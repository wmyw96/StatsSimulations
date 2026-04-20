"""Estimators for partial linear models."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from simlab.core.records import EstimateResult, SampledData
from simlab.core.types import ConfigDict
from simlab.estimators.base import Estimator

PLMRegressionFunction = Callable[[np.ndarray], np.ndarray]


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Batch normalization that gracefully handles batch size one during training."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training and inputs.shape[0] == 1:
            return nn.functional.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )
        return super().forward(inputs)


class ResidualBlock(nn.Module):
    """A width-preserving residual block with batch normalization and ReLU."""

    def __init__(self, width: int):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.bn1 = SafeBatchNorm1d(width)
        self.linear2 = nn.Linear(width, width)
        self.bn2 = SafeBatchNorm1d(width)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.linear1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + residual
        return self.activation(outputs)


class ResidualReLUNet(nn.Module):
    """Fully connected ReLU network with batch normalization and residual blocks."""

    def __init__(self, input_dim: int, depth: int, width: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if width <= 0:
            raise ValueError("width must be positive.")

        self.input_layer = nn.Linear(input_dim, width)
        self.input_bn = SafeBatchNorm1d(width)
        self.activation = nn.ReLU()
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(depth - 1))
        self.output_layer = nn.Linear(width, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.input_layer(inputs)
        outputs = self.input_bn(outputs)
        outputs = self.activation(outputs)
        for block in self.blocks:
            outputs = block(outputs)
        return self.output_layer(outputs)


class PLMDMLEstimator(Estimator):
    """Double machine learning estimator for the partial linear model."""

    def __init__(
        self,
        name: str,
        hyper_parameters: ConfigDict,
        d: int,
        device: str = "cpu",
    ):
        super().__init__(name=name, hyper_parameters=hyper_parameters)
        if d <= 0:
            raise ValueError("d must be a positive integer.")

        self.d = d
        self.device = _resolve_device(device)
        self.L = int(hyper_parameters["L"])
        self.N = int(hyper_parameters["N"])
        self.lambda_mu = float(hyper_parameters["lambda_mu"])
        self.lambda_pi = float(hyper_parameters["lambda_pi"])
        self.niter = int(hyper_parameters["niter"])
        self.lr = float(hyper_parameters.get("lr", 1e-3))
        self.batch_size = int(hyper_parameters.get("batch_size", 1024))
        self.seed = hyper_parameters.get("seed")

        if self.seed is not None:
            _set_torch_seed(int(self.seed))

        self.est_mu = ResidualReLUNet(input_dim=d, depth=self.L, width=self.N).to(self.device)
        self.est_pi = ResidualReLUNet(input_dim=d, depth=self.L, width=self.N).to(self.device)
        self.beta = nn.Parameter(torch.zeros(1, device=self.device))
        self.optimizer = torch.optim.Adam(
            list(self.est_mu.parameters()) + list(self.est_pi.parameters()),
            lr=self.lr,
        )

    def fit(self, data: SampledData) -> EstimateResult:
        return self._fit_internal(data=data, record_oracle_paths=False)

    def _fit_internal(
        self,
        data: SampledData,
        record_oracle_paths: bool,
    ) -> EstimateResult:
        x, t, y = _extract_plm_arrays(data)
        d1_x, d1_t, d1_y, d2_x, d2_t, d2_y = _split_plm_data(x, t, y)

        x2_tensor = torch.as_tensor(d2_x, dtype=torch.float32, device=self.device)
        t2_tensor = torch.as_tensor(d2_t, dtype=torch.float32, device=self.device)
        y2_tensor = torch.as_tensor(d2_y, dtype=torch.float32, device=self.device)
        oracle_tracking = None
        if record_oracle_paths:
            oracle_tracking = _build_oracle_tracking_state(
                data=data,
                n_total=len(x),
                device=self.device,
            )

        dataset = TensorDataset(x2_tensor, t2_tensor, y2_tensor)
        batch_size = min(self.batch_size, len(dataset))
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        last_joint_loss = float("nan")
        last_pi_loss = float("nan")
        full_joint_loss = float("nan")
        full_pi_loss = float("nan")
        epoch_grid: list[int] = []
        mu_mse_path: list[float] = []
        pi_mse_path: list[float] = []

        if oracle_tracking is not None:
            self.est_mu.eval()
            self.est_pi.eval()
            with torch.no_grad():
                initial_mu_mse, initial_pi_mse = _compute_oracle_tracking_mse(
                    est_mu=self.est_mu,
                    est_pi=self.est_pi,
                    tracking_x=oracle_tracking["x"],
                    tracking_mu=oracle_tracking["mu"],
                    tracking_pi=oracle_tracking["pi"],
                )
            epoch_grid.append(0)
            mu_mse_path.append(initial_mu_mse)
            pi_mse_path.append(initial_pi_mse)

        self.est_mu.train()
        self.est_pi.train()
        for epoch in range(self.niter):
            for batch_x, batch_t, batch_y in loader:
                self.optimizer.zero_grad()
                mu_pred = self.est_mu(batch_x)
                pi_pred = self.est_pi(batch_x)
                joint_loss = torch.mean((self.beta.detach() * batch_t + mu_pred - batch_y) ** 2)
                pi_loss = torch.mean((batch_t - pi_pred) ** 2)
                total_loss = (
                    joint_loss
                    + pi_loss
                    + self.lambda_mu * _weight_l2_penalty(self.est_mu)
                    + self.lambda_pi * _weight_l2_penalty(self.est_pi)
                )
                total_loss.backward()
                self.optimizer.step()
                last_joint_loss = float(joint_loss.detach().cpu().item())
                last_pi_loss = float(pi_loss.detach().cpu().item())

            self.est_mu.eval()
            self.est_pi.eval()
            with torch.no_grad():
                mu_d2 = self.est_mu(x2_tensor)
                pi_d2 = self.est_pi(x2_tensor)
                profiled_beta = _profile_beta_from_tensors(
                    t=t2_tensor,
                    y=y2_tensor,
                    mu=mu_d2,
                )
                self.beta.data.fill_(profiled_beta)
                full_joint_loss = float(torch.mean((self.beta * t2_tensor + mu_d2 - y2_tensor) ** 2).cpu().item())
                full_pi_loss = float(torch.mean((t2_tensor - pi_d2) ** 2).cpu().item())
                if oracle_tracking is not None:
                    tracked_mu_mse, tracked_pi_mse = _compute_oracle_tracking_mse(
                        est_mu=self.est_mu,
                        est_pi=self.est_pi,
                        tracking_x=oracle_tracking["x"],
                        tracking_mu=oracle_tracking["mu"],
                        tracking_pi=oracle_tracking["pi"],
                    )
                    epoch_grid.append(epoch + 1)
                    mu_mse_path.append(tracked_mu_mse)
                    pi_mse_path.append(tracked_pi_mse)
            self.est_mu.train()
            self.est_pi.train()

        self.est_mu.eval()
        self.est_pi.eval()
        with torch.no_grad():
            d1_x_tensor = torch.as_tensor(d1_x, dtype=torch.float32, device=self.device)
            mu_hat = self.est_mu(d1_x_tensor).detach().cpu().numpy()
            pi_hat = self.est_pi(d1_x_tensor).detach().cpu().numpy()

        beta_hat = _aipw_beta(d1_y, d1_t, mu_hat, pi_hat)
        diagnostics = {
            "beta_joint": float(self.beta.detach().cpu().item()),
            "final_joint_loss": full_joint_loss if np.isfinite(full_joint_loss) else last_joint_loss,
            "final_pi_loss": full_pi_loss if np.isfinite(full_pi_loss) else last_pi_loss,
            "n_d1": len(d1_x),
            "n_d2": len(d2_x),
            "device": str(self.device),
        }
        if oracle_tracking is not None:
            diagnostics.update(
                {
                    "epoch_grid": epoch_grid,
                    "mu_mse_path": mu_mse_path,
                    "pi_mse_path": pi_mse_path,
                    "tracking_split": "D2",
                    "tracking_n": int(oracle_tracking["x"].shape[0]),
                }
            )
        self.est_params = EstimateResult(
            target="beta",
            estimate=beta_hat,
            diagnostics=diagnostics,
        )
        return self.est_params

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.est_params is None:
            raise RuntimeError("Call fit() before predict().")

        features = _validate_feature_matrix(X, expected_dim=self.d)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        self.est_mu.eval()
        self.est_pi.eval()
        with torch.no_grad():
            mu_hat = self.est_mu(feature_tensor).detach().cpu().numpy()
            pi_hat = self.est_pi(feature_tensor).detach().cpu().numpy()
        return {
            "mu": mu_hat,
            "pi": pi_hat,
        }


class PLMOracleAIPWEstimator(Estimator):
    """Oracle estimator using the ground-truth nuisance functions in the AIPW formula."""

    def __init__(
        self,
        name: str,
        ground_truth_func_mu: PLMRegressionFunction,
        ground_truth_func_pi: PLMRegressionFunction,
        hyper_parameters: ConfigDict | None = None,
    ):
        super().__init__(name=name, hyper_parameters=hyper_parameters or {})
        if not callable(ground_truth_func_mu):
            raise TypeError("ground_truth_func_mu must be callable.")
        if not callable(ground_truth_func_pi):
            raise TypeError("ground_truth_func_pi must be callable.")
        self.ground_truth_func_mu = deepcopy(ground_truth_func_mu)
        self.ground_truth_func_pi = deepcopy(ground_truth_func_pi)

    def fit(self, data: SampledData) -> EstimateResult:
        x, t, y = _extract_plm_arrays(data)
        d1_x, d1_t, d1_y, _, _, _ = _split_plm_data(x, t, y)
        mu_hat = _as_float_column(self.ground_truth_func_mu(d1_x), len(d1_x), label="ground_truth_func_mu")
        pi_hat = _as_float_column(self.ground_truth_func_pi(d1_x), len(d1_x), label="ground_truth_func_pi")
        beta_hat = _aipw_beta(d1_y, d1_t, mu_hat, pi_hat)
        self.est_params = EstimateResult(
            target="beta",
            estimate=beta_hat,
            diagnostics={
                "n_d1": len(d1_x),
                "n_d2": len(x) - len(d1_x),
            },
        )
        return self.est_params

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        features = _validate_feature_matrix(X)
        mu_hat = _as_float_column(
            self.ground_truth_func_mu(features),
            len(features),
            label="ground_truth_func_mu",
        )
        pi_hat = _as_float_column(
            self.ground_truth_func_pi(features),
            len(features),
            label="ground_truth_func_pi",
        )
        return {
            "mu": mu_hat,
            "pi": pi_hat,
        }


class PLMDMLOracleTrackingEstimator(PLMDMLEstimator):
    """DML estimator variant that records oracle nuisance MSE paths across epochs."""

    def fit(self, data: SampledData) -> EstimateResult:
        return self._fit_internal(data=data, record_oracle_paths=True)


def _resolve_device(device: str) -> torch.device:
    """Resolve and validate a torch device string."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return resolved


def _set_torch_seed(seed: int) -> None:
    """Set PyTorch random seeds for reproducible initialization and shuffling."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weight_l2_penalty(module: nn.Module) -> torch.Tensor:
    """Compute the sum of squared weights, excluding biases and batch-norm scales."""
    penalty = torch.zeros(1, device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.ndim > 1:
            penalty = penalty + torch.sum(parameter ** 2)
    return penalty


def _profile_beta_from_tensors(t: torch.Tensor, y: torch.Tensor, mu: torch.Tensor) -> float:
    """Return the profiled least-squares beta for fixed mu on one split."""
    denominator = torch.mean(t * t)
    if float(torch.abs(denominator).cpu().item()) <= 1e-12:
        raise ZeroDivisionError("Cannot profile beta because mean(T^2) is numerically zero.")
    numerator = torch.mean(t * (y - mu))
    return float((numerator / denominator).detach().cpu().item())


def _build_oracle_tracking_state(
    data: SampledData,
    n_total: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build tensors for oracle nuisance tracking on the fitted split."""
    if "mu_x" not in data.oracle or "pi_x" not in data.oracle:
        raise KeyError(
            "Oracle nuisance tracking requires oracle keys 'mu_x' and 'pi_x' in the sampled data."
        )
    split = n_total // 2
    tracking_x = _validate_feature_matrix(data.observed["x"][split:])
    tracking_mu = _as_float_column(data.oracle["mu_x"], n_total, label="oracle mu_x")[split:]
    tracking_pi = _as_float_column(data.oracle["pi_x"], n_total, label="oracle pi_x")[split:]
    return {
        "x": torch.as_tensor(tracking_x, dtype=torch.float32, device=device),
        "mu": torch.as_tensor(tracking_mu, dtype=torch.float32, device=device),
        "pi": torch.as_tensor(tracking_pi, dtype=torch.float32, device=device),
    }


def _compute_oracle_tracking_mse(
    est_mu: nn.Module,
    est_pi: nn.Module,
    tracking_x: torch.Tensor,
    tracking_mu: torch.Tensor,
    tracking_pi: torch.Tensor,
) -> tuple[float, float]:
    """Compute oracle nuisance MSEs for one evaluation checkpoint."""
    mu_pred = est_mu(tracking_x)
    pi_pred = est_pi(tracking_x)
    mu_mse = torch.mean((mu_pred - tracking_mu) ** 2)
    pi_mse = torch.mean((pi_pred - tracking_pi) ** 2)
    return (
        float(mu_mse.detach().cpu().item()),
        float(pi_mse.detach().cpu().item()),
    )


def _validate_feature_matrix(X: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
    """Validate that X is a two-dimensional NumPy array."""
    features = np.asarray(X, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(f"X must have shape (n, {expected_dim}).")
    return features


def _extract_plm_arrays(data: SampledData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract x, t, y arrays from generic sampled data."""
    try:
        x = data.observed["x"]
        t = data.observed["t"]
        y = data.observed["y"]
    except KeyError as error:
        raise KeyError("PLM estimators require observed keys 'x', 't', and 'y'.") from error

    x = _validate_feature_matrix(x)
    t = _as_float_column(t, len(x), label="t")
    y = _as_float_column(y, len(x), label="y")
    return x, t, y


def _split_plm_data(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split PLM data so the first floor(n/2) samples are in D1 and the rest in D2."""
    n = len(x)
    if n < 2:
        raise ValueError("At least two observations are required for the D1/D2 split.")
    split = n // 2
    return (
        x[:split],
        t[:split],
        y[:split],
        x[split:],
        t[split:],
        y[split:],
    )


def _as_float_column(values: np.ndarray, n: int, label: str) -> np.ndarray:
    """Validate and reshape values as an n by 1 floating-point array."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape == (n,):
        return array.reshape(n, 1)
    if array.shape == (n, 1):
        return array
    raise ValueError(f"{label} must have shape (n,) or (n, 1).")


def _aipw_beta(
    y: np.ndarray,
    t: np.ndarray,
    mu_hat: np.ndarray,
    pi_hat: np.ndarray,
) -> float:
    """Compute the AIPW-style estimate in Eq. (1.2) of the PLM paper."""
    numerator = float(np.mean((y - mu_hat) * (t - pi_hat)))
    denominator = float(np.mean(t * (t - pi_hat)))
    if abs(denominator) <= 1e-8:
        raise ZeroDivisionError("The AIPW denominator is too close to zero.")
    return numerator / denominator
