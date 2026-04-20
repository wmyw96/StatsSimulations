"""Estimators for partial linear models."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

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


class DifferenceResidualReLUNet(nn.Module):
    """Difference of two residual ReLU networks for representing a difference class."""

    def __init__(self, input_dim: int, depth: int, width: int):
        super().__init__()
        self.net_plus = ResidualReLUNet(input_dim=input_dim, depth=depth, width=width)
        self.net_minus = ResidualReLUNet(input_dim=input_dim, depth=depth, width=width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net_plus(inputs) - self.net_minus(inputs)


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
        oracle_tracking_states = None
        if record_oracle_paths:
            oracle_tracking_states = _build_oracle_tracking_states(
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
        tracking_paths: dict[str, dict[str, Any]] = {}

        if oracle_tracking_states is not None:
            for split_label, tracking_state in oracle_tracking_states.items():
                tracking_paths[split_label] = {
                    "mu_mse_path": [],
                    "pi_mse_path": [],
                    "tracking_n": int(tracking_state["x"].shape[0]),
                }
            self.est_mu.eval()
            self.est_pi.eval()
            with torch.no_grad():
                for split_label, tracking_state in oracle_tracking_states.items():
                    initial_mu_mse, initial_pi_mse = _compute_oracle_tracking_mse(
                        est_mu=self.est_mu,
                        est_pi=self.est_pi,
                        tracking_x=tracking_state["x"],
                        tracking_mu=tracking_state["mu"],
                        tracking_pi=tracking_state["pi"],
                    )
                    tracking_paths[split_label]["mu_mse_path"].append(initial_mu_mse)
                    tracking_paths[split_label]["pi_mse_path"].append(initial_pi_mse)
            epoch_grid.append(0)

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
                if oracle_tracking_states is not None:
                    epoch_grid.append(epoch + 1)
                    for split_label, tracking_state in oracle_tracking_states.items():
                        tracked_mu_mse, tracked_pi_mse = _compute_oracle_tracking_mse(
                            est_mu=self.est_mu,
                            est_pi=self.est_pi,
                            tracking_x=tracking_state["x"],
                            tracking_mu=tracking_state["mu"],
                            tracking_pi=tracking_state["pi"],
                        )
                        tracking_paths[split_label]["mu_mse_path"].append(tracked_mu_mse)
                        tracking_paths[split_label]["pi_mse_path"].append(tracked_pi_mse)
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
        if oracle_tracking_states is not None:
            diagnostics["epoch_grid"] = epoch_grid
            if len(tracking_paths) == 1:
                split_label, path_record = next(iter(tracking_paths.items()))
                diagnostics.update(
                    {
                        "mu_mse_path": path_record["mu_mse_path"],
                        "pi_mse_path": path_record["pi_mse_path"],
                        "tracking_split": split_label,
                        "tracking_n": path_record["tracking_n"],
                    }
                )
            else:
                diagnostics["tracking_paths"] = tracking_paths
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


class PLMMinimaxDebiasEstimator(PLMDMLEstimator):
    """Paper-style minimax debiasing estimator based on Eq. (2.3)."""

    def __init__(
        self,
        name: str,
        hyper_parameters: ConfigDict,
        d: int,
        device: str = "cpu",
    ):
        super().__init__(name=name, hyper_parameters=hyper_parameters, d=d, device=device)
        self.lambda_debias = hyper_parameters.get("lambda_debias")
        self.weight_bound = float(hyper_parameters.get("weight_bound", 5.0))
        self.niter_debias = int(hyper_parameters.get("niter_debias", self.niter))
        self.niter_adversary = int(hyper_parameters.get("niter_adversary", 5))
        self.debias_lr = float(hyper_parameters.get("debias_lr", self.lr))
        self.smooth_eps = float(hyper_parameters.get("smooth_eps", 1e-6))
        self.variance_mode = str(hyper_parameters.get("variance_mode", "constant_one"))

        if self.weight_bound <= 0:
            raise ValueError("weight_bound must be positive.")
        if self.niter_debias <= 0:
            raise ValueError("niter_debias must be positive.")
        if self.niter_adversary <= 0:
            raise ValueError("niter_adversary must be positive.")
        if self.debias_lr <= 0:
            raise ValueError("debias_lr must be positive.")
        if self.variance_mode != "constant_one":
            raise ValueError("Only variance_mode='constant_one' is currently supported.")

    def fit(self, data: SampledData) -> EstimateResult:
        x, t, y = _extract_plm_arrays(data)
        d1_x, d1_t, d1_y, d2_x, d2_t, d2_y = _split_plm_data(x, t, y)

        x2_tensor = torch.as_tensor(d2_x, dtype=torch.float32, device=self.device)
        t2_tensor = torch.as_tensor(d2_t, dtype=torch.float32, device=self.device)
        y2_tensor = torch.as_tensor(d2_y, dtype=torch.float32, device=self.device)

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

        self.est_mu.train()
        self.est_pi.train()
        for _ in range(self.niter):
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
            self.est_mu.train()
            self.est_pi.train()

        self.est_mu.eval()
        self.est_pi.eval()
        with torch.no_grad():
            d1_x_tensor = torch.as_tensor(d1_x, dtype=torch.float32, device=self.device)
            mu_hat = self.est_mu(d1_x_tensor).detach().cpu().numpy()

        lambda_debias = (
            float(self.lambda_debias)
            if self.lambda_debias is not None
            else _default_lambda_debias(len(d1_x))
        )
        a_hat, debias_diagnostics = _fit_minimax_debiasing_weights(
            x=d1_x,
            t=d1_t,
            est_mu=self.est_mu,
            d=self.d,
            depth=self.L,
            width=self.N,
            device=self.device,
            weight_bound=self.weight_bound,
            lambda_debias=lambda_debias,
            niter_debias=self.niter_debias,
            niter_adversary=self.niter_adversary,
            debias_lr=self.debias_lr,
            smooth_eps=self.smooth_eps,
            seed=self.seed,
        )

        beta_hat = float(np.mean((d1_y - mu_hat) * a_hat))
        diagnostics = {
            "beta_joint": float(self.beta.detach().cpu().item()),
            "final_joint_loss": full_joint_loss if np.isfinite(full_joint_loss) else last_joint_loss,
            "final_pi_loss": full_pi_loss if np.isfinite(full_pi_loss) else last_pi_loss,
            "n_d1": len(d1_x),
            "n_d2": len(d2_x),
            "device": str(self.device),
            "variance_mode": self.variance_mode,
            "lambda_debias": lambda_debias,
            **debias_diagnostics,
        }
        self.est_params = EstimateResult(
            target="beta",
            estimate=beta_hat,
            diagnostics=diagnostics,
        )
        return self.est_params


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


def _fit_minimax_debiasing_weights(
    x: np.ndarray,
    t: np.ndarray,
    est_mu: nn.Module,
    d: int,
    depth: int,
    width: int,
    device: torch.device,
    weight_bound: float,
    lambda_debias: float,
    niter_debias: int,
    niter_adversary: int,
    debias_lr: float,
    smooth_eps: float,
    seed: int | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit empirical debiasing weights with the stabilized paper objective."""
    if seed is not None:
        _set_torch_seed(int(seed) + 1)

    x_tensor = torch.as_tensor(_validate_feature_matrix(x, expected_dim=d), dtype=torch.float32, device=device)
    t_tensor = torch.as_tensor(_as_float_column(t, len(x), label="t"), dtype=torch.float32, device=device)
    vhat_tensor = torch.ones_like(t_tensor, device=device)

    initial_weights = t_tensor / torch.clamp(torch.mean(t_tensor * t_tensor), min=1e-6)
    initial_weights = torch.clamp(initial_weights, -0.99 * weight_bound, 0.99 * weight_bound)
    raw_a = nn.Parameter(torch.atanh(initial_weights / weight_bound))
    adv_beta = nn.Parameter(torch.zeros(1, device=device))
    adv_f = DifferenceResidualReLUNet(input_dim=d, depth=depth, width=width).to(device)

    optimizer_a = torch.optim.Adam([raw_a], lr=debias_lr)
    optimizer_adv = torch.optim.Adam(list(adv_f.parameters()) + [adv_beta], lr=debias_lr)

    def current_weights() -> torch.Tensor:
        return weight_bound * torch.tanh(raw_a)

    def stabilized_value(a_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal = adv_beta * t_tensor + adv_f(x_tensor)
        imbalance = torch.mean(signal * a_tensor) - adv_beta
        smooth_abs = torch.sqrt(imbalance * imbalance + smooth_eps)
        stability = torch.mean(signal * signal)
        return smooth_abs - stability, imbalance, stability

    final_imbalance = 0.0
    final_stability = 0.0
    final_adversary_value = 0.0
    final_objective = 0.0
    for _ in range(niter_debias):
        for _ in range(niter_adversary):
            optimizer_adv.zero_grad()
            a_tensor = current_weights().detach()
            adversary_value, _, _ = stabilized_value(a_tensor)
            (-adversary_value).backward()
            optimizer_adv.step()

        optimizer_a.zero_grad()
        a_tensor = current_weights()
        adversary_value, imbalance, stability = stabilized_value(a_tensor)
        objective = lambda_debias * torch.mean(vhat_tensor * (a_tensor * a_tensor)) + adversary_value
        objective.backward()
        optimizer_a.step()

        final_imbalance = float(imbalance.detach().cpu().item())
        final_stability = float(stability.detach().cpu().item())
        final_adversary_value = float(adversary_value.detach().cpu().item())
        final_objective = float(objective.detach().cpu().item())

    a_hat = current_weights().detach().cpu().numpy()
    diagnostics = {
        "weight_bound": float(weight_bound),
        "mean_ta": float(np.mean(t * a_hat)),
        "mean_a_sq": float(np.mean(a_hat * a_hat)),
        "max_abs_weight": float(np.max(np.abs(a_hat))),
        "final_debias_objective": final_objective,
        "final_adversary_value": final_adversary_value,
        "final_imbalance": final_imbalance,
        "final_stability_term": final_stability,
    }
    return a_hat, diagnostics


def _build_oracle_tracking_states(
    data: SampledData,
    n_total: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build tensors for oracle nuisance tracking on D2 and optional validation data."""
    tracking_states: dict[str, dict[str, torch.Tensor]] = {}
    if "mu_x" not in data.oracle or "pi_x" not in data.oracle:
        raise KeyError(
            "Oracle nuisance tracking requires oracle keys 'mu_x' and 'pi_x' in the sampled data."
        )
    split = n_total // 2
    tracking_x = _validate_feature_matrix(data.observed["x"][split:])
    tracking_mu = _as_float_column(data.oracle["mu_x"], n_total, label="oracle mu_x")[split:]
    tracking_pi = _as_float_column(data.oracle["pi_x"], n_total, label="oracle pi_x")[split:]
    tracking_states["D2"] = {
        "x": torch.as_tensor(tracking_x, dtype=torch.float32, device=device),
        "mu": torch.as_tensor(tracking_mu, dtype=torch.float32, device=device),
        "pi": torch.as_tensor(tracking_pi, dtype=torch.float32, device=device),
    }
    validation_keys = {"validation_x", "validation_mu_x", "validation_pi_x"}
    if validation_keys.issubset(data.oracle):
        tracking_x = _validate_feature_matrix(data.oracle["validation_x"])
        tracking_mu = _as_float_column(
            data.oracle["validation_mu_x"],
            len(tracking_x),
            label="validation oracle mu_x",
        )
        tracking_pi = _as_float_column(
            data.oracle["validation_pi_x"],
            len(tracking_x),
            label="validation oracle pi_x",
        )
        tracking_states["validation"] = {
            "x": torch.as_tensor(tracking_x, dtype=torch.float32, device=device),
            "mu": torch.as_tensor(tracking_mu, dtype=torch.float32, device=device),
            "pi": torch.as_tensor(tracking_pi, dtype=torch.float32, device=device),
        }
    return tracking_states


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


def _default_lambda_debias(n: int) -> float:
    """Return the default debiasing penalty level based on the D1 sample size."""
    if n <= 0:
        raise ValueError("n must be positive for the default debiasing penalty.")
    return 1.0 / (float(np.sqrt(n)) * float(np.log2(max(n, 2))))
