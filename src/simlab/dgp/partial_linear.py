"""Partial linear model data-generating process implementations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from simlab.core.records import SampledData
from simlab.core.types import ConfigDict
from simlab.dgp.base import DataGeneratingProcess

PLMRegressionFunction = Callable[[np.ndarray], np.ndarray]


class PartialLinearModelUniformNoiseDGP(DataGeneratingProcess):
    """Partial linear model with uniform covariates and uniform additive noise."""

    def __init__(
        self,
        beta: float,
        func_mu: PLMRegressionFunction,
        func_pi: PLMRegressionFunction,
        d: int,
        sigma_u: float,
        sigma_eps: float,
        name: str = "partial_linear_uniform_noise",
    ):
        if d <= 0:
            raise ValueError("d must be a positive integer.")
        if sigma_u < 0:
            raise ValueError("sigma_u must be non-negative.")
        if sigma_eps < 0:
            raise ValueError("sigma_eps must be non-negative.")
        if not callable(func_mu):
            raise TypeError("func_mu must be callable.")
        if not callable(func_pi):
            raise TypeError("func_pi must be callable.")

        params: ConfigDict = {
            "beta": beta,
            "func_mu": func_mu,
            "func_pi": func_pi,
            "d": d,
            "sigma_u": sigma_u,
            "sigma_eps": sigma_eps,
        }
        super().__init__(name=name, params=params)
        self.beta = float(beta)
        self.func_mu = func_mu
        self.func_pi = func_pi
        self.d = d
        self.sigma_u = float(sigma_u)
        self.sigma_eps = float(sigma_eps)

    def sample(
        self,
        n: int,
        seed: int | None = None,
        oracle: bool = False,
    ) -> SampledData:
        """Sample from the partial linear model with uniform covariates and noise."""
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, size=(n, self.d))

        pi_x = self._ensure_column_vector(self.func_pi(x), n=n, label="func_pi")
        u = rng.uniform(-self.sigma_u, self.sigma_u, size=(n, 1))
        t = pi_x + u

        mu_x = self._ensure_column_vector(self.func_mu(x), n=n, label="func_mu")
        eps = rng.uniform(-self.sigma_eps, self.sigma_eps, size=(n, 1))
        y = self.beta * t + mu_x + eps

        return SampledData(
            observed={
                "x": x,
                "t": t,
                "y": y,
            },
            oracle={
                "pi_x": pi_x,
                "mu_x": mu_x,
            }
            if oracle
            else {},
        )

    def true_parameter(self) -> float:
        """Return the ground-truth linear effect coefficient."""
        return self.beta

    @staticmethod
    def _ensure_column_vector(values: np.ndarray, n: int, label: str) -> np.ndarray:
        """Validate that a regression function returns an n by 1 array."""
        array = np.asarray(values, dtype=float)
        if array.shape == (n,):
            return array.reshape(n, 1)
        if array.shape == (n, 1):
            return array
        raise ValueError(f"{label} must return an array of shape (n,) or (n, 1).")
