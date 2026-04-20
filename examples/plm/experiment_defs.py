"""Experiment definitions for partial linear model simulations."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from simlab.dgp.partial_linear import PartialLinearModelUniformNoiseDGP
from simlab.estimators.plm_est import (
    PLMDMLEstimator,
    PLMDMLOracleTrackingEstimator,
    PLMOracleAIPWEstimator,
)
from simlab.evaluation.plm_eval import PLMEvaluator

EXPERIMENT_NAME = "plm"
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parents[2] / "simulation_results"


def sin_2pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sin(2 pi x) using the first coordinate of x."""
    return np.sin(2.0 * np.pi * x[:, [0]])


def sin_4pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sin(4 pi x) using the first coordinate of x."""
    return np.sin(4.0 * np.pi * x[:, [0]])


def sin_8pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sin(8 pi x) using the first coordinate of x."""
    return np.sin(8.0 * np.pi * x[:, [0]])


def sign_sin_2pi_times_sin_2pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sign(sin(2 pi x)) * sin(2 pi x) using the first coordinate."""
    x0 = x[:, [0]]
    return np.sign(np.sin(2.0 * np.pi * x0)) * np.sin(2.0 * np.pi * x0)


def sign_sin_2pi_times_sin_4pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sign(sin(2 pi x)) * sin(4 pi x) using the first coordinate."""
    x0 = x[:, [0]]
    return np.sign(np.sin(2.0 * np.pi * x0)) * np.sin(4.0 * np.pi * x0)


def sign_sin_2pi_times_sin_8pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sign(sin(2 pi x)) * sin(8 pi x) using the first coordinate."""
    x0 = x[:, [0]]
    return np.sign(np.sin(2.0 * np.pi * x0)) * np.sin(8.0 * np.pi * x0)


def sign_sin_2pi_times_abs_sin_2pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sign(sin(2 pi x)) * abs(sin(2 pi x)) using the first coordinate."""
    x0 = x[:, [0]]
    return np.sign(np.sin(2.0 * np.pi * x0)) * np.abs(np.sin(2.0 * np.pi * x0))


def sign_sin_2pi_times_abs_sin_4pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sign(sin(2 pi x)) * abs(sin(4 pi x)) using the first coordinate."""
    x0 = x[:, [0]]
    return np.sign(np.sin(2.0 * np.pi * x0)) * np.abs(np.sin(4.0 * np.pi * x0))


def sign_sin_2pi_times_abs_sin_8pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sign(sin(2 pi x)) * abs(sin(8 pi x)) using the first coordinate."""
    x0 = x[:, [0]]
    return np.sign(np.sin(2.0 * np.pi * x0)) * np.abs(np.sin(8.0 * np.pi * x0))


def _progressive_pi_multiscale_component(x: np.ndarray) -> np.ndarray:
    """Return a rough multiscale signed-wave component on the first coordinate."""
    x0 = x[:, [0]]
    raw = (
        np.sign(np.sin(8.0 * np.pi * x0))
        + 0.5 * np.sign(np.sin(16.0 * np.pi * x0))
        + 0.25 * np.sign(np.sin(32.0 * np.pi * x0))
    ) / 1.75
    return 1.08 * raw


def progressive_pi_1_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Smooth and weakly aligned candidate for a progressive pi family."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sin(4.0 * np.pi * x0)
    return 0.25 * mu + np.sqrt(1.0 - 0.25**2) * aux


def progressive_pi_2_first_coordinate(x: np.ndarray) -> np.ndarray:
    """More oscillatory and more aligned candidate for a progressive pi family."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sin(8.0 * np.pi * x0)
    return 0.5 * mu + np.sqrt(1.0 - 0.5**2) * aux


def progressive_pi_3_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Discontinuous and strongly aligned candidate for a progressive pi family."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sqrt(0.5) * np.sign(np.sin(8.0 * np.pi * x0))
    return 0.75 * mu + np.sqrt(1.0 - 0.75**2) * aux


def progressive_pi_4_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Very rough multiscale candidate for a progressive pi family."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = _progressive_pi_multiscale_component(x)
    return 0.9 * mu + np.sqrt(1.0 - 0.9**2) * aux


def fixed_overlap_pi_1_first_coordinate(x: np.ndarray) -> np.ndarray:
    """High-overlap and smooth treatment regression candidate."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sqrt(2.0) * np.cos(2.0 * np.pi * x0)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def fixed_overlap_pi_2_first_coordinate(x: np.ndarray) -> np.ndarray:
    """High-overlap and higher-frequency smooth treatment regression candidate."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sqrt(2.0) * np.cos(8.0 * np.pi * x0)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def fixed_overlap_pi_3_first_coordinate(x: np.ndarray) -> np.ndarray:
    """High-overlap and discontinuous treatment regression candidate."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sign(np.sin(8.0 * np.pi * x0))
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def fixed_overlap_pi_4_first_coordinate(x: np.ndarray) -> np.ndarray:
    """High-overlap and very high-frequency discontinuous treatment regression candidate."""
    x0 = x[:, [0]]
    mu = np.sin(2.0 * np.pi * x0)
    aux = np.sign(np.sin(64.0 * np.pi * x0))
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def sin_2pi_four_coordinates(x: np.ndarray) -> np.ndarray:
    """Return a four-dimensional smooth outcome regression with matched scale."""
    return 0.5 * np.sum(np.sin(2.0 * np.pi * x), axis=1, keepdims=True)


def fixed_overlap_d4_pi_1(x: np.ndarray) -> np.ndarray:
    """High-overlap smooth low-frequency treatment regression in four dimensions."""
    mu = sin_2pi_four_coordinates(x)
    aux = 0.5 * np.sum(np.sqrt(2.0) * np.cos(2.0 * np.pi * x), axis=1, keepdims=True)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def fixed_overlap_d4_pi_2(x: np.ndarray) -> np.ndarray:
    """High-overlap smooth high-frequency treatment regression in four dimensions."""
    mu = sin_2pi_four_coordinates(x)
    aux = 0.5 * np.sum(np.sqrt(2.0) * np.cos(8.0 * np.pi * x), axis=1, keepdims=True)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def fixed_overlap_d4_pi_3(x: np.ndarray) -> np.ndarray:
    """High-overlap discontinuous high-frequency treatment regression in four dimensions."""
    mu = sin_2pi_four_coordinates(x)
    aux = 0.5 * np.sum(np.sign(np.sin(32.0 * np.pi * x)), axis=1, keepdims=True)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def isolated_d4_pi_1(x: np.ndarray) -> np.ndarray:
    """Keep mu on x1 but let pi depend smoothly on x2, x3, and x4."""
    x1 = x[:, [0]]
    x234 = x[:, 1:4]
    mu = np.sin(2.0 * np.pi * x1)
    aux = np.sum(np.sin(2.0 * np.pi * x234), axis=1, keepdims=True) / np.sqrt(3.0)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def isolated_d4_pi_2(x: np.ndarray) -> np.ndarray:
    """Increase pi frequency on x2, x3, and x4 while keeping mu unchanged."""
    x1 = x[:, [0]]
    x234 = x[:, 1:4]
    mu = np.sin(2.0 * np.pi * x1)
    aux = np.sum(np.sin(8.0 * np.pi * x234), axis=1, keepdims=True) / np.sqrt(3.0)
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def isolated_d4_pi_3(x: np.ndarray) -> np.ndarray:
    """Use a discontinuous three-way interaction on x2, x3, and x4 for pi."""
    x1 = x[:, [0]]
    x234 = x[:, 1:4]
    mu = np.sin(2.0 * np.pi * x1)
    aux = np.sign(np.prod(np.sin(8.0 * np.pi * x234), axis=1, keepdims=True))
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def isolated_d4_pi_4(x: np.ndarray) -> np.ndarray:
    """Make pi very rough through a high-frequency three-way interaction."""
    x1 = x[:, [0]]
    x234 = x[:, 1:4]
    mu = np.sin(2.0 * np.pi * x1)
    aux = np.sign(np.prod(np.sin(32.0 * np.pi * x234), axis=1, keepdims=True))
    return 0.98 * mu + np.sqrt(1.0 - 0.98**2) * aux


def easy_mu_sin_pi_x1_plus_cos_pi_x2(x: np.ndarray) -> np.ndarray:
    """Return a relatively easy outcome regression on the first two coordinates."""
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    return np.sin(np.pi * x1) + np.cos(np.pi * x2)


def increasing_beta_pi_1(x: np.ndarray) -> np.ndarray:
    """Smooth low-amplitude perturbation of the easy mu design."""
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    mu = easy_mu_sin_pi_x1_plus_cos_pi_x2(x)
    aux = (np.sin(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)) / np.sqrt(2.0)
    return mu + 0.05 * aux


def increasing_beta_pi_2(x: np.ndarray) -> np.ndarray:
    """Add a moderate-amplitude rough interaction on the same coordinates."""
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    mu = easy_mu_sin_pi_x1_plus_cos_pi_x2(x)
    aux = np.sign(np.sin(8.0 * np.pi * x1)) * np.sign(np.cos(8.0 * np.pi * x2))
    return mu + 0.18 * aux


def increasing_beta_pi_3(x: np.ndarray) -> np.ndarray:
    """Increase the amplitude of the rough interaction to raise pi error further."""
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    mu = easy_mu_sin_pi_x1_plus_cos_pi_x2(x)
    aux = np.sign(np.sin(8.0 * np.pi * x1)) * np.sign(np.cos(8.0 * np.pi * x2))
    return mu + 0.20 * aux


def increasing_beta_pi_4(x: np.ndarray) -> np.ndarray:
    """Use the same rough interaction with the largest amplitude in the family."""
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    mu = easy_mu_sin_pi_x1_plus_cos_pi_x2(x)
    aux = np.sign(np.sin(8.0 * np.pi * x1)) * np.sign(np.cos(8.0 * np.pi * x2))
    return mu + 0.25 * aux


def correlated_hard_component_g2(x: np.ndarray) -> np.ndarray:
    """Return a rough component that stays aligned with the easy smooth signal."""
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    rough_magnitude = 0.5 * (
        np.abs(np.sin(8.0 * np.pi * x1)) + np.abs(np.cos(8.0 * np.pi * x2))
    )
    return np.sign(easy_mu_sin_pi_x1_plus_cos_pi_x2(x)) * rough_magnitude


def correlated_mu_eps005(x: np.ndarray) -> np.ndarray:
    """Mostly smooth outcome regression with a small rough correlated component."""
    return 0.95 * easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 0.05 * correlated_hard_component_g2(x)


def correlated_pi_1(x: np.ndarray) -> np.ndarray:
    """First correlated treatment regression in the 1.5.9 family."""
    return easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 0.05 * correlated_hard_component_g2(x)


def correlated_pi_2(x: np.ndarray) -> np.ndarray:
    """Second correlated treatment regression in the 1.5.9 family."""
    return easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 0.10 * correlated_hard_component_g2(x)


def correlated_pi_3(x: np.ndarray) -> np.ndarray:
    """Third correlated treatment regression in the 1.5.9 family."""
    return easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 0.20 * correlated_hard_component_g2(x)


def correlated_wide_pi_1(x: np.ndarray) -> np.ndarray:
    """First wide-range correlated treatment regression."""
    return easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 0.05 * correlated_hard_component_g2(x)


def correlated_wide_pi_2(x: np.ndarray) -> np.ndarray:
    """Second wide-range correlated treatment regression."""
    return easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 0.50 * correlated_hard_component_g2(x)


def correlated_wide_pi_3(x: np.ndarray) -> np.ndarray:
    """Third wide-range correlated treatment regression."""
    return easy_mu_sin_pi_x1_plus_cos_pi_x2(x) + 1.00 * correlated_hard_component_g2(x)


def sin_pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sin(pi x) using the first coordinate of x."""
    return np.sin(np.pi * x[:, [0]])


def unit_variance_correlated_g2(x: np.ndarray) -> np.ndarray:
    """Return a rough 1D component aligned with the sign of sin(pi x)."""
    x0 = x[:, [0]]
    rough = 0.5 * np.abs(np.sin(8.0 * np.pi * x0)) + 0.5 * np.abs(np.sin(16.0 * np.pi * x0))
    return np.sign(sin_pi_first_coordinate(x)) * rough


def unit_variance_correlated_mu_eps005(x: np.ndarray) -> np.ndarray:
    """Mostly smooth 1D outcome regression with a small rough correlated component."""
    return 0.95 * sin_pi_first_coordinate(x) + 0.05 * unit_variance_correlated_g2(x)


def unit_variance_correlated_pi_1(x: np.ndarray) -> np.ndarray:
    """First 1D treatment regression in the unit-variance correlated family."""
    return sin_pi_first_coordinate(x) + 0.05 * unit_variance_correlated_g2(x)


def unit_variance_correlated_pi_2(x: np.ndarray) -> np.ndarray:
    """Second 1D treatment regression in the unit-variance correlated family."""
    return sin_pi_first_coordinate(x) + 0.50 * unit_variance_correlated_g2(x)


def unit_variance_correlated_pi_3(x: np.ndarray) -> np.ndarray:
    """Third 1D treatment regression in the unit-variance correlated family."""
    return sin_pi_first_coordinate(x) + 1.00 * unit_variance_correlated_g2(x)


_SHARED_RESIDUAL_GRID = np.linspace(-1.0, 1.0, 200001, dtype=float).reshape(-1, 1)


def shared_residual_easy_signal(x: np.ndarray) -> np.ndarray:
    """Return the easy 1D signal used in the shared residual-correlation family."""
    return sin_pi_first_coordinate(x)


def _shared_residual_hard_signal_raw(x: np.ndarray) -> np.ndarray:
    """Return the rough aligned component before centering and variance normalization."""
    x0 = x[:, [0]]
    rough = 0.5 * np.abs(np.sin(8.0 * np.pi * x0)) + 0.5 * np.abs(np.sin(16.0 * np.pi * x0))
    return np.sign(shared_residual_easy_signal(x)) * rough


_SHARED_RESIDUAL_HARD_MEAN = float(np.mean(_shared_residual_hard_signal_raw(_SHARED_RESIDUAL_GRID)))
_SHARED_RESIDUAL_HARD_STD = float(
    np.std(_shared_residual_hard_signal_raw(_SHARED_RESIDUAL_GRID) - _SHARED_RESIDUAL_HARD_MEAN)
)


def shared_residual_hard_signal(x: np.ndarray) -> np.ndarray:
    """Return the centered, unit-variance rough component shared by mu and pi."""
    return (_shared_residual_hard_signal_raw(x) - _SHARED_RESIDUAL_HARD_MEAN) / _SHARED_RESIDUAL_HARD_STD


def _shared_residual_combo_scale(h_weight: float) -> float:
    """Return the deterministic scale used to keep Var(g + h_weight * h) fixed."""
    combo = shared_residual_easy_signal(_SHARED_RESIDUAL_GRID) + h_weight * shared_residual_hard_signal(
        _SHARED_RESIDUAL_GRID
    )
    return float(np.std(combo))


_SHARED_RESIDUAL_MU_SCALE = _shared_residual_combo_scale(1.0)
_SHARED_RESIDUAL_PI_1_SCALE = _shared_residual_combo_scale(0.5)
_SHARED_RESIDUAL_PI_2_SCALE = _shared_residual_combo_scale(1.0)
_SHARED_RESIDUAL_PI_3_SCALE = _shared_residual_combo_scale(2.0)


def shared_residual_mu(x: np.ndarray) -> np.ndarray:
    """Outcome regression in the shared hard-component residual-correlation family."""
    combo = shared_residual_easy_signal(x) + 1.0 * shared_residual_hard_signal(x)
    return combo / _SHARED_RESIDUAL_MU_SCALE


def shared_residual_pi_1(x: np.ndarray) -> np.ndarray:
    """First variance-normalized treatment regression in the shared residual family."""
    combo = shared_residual_easy_signal(x) + 0.5 * shared_residual_hard_signal(x)
    return combo / _SHARED_RESIDUAL_PI_1_SCALE


def shared_residual_pi_2(x: np.ndarray) -> np.ndarray:
    """Second variance-normalized treatment regression in the shared residual family."""
    combo = shared_residual_easy_signal(x) + 1.0 * shared_residual_hard_signal(x)
    return combo / _SHARED_RESIDUAL_PI_2_SCALE


def shared_residual_pi_3(x: np.ndarray) -> np.ndarray:
    """Third variance-normalized treatment regression in the shared residual family."""
    combo = shared_residual_easy_signal(x) + 2.0 * shared_residual_hard_signal(x)
    return combo / _SHARED_RESIDUAL_PI_3_SCALE


FUNCTION_REGISTRY = {
    "sin_2pi_first_coordinate": sin_2pi_first_coordinate,
    "sin_4pi_first_coordinate": sin_4pi_first_coordinate,
    "sin_8pi_first_coordinate": sin_8pi_first_coordinate,
    "sign_sin_2pi_times_sin_2pi_first_coordinate": sign_sin_2pi_times_sin_2pi_first_coordinate,
    "sign_sin_2pi_times_sin_4pi_first_coordinate": sign_sin_2pi_times_sin_4pi_first_coordinate,
    "sign_sin_2pi_times_sin_8pi_first_coordinate": sign_sin_2pi_times_sin_8pi_first_coordinate,
    "sign_sin_2pi_times_abs_sin_2pi_first_coordinate": sign_sin_2pi_times_abs_sin_2pi_first_coordinate,
    "sign_sin_2pi_times_abs_sin_4pi_first_coordinate": sign_sin_2pi_times_abs_sin_4pi_first_coordinate,
    "sign_sin_2pi_times_abs_sin_8pi_first_coordinate": sign_sin_2pi_times_abs_sin_8pi_first_coordinate,
    "progressive_pi_1_first_coordinate": progressive_pi_1_first_coordinate,
    "progressive_pi_2_first_coordinate": progressive_pi_2_first_coordinate,
    "progressive_pi_3_first_coordinate": progressive_pi_3_first_coordinate,
    "progressive_pi_4_first_coordinate": progressive_pi_4_first_coordinate,
    "fixed_overlap_pi_1_first_coordinate": fixed_overlap_pi_1_first_coordinate,
    "fixed_overlap_pi_2_first_coordinate": fixed_overlap_pi_2_first_coordinate,
    "fixed_overlap_pi_3_first_coordinate": fixed_overlap_pi_3_first_coordinate,
    "fixed_overlap_pi_4_first_coordinate": fixed_overlap_pi_4_first_coordinate,
    "sin_2pi_four_coordinates": sin_2pi_four_coordinates,
    "fixed_overlap_d4_pi_1": fixed_overlap_d4_pi_1,
    "fixed_overlap_d4_pi_2": fixed_overlap_d4_pi_2,
    "fixed_overlap_d4_pi_3": fixed_overlap_d4_pi_3,
    "isolated_d4_pi_1": isolated_d4_pi_1,
    "isolated_d4_pi_2": isolated_d4_pi_2,
    "isolated_d4_pi_3": isolated_d4_pi_3,
    "isolated_d4_pi_4": isolated_d4_pi_4,
    "easy_mu_sin_pi_x1_plus_cos_pi_x2": easy_mu_sin_pi_x1_plus_cos_pi_x2,
    "increasing_beta_pi_1": increasing_beta_pi_1,
    "increasing_beta_pi_2": increasing_beta_pi_2,
    "increasing_beta_pi_3": increasing_beta_pi_3,
    "increasing_beta_pi_4": increasing_beta_pi_4,
    "correlated_mu_eps005": correlated_mu_eps005,
    "correlated_pi_1": correlated_pi_1,
    "correlated_pi_2": correlated_pi_2,
    "correlated_pi_3": correlated_pi_3,
    "correlated_wide_pi_1": correlated_wide_pi_1,
    "correlated_wide_pi_2": correlated_wide_pi_2,
    "correlated_wide_pi_3": correlated_wide_pi_3,
    "sin_pi_first_coordinate": sin_pi_first_coordinate,
    "unit_variance_correlated_mu_eps005": unit_variance_correlated_mu_eps005,
    "unit_variance_correlated_pi_1": unit_variance_correlated_pi_1,
    "unit_variance_correlated_pi_2": unit_variance_correlated_pi_2,
    "unit_variance_correlated_pi_3": unit_variance_correlated_pi_3,
    "shared_residual_mu": shared_residual_mu,
    "shared_residual_pi_1": shared_residual_pi_1,
    "shared_residual_pi_2": shared_residual_pi_2,
    "shared_residual_pi_3": shared_residual_pi_3,
}

FUNCTION_LABELS = {
    "sin_2pi_first_coordinate": r"$\sin(2\pi x)$",
    "sin_4pi_first_coordinate": r"$\sin(4\pi x)$",
    "sin_8pi_first_coordinate": r"$\sin(8\pi x)$",
    "sign_sin_2pi_times_sin_2pi_first_coordinate": r"$\operatorname{sign}(\sin(2\pi x))\sin(2\pi x)$",
    "sign_sin_2pi_times_sin_4pi_first_coordinate": r"$\operatorname{sign}(\sin(2\pi x))\sin(4\pi x)$",
    "sign_sin_2pi_times_sin_8pi_first_coordinate": r"$\operatorname{sign}(\sin(2\pi x))\sin(8\pi x)$",
    "sign_sin_2pi_times_abs_sin_2pi_first_coordinate": r"$\operatorname{sign}(\sin(2\pi x))|\sin(2\pi x)|$",
    "sign_sin_2pi_times_abs_sin_4pi_first_coordinate": r"$\operatorname{sign}(\sin(2\pi x))|\sin(4\pi x)|$",
    "sign_sin_2pi_times_abs_sin_8pi_first_coordinate": r"$\operatorname{sign}(\sin(2\pi x))|\sin(8\pi x)|$",
    "progressive_pi_1_first_coordinate": r"$0.25\,\mu(x)+\sqrt{1-0.25^2}\sin(4\pi x)$",
    "progressive_pi_2_first_coordinate": r"$0.5\,\mu(x)+\sqrt{1-0.5^2}\sin(8\pi x)$",
    "progressive_pi_3_first_coordinate": r"$0.75\,\mu(x)+\sqrt{1-0.75^2}\sqrt{0.5}\,\operatorname{sign}(\sin(8\pi x))$",
    "progressive_pi_4_first_coordinate": r"$0.9\,\mu(x)+\sqrt{1-0.9^2}\,r(x)$",
    "fixed_overlap_pi_1_first_coordinate": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\sqrt{2}\cos(2\pi x)$",
    "fixed_overlap_pi_2_first_coordinate": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\sqrt{2}\cos(8\pi x)$",
    "fixed_overlap_pi_3_first_coordinate": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\operatorname{sign}(\sin(8\pi x))$",
    "fixed_overlap_pi_4_first_coordinate": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\operatorname{sign}(\sin(64\pi x))$",
    "sin_2pi_four_coordinates": r"$0.5\sum_{j=1}^4 \sin(2\pi x_j)$",
    "fixed_overlap_d4_pi_1": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\,0.5\sum_{j=1}^4 \sqrt{2}\cos(2\pi x_j)$",
    "fixed_overlap_d4_pi_2": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\,0.5\sum_{j=1}^4 \sqrt{2}\cos(8\pi x_j)$",
    "fixed_overlap_d4_pi_3": r"$0.98\,\mu(x)+\sqrt{1-0.98^2}\,0.5\sum_{j=1}^4 \operatorname{sign}(\sin(32\pi x_j))$",
    "isolated_d4_pi_1": r"$0.98\,\sin(2\pi x_1)+\sqrt{1-0.98^2}\frac{\sin(2\pi x_2)+\sin(2\pi x_3)+\sin(2\pi x_4)}{\sqrt{3}}$",
    "isolated_d4_pi_2": r"$0.98\,\sin(2\pi x_1)+\sqrt{1-0.98^2}\frac{\sin(8\pi x_2)+\sin(8\pi x_3)+\sin(8\pi x_4)}{\sqrt{3}}$",
    "isolated_d4_pi_3": r"$0.98\,\sin(2\pi x_1)+\sqrt{1-0.98^2}\operatorname{sign}(\prod_{j=2}^4\sin(8\pi x_j))$",
    "isolated_d4_pi_4": r"$0.98\,\sin(2\pi x_1)+\sqrt{1-0.98^2}\operatorname{sign}(\prod_{j=2}^4\sin(32\pi x_j))$",
    "easy_mu_sin_pi_x1_plus_cos_pi_x2": r"$\sin(\pi x_1)+\cos(\pi x_2)$",
    "increasing_beta_pi_1": r"$\mu(x)+0.05\frac{\sin(2\pi x_1)+\cos(2\pi x_2)}{\sqrt{2}}$",
    "increasing_beta_pi_2": r"$\mu(x)+0.18\,\operatorname{sign}(\sin(8\pi x_1))\operatorname{sign}(\cos(8\pi x_2))$",
    "increasing_beta_pi_3": r"$\mu(x)+0.20\,\operatorname{sign}(\sin(8\pi x_1))\operatorname{sign}(\cos(8\pi x_2))$",
    "increasing_beta_pi_4": r"$\mu(x)+0.25\,\operatorname{sign}(\sin(8\pi x_1))\operatorname{sign}(\cos(8\pi x_2))$",
    "correlated_mu_eps005": r"$0.95\,g_1(x)+0.05\,g_2(x)$",
    "correlated_pi_1": r"$g_1(x)+0.05\,g_2(x)$",
    "correlated_pi_2": r"$g_1(x)+0.10\,g_2(x)$",
    "correlated_pi_3": r"$g_1(x)+0.20\,g_2(x)$",
    "correlated_wide_pi_1": r"$g_1(x)+0.05\,g_2(x)$",
    "correlated_wide_pi_2": r"$g_1(x)+0.50\,g_2(x)$",
    "correlated_wide_pi_3": r"$g_1(x)+1.00\,g_2(x)$",
    "sin_pi_first_coordinate": r"$\sin(\pi x)$",
    "unit_variance_correlated_mu_eps005": r"$0.95\,\sin(\pi x)+0.05\,g_2(x)$",
    "unit_variance_correlated_pi_1": r"$\sin(\pi x)+0.05\,g_2(x)$",
    "unit_variance_correlated_pi_2": r"$\sin(\pi x)+0.50\,g_2(x)$",
    "unit_variance_correlated_pi_3": r"$\sin(\pi x)+1.00\,g_2(x)$",
    "shared_residual_mu": r"$s_\mu^{-1}\{\sin(\pi x)+h(x)\}$",
    "shared_residual_pi_1": r"$s_{0.5}^{-1}\{\sin(\pi x)+0.5\,h(x)\}$",
    "shared_residual_pi_2": r"$s_{1.0}^{-1}\{\sin(\pi x)+1.0\,h(x)\}$",
    "shared_residual_pi_3": r"$s_{2.0}^{-1}\{\sin(\pi x)+2.0\,h(x)\}$",
}


def normalize_exp_id(exp_id: str) -> tuple[str, str]:
    """Return the storage id and display id for an experiment identifier."""
    if "_" in exp_id:
        storage_id = exp_id
        display_id = exp_id.replace("_", ".")
        return storage_id, display_id

    parts = exp_id.split(".")
    if len(parts) < 3:
        raise ValueError(
            "Experiment identifiers should look like '1.1.2' or the storage form '1.1_2'."
        )
    storage_id = f"{'.'.join(parts[:-1])}_{parts[-1]}"
    display_id = exp_id
    return storage_id, display_id


def plm_uniform_noise_dgp_generator(param_config: dict, seed: int | None = None) -> PartialLinearModelUniformNoiseDGP:
    """Build a partial linear DGP from a serializable parameter configuration."""
    del seed  # Reserved for future generators that may use NumPy randomness.
    func_mu = FUNCTION_REGISTRY[param_config["func_mu_name"]]
    func_pi = FUNCTION_REGISTRY[param_config["func_pi_name"]]
    if "beta" in param_config:
        beta = float(param_config["beta"])
    elif param_config.get("beta_sampler_name") == "uniform":
        beta = float(np.random.uniform(float(param_config["beta_low"]), float(param_config["beta_high"])))
    else:
        raise KeyError("PLM DGP config must provide either 'beta' or a supported beta sampler.")
    return PartialLinearModelUniformNoiseDGP(
        beta=beta,
        func_mu=func_mu,
        func_pi=func_pi,
        d=int(param_config["d"]),
        sigma_u=float(param_config["sigma_u"]),
        sigma_eps=float(param_config["sigma_eps"]),
        name="partial_linear_uniform_noise",
    )


def make_plm_dml_estimator(method_config: dict) -> PLMDMLEstimator:
    """Construct a neural DML estimator for the PLM."""
    hyper_parameters = {
        "L": method_config["L"],
        "N": method_config["N"],
        "lambda_mu": method_config["lambda_mu"],
        "lambda_pi": method_config["lambda_pi"],
        "niter": method_config["niter"],
        "lr": method_config["lr"],
        "batch_size": method_config["batch_size"],
        "seed": method_config.get("seed"),
    }
    return PLMDMLEstimator(
        name="dml_nn",
        hyper_parameters=hyper_parameters,
        d=int(method_config["d"]),
        device=str(method_config.get("device", "cpu")),
    )


def make_plm_oracle_estimator(method_config: dict) -> PLMOracleAIPWEstimator:
    """Construct an oracle AIPW estimator for the PLM."""
    return PLMOracleAIPWEstimator(
        name="oracle_aipw",
        ground_truth_func_mu=FUNCTION_REGISTRY[method_config["func_mu_name"]],
        ground_truth_func_pi=FUNCTION_REGISTRY[method_config["func_pi_name"]],
    )


def make_plm_dml_tracking_estimator(method_config: dict) -> PLMDMLOracleTrackingEstimator:
    """Construct a neural DML estimator that records oracle nuisance MSE paths."""
    hyper_parameters = {
        "L": method_config["L"],
        "N": method_config["N"],
        "lambda_mu": method_config["lambda_mu"],
        "lambda_pi": method_config["lambda_pi"],
        "niter": method_config["niter"],
        "lr": method_config["lr"],
        "batch_size": method_config["batch_size"],
        "seed": method_config.get("seed"),
    }
    return PLMDMLOracleTrackingEstimator(
        name="dml_nn_tracking",
        hyper_parameters=hyper_parameters,
        d=int(method_config["d"]),
        device=str(method_config.get("device", "cpu")),
    )


def _make_trial_seeded_dml_factory(method_config: dict):
    """Return a factory that injects the trial seed into the DML estimator config."""
    base_config = deepcopy(method_config)

    def factory(*, trial_seed: int | None = None) -> PLMDMLEstimator:
        config = deepcopy(base_config)
        if trial_seed is not None:
            config["seed"] = int(trial_seed)
        return make_plm_dml_estimator(config)

    return factory


def _make_fixed_dml_factory(method_config: dict):
    """Return a factory for a fixed-seed DML estimator."""
    base_config = deepcopy(method_config)

    def factory() -> PLMDMLEstimator:
        return make_plm_dml_estimator(deepcopy(base_config))

    return factory


def _make_oracle_factory(method_config: dict):
    """Return a factory for the oracle estimator with a uniform call signature."""
    base_config = deepcopy(method_config)

    def factory(
        *,
        trial_seed: int | None = None,
        dgp_config: dict | None = None,
    ) -> PLMOracleAIPWEstimator:
        del trial_seed
        config = deepcopy(base_config)
        if config.get("follows_dgp_pi"):
            if dgp_config is None:
                raise ValueError("Oracle factory with follows_dgp_pi=True requires dgp_config.")
            config["func_pi_name"] = dgp_config["func_pi_name"]
        return make_plm_oracle_estimator(config)

    return factory


def _make_trial_seeded_tracking_factory(method_config: dict):
    """Return a factory that injects the trial seed into the tracking estimator config."""
    base_config = deepcopy(method_config)

    def factory(*, trial_seed: int | None = None) -> PLMDMLOracleTrackingEstimator:
        config = deepcopy(base_config)
        if trial_seed is not None:
            config["seed"] = int(trial_seed)
        return make_plm_dml_tracking_estimator(config)

    return factory


def _format_lambda_label(value: float) -> str:
    """Format a positive regularization value in compact scientific notation."""
    formatted = f"{float(value):.3e}"
    mantissa, exponent = formatted.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def build_tracking_experiment(
    exp_id: str,
    lambda_values: list[float],
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    tracking_source: str = "d2",
    validation_n: int | None = None,
) -> PLMEvaluator:
    """Build a nuisance-tracking experiment with one estimator per lambda choice."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": "sin_2pi_first_coordinate",
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    estimators = []
    for lambda_value in lambda_values:
        lambda_label = _format_lambda_label(lambda_value)
        tracking_method_config = {
            "L": 3,
            "N": 512,
            "lambda_mu": float(lambda_value),
            "lambda_pi": float(lambda_value),
            "lambda_label": lambda_label,
            "tracking_source": tracking_source,
            "niter": 200,
            "lr": 1e-3,
            "batch_size": 1024,
            "device": device,
            "seed_mode": "trial_seed",
            "d": 1,
        }
        if validation_n is not None:
            tracking_method_config["validation_n"] = int(validation_n)
        estimators.append(
            {
                "name": f"dml_nn_tracking_lambda_{lambda_label}",
                "is_oracle": True,
                "factory_name": "make_plm_dml_tracking_estimator",
                "method_config": deepcopy(tracking_method_config),
                "accepts_trial_seed": True,
                "factory": _make_trial_seeded_tracking_factory(tracking_method_config),
            }
        )

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_1(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build evaluator configuration for experiment family 1.1."""
    return build_plm_sine_experiment(
        exp_id=exp_id,
        beta=0.0,
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
        trial_seeded_dml=False,
    )


def build_experiment_1_2(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build evaluator configuration for experiment family 1.2."""
    return build_plm_sine_experiment(
        exp_id=exp_id,
        beta=0.5,
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
        trial_seeded_dml=False,
    )


def build_experiment_1_3(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    ) -> PLMEvaluator:
    """Build evaluator configuration for experiment family 1.3."""
    return build_random_beta_experiment(
        exp_id=exp_id,
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
        trial_seeded_dml=False,
    )


def build_experiment_1_3_2(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build evaluator configuration for the trial-seeded 1.3.2 experiment."""
    return build_random_beta_experiment(
        exp_id=exp_id,
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
        trial_seeded_dml=True,
    )


def build_experiment_1_4_1(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build evaluator configuration for nuisance-path tracking in the random-beta PLM."""
    return build_tracking_experiment(
        exp_id=exp_id,
        lambda_values=[1e-4],
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
    )


def build_experiment_1_4_2(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the lambda-sweep nuisance-tracking experiment."""
    return build_tracking_experiment(
        exp_id=exp_id,
        lambda_values=[2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4],
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
    )


def build_experiment_1_4_3(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the wide-range lambda-sweep nuisance-tracking experiment."""
    return build_tracking_experiment(
        exp_id=exp_id,
        lambda_values=[(5.0**power) * 1e-4 for power in (-3, -2, -1, 0, 1, 2, 3)],
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
    )


def build_experiment_1_4_4(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build dual-source nuisance-path tracking for the wide lambda sweep."""
    return build_tracking_experiment(
        exp_id=exp_id,
        lambda_values=[(5.0**power) * 1e-4 for power in (-3, -2, -1, 0, 1, 2, 3)],
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
        tracking_source="validation",
        validation_n=1024,
    )


def build_experiment_1_5(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the pi-complexity experiment at fixed n and lambda."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": [
            "sin_2pi_first_coordinate",
            "sin_4pi_first_coordinate",
            "sin_8pi_first_coordinate",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_random_beta_experiment(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    trial_seeded_dml: bool = False,
) -> PLMEvaluator:
    """Build a random-beta sine/sine PLM experiment."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": "sin_2pi_first_coordinate",
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [256, 512, 1024, 2048],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 1e-4,
        "lambda_pi": 1e-4,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed": seed_offset,
        "d": 1,
    }
    if trial_seeded_dml:
        dml_method_config["seed_mode"] = "trial_seed"
        dml_method_config.pop("seed", None)
    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": "sin_2pi_first_coordinate",
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": trial_seeded_dml,
            "factory": _make_trial_seeded_dml_factory(dml_method_config)
            if trial_seeded_dml
            else _make_fixed_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "factory": _make_oracle_factory(oracle_method_config) if trial_seeded_dml else (lambda cfg=deepcopy(oracle_method_config): make_plm_oracle_estimator(cfg)),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_plm_sine_experiment(
    exp_id: str,
    beta: float,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    trial_seeded_dml: bool = False,
) -> PLMEvaluator:
    """Build a sine/sine PLM experiment with configurable beta."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": "sin_2pi_first_coordinate",
        "beta": float(beta),
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [256, 512, 1024, 2048],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 1e-4,
        "lambda_pi": 1e-4,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed": seed_offset,
        "d": 1,
    }
    if trial_seeded_dml:
        dml_method_config["seed_mode"] = "trial_seed"
        dml_method_config.pop("seed", None)
    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": "sin_2pi_first_coordinate",
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": trial_seeded_dml,
            "factory": _make_trial_seeded_dml_factory(dml_method_config)
            if trial_seeded_dml
            else _make_fixed_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "factory": _make_oracle_factory(oracle_method_config) if trial_seeded_dml else (lambda cfg=deepcopy(oracle_method_config): make_plm_oracle_estimator(cfg)),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_2(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the signed pi-complexity experiment at fixed n and lambda."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": [
            "sign_sin_2pi_times_sin_2pi_first_coordinate",
            "sign_sin_2pi_times_sin_4pi_first_coordinate",
            "sign_sin_2pi_times_sin_8pi_first_coordinate",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_3(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the signed-absolute pi-complexity experiment at fixed n and lambda."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": [
            "sign_sin_2pi_times_abs_sin_2pi_first_coordinate",
            "sign_sin_2pi_times_abs_sin_4pi_first_coordinate",
            "sign_sin_2pi_times_abs_sin_8pi_first_coordinate",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_4(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the four-level progressive pi-complexity experiment."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": [
            "progressive_pi_1_first_coordinate",
            "progressive_pi_2_first_coordinate",
            "progressive_pi_3_first_coordinate",
            "progressive_pi_4_first_coordinate",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_5(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the fixed-overlap pi-complexity experiment."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": [
            "fixed_overlap_pi_1_first_coordinate",
            "fixed_overlap_pi_2_first_coordinate",
            "fixed_overlap_pi_3_first_coordinate",
            "fixed_overlap_pi_4_first_coordinate",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_6(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the four-dimensional fixed-overlap pi-complexity experiment."""
    dgp_param_grid = {
        "d": 4,
        "func_mu_name": "sin_2pi_four_coordinates",
        "func_pi_name": [
            "fixed_overlap_d4_pi_1",
            "fixed_overlap_d4_pi_2",
            "fixed_overlap_d4_pi_3",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 4,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_four_coordinates",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_7(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the d=4 experiment that isolates pi difficulty from mu difficulty."""
    dgp_param_grid = {
        "d": 4,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": [
            "isolated_d4_pi_1",
            "isolated_d4_pi_2",
            "isolated_d4_pi_3",
            "isolated_d4_pi_4",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 4,
    }

    oracle_method_config = {
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_8(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the easy-mu family where pi and beta error rise together in pilot runs."""
    dgp_param_grid = {
        "d": 4,
        "func_mu_name": "easy_mu_sin_pi_x1_plus_cos_pi_x2",
        "func_pi_name": [
            "increasing_beta_pi_1",
            "increasing_beta_pi_2",
            "increasing_beta_pi_3",
            "increasing_beta_pi_4",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 4,
    }

    oracle_method_config = {
        "func_mu_name": "easy_mu_sin_pi_x1_plus_cos_pi_x2",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_9(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the correlated g1/g2 family proposed after the 1.5.8 follow-up."""
    dgp_param_grid = {
        "d": 4,
        "func_mu_name": "correlated_mu_eps005",
        "func_pi_name": [
            "correlated_pi_1",
            "correlated_pi_2",
            "correlated_pi_3",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 4,
    }

    oracle_method_config = {
        "func_mu_name": "correlated_mu_eps005",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_10(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the wide-range correlated g1/g2 family requested after 1.5.9."""
    dgp_param_grid = {
        "d": 4,
        "func_mu_name": "correlated_mu_eps005",
        "func_pi_name": [
            "correlated_wide_pi_1",
            "correlated_wide_pi_2",
            "correlated_wide_pi_3",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": 0.5,
        "sigma_eps": 0.5,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 4,
    }

    oracle_method_config = {
        "func_mu_name": "correlated_mu_eps005",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_11(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the 1D unit-variance correlated family motivated by denominator stability."""
    unit_variance_scale = math.sqrt(3.0)
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "unit_variance_correlated_mu_eps005",
        "func_pi_name": [
            "unit_variance_correlated_pi_1",
            "unit_variance_correlated_pi_2",
            "unit_variance_correlated_pi_3",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": unit_variance_scale,
        "sigma_eps": unit_variance_scale,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "unit_variance_correlated_mu_eps005",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


def build_experiment_1_5_12(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build the shared hard-component family aimed at aligned fitted residuals."""
    unit_variance_scale = math.sqrt(3.0)
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "shared_residual_mu",
        "func_pi_name": [
            "shared_residual_pi_1",
            "shared_residual_pi_2",
            "shared_residual_pi_3",
        ],
        "beta_sampler_name": "uniform",
        "beta_low": -0.5,
        "beta_high": 0.5,
        "sigma_u": unit_variance_scale,
        "sigma_eps": unit_variance_scale,
        "n_test": 10000,
        "n": [1024],
    }

    dml_method_config = {
        "L": 3,
        "N": 512,
        "lambda_mu": 2e-5,
        "lambda_pi": 2e-5,
        "niter": 200,
        "lr": 1e-3,
        "batch_size": 1024,
        "device": device,
        "seed_mode": "trial_seed",
        "d": 1,
    }

    oracle_method_config = {
        "func_mu_name": "shared_residual_mu",
        "func_pi_name": None,
        "follows_dgp_pi": True,
    }

    estimators = [
        {
            "name": "dml_nn",
            "is_oracle": False,
            "factory_name": "make_plm_dml_estimator",
            "method_config": deepcopy(dml_method_config),
            "accepts_trial_seed": True,
            "factory": _make_trial_seeded_dml_factory(dml_method_config),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "accepts_dgp_config": True,
            "factory": _make_oracle_factory(oracle_method_config),
        },
    ]

    return PLMEvaluator(
        exp_name=EXPERIMENT_NAME,
        exp_id=exp_id,
        dgp_generator=plm_uniform_noise_dgp_generator,
        dgp_param_grid=dgp_param_grid,
        estimators=estimators,
        n_trials=n_trials,
        seed_offset=seed_offset,
        result_root=result_root,
    )


EXPERIMENT_FAMILY_BUILDERS = {
    "1.1": build_experiment_1_1,
    "1.2": build_experiment_1_2,
    "1.3": build_experiment_1_3,
    "1.4": build_experiment_1_4_1,
    "1.5": build_experiment_1_5,
}

EXPERIMENT_ID_BUILDERS = {
    "1.1_1": build_experiment_1_1,
    "1.1_2": build_experiment_1_1,
    "1.2_1": build_experiment_1_2,
    "1.2_2": build_experiment_1_2,
    "1.3_1": build_experiment_1_3,
    "1.3_2": build_experiment_1_3_2,
    "1.4_1": build_experiment_1_4_1,
    "1.4_2": build_experiment_1_4_2,
    "1.4_3": build_experiment_1_4_3,
    "1.4_4": build_experiment_1_4_4,
    "1.5_1": build_experiment_1_5,
    "1.5_2": build_experiment_1_5_2,
    "1.5_3": build_experiment_1_5_3,
    "1.5_4": build_experiment_1_5_4,
    "1.5_5": build_experiment_1_5_5,
    "1.5_6": build_experiment_1_5_6,
    "1.5_7": build_experiment_1_5_7,
    "1.5_8": build_experiment_1_5_8,
    "1.5_9": build_experiment_1_5_9,
    "1.5_10": build_experiment_1_5_10,
    "1.5_11": build_experiment_1_5_11,
    "1.5_12": build_experiment_1_5_12,
}


def build_evaluator_from_exp_id(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build an evaluator from the experiment family encoded in exp_id."""
    storage_id, _ = normalize_exp_id(exp_id)
    if storage_id in EXPERIMENT_ID_BUILDERS:
        builder = EXPERIMENT_ID_BUILDERS[storage_id]
    else:
        family = storage_id.split("_", 1)[0]
        if family not in EXPERIMENT_FAMILY_BUILDERS:
            raise ValueError(f"Unknown experiment family '{family}'.")
        builder = EXPERIMENT_FAMILY_BUILDERS[family]
    return builder(
        exp_id=storage_id,
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
    )
