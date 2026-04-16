"""Structured records shared across DGPs, estimators, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    np = Any


@dataclass
class SampledData:
    """A generic sampled dataset with observed and optional oracle arrays."""

    observed: dict[str, np.ndarray]
    oracle: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EstimateResult:
    """Structured output returned by an estimator fit."""

    target: str
    estimate: float
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    """One evaluator row for a single estimator on a single trial."""

    study_name: str
    dgp_name: str
    estimator_name: str
    trial_id: int
    data_seed: int
    estimator_seed: int | None
    theta_true: float | None
    estimate_result: EstimateResult
    runtime_sec: float
    fit_status: str = "success"
    fit_message: str = ""
    dgp_config: dict[str, Any] = field(default_factory=dict)
    estimator_config: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
