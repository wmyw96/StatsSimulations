"""Experiment definitions for partial linear model simulations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from simlab.dgp.partial_linear import PartialLinearModelUniformNoiseDGP
from simlab.estimators.plm_est import PLMDMLEstimator, PLMOracleAIPWEstimator
from simlab.evaluation.plm_eval import PLMEvaluator

EXPERIMENT_NAME = "plm"
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parents[2] / "simulation_results"


def sin_2pi_first_coordinate(x: np.ndarray) -> np.ndarray:
    """Return sin(2 pi x) using the first coordinate of x."""
    return np.sin(2.0 * np.pi * x[:, [0]])


FUNCTION_REGISTRY = {
    "sin_2pi_first_coordinate": sin_2pi_first_coordinate,
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
    return PartialLinearModelUniformNoiseDGP(
        beta=float(param_config["beta"]),
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


def build_experiment_1_1(
    exp_id: str,
    n_trials: int,
    seed_offset: int = 0,
    device: str = "cpu",
    result_root: str | Path = DEFAULT_RESULT_ROOT,
) -> PLMEvaluator:
    """Build evaluator configuration for experiment family 1.1."""
    dgp_param_grid = {
        "d": 1,
        "func_mu_name": "sin_2pi_first_coordinate",
        "func_pi_name": "sin_2pi_first_coordinate",
        "beta": 0.0,
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
            "factory": lambda cfg=deepcopy(dml_method_config): make_plm_dml_estimator(cfg),
        },
        {
            "name": "oracle_aipw",
            "is_oracle": True,
            "factory_name": "make_plm_oracle_estimator",
            "method_config": deepcopy(oracle_method_config),
            "factory": lambda cfg=deepcopy(oracle_method_config): make_plm_oracle_estimator(cfg),
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
    family = storage_id.split("_", 1)[0]
    if family not in EXPERIMENT_FAMILY_BUILDERS:
        raise ValueError(f"Unknown experiment family '{family}'.")
    return EXPERIMENT_FAMILY_BUILDERS[family](
        exp_id=storage_id,
        n_trials=n_trials,
        seed_offset=seed_offset,
        device=device,
        result_root=result_root,
    )
