"""Experiment definitions for partial linear model simulations."""

from __future__ import annotations

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

    def factory(*, trial_seed: int | None = None) -> PLMOracleAIPWEstimator:
        del trial_seed
        return make_plm_oracle_estimator(deepcopy(base_config))

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


EXPERIMENT_FAMILY_BUILDERS = {
    "1.1": build_experiment_1_1,
    "1.2": build_experiment_1_2,
    "1.3": build_experiment_1_3,
    "1.4": build_experiment_1_4_1,
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
