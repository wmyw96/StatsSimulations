"""Evaluation utilities for partial linear model experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from simlab.core.records import SampledData
from simlab.estimators.plm_est import PLMDMLEstimator
from simlab.evaluation.base import Evaluator

EstimatorFactory = Callable[[], Any]
DGPGenerator = Callable[[dict[str, Any], int | None], Any]


class PLMEvaluator(Evaluator):
    """Experiment runner for partial linear model simulations."""

    def __init__(
        self,
        exp_name: str,
        exp_id: str,
        dgp_generator: DGPGenerator,
        dgp_param_grid: dict[str, Any],
        estimators: list[dict[str, Any]],
        n_trials: int = 1,
        seed_offset: int = 0,
        result_root: str | Path = "simulation_results",
    ):
        super().__init__(
            name=exp_name,
            dgp_name=getattr(dgp_generator, "__name__", "dgp_generator"),
            dgp_param_grid=dgp_param_grid,
            estimators=estimators,
            n_trials=n_trials,
            seed=seed_offset,
        )
        self.exp_name = exp_name
        self.exp_id = exp_id
        self.dgp_generator = dgp_generator
        self.seed_offset = seed_offset
        self.result_root = Path(result_root)
        self.result_path = self.result_root / exp_name / f"{exp_id}.json"
        self.dgp_param_grid = deepcopy(dgp_param_grid)
        self.estimators = [self._normalize_estimator_spec(spec) for spec in estimators]
        self.results: dict[str, Any] | None = None

    def __run__(self) -> dict[str, Any]:
        """Alias requested by the experiment specification."""
        return self.run()

    def run(self) -> dict[str, Any]:
        """Run the configured experiment, resuming from disk when possible."""
        self.results = self._load_results()
        metadata_changed = self._refresh_loaded_results_metadata(self.results)
        results_changed = metadata_changed
        completed_trials = {
            (self._config_signature(record["dgp_config"]), int(record["trial_id"]))
            for record in self.results["trial_results"]
        }
        expanded_configs = self._expand_param_grid(self.dgp_param_grid)

        for dgp_config in expanded_configs:
            config_signature = self._config_signature(dgp_config)
            for trial_id in range(self.n_trials):
                if (config_signature, trial_id) in completed_trials:
                    continue

                trial_seed = self.seed_offset + trial_id
                trial_record = self._run_single_trial(
                    dgp_config=dgp_config,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                )
                self.results["trial_results"].append(trial_record)
                completed_trials.add((config_signature, trial_id))
                results_changed = True
                self.save()

        if results_changed:
            self.save()
        return self.results

    def save(self, path: str | None = None) -> None:
        """Persist the current experiment state to disk."""
        if self.results is None:
            raise RuntimeError("No results are available to save.")
        save_path = Path(path) if path is not None else self.result_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2, sort_keys=True)

    def query_results(
        self,
        param_config: dict[str, Any],
        mode: str = "summary",
    ) -> dict[str, dict[str, float | int]]:
        """Query saved results for a specific DGP configuration."""
        results = self._load_results()
        matching_trials = [
            record
            for record in results["trial_results"]
            if self._config_signature(record["dgp_config"]) == self._config_signature(param_config)
        ]

        if mode != "summary":
            raise ValueError("Only mode='summary' is currently implemented.")

        summary: dict[str, dict[str, float | int]] = {}
        for trial in matching_trials:
            for estimator_record in trial["estimator_results"]:
                method_name = estimator_record["estimator_name"]
                method_summary = summary.setdefault(
                    method_name,
                    {
                        "beta_hat_mse": 0.0,
                        "beta_init_mse": 0.0,
                        "mu_mse": 0.0,
                        "pi_mse": 0.0,
                        "num_trials": 0,
                    },
                )
                method_summary["beta_hat_mse"] += float(estimator_record["beta_sq_error"])
                method_summary["beta_init_mse"] += float(estimator_record["beta_init_sq_error"])
                method_summary["mu_mse"] += float(estimator_record["mu_mse"])
                method_summary["pi_mse"] += float(estimator_record["pi_mse"])
                method_summary["num_trials"] += 1

        for method_summary in summary.values():
            num_trials = int(method_summary["num_trials"])
            if num_trials == 0:
                continue
            method_summary["beta_hat_mse"] /= num_trials
            method_summary["beta_init_mse"] /= num_trials
            method_summary["mu_mse"] /= num_trials
            method_summary["pi_mse"] /= num_trials

        return summary

    def _run_single_trial(
        self,
        dgp_config: dict[str, Any],
        trial_id: int,
        trial_seed: int,
    ) -> dict[str, Any]:
        """Run one trial for one DGP configuration."""
        dgp = self._build_dgp(dgp_config=dgp_config, seed=trial_seed)
        train_seed = trial_seed + 1
        test_seed = trial_seed + 2
        train_n = int(dgp_config["n"])
        test_n = int(dgp_config["n_test"])

        data_oracle = dgp.sample(n=train_n, seed=train_seed, oracle=True)
        data_observed = SampledData(observed=deepcopy(data_oracle.observed))
        data_test = dgp.sample(n=test_n, seed=test_seed, oracle=True)
        beta_true = float(dgp.true_parameter())

        estimator_results = []
        for estimator_spec in self.estimators:
            estimator = self._instantiate_estimator(estimator_spec=estimator_spec, trial_seed=trial_seed)
            start_time = time.perf_counter()
            fit_data = data_oracle if estimator_spec["is_oracle"] else data_observed
            fit_result = estimator.fit(fit_data)
            runtime_sec = time.perf_counter() - start_time
            estimator_results.append(
                self._evaluate_estimator(
                    estimator=estimator,
                    estimator_spec=estimator_spec,
                    fit_result=fit_result,
                    beta_true=beta_true,
                    data_test=data_test,
                    runtime_sec=runtime_sec,
                )
            )

        return {
            "dgp_name": dgp.name,
            "dgp_config": deepcopy(dgp_config),
            "beta_true": beta_true,
            "trial_id": trial_id,
            "trial_seed": trial_seed,
            "train_seed": train_seed,
            "test_seed": test_seed,
            "estimator_results": estimator_results,
        }

    def _evaluate_estimator(
        self,
        estimator: Any,
        estimator_spec: dict[str, Any],
        fit_result: Any,
        beta_true: float,
        data_test: SampledData,
        runtime_sec: float,
    ) -> dict[str, Any]:
        """Evaluate one fitted estimator on the PLM metrics."""
        predictions = estimator.predict(data_test.observed["x"])
        mu_pred = _as_column(predictions["mu"], label="mu prediction")
        pi_pred = _as_column(predictions["pi"], label="pi prediction")
        mu_true = _as_column(data_test.oracle["mu_x"], label="oracle mu_x")
        pi_true = _as_column(data_test.oracle["pi_x"], label="oracle pi_x")
        beta_hat = float(fit_result.estimate)
        beta_sq_error = float((beta_hat - beta_true) ** 2)

        if estimator_spec["is_oracle"]:
            beta_initial = beta_hat
        else:
            diagnostics = getattr(fit_result, "diagnostics", {})
            beta_initial = float(diagnostics.get("beta_joint", beta_hat))
        diagnostics = getattr(fit_result, "diagnostics", {})

        result_record = {
            "estimator_name": estimator_spec["name"],
            "is_oracle": estimator_spec["is_oracle"],
            "factory_name": estimator_spec["factory_name"],
            "method_config": deepcopy(estimator_spec["method_config"]),
            "beta_hat": beta_hat,
            "beta_sq_error": beta_sq_error,
            "beta_initial": beta_initial,
            "beta_init_sq_error": float((beta_initial - beta_true) ** 2),
            "mu_mse": float(np.mean((mu_pred - mu_true) ** 2)),
            "pi_mse": float(np.mean((pi_pred - pi_true) ** 2)),
            "mu_pi_product_mean": float(np.mean(mu_pred * pi_pred)),
            "mu_pi_product_true_mean": float(np.mean(mu_true * pi_true)),
            "mu_pi_product_mse": float(np.mean((mu_pred * pi_pred - mu_true * pi_true) ** 2)),
            "runtime_sec": runtime_sec,
        }
        for key in ("epoch_grid", "mu_mse_path", "pi_mse_path", "tracking_split", "tracking_n"):
            if key in diagnostics:
                result_record[key] = deepcopy(diagnostics[key])
        return result_record

    def _build_dgp(self, dgp_config: dict[str, Any], seed: int | None) -> Any:
        """Build a DGP with a reproducible NumPy global state for the generator."""
        if seed is None:
            return self.dgp_generator(deepcopy(dgp_config), seed)

        old_state = np.random.get_state()
        np.random.seed(seed)
        try:
            return self.dgp_generator(deepcopy(dgp_config), seed)
        finally:
            np.random.set_state(old_state)

    def _load_results(self) -> dict[str, Any]:
        """Load persisted results or create a fresh result container."""
        if self.result_path.exists():
            with self.result_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self._validate_loaded_results(loaded)
            return loaded

        return {
            "exp_name": self.exp_name,
            "exp_id": self.exp_id,
            "dgp_generator_name": getattr(self.dgp_generator, "__name__", "dgp_generator"),
            "dgp_param_grid": deepcopy(self.dgp_param_grid),
            "n_trials": self.n_trials,
            "seed_offset": self.seed_offset,
            "estimator_specs": [
                {
                    "name": spec["name"],
                    "is_oracle": spec["is_oracle"],
                    "factory_name": spec["factory_name"],
                    "method_config": deepcopy(spec["method_config"]),
                }
                for spec in self.estimators
            ],
            "trial_results": [],
        }

    def _refresh_loaded_results_metadata(self, results: dict[str, Any]) -> bool:
        """Refresh mutable top-level metadata after loading a resumable result file."""
        changed = False
        completed_counts = {}
        for trial in results.get("trial_results", []):
            signature = self._config_signature(trial["dgp_config"])
            completed_counts[signature] = completed_counts.get(signature, 0) + 1

        target_n_trials = max(
            int(results.get("n_trials", 0)),
            int(self.n_trials),
            max(completed_counts.values(), default=0),
        )
        if int(results.get("n_trials", 0)) != target_n_trials:
            results["n_trials"] = target_n_trials
            changed = True

        expected_specs = self._serializable_estimator_specs()
        if self._normalize_loaded_estimator_specs(results.get("estimator_specs", [])) != expected_specs:
            results["estimator_specs"] = expected_specs
            changed = True

        if results.get("dgp_param_grid") != deepcopy(self.dgp_param_grid):
            results["dgp_param_grid"] = deepcopy(self.dgp_param_grid)
            changed = True

        if results.get("seed_offset") != self.seed_offset:
            results["seed_offset"] = self.seed_offset
            changed = True

        return changed

    @staticmethod
    def _config_signature(config: dict[str, Any]) -> str:
        """Create a stable signature for a serializable configuration dictionary."""
        return json.dumps(config, sort_keys=True)

    @staticmethod
    def _expand_param_grid(param_grid: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand a dictionary where lists and tuples represent grid axes."""
        grid_keys = []
        grid_values = []
        fixed_values = {}
        for key, value in param_grid.items():
            if isinstance(value, (list, tuple)):
                grid_keys.append(key)
                grid_values.append(list(value))
            else:
                fixed_values[key] = value

        if not grid_keys:
            return [fixed_values]

        expanded = []
        for value_combo in product(*grid_values):
            config = deepcopy(fixed_values)
            for key, value in zip(grid_keys, value_combo, strict=True):
                config[key] = value
            expanded.append(config)
        return expanded

    @staticmethod
    def _normalize_estimator_spec(spec: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize an estimator specification."""
        if "name" not in spec:
            raise KeyError("Each estimator spec must include a 'name'.")
        if "factory" not in spec:
            raise KeyError("Each estimator spec must include a 'factory'.")
        if "is_oracle" not in spec:
            raise KeyError("Each estimator spec must include an 'is_oracle' flag.")
        factory = spec["factory"]
        if not callable(factory):
            raise TypeError("Estimator factory must be callable.")

        return {
            "name": str(spec["name"]),
            "factory": factory,
            "factory_name": str(spec.get("factory_name", getattr(factory, "__name__", "factory"))),
            "is_oracle": bool(spec["is_oracle"]),
            "method_config": deepcopy(spec.get("method_config", {})),
            "accepts_trial_seed": bool(spec.get("accepts_trial_seed", False)),
        }

    def _instantiate_estimator(self, estimator_spec: dict[str, Any], trial_seed: int) -> Any:
        """Build an estimator instance, optionally passing a trial-specific seed."""
        factory = estimator_spec["factory"]
        if estimator_spec["accepts_trial_seed"]:
            return factory(trial_seed=trial_seed)
        return factory()

    def _serializable_estimator_specs(self) -> list[dict[str, Any]]:
        """Return the serializable estimator spec metadata used in result headers."""
        serializable_specs = []
        for spec in self.estimators:
            item = {
                "name": spec["name"],
                "is_oracle": spec["is_oracle"],
                "factory_name": spec["factory_name"],
                "method_config": deepcopy(spec["method_config"]),
            }
            if spec["accepts_trial_seed"]:
                item["accepts_trial_seed"] = True
            serializable_specs.append(item)
        return serializable_specs

    def _validate_loaded_results(self, results: dict[str, Any]) -> None:
        """Validate that an on-disk result file matches the requested experiment."""
        expected = {
            "exp_name": self.exp_name,
            "exp_id": self.exp_id,
            "dgp_generator_name": getattr(self.dgp_generator, "__name__", "dgp_generator"),
            "dgp_param_grid": deepcopy(self.dgp_param_grid),
            "seed_offset": self.seed_offset,
        }
        for key, expected_value in expected.items():
            observed_value = results.get(key)
            if observed_value != expected_value:
                raise ValueError(
                    f"Existing result file '{self.result_path}' does not match the requested "
                    f"experiment definition for key '{key}'."
                )
        if self._normalize_loaded_estimator_specs(results.get("estimator_specs", [])) != self._serializable_estimator_specs():
            raise ValueError(
                f"Existing result file '{self.result_path}' does not match the requested "
                "experiment definition for key 'estimator_specs'."
            )

    @staticmethod
    def _normalize_loaded_estimator_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize estimator specs loaded from disk for compatibility checks."""
        normalized = []
        for spec in specs:
            item = {
                "name": spec["name"],
                "is_oracle": spec["is_oracle"],
                "factory_name": spec["factory_name"],
                "method_config": deepcopy(spec["method_config"]),
            }
            if spec.get("accepts_trial_seed"):
                item["accepts_trial_seed"] = True
            normalized.append(item)
        return normalized


def _as_column(values: np.ndarray, label: str) -> np.ndarray:
    """Validate an array as an n by 1 column."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    raise ValueError(f"{label} must have shape (n,) or (n, 1).")
