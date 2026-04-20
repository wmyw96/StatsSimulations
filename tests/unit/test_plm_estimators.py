"""Unit tests for partial linear model estimators."""

from __future__ import annotations

import unittest

import numpy as np

from simlab.core.records import SampledData
from simlab.dgp.partial_linear import PartialLinearModelUniformNoiseDGP
from simlab.estimators.plm_est import (
    PLMDMLEstimator,
    PLMDMLOracleTrackingEstimator,
    PLMMinimaxDebiasEstimator,
    PLMOracleAIPWEstimator,
)


def zero_mu(x: np.ndarray) -> np.ndarray:
    return np.zeros((len(x), 1), dtype=np.float32)


def linear_pi(x: np.ndarray) -> np.ndarray:
    return (0.5 * x[:, [0]] - 0.25 * x[:, [1]]).astype(np.float32)


class PLMEstimatorTests(unittest.TestCase):
    def test_oracle_estimator_matches_manual_aipw_formula(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=1.75,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        sample = dgp.sample(n=10, seed=314, oracle=False)
        estimator = PLMOracleAIPWEstimator(
            name="oracle",
            ground_truth_func_mu=zero_mu,
            ground_truth_func_pi=linear_pi,
        )

        result = estimator.fit(sample)
        x_d1 = sample.observed["x"][:5]
        t_d1 = sample.observed["t"][:5]
        y_d1 = sample.observed["y"][:5]
        mu_d1 = zero_mu(x_d1)
        pi_d1 = linear_pi(x_d1)
        manual = np.mean((y_d1 - mu_d1) * (t_d1 - pi_d1)) / np.mean(t_d1 * (t_d1 - pi_d1))

        self.assertAlmostEqual(result.estimate, float(manual), places=6)

    def test_dml_estimator_fit_and_predict_run(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=2.0,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        sample = dgp.sample(n=12, seed=123, oracle=False)
        estimator = PLMDMLEstimator(
            name="dml",
            hyper_parameters={
                "L": 2,
                "N": 8,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 3,
                "lr": 1e-3,
                "batch_size": 4,
                "seed": 9,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(sample)
        predictions = estimator.predict(sample.observed["x"])

        self.assertIsNotNone(estimator.est_params)
        self.assertTrue(np.isfinite(result.estimate))
        self.assertEqual(predictions["mu"].shape, (12, 1))
        self.assertEqual(predictions["pi"].shape, (12, 1))
        self.assertEqual(result.diagnostics["n_d1"], 6)
        self.assertEqual(result.diagnostics["n_d2"], 6)

    def test_dml_split_puts_extra_observation_in_d2(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=1.0,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        sample = dgp.sample(n=9, seed=999, oracle=False)
        estimator = PLMDMLEstimator(
            name="dml-odd-split",
            hyper_parameters={
                "L": 1,
                "N": 4,
                "lambda_mu": 0.0,
                "lambda_pi": 0.0,
                "niter": 2,
                "lr": 1e-3,
                "batch_size": 16,
                "seed": 7,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(sample)

        self.assertEqual(result.diagnostics["n_d1"], 4)
        self.assertEqual(result.diagnostics["n_d2"], 5)

    def test_profiled_joint_lse_beta_tracks_nonzero_signal(self) -> None:
        def sine_first_coordinate(x: np.ndarray) -> np.ndarray:
            return np.sin(2.0 * np.pi * x[:, [0]]).astype(np.float32)

        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=sine_first_coordinate,
            func_pi=sine_first_coordinate,
            d=1,
            sigma_u=0.5,
            sigma_eps=0.5,
        )
        sample = dgp.sample(n=1024, seed=21, oracle=False)
        estimator = PLMDMLEstimator(
            name="dml-profiled-beta",
            hyper_parameters={
                "L": 3,
                "N": 128,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 200,
                "lr": 1e-3,
                "batch_size": 512,
                "seed": 0,
            },
            d=1,
            device="cpu",
        )

        result = estimator.fit(sample)

        self.assertGreater(result.diagnostics["beta_joint"], 0.25)
        self.assertLess(result.diagnostics["beta_joint"], 0.75)

    def test_tracking_estimator_records_oracle_nuisance_paths(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.25,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        sample = dgp.sample(n=12, seed=1234, oracle=True)
        estimator = PLMDMLOracleTrackingEstimator(
            name="dml-track",
            hyper_parameters={
                "L": 2,
                "N": 8,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 3,
                "lr": 1e-3,
                "batch_size": 4,
                "seed": 5,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(sample)

        self.assertEqual(result.diagnostics["epoch_grid"], [0, 1, 2, 3])
        self.assertEqual(len(result.diagnostics["mu_mse_path"]), 4)
        self.assertEqual(len(result.diagnostics["pi_mse_path"]), 4)
        self.assertEqual(result.diagnostics["tracking_split"], "D2")
        self.assertEqual(result.diagnostics["tracking_n"], 6)

    def test_tracking_estimator_uses_validation_sample_when_available(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.25,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        train_sample = dgp.sample(n=12, seed=1234, oracle=True)
        validation_sample = dgp.sample(n=8, seed=4321, oracle=True)
        fit_sample = SampledData(
            observed=train_sample.observed,
            oracle={
                **train_sample.oracle,
                "validation_x": validation_sample.observed["x"],
                "validation_mu_x": validation_sample.oracle["mu_x"],
                "validation_pi_x": validation_sample.oracle["pi_x"],
            },
        )
        estimator = PLMDMLOracleTrackingEstimator(
            name="dml-track-validation",
            hyper_parameters={
                "L": 2,
                "N": 8,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 3,
                "lr": 1e-3,
                "batch_size": 4,
                "seed": 5,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(fit_sample)

        self.assertIn("tracking_paths", result.diagnostics)
        self.assertIn("D2", result.diagnostics["tracking_paths"])
        self.assertIn("validation", result.diagnostics["tracking_paths"])
        self.assertEqual(result.diagnostics["tracking_paths"]["D2"]["tracking_n"], 6)
        self.assertEqual(result.diagnostics["tracking_paths"]["validation"]["tracking_n"], 8)

    def test_minimax_debias_estimator_fit_and_predict_run(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        sample = dgp.sample(n=12, seed=2024, oracle=False)
        estimator = PLMMinimaxDebiasEstimator(
            name="minimax-debias",
            hyper_parameters={
                "L": 2,
                "N": 8,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 3,
                "lr": 1e-3,
                "batch_size": 4,
                "seed": 11,
                "lambda_debias": 0.1,
                "weight_bound": 2.0,
                "niter_debias": 4,
                "niter_adversary": 2,
                "debias_lr": 5e-3,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(sample)
        predictions = estimator.predict(sample.observed["x"])

        self.assertTrue(np.isfinite(result.estimate))
        self.assertEqual(predictions["mu"].shape, (12, 1))
        self.assertEqual(predictions["pi"].shape, (12, 1))
        self.assertEqual(result.diagnostics["n_d1"], 6)
        self.assertEqual(result.diagnostics["n_d2"], 6)
        self.assertLessEqual(result.diagnostics["max_abs_weight"], 2.0 + 1e-5)
        self.assertIn("final_debias_objective", result.diagnostics)
        self.assertIn("final_adversary_value", result.diagnostics)
        self.assertIn("final_stability_term", result.diagnostics)

    def test_minimax_debias_estimator_uses_default_lambda_debias(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        sample = dgp.sample(n=10, seed=77, oracle=False)
        estimator = PLMMinimaxDebiasEstimator(
            name="minimax-debias-default",
            hyper_parameters={
                "L": 1,
                "N": 4,
                "lambda_mu": 0.0,
                "lambda_pi": 0.0,
                "niter": 2,
                "lr": 1e-3,
                "batch_size": 8,
                "seed": 3,
                "weight_bound": 3.0,
                "niter_debias": 2,
                "niter_adversary": 1,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(sample)

        self.assertAlmostEqual(result.diagnostics["lambda_debias"], 1.0 / np.sqrt(5), places=6)


if __name__ == "__main__":
    unittest.main()
