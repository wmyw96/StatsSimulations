"""Unit tests for partial linear model estimators."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from simlab.core.records import SampledData
from simlab.dgp.partial_linear import PartialLinearModelUniformNoiseDGP
from simlab.estimators.plm_est import (
    PLMDMLEstimator,
    PLMDMLOracleTrackingEstimator,
    PLMMinimaxDebiasEstimator,
    PLMMinimaxDebiasTrackingEstimator,
    PLMOracleAIPWEstimator,
    PLMValidationSelectedDMLEstimator,
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

    def test_validation_selected_dml_records_selection_paths(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        train_sample = dgp.sample(n=24, seed=101, oracle=False)
        validation_sample = dgp.sample(n=9, seed=202, oracle=False)
        estimator = PLMValidationSelectedDMLEstimator(
            name="dml-valid-select",
            hyper_parameters={
                "L": 2,
                "N": 8,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 10,
                "lr": 1e-3,
                "batch_size": 6,
                "seed": 17,
                "validation_check_interval": 4,
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(train_sample, validation_sample)
        predictions = estimator.predict(validation_sample.observed["x"])

        self.assertTrue(np.isfinite(result.estimate))
        self.assertEqual(predictions["mu"].shape, (9, 1))
        self.assertEqual(predictions["pi"].shape, (9, 1))
        self.assertTrue(result.diagnostics["used_validation_selection"])
        self.assertEqual(result.diagnostics["validation_n"], 9)
        self.assertEqual(result.diagnostics["validation_check_interval"], 4)
        self.assertEqual(result.diagnostics["validation_epoch_grid"], [4, 8, 10])
        self.assertEqual(len(result.diagnostics["validation_mu_loss_path"]), 3)
        self.assertEqual(len(result.diagnostics["validation_pi_loss_path"]), 3)
        self.assertIn(
            result.diagnostics["selected_mu_epoch"],
            result.diagnostics["validation_epoch_grid"],
        )
        self.assertIn(
            result.diagnostics["selected_pi_epoch"],
            result.diagnostics["validation_epoch_grid"],
        )
        self.assertAlmostEqual(
            result.diagnostics["best_validation_mu_loss"],
            min(result.diagnostics["validation_mu_loss_path"]),
        )
        self.assertAlmostEqual(
            result.diagnostics["best_validation_pi_loss"],
            min(result.diagnostics["validation_pi_loss_path"]),
        )

    def test_validation_selected_dml_warns_and_falls_back_without_validation(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        train_sample = dgp.sample(n=20, seed=303, oracle=False)
        estimator = PLMValidationSelectedDMLEstimator(
            name="dml-valid-select-fallback",
            hyper_parameters={
                "L": 1,
                "N": 6,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 3,
                "lr": 1e-3,
                "batch_size": 5,
                "seed": 19,
            },
            d=2,
            device="cpu",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(train_sample, valid_data=None)

        predictions = estimator.predict(train_sample.observed["x"])
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, RuntimeWarning))
        self.assertTrue(np.isfinite(result.estimate))
        self.assertEqual(predictions["mu"].shape, (20, 1))
        self.assertEqual(predictions["pi"].shape, (20, 1))
        self.assertFalse(result.diagnostics["used_validation_selection"])
        self.assertEqual(result.diagnostics["validation_n"], 0)
        self.assertEqual(result.diagnostics["validation_epoch_grid"], [])
        self.assertEqual(result.diagnostics["validation_mu_loss_path"], [])
        self.assertEqual(result.diagnostics["validation_pi_loss_path"], [])
        self.assertIsNone(result.diagnostics["selected_mu_epoch"])
        self.assertIsNone(result.diagnostics["selected_pi_epoch"])

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

        self.assertAlmostEqual(
            result.diagnostics["lambda_debias"],
            1.0 / (np.sqrt(5) * np.log2(5)),
            places=6,
        )

    def test_minimax_tracking_estimator_records_mu_and_beta_paths(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        train_sample = dgp.sample(n=12, seed=515, oracle=True)
        validation_sample = dgp.sample(n=8, seed=616, oracle=True)
        augmented_oracle = dict(train_sample.oracle)
        augmented_oracle.update(
            {
                "validation_x": validation_sample.observed["x"],
                "validation_mu_x": validation_sample.oracle["mu_x"],
            }
        )
        tracking_sample = SampledData(
            observed=train_sample.observed,
            oracle=augmented_oracle,
        )
        estimator = PLMMinimaxDebiasTrackingEstimator(
            name="minimax-tracking",
            hyper_parameters={
                "L": 1,
                "N": 4,
                "lambda_mu": 1e-4,
                "lambda_pi": 1e-4,
                "niter": 4,
                "lr": 1e-3,
                "batch_size": 4,
                "seed": 23,
                "lambda_debias": 0.1,
                "weight_bound": 2.0,
                "niter_debias": 3,
                "niter_adversary": 1,
                "debias_lr": 5e-3,
                "tracking_interval": 2,
                "tracking_source": "validation",
            },
            d=2,
            device="cpu",
        )

        result = estimator.fit(tracking_sample)

        self.assertEqual(result.diagnostics["epoch_grid"], [0, 2, 4])
        self.assertEqual(result.diagnostics["tracking_split"], "validation")
        self.assertEqual(result.diagnostics["tracking_n"], 8)
        self.assertEqual(result.diagnostics["tracking_interval"], 2)
        self.assertEqual(len(result.diagnostics["mu_mse_path"]), 3)
        self.assertEqual(len(result.diagnostics["beta_path"]), 3)
        self.assertEqual(len(result.diagnostics["debias_weights"]), 6)
        self.assertTrue(np.all(np.isfinite(result.diagnostics["mu_mse_path"])))
        self.assertTrue(np.all(np.isfinite(result.diagnostics["beta_path"])))
        self.assertTrue(np.all(np.isfinite(result.diagnostics["debias_weights"])))
        self.assertAlmostEqual(result.diagnostics["beta_path"][-1], result.estimate, places=6)

    def test_minimax_tracking_mu_path_matches_dml_tracking_mu_path(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=zero_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.1,
        )
        train_sample = dgp.sample(n=12, seed=717, oracle=True)
        validation_sample = dgp.sample(n=8, seed=818, oracle=True)
        augmented_oracle = dict(train_sample.oracle)
        augmented_oracle.update(
            {
                "validation_x": validation_sample.observed["x"],
                "validation_mu_x": validation_sample.oracle["mu_x"],
                "validation_pi_x": validation_sample.oracle["pi_x"],
            }
        )
        tracking_sample = SampledData(
            observed=train_sample.observed,
            oracle=augmented_oracle,
        )
        common_hyper_parameters = {
            "L": 1,
            "N": 4,
            "lambda_mu": 1e-4,
            "lambda_pi": 1e-4,
            "niter": 3,
            "lr": 1e-3,
            "batch_size": 4,
            "seed": 29,
        }
        dml_estimator = PLMDMLOracleTrackingEstimator(
            name="dml-tracking",
            hyper_parameters=dict(common_hyper_parameters),
            d=2,
            device="cpu",
        )
        minimax_estimator = PLMMinimaxDebiasTrackingEstimator(
            name="minimax-tracking-match",
            hyper_parameters={
                **common_hyper_parameters,
                "lambda_debias": 0.1,
                "weight_bound": 2.0,
                "niter_debias": 2,
                "niter_adversary": 1,
                "debias_lr": 5e-3,
                "tracking_interval": 1,
                "tracking_source": "validation",
            },
            d=2,
            device="cpu",
        )

        dml_result = dml_estimator.fit(tracking_sample)
        minimax_result = minimax_estimator.fit(tracking_sample)

        self.assertEqual(minimax_result.diagnostics["epoch_grid"], [0, 1, 2, 3])
        self.assertTrue(
            np.allclose(
                dml_result.diagnostics["tracking_paths"]["validation"]["mu_mse_path"],
                minimax_result.diagnostics["mu_mse_path"],
            )
        )


if __name__ == "__main__":
    unittest.main()
