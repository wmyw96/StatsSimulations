"""Unit tests for the PLM evaluator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

from examples.plm.experiment_defs import build_evaluator_from_exp_id, build_experiment_1_1, normalize_exp_id


class PLMEvaluatorTests(unittest.TestCase):
    def test_normalize_exp_id_accepts_dotted_and_storage_forms(self) -> None:
        self.assertEqual(normalize_exp_id("1.1.2"), ("1.1_2", "1.1.2"))
        self.assertEqual(normalize_exp_id("1.1_2"), ("1.1_2", "1.1.2"))
        self.assertEqual(normalize_exp_id("1.2.1"), ("1.2_1", "1.2.1"))
        self.assertEqual(normalize_exp_id("1.3.1"), ("1.3_1", "1.3.1"))
        self.assertEqual(normalize_exp_id("1.5.1"), ("1.5_1", "1.5.1"))

        evaluator = build_evaluator_from_exp_id(
            exp_id="1.1.2",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator.exp_id, "1.1_2")
        self.assertEqual(evaluator.result_path.name, "1.1_2.json")

        evaluator_12 = build_evaluator_from_exp_id(
            exp_id="1.2.1",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_12.exp_id, "1.2_1")
        self.assertEqual(evaluator_12.result_path.name, "1.2_1.json")
        self.assertEqual(evaluator_12.dgp_param_grid["beta"], 0.5)

        evaluator_13 = build_evaluator_from_exp_id(
            exp_id="1.3.1",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_13.exp_id, "1.3_1")
        self.assertEqual(evaluator_13.result_path.name, "1.3_1.json")
        self.assertEqual(evaluator_13.dgp_param_grid["beta_sampler_name"], "uniform")
        self.assertEqual(evaluator_13.dgp_param_grid["beta_low"], -0.5)
        self.assertEqual(evaluator_13.dgp_param_grid["beta_high"], 0.5)
        self.assertFalse(evaluator_13.estimators[0]["accepts_trial_seed"])

        evaluator_13_2 = build_evaluator_from_exp_id(
            exp_id="1.3.2",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertTrue(evaluator_13_2.estimators[0]["accepts_trial_seed"])

        evaluator_14 = build_evaluator_from_exp_id(
            exp_id="1.4.1",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_14.exp_id, "1.4_1")
        self.assertEqual(evaluator_14.result_path.name, "1.4_1.json")
        self.assertEqual(evaluator_14.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_14.estimators[0]["name"], "dml_nn_tracking_lambda_1e-4")
        self.assertTrue(evaluator_14.estimators[0]["accepts_trial_seed"])

        evaluator_14_2 = build_evaluator_from_exp_id(
            exp_id="1.4.2",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_14_2.exp_id, "1.4_2")
        self.assertEqual(evaluator_14_2.result_path.name, "1.4_2.json")
        self.assertEqual(len(evaluator_14_2.estimators), 6)
        lambda_values = [spec["method_config"]["lambda_mu"] for spec in evaluator_14_2.estimators]
        self.assertEqual(lambda_values, [2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4])

        evaluator_14_3 = build_evaluator_from_exp_id(
            exp_id="1.4.3",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_14_3.exp_id, "1.4_3")
        self.assertEqual(evaluator_14_3.result_path.name, "1.4_3.json")
        self.assertEqual(len(evaluator_14_3.estimators), 7)
        lambda_values = [spec["method_config"]["lambda_mu"] for spec in evaluator_14_3.estimators]
        expected_lambda_values = [8e-07, 4e-06, 2e-05, 1e-04, 5e-04, 2.5e-03, 1.25e-02]
        self.assertEqual(len(lambda_values), len(expected_lambda_values))
        for observed, expected in zip(lambda_values, expected_lambda_values):
            self.assertAlmostEqual(observed, expected)

        evaluator_14_4 = build_evaluator_from_exp_id(
            exp_id="1.4.4",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_14_4.exp_id, "1.4_4")
        self.assertEqual(evaluator_14_4.result_path.name, "1.4_4.json")
        self.assertEqual(len(evaluator_14_4.estimators), 7)
        self.assertEqual(evaluator_14_4.estimators[0]["method_config"]["tracking_source"], "validation")
        self.assertEqual(evaluator_14_4.estimators[0]["method_config"]["validation_n"], 1024)
        lambda_values = [spec["method_config"]["lambda_mu"] for spec in evaluator_14_4.estimators]
        expected_lambda_values = [8e-07, 4e-06, 2e-05, 1e-04, 5e-04, 2.5e-03, 1.25e-02]
        self.assertEqual(len(lambda_values), len(expected_lambda_values))
        for observed, expected in zip(lambda_values, expected_lambda_values):
            self.assertAlmostEqual(observed, expected)

        evaluator_15 = build_evaluator_from_exp_id(
            exp_id="1.5.1",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15.exp_id, "1.5_1")
        self.assertEqual(evaluator_15.result_path.name, "1.5_1.json")
        self.assertEqual(
            evaluator_15.dgp_param_grid["func_pi_name"],
            [
                "sin_2pi_first_coordinate",
                "sin_4pi_first_coordinate",
                "sin_8pi_first_coordinate",
            ],
        )
        self.assertEqual(evaluator_15.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15.estimators[1]["accepts_dgp_config"])

        evaluator_15_2 = build_evaluator_from_exp_id(
            exp_id="1.5.2",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_2.exp_id, "1.5_2")
        self.assertEqual(evaluator_15_2.result_path.name, "1.5_2.json")
        self.assertEqual(
            evaluator_15_2.dgp_param_grid["func_pi_name"],
            [
                "sign_sin_2pi_times_sin_2pi_first_coordinate",
                "sign_sin_2pi_times_sin_4pi_first_coordinate",
                "sign_sin_2pi_times_sin_8pi_first_coordinate",
            ],
        )
        self.assertEqual(evaluator_15_2.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_2.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_2.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_2.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_2.estimators[1]["accepts_dgp_config"])

        evaluator_15_3 = build_evaluator_from_exp_id(
            exp_id="1.5.3",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_3.exp_id, "1.5_3")
        self.assertEqual(evaluator_15_3.result_path.name, "1.5_3.json")
        self.assertEqual(
            evaluator_15_3.dgp_param_grid["func_pi_name"],
            [
                "sign_sin_2pi_times_abs_sin_2pi_first_coordinate",
                "sign_sin_2pi_times_abs_sin_4pi_first_coordinate",
                "sign_sin_2pi_times_abs_sin_8pi_first_coordinate",
            ],
        )
        self.assertEqual(evaluator_15_3.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_3.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_3.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_3.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_3.estimators[1]["accepts_dgp_config"])

        evaluator_15_4 = build_evaluator_from_exp_id(
            exp_id="1.5.4",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_4.exp_id, "1.5_4")
        self.assertEqual(evaluator_15_4.result_path.name, "1.5_4.json")
        self.assertEqual(
            evaluator_15_4.dgp_param_grid["func_pi_name"],
            [
                "progressive_pi_1_first_coordinate",
                "progressive_pi_2_first_coordinate",
                "progressive_pi_3_first_coordinate",
                "progressive_pi_4_first_coordinate",
            ],
        )
        self.assertEqual(evaluator_15_4.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_4.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_4.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_4.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_4.estimators[1]["accepts_dgp_config"])

        evaluator_15_5 = build_evaluator_from_exp_id(
            exp_id="1.5.5",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_5.exp_id, "1.5_5")
        self.assertEqual(evaluator_15_5.result_path.name, "1.5_5.json")
        self.assertEqual(
            evaluator_15_5.dgp_param_grid["func_pi_name"],
            [
                "fixed_overlap_pi_1_first_coordinate",
                "fixed_overlap_pi_2_first_coordinate",
                "fixed_overlap_pi_3_first_coordinate",
                "fixed_overlap_pi_4_first_coordinate",
            ],
        )
        self.assertEqual(evaluator_15_5.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_5.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_5.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_5.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_5.estimators[1]["accepts_dgp_config"])

        evaluator_15_6 = build_evaluator_from_exp_id(
            exp_id="1.5.6",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_6.exp_id, "1.5_6")
        self.assertEqual(evaluator_15_6.result_path.name, "1.5_6.json")
        self.assertEqual(evaluator_15_6.dgp_param_grid["d"], 4)
        self.assertEqual(evaluator_15_6.dgp_param_grid["func_mu_name"], "sin_2pi_four_coordinates")
        self.assertEqual(
            evaluator_15_6.dgp_param_grid["func_pi_name"],
            [
                "fixed_overlap_d4_pi_1",
                "fixed_overlap_d4_pi_2",
                "fixed_overlap_d4_pi_3",
            ],
        )
        self.assertEqual(evaluator_15_6.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_6.estimators[0]["method_config"]["d"], 4)
        self.assertEqual(evaluator_15_6.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_6.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_6.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_6.estimators[1]["accepts_dgp_config"])

        evaluator_15_7 = build_evaluator_from_exp_id(
            exp_id="1.5.7",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_7.exp_id, "1.5_7")
        self.assertEqual(evaluator_15_7.result_path.name, "1.5_7.json")
        self.assertEqual(evaluator_15_7.dgp_param_grid["d"], 4)
        self.assertEqual(evaluator_15_7.dgp_param_grid["func_mu_name"], "sin_2pi_first_coordinate")
        self.assertEqual(
            evaluator_15_7.dgp_param_grid["func_pi_name"],
            [
                "isolated_d4_pi_1",
                "isolated_d4_pi_2",
                "isolated_d4_pi_3",
                "isolated_d4_pi_4",
            ],
        )
        self.assertEqual(evaluator_15_7.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_7.estimators[0]["method_config"]["d"], 4)
        self.assertEqual(evaluator_15_7.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_7.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_7.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_7.estimators[1]["accepts_dgp_config"])

        evaluator_15_8 = build_evaluator_from_exp_id(
            exp_id="1.5.8",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_8.exp_id, "1.5_8")
        self.assertEqual(evaluator_15_8.result_path.name, "1.5_8.json")
        self.assertEqual(evaluator_15_8.dgp_param_grid["d"], 4)
        self.assertEqual(
            evaluator_15_8.dgp_param_grid["func_mu_name"],
            "easy_mu_sin_pi_x1_plus_cos_pi_x2",
        )
        self.assertEqual(
            evaluator_15_8.dgp_param_grid["func_pi_name"],
            [
                "increasing_beta_pi_1",
                "increasing_beta_pi_2",
                "increasing_beta_pi_3",
                "increasing_beta_pi_4",
            ],
        )
        self.assertEqual(evaluator_15_8.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_8.estimators[0]["method_config"]["d"], 4)
        self.assertEqual(evaluator_15_8.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_8.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_8.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_8.estimators[1]["accepts_dgp_config"])

        evaluator_15_9 = build_evaluator_from_exp_id(
            exp_id="1.5.9",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_9.exp_id, "1.5_9")
        self.assertEqual(evaluator_15_9.result_path.name, "1.5_9.json")
        self.assertEqual(evaluator_15_9.dgp_param_grid["d"], 4)
        self.assertEqual(evaluator_15_9.dgp_param_grid["func_mu_name"], "correlated_mu_eps005")
        self.assertEqual(
            evaluator_15_9.dgp_param_grid["func_pi_name"],
            [
                "correlated_pi_1",
                "correlated_pi_2",
                "correlated_pi_3",
            ],
        )
        self.assertEqual(evaluator_15_9.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_9.estimators[0]["method_config"]["d"], 4)
        self.assertEqual(evaluator_15_9.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_9.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_9.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_9.estimators[1]["accepts_dgp_config"])

        evaluator_15_10 = build_evaluator_from_exp_id(
            exp_id="1.5.10",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_10.exp_id, "1.5_10")
        self.assertEqual(evaluator_15_10.result_path.name, "1.5_10.json")
        self.assertEqual(evaluator_15_10.dgp_param_grid["d"], 4)
        self.assertEqual(evaluator_15_10.dgp_param_grid["func_mu_name"], "correlated_mu_eps005")
        self.assertEqual(
            evaluator_15_10.dgp_param_grid["func_pi_name"],
            [
                "correlated_wide_pi_1",
                "correlated_wide_pi_2",
                "correlated_wide_pi_3",
            ],
        )
        self.assertEqual(evaluator_15_10.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_10.estimators[0]["method_config"]["d"], 4)
        self.assertEqual(evaluator_15_10.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_10.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_10.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_10.estimators[1]["accepts_dgp_config"])

        evaluator_15_11 = build_evaluator_from_exp_id(
            exp_id="1.5.11",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        self.assertEqual(evaluator_15_11.exp_id, "1.5_11")
        self.assertEqual(evaluator_15_11.result_path.name, "1.5_11.json")
        self.assertEqual(evaluator_15_11.dgp_param_grid["d"], 1)
        self.assertEqual(
            evaluator_15_11.dgp_param_grid["func_pi_name"],
            [
                "unit_variance_correlated_pi_1",
                "unit_variance_correlated_pi_2",
                "unit_variance_correlated_pi_3",
            ],
        )
        self.assertAlmostEqual(evaluator_15_11.dgp_param_grid["sigma_u"], 3.0**0.5)
        self.assertAlmostEqual(evaluator_15_11.dgp_param_grid["sigma_eps"], 3.0**0.5)
        self.assertEqual(evaluator_15_11.dgp_param_grid["n"], [1024])
        self.assertEqual(evaluator_15_11.estimators[0]["method_config"]["d"], 1)
        self.assertEqual(evaluator_15_11.estimators[0]["method_config"]["lambda_mu"], 2e-5)
        self.assertEqual(evaluator_15_11.estimators[0]["method_config"]["lambda_pi"], 2e-5)
        self.assertTrue(evaluator_15_11.estimators[0]["accepts_trial_seed"])
        self.assertTrue(evaluator_15_11.estimators[1]["accepts_dgp_config"])

    def test_run_and_resume_without_duplicate_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_experiment_1_1(
                exp_id="1.1_test",
                n_trials=2,
                seed_offset=5,
                device="cpu",
                result_root=result_root,
            )
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            dml_config = dict(evaluator.estimators[0]["method_config"])
            dml_config["N"] = 8
            dml_config["niter"] = 2
            dml_config["batch_size"] = 8
            dml_config["d"] = 1
            evaluator.estimators[0]["method_config"] = dml_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(dml_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator({**cfg, "seed": trial_seed})
            evaluator.estimators[0]["accepts_trial_seed"] = True

            first_results = evaluator.run()
            self.assertEqual(len(first_results["trial_results"]), 2)

            second_results = evaluator.run()
            self.assertEqual(len(second_results["trial_results"]), 2)

            summary = evaluator.query_results(
                {
                    "d": 1,
                    "func_mu_name": "sin_2pi_first_coordinate",
                    "func_pi_name": "sin_2pi_first_coordinate",
                    "beta": 0.0,
                    "sigma_u": 0.5,
                    "sigma_eps": 0.5,
                    "n_test": 32,
                    "n": 16,
                },
                mode="summary",
            )

            self.assertIn("dml_nn", summary)
            self.assertIn("oracle_aipw", summary)
            self.assertEqual(summary["dml_nn"]["num_trials"], 2)
            self.assertEqual(summary["oracle_aipw"]["num_trials"], 2)

            first_trial = first_results["trial_results"][0]["estimator_results"][0]
            self.assertIn("mu_pi_product_mean", first_trial)
            self.assertIn("mu_pi_product_true_mean", first_trial)
            self.assertIn("mu_pi_product_mse", first_trial)

    def test_resume_updates_n_trials_metadata_and_reuses_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_experiment_1_1(
                exp_id="1.1_meta",
                n_trials=1,
                seed_offset=7,
                device="cpu",
                result_root=result_root,
            )
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            dml_config = dict(evaluator.estimators[0]["method_config"])
            dml_config["N"] = 8
            dml_config["niter"] = 2
            dml_config["batch_size"] = 8
            dml_config["d"] = 1
            evaluator.estimators[0]["method_config"] = dml_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(dml_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator({**cfg, "seed": trial_seed})
            evaluator.estimators[0]["accepts_trial_seed"] = True
            evaluator.run()

            resumed = build_experiment_1_1(
                exp_id="1.1_meta",
                n_trials=3,
                seed_offset=7,
                device="cpu",
                result_root=result_root,
            )
            resumed.dgp_param_grid["n"] = [16]
            resumed.dgp_param_grid["n_test"] = 32
            resumed_config = dict(resumed.estimators[0]["method_config"])
            resumed_config["N"] = 8
            resumed_config["niter"] = 2
            resumed_config["batch_size"] = 8
            resumed_config["d"] = 1
            resumed.estimators[0]["method_config"] = resumed_config
            resumed.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(resumed_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator({**cfg, "seed": trial_seed})
            resumed.estimators[0]["accepts_trial_seed"] = True

            results = resumed.run()
            self.assertEqual(results["n_trials"], 3)
            self.assertEqual(len(results["trial_results"]), 3)

    def test_pi_complexity_experiment_oracle_tracks_current_dgp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_evaluator_from_exp_id(
                exp_id="1.5.1",
                n_trials=1,
                seed_offset=3,
                device="cpu",
                result_root=result_root,
            )
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            dml_config = dict(evaluator.estimators[0]["method_config"])
            dml_config["N"] = 8
            dml_config["niter"] = 2
            dml_config["batch_size"] = 8
            dml_config["d"] = 1
            evaluator.estimators[0]["method_config"] = dml_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(dml_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator({**cfg, "seed": trial_seed})

            results = evaluator.run()
            self.assertEqual(len(results["trial_results"]), 3)
            oracle_errors = [
                record["pi_mse"]
                for trial in results["trial_results"]
                for record in trial["estimator_results"]
                if record["estimator_name"] == "oracle_aipw"
            ]
            self.assertEqual(len(oracle_errors), 3)
            for error in oracle_errors:
                self.assertAlmostEqual(error, 0.0, places=12)

    def test_run_rejects_existing_results_with_mismatched_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_experiment_1_1(
                exp_id="1.1_validate",
                n_trials=1,
                seed_offset=11,
                device="cpu",
                result_root=result_root,
            )
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            dml_config = dict(evaluator.estimators[0]["method_config"])
            dml_config["N"] = 8
            dml_config["niter"] = 2
            dml_config["batch_size"] = 8
            dml_config["d"] = 1
            evaluator.estimators[0]["method_config"] = dml_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(dml_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator({**cfg, "seed": trial_seed})
            evaluator.estimators[0]["accepts_trial_seed"] = True
            evaluator.run()

            mismatched = build_experiment_1_1(
                exp_id="1.1_validate",
                n_trials=2,
                seed_offset=11,
                device="cpu",
                result_root=result_root,
            )
            mismatched.dgp_param_grid["n"] = [16]
            mismatched.dgp_param_grid["n_test"] = 64

            with self.assertRaises(ValueError):
                mismatched.run()

    def test_trial_seeded_dml_factory_uses_trial_specific_seed(self) -> None:
        evaluator = build_evaluator_from_exp_id(
            exp_id="1.3.2",
            n_trials=1,
            seed_offset=0,
            device="cpu",
        )
        estimator_one = evaluator.estimators[0]["factory"](trial_seed=3)
        estimator_two = evaluator.estimators[0]["factory"](trial_seed=4)
        self.assertEqual(estimator_one.seed, 3)
        self.assertEqual(estimator_two.seed, 4)

    def test_random_beta_family_records_realized_beta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_evaluator_from_exp_id(
                exp_id="1.3.1",
                n_trials=1,
                seed_offset=3,
                device="cpu",
                result_root=result_root,
            )
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            dml_config = dict(evaluator.estimators[0]["method_config"])
            dml_config["N"] = 8
            dml_config["niter"] = 2
            dml_config["batch_size"] = 8
            dml_config["d"] = 1
            evaluator.estimators[0]["method_config"] = dml_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(dml_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator({**cfg, "seed": trial_seed})
            evaluator.estimators[0]["accepts_trial_seed"] = True

            results = evaluator.run()
            trial_record = results["trial_results"][0]

            self.assertIn("beta_true", trial_record)
            self.assertGreaterEqual(trial_record["beta_true"], -0.5)
            self.assertLessEqual(trial_record["beta_true"], 0.5)

    def test_tracking_family_serializes_epoch_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_evaluator_from_exp_id(
                exp_id="1.4.1",
                n_trials=1,
                seed_offset=2,
                device="cpu",
                result_root=result_root,
            )
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            tracking_config = dict(evaluator.estimators[0]["method_config"])
            tracking_config["N"] = 8
            tracking_config["niter"] = 2
            tracking_config["batch_size"] = 8
            tracking_config["d"] = 1
            evaluator.estimators[0]["method_config"] = tracking_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(tracking_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_tracking_estimator"],
            ).make_plm_dml_tracking_estimator({**cfg, "seed": trial_seed})
            evaluator.estimators[0]["accepts_trial_seed"] = True

            results = evaluator.run()
            estimator_record = results["trial_results"][0]["estimator_results"][0]

            self.assertEqual(estimator_record["estimator_name"], "dml_nn_tracking_lambda_1e-4")
            self.assertEqual(estimator_record["epoch_grid"], [0, 1, 2])
            self.assertEqual(len(estimator_record["mu_mse_path"]), 3)
            self.assertEqual(len(estimator_record["pi_mse_path"]), 3)
            self.assertEqual(estimator_record["tracking_split"], "D2")

    def test_validation_tracking_family_serializes_validation_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "simulation_results"
            evaluator = build_evaluator_from_exp_id(
                exp_id="1.4.4",
                n_trials=1,
                seed_offset=2,
                device="cpu",
                result_root=result_root,
            )
            evaluator.estimators = [evaluator.estimators[0]]
            evaluator.dgp_param_grid["n"] = [16]
            evaluator.dgp_param_grid["n_test"] = 32
            tracking_config = dict(evaluator.estimators[0]["method_config"])
            tracking_config["N"] = 8
            tracking_config["niter"] = 2
            tracking_config["batch_size"] = 8
            tracking_config["d"] = 1
            tracking_config["validation_n"] = 16
            evaluator.estimators[0]["method_config"] = tracking_config
            evaluator.estimators[0]["factory"] = lambda *, trial_seed=None, cfg=dict(tracking_config): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_tracking_estimator"],
            ).make_plm_dml_tracking_estimator({**cfg, "seed": trial_seed})
            evaluator.estimators[0]["accepts_trial_seed"] = True

            results = evaluator.run()
            estimator_record = results["trial_results"][0]["estimator_results"][0]

            self.assertIn("tracking_paths", estimator_record)
            self.assertEqual(estimator_record["tracking_paths"]["D2"]["tracking_n"], 8)
            self.assertEqual(estimator_record["tracking_paths"]["validation"]["tracking_n"], 16)


if __name__ == "__main__":
    unittest.main()
