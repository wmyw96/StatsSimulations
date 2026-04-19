"""Unit tests for the PLM evaluator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.plm.experiment_defs import build_experiment_1_1


class PLMEvaluatorTests(unittest.TestCase):
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
            evaluator.estimators[0]["method_config"]["N"] = 8
            evaluator.estimators[0]["method_config"]["niter"] = 2
            evaluator.estimators[0]["method_config"]["batch_size"] = 8
            evaluator.estimators[0]["method_config"]["d"] = 1
            evaluator.estimators[0]["factory"] = lambda cfg=dict(evaluator.estimators[0]["method_config"]): __import__(
                "examples.plm.experiment_defs",
                fromlist=["make_plm_dml_estimator"],
            ).make_plm_dml_estimator(cfg)

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


if __name__ == "__main__":
    unittest.main()
