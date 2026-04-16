"""Unit tests for the partial linear model DGP."""

from __future__ import annotations

import unittest

import numpy as np

from simlab.dgp.partial_linear import PartialLinearModelUniformNoiseDGP


def linear_mu(x: np.ndarray) -> np.ndarray:
    return x[:, [0]] + 0.5 * x[:, [1]]


def linear_pi(x: np.ndarray) -> np.ndarray:
    return 0.25 * x[:, [0]] - 0.75 * x[:, [1]]


class PartialLinearModelUniformNoiseDGPTests(unittest.TestCase):
    def test_sample_shapes_without_oracle(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=1.5,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.2,
            sigma_eps=0.3,
        )

        sample = dgp.sample(n=25, seed=123, oracle=False)

        self.assertEqual(sample.observed["x"].shape, (25, 2))
        self.assertEqual(sample.observed["t"].shape, (25, 1))
        self.assertEqual(sample.observed["y"].shape, (25, 1))
        self.assertEqual(sample.oracle, {})

    def test_sample_includes_oracle_outputs(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=2.0,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.0,
            sigma_eps=0.0,
        )

        sample = dgp.sample(n=10, seed=7, oracle=True)

        self.assertEqual(sample.oracle["pi_x"].shape, (10, 1))
        self.assertEqual(sample.oracle["mu_x"].shape, (10, 1))
        np.testing.assert_allclose(sample.observed["t"], sample.oracle["pi_x"])
        np.testing.assert_allclose(
            sample.observed["y"],
            dgp.beta * sample.observed["t"] + sample.oracle["mu_x"],
        )

    def test_sampling_is_reproducible_for_fixed_seed(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=0.5,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.4,
            sigma_eps=0.1,
        )

        sample_one = dgp.sample(n=12, seed=999, oracle=True)
        sample_two = dgp.sample(n=12, seed=999, oracle=True)

        np.testing.assert_allclose(
            sample_one.observed["x"],
            sample_two.observed["x"],
        )
        np.testing.assert_allclose(
            sample_one.observed["t"],
            sample_two.observed["t"],
        )
        np.testing.assert_allclose(
            sample_one.observed["y"],
            sample_two.observed["y"],
        )
        np.testing.assert_allclose(
            sample_one.oracle["pi_x"],
            sample_two.oracle["pi_x"],
        )
        np.testing.assert_allclose(
            sample_one.oracle["mu_x"],
            sample_two.oracle["mu_x"],
        )

    def test_true_parameter_returns_beta(self) -> None:
        dgp = PartialLinearModelUniformNoiseDGP(
            beta=3.25,
            func_mu=linear_mu,
            func_pi=linear_pi,
            d=2,
            sigma_u=0.4,
            sigma_eps=0.1,
        )

        self.assertEqual(dgp.true_parameter(), 3.25)


if __name__ == "__main__":
    unittest.main()
