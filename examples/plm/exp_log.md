# General setting

## Data generating process

### Class DGP1: `PartialLinearModelUniformNoiseDGP`

This class implements a partial linear model with uniformly distributed covariates and additive uniform noise. For a covariate vector `X in [-1, 1]^d`, treatment variable `T`, and response variable `Y`, the model takes the form

```text
T = pi(X) + u,
Y = beta * T + mu(X) + eps,
```

where `mu` is the outcome regression, `pi` is the treatment regression, `u` is treatment noise, and `eps` is outcome noise. The class draws the covariates independently from a uniform distribution on `[-1, 1]^d`, then evaluates the user-specified functions `mu` and `pi`, and finally adds bounded uniform noise to produce the observed treatment and response. When requested in oracle mode, the class also returns the oracle nuisance quantities `mu(X)` and `pi(X)`, which are used only for evaluation and oracle procedures.

This class is designed to separate the model family from the experiment-specific parameter choice. The concrete selections of the nuisance functions, signal parameter, noise scales, sample sizes, and test design are specified in the individual experiment sections below.

### Class DGP2

Reserved for future partial linear model designs.

## Methods

### Method1: Neural DML (`PLMDMLEstimator`)

This method implements the double machine learning pipeline for the partial linear model. The data are split into two parts. On the second split, the method fits a neural network approximation to the outcome regression `mu(X)` jointly with an initial coefficient estimate for `beta`, and separately fits a neural network approximation to the treatment regression `pi(X)`. On the first split, it plugs these nuisance estimates into the augmented inverse-propensity-weighted estimator given by Eq. (1.2) of the PLM paper to produce the final estimate of `beta`.

The nuisance models are fully connected ReLU networks with batch normalization and residual connections. The implementation uses Adam for stochastic optimization and supports either CPU or accelerator devices supported by PyTorch.

Default hyper-parameters:

- `L = 3`: network depth. This controls the number of nonlinear transformation stages used to represent the nuisance functions.
- `N = 512`: network width. This controls the number of hidden units per layer and therefore the expressive capacity of the nuisance learner.
- `lambda_mu = 1e-4`: L2 regularization level for the outcome network. This shrinks the network weights and stabilizes training by discouraging overly large coefficients.
- `lambda_pi = 1e-4`: L2 regularization level for the treatment network. This plays the same role for the auxiliary regression `pi`.
- `niter`: number of training epochs over the second split of the sample.
- `lr = 1e-3`: Adam learning rate. This controls the step size of the stochastic optimizer.
- `batch_size = 1024`: mini-batch size used in stochastic optimization.
- `device = "cpu"` by default: computation device for the neural network fitting routine.

The regularization values `lambda_mu = lambda_pi = 1e-4` follow the common small-weight-decay convention in neural network regression and are intended as conservative defaults that stabilize optimization without dominating the data-fit term.

### Method2: Oracle AIPW (`PLMOracleAIPWEstimator`)

This method is an oracle benchmark. It does not estimate the nuisance functions from data. Instead, it directly uses the ground-truth `mu(X)` and `pi(X)` in the same augmented inverse-propensity-weighted formula used by the DML estimator. This isolates the contribution of nuisance estimation error and provides a benchmark for the best performance one can expect from the AIPW stage when the nuisance functions are known.

Default hyper-parameters:

- none beyond the specification of the oracle nuisance functions themselves.

## Evaluation

For each estimator, the experiment records the following quantities.

- AIPW estimate of `beta`: the final coefficient estimate returned by the estimator.
- Squared error of the final `beta` estimate: `(beta_hat - beta_true)^2`.
- Squared error of the initial `beta` estimate: for the neural DML estimator, this compares the jointly trained neural-network coefficient parameter to the ground truth before the AIPW correction; for the oracle estimator, this is set equal to the final squared error.
- Mean squared error of `mu`: the average squared difference between the predicted outcome regression and the oracle nuisance value on an independent test sample.
- Mean squared error of `pi`: the average squared difference between the predicted treatment regression and the oracle nuisance value on an independent test sample.

The summary view aggregates these quantities across repeated trials for each experimental configuration.

# Experimental results

## 1.1_1

### Goal

This experiment is intended as a first validation study. Its goal is to verify that the simulation pipeline is working correctly and to assess whether there is a visible performance gap between the neural DML estimator and the oracle benchmark when both are evaluated under the same partial linear model.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Target coefficient: `beta = 0`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample sizes: `n in {256, 512, 1024, 2048}`
- Test sample size: `n_test = 10000`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 1e-4`
- Treatment-network regularization: `lambda_pi = 1e-4`
- Optimizer: Adam
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

This design uses a relatively wide network so that the nonlinear nuisance functions can be approximated flexibly, while the modest L2 penalties provide a conservative stabilization mechanism during optimization. The oracle method uses the same AIPW correction formula but removes nuisance estimation error entirely.

### Results

The experiment was run with `10` independent trials for each sample size in `{256, 512, 1024, 2048}`. The summary metrics below are trial averages.

| n | Method | Beta MSE | Initial Beta MSE | Mu MSE | Pi MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | Neural DML | 0.02686 | 0.000214 | 0.02342 | 0.02254 |
| 256 | Oracle AIPW | 0.002669 | 0.002669 | 0.000000 | 0.000000 |
| 512 | Neural DML | 0.01314 | 0.000080 | 0.01394 | 0.01803 |
| 512 | Oracle AIPW | 0.003089 | 0.003089 | 0.000000 | 0.000000 |
| 1024 | Neural DML | 0.004042 | 0.000059 | 0.008748 | 0.009004 |
| 1024 | Oracle AIPW | 0.002424 | 0.002424 | 0.000000 | 0.000000 |
| 2048 | Neural DML | 0.002253 | 0.000027 | 0.006470 | 0.006261 |
| 2048 | Oracle AIPW | 0.000826 | 0.000826 | 0.000000 | 0.000000 |

Main observations:

- The AIPW beta mean squared error decreases with sample size for both methods, which is the expected large-sample trend.
- The oracle benchmark achieves lower beta mean squared error than neural DML at every sample size in this run, so there is a visible gap between the method with estimated nuisance functions and the method with oracle nuisance functions.
- The nuisance mean squared errors of the neural DML estimator decrease as `n` grows, while the oracle nuisance errors are numerically zero up to floating-point precision, as they should be.
- A striking feature of this experiment is that the initial joint least-squares beta parameter from neural DML is substantially more accurate than the final AIPW beta estimate. In this design, the AIPW correction appears to introduce noticeable variance relative to the jointly trained beta parameter, even though it is theoretically motivated as the debiased target.
- The gap between neural DML and oracle narrows as `n` increases, but it remains present at `n = 2048` in this 10-trial run.

Generated figures:

- `examples/plm/figs/1.1_1_beta_hat_mse.png`
- `examples/plm/figs/1.1_1_beta_init_mse.png`
- `examples/plm/figs/1.1_1_mu_mse.png`
- `examples/plm/figs/1.1_1_pi_mse.png`

Suggested presentation items:

- a summary table of the averaged performance metrics by sample size,
- a separate plot of beta-estimation mean squared error against `n`,
- a separate plot of initial-beta mean squared error against `n`,
- a separate plot of `mu` mean squared error against `n`,
- a separate plot of `pi` mean squared error against `n`,
- a short interpretation discussing whether the oracle benchmark reveals a meaningful gap relative to neural DML.
