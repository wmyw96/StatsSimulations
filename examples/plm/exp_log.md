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

This method implements the double machine learning pipeline for the partial linear model. The data are split into two parts. On the second split, the method fits a neural network approximation to the outcome regression `mu(X)` jointly with a joint least-squares coefficient estimate for `beta`, and separately fits a neural network approximation to the treatment regression `pi(X)`. On the first split, it plugs these nuisance estimates into the augmented inverse-propensity-weighted estimator given by Eq. (1.2) of the PLM paper to produce the final estimate of `beta`.

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

### Method3: Neural DML With Oracle Nuisance Tracking (`PLMDMLOracleTrackingEstimator`)

This method shares the same estimation and prediction pipeline as the neural DML estimator, but it augments the fit routine with an oracle monitoring path. After every training epoch, it evaluates the current nuisance networks on the fitted split and records the mean squared error of `mu_hat(X)` against the oracle `mu(X)` and of `pi_hat(X)` against the oracle `pi(X)`. These trajectory diagnostics are used only for optimization monitoring and do not affect the fitted parameter updates.

Default hyper-parameters:

- `L = 3`: network depth for the nuisance models.
- `N = 512`: hidden-layer width for the nuisance models.
- `lambda_mu = 1e-4`: L2 regularization level for the outcome network.
- `lambda_pi = 1e-4`: L2 regularization level for the treatment network.
- `niter = 200`: number of training epochs recorded in the trajectory plot.
- `lr = 1e-3`: Adam learning rate.
- `batch_size = 1024`: mini-batch size used during training.
- `device = "cpu"` by default: computation device for the neural network fitting routine.

### Method4: Paper Minimax Debiasing Estimator (`PLMMinimaxDebiasEstimator`)

This method follows the estimator construction in equation `(2.3)` of `aplm.pdf`. It first fits the initial outcome model and joint least-squares coefficient on the second split, as in the neural DML baseline. It then returns to the first split and solves a minimax empirical debiasing problem over bounded weights `a_1, ..., a_n`, where an adversarial difference-class neural network searches for the hardest imbalance in the class `{\beta T + f(X) : \beta \in \mathbb{R}, f \in \mathcal{G}_\mu - \mathcal{G}_\mu}`. The final coefficient estimate is the weighted average `n^{-1} \sum_i (Y_i - \hat{g}(X_i)) a_i`.

In our implementation, the inner stabilization term uses

```text
(1 / n) * sum_i (beta * T_i + f(X_i))^2
```

instead of the paper’s original `(1 / n) * sum_i f^2(X_i)`, following the requested numerical-stability adjustment.

Default hyper-parameters:

- `L = 3`: depth of the neural networks used both for the initial nuisance fit and for the adversarial difference-class learner.
- `N = 512`: width of those neural networks.
- `lambda_mu = 1e-4`: L2 regularization level for the initial outcome learner.
- `lambda_pi = 1e-4`: L2 regularization level for the initial treatment learner.
- `niter = 200`: number of epochs used for the initial joint least-squares nuisance fit on the second split.
- `lr = 1e-3`: Adam learning rate for the initial nuisance fit.
- `batch_size = 1024`: mini-batch size for the initial nuisance fit.
- `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default, where `n` denotes the size of the debiasing split `D1`.
- `weight_bound = 5.0`: absolute bound on each empirical debiasing weight.
- `niter_debias = 200` by default: number of outer optimization steps for the debiasing weights.
- `niter_adversary = 5` by default: number of adversary updates per outer debiasing step.
- `debias_lr = 1e-3` by default: learning rate for both the debiasing weights and the adversarial network.
- `variance_mode = "constant_one"` by default: uses `\hat v \equiv 1`, which is appropriate for the current homoskedastic simulation studies.
- `device = "cpu"` by default: computation device for the neural-network routines.

## Evaluation

For each estimator, the experiment records the following quantities.

- AIPW estimate of `beta`: the final coefficient estimate returned by the estimator.
- Squared error of the final `beta` estimate: `(beta_hat - beta_true)^2`.
- Squared error of the joint least-squares `beta` estimate: for the neural DML estimator, this compares the coefficient parameter from the joint least-squares neural-network fit on the second split to the ground truth before the AIPW correction; for the oracle estimator, this is set equal to the final squared error.
- Mean squared error of `mu`: the average squared difference between the predicted outcome regression and the oracle nuisance value on an independent test sample.
- Mean squared error of `pi`: the average squared difference between the predicted treatment regression and the oracle nuisance value on an independent test sample.
- For experiments that study nuisance-product behavior, the evaluator also records the empirical test-sample mean of the fitted product `mu_hat(X) * pi_hat(X)`, together with the oracle mean of `mu(X) * pi(X)`, so that cross-trial scatter plots can compare this nuisance summary against the final beta estimation error.
- For experiments that study optimization dynamics, the evaluator can also store epoch-by-epoch oracle nuisance MSE paths for `mu` and `pi` on the fitted split.

The summary view aggregates these quantities across repeated trials for each experimental configuration.

# Experimental results

## 1.1.1

Archived run for Experiment `1.1.1`, stored in the simulation artifact `1.1_1`.

### Goal

This experiment is intended as a first validation study. Its goal is to verify that the simulation pipeline is working correctly and to assess whether there is a visible performance gap between the neural DML estimator and the oracle benchmark when both are evaluated under the same partial linear model. This archived run focuses on the standard error metrics for `beta`, `mu`, and `pi`; it does not yet provide the product-based nuisance diagnostic that will be treated separately in Experiment `1.1.2`.

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

| n | Method | Beta MSE | Joint LSE Beta MSE | Mu MSE | Pi MSE |
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

- `examples/plm/figs/1.1/1.1_1_beta_hat_mse.png`
- `examples/plm/figs/1.1/1.1_1_beta_init_mse.png`
- `examples/plm/figs/1.1/1.1_1_mu_mse.png`
- `examples/plm/figs/1.1/1.1_1_pi_mse.png`

Suggested presentation items:

- a summary table of the averaged performance metrics by sample size,
- a separate plot of beta-estimation mean squared error against `n`,
- a separate plot of joint least-squares beta mean squared error against `n`,
- a separate plot of `mu` mean squared error against `n`,
- a separate plot of `pi` mean squared error against `n`,
- a short interpretation discussing whether the oracle benchmark reveals a meaningful gap relative to neural DML.

## 1.1.2

Experiment `1.1.2`, stored in the simulation artifact `1.1_2`.

### Goal

This experiment keeps the same one-dimensional partial linear model family but shifts the focus to a nuisance-product diagnostic. The goal is to study how a summary built from the fitted nuisance functions scales with the final beta estimation error across repeated trials, while still comparing neural DML against the oracle benchmark on the standard metrics.

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

Additional diagnostic target:

- In addition to the standard summary metrics, this experiment records the empirical test-sample mean of `mu_hat(X) * pi_hat(X)` for each trial, along with the oracle mean of `mu(X) * pi(X)`.
- The corresponding scatter plot studies how this fitted nuisance-product summary varies with the final beta estimation error across trials.

### Results

The experiment was run with `10` independent trials for each sample size in `{256, 512, 1024, 2048}`. The summary metrics below are trial averages.

| n | Method | Beta MSE | Joint LSE Beta MSE | Mu MSE | Pi MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | Neural DML | 0.01740 | 0.000174 | 0.02631 | 0.03143 |
| 256 | Oracle AIPW | 0.002669 | 0.002669 | 0.000000 | 0.000000 |
| 512 | Neural DML | 0.01140 | 0.000038 | 0.01891 | 0.01590 |
| 512 | Oracle AIPW | 0.003089 | 0.003089 | 0.000000 | 0.000000 |
| 1024 | Neural DML | 0.004671 | 0.000068 | 0.008853 | 0.009395 |
| 1024 | Oracle AIPW | 0.002424 | 0.002424 | 0.000000 | 0.000000 |
| 2048 | Neural DML | 0.002102 | 0.000047 | 0.005552 | 0.005913 |
| 2048 | Oracle AIPW | 0.000826 | 0.000826 | 0.000000 | 0.000000 |

Product diagnostic summary for Neural DML:

| n | Mean of `mu_hat * pi_hat` | Oracle mean of `mu * pi` | Raw corr. with beta squared error |
| --- | ---: | ---: | ---: |
| 256 | 0.49958 | 0.50110 | -0.223 |
| 512 | 0.48679 | 0.50110 | 0.303 |
| 1024 | 0.49949 | 0.50110 | -0.263 |
| 2048 | 0.50477 | 0.50110 | -0.431 |

Main observations:

- The standard beta, `mu`, and `pi` error trends are qualitatively similar to the archived baseline run: oracle AIPW remains better than neural DML at every sample size, and the DML nuisance errors decrease as `n` grows.
- The fitted nuisance-product mean `mean(mu_hat(X) * pi_hat(X))` is already quite close to the oracle benchmark `mean(mu(X) * pi(X))` at every sample size. In this design, the product mean itself is therefore estimated fairly stably.
- Despite that stability, the combined scatter plot does not reveal a strong monotone relationship between the product mean and the final beta squared error across trials. The overall raw correlation is `-0.053`, and the overall log-scale correlation is `-0.091`.
- The per-`n` scatter plots also do not show a consistent direction of association. This suggests that, for this sine-sine one-dimensional design, variation in the scalar nuisance-product mean is not a strong driver of the remaining trial-to-trial beta error.

Generated figures:

- `examples/plm/figs/1.1/1.1.2_beta_hat_mse.png`
- `examples/plm/figs/1.1/1.1.2_beta_joint_lse_mse.png`
- `examples/plm/figs/1.1/1.1.2_mu_mse.png`
- `examples/plm/figs/1.1/1.1.2_pi_mse.png`
- `examples/plm/figs/1.1/1.1.2_beta_error_comparison.png`
- `examples/plm/figs/1.1/1.1.2_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.1/1.1.2_n256_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.1/1.1.2_n512_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.1/1.1.2_n1024_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.1/1.1.2_n2048_mu_pi_product_mean_vs_beta_hat_scatter.png`

## 1.2.1

Archived pre-fix run for Experiment `1.2.1`, stored in the simulation artifact `1.2_1`.

### Goal

This experiment keeps the same sine-sine one-dimensional partial linear model as the `1.1` family, but changes the target coefficient to a nonzero value. The goal is to study how the beta-estimation error scales with the sample size when `beta = 0.5`, focusing on three quantities:

- the neural DML AIPW estimate of `beta`,
- the oracle AIPW estimate of `beta`,
- the neural-network joint least-squares estimate of `beta`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Target coefficient: `beta = 0.5`
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

### Results

This run was produced before the joint-LSE beta training bug was fixed. It is kept for reference, but the corrected rerun in Experiment `1.2.2` should be used for substantive interpretation.

The experiment was run with `10` independent trials for each sample size in `{256, 512, 1024, 2048}`. The table below focuses on the three beta-estimation errors requested for this setting.

| n | DML AIPW Beta MSE | Oracle AIPW Beta MSE | Joint LSE Beta MSE |
| --- | ---: | ---: | ---: |
| 256 | 0.23944 | 0.002669 | 0.19957 |
| 512 | 0.74010 | 0.003089 | 0.19320 |
| 1024 | 0.01988 | 0.002424 | 0.20129 |
| 2048 | 0.009364 | 0.000826 | 0.19508 |

For reference, the nuisance-estimation errors for Neural DML were:

| n | Mu MSE | Pi MSE |
| --- | ---: | ---: |
| 256 | 0.12737 | 0.03157 |
| 512 | 0.11929 | 0.01975 |
| 1024 | 0.11841 | 0.009404 |
| 2048 | 0.11333 | 0.005660 |

Main observations:

- The oracle AIPW estimator remains the strongest benchmark across all sample sizes, with beta mean squared error decreasing from `0.002669` at `n = 256` to `0.000826` at `n = 2048`.
- The neural DML AIPW estimator is much more variable in this nonzero-beta setting. Its error is large at `n = 256`, becomes even worse at `n = 512`, and then drops substantially at `n = 1024` and `n = 2048`.
- The joint least-squares beta estimate is much more stable than the DML AIPW estimator at small and moderate sample sizes, but it does not show the same clear improvement with `n`. Its mean squared error stays near `0.2` across the whole range.
- At the larger sample sizes, the DML AIPW estimator overtakes the joint least-squares estimator by a wide margin: at `n = 2048`, the DML AIPW beta MSE is `0.009364`, versus `0.19508` for the joint least-squares beta estimate.
- The new beta-comparison figure therefore highlights a useful tradeoff in this design: the joint least-squares beta is more stable in finite samples around `n = 256` to `512`, while the debiased DML AIPW estimate scales much better once the sample size is large enough.

Generated figures:

- `examples/plm/figs/1.2/1.2.1_beta_hat_mse.png`
- `examples/plm/figs/1.2/1.2.1_beta_joint_lse_mse.png`
- `examples/plm/figs/1.2/1.2.1_beta_error_comparison.png`
- `examples/plm/figs/1.2/1.2.1_mu_mse.png`
- `examples/plm/figs/1.2/1.2.1_pi_mse.png`

## 1.2.2

Corrected rerun for Experiment `1.2.2`, stored in the simulation artifact `1.2_2`.

### Goal

This experiment reruns the `beta = 0.5` sine-sine PLM after fixing the joint least-squares beta training routine. The goal is to re-evaluate how the beta-estimation error scales with the sample size for:

- the neural DML AIPW estimate of `beta`,
- the oracle AIPW estimate of `beta`,
- the neural-network joint least-squares estimate of `beta`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Target coefficient: `beta = 0.5`
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
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

The corrected experiment was run with `10` independent trials for each sample size in `{256, 512, 1024, 2048}`. The table below focuses on the three beta-estimation errors requested for this setting.

| n | DML AIPW Beta MSE | Oracle AIPW Beta MSE | Joint LSE Beta MSE |
| --- | ---: | ---: | ---: |
| 256 | 0.02043 | 0.002669 | 0.02687 |
| 512 | 0.01071 | 0.003089 | 0.01860 |
| 1024 | 0.002623 | 0.002424 | 0.004766 |
| 2048 | 0.002177 | 0.000826 | 0.002113 |

For reference, the nuisance-estimation errors for Neural DML were:

| n | Mu MSE | Pi MSE |
| --- | ---: | ---: |
| 256 | 0.03756 | 0.02358 |
| 512 | 0.02983 | 0.01930 |
| 1024 | 0.01235 | 0.008522 |
| 2048 | 0.007612 | 0.005096 |

Main observations:

- The corrected joint least-squares beta estimate is now consistent with the intended behavior: its mean squared error decreases sharply with `n`, from `0.02687` at `n = 256` to `0.002113` at `n = 2048`.
- The neural DML AIPW estimator also improves steadily with `n`, from `0.02043` at `n = 256` to `0.002177` at `n = 2048`.
- The oracle AIPW benchmark remains the best method overall, but the gap is now much smaller than in the broken `1.2.1` run. By `n = 1024` and `n = 2048`, the corrected DML and joint-LSE beta errors are both close to the oracle benchmark.
- Compared with the pre-fix run `1.2.1`, the corrected `1.2.2` results resolve the earlier inconsistency: the joint-LSE beta estimate no longer stays flat around an error level near `0.2`, and the DML AIPW curve no longer shows the pathological spike at `n = 512`.

Generated figures:

- `examples/plm/figs/1.2/1.2.2_beta_hat_mse.png`
- `examples/plm/figs/1.2/1.2.2_beta_joint_lse_mse.png`
- `examples/plm/figs/1.2/1.2.2_beta_error_comparison.png`
- `examples/plm/figs/1.2/1.2.2_mu_mse.png`
- `examples/plm/figs/1.2/1.2.2_pi_mse.png`
- `examples/plm/figs/1.2/1.2.2_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.2/1.2.2_n256_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.2/1.2.2_n512_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.2/1.2.2_n1024_mu_pi_product_mean_vs_beta_hat_scatter.png`
- `examples/plm/figs/1.2/1.2.2_n2048_mu_pi_product_mean_vs_beta_hat_scatter.png`

## 1.3.1

Archived preliminary run for Experiment `1.3.1`, stored in the simulation artifact `1.3_1`.

### Goal

This experiment keeps the same one-dimensional sine-sine partial linear model as the `1.1` and `1.2` families, but now randomizes the ground-truth coefficient across trials. The goal is to study how the average error scales with the sample size when each trial draws

```text
beta ~ Unif[-0.5, 0.5],
```

and to summarize the scaling behavior of the main beta and nuisance quantities in a single unified plot. This archived run was the first implementation of the random-beta design and is kept for reference.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample sizes: `n in {256, 512, 1024, 2048}`
- Test sample size: `n_test = 10000`
- Number of trials per sample size: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 1e-4`
- Treatment-network regularization: `lambda_pi = 1e-4`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Design note:

- The evaluator seeds each trial by its trial index. As a result, for a fixed trial id the same realized `beta` value is reused across the different sample sizes. This keeps the cross-`n` comparison aligned at the trial level while still averaging over a broad range of coefficient values across the 30 trials.
- In this preliminary run, the neural DML estimator used a fixed PyTorch seed across trials. That means the Monte Carlo variation comes from the sampled data and coefficient draws, but not from trial-to-trial variation in network initialization or DataLoader shuffling.

### Results

The experiment was run with `30` trials for each sample size in `{256, 512, 1024, 2048}`. Across the 30 seeded trials, the realized coefficients ranged from approximately `-0.4896` to `0.4670`, with an average close to zero (`0.000863`).

Average mean squared errors:

| n | Oracle AIPW Beta MSE | DML AIPW Beta MSE | Joint LSE Beta MSE | DML Mu MSE | DML Pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.004472 | 0.156432 | 0.024272 | 0.039191 | 0.027801 |
| 512 | 0.004564 | 0.013441 | 0.023545 | 0.029614 | 0.016700 |
| 1024 | 0.002332 | 0.006339 | 0.007433 | 0.014810 | 0.008994 |
| 2048 | 0.000933 | 0.003363 | 0.002195 | 0.007766 | 0.005506 |

Main observations:

- This preliminary 30-trial run was useful for smoke-testing the random-beta pipeline, but its DML curves are visibly noisier than we would like for a reference figure.
- The oracle AIPW benchmark decreases with sample size as expected, and the nuisance errors also shrink with `n`.
- The DML AIPW estimate shows the largest instability at `n = 256`, which is one of the reasons this run is archived rather than treated as the preferred reference.

Generated figure:

- `examples/plm/figs/1.3/1.3.1_unified_mse_scaling.png`

## 1.3.2

Corrected follow-up run for Experiment `1.3.2`, stored in the simulation artifact `1.3_2`.

### Goal

This experiment repeats the random-beta setting of `1.3.1`, but fixes two bookkeeping issues in the simulation framework:

- result files are now validated before a resume appends new trials,
- the neural DML estimator now uses trial-specific PyTorch randomness instead of a single fixed training seed across the whole experiment.

The goal is to provide the clean reference record for the random-beta design.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample sizes: `n in {256, 512, 1024, 2048}`
- Test sample size: `n_test = 10000`
- Number of trials per sample size: `100`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 1e-4`
- Treatment-network regularization: `lambda_pi = 1e-4`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Design note:

- The evaluator still uses common random numbers across sample sizes: for a fixed trial id, the same realized `beta` is reused across `n`.
- Unlike `1.3.1`, the neural DML estimator now receives a trial-specific PyTorch seed equal to the trial seed, so the 100-trial average includes both data randomness and trial-to-trial training randomness.

### Results

The experiment was run with `100` trials for each sample size in `{256, 512, 1024, 2048}`. Across the 100 seeded trials, the realized coefficients ranged from approximately `-0.4896` to `0.4890`, with an average of `-0.0262`.

Average mean squared errors:

| n | Oracle AIPW Beta MSE | DML AIPW Beta MSE | Joint LSE Beta MSE | DML Mu MSE | DML Pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.012023 | 0.061531 | 0.031601 | 0.049235 | 0.028331 |
| 512 | 0.005071 | 0.012846 | 0.019718 | 0.027730 | 0.016451 |
| 1024 | 0.001822 | 0.006444 | 0.007043 | 0.014507 | 0.009335 |
| 2048 | 0.000941 | 0.004376 | 0.002716 | 0.008607 | 0.006274 |

Main observations:

- The 100-trial averages give a much smoother picture of the scaling behavior than the archived `1.3.1` run.
- The oracle AIPW benchmark remains the strongest line throughout the experiment, with beta MSE falling from about `1.20e-2` at `n = 256` to `9.41e-4` at `n = 2048`.
- The neural DML AIPW estimator is still the noisiest beta curve, but its scaling is now clearer and more stable under the corrected seed policy: the average beta MSE decreases from `0.061531` at `n = 256` to `0.012846`, `0.006444`, and `0.004376` as `n` doubles.
- The neural-network joint least-squares beta estimate is more stable than DML AIPW at the smaller sample sizes and remains competitive at the larger ones, with average beta MSE `0.031601`, `0.019718`, `0.007043`, and `0.002716`.
- The nuisance curves also improve steadily with the sample size. The `mu` error remains above the `pi` error at every `n`, but both decrease substantially over the observed range.

Generated figure:

- `examples/plm/figs/1.3/1.3.2_unified_mse_scaling.png`

## 1.4.1

Experiment `1.4.1`, stored in the simulation artifact `1.4_1`.

### Goal

This experiment studies whether the neural nuisance learners appear fully optimized by epoch `200`. Instead of focusing on the final beta estimate, it tracks how the oracle mean squared errors of `mu_hat` and `pi_hat` evolve over training epochs.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `20`

Method design:

- Compared method: `PLMDMLOracleTrackingEstimator`
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 1e-4`
- Treatment-network regularization: `lambda_pi = 1e-4`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Tracking design:

- The estimator receives oracle data only so it can measure nuisance MSE during training.
- The recorded `mu` and `pi` trajectories are evaluated on `D2`, the split used to fit the nuisance networks.
- The visualization overlays `20` red `mu` curves and `20` blue `pi` curves on the same axes, with a log-scaled vertical axis for the nuisance MSE.

### Results

The trajectory plot suggests that the nuisance learners improve rapidly early in training, then largely plateau before epoch `200`.

Average oracle nuisance MSE along the training path:

| Epoch | Mu MSE | Pi MSE |
| --- | ---: | ---: |
| 0 | 0.563377 | 0.582682 |
| 50 | 0.121951 | 0.058196 |
| 100 | 0.015878 | 0.007435 |
| 150 | 0.010380 | 0.007901 |
| 200 | 0.011974 | 0.009215 |

Additional trajectory summary:

- For `mu`, the average epoch of the minimum path value is `152.35`, with median `147.5`.
- For `pi`, the average epoch of the minimum path value is `128.65`, with median `119.5`.
- In `11` out of `20` trials, the `mu` minimum is reached by epoch `150`.
- In `15` out of `20` trials, the `pi` minimum is reached by epoch `150`.
- On average, the final epoch is slightly worse than the best epoch by about `0.00527` for `mu` and `0.00251` for `pi`.

Interpretation:

- The first `100` epochs matter a great deal; both nuisance curves fall sharply over that range.
- After about epoch `150`, the gains are small and the average paths start to wobble rather than continue improving monotonically.
- In this design, `200` epochs does not look clearly too short. If anything, it may be mildly longer than necessary for the nuisance learners, since the average `mu` and `pi` errors are slightly higher at epoch `200` than around their best epochs.

Generated figure:

- `examples/plm/figs/1.4/1.4.1_nuisance_mse_paths.png`

## 1.4.2

Experiment `1.4.2`, stored in the simulation artifact `1.4_2`.

### Goal

This experiment keeps the same nuisance-tracking design as `1.4.1`, but varies the shared regularization level `lambda_mu = lambda_pi` to see how the choice of weight decay changes the optimization path of the neural nuisance learners. The goal is to understand whether the baseline choice `lambda = 1e-4` is in a reasonable range and how strongly larger or smaller penalties slow down or stabilize learning.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `20`

Method design:

- Compared method family: `PLMDMLOracleTrackingEstimator`
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration
- Regularization sweep: `lambda_mu = lambda_pi in {2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4}`

Visualization design:

- Figure 1 is a six-panel path plot. Each panel corresponds to one lambda value and overlays the `20` red `mu` curves and `20` blue `pi` curves on the same axes.
- The six panels share the same vertical limits and use a logarithmic MSE axis, so the regularization effect can be compared directly across lambdas.
- Figure 2 is an amortized plot of the trial averages. Each color represents one lambda value, with solid lines for the average `mu` path and dashed lines for the average `pi` path.

### Results

The lambda sweep shows a clear bias-variance pattern: stronger regularization slows nuisance learning and leaves both `mu` and `pi` at higher error levels throughout the path, while the smallest regularization values produce the best late-epoch fits in this design.

Average nuisance MSE at epoch `200` and the epoch of the best average path value:

| lambda | Mu MSE @ 200 | Pi MSE @ 200 | Mu best epoch | Pi best epoch |
| --- | ---: | ---: | ---: | ---: |
| 2e-5 | 0.014589 | 0.009072 | 157 | 118 |
| 5e-5 | 0.013047 | 0.008978 | 190 | 113 |
| 1e-4 | 0.011974 | 0.009215 | 189 | 113 |
| 2e-4 | 0.012257 | 0.010021 | 148 | 110 |
| 4e-4 | 0.021217 | 0.012560 | 152 | 112 |
| 8e-4 | 0.036417 | 0.015509 | 148 | 114 |

Main observations:

- The large-penalty settings `4e-4` and `8e-4` are clearly too aggressive here. They slow learning early and also finish at much higher `mu` and `pi` errors than the smaller penalties.
- The baseline setting `1e-4` remains a sensible middle choice. Its final average `mu` path is slightly better than `5e-5` and close to the best of the sweep, while its `pi` path stays competitive.
- The smallest penalties `2e-5` and `5e-5` learn fastest in the early and middle stages, especially for `pi`, but they also show more late-epoch wobble than the best medium-penalty runs.
- Overall, the sweep supports using `1e-4` as a stable default, while suggesting that mild reductions such as `5e-5` may also be worth considering if we want slightly less shrinkage without moving into the clearly under-regularized regime.

Generated figures:

- `examples/plm/figs/1.4/1.4.2_lambda_path_panels.png`
- `examples/plm/figs/1.4/1.4.2_lambda_average_paths.png`

## 1.4.3

Experiment `1.4.3`, stored in the simulation artifact `1.4_3`.

### Goal

This experiment repeats the lambda-sweep study of `1.4.2`, but broadens the regularization range to cover much smaller and much larger penalties. The goal is to see whether the earlier sweep was too narrow to reveal the true optimization tradeoff and to check whether the baseline choice `lambda = 1e-4` still looks reasonable once the search range spans several multiplicative scales.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `20`

Method design:

- Compared method family: `PLMDMLOracleTrackingEstimator`
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration
- Wide regularization sweep: `lambda_mu = lambda_pi = 5^l * 1e-4` for `l in {-3, -2, -1, 0, 1, 2, 3}`
- Concrete lambda values: `{8e-7, 4e-6, 2e-5, 1e-4, 5e-4, 2.5e-3, 1.25e-2}`

Visualization design:

- Figure 1 is a seven-panel path plot with shared log-scaled vertical limits. Each panel corresponds to one lambda value and overlays the `20` red `mu` curves and `20` blue `pi` curves.
- Figure 2 is the amortized plot of trial averages. Each color represents one lambda value, with solid lines for the average `mu` path and dashed lines for the average `pi` path.

### Results

The wider sweep makes the tradeoff much clearer than `1.4.2`. Very large penalties clearly underfit, while the smallest penalties can reach the best average nuisance errors at intermediate epochs, but they also show more late-epoch wobble than the moderate penalties.

Average nuisance MSE at epoch `200` and the epoch of the best average path value:

| lambda | Mu MSE @ 200 | Pi MSE @ 200 | Mu best epoch | Pi best epoch |
| --- | ---: | ---: | ---: | ---: |
| 8e-7 | 0.009894 | 0.005321 | 169 | 116 |
| 4e-6 | 0.021311 | 0.006475 | 160 | 118 |
| 2e-5 | 0.014589 | 0.009072 | 157 | 118 |
| 1e-4 | 0.011974 | 0.009215 | 189 | 113 |
| 5e-4 | 0.016704 | 0.013527 | 127 | 118 |
| 2.5e-3 | 0.034410 | 0.027198 | 154 | 114 |
| 1.25e-2 | 0.092431 | 0.058068 | 196 | 194 |

Main observations:

- The largest penalties `2.5e-3` and `1.25e-2` are decisively too strong. They underfit throughout training and never come close to the nuisance errors achieved by the smaller lambdas.
- The smallest penalty `8e-7` gives the best final average `pi` error and one of the best final `mu` errors, so the earlier `1.4.2` sweep was indeed missing some stronger-performing low-regularization settings.
- The tiny penalties are not uniformly stable, though. The `4e-6` run achieves one of the best average `mu` minima around epoch `160`, but its final `mu` error at epoch `200` is much worse because the average path rises again late in training.
- The baseline `1e-4` still looks like a robust compromise. It is not the most aggressive fitter, but it stays competitive and substantially more stable than the smallest settings while avoiding the obvious underfitting of the large penalties.
- Taken together, the new sweep suggests that if we want a conservative default for downstream experiments, `1e-4` remains defensible, while very small values such as `8e-7` are interesting candidates when the focus is best attainable nuisance fit rather than path stability.

Generated figures:

- `examples/plm/figs/1.4/1.4.3_lambda_path_panels.png`
- `examples/plm/figs/1.4/1.4.3_lambda_average_paths.png`

## 1.4.4

Experiment `1.4.4`, stored in the simulation artifact `1.4_4`.

### Goal

This experiment revisits the lambda-sweep question from `1.4.3`, but now records the nuisance-learning paths on both the fitted split `D2` and on an independent validation sample. The goal is to separate the in-sample optimization picture from the out-of-sample one and see whether the preferred lambda values change once the nuisance learners are judged on fresh data.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression: `pi(x) = sin(2 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Validation sample size: `n_val = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `50`

Method design:

- Compared method family: `PLMDMLOracleTrackingEstimator`
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Wide regularization sweep: `lambda_mu = lambda_pi = 5^l * 1e-4` for `l in {-3, -2, -1, 0, 1, 2, 3}`
- Concrete lambda values: `{8e-7, 4e-6, 2e-5, 1e-4, 5e-4, 2.5e-3, 1.25e-2}`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Tracking design:

- The nuisance networks are fitted on `D2`.
- After every epoch, the estimator evaluates `mu_hat` and `pi_hat` both on `D2` itself and on an independent oracle validation sample of size `1024`.
- The visualization reports two averaged path figures: one for oracle nuisance MSE on `D2` and one for oracle nuisance MSE on the validation sample.

### Results

This experiment compares the wide lambda sweep side by side on `D2` and on validation. In this run, the two views are much closer than one might fear: the validation curves do not overturn the main ranking from `D2`, although they do make the tradeoff between `8e-7` and `1e-4` a bit clearer.

Final average nuisance MSE at epoch `200`:

| lambda | D2 mu | D2 pi | Validation mu | Validation pi |
| --- | ---: | ---: | ---: | ---: |
| 8e-7 | 0.011243 | 0.005696 | 0.012591 | 0.005480 |
| 4e-6 | 0.018524 | 0.006505 | 0.017296 | 0.006351 |
| 2e-5 | 0.019397 | 0.010953 | 0.019184 | 0.012106 |
| 1e-4 | 0.012636 | 0.009490 | 0.012213 | 0.009210 |
| 5e-4 | 0.015937 | 0.014450 | 0.015928 | 0.014070 |
| 2.5e-3 | 0.033103 | 0.028254 | 0.032252 | 0.028417 |
| 1.25e-2 | 0.104797 | 0.061855 | 0.106776 | 0.061313 |

Interpretation:

- The largest penalties `2.5e-3` and `1.25e-2` still clearly underfit on both `D2` and validation, so the earlier underfitting conclusion was not an artifact of the tracking split.
- The low-penalty end remains strongest, but the winner now depends on which nuisance target we emphasize: `8e-7` gives the best final validation `pi` error, while `1e-4` gives the best final validation `mu` error.
- The `D2` and validation rankings are broadly aligned. That means the earlier `D2`-only tracking was optimistic in the usual sense, but it was not qualitatively misleading for this design.
- As a practical default, `1e-4` still looks like the most balanced choice. If we specifically care about pushing the treatment nuisance as low as possible, then `8e-7` is a serious alternative worth revisiting.

Generated figures:

- `examples/plm/figs/1.4/1.4.4_d2_average_paths.png`
- `examples/plm/figs/1.4/1.4.4_validation_average_paths.png`

## 1.5.1

Experiment `1.5.1`, stored in the simulation artifact `1.5_1`.

### Goal

This experiment studies whether making the treatment regression `pi(x)` harder to estimate also degrades the final beta estimation error. The key question is whether higher treatment-nuisance complexity translates into a clear deterioration of the DML AIPW estimator, or whether the effect is weaker and more indirect. To isolate that question, the experiment keeps the network architecture, sample size, and regularization fixed, and changes only the oscillation level of `pi(x)`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression candidates:
  - `pi(x) = sin(2 pi x)`
  - `pi(x) = sin(4 pi x)`
  - `pi(x) = sin(8 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Visualization design:

- The single summary figure reports the average mean squared error for four quantities as the treatment regression becomes more oscillatory:
  - Oracle AIPW beta MSE,
  - DML AIPW beta MSE,
  - DML `mu` MSE,
  - DML `pi` MSE.
- The horizontal axis indexes the three treatment-regression choices, and the vertical axis uses a logarithmic scale so the four error types can be compared on the same panel.

### Results

Average MSE over `30` trials:

| pi(x) | Oracle AIPW beta MSE | DML AIPW beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `sin(2 pi x)` | 0.002332 | 0.187253 | 0.012780 | 0.009745 |
| `sin(4 pi x)` | 0.002115 | 0.009719 | 0.022983 | 0.019237 |
| `sin(8 pi x)` | 0.001988 | 0.003350 | 0.008293 | 0.084803 |

Main observations:

- The treatment nuisance does become harder to learn as the oscillation frequency increases. The DML `pi` MSE roughly doubles from `sin(2 pi x)` to `sin(4 pi x)`, and then jumps sharply to `0.084803` at `sin(8 pi x)`.
- The final DML beta error does not move in lockstep with the treatment-nuisance error. In this run, the DML AIPW beta MSE is actually worst for the easiest treatment regression `sin(2 pi x)`, then drops substantially for `sin(4 pi x)`, and is smallest for `sin(8 pi x)`.
- The oracle AIPW beta benchmark stays stable across the three designs, with only a mild improvement as the treatment regression becomes more oscillatory. That indicates the large variation is coming from nuisance learning rather than from a fundamental change in the oracle identification difficulty.
- Taken together, these results suggest that treatment-nuisance error alone is not enough to explain the final beta error in this setup. The relation is clearly not monotone here, so the DML beta error appears to depend on a more complicated interaction between the learned nuisances and the orthogonal-score correction.

Generated figures:

- `examples/plm/figs/1.5/1.5.1_pi_complexity_mse_comparison.png`

## 1.5.2

Experiment `1.5.2`, stored in the simulation artifact `1.5_2`.

### Goal

This experiment revisits the `1.5` question with a sharper treatment-regression family. Instead of using plain sine functions, it defines the treatment regression as

```text
pi(x) = sign(sin(2 pi x)) * sin(k pi x),
```

for `k in {2, 4, 8}`. The goal is to see whether a progressively harder and more oscillatory treatment nuisance creates a clearer relationship between treatment-regression estimation error and the final beta error.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression candidates:
  - `pi(x) = sign(sin(2 pi x)) * sin(2 pi x)`
  - `pi(x) = sign(sin(2 pi x)) * sin(4 pi x)`
  - `pi(x) = sign(sin(2 pi x)) * sin(8 pi x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Visualization design:

- The single summary figure again reports the average mean squared error for four quantities:
  - Oracle AIPW beta MSE,
  - DML AIPW beta MSE,
  - DML `mu` MSE,
  - DML `pi` MSE.
- The horizontal axis now indexes the three signed treatment-regression choices above, and the vertical axis uses a logarithmic scale.

### Results

Average MSE over `30` trials:

| pi(x) | Oracle AIPW beta MSE | DML AIPW beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `sign(sin(2 pi x)) * sin(2 pi x)` | 0.002124 | 0.004375 | 0.010964 | 0.009093 |
| `sign(sin(2 pi x)) * sin(4 pi x)` | 0.002074 | 0.002998 | 0.008367 | 0.019201 |
| `sign(sin(2 pi x)) * sin(8 pi x)` | 0.002130 | 0.002219 | 0.007388 | 0.117717 |

Main observations:

- The treatment nuisance error now increases very clearly with oscillation frequency. The DML `pi` MSE rises from `0.009093` to `0.019201` and then to `0.117717`, so this signed family does make the treatment regression substantially harder.
- Despite that, the DML AIPW beta error does not deteriorate. In fact, it decreases mildly across the three settings, from `0.004375` to `0.002998` to `0.002219`.
- The oracle AIPW benchmark remains essentially flat across all three signed designs, with beta MSE around `0.0021`. By the hardest setting, the DML beta error is very close to the oracle benchmark even though the treatment nuisance error is much larger.
- In this signed family, the evidence against a simple monotone link is even stronger than in `1.5.1`: making `pi(x)` much harder to estimate does not by itself force a larger DML beta error.

Generated figures:

- `examples/plm/figs/1.5/1.5.2_pi_complexity_mse_comparison.png`

## 1.5.3

Experiment `1.5.3`, stored in the simulation artifact `1.5_3`.

### Goal

This experiment corrects the `1.5.2` specification. The treatment regression is now

```text
pi(x) = sign(sin(2 pi x)) * |sin(k pi x)|,
```

for `k in {2, 4, 8}`. The goal is again to test whether increasingly hard treatment-regression estimation translates into larger beta error, but now under the absolute-value version you intended.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression candidates:
  - `pi(x) = sign(sin(2 pi x)) * |sin(2 pi x)|`
  - `pi(x) = sign(sin(2 pi x)) * |sin(4 pi x)|`
  - `pi(x) = sign(sin(2 pi x)) * |sin(8 pi x)|`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Visualization design:

- The single summary figure reports the average mean squared error for:
  - Oracle AIPW beta,
  - DML AIPW beta,
  - DML `mu`,
  - DML `pi`.
- The horizontal axis indexes the three corrected treatment-regression choices, and the vertical axis is logarithmic.

### Results

Average MSE over `30` trials:

| pi(x) | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sign(sin(2 pi x)) * |sin(2 pi x)|` | 0.002332 | 0.187253 | 0.005774 | 0.012780 | 0.009745 |
| `sign(sin(2 pi x)) * |sin(4 pi x)|` | 0.002194 | 0.005391 | 0.003864 | 0.009786 | 0.027950 |
| `sign(sin(2 pi x)) * |sin(8 pi x)|` | 0.002222 | 0.003443 | 0.001744 | 0.008389 | 0.083315 |

Main observations:

- The first case reduces exactly to `sin(2 pi x)`, so its numbers match the `1.5.1` baseline. That provides a useful sanity check that the corrected function family is wired properly.
- The treatment nuisance error again grows sharply with oscillation frequency: DML `pi` MSE goes from `0.009745` to `0.027950` to `0.083315`.
- The key point is that both beta estimators improve as `k` increases. The joint LSE beta MSE drops from `0.005774` to `0.003864` to `0.001744`, and the DML AIPW beta MSE drops even more dramatically from `0.187253` to `0.005391` to `0.003443`.
- That pattern suggests the improvement is not caused by the AIPW correction alone. Instead, the data geometry itself is becoming more favorable for beta estimation: when `pi(x)` is exactly `mu(x)`, the treatment signal carried by `T` is highly aligned with the outcome nuisance, making `beta * T` and `mu(X)` difficult to separate in the regression fit. As `k` increases, `pi(x)` is still correlated with `mu(x)`, but it is no longer identical, so the collinearity between the treatment-related structure and the outcome nuisance weakens.
- The oracle AIPW beta benchmark stays essentially flat across the three settings, so the main change is still happening in how learnable and separable the nuisance and target components are for the neural estimator, rather than in the oracle target itself.
- Taken together, the corrected absolute-value family suggests that nuisance difficulty alone is not the right explanatory variable. What matters much more here is the overlap between `mu(X)` and the systematic part of `T`.

Generated figures:

- `examples/plm/figs/1.5/1.5.3_pi_complexity_mse_comparison.png`

## 1.5.4

Experiment `1.5.4`, stored in the simulation artifact `1.5_4`.

### Goal

This experiment tests the four-function progressive `pi` family designed to increase both approximation difficulty and overlap with `mu(x) = sin(2 pi x)`. The intended goal was to find a sequence in which both the DML beta estimator and the joint least-squares beta estimator degrade as the treatment regression becomes harder and more confounded with the outcome nuisance.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression candidates:
  - `pi_1(x) = 0.25 * mu(x) + sqrt(1 - 0.25^2) * sin(4 pi x)`
  - `pi_2(x) = 0.5 * mu(x) + sqrt(1 - 0.5^2) * sin(8 pi x)`
  - `pi_3(x) = 0.75 * mu(x) + sqrt(1 - 0.75^2) * sqrt(0.5) * sign(sin(8 pi x))`
  - `pi_4(x) = 0.9 * mu(x) + sqrt(1 - 0.9^2) * r(x)`
- Here `r(x)` is a multiscale signed-wave component built from `sign(sin(8 pi x))`, `sign(sin(16 pi x))`, and `sign(sin(32 pi x))`.
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002154 | 0.013797 | 0.021027 | 0.020802 | 0.022323 |
| `pi_2` | 0.002073 | 0.002373 | 0.001469 | 0.007759 | 0.086517 |
| `pi_3` | 0.002110 | 0.003856 | 0.001862 | 0.008438 | 0.081240 |
| `pi_4` | 0.002128 | 0.002578 | 0.004561 | 0.011688 | 0.043856 |

Main observations:

- This family did not achieve the monotone degradation target. The treatment nuisance error does rise substantially from `pi_1` to `pi_2`, but the DML beta and joint LSE beta errors both improve sharply rather than worsen.
- The worst beta performance is actually at `pi_1`, the smoothest and least aligned member of the family. After that, the beta errors stay relatively small, even though the nuisance fits become rougher and the treatment-regression error is much larger.
- The progressive family was still informative: it shows that simply increasing both roughness and overlap in an informal way is not enough. The geometry of the regression problem is still changing in a way that sometimes helps beta estimation rather than hurting it.
- In particular, the rougher members seem to make `pi` harder to approximate, but not in a way that consistently damages the target estimation stage. So this is not yet the right family for constructing a clean “harder nuisance implies worse beta” stress test.

Generated figures:

- `examples/plm/figs/1.5/1.5.4_pi_complexity_mse_comparison.png`

## 1.5.5

Experiment `1.5.5`, stored in the simulation artifact `1.5_5`.

### Goal

This experiment revisits the `pi`-complexity question with a more controlled family. The key design change is to keep the overlap between `pi(x)` and `mu(x) = sin(2 pi x)` very high and nearly fixed, while increasing the roughness of the treatment regression in a clearer progression. The main purpose is to make the treatment nuisance genuinely harder to estimate in a monotone way before asking whether the beta estimators degrade with it.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Outcome regression: `mu(x) = sin(2 pi x)`
- Treatment regression candidates:
  - `pi_1(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * sqrt(2) * cos(2 pi x)`
  - `pi_2(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * sqrt(2) * cos(8 pi x)`
  - `pi_3(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * sign(sin(8 pi x))`
  - `pi_4(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * sign(sin(64 pi x))`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002319 | 0.002882 | 0.005373 | 0.011451 | 0.008196 |
| `pi_2` | 0.002379 | 0.023270 | 0.005264 | 0.011744 | 0.013576 |
| `pi_3` | 0.002250 | 0.005908 | 0.006684 | 0.011754 | 0.018740 |
| `pi_4` | 0.002312 | 0.003795 | 0.002493 | 0.009036 | 0.049803 |

Main observations:

- This family succeeds at the first design goal: the treatment nuisance really does get harder in a clean progression. The DML `pi` MSE increases monotonically across the four candidates: `0.008196 -> 0.013576 -> 0.018740 -> 0.049803`.
- The outcome nuisance does not change much across the first three candidates, which is exactly what we wanted from a cleaner stress test. The DML `mu` MSE stays near `0.0115` before improving slightly in the roughest case.
- The beta estimators still do not degrade monotonically with that nuisance difficulty. The DML AIPW beta error is worst for `pi_2`, improves for `pi_3`, and improves again for `pi_4`; the joint LSE beta error is fairly flat through `pi_3` and then actually drops for `pi_4`.
- So `1.5.5` is a better experiment than the earlier families in one important sense: it clearly isolates a monotone increase in treatment-regression difficulty. What it shows is that, even after doing that, beta performance is still driven by more than just the standalone MSE of `pi`. The precise geometry of how the treatment regression interacts with `mu(x)` and the orthogonal score still matters.

Generated figures:

- `examples/plm/figs/1.5/1.5.5_pi_complexity_mse_comparison.png`

## 1.5.6

Experiment `1.5.6`, stored in the simulation artifact `1.5_6`.

### Goal

This experiment tests whether moving from a one-dimensional nuisance structure to a genuinely four-dimensional one makes the DML problem meaningfully harder. The goal is not only to increase the difficulty of estimating `pi(x)`, but also to make the overall target-estimation problem materially harsher for both DML and joint least-squares.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 4`
- Outcome regression: `mu(x) = 0.5 * sum_{j=1}^4 sin(2 pi x_j)`
- Treatment regression candidates:
  - `pi_1(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * 0.5 * sum_{j=1}^4 sqrt(2) * cos(2 pi x_j)`
  - `pi_2(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * 0.5 * sum_{j=1}^4 sqrt(2) * cos(8 pi x_j)`
  - `pi_3(x) = 0.98 * mu(x) + sqrt(1 - 0.98^2) * 0.5 * sum_{j=1}^4 sign(sin(32 pi x_j))`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002639 | 0.195190 | 0.435389 | 0.383664 | 0.195469 |
| `pi_2` | 0.002658 | 0.120992 | 0.352667 | 0.348962 | 0.254023 |
| `pi_3` | 0.002554 | 0.121511 | 0.334499 | 0.345050 | 0.251645 |

Main observations:

- Moving to `d = 4` succeeds at the main goal: the entire estimation problem becomes dramatically harder. Compared with the one-dimensional `1.5.5` family, the DML nuisance errors jump from roughly `10^{-2}` to roughly `10^{-1}`, and the beta estimators degrade by one to two orders of magnitude.
- The treatment nuisance does get harder as we move from the low-frequency smooth `pi_1` to the higher-frequency smooth `pi_2`. The discontinuous `pi_3` stays comparably difficult to `pi_2`, rather than becoming much worse again.
- The DML and joint LSE beta errors are both very large in all three settings, which is the key qualitative change relative to the one-dimensional experiments. In that sense, the four-dimensional setup does make DML genuinely hard to estimate.
- The beta errors still do not rise monotonically with the treatment-regression difficulty. Instead, the hardest overall beta case is the smooth but highly confounded `pi_1`, while `pi_2` and `pi_3` have similarly bad but slightly smaller beta errors. This again points to an interaction between nuisance difficulty and identification geometry, rather than a simple one-number control through `pi` MSE alone.

Generated figures:

- `examples/plm/figs/1.5/1.5.6_pi_complexity_mse_comparison.png`

## 1.5.7

Experiment `1.5.7`, stored in the simulation artifact `1.5_7`.

### Goal

This experiment tries to isolate treatment-regression difficulty more cleanly than `1.5.6`. The design keeps the outcome regression easy, depending only on `x_1`, while the treatment regression depends on `x_2`, `x_3`, and `x_4` through increasingly rough structures. The intention is to make `pi(x)` harder without simultaneously making the true `mu(x)` more complicated.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 4`
- Outcome regression: `mu(x) = sin(2 pi x_1)`
- Treatment regression candidates:
  - `pi_1(x) = 0.98 * sin(2 pi x_1) + sqrt(1 - 0.98^2) * (sin(2 pi x_2) + sin(2 pi x_3) + sin(2 pi x_4)) / sqrt(3)`
  - `pi_2(x) = 0.98 * sin(2 pi x_1) + sqrt(1 - 0.98^2) * (sin(8 pi x_2) + sin(8 pi x_3) + sin(8 pi x_4)) / sqrt(3)`
  - `pi_3(x) = 0.98 * sin(2 pi x_1) + sqrt(1 - 0.98^2) * sign(prod_{j=2}^4 sin(8 pi x_j))`
  - `pi_4(x) = 0.98 * sin(2 pi x_1) + sqrt(1 - 0.98^2) * sign(prod_{j=2}^4 sin(32 pi x_j))`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002712 | 0.113622 | 0.452936 | 0.384989 | 0.134818 |
| `pi_2` | 0.002747 | 0.081255 | 0.413697 | 0.372124 | 0.157720 |
| `pi_3` | 0.002783 | 0.062631 | 0.340507 | 0.344817 | 0.200159 |
| `pi_4` | 0.002674 | 0.057620 | 0.362190 | 0.358094 | 0.192388 |

Main observations:

- The treatment nuisance does get harder in a much cleaner way than before. The DML `pi` MSE rises from `0.134818` to `0.157720` to `0.200159`, then stays at a similarly large level for the roughest case.
- However, this experiment still does not isolate `pi` difficulty as cleanly as intended, because the DML `mu` MSE remains very large throughout, even though the true `mu(x)` only depends on `x_1`. So the four-dimensional learning problem itself is still making the outcome nuisance difficult for the neural estimator.
- Both beta estimators remain very poor compared with the oracle benchmark, but they again do not degrade monotonically with the treatment-regression difficulty. In fact, the DML beta MSE decreases across the family, and the joint LSE beta MSE also improves from `pi_1` to `pi_3` before bouncing slightly at `pi_4`.
- So `1.5.7` improves on `1.5.6` in one sense: it makes `pi` harder while keeping the true `mu` structurally simple. But empirically, the fitted `mu` is still hard, which means this design still does not produce a clean one-factor stress test for the role of treatment-regression error in the beta estimator.

Generated figures:

- `examples/plm/figs/1.5/1.5.7_pi_complexity_mse_comparison.png`

## 1.5.8

Experiment `1.5.8`, stored in the simulation artifact `1.5_8`.

### Goal

This experiment keeps `mu(x)` relatively easy, as requested, using `mu(x) = sin(pi x_1) + cos(pi x_2)`. The design goal is to find a treatment-regression family for which both the DML treatment-regression MSE and the DML beta-estimation MSE increase together.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 4`
- Outcome regression: `mu(x) = sin(pi x_1) + cos(pi x_2)`
- Treatment regression candidates:
  - `pi_1(x) = mu(x) + 0.05 * (sin(2 pi x_1) + cos(2 pi x_2)) / sqrt(2)`
  - `pi_2(x) = mu(x) + 0.18 * sign(sin(8 pi x_1)) * sign(cos(8 pi x_2))`
  - `pi_3(x) = mu(x) + 0.20 * sign(sin(8 pi x_1)) * sign(cos(8 pi x_2))`
  - `pi_4(x) = mu(x) + 0.25 * sign(sin(8 pi x_1)) * sign(cos(8 pi x_2))`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002802 | 0.044891 | 0.267990 | 0.348808 | 0.055569 |
| `pi_2` | 0.002902 | 0.045978 | 0.184998 | 0.263691 | 0.113965 |
| `pi_3` | 0.002919 | 0.082161 | 0.168694 | 0.248940 | 0.128094 |
| `pi_4` | 0.002963 | 0.056878 | 0.135596 | 0.214755 | 0.165524 |

Main observations:

- This is the first family in the `1.5` sequence where the intended pattern shows up clearly for the first three settings. Both DML `pi` MSE and DML beta MSE increase together from `pi_1` to `pi_3`: `0.055569 -> 0.113965 -> 0.128094` for `pi`, and `0.044891 -> 0.045978 -> 0.082161` for beta.
- The fourth setting increases the treatment-regression error further, but the DML beta error drops back from `0.082161` to `0.056878`. So the family is only partially successful at the full four-point level.
- The true `mu(x)` is indeed simpler than in the earlier four-dimensional experiments, but the fitted DML `mu` error is still not small. It decreases steadily across the family, which suggests that part of the improvement in beta estimation may still be coming from easier nuisance fitting on the outcome side, not only from the treatment side.
- The practical conclusion is that `1.5.8` is the closest match so far to the requested design. If we want a clean monotone experiment, the first three settings of `1.5.8` are the most defensible subset.

Generated figures:

- `examples/plm/figs/1.5/1.5.8_pi_complexity_mse_comparison.png`

## 1.5.9

Experiment `1.5.9`, stored in the simulation artifact `1.5_9`.

### Goal

This experiment implements the correlated `g_1/g_2` idea directly. The intention is to keep the outcome regression mostly smooth, while building the treatment regression from the same smooth component plus an increasingly amplified rough component that stays highly aligned with it.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 4`
- Smooth component: `g_1(x) = sin(pi x_1) + cos(pi x_2)`
- Rough correlated component: `g_2(x) = sign(g_1(x)) * 0.5 * (|sin(8 pi x_1)| + |cos(8 pi x_2)|)`
- Outcome regression: `mu(x) = 0.95 * g_1(x) + 0.05 * g_2(x)`
- Treatment regression candidates:
  - `pi_1(x) = g_1(x) + 0.05 * g_2(x)`
  - `pi_2(x) = g_1(x) + 0.10 * g_2(x)`
  - `pi_3(x) = g_1(x) + 0.20 * g_2(x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002801 | 0.037985 | 0.249516 | 0.342280 | 0.056181 |
| `pi_2` | 0.002833 | 0.042018 | 0.248710 | 0.356475 | 0.058495 |
| `pi_3` | 0.002903 | 0.033719 | 0.229804 | 0.357388 | 0.064695 |

Main observations:

- The treatment nuisance error does increase across the family, but only mildly: `0.056181 -> 0.058495 -> 0.064695`.
- The DML beta error does not increase cleanly with it. It rises slightly from `pi_1` to `pi_2`, then drops at `pi_3`.
- So the correlated `g_1/g_2` construction is conceptually appealing, but in this concrete parameterization it still does not produce the monotone beta degradation we are looking for.
- One likely reason is that the rough component is still too small relative to the dominant smooth component. The treatment-regression difficulty changes, but not enough to dominate the overall fitting geometry.

Generated figures:

- `examples/plm/figs/1.5/1.5.9_pi_complexity_mse_comparison.png`

## 1.5.10

Experiment `1.5.10`, stored in the simulation artifact `1.5_10`.

### Goal

This experiment revisits the correlated `g_1/g_2` design using a much wider coefficient range in the treatment regression: `g_1(x) + 0.05 g_2(x)`, `g_1(x) + 0.5 g_2(x)`, and `g_1(x) + 1.0 g_2(x)`. The purpose is to make the treatment-regression difficulty change much more substantially than it did in `1.5.9`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 4`
- Smooth component: `g_1(x) = sin(pi x_1) + cos(pi x_2)`
- Rough correlated component: `g_2(x) = sign(g_1(x)) * 0.5 * (|sin(8 pi x_1)| + |cos(8 pi x_2)|)`
- Outcome regression: `mu(x) = 0.95 * g_1(x) + 0.05 * g_2(x)`
- Treatment regression candidates:
  - `pi_1(x) = g_1(x) + 0.05 * g_2(x)`
  - `pi_2(x) = g_1(x) + 0.50 * g_2(x)`
  - `pi_3(x) = g_1(x) + 1.00 * g_2(x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = 0.5`
- Outcome noise scale: `sigma_eps = 0.5`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002801 | 0.037985 | 0.249516 | 0.342280 | 0.056181 |
| `pi_2` | 0.003166 | 0.024524 | 0.183829 | 0.370814 | 0.111811 |
| `pi_3` | 0.003866 | 0.009362 | 0.098294 | 0.306001 | 0.276027 |

Main observations:

- The wider range succeeds at making the treatment-regression error much more visibly different. The DML `pi` MSE now increases strongly across the three settings: `0.056181 -> 0.111811 -> 0.276027`.
- However, the DML beta MSE moves in the opposite direction: `0.037985 -> 0.024524 -> 0.009362`. The joint LSE beta MSE also decreases strongly across the family.
- So this wider range confirms that the main issue in `1.5.9` was not merely that the coefficient range was too small. Even after making the treatment-regression difficulty much more pronounced, beta estimation still becomes easier in this family.
- The most likely explanation is that increasing the rough correlated component is also changing the geometry of the target problem in a favorable way for identifying beta, even while it makes `pi` harder to approximate pointwise.

Generated figures:

- `examples/plm/figs/1.5/1.5.10_pi_complexity_mse_comparison.png`

## 1.5.11

Experiment `1.5.11`, stored in the simulation artifact `1.5_11`.

### Goal

This experiment revisits the correlated `g_1/g_2` idea in a one-dimensional setting while fixing the noise-stability concern from the previous families. The design makes both the treatment noise `u` and the outcome noise `eps` have unit variance, keeps the dominant smooth signal simple with `g_1(x) = sin(pi x)`, and uses a rough component `g_2(x)` that is still highly correlated with `g_1(x)`. The goal is to see whether a more stable denominator and a simpler one-dimensional geometry produce a clearer relationship between treatment-regression difficulty and beta estimation.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Dominant smooth component: `g_1(x) = sin(pi x)`
- Rough correlated component: `g_2(x) = sign(sin(pi x)) * 0.5 * (|sin(8 pi x)| + |sin(16 pi x)|)`
- Approximate correlation between `g_1(x)` and `g_2(x)` under `X ~ Unif[-1,1]`: `0.845`
- Outcome regression: `mu(x) = 0.95 * g_1(x) + 0.05 * g_2(x)`
- Treatment regression candidates:
  - `pi_1(x) = g_1(x) + 0.05 * g_2(x)`
  - `pi_2(x) = g_1(x) + 0.50 * g_2(x)`
  - `pi_3(x) = g_1(x) + 1.00 * g_2(x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average MSE over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002080 | 0.002178 | 0.002522 | 0.068690 | 0.062680 |
| `pi_2` | 0.002080 | 0.002317 | 0.006083 | 0.084863 | 0.079773 |
| `pi_3` | 0.002082 | 0.002008 | 0.008175 | 0.092341 | 0.124890 |

Main observations:

- The treatment nuisance now clearly gets harder across the family: DML `pi` MSE increases from `0.062680` to `0.079773` to `0.124890`.
- The outcome nuisance also gets modestly harder, but much more slowly: DML `mu` MSE rises from `0.068690` to `0.084863` to `0.092341`.
- Under this unit-variance one-dimensional design, the DML beta MSE stays very stable and remains close to the oracle AIPW beta MSE across all three settings, rather than drifting downward as it did in `1.5.10`.
- The quantity that degrades more visibly is the joint least-squares beta estimate: its MSE rises from `0.002522` to `0.006083` to `0.008175`.
- So the variance correction and the simpler one-dimensional geometry do make the experiment more stable, but they still do not produce a strong monotone deterioration of the final DML AIPW beta estimator.

Generated figures:

- `examples/plm/figs/1.5/1.5.11_pi_complexity_mse_comparison.png`

## 1.5.12

Experiment `1.5.12`, stored in the simulation artifact `1.5_12`.

### Goal

This experiment tries to make the fitted nuisance residuals align more directly by forcing the outcome regression and the treatment regression to share the same hard-to-learn component with the same sign. The design keeps the one-dimensional unit-variance noise setting from `1.5.11`, but changes the signal structure so that both `mu(x)` and `pi(x)` contain the same rough component `h(x)` and the treatment family increases the weight on `h(x)` while keeping the overall variance of `pi(X)` approximately fixed.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 1`
- Easy signal: `g(x) = sin(pi x)`
- Rough aligned component before normalization: `h_raw(x) = sign(sin(pi x)) * 0.5 * (|sin(8 pi x)| + |sin(16 pi x)|)`
- Centered and scaled hard component: `h(x) = (h_raw(x) - E[h_raw(X)]) / sd(h_raw(X))`
- Approximate correlation between `g(x)` and `h(x)` under `X ~ Unif[-1,1]`: `0.845`
- Outcome regression:
  - `mu(x) = s_mu^{-1} * (g(x) + 1.0 * h(x))`
- Treatment regression candidates:
  - `pi_1(x) = s_0.5^{-1} * (g(x) + 0.5 * h(x))`
  - `pi_2(x) = s_1.0^{-1} * (g(x) + 1.0 * h(x))`
  - `pi_3(x) = s_2.0^{-1} * (g(x) + 2.0 * h(x))`
- Here each scale factor `s_*` is chosen deterministically on a dense `[-1,1]` grid so that the variance of the corresponding signal stays approximately fixed across the family.
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `30`

Method design:

- Compared methods: Neural DML and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `30` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE | DML nuisance-error corr | DML oracle-residual corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002082 | 0.004127 | 0.011520 | 0.130356 | 0.079281 | 0.285431 | 0.021949 |
| `pi_2` | 0.002085 | 0.003940 | 0.004141 | 0.117876 | 0.132520 | 0.323403 | 0.029892 |
| `pi_3` | 0.002087 | 0.004422 | 0.008486 | 0.133819 | 0.125359 | 0.376466 | 0.036541 |

Main observations:

- The alignment diagnostics do move in the intended direction. The DML nuisance-error correlation increases from `0.285` to `0.323` to `0.376`, and the oracle-beta residual correlation rises from `0.0219` to `0.0299` to `0.0365`.
- So this family does a better job than `1.5.11` of making the fitted nuisance errors co-move through the shared hard component.
- The treatment nuisance error also rises substantially relative to `1.5.11`, especially from `pi_1` to `pi_2`, although it is not perfectly monotone in the last step: `0.079281 -> 0.132520 -> 0.125359`.
- The final DML beta MSE remains fairly flat overall (`0.004127, 0.003940, 0.004422`) rather than showing a strong monotone degradation. The joint LSE beta estimate is also unstable rather than cleanly ordered across the family.
- The practical takeaway is that explicitly sharing the hard component between `mu` and `pi` does increase nuisance-error alignment, but with unit-variance noise it still only translates into a modest change in the final DML beta accuracy.

Generated figures:

- `examples/plm/figs/1.5/1.5.12_pi_complexity_mse_comparison.png`

## 1.6.1

Experiment `1.6.1`, stored in the simulation artifact `1.6_1`.

### Goal

This experiment moves to a two-dimensional unit-variance design where the outcome regression and the treatment regression share the same smooth first-coordinate signal, but the treatment family gradually increases the amplitude of a high-frequency second-coordinate component. The goal is to see how the DML, paper minimax-debias, oracle, and joint least-squares beta estimators respond as the treatment regression becomes more aligned with that rough second-coordinate component.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Outcome regression:
  - `mu(x) = sin(pi x_1) + 0.33 cos(8 pi x_2)`
- Treatment regression candidates:
  - `pi_1(x) = sin(pi x_1) + (1/3) cos(8 pi x_2)`
  - `pi_2(x) = sin(pi x_1) + (2/3) cos(8 pi x_2)`
  - `pi_3(x) = sin(pi x_1) + cos(8 pi x_2)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `50`

Method design:

- Compared methods: Neural DML, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 1024`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `50` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002032 | 0.009927 | 0.004374 | 0.008302 | 0.346520 | 0.329056 |
| `pi_2` | 0.002043 | 0.059280 | 0.006881 | 0.012164 | 0.357995 | 0.613800 |
| `pi_3` | 0.002055 | 0.012831 | 0.008143 | 0.013810 | 0.422049 | 0.877659 |

Main observations:

- This family still makes the treatment nuisance progressively harder at the 50-trial scale: DML `pi` MSE rises from `0.329056` to `0.613800` to `0.877659`.
- The joint least-squares beta estimate now shows a clean monotone degradation as well: `0.008302 -> 0.012164 -> 0.013810`.
- The plain DML AIPW beta estimate remains the least stable of the three beta procedures. It is worst at the middle setting and remains much larger than oracle throughout: `0.009927 -> 0.059280 -> 0.012831`.
- The paper minimax-debias estimator remains substantially more stable than plain DML AIPW after averaging over 50 trials. Its beta MSE only moves from `0.004374` to `0.006881` to `0.008143`, and it beats the plain DML AIPW beta in all three settings.
- So `1.6.1` is a more informative stress test than many of the earlier `1.5` families. Increasing the rough second-coordinate component in `pi(x)` does make nuisance estimation harder and eventually worsens the simple joint LSE beta fit, while the minimax-debias estimator appears substantially more robust than the baseline DML AIPW estimate in this design.

Generated figures:

- `examples/plm/figs/1.6/1.6.1_pi_complexity_mse_comparison.png`
