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

### Method5: Validation-Selected Neural DML (`PLMValidationSelectedDMLEstimator`)

This method is a standalone neural DML implementation that uses the same residual ReLU nuisance architecture as `PLMDMLEstimator`, but adds model selection through an independent observed validation sample. The validation sample contains only `X`, `T`, and `Y`; oracle nuisance values are not exposed to the estimator.

Training follows the standard DML nuisance stage on the second split of the training sample. Every `validation_check_interval` epochs, the treatment network is evaluated by the observed treatment-prediction loss `mean((T - pi_hat(X))^2)` on the validation sample. The outcome network is evaluated by a profiled observed outcome-prediction loss: for the current `mu_hat`, the method first computes the least-squares coefficient `beta` on the validation sample and then records `mean((beta T + mu_hat(X) - Y)^2)`. The selected `mu` and selected `pi` may therefore come from different epochs. After selection, the method restores the best nuisance networks, recomputes the joint least-squares coefficient on the second split, and computes the final AIPW beta estimate on the first split.

Default hyper-parameters:

- `L = 3`: depth of the residual ReLU nuisance networks.
- `N = 512`: hidden-layer width of the nuisance networks.
- `lambda_mu = 1e-4`: L2 weight regularization level for the outcome network.
- `lambda_pi = 1e-4`: L2 weight regularization level for the treatment network.
- `niter = 200`: number of training epochs used for the nuisance fit.
- `lr = 1e-3`: Adam learning rate.
- `batch_size = 1024`: mini-batch size used during training.
- `validation_n = n // 3` by default in the evaluator: size of the independent validation sample used for model selection.
- `validation_check_interval = 10`: number of epochs between validation-loss evaluations; the final epoch is also evaluated when it is not already on the validation grid.
- `device = "cpu"` by default: computation device for neural-network training.

The diagnostics record the validation epoch grid, the validation loss paths for both nuisances, the selected epochs for `mu` and `pi`, the best validation losses, and whether validation selection was actually used. If no validation sample is supplied, the estimator emits a runtime warning and reduces to final-epoch neural DML behavior.

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

Median metrics over `50` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000969 | 0.004912 | 0.002613 | 0.005557 | 0.284516 | 0.281471 |
| `pi_2` | 0.000961 | 0.006426 | 0.004580 | 0.009714 | 0.302220 | 0.461521 |
| `pi_3` | 0.000946 | 0.006774 | 0.005959 | 0.012644 | 0.341615 | 0.744838 |

Main observations:

- This family still makes the treatment nuisance progressively harder at the 50-trial scale: DML `pi` MSE rises from `0.329056` to `0.613800` to `0.877659`.
- The joint least-squares beta estimate now shows a clean monotone degradation as well: `0.008302 -> 0.012164 -> 0.013810`.
- The plain DML AIPW beta estimate remains the least stable of the three beta procedures. It is worst at the middle setting and remains much larger than oracle throughout: `0.009927 -> 0.059280 -> 0.012831`.
- The paper minimax-debias estimator remains substantially more stable than plain DML AIPW after averaging over 50 trials. Its beta MSE only moves from `0.004374` to `0.006881` to `0.008143`, and it beats the plain DML AIPW beta in all three settings.
- The medians tell a cleaner story than the means. The DML median beta MSE is almost monotone: `0.004912 -> 0.006426 -> 0.006774`. So the non-monotone mean is mainly a tail effect rather than a shift in the typical trial.
- The middle setting `pi_2` has a much heavier right tail than `pi_3`. For DML beta MSE, the `pi_2` maximum is about `1.92` and the 95th percentile is about `0.090`, while for `pi_3` the maximum is only about `0.049` and the 95th percentile is about `0.040`.
- In the saved trial records, the DML beta error is also much more tightly tied to treatment-nuisance error in the `pi_2` case than in the other two settings: the empirical correlation between DML beta MSE and DML `pi` MSE is about `0.92` for `pi_2`, versus about `0.26` for `pi_1` and `0.19` for `pi_3`.
- So the current evidence suggests that `pi_2` is not harder in a typical-case sense than `pi_3`; instead, it creates occasional catastrophic DML failures that inflate the mean. We did not save the AIPW denominator itself, so the next diagnostic would be to record that denominator and check whether those `pi_2` outliers coincide with unusually unstable score normalization.
- So `1.6.1` is a more informative stress test than many of the earlier `1.5` families. Increasing the rough second-coordinate component in `pi(x)` does make nuisance estimation harder and eventually worsens the simple joint LSE beta fit, while the minimax-debias estimator appears substantially more robust than the baseline DML AIPW estimate in this design.

Generated figures:

- `examples/plm/figs/1.6/1.6.1_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.1_pi_complexity_median_mse_comparison.png`

## 1.6.2

Experiment `1.6.2`, stored in the simulation artifact `1.6_2`.

### Goal

This experiment keeps the two-dimensional unit-variance setup but changes the signal geometry so that both the outcome regression and the treatment regression depend only on the same first-coordinate sine signal. The outcome signal is deliberately weaker, while the treatment signal amplitude increases across the family. The goal is to see whether increasing the strength of the aligned treatment component worsens the beta estimators and whether the paper minimax-debias estimator remains stable.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Outcome regression:
  - `mu(x) = 0.25 sin(2 pi x_1)`
- Treatment regression candidates:
  - `pi_1(x) = sin(2 pi x_1)`
  - `pi_2(x) = 2 sin(2 pi x_1)`
  - `pi_3(x) = 3 sin(2 pi x_1)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002989 | 0.006250 | 0.002186 | 0.001321 | 0.279357 | 0.293424 |
| `pi_2` | 0.002979 | 0.007137 | 0.002864 | 0.003230 | 0.346451 | 0.281022 |
| `pi_3` | 0.002983 | 0.033253 | 0.003198 | 0.005338 | 0.447414 | 0.573878 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001541 | 0.003786 | 0.001187 | 0.000620 | 0.248620 | 0.272602 |
| `pi_2` | 0.001508 | 0.003181 | 0.001518 | 0.001501 | 0.226542 | 0.285226 |
| `pi_3` | 0.001499 | 0.013955 | 0.001818 | 0.005291 | 0.360612 | 0.344510 |

Main observations:

- The DML AIPW beta error worsens sharply at the largest treatment amplitude. The mean beta MSE changes as `0.006250 -> 0.007137 -> 0.033253`, and the median changes as `0.003786 -> 0.003181 -> 0.013955`.
- The joint least-squares beta error increases monotonically in both mean and median, suggesting that stronger alignment between `mu(x)` and the systematic part of `T` makes direct beta separation harder.
- The treatment-network MSE is not perfectly monotone in the first two settings, but it becomes substantially larger at `pi_3`. This suggests the largest-amplitude treatment signal is the main stress point in this design.
- The paper minimax-debias estimator remains close to oracle and is much more stable than DML AIPW here. Its mean beta MSE is `0.002186 -> 0.002864 -> 0.003198`, and its median beta MSE is `0.001187 -> 0.001518 -> 0.001818`.

Generated figures:

- `examples/plm/figs/1.6/1.6.2_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.2_pi_complexity_median_mse_comparison.png`

## 1.6.3

Experiment `1.6.3`, stored in the simulation artifact `1.6_3`.

### Goal

This experiment keeps the two-dimensional unit-variance setting but changes the frequency convention to the direct signals requested in the latest diagnostic design. The goal is to test whether increasing the amplitude of a shared second-coordinate sine component in the treatment regression worsens DML beta estimation, while comparing against the oracle AIPW estimator and the paper minimax-debias estimator.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Outcome regression:
  - `mu(x) = sin(2 x_1) + 0.25 sin(6 x_2)`
- Treatment regression candidates:
  - `pi_1(x) = sin(2 x_1) + sin(6 x_2)`
  - `pi_2(x) = sin(2 x_1) + 2 sin(6 x_2)`
  - `pi_3(x) = sin(2 x_1) + 3 sin(6 x_2)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.003041 | 0.008215 | 0.003128 | 0.004340 | 0.571027 | 0.229157 |
| `pi_2` | 0.003116 | 0.019229 | 0.005086 | 0.002860 | 0.250123 | 0.349541 |
| `pi_3` | 0.003217 | 0.018763 | 0.005020 | 0.003727 | 0.313485 | 0.296523 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001654 | 0.003181 | 0.001103 | 0.001201 | 0.304145 | 0.230567 |
| `pi_2` | 0.001675 | 0.012911 | 0.002324 | 0.000527 | 0.228291 | 0.244616 |
| `pi_3` | 0.001704 | 0.003607 | 0.003067 | 0.001015 | 0.292745 | 0.283299 |

Main observations:

- The DML AIPW beta error is larger once the second-coordinate treatment component is amplified. The mean beta MSE changes as `0.008215 -> 0.019229 -> 0.018763`.
- The median DML beta MSE is most severe at `pi_2` in this 10-trial run: `0.003181 -> 0.012911 -> 0.003607`. This suggests some finite-sample instability rather than a clean monotone ordering.
- The right tail grows with the amplified treatment component. The DML beta-MSE maxima are approximately `0.027288`, `0.104141`, and `0.126715` for `pi_1`, `pi_2`, and `pi_3`, respectively.
- The DML treatment nuisance MSE is mildly non-monotone at 10 trials: `0.229157 -> 0.349541 -> 0.296523` in mean and `0.230567 -> 0.244616 -> 0.283299` in median.
- The paper minimax-debias estimator remains much closer to oracle than plain DML AIPW in all three configurations. Its mean beta MSE is `0.003128 -> 0.005086 -> 0.005020`, compared with oracle around `0.0030`.
- The joint least-squares beta estimate remains small in this design and does not show monotone degradation at 10 trials. More trials may be useful if we want to separate systematic behavior from stochastic variation.

Generated figures:

- `examples/plm/figs/1.6/1.6.3_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.3_pi_complexity_median_mse_comparison.png`

## 1.6.4

Experiment `1.6.4`, stored in the simulation artifact `1.6_4`.

### Goal

This experiment keeps the two-dimensional unit-variance protocol but puts both the smooth and rough sine components on the same coordinate. The goal is to see whether making the treatment regression share the same `x_1` rough direction as the outcome regression produces stronger DML instability than the previous two-coordinate design.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Outcome regression:
  - `mu(x) = sin(2 x_1) + 0.25 sin(6 x_1)`
- Treatment regression candidates:
  - `pi_1(x) = sin(2 x_1) + sin(6 x_1)`
  - `pi_2(x) = sin(2 x_1) + 2 sin(6 x_1)`
  - `pi_3(x) = sin(2 x_1) + 3 sin(6 x_1)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002955 | 0.021288 | 0.003623 | 0.002337 | 0.316321 | 0.372796 |
| `pi_2` | 0.002934 | 0.007331 | 0.002172 | 0.003414 | 0.269126 | 0.262688 |
| `pi_3` | 0.002925 | 0.006284 | 0.002309 | 0.002580 | 0.267601 | 0.339726 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001587 | 0.004775 | 0.000666 | 0.001470 | 0.303079 | 0.229942 |
| `pi_2` | 0.001536 | 0.008130 | 0.001189 | 0.000967 | 0.270605 | 0.234034 |
| `pi_3` | 0.001487 | 0.003271 | 0.001320 | 0.002476 | 0.187203 | 0.296727 |

Main observations:

- This same-coordinate design does not produce a monotone DML degradation over 10 trials. The mean DML beta MSE changes as `0.021288 -> 0.007331 -> 0.006284`, while the median changes as `0.004775 -> 0.008130 -> 0.003271`.
- The large mean for `pi_1` is driven by a tail event. The maximum DML beta MSE is about `0.157407` for `pi_1`, compared with about `0.017252` for `pi_2` and `0.017696` for `pi_3`.
- Treatment nuisance error is also not monotone in mean: `0.372796 -> 0.262688 -> 0.339726`, although the median increases from `pi_1` to `pi_3` after a nearly flat first step: `0.229942 -> 0.234034 -> 0.296727`.
- The paper minimax-debias estimator again stays close to oracle. Its mean beta MSE is `0.003623 -> 0.002172 -> 0.002309`, while oracle is about `0.0029` in all three configurations.
- Joint least squares remains small and non-monotone. This suggests that merely putting the rough component on the same coordinate is not enough to create a clean amplitude-driven deterioration under the current network and sample-size regime.

Generated figures:

- `examples/plm/figs/1.6/1.6.4_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.4_pi_complexity_median_mse_comparison.png`

## 1.6.5

Experiment `1.6.5`, stored in the simulation artifact `1.6_5`.

### Goal

This experiment isolates the rough shared direction by removing the smooth `sin(2 x_1)` component from both the outcome and treatment regressions. The goal is to test whether a pure shared rough signal gives a cleaner monotone deterioration in DML beta estimation as the treatment amplitude increases.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Outcome regression:
  - `mu(x) = 0.25 sin(6 x_1)`
- Treatment regression candidates:
  - `pi_1(x) = sin(6 x_1)`
  - `pi_2(x) = 2 sin(6 x_1)`
  - `pi_3(x) = 3 sin(6 x_1)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.003002 | 0.006077 | 0.002248 | 0.001422 | 0.220052 | 0.263133 |
| `pi_2` | 0.003006 | 0.007392 | 0.002413 | 0.003513 | 0.259244 | 0.339013 |
| `pi_3` | 0.003023 | 0.014634 | 0.003084 | 0.004268 | 0.306927 | 0.351700 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001537 | 0.002741 | 0.001013 | 0.000803 | 0.220809 | 0.240252 |
| `pi_2` | 0.001487 | 0.003160 | 0.001190 | 0.000658 | 0.226601 | 0.254589 |
| `pi_3` | 0.001457 | 0.004421 | 0.000989 | 0.003315 | 0.301521 | 0.274701 |

Main observations:

- This rough-only design gives a cleaner monotone DML beta pattern than `1.6.4`. The mean DML beta MSE increases as `0.006077 -> 0.007392 -> 0.014634`, and the median increases as `0.002741 -> 0.003160 -> 0.004421`.
- The DML nuisance errors also increase overall as the treatment amplitude grows. Mean `mu` MSE changes as `0.220052 -> 0.259244 -> 0.306927`, while mean `pi` MSE changes as `0.263133 -> 0.339013 -> 0.351700`.
- The high-amplitude setting has the largest DML right tail. The maximum DML beta MSE is about `0.110240` for `pi_3`, compared with `0.029643` for `pi_1` and `0.022185` for `pi_2`.
- The paper minimax-debias estimator remains close to oracle. Its mean beta MSE is `0.002248 -> 0.002413 -> 0.003084`, while oracle stays around `0.0030`.
- Joint least squares also worsens in mean from `pi_1` to `pi_3`, although the median is not monotone between `pi_1` and `pi_2`. Overall, this is the most orderly amplitude-stress design in the recent `1.6` sequence.

Generated figures:

- `examples/plm/figs/1.6/1.6.5_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.5_pi_complexity_median_mse_comparison.png`

## 1.6.6

Experiment `1.6.6`, stored in the simulation artifact `1.6_6`.

### Goal

This experiment uses a shared two-frequency signal with a lower-frequency component and a smaller high-frequency component. The goal is to test whether amplifying a treatment regression that is exactly aligned with the outcome regression, but contains both `sin(5 x_1)` and `sin(20 x_1)`, produces a clear increase in DML beta error.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Shared base signal:
  - `g(x) = 0.25 sin(5 x_1) + 0.125 sin(20 x_1)`
- Outcome regression:
  - `mu(x) = g(x)`
- Treatment regression candidates:
  - `pi_1(x) = 4 * 0.5 * g(x)`
  - `pi_2(x) = 4 * 1.0 * g(x)`
  - `pi_3(x) = 4 * 2.0 * g(x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.003055 | 0.004863 | 0.002738 | 0.001591 | 0.330841 | 0.246151 |
| `pi_2` | 0.003101 | 0.005385 | 0.003045 | 0.002086 | 0.279957 | 0.446045 |
| `pi_3` | 0.003203 | 0.005056 | 0.003251 | 0.002944 | 0.265101 | 0.598324 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001585 | 0.002511 | 0.000987 | 0.000684 | 0.267868 | 0.237757 |
| `pi_2` | 0.001582 | 0.003493 | 0.001611 | 0.001380 | 0.250830 | 0.301065 |
| `pi_3` | 0.001578 | 0.002286 | 0.001916 | 0.001975 | 0.248944 | 0.544050 |

Main observations:

- The treatment nuisance gets substantially harder as the treatment scaling increases. Mean DML `pi` MSE changes as `0.246151 -> 0.446045 -> 0.598324`, and median DML `pi` MSE changes as `0.237757 -> 0.301065 -> 0.544050`.
- The final DML AIPW beta MSE does not increase monotonically over 10 trials. The mean changes as `0.004863 -> 0.005385 -> 0.005056`, while the median changes as `0.002511 -> 0.003493 -> 0.002286`.
- The DML right tail is also similar across the three settings. The maximum DML beta MSE is about `0.019384`, `0.019500`, and `0.017222` for `pi_1`, `pi_2`, and `pi_3`, respectively.
- The joint least-squares beta estimate worsens monotonically in both mean and median, moving from `0.001591` to `0.002944` in mean and from `0.000684` to `0.001975` in median.
- The paper minimax-debias estimator remains close to oracle. Its mean beta MSE is `0.002738 -> 0.003045 -> 0.003251`, while oracle is about `0.0031`.
- Compared with `1.6.5`, adding the lower-frequency component and using this two-frequency base makes the treatment regression harder to learn, but it does not translate into a strong monotone DML AIPW beta deterioration at this 10-trial scale.

Generated figures:

- `examples/plm/figs/1.6/1.6.6_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.6_pi_complexity_median_mse_comparison.png`

## 1.6.7

Experiment `1.6.7`, stored in the simulation artifact `1.6_7`.

### Goal

This experiment uses a two-dimensional shared signal `eta(x)` as both the outcome regression and the treatment-regression direction. The goal is to test whether scaling a treatment regression that is exactly aligned with the full outcome regression creates a stronger instability than the previous partially aligned two-frequency designs.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Shared signal:
  - `eta(x) = sin(x_1) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = 4 * 0.5 * eta(x)`
  - `pi_2(x) = 4 * 1.0 * eta(x)`
  - `pi_3(x) = 4 * 2.0 * eta(x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.003036 | 0.004447 | 0.004072 | 0.011173 | 0.295831 | 0.264498 |
| `pi_2` | 0.003177 | 132.099823 | 9.374499 | 161.501047 | 1330.864284 | 0.281679 |
| `pi_3` | 0.003929 | 330.959098 | 8.891743 | 263.952510 | 6103.634289 | 0.321006 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001660 | 0.002225 | 0.001054 | 0.003631 | 0.260949 | 0.205777 |
| `pi_2` | 0.001745 | 11.038864 | 5.172272 | 95.723028 | 668.882169 | 0.230639 |
| `pi_3` | 0.001963 | 11.298917 | 5.463956 | 232.718997 | 5285.603231 | 0.332364 |

Main observations:

- This design creates a sharp instability once the treatment regression is scaled to `4 eta(x)` or `8 eta(x)`. The mean DML AIPW beta MSE jumps from `0.004447` at `pi_1` to `132.099823` at `pi_2` and `330.959098` at `pi_3`.
- The median shows that this is not only a single outlier. Median DML AIPW beta MSE moves from `0.002225` to `11.038864` and `11.298917`.
- The joint least-squares beta estimate is the most visibly unstable component. Mean joint-LSE beta MSE changes as `0.011173 -> 161.501047 -> 263.952510`, and median joint-LSE beta MSE changes as `0.003631 -> 95.723028 -> 232.718997`.
- The DML outcome nuisance estimate also collapses in the high-scaling regimes. Mean `mu` MSE changes as `0.295831 -> 1330.864284 -> 6103.634289`, while the treatment nuisance MSE remains modest: `0.264498 -> 0.281679 -> 0.321006`.
- The paper minimax-debias estimator is more stable than plain DML AIPW in the high-scaling settings, but it is still far from oracle. Its mean beta MSE is `0.004072 -> 9.374499 -> 8.891743`.
- The oracle AIPW estimator remains stable throughout, with beta MSE around `0.003`. This suggests that the failure is coming from the learned nuisance/joint-LSE stage rather than from the oracle score itself.
- The practical interpretation is that exactly aligning `mu` and the scaled systematic treatment component can make the neural joint least-squares decomposition extremely ill-conditioned. A useful next diagnostic would be to record the DML denominator and the learned joint-LSE beta directly for these trials.

Generated figures:

- `examples/plm/figs/1.6/1.6.7_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.7_pi_complexity_median_mse_comparison.png`

## 1.6.8

Experiment `1.6.8`, stored in the simulation artifact `1.6_8`.

### Goal

This experiment separates the smooth component in the outcome regression from the treatment regression. The outcome regression uses the full signal `eta(x)`, while the treatment regression uses only the residual high-frequency part `eta(x) - sin(x_2)`. The goal is to see whether removing the shared smooth direction from the treatment regression avoids the catastrophic instability observed in `1.6.7`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Shared outcome signal:
  - `eta(x) = sin(x_2) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = 4 * 1 * (eta(x) - sin(x_2))`
  - `pi_2(x) = 4 * 2 * (eta(x) - sin(x_2))`
  - `pi_3(x) = 4 * 3 * (eta(x) - sin(x_2))`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `10`

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

Average metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.003109 | 0.004825 | 0.003093 | 0.002459 | 0.244481 | 0.233958 |
| `pi_2` | 0.003218 | 0.007459 | 0.003192 | 0.003743 | 0.367582 | 0.295433 |
| `pi_3` | 0.003340 | 0.018328 | 0.003526 | 0.002826 | 0.336413 | 0.839971 |

Median metrics over `10` trials:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | Joint LSE beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001581 | 0.002237 | 0.000917 | 0.000909 | 0.231737 | 0.225229 |
| `pi_2` | 0.001636 | 0.001093 | 0.001050 | 0.001308 | 0.332055 | 0.279530 |
| `pi_3` | 0.001695 | 0.008289 | 0.001050 | 0.000668 | 0.280402 | 0.374360 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | DML beta MSE | DML squared bias | DML variance | Minimax beta MSE | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.004825 | 0.000797 | 0.004028 | 0.003093 | 0.000989 | 0.002104 |
| `pi_2` | 0.007459 | 0.003547 | 0.003911 | 0.003192 | 0.001245 | 0.001947 |
| `pi_3` | 0.018328 | 0.000541 | 0.017787 | 0.003526 | 0.001417 | 0.002109 |

Main observations:

- Removing the smooth `sin(x_2)` component from the treatment regression avoids the catastrophic blow-up observed in `1.6.7`. DML AIPW beta MSE remains finite and much smaller across all three settings.
- The mean DML AIPW beta MSE increases with the residual-treatment scaling: `0.004825 -> 0.007459 -> 0.018328`.
- The DML treatment nuisance error also increases, especially at the largest scaling. Mean `pi` MSE changes as `0.233958 -> 0.295433 -> 0.839971`, and median `pi` MSE changes as `0.225229 -> 0.279530 -> 0.374360`.
- The DML right tail grows with scaling. The maximum DML beta MSE is about `0.024080`, `0.034165`, and `0.076813` for `pi_1`, `pi_2`, and `pi_3`, respectively.
- The paper minimax-debias estimator stays close to oracle throughout. Its mean beta MSE is `0.003093 -> 0.003192 -> 0.003526`, while oracle is `0.003109 -> 0.003218 -> 0.003340`.
- The bias-variance decomposition shows that the large DML MSE at `pi_3` is mainly variance-driven: DML variance is `0.017787`, while its squared bias is only `0.000541`. The minimax-debias estimator keeps variance close to `0.002` across all three treatment scalings.
- Joint least squares does not show the same monotone deterioration as DML AIPW here. This suggests that the main stress in `1.6.8` comes through the AIPW nuisance stage rather than through an obviously exploding joint-LSE beta.

Generated figures:

- `examples/plm/figs/1.6/1.6.8_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.8_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.6/1.6.8_pi_complexity_beta_variance.png`

## 1.6.9

Experiment `1.6.9`, stored in the simulation artifact `1.6_9`.

### Goal

This experiment adds the smooth component `sin(x_2)` back into the treatment regression while keeping the high-frequency residual scaling from `1.6.8`. The target coefficient is sampled from a balanced three-point support. The goal is to compare DML AIPW and the paper minimax-debias estimator when the treatment regression shares the smooth outcome component but becomes progressively more dominated by the high-frequency residual.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Shared signal:
  - `eta(x) = sin(x_2) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = sin(x_2) + 4 * 1 * (eta(x) - sin(x_2))`
  - `pi_2(x) = sin(x_2) + 4 * 2 * (eta(x) - sin(x_2))`
  - `pi_3(x) = sin(x_2) + 4 * 3 * (eta(x) - sin(x_2))`
- Trial-level target coefficient: `beta in {-0.5, 0, 0.5}`, balanced by trial seed with `20` trials per beta value for each treatment-regression candidate
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `60` per treatment-regression candidate

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

Average metrics over `60` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002228 | 0.006222 | 0.001981 | 0.270178 | 0.363978 |
| `pi_2` | 0.002267 | 0.004747 | 0.002461 | 0.297006 | 0.410299 |
| `pi_3` | 0.002323 | 0.011104 | 0.003011 | 0.281403 | 0.503328 |

Grouped bias-variance decomposition over the balanced beta support:

| pi family | DML mean squared bias | DML mean variance | Minimax mean squared bias | Minimax mean variance |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000914 | 0.005308 | 0.000455 | 0.001526 |
| `pi_2` | 0.000524 | 0.004223 | 0.001254 | 0.001207 |
| `pi_3` | 0.000928 | 0.010177 | 0.001999 | 0.001012 |

Main observations:

- The DML treatment nuisance error increases clearly with treatment-regression complexity: mean `pi` MSE changes as `0.363978 -> 0.410299 -> 0.503328`.
- The DML AIPW beta MSE is not monotone between `pi_1` and `pi_2`, but it becomes substantially larger at `pi_3`: `0.006222 -> 0.004747 -> 0.011104`.
- The increase in DML beta MSE at `pi_3` is mainly variance-driven. Its grouped variance increases to `0.010177`, while grouped squared bias remains below `0.001`.
- The paper minimax-debias estimator has lower beta MSE than DML in all three settings. Its MSE increases moderately as the high-frequency residual receives more weight: `0.001981 -> 0.002461 -> 0.003011`.
- The minimax-debias decomposition shows the opposite bias-variance movement: grouped squared bias increases with the treatment residual scaling, while grouped variance decreases slightly.
- Oracle AIPW stays stable around `0.0023`, indicating that the extra beta-estimation difficulty is coming from nuisance estimation and debiasing rather than from the oracle score itself.

Generated figures:

- `examples/plm/figs/1.6/1.6.9_pi_complexity_requested_mse.png`
- `examples/plm/figs/1.6/1.6.9_beta_grouped_bias_variance_hist.png`

## 1.6.10

Experiment `1.6.10`, stored in the simulation artifact `1.6_10`.

### Goal

This experiment returns to the residual-only treatment regression from `1.6.8`, but uses the balanced three-point beta support from `1.6.9`. The goal is to compare the residual-only design against the smooth-plus-residual design under the same beta grouping used for the requested grouped bias-variance decomposition.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Shared signal:
  - `eta(x) = sin(x_2) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = 4 * 1 * (eta(x) - sin(x_2))`
  - `pi_2(x) = 4 * 2 * (eta(x) - sin(x_2))`
  - `pi_3(x) = 4 * 3 * (eta(x) - sin(x_2))`
- Trial-level target coefficient: `beta in {-0.5, 0, 0.5}`, balanced by trial seed with `20` trials per beta value for each treatment-regression candidate
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 1024`
- Test sample size: `n_test = 10000`
- Number of trials: `60` per treatment-regression candidate

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

Average metrics over `60` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.002239 | 0.005912 | 0.001948 | 0.301685 | 0.315998 |
| `pi_2` | 0.002280 | 0.010008 | 0.002953 | 0.294406 | 0.379556 |
| `pi_3` | 0.002339 | 0.011800 | 0.003510 | 0.311399 | 0.560491 |

Grouped bias-variance decomposition over the balanced beta support:

| pi family | DML mean squared bias | DML mean variance | Minimax mean squared bias | Minimax mean variance |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000400 | 0.005512 | 0.000324 | 0.001624 |
| `pi_2` | 0.000438 | 0.009569 | 0.001767 | 0.001186 |
| `pi_3` | 0.002378 | 0.009422 | 0.002439 | 0.001071 |

Main observations:

- The DML treatment nuisance error increases monotonically with the residual-treatment scaling: mean `pi` MSE changes as `0.315998 -> 0.379556 -> 0.560491`.
- The DML AIPW beta MSE also increases across the same settings: `0.005912 -> 0.010008 -> 0.011800`.
- The DML increase from `pi_1` to `pi_2` is mainly variance-driven, with grouped variance moving from `0.005512` to `0.009569`. At `pi_3`, grouped squared bias also rises to `0.002378`.
- The paper minimax-debias estimator has smaller beta MSE than DML throughout, but its mean squared bias increases with treatment complexity: `0.000324 -> 0.001767 -> 0.002439`.
- Oracle AIPW remains stable around `0.0023`, again suggesting that the degradation is due to learned nuisance/debiasing behavior rather than the oracle score.

Generated figures:

- `examples/plm/figs/1.6/1.6.10_pi_complexity_requested_mse.png`
- `examples/plm/figs/1.6/1.6.10_beta_grouped_bias_variance_hist.png`

## 1.6.11

Experiment `1.6.11`, stored in the simulation artifact `1.6_11`.

### Goal

This experiment increases the covariate dimension to `d = 10` and doubles the training sample size to `n = 2048`. The goal is to test whether the residual-only treatment design remains stable when the outcome and treatment nuisance functions include a weak high-frequency tail over eight additional coordinates.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 10`
- Shared signal:
  - `eta(x) = sin(x_1) + 0.25 sin(4 x_2) + 0.05 sin(11 x_2) + sum_{j=3}^{10} 0.005 sin(30 x_j)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = 4 * 1 * (eta(x) - sin(x_1))`
  - `pi_2(x) = 4 * 2 * (eta(x) - sin(x_1))`
  - `pi_3(x) = 4 * 3 * (eta(x) - sin(x_1))`
- Trial-level target coefficient: `beta in {-0.5, 0, 0.5}`, balanced by trial seed with `20` trials per beta value for each treatment-regression candidate
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `60` per treatment-regression candidate

Method design:

- Compared methods: Neural DML, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `60` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000949 | 0.002390 | 0.068152 | 0.426297 | 0.682598 |
| `pi_2` | 0.000959 | 0.002732 | 0.095625 | 0.430358 | 1.243759 |
| `pi_3` | 0.000972 | 0.002346 | 0.118819 | 0.434224 | 1.835003 |

Grouped bias-variance decomposition over the balanced beta support:

| pi family | DML mean squared bias | DML mean variance | Minimax mean squared bias | Minimax mean variance |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000938 | 0.001452 | 0.067998 | 0.000155 |
| `pi_2` | 0.001960 | 0.000772 | 0.095586 | 0.000039 |
| `pi_3` | 0.001881 | 0.000465 | 0.118791 | 0.000028 |

Main observations:

- The DML treatment nuisance error increases strongly with residual-treatment scaling: mean `pi` MSE changes as `0.682598 -> 1.243759 -> 1.835003`.
- DML AIPW beta MSE stays small and relatively flat in this run: `0.002390 -> 0.002732 -> 0.002346`, despite the increasing treatment nuisance error.
- The paper minimax-debias estimator performs poorly in this high-dimensional setting under the current hyper-parameters. Its beta MSE increases as `0.068152 -> 0.095625 -> 0.118819`.
- The minimax-debias error is almost entirely squared bias rather than variance. Grouped squared bias is `0.067998 -> 0.095586 -> 0.118791`, while grouped variance is below `0.0002`.
- Oracle AIPW beta MSE is lower than in the previous `n = 1024` runs, around `0.001`, as expected from the larger sample size.
- This run suggests that the current minimax-debias hyper-parameters or optimization setup may not transfer cleanly to the `d = 10`, `n = 2048` nuisance family, even though DML remains stable.

Generated figures:

- `examples/plm/figs/1.6/1.6.11_pi_complexity_requested_mse.png`
- `examples/plm/figs/1.6/1.6.11_beta_grouped_bias_variance_hist.png`

## 1.6.12

Experiment `1.6.12`, stored in the simulation artifact `1.6_12`.

### Goal

This experiment returns to a lower-dimensional setting with `d = 3`, keeps the larger training size `n = 2048`, and uses a smooth-plus-residual treatment regression. The target coefficient is sampled continuously from `[-0.5, 0.5]`, so the bias-variance decomposition is computed over trial-level beta estimation errors rather than grouped by beta value.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 3`
- Shared signal:
  - `eta(x) = sin(x_2) + 0.25 sin(4 x_2) + 0.05 sin(7 x_2) + 0.005 sin(23 x_2)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = sin(x_2) + 4 * 1 * (eta(x) - sin(x_2))`
  - `pi_2(x) = sin(x_2) + 4 * 2 * (eta(x) - sin(x_2))`
  - `pi_3(x) = sin(x_2) + 4 * 3 * (eta(x) - sin(x_2))`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: Neural DML, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | DML AIPW beta MSE | Minimax debias beta MSE | DML mu MSE | DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001227 | 0.001871 | 0.005190 | 0.339131 | 0.259277 |
| `pi_2` | 0.001252 | 0.001804 | 0.006103 | 0.372115 | 0.734264 |
| `pi_3` | 0.001278 | 0.696105 | 0.006374 | 0.349373 | 0.402146 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | DML squared bias | DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000219 | 0.001652 | 0.000040 | 0.005149 |
| `pi_2` | 0.000007 | 0.001797 | 0.000003 | 0.006099 |
| `pi_3` | 0.086055 | 0.610051 | 0.000002 | 0.006372 |

Main observations:

- The DML AIPW beta estimate is stable for `pi_1` and `pi_2`, with beta MSE around `0.0018`.
- At `pi_3`, DML AIPW has a large failure over the 10-trial run: mean beta MSE is `0.696105`.
- The `pi_3` DML failure is mostly variance-driven. The largest trial has beta truth `0.373429`, DML beta estimate `-2.233720`, and squared error `6.797230`.
- The paper minimax-debias estimator is much more stable in this setting, with beta MSE around `0.005` to `0.0064` across all three treatment regressions.
- The DML treatment nuisance MSE is not monotone over only 10 trials: `0.259277 -> 0.734264 -> 0.402146`. This run should be treated as a first diagnostic rather than a final trend estimate.

Generated figures:

- `examples/plm/figs/1.6/1.6.12_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.12_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.6/1.6.12_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.6/1.6.12_unified_mse_mean_curve.png`

## 1.6.13

Experiment `1.6.13`, stored in the simulation artifact `1.6_13`.

### Goal

This experiment reruns the two-dimensional `1.6.13` design with `n = 2048` and `batch_size = 2048`. Compared with `1.6.12`, the treatment regression changes the amplitude of the `sin(5x_2)` component directly while keeping the `sin(20x_2)` component fixed. The previous `n = 1024` artifact was archived as `simulation_results/plm/1.6_13_n1024_archive.json`, and the standard-DML `n = 2048` artifact was archived as `simulation_results/plm/1.6_13_standard_dml_archive.json`. The active `1.6_13` artifact now uses validation-selected neural DML.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Shared signal:
  - `eta(x) = sin(x_2) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`
- Outcome regression:
  - `mu(x) = eta(x)`
- Treatment regression candidates:
  - `pi_1(x) = sin(x_2) + 1 * sin(5 x_2) + 0.05 sin(20 x_2)`
  - `pi_2(x) = sin(x_2) + 2 * sin(5 x_2) + 0.05 sin(20 x_2)`
  - `pi_3(x) = sin(x_2) + 3 * sin(5 x_2) + 0.05 sin(20 x_2)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `10` trials for each treatment-regression candidate. The Neural DML nuisance MSEs below are computed from the nuisance networks restored by validation selection.

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Minimax debias beta MSE | Selected DML mu MSE | Selected DML pi MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000734 | 0.000991 | 0.000686 | 0.039362 | 0.047925 |
| `pi_2` | 0.000730 | 0.001065 | 0.000629 | 0.047026 | 0.056926 |
| `pi_3` | 0.000726 | 0.001039 | 0.000721 | 0.056913 | 0.068254 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Validation-selected DML squared bias | Validation-selected DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000087 | 0.000904 | 0.000001 | 0.000685 |
| `pi_2` | 0.000086 | 0.000979 | 0.000006 | 0.000623 |
| `pi_3` | 0.000183 | 0.000855 | 0.000035 | 0.000686 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `pi_1` | 682 | `10, 20, ..., 200` | 37.0 | 40.0 |
| `pi_2` | 682 | `10, 20, ..., 200` | 41.0 | 57.0 |
| `pi_3` | 682 | `10, 20, ..., 200` | 40.0 | 61.0 |

Main observations:

- Validation selection substantially lowers the Neural DML nuisance errors relative to the archived standard-DML `n = 2048` run, and the DML beta MSE falls from the earlier range of roughly `0.0025` to `0.0037` down to roughly `0.0010`.
- Validation-selected DML remains above the oracle benchmark and the minimax-debias estimator, but the gap is now much smaller.
- The minimax-debias estimator remains close to oracle in all three settings, with beta MSE around `0.0006` to `0.0007`.
- The selected DML nuisance errors increase with the treatment-regression amplitude in this run: both `mu` and `pi` MSE are smallest for `pi_1` and largest for `pi_3`.
- The unified mean-curve figure now shows solid mean beta-MSE curves for validation-selected DML and minimax debias, a dashed oracle reference, and dotted nuisance-MSE curves from the selected DML nuisances.

Generated figures:

- `examples/plm/figs/1.6/1.6.13_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.13_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.6/1.6.13_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.6/1.6.13_unified_mse_mean_curve.png`

### Nuisance-learning path diagnostic

Diagnostic artifact: `simulation_results/plm/1.6_13_tracking.json`.

This follow-up tracks the oracle nuisance MSE paths for the neural DML nuisance learners in the same `1.6.13` DGP. The estimator records both the in-sample path on `D2` and the out-of-sample path on an independent validation sample of size `2048`, but the current summary figure focuses on the validation path only. The figure averages the paths over `10` trials per treatment-regression candidate and overlays the three treatment-regression settings in one axes. Red curves represent `pi`, blue curves represent `mu`, and the alpha level encodes the treatment amplitude, with alpha equal to `1` for the largest `k`.

Average oracle nuisance MSE at epoch `0` and epoch `200`:

| pi family | Source | Mu MSE @ 0 | Mu MSE @ 200 | Pi MSE @ 0 | Pi MSE @ 200 |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi_1` | D2 | 0.326047 | 0.156994 | 0.709777 | 0.200147 |
| `pi_1` | validation | 0.326705 | 0.164995 | 0.720087 | 0.204579 |
| `pi_2` | D2 | 0.326047 | 0.242155 | 2.125450 | 0.157932 |
| `pi_2` | validation | 0.326705 | 0.245028 | 2.148390 | 0.163688 |
| `pi_3` | D2 | 0.326047 | 0.180692 | 4.584240 | 0.161562 |
| `pi_3` | validation | 0.326705 | 0.181305 | 4.620900 | 0.171805 |

Main observations:

- The D2 and validation curves are close across the full training path, so this diagnostic does not show a large in-sample/out-of-sample tracking gap.
- The initial `pi` MSE increases sharply with the amplitude of the treatment regression, but by epoch `200` all three treatment learners finish in the narrower range `0.158` to `0.205` on D2 and `0.164` to `0.205` on validation.
- The final `mu` path is largest for `pi_2`, matching the earlier non-monotone nuisance behavior observed in the standard `1.6.13` summary.

Generated diagnostic figures:

- `examples/plm/figs/1.6/1.6.13_nuisance_validation_overlay_paths.png`
- `examples/plm/figs/1.6/1.6.13_nuisance_in_out_average_paths.png`

## 1.6.14

Experiment `1.6.14`, stored in the simulation artifact `1.6_14`.

### Goal

This experiment keeps `n = 2048`, `batch_size = 2048`, and `beta ~ Unif[-0.5, 0.5]`, but switches to a three-dimensional design with a shared `cos(x_1)` component and treatment-specific scaling of the `sin(6x_2)` component. The goal is to test whether a treatment regression that changes its middle-frequency component produces a clearer degradation pattern for DML AIPW and the paper minimax-debias estimator. The earlier standard-DML-only artifact was archived as `simulation_results/plm/1.6_14_standard_dml_archive.json`; the active artifact now compares validation-selected DML against the original DML estimator without model selection.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 3`
- Outcome regression: `mu(x) = cos(x_1) + 0.25 sin(6 x_2) + 0.05 sin(19 x_3)`
- Treatment regression candidates:
  - `pi_1(x) = cos(x_1) + 0.5 sin(6 x_2) + 0.05 sin(19 x_3)`
  - `pi_2(x) = cos(x_1) + sin(6 x_2) + 0.05 sin(19 x_3)`
  - `pi_4(x) = cos(x_1) + 2 sin(6 x_2) + 0.05 sin(19 x_3)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, standard Neural DML without model selection, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for validation-selected Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for validation-selected Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Standard DML beta MSE | Minimax debias beta MSE |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.001218 | 0.001000 | 0.003845 | 0.003100 |
| `pi_2` | 0.001228 | 0.001015 | 0.002892 | 0.003359 |
| `pi_4` | 0.001247 | 0.001217 | 0.006117 | 0.004168 |

Nuisance MSEs for the two Neural DML variants:

| pi family | Selected DML mu MSE | Selected DML pi MSE | Standard DML mu MSE | Standard DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.094339 | 0.088853 | 0.358750 | 0.423948 |
| `pi_2` | 0.103533 | 0.128702 | 0.283530 | 0.874364 |
| `pi_4` | 0.118356 | 0.203361 | 0.339523 | 0.353218 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Selected DML squared bias | Selected DML variance | Standard DML squared bias | Standard DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi_1` | 0.000206 | 0.000794 | 0.000229 | 0.003616 | 0.000143 | 0.002957 |
| `pi_2` | 0.000084 | 0.000931 | 0.000511 | 0.002382 | 0.000141 | 0.003218 |
| `pi_4` | 0.000149 | 0.001068 | 0.000095 | 0.006022 | 0.000091 | 0.004077 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `pi_1` | 682 | `10, 20, ..., 200` | 41.0 | 34.0 |
| `pi_2` | 682 | `10, 20, ..., 200` | 39.0 | 46.0 |
| `pi_4` | 682 | `10, 20, ..., 200` | 38.0 | 56.0 |

Main observations:

- Validation-selected DML substantially improves both nuisance MSEs and beta MSE relative to standard DML without model selection in all three treatment settings.
- The validation-selected DML beta MSE is very close to the oracle AIPW benchmark in this run, while standard DML remains much more variable.
- Standard DML still shows the largest beta MSE at `pi_4`, while validation-selected DML is comparatively stable across the three treatment-regression candidates.
- For standard DML and minimax-debias, variance dominates beta error. Validation-selected DML also remains variance-dominated, but the variance is much smaller than the standard DML variance.

Generated figures:

- `examples/plm/figs/1.6/1.6.14_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.6/1.6.14_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.6/1.6.14_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.6/1.6.14_unified_mse_mean_curve.png`

### Nuisance-learning path diagnostic

Diagnostic artifact: `simulation_results/plm/1.6_14_tracking.json`.

This follow-up uses the same validation-only overlay convention as the `1.6.13` diagnostic. The neural DML tracking estimator records oracle nuisance MSE paths on `D2` and on an independent validation sample of size `2048`; the summary figure uses the validation paths only. Red curves represent `pi`, blue curves represent `mu`, and the alpha level encodes the treatment amplitude, with alpha equal to `1` for the largest `k`.

Average validation oracle nuisance MSE at epoch `0` and epoch `200`:

| pi family | Mu MSE @ 0 | Mu MSE @ 200 | Pi MSE @ 0 | Pi MSE @ 200 |
| --- | ---: | ---: | ---: | ---: |
| `pi_1` | 0.822581 | 0.361566 | 0.820306 | 0.425776 |
| `pi_2` | 0.822581 | 0.283758 | 1.197750 | 0.882550 |
| `pi_4` | 0.822581 | 0.340963 | 2.736380 | 0.357894 |

Main observations:

- The initial validation `pi` MSE increases sharply with treatment amplitude, as expected from the larger treatment regression signal.
- By epoch `200`, the `pi_2` treatment learner remains the hardest in validation MSE, matching the non-monotone nuisance behavior observed in the standard `1.6.14` summary.
- The validation `mu` MSE remains highest for `pi_1` and lowest for `pi_2`, so the outcome learner's path is also not monotone in the treatment amplitude.

Generated diagnostic figures:

- `examples/plm/figs/1.6/1.6.14_nuisance_validation_overlay_paths.png`
- `examples/plm/figs/1.6/1.6.14_nuisance_in_out_average_paths.png`

## 1.7.1

Experiment `1.7.1`, stored in the simulation artifact `1.7_1`.

### Goal

This experiment introduces a shared-random projected Fourier family in ambient dimension `d = 5`. The goal is to compare the PLM estimators when the outcome regression is fixed at `mu(x) = g_1(q^\top x)` and the treatment regression varies across the roughness levels `g_r(q^\top x)` for `r in {1, 2, 4, 8}`. This allows us to keep the same random coefficient bank and projection direction while increasing the effective high-frequency content of `pi`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 5`
- Shared one-dimensional projection:
  - `q` is drawn once from a standard Gaussian vector and normalized to unit Euclidean norm, so that `Var(q^\top X) = Var(Unif[-1,1])` when `X ~ Unif([-1,1]^5)`
  - realized `q = (0.052520, 0.579907, -0.767070, -0.206127, -0.173391)`
- Shared random Fourier coefficient bank:
  - `a_k, b_k ~ Unif[0, 2]`, sampled once with seed `20260426`, for `k = 1, ..., 34`
- Fourier family:
  - `g_r(z) = {sum_{k=1}^{34} [a_k k^{-1/r} sin(pi k z) + b_k k^{-1/r} cos(pi k z)]} / sqrt(sum_{k=1}^{34} (a_k^2 + b_k^2) k^{-2/r})`
- Outcome regression:
  - `mu(x) = g_1(q^\top x)`
- Treatment regression candidates:
  - `pi_1(x) = g_1(q^\top x)`
  - `pi_2(x) = g_2(q^\top x)`
  - `pi_4(x) = g_4(q^\top x)`
  - `pi_8(x) = g_8(q^\top x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, standard Neural DML without model selection, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for validation-selected Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for validation-selected Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Ground-truth family preview:

- `examples/plm/figs/1.7/1.7.1_g_r_preview.png`

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Standard DML beta MSE | Minimax debias beta MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.001062 | 0.041062 | 0.015200 | 0.046676 |
| `r = 2` | 0.001062 | 0.045298 | 0.025499 | 0.044750 |
| `r = 4` | 0.001060 | 0.036063 | 0.023493 | 0.043782 |
| `r = 8` | 0.001059 | 0.030993 | 0.022647 | 0.042193 |

Nuisance MSEs for the two Neural DML variants:

| pi family | Selected DML mu MSE | Selected DML pi MSE | Standard DML mu MSE | Standard DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.425288 | 0.375127 | 0.902073 | 0.863513 |
| `r = 2` | 0.393755 | 0.600040 | 0.924398 | 1.314070 |
| `r = 4` | 0.376579 | 0.645104 | 0.861623 | 1.479010 |
| `r = 8` | 0.371562 | 0.654106 | 0.841569 | 1.527620 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Selected DML squared bias | Selected DML variance | Standard DML squared bias | Standard DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r = 1` | 0.039437 | 0.001625 | 0.011270 | 0.003930 | 0.003901 | 0.042775 |
| `r = 2` | 0.044110 | 0.001188 | 0.023149 | 0.002351 | 0.006656 | 0.038095 |
| `r = 4` | 0.034948 | 0.001115 | 0.019957 | 0.003536 | 0.007190 | 0.036591 |
| `r = 8` | 0.029877 | 0.001116 | 0.019731 | 0.002915 | 0.006481 | 0.035712 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `r = 1` | 682 | `10, 20, ..., 200` | 24.0 | 28.0 |
| `r = 2` | 682 | `10, 20, ..., 200` | 26.0 | 18.0 |
| `r = 4` | 682 | `10, 20, ..., 200` | 29.0 | 12.0 |
| `r = 8` | 682 | `10, 20, ..., 200` | 29.0 | 12.0 |

Main observations:

- This projected Fourier family is substantially harder than the recent `1.6` families: oracle AIPW remains around `10^{-3}`, while all learned estimators remain between roughly `1.5e-2` and `4.7e-2`.
- Validation-selected DML improves nuisance MSE sharply relative to standard DML in every `r` setting, but its beta MSE is worse than standard DML because its error is dominated by large squared bias rather than variance.
- Standard DML also remains strongly biased, but it keeps a smaller bias term than validation-selected DML in this experiment, which is why it outperforms the validation-selected variant on beta MSE despite worse nuisance MSE.
- The minimax-debias estimator is the most unstable learned method in this family: it has the largest variance across all four `r` settings and beta MSE above `0.04` throughout.
- As `r` increases, the selected DML beta MSE decreases somewhat, while the selected `pi` nuisance MSE increases and then plateaus. This suggests the treatment-regression roughness alone is not the only driver of beta error in this projected shared-coefficient family.

Generated figures:

- `examples/plm/figs/1.7/1.7.1_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.7/1.7.1_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.7/1.7.1_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.7/1.7.1_unified_mse_mean_curve.png`

## 1.7.2

Experiment `1.7.2`, stored in the simulation artifact `1.7_2`.

### Goal

This experiment revisits the projected one-dimensional Fourier family from `1.7.1`, but removes coefficient randomness by fixing `a_k = b_k = 1` and truncating the harmonic range to `k = 1, ..., 10`. The goal is to see whether the same estimators behave more regularly when the treatment regression varies only through the decay exponent `r in {1, 2, 4, 8}` rather than through a shared random coefficient bank.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 5`
- Shared one-dimensional projection:
  - `q` is fixed to the same normalized Gaussian draw used in `1.7.1`, so that `Var(q^\top X) = Var(Unif[-1,1])` when `X ~ Unif([-1,1]^5)`
  - realized `q = (0.052520, 0.579907, -0.767070, -0.206127, -0.173391)`
- Fixed-coefficient Fourier family:
  - `g_r(z) = {sum_{k=1}^{10} k^{-1/r} [sin(pi k z) + cos(pi k z)]} / sqrt(sum_{k=1}^{10} 2 k^{-2/r})`
- Outcome regression:
  - `mu(x) = g_1(q^\top x)`
- Treatment regression candidates:
  - `pi_1(x) = g_1(q^\top x)`
  - `pi_2(x) = g_2(q^\top x)`
  - `pi_4(x) = g_4(q^\top x)`
  - `pi_8(x) = g_8(q^\top x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, standard Neural DML without model selection, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for validation-selected Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for validation-selected Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

Ground-truth family preview:

- `examples/plm/figs/1.7/1.7.2_g_r_preview.png`

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Standard DML beta MSE | Minimax debias beta MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.001060 | 0.029436 | 0.009773 | 0.043522 |
| `r = 2` | 0.001063 | 0.038655 | 0.017556 | 0.043983 |
| `r = 4` | 0.001064 | 0.037251 | 0.019662 | 0.042629 |
| `r = 8` | 0.001064 | 0.035078 | 0.020496 | 0.042863 |

Nuisance MSEs for the two Neural DML variants:

| pi family | Selected DML mu MSE | Selected DML pi MSE | Standard DML mu MSE | Standard DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.378373 | 0.317793 | 0.833861 | 0.788790 |
| `r = 2` | 0.358136 | 0.521586 | 0.808260 | 1.077288 |
| `r = 4` | 0.340858 | 0.609282 | 0.780802 | 1.256180 |
| `r = 8` | 0.333525 | 0.636665 | 0.790957 | 1.341330 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Selected DML squared bias | Selected DML variance | Standard DML squared bias | Standard DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r = 1` | 0.027927 | 0.001510 | 0.005804 | 0.003969 | 0.002480 | 0.041042 |
| `r = 2` | 0.036394 | 0.002260 | 0.013750 | 0.003806 | 0.004740 | 0.039243 |
| `r = 4` | 0.035835 | 0.001415 | 0.016769 | 0.002892 | 0.005661 | 0.036969 |
| `r = 8` | 0.033815 | 0.001263 | 0.017617 | 0.002879 | 0.005451 | 0.037412 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `r = 1` | 682 | `10, 20, ..., 200` | 26.0 | 28.0 |
| `r = 2` | 682 | `10, 20, ..., 200` | 26.0 | 23.0 |
| `r = 4` | 682 | `10, 20, ..., 200` | 28.0 | 18.0 |
| `r = 8` | 682 | `10, 20, ..., 200` | 28.0 | 17.0 |

Main observations:

- Relative to `1.7.1`, fixing the coefficients to `a_k = b_k = 1` and truncating to `k = 1, ..., 10` makes the family materially easier for the learned estimators, especially for standard DML.
- Standard DML outperforms validation-selected DML on beta MSE throughout this experiment, even though validation selection sharply improves both `mu` and `pi` nuisance MSE. The same qualitative pattern from `1.7.1` remains: better nuisance prediction does not translate into better beta estimation because the selected model is more biased.
- As `r` increases, both DML variants see larger `pi` nuisance MSE, which is consistent with the slower decay making the treatment regression rougher. Standard DML beta MSE also increases with `r`, while validation-selected DML is worst at `r = 2` and then improves slightly.
- The minimax-debias estimator is again dominated by variance rather than bias: its squared bias remains around `0.0025` to `0.0057`, but its variance stays near `0.037` to `0.041`, which keeps its total beta MSE above `0.042` across all four treatment-regression choices.

Generated figures:

- `examples/plm/figs/1.7/1.7.2_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.7/1.7.2_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.7/1.7.2_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.7/1.7.2_unified_mse_mean_curve.png`

## 1.7.3

Experiment `1.7.3`, stored in the simulation artifact `1.7_3`.

### Goal

This experiment keeps the fixed-coefficient family from `1.7.2`, but removes the random projection and instead places the signal directly on the first covariate. The goal is to compare the same estimators when `mu(x) = g_1(x_1)` and `pi_r(x) = g_r(x_1)` in ambient dimension `d = 3`, so that the target and treatment regressions are truly one-dimensional even though the learner still receives a three-dimensional input vector.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 3`
- Fixed-coefficient Fourier family:
  - `g_r(z) = {sum_{k=1}^{10} k^{-1/r} [sin(pi k z) + cos(pi k z)]} / sqrt(sum_{k=1}^{10} 2 k^{-2/r})`
- Outcome regression:
  - `mu(x) = g_1(x_1)`
- Treatment regression candidates:
  - `pi_1(x) = g_1(x_1)`
  - `pi_2(x) = g_2(x_1)`
  - `pi_4(x) = g_4(x_1)`
  - `pi_8(x) = g_8(x_1)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, standard Neural DML without model selection, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for validation-selected Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for validation-selected Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Standard DML beta MSE | Minimax debias beta MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.001224 | 0.005208 | 0.002191 | 0.002925 |
| `r = 2` | 0.001227 | 0.009884 | 0.003336 | 0.004298 |
| `r = 4` | 0.001226 | 0.013263 | 0.008230 | 0.005091 |
| `r = 8` | 0.001226 | 0.010551 | 0.005508 | 0.004498 |

Nuisance MSEs for the two Neural DML variants:

| pi family | Selected DML mu MSE | Selected DML pi MSE | Standard DML mu MSE | Standard DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.167523 | 0.156208 | 0.358922 | 0.396793 |
| `r = 2` | 0.168358 | 0.292011 | 0.268619 | 0.449203 |
| `r = 4` | 0.160984 | 0.391370 | 0.701438 | 0.516370 |
| `r = 8` | 0.159181 | 0.423725 | 0.464406 | 0.602110 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Selected DML squared bias | Selected DML variance | Standard DML squared bias | Standard DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r = 1` | 0.004148 | 0.001060 | 0.000248 | 0.001943 | 0.001322 | 0.001604 |
| `r = 2` | 0.009028 | 0.000856 | 0.002042 | 0.001294 | 0.003029 | 0.001269 |
| `r = 4` | 0.012294 | 0.000968 | 0.002502 | 0.005727 | 0.004323 | 0.000768 |
| `r = 8` | 0.009846 | 0.000704 | 0.004514 | 0.000994 | 0.004060 | 0.000438 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `r = 1` | 682 | `10, 20, ..., 200` | 59.0 | 43.0 |
| `r = 2` | 682 | `10, 20, ..., 200` | 42.0 | 70.0 |
| `r = 4` | 682 | `10, 20, ..., 200` | 42.0 | 65.0 |
| `r = 8` | 682 | `10, 20, ..., 200` | 43.0 | 116.0 |

Main observations:

- Moving from the projected `d = 5` setting in `1.7.2` to the direct first-coordinate `d = 3` setting in `1.7.3` makes the problem substantially easier for all learned estimators. Both DML variants improve sharply in beta MSE and nuisance MSE.
- Standard DML still achieves the best beta MSE among the learned methods at every `r`, although the gap to validation-selected DML is much smaller than in `1.7.2` for `r = 1` and `r = 2`.
- Validation-selected DML continues to win decisively on nuisance estimation error: its `mu` MSE stays around `0.16` to `0.17`, while the standard DML `mu` MSE ranges from about `0.27` up to `0.70`.
- As `r` increases, the selected `pi` nuisance MSE grows monotonically, and the selected validation epoch for `pi` shifts later, reaching an average of `116` epochs at `r = 8`. This is consistent with rougher treatment regressions requiring longer optimization to reach their best validation fit.
- The minimax-debias estimator is much more competitive here than in `1.7.2`: it outperforms standard DML at `r = 4` and `r = 8`, because its variance remains relatively small in this simpler one-coordinate setting.

Generated figures:

- `examples/plm/figs/1.7/1.7.3_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.7/1.7.3_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.7/1.7.3_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.7/1.7.3_unified_mse_mean_curve.png`

## 1.7.4

Experiment `1.7.4`, stored in the simulation artifact `1.7_4`.

### Goal

This experiment keeps the direct first-coordinate design from `1.7.3`, but changes the Fourier family to the alternating-cosine-sign version with `a_k = 1` and `b_k = (-1)^k`, while also reducing the ambient dimension to `d = 2`. The goal is to see how this sign-flipped family changes the relative difficulty of nuisance estimation and beta estimation across the same roughness grid `r in {1, 2, 4, 8}`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 2`
- Fixed-coefficient Fourier family:
  - `\tilde g_r(z) = {sum_{k=1}^{10} k^{-1/r} [sin(pi k z) + (-1)^k cos(pi k z)]} / sqrt(sum_{k=1}^{10} 2 k^{-2/r})`
- Outcome regression:
  - `mu(x) = \tilde g_1(x_1)`
- Treatment regression candidates:
  - `pi_1(x) = \tilde g_1(x_1)`
  - `pi_2(x) = \tilde g_2(x_1)`
  - `pi_4(x) = \tilde g_4(x_1)`
  - `pi_8(x) = \tilde g_8(x_1)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, standard Neural DML without model selection, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for validation-selected Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for validation-selected Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Standard DML beta MSE | Minimax debias beta MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.000751 | 0.002211 | 0.001344 | 0.000688 |
| `r = 2` | 0.000756 | 0.003061 | 0.003264 | 0.001316 |
| `r = 4` | 0.000759 | 0.002536 | 0.005392 | 0.001461 |
| `r = 8` | 0.000761 | 0.002369 | 0.007442 | 0.001468 |

Nuisance MSEs for the two Neural DML variants:

| pi family | Selected DML mu MSE | Selected DML pi MSE | Standard DML mu MSE | Standard DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.093212 | 0.078679 | 0.182783 | 0.197051 |
| `r = 2` | 0.097175 | 0.159047 | 0.194267 | 0.270246 |
| `r = 4` | 0.097376 | 0.201071 | 0.196013 | 0.360094 |
| `r = 8` | 0.094856 | 0.217752 | 0.234919 | 0.432585 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Selected DML squared bias | Selected DML variance | Standard DML squared bias | Standard DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r = 1` | 0.001155 | 0.001057 | 0.000091 | 0.001253 | 0.000144 | 0.000543 |
| `r = 2` | 0.001332 | 0.001729 | 0.000000 | 0.003264 | 0.000635 | 0.000681 |
| `r = 4` | 0.001053 | 0.001483 | 0.000308 | 0.005084 | 0.000775 | 0.000686 |
| `r = 8` | 0.001232 | 0.001136 | 0.001378 | 0.006063 | 0.000794 | 0.000674 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `r = 1` | 682 | `10, 20, ..., 200` | 62.0 | 64.0 |
| `r = 2` | 682 | `10, 20, ..., 200` | 67.0 | 117.0 |
| `r = 4` | 682 | `10, 20, ..., 200` | 64.0 | 144.0 |
| `r = 8` | 682 | `10, 20, ..., 200` | 65.0 | 168.0 |

Main observations:

- This alternating-cosine-sign `d = 2` family is the easiest of the recent `1.7` variants. All learned estimators improve substantially relative to `1.7.3`, and oracle AIPW remains below `8e-4` throughout.
- The minimax-debias estimator performs best among the learned methods in every `r` setting. Its main advantage here is variance control: its variance stays around `5e-4` to `7e-4`, far below the standard DML variance at larger `r`.
- Validation-selected DML continues to dominate nuisance estimation, with `mu` MSE near `0.095` and `pi` MSE increasing gradually from `0.079` to `0.218` as `r` grows. However, its beta error remains mostly bias-driven, so it does not beat minimax on beta MSE.
- Standard DML becomes increasingly unstable as `r` grows: both `pi` MSE and beta variance rise sharply, and the beta MSE climbs from `0.0013` at `r = 1` to `0.0074` at `r = 8`.
- The selected validation epoch for `pi` shifts markedly later as `r` increases, reaching an average of `168` epochs at `r = 8`. This is consistent with the rougher treatment regressions taking longer to fit well even in this lower-dimensional setting.

Generated figures:

- `examples/plm/figs/1.7/1.7.4_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.7/1.7.4_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.7/1.7.4_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.7/1.7.4_unified_mse_mean_curve.png`

## 1.7.5

Experiment `1.7.5`, stored in the simulation artifact `1.7_5`.

### Goal

This experiment takes the current alternating-sign family from `1.7.4`, applies the tanh wrapping, and then projects the covariates onto the fixed direction `w = (1, 1, 1) / sqrt(3)`. The goal is to compare the same estimators when `mu(x) = f_{0.8}(w^\top x)` and `pi_r(x) = f_r(w^\top x)` in ambient dimension `d = 3`.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 3`
- Fixed projection:
  - `w = (1, 1, 1) / sqrt(3)`
- Base Fourier family:
  - `g_r(z) = {sum_{k=1}^{10} k^{-1/r} [sin(pi k z) + (-1)^k cos(pi k z)]} / sqrt(sum_{k=1}^{10} 2 k^{-2/r})`
- Tanh-wrapped family:
  - `f_r(z) = g_r(tanh(z))`
- Outcome regression:
  - `mu(x) = f_{0.8}(w^\top x)`
- Treatment regression candidates:
  - `pi_1(x) = f_1(w^\top x)`
  - `pi_2(x) = f_2(w^\top x)`
  - `pi_4(x) = f_4(w^\top x)`
  - `pi_8(x) = f_8(w^\top x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Test sample size: `n_test = 10000`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Compared methods: validation-selected Neural DML, standard Neural DML without model selection, paper minimax-debias estimator, and Oracle AIPW
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Validation sample size for validation-selected Neural DML: `validation_n = floor(n / 3) = 682`
- Validation-selection interval for validation-selected Neural DML: every `10` epochs, including epoch `200`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))` by default on the `D1` split
- Optimizer: Adam with profiled closed-form updates for the joint least-squares beta on `D2`
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Device: CPU by default unless explicitly changed in the simulation configuration

### Results

Average metrics over `10` trials for each treatment-regression candidate:

| pi family | Oracle AIPW beta MSE | Validation-selected DML beta MSE | Standard DML beta MSE | Minimax debias beta MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.001223 | 0.002193 | 0.002083 | 0.001937 |
| `r = 2` | 0.001230 | 0.002900 | 0.005638 | 0.001959 |
| `r = 4` | 0.001234 | 0.004059 | 0.002441 | 0.001404 |
| `r = 8` | 0.001235 | 0.003916 | 0.021411 | 0.001825 |

Nuisance MSEs for the two Neural DML variants:

| pi family | Selected DML mu MSE | Selected DML pi MSE | Standard DML mu MSE | Standard DML pi MSE |
| --- | ---: | ---: | ---: | ---: |
| `r = 1` | 0.105243 | 0.134028 | 0.279746 | 0.334412 |
| `r = 2` | 0.099622 | 0.228342 | 0.311210 | 0.490355 |
| `r = 4` | 0.099345 | 0.302466 | 0.327535 | 0.467517 |
| `r = 8` | 0.097011 | 0.331683 | 0.449055 | 0.532474 |

Bias-variance decomposition of beta estimation error over `10` trials:

| pi family | Selected DML squared bias | Selected DML variance | Standard DML squared bias | Standard DML variance | Minimax squared bias | Minimax variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r = 1` | 0.001196 | 0.000997 | 0.000122 | 0.001962 | 0.000128 | 0.001809 |
| `r = 2` | 0.002145 | 0.000755 | 0.000079 | 0.005559 | 0.000343 | 0.001616 |
| `r = 4` | 0.003320 | 0.000739 | 0.000573 | 0.001868 | 0.000485 | 0.000918 |
| `r = 8` | 0.003399 | 0.000517 | 0.000011 | 0.021400 | 0.000733 | 0.001092 |

Validation-selection diagnostics for Neural DML:

| pi family | Validation n | Checkpoint grid | Mean selected mu epoch | Mean selected pi epoch |
| --- | ---: | --- | ---: | ---: |
| `r = 1` | 682 | `10, 20, ..., 200` | 35.0 | 43.0 |
| `r = 2` | 682 | `10, 20, ..., 200` | 34.0 | 55.0 |
| `r = 4` | 682 | `10, 20, ..., 200` | 31.0 | 52.0 |
| `r = 8` | 682 | `10, 20, ..., 200` | 36.0 | 65.0 |

Main observations:

- Relative to the untanh-transformed `1.7.4` setting, the projected tanh-wrapped family makes the problem somewhat harder for all learned methods, but the minimax-debias estimator remains very competitive and is best among the learned methods at `r = 1`, `2`, and `8`.
- Validation-selected DML again gives the strongest nuisance fits, with `mu` MSE near `0.10` and `pi` MSE increasing from `0.13` to `0.33` as `r` grows. However, its beta error is still bias-driven rather than variance-driven.
- Standard DML is especially unstable at `r = 8`: its beta variance explodes to about `0.0214`, which drives the mean beta MSE above `0.02` even though its squared bias is tiny.
- The minimax-debias estimator is the most stable learned method across the whole grid. Its beta MSE stays between about `0.0014` and `0.0020`, close to the oracle benchmark, while its variance stays around `9e-4` to `1.8e-3`.
- The selected validation epoch for `pi` moves later as `r` increases, but much less dramatically than in `1.7.4`, suggesting that the `tanh` compression may smooth the effective learning problem along the projection direction.

Generated figures:

- `examples/plm/figs/1.7/1.7.5_pi_complexity_mean_mse_comparison.png`
- `examples/plm/figs/1.7/1.7.5_pi_complexity_beta_bias_sq.png`
- `examples/plm/figs/1.7/1.7.5_pi_complexity_beta_variance.png`
- `examples/plm/figs/1.7/1.7.5_unified_mse_mean_curve.png`

### Nuisance-learning diagnostic

Diagnostic artifact: `simulation_results/plm/1.7_5_tracking.json`.

This follow-up keeps the same `1.7.5` DGP and neural-network hyper-parameters, but replaces the comparison of final estimators by a single oracle tracking estimator. For each trial and each treatment-regression candidate, the estimator records the oracle nuisance MSE of `mu` and `pi` at every epoch on an independent validation sample of size `2048`. The plotting script then averages those epoch paths over all `10` trials and overlays the four `r` settings in one log-scale figure.

Main tracking observations from the validation-average paths:

- The average `mu` curve is nearly identical across all four `pi_r` settings, which is expected because `mu(x) = f_{0.8}(w^\top x)` is unchanged across the whole experiment.
- The average `mu` validation MSE drops rapidly from about `0.47` at epoch `0` to its minimum near epochs `37` to `38`, with minimum values around `0.10`, and then rises again by epoch `200`.
- The average `pi` validation MSE also improves fastest in the first few dozen epochs, reaching its minimum around epoch `27` for all four `r` settings.
- As `r` increases, the best achievable `pi` validation MSE worsens substantially: about `0.128` for `r = 1`, `0.224` for `r = 2`, `0.287` for `r = 4`, and `0.318` for `r = 8`.
- After those minima, the `pi` curves drift upward, especially for `r = 4` and `r = 8`, which is consistent with the validation-selected DML runs preferring much earlier nuisance checkpoints than the fixed final epoch.

Generated tracking figures:

- `examples/plm/figs/1.7/1.7.5_nuisance_validation_overlay_paths.png`
- `examples/plm/figs/1.7/1.7.5_nuisance_in_out_average_paths.png`

## 1.7.6

Experiment `1.7.6`, stored in the simulation artifact `1.7_6`.

### Goal

This experiment is an ablation study for the paper minimax-debias estimator on the same projected tanh-wrapped family as `1.7.5`. The goal is to separate the effect of learning the outcome regression `mu(x)` from the effect of the empirical debiasing weights `a_i`, and to see how the resulting minimax beta error evolves over the training path for different treatment-regression complexities.

### Setting and design

Specific data-generating setting:

- DGP class: `PartialLinearModelUniformNoiseDGP`
- Covariate dimension: `d = 3`
- Fixed projection:
  - `w = (1, 1, 1) / sqrt(3)`
- Tanh-wrapped family:
  - `mu(x) = f_{0.8}(w^\top x)`
  - `pi_1(x) = f_1(w^\top x)`
  - `pi_2(x) = f_2(w^\top x)`
  - `pi_4(x) = f_4(w^\top x)`
  - `pi_8(x) = f_8(w^\top x)`
- Trial-level target coefficient: `beta ~ Unif[-0.5, 0.5]`
- Treatment noise scale: `sigma_u = sqrt(3)` so that `Var(u) = 1`
- Outcome noise scale: `sigma_eps = sqrt(3)` so that `Var(eps) = 1`
- Training sample size: `n = 2048`
- Validation tracking sample size: `2048`
- Number of trials: `10` per treatment-regression candidate

Method design:

- Tracked method: paper minimax-debias estimator only
- Neural network depth: `L = 3`
- Neural network width: `N = 512`
- Outcome-network regularization: `lambda_mu = 2e-5`
- Treatment-network regularization: `lambda_pi = 2e-5`
- Paper debiasing penalty: `lambda_debias = 1 / (sqrt(n) * log_2(n))`
- Optimizer: Adam
- Learning rate: `lr = 1e-3`
- Mini-batch size: `batch_size = 2048`
- Training epochs: `niter = 200`
- Tracking grid: epochs `0, 10, 20, ..., 200`
- Debiasing-weight construction: the empirical weights `a_i` are solved once on the `D1` split after the full minimax fit is complete, using the final trained nuisance network. Those fixed final-trial weights are then paired with the saved `mu` checkpoints from epochs `0, 10, ..., 200`, so the last point on the beta path exactly matches the standard minimax estimate reported in `1.7.5`.

### Results

Average path summaries over `10` trials:

| pi family | mu MSE at epoch 0 | Minimum average mu MSE | Epoch of minimum mu MSE | Minimum average beta MSE | Epoch of minimum beta MSE | Beta MSE at epoch 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r = 1` | 0.469836 | 0.106094 | 40 | 0.001379 | 100 | 0.001937 |
| `r = 2` | 0.469836 | 0.105040 | 30 | 0.001119 | 190 | 0.001959 |
| `r = 4` | 0.469836 | 0.102529 | 30 | 0.001137 | 120 | 0.001404 |
| `r = 8` | 0.469836 | 0.101675 | 30 | 0.001607 | 110 | 0.001825 |

Main observations:

- The validation oracle `mu` MSE path now matches the DML nuisance-tracking diagnostic `1.7.5_tracking` exactly on the shared checkpoint grid, confirming that the minimax ablation uses the same nuisance-learning trajectory as the neural DML fit.
- The average `mu` path reaches its minimum around epoch `30` to `40` for all four treatment-regression families, with minimum validation oracle MSE between about `0.1017` and `0.1061`.
- The minimax beta error attains its minimum noticeably later than the oracle `mu` MSE. The best average beta MSE occurs around epoch `100` for `r = 1`, epoch `190` for `r = 2`, epoch `120` for `r = 4`, and epoch `110` for `r = 8`.
- The final beta MSE at epoch `200` exactly matches the standard minimax results reported in `1.7.5`, which confirms that the tracked beta path is now calibrated to the actual minimax estimator rather than to an earlier buggy weight construction.

Generated figure:

- `examples/plm/figs/1.7/1.7.6_minimax_ablation_paths.png`
