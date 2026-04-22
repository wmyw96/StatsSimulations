# Change Log

## 2026-04-15 21:01:49 EDT

- Added the initial framework design note in `docs/simulation_framework_outline.md`.
- Documented the proposed package structure, abstract interfaces, evaluator design, and testing strategy for future simulation studies.
- Established `log.md` as the running record of meaningful updates in the repository.

## 2026-04-15 21:14:46 EDT

- Scaffolded the `src/simlab` package hierarchy with `core`, `dgp`, `estimators`, `evaluation`, and `utils` modules.
- Added the first core dataclasses: `SampledData`, `EstimateResult`, and `TrialRecord`.
- Added abstract base classes for `DataGeneratingProcess`, `Estimator`, and `Evaluator`, using dictionaries for configs and defensive copying on initialization.
- Updated the framework note to reflect the adopted design choice of dataclasses for records and dictionaries for parameters.
- Kept the initial scaffold importable without requiring `numpy` at package import time.

## 2026-04-15 21:23:15 EDT

- Updated the framework note to make the base stack explicitly NumPy-first and to avoid pandas in the framework layer.
- Added `docs/implementation_status.md` as a quick reference for what is already implemented versus what is still planned.

## 2026-04-15 21:34:00 EDT

- Implemented `PartialLinearModelUniformNoiseDGP` with uniform covariates, uniform additive noise, and optional oracle outputs.
- Extended `SampledData` so it can carry oracle quantities `pi_x` and `mu_x`.
- Added a unit test for the new DGP covering shapes, oracle behavior, reproducibility, and zero-noise identities.

## 2026-04-15 21:50:47 EDT

- Created a project-local conda environment at `.conda/simlab`.
- Installed NumPy in that environment and used it to run the new DGP unit test successfully.
- Updated the implementation status document with the validated test command and environment location.

## 2026-04-15 21:57:30 EDT

- Refactored `SampledData` into a generic container with `observed` and `oracle` dictionaries.
- Removed `metadata` from sampled data so estimators cannot accidentally access truth-related information through the sample object.
- Updated the partial linear DGP and its unit test to use the generic sampled-data interface.

## 2026-04-15 22:28:00 EDT

- Extracted Eq. (1.2) from `PLM.pdf` and implemented PLM estimators around that exact AIPW formula.
- Added `src/simlab/estimators/plm_est.py` with the residual ReLU network, the neural DML estimator, and the oracle AIPW estimator.
- Added estimator unit tests covering the oracle formula, neural fit/predict behavior, and the odd-sample D1/D2 split rule.
- Installed PyTorch in the project-local conda environment and documented the new estimator implementation in `docs/implementation_status.md`.
- Removed the old PLM estimator placeholder files so the package now has a single PLM estimator module.

## 2026-04-15 23:10:21 EDT

- Updated the local git `origin` remote to `git@github.com:wmyw96/StatsSimulations.git`.

## 2026-04-19 12:22:49 EDT

- Added a draft `examples/plm/exp_log.md` for the first PLM experiment.
- Rewrote the experiment log structure so the general section describes the DGP class and method semantics, while experiment-specific parameter values live in the detailed results section.

## 2026-04-19 13:05:00 EDT

- Implemented `PLMEvaluator` with JSON-based resume, per-trial saving, PLM metric evaluation, and summary queries.
- Added the PLM experiment-definition file and example scripts for running simulations and plotting separate result figures.
- Added a unit test for evaluator resume/query behavior and updated the experiment log with the fixed training-epoch setting.

## 2026-04-19 14:10:00 EDT

- Ran experiment `1.1_1` with `10` trials for each sample size and saved the full results to `simulation_results/plm/1.1_1.json`.
- Generated the separate result figures under `examples/plm/figs`.
- Wrote a numerical summary and short interpretation of the experiment into `examples/plm/exp_log.md`.

## 2026-04-19 14:38:00 EDT

- Extended `examples/plm/vis_result.py` to generate trial-level scatter plots of `mu_mse * pi_mse` against the final beta squared error for the DML estimator.
- Regenerated the PLM figures for experiment `1.1_1`, including one combined scatter plot and one per sample size.

## 2026-04-19 14:55:00 EDT

- Reorganized `examples/plm/exp_log.md` so the completed baseline run is archived as Experiment `1.1.1`.
- Added a new Experiment `1.1.2` section describing the next run, which will keep the same sine-sine PLM setting but focus on a product-based nuisance diagnostic built from the fitted nuisance functions themselves.

## 2026-04-19 16:20:00 EDT

- Extended the PLM evaluator to record trial-level nuisance-product summaries from the fitted nuisance functions on the test sample.
- Updated `examples/plm/vis_result.py` so the scatter plots for Experiment `1.1.2` use `mean(mu_hat * pi_hat)` rather than the earlier `mu_mse * pi_mse` proxy.
- Ran Experiment `1.1_2` with `10` trials per sample size, generated the new figures, and filled in the `1.1.2` results section of `examples/plm/exp_log.md`.

## 2026-04-19 16:42:00 EDT

- Added dotted experiment-id normalization so the example scripts accept user-facing ids like `1.1.2` while still loading the existing storage artifact `1.1_2`.
- Extended `examples/plm/vis_result.py` with a single beta-error comparison plot for Experiment `1.1.2`, comparing the DML AIPW estimate, the neural-network joint least-squares beta estimate, and the oracle AIPW estimate across sample sizes.
- Regenerated the `1.1.2` figure set using the dotted experiment label and updated `examples/plm/exp_log.md` to list the new comparison figure.

## 2026-04-19 16:50:00 EDT

- Relabeled the user-facing `beta_init_mse` metric as the joint least-squares beta estimate in the PLM report and plotting script.
- Regenerated the `1.1.2` figures so the standalone beta curve and the beta-error comparison plot use the corrected joint-LSE terminology.

## 2026-04-19 17:15:00 EDT

- Added experiment family `1.2` for the sine-sine PLM with the same settings as `1.1`, except the ground-truth coefficient is `beta = 0.5`.
- Updated `examples/plm/vis_result.py` so it derives the fixed DGP query configuration from the experiment definition instead of assuming `beta = 0.0`.
- Ran Experiment `1.2.1` with `10` trials per sample size, generated the new figure set, and filled in the `1.2.1` results section of `examples/plm/exp_log.md`.

## 2026-04-19 18:05:00 EDT

- Diagnosed the inconsistent joint-LSE beta behavior in the `beta = 0.5` PLM setting: under the old single-optimizer joint fit, the flexible `mu` network absorbed most of the `beta * pi(X)` signal, leaving the scalar beta update stuck near zero.
- Reworked `PLMDMLEstimator` so the nuisance networks are trained with Adam while the joint-LSE beta is updated in profiled closed form on `D2` after each epoch.
- Added deterministic PyTorch seeding when an estimator seed is provided and added a regression test showing the joint-LSE beta tracks a nonzero signal in the sine-sine PLM.

## 2026-04-19 18:35:00 EDT

- Reorganized local visualization outputs into family folders under `examples/plm/figs/`, including `examples/plm/figs/1.1/` and `examples/plm/figs/1.2/`.
- Updated `examples/plm/vis_result.py` so new figures are saved into `figs/<experiment-family>/` instead of directly into `figs/`.
- Reran the corrected `beta = 0.5` setting as Experiment `1.2.2`, generated the full figure set in `examples/plm/figs/1.2/`, and updated `examples/plm/exp_log.md` so `1.2.1` is archived as the pre-fix run and `1.2.2` is the corrected reference run.

## 2026-04-19 20:05:00 EDT

- Added experiment family `1.3` with trial-level random coefficients `beta ~ Unif[-0.5, 0.5]`, while keeping the sine-sine PLM and the existing DML/oracle method settings.
- Extended the PLM evaluator to store the realized `beta_true` for each trial so randomized-coefficient experiments can be summarized cleanly.
- Updated `examples/plm/vis_result.py` with the requested custom color bank and a dedicated unified `log_2(n)` versus `log_2(MSE)` plot for the `1.3` family.
- Ran Experiment `1.3.1` with `30` trials per sample size, saved the results to `simulation_results/plm/1.3_1.json`, generated `examples/plm/figs/1.3/1.3.1_unified_mse_scaling.png`, and documented the outcome in `examples/plm/exp_log.md`.

## 2026-04-19 21:10:00 EDT

- Continued Experiment `1.3.1` from the saved 30-trial checkpoint to `100` trials per sample size using the evaluator's resume mechanism.
- Regenerated the unified `1.3.1` scaling figure from the 100-trial averages.
- Updated the `1.3.1` section of `examples/plm/exp_log.md` so the table and interpretation now reflect the more stable 100-trial run.

## 2026-04-19 23:40:00 EDT

- Tightened `PLMEvaluator` resume behavior so existing result files are validated against the requested experiment definition before new trials are appended.
- Updated the resume path so top-level result metadata such as `n_trials` is refreshed when a run is extended, instead of leaving stale headers in the saved JSON.
- Added trial-seeded DML support for exact experiment ids like `1.3.2`, while preserving the archived fixed-seed behavior for `1.3.1`.
- Reconstructed `simulation_results/plm/1.3_1.json` back to the original 30-trial archived run, reran the corrected random-seed experiment as `1.3_2.json` with 100 trials, regenerated the `1.3.1` and `1.3.2` unified figures, and rewrote the `1.3` sections of `examples/plm/exp_log.md`.

## 2026-04-20 00:35:00 EDT

- Added `PLMDMLOracleTrackingEstimator`, which records oracle `mu` and `pi` MSE trajectories across epochs while keeping the same DML fitting pipeline.
- Extended the PLM evaluator so epoch-path diagnostics such as `epoch_grid`, `mu_mse_path`, and `pi_mse_path` are persisted in experiment result files when available.
- Added experiment family `1.4` with exact id `1.4_1` for nuisance-learning trajectory diagnostics at fixed `n = 1024`.
- Ran Experiment `1.4.1` with `20` trials, generated `examples/plm/figs/1.4/1.4.1_nuisance_mse_paths.png`, and documented the optimization-path summary in `examples/plm/exp_log.md`.

## 2026-04-20 00:45:00 EDT

- Updated the `1.4.1` nuisance-trajectory figure so the vertical axis uses a logarithmic scale, making late-epoch improvements easier to inspect.
- Revised the `1.4.1` experiment note in `examples/plm/exp_log.md` to describe the log-scaled trajectory visualization.

## 2026-04-20 00:55:00 EDT

- Added Experiment `1.4.2`, a six-value lambda sweep for the oracle-tracked nuisance-learning study with `lambda_mu = lambda_pi in {2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4}`.
- Extended the PLM visualizer with two `1.4.2` outputs: a six-panel raw trajectory figure with shared log-scaled y-limits and an amortized average-path figure colored by lambda and styled by nuisance target.
- Ran `1.4.2` with `20` trials, generated both `examples/plm/figs/1.4/1.4.2_lambda_path_panels.png` and `examples/plm/figs/1.4/1.4.2_lambda_average_paths.png`, and documented the lambda-sweep summary in `examples/plm/exp_log.md`.

## 2026-04-20 01:45:00 EDT

- Added Experiment `1.4.3`, a wider seven-value lambda sweep with `lambda_mu = lambda_pi = 5^l * 1e-4` for `l in {-3, -2, -1, 0, 1, 2, 3}`.
- Generalized the `1.4` plotting code so lambda-sweep experiments can render an arbitrary number of panels instead of assuming a fixed six-panel layout.
- Ran `1.4.3` with `20` trials, generated `examples/plm/figs/1.4/1.4.3_lambda_path_panels.png` and `examples/plm/figs/1.4/1.4.3_lambda_average_paths.png`, and added the new wide-range summary to `examples/plm/exp_log.md`.

## 2026-04-20 11:05:00 EDT

- Extended oracle nuisance tracking so tracked estimators can evaluate epoch-by-epoch nuisance MSE on an independent validation sample instead of only on `D2`.
- Added exact experiment id `1.4.4` for validation-based nuisance tracking with `n = n_val = 1024` and the baseline `lambda_mu = lambda_pi = 1e-4`.
- Generated separate validation-path figures for `mu` and `pi`, ran `1.4.4` with `10` trials, and documented the validation-based checkpoint summary in `examples/plm/exp_log.md`.

## 2026-04-20 11:35:00 EDT

- Reworked `1.4.4` into a seven-value lambda sweep that tracks oracle nuisance MSE on both `D2` and an independent validation sample under the same fitted networks.
- Generalized the tracking diagnostics so one fit can persist multiple tracked sources at once, and updated the visualizer to emit separate averaged figures for `D2` and validation.
- Reran `1.4.4` with `10` trials, generated `examples/plm/figs/1.4/1.4.4_d2_average_paths.png` and `examples/plm/figs/1.4/1.4.4_validation_average_paths.png`, and rewrote the `1.4.4` section of `examples/plm/exp_log.md`.

## 2026-04-20 12:05:00 EDT

- Audited the saved `1.4.4` artifact after the D2 and validation figures looked nearly identical, and verified that the tracked paths are distinct in every record.
- Confirmed the visual similarity comes from very small generalization gaps in this smooth one-dimensional setup, not from reusing `D2` as the validation sample.

## 2026-04-20 13:15:00 EDT

- Added Experiment `1.5.1`, which fixes `n = 1024`, `lambda_mu = lambda_pi = 2e-5`, and compares three treatment regressions `sin(2 pi x)`, `sin(4 pi x)`, and `sin(8 pi x)` under the random-beta PLM design.
- Extended the evaluator/experiment factory interface so oracle estimators can follow the current DGP nuisance function when a nuisance function name varies across the parameter grid.
- Added the `1.5.1` visualization, ran the experiment with `30` trials per `pi` choice, generated `examples/plm/figs/1.5/1.5.1_pi_complexity_mse_comparison.png`, and documented the results in `examples/plm/exp_log.md`.

## 2026-04-20 14:10:00 EDT

- Added Experiment `1.5.2`, which replaces the treatment-regression family with `sign(sin(2 pi x)) * sin(k pi x)` for `k in {2, 4, 8}` while keeping `n = 1024` and `lambda_mu = lambda_pi = 2e-5`.
- Generalized the `1.5` plotting helper to pull the treatment-regression labels from the experiment definition, so the figure stays aligned with whichever `pi` family a specific exact id uses.
- Ran `1.5.2` with `30` trials per signed treatment-regression choice, generated `examples/plm/figs/1.5/1.5.2_pi_complexity_mse_comparison.png`, and added the new results section to `examples/plm/exp_log.md`.

## 2026-04-20 15:05:00 EDT

- Added Experiment `1.5.3`, correcting the `1.5.2` family to use `sign(sin(2 pi x)) * |sin(k pi x)|` for `k in {2, 4, 8}` while keeping `n = 1024` and `lambda_mu = lambda_pi = 2e-5`.
- Extended the function registry and experiment-id tests for the corrected absolute-value family and reran the `1.5` comparison with `30` trials per treatment-regression choice.
- Generated `examples/plm/figs/1.5/1.5.3_pi_complexity_mse_comparison.png` and documented the corrected results in `examples/plm/exp_log.md`.

## 2026-04-20 15:30:00 EDT

- Updated the `1.5.3` comparison to include the joint least-squares beta curve and added an interpretation note explaining that the main driver appears to be the overlap between `mu(X)` and the systematic part of `T`, not the standalone difficulty of estimating `pi(X)`.

## 2026-04-20 15:45:00 EDT

- Added a four-function progressive `pi` candidate family with fixed `mu(x)=sin(2 pi x)` to the shared function registry, designed so both nuisance roughness and overlap with `mu` can be increased together in a later experiment.
- The new candidates use correlations with `mu` of roughly `0.25`, `0.50`, `0.75`, and `0.90`, which should make it easier to tune a sequence where both DML and joint LSE beta performance degrade rather than improve.

## 2026-04-20 16:15:00 EDT

- Added Experiment `1.5.4` using the new four-level progressive `pi` family at `n = 1024` with `lambda_mu = lambda_pi = 2e-5`, and ran it with `30` trials per treatment-regression choice.
- Generated `examples/plm/figs/1.5/1.5.4_pi_complexity_mse_comparison.png` and documented the result that this family did not produce the desired monotone degradation of DML and joint LSE beta performance.

## 2026-04-20 18:35:00 EDT

- Added Experiment `1.5.5`, a fixed-overlap `pi` family that keeps the treatment regression highly aligned with `mu(x) = sin(2 pi x)` while increasing its roughness from smooth cosine to high-frequency discontinuous sign waves.
- Extended the exact-id registry/tests for `1.5.5`, ran the new experiment with `30` trials per treatment-regression choice, and generated `examples/plm/figs/1.5/1.5.5_pi_complexity_mse_comparison.png`.
- Documented that `1.5.5` finally makes DML `pi` error increase monotonically across the four candidates, even though the DML and joint-LSE beta errors still do not degrade monotonically with that nuisance difficulty.

## 2026-04-20 19:20:00 EDT

- Added Experiment `1.5.6`, which moves the nuisance structure to `d = 4` with `mu(x) = 0.5 * sum_{j=1}^4 sin(2 pi x_j)` and a three-level fixed-overlap `pi` family ranging from smooth low-frequency to discontinuous high-frequency.
- Extended the function registry and exact-id evaluator tests for the new four-dimensional family, then ran `1.5.6` with `30` trials and generated `examples/plm/figs/1.5/1.5.6_pi_complexity_mse_comparison.png`.
- Documented that the `d = 4` setup makes the nuisance fits and both beta estimators dramatically harder overall, even though the beta errors still do not degrade monotonically with the standalone treatment-regression MSE.

## 2026-04-20 20:25:00 EDT

- Added Experiment `1.5.7`, which keeps `mu(x) = sin(2 pi x_1)` structurally simple in `d = 4` while making `pi(x)` depend on `x_2`, `x_3`, and `x_4` through progressively rougher smooth and discontinuous components.
- Extended the function registry, exact-id evaluator tests, and figure labels for the new isolated-`pi` family, then ran `1.5.7` with `30` trials and generated `examples/plm/figs/1.5/1.5.7_pi_complexity_mse_comparison.png`.
- Documented that `1.5.7` does make the treatment nuisance harder in a cleaner way, but the fitted `mu` error remains large in `d = 4`, so the experiment still does not cleanly isolate the effect of treatment-regression difficulty on beta estimation.

## 2026-04-20 21:35:00 EDT

- Added Experiment `1.5.8`, using the easier outcome regression `mu(x) = sin(pi x_1) + cos(pi x_2)` and a four-level treatment-regression family that perturbs `mu(x)` by increasingly rough same-coordinate components.
- Extended the function registry and exact-id evaluator tests for `1.5.8`, ran the full `30`-trial experiment, and generated `examples/plm/figs/1.5/1.5.8_pi_complexity_mse_comparison.png`.
- Documented that `1.5.8` is the closest match so far to the requested design: the first three settings show both DML `pi` MSE and DML beta MSE increasing together, while the fourth setting still breaks the monotone beta trend.

## 2026-04-20 22:40:00 EDT

- Added Experiment `1.5.9`, implementing the correlated `g1/g2` design with `mu(x) = 0.95 g_1(x) + 0.05 g_2(x)` and a three-level treatment-regression family `g_1(x) + b g_2(x)` for increasing `b`.
- Extended the function registry and exact-id evaluator tests for `1.5.9`, ran the full `30`-trial experiment, and generated `examples/plm/figs/1.5/1.5.9_pi_complexity_mse_comparison.png`.
- Documented that the treatment-regression MSE does increase under this correlated construction, but the DML beta MSE still does not rise monotonically, so the family remains only partially successful.

## 2026-04-20 23:15:00 EDT

- Added Experiment `1.5.10`, which keeps the same correlated `g1/g2` construction as `1.5.9` but widens the treatment-regression coefficients to `g_1(x) + 0.05 g_2(x)`, `g_1(x) + 0.5 g_2(x)`, and `g_1(x) + 1.0 g_2(x)`.
- Extended the exact-id evaluator tests for `1.5.10`, ran the full `30`-trial experiment, and generated `examples/plm/figs/1.5/1.5.10_pi_complexity_mse_comparison.png`.
- Documented that the wider range does make the treatment-regression MSE much larger, but both DML and joint-LSE beta errors still decrease across the family, so the larger coefficient spread does not fix the core issue.

## 2026-04-20 23:45:00 EDT

- Audited the `1.5.10` beta formulas and found the main explanation for the counterintuitive trend: as the rough `g_2` coefficient grows, the estimated DML denominator `E[T (T - \hat{\pi}(X))]` increases sharply because the learned treatment model leaves more structured signal in the residual.
- In a small replay of the `1.5.10` family, the average DML denominator roughly doubled from `0.111` to `0.264`, while the nuisance cross-term stayed small, so the final beta ratio became better conditioned even though the pointwise `\pi` MSE got worse.

## 2026-04-21 00:05:00 EDT

- Recorded the follow-up diagnosis that this denominator inflation is itself a sign that the current `1.5.10` family is not the right stress test for the intended question.
- Since `u ~ Unif[-0.5, 0.5]`, the target denominator should stay near `Var(u) = 1/12 ≈ 0.0833` under a good nuisance fit. The fact that the learned DML denominator moves far above that means the residual `T - \hat{\pi}(X)` is still carrying structured signal, so the experiment is changing identification strength instead of only changing nuisance difficulty.

## 2026-04-20 22:05:00 EDT

- Added a design note for the next `1.5` direction: split `mu` into a dominant smooth component `g1` and a small-amplitude rough component `g2`, then build `pi` so the rough `g2` direction is amplified relative to `mu`.
- This is intended to keep the outcome regression mostly easy while making the treatment regression increasingly aligned with the hard-to-learn component, which is a better candidate for forcing both nuisance error and DML beta error upward together.

## 2026-04-21 16:55:00 EDT

- Added Experiment `1.5.11`, a new one-dimensional unit-variance stress test with `g_1(x) = sin(pi x)`, a rough correlated component `g_2(x) = sign(sin(pi x)) * 0.5 * (|sin(8 pi x)| + |sin(16 pi x)|)`, and `sigma_u = sigma_eps = sqrt(3)` so both noises have variance one.
- Extended the function registry, exact-id evaluator tests, and experiment builder mapping for `1.5.11`, then ran the full `30`-trial experiment and generated `examples/plm/figs/1.5/1.5.11_pi_complexity_mse_comparison.png`.
- Documented that the new design finally keeps the DML beta MSE stable and close to oracle while `pi` clearly gets harder, but it still does not produce a strong monotone deterioration in the final DML AIPW beta estimator; the joint least-squares beta is the quantity that degrades more noticeably.

## 2026-04-21 17:25:00 EDT

- Recorded the next design principle for the PLM stress tests: if we want the fitted residuals `Y - \hat{\mu}(X)` and `T - \hat{\pi}(X)` to be highly correlated, we should make `mu` and `pi` share the same hard-to-learn component with the same sign, not just make `pi` harder in isolation.
- The suggested template is `mu(x) = a_g g(x) + a_h h(x)` and `pi_k(x) = c_k (b_g g(x) + b_{h,k} h(x))`, where `g` is easy, `h` is rough but positively aligned with `g`, `a_h` is large enough that `\hat{\mu}` still misses `h`, and the scale factor `c_k` is chosen to keep the variance of `pi_k(X)` roughly fixed across `k` so the denominator does not drift.

## 2026-04-21 19:05:00 EDT

- Added Experiment `1.5.12`, a new one-dimensional shared-`g/h` family where both `mu(x)` and `pi(x)` contain the same centered, variance-normalized hard component `h(x)` and each `pi_k` is rescaled so the variance of `pi_k(X)` stays approximately fixed.
- Extended the PLM evaluator to record two new diagnostics on the oracle test sample: the nuisance-error correlation and the oracle-beta residual correlation, then added exact-id coverage for `1.5.12` in the evaluator tests.
- Ran the full `30`-trial `1.5.12` experiment and generated `examples/plm/figs/1.5/1.5.12_pi_complexity_mse_comparison.png`; the new family does increase nuisance-error alignment and residual correlation, but it still only produces a mild change in the final DML AIPW beta MSE.

## 2026-04-21 20:05:00 EDT

- Implemented a new paper-style PLM estimator based on the minimax debiasing program in equation `(2.3)` of `aplm.pdf`, with the requested stabilized inner penalty `n^{-1} \\, \\sum_i (\\beta T_i + f(X_i))^2` replacing the paper’s original `n^{-1} \\, \\sum_i f^2(X_i)`.
- Added the new `PLMMinimaxDebiasEstimator`, a difference-class adversary network, a factory helper in `examples/plm/experiment_defs.py`, and package exports so the estimator can be imported and used alongside the existing DML and oracle baselines.
- Added unit tests covering the new estimator’s fit/predict path, bounded debiasing weights, default `lambda_debias` behavior, and evaluator compatibility; `17` unit tests passed after the change.

## 2026-04-21 20:25:00 EDT

- Changed the paper estimator’s default debiasing penalty to `1 / (sqrt(n) * log_2(n))`, with `n` interpreted as the size of the debiasing split `D1`.
- Added a reusable trial-seeded factory helper for the paper estimator and updated the PLM visualization code so future experiments can show the minimax-debias beta curve automatically whenever that estimator is included.
- Expanded the experiment log’s method section to document the paper estimator and its default hyper-parameters for future simulation writeups.

## 2026-04-20 23:55:00 EDT

- Added experiment `1.6.1`, a new two-dimensional unit-variance PLM family with `mu(x) = sin(pi x_1) + 0.33 cos(8 pi x_2)` and `pi_k(x) = sin(pi x_1) + ((k+1)/3) cos(8 pi x_2)` for `k in {0,1,2}`.
- Included the paper minimax-debias estimator in the `1.6.1` comparison set, extended the experiment registry/tests for the new exact id, and reused the existing pi-complexity plotting pipeline for the new family.
- Fixed a PLM evaluator resume bug so a stale larger `n_trials` header from an interrupted run can now be lowered back to the current target when the completed trial counts support it.
- Ran the full `10`-trial first-pass `1.6.1` experiment and generated `examples/plm/figs/1.6/1.6.1_pi_complexity_mse_comparison.png`; in this design the treatment nuisance gets progressively harder, the joint LSE beta degrades monotonically, and the minimax-debias beta estimate is markedly more stable than the plain DML AIPW beta.

## 2026-04-21 00:35:00 EDT

- Extended experiment `1.6.1` from `10` to `50` trials by resuming the saved artifact, regenerated the `examples/plm/figs/1.6/1.6.1_pi_complexity_mse_comparison.png` figure, and refreshed the experiment report with the new averages.
- At the 50-trial scale, the treatment nuisance difficulty still grows clearly across the three `pi_k` settings, the joint LSE beta estimate degrades monotonically, and the paper minimax-debias beta estimate remains substantially more stable than the plain DML AIPW beta.

## 2026-04-21 00:50:00 EDT

- Added median summaries for experiment `1.6.1` and a short tail-risk diagnosis to the experiment log.
- The new analysis shows that the DML median beta MSE is almost monotone across the three `pi_k` settings; the non-monotone mean is mainly driven by a few catastrophic `pi_2` trials with very large beta error.

## 2026-04-21 01:05:00 EDT

- Updated the pi-complexity visualization pipeline so families like `1.6.1` now emit separate mean-based and median-based summary figures, while keeping the old mean filename as a compatibility alias.
- Extended `PLMEvaluator.query_results()` to support `mode="median"` and added evaluator test coverage for the new aggregation path.

## 2026-04-21 01:35:00 EDT

- Added experiment `1.6.2`, a two-dimensional unit-variance PLM family with `mu(x) = 0.25 sin(2 pi x_1)` and `pi_k(x) = (k+1) sin(2 pi x_1)` for `k in {0,1,2}`.
- Ran the full `10`-trial `1.6.2` experiment with DML, paper minimax-debias, and oracle estimators; generated mean and median pi-complexity figures under `examples/plm/figs/1.6/`.
- Updated `examples/plm/exp_log.md` with mean/median tables and noted that DML AIPW degrades sharply at the largest treatment amplitude while the minimax-debias beta estimate remains close to oracle.

## 2026-04-21 09:28:00 EDT

- Added experiment `1.6.3`, a two-dimensional unit-variance PLM family with `mu(x) = sin(2 x_1) + 0.25 sin(6 x_2)` and `pi_k(x) = sin(2 x_1) + (k+1) sin(6 x_2)` for `k in {0,1,2}`.
- Extended the exact-id evaluator tests for `1.6.3`, ran the full `10`-trial comparison with DML, paper minimax-debias, and oracle estimators, and generated separate mean and median figures under `examples/plm/figs/1.6/`.
- Updated the experiment log with mean/median tables; DML AIPW worsens once the second-coordinate treatment amplitude increases, while the paper minimax-debias estimator remains much closer to oracle in this run.

## 2026-04-21 10:13:00 EDT

- Added experiment `1.6.4`, a two-dimensional unit-variance PLM family with `mu(x) = sin(2 x_1) + 0.25 sin(6 x_1)` and `pi_k(x) = sin(2 x_1) + (k+1) sin(6 x_1)` for `k in {0,1,2}`.
- Extended the exact-id evaluator tests for `1.6.4`, ran the full `10`-trial comparison with DML, paper minimax-debias, and oracle estimators, and generated separate mean and median figures under `examples/plm/figs/1.6/`.
- Updated the experiment log with mean/median tables; this same-coordinate design does not produce monotone DML degradation at 10 trials, but the paper minimax-debias estimator remains close to oracle.

## 2026-04-21 10:34:00 EDT

- Added experiment `1.6.5`, a two-dimensional unit-variance PLM family with `mu(x) = 0.25 sin(6 x_1)` and `pi_k(x) = (k+1) sin(6 x_1)` for `k in {0,1,2}`.
- Extended the exact-id evaluator tests for `1.6.5`, ran the full `10`-trial comparison with DML, paper minimax-debias, and oracle estimators, and generated separate mean and median figures under `examples/plm/figs/1.6/`.
- Updated the experiment log with mean/median tables; this rough-only design gives a cleaner monotone increase in DML beta MSE and nuisance MSEs as the treatment amplitude grows.

## 2026-04-21 12:07:00 EDT

- Added experiment `1.6.6`, a two-dimensional unit-variance PLM family with shared base signal `g(x) = 0.25 sin(5 x_1) + 0.125 sin(20 x_1)`, `mu(x) = g(x)`, and `pi_k(x) = 4k g(x)` for `k in {0.5, 1, 2}`.
- Extended the exact-id evaluator tests for `1.6.6`, ran the full `10`-trial comparison with DML, paper minimax-debias, and oracle estimators, and generated separate mean and median figures under `examples/plm/figs/1.6/`.
- Updated the experiment log with mean/median tables; treatment nuisance MSE grows strongly with the scaling, but DML AIPW beta MSE remains fairly flat and non-monotone at this 10-trial scale.

## 2026-04-21 12:26:00 EDT

- Added experiment `1.6.7`, a two-dimensional unit-variance PLM family with `eta(x) = sin(x_1) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`, `mu(x) = eta(x)`, and `pi_k(x) = 4k eta(x)` for `k in {0.5, 1, 2}`.
- Extended the exact-id evaluator tests for `1.6.7`, ran the full `10`-trial comparison with DML, paper minimax-debias, and oracle estimators, and generated separate mean and median figures under `examples/plm/figs/1.6/`.
- Updated the experiment log with mean/median tables; the high-scaling settings produce catastrophic DML and joint-LSE instability while oracle AIPW remains stable.

## 2026-04-21 12:46:00 EDT

- Added experiment `1.6.8`, a two-dimensional unit-variance PLM family with `eta(x) = sin(x_2) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`, `mu(x) = eta(x)`, and `pi_k(x) = 4k(eta(x) - sin(x_2))` for `k in {1, 2, 3}`.
- Extended the exact-id evaluator tests for `1.6.8`, ran the full `10`-trial comparison with DML, paper minimax-debias, and oracle estimators, and generated separate mean and median figures under `examples/plm/figs/1.6/`.
- Updated the experiment log with mean/median tables; the residual-treatment design avoids the catastrophic instability of `1.6.7` while still increasing DML beta and treatment-nuisance errors with scaling.

## 2026-04-21 13:05:00 EDT

- Updated the `1.5`/`1.6` pi-complexity visualization path to focus on DML AIPW and paper minimax-debias beta estimators only, removing median-plot generation and joint-LSE curves from the regenerated performance figure.
- Added separate squared-bias and variance figures for beta estimation error, using trial-level errors `hat_beta - beta_true` so the decomposition remains valid when `beta_true` is sampled independently by trial.
- Regenerated the `1.6.8` figures and updated the experiment log with the beta MSE, squared-bias, and variance decomposition over the `10` trials.

## 2026-04-21 14:48:08 EDT

- Added experiment `1.6.9`, with `eta(x) = sin(x_2) + 0.25 sin(5 x_2) + 0.05 sin(20 x_2)`, `mu(x) = eta(x)`, and `pi_k(x) = sin(x_2) + 4k(eta(x) - sin(x_2))` for `k in {1,2,3}`.
- Added a balanced discrete beta sampler for `beta in {-0.5, 0, 0.5}` and verified that the completed `60`-trial run has `20` trials at each beta value for every treatment-regression candidate.
- Added requested `1.6.9` figures for MSE and grouped beta bias/variance, then updated the experiment log with the numerical summary from the saved simulation artifact.

## 2026-04-21 16:18:50 EDT

- Added experiment `1.6.10`, reusing the residual-only `1.6.8` treatment family with balanced `beta in {-0.5, 0, 0.5}` for grouped bias-variance analysis.
- Ran the full `60`-trial-per-treatment-setting simulation, verified balanced beta counts, and generated the requested MSE and grouped bias/variance figures under `examples/plm/figs/1.6/`.
- Updated `examples/plm/exp_log.md` with the `1.6.10` numerical tables and main observations from the saved simulation artifact.

## 2026-04-21 21:08:41 EDT

- Added experiment `1.6.11`, a ten-dimensional PLM with `n = 2048`, `batch_size = 2048`, balanced `beta in {-0.5, 0, 0.5}`, and residual-only treatment regressions `pi_k(x) = 4k(eta(x) - sin(x_1))`.
- Ran the full `60`-trial-per-treatment-setting simulation, verified balanced beta counts, and generated the requested MSE and grouped beta bias/variance figures.
- Updated `examples/plm/exp_log.md` with the `1.6.11` results, including the observed high bias of the current minimax-debias estimator in this high-dimensional setting.

## 2026-04-21 21:42:38 EDT

- Added experiment `1.6.12`, a three-dimensional PLM with `n = 2048`, `batch_size = 2048`, `beta ~ Unif[-0.5, 0.5]`, and smooth-plus-residual treatment regressions on the second coordinate.
- Ran the requested `10` trials per treatment-regression setting, generated the standard beta MSE, squared-bias, and variance figures, and extracted nuisance MSE summaries from the saved artifact.
- Updated `examples/plm/exp_log.md` with the `1.6.12` numerical results, including the observed DML AIPW failure at the largest treatment scaling.

## 2026-04-21 21:51:07 EDT

- Added a unified `1.6.12` visualization that combines oracle beta MSE, DML nuisance MSE, and beta-MSE boxplots for DML and minimax debias in one log-scale figure.
- Regenerated the `1.6.12` visualization outputs and recorded the new unified figure path in `examples/plm/exp_log.md`.

## 2026-04-21 22:18:35 EDT

- Added experiment `1.6.13`, a two-dimensional PLM with `n = 1024`, `batch_size = 1024`, `beta ~ Unif[-0.5, 0.5]`, and treatment regressions `sin(x_2) + k sin(5x_2) + 0.05 sin(20x_2)`.
- Ran the requested `10` trials per treatment-regression setting and generated standard beta decomposition figures plus the unified nuisance/beta-MSE boxplot.
- Updated `examples/plm/exp_log.md` with the `1.6.13` numerical results and comparison notes against `1.6.12`.
