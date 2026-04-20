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
