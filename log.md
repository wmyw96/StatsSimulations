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
