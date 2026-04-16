# Generic Simulation Framework Outline

## Purpose

We want a simulation package that works for the immediate partial linear model (PLM) study, but is not tied to PLM or to one comparison against double machine learning (DML). The framework should make it easy to:

- define a new data-generating process (DGP),
- plug in one or more estimators,
- run controlled simulation experiments across parameter grids,
- save results in a structured format,
- test each new DGP and estimator with a repeatable contract.

The first concrete use case is:

- a PLM DGP generating `(X, T, Y)`,
- a baseline DML estimator with neural-network nuisance models,
- an estimator implementing the paper's proposed method,
- an evaluator that compares them fairly across repeated trials.

## Design Principles

### 1. Separate scientific roles cleanly

The framework should distinguish:

- the **true world**: the DGP and its ground-truth target,
- the **algorithm**: an estimator and its fitted state,
- the **experiment runner**: how we vary settings and aggregate results.

This prevents estimator code from implicitly depending on DGP internals while still allowing evaluation against known truth.

### 2. Make randomness explicit

Every simulation component should receive a seed or RNG object explicitly. This matters for:

- reproducibility,
- fair comparison between estimators,
- debugging failures in repeated trials.

### 3. Keep interfaces minimal but typed

The base classes should be small. Rich behavior should live in helper dataclasses and result objects rather than in a large abstract API.

### 4. Separate configuration from execution

Simulation settings should be declarative. A study should be representable as a config object or YAML/JSON entry later, even if we start with Python constructors now.

### 5. Make outputs analysis-friendly

The evaluator should save results in a tidy tabular format so downstream analysis is easy in pandas, polars, or R.

### 6. Support extension beyond one model

PLM should be one module under a broader simulation framework. The same evaluator should later handle:

- partially linear IV,
- treatment effect simulations,
- missing-data settings,
- high-dimensional linear and nonlinear benchmarks.

## Proposed Package Layout

```text
Simulation/
  docs/
    simulation_framework_outline.md
  src/
    simlab/
      __init__.py
      core/
        types.py
        randomness.py
        records.py
      dgp/
        __init__.py
        base.py
        partial_linear.py
      estimators/
        __init__.py
        base.py
        dml_nn.py
        proposed_plm.py
      evaluation/
        __init__.py
        experiment.py
        metrics.py
        results.py
      utils/
        io.py
        grids.py
        validation.py
  tests/
    unit/
      test_partial_linear_dgp.py
      test_dml_nn_estimator.py
      test_proposed_plm_estimator.py
      test_experiment_runner.py
    integration/
      test_plm_study_smoke.py
  scripts/
    run_plm_study.py
```

`simlab` is only a placeholder name. We can rename it later.

## Core Abstractions

The three abstract classes you proposed are the right backbone. I would make two small refinements:

- treat `params` and `hyper_parameters` as typed config objects or plain dictionaries that are stored but not mutated silently,
- make `predict()` accept data explicitly so the contract is unambiguous.

### A. DataGeneratingProcess

Responsibilities:

- store the model definition and its parameters,
- generate samples,
- expose ground-truth quantities needed for evaluation.

Recommended interface:

```python
class DataGeneratingProcess(ABC):
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def sample(self, n: int, seed: int | None = None):
        """Return one simulated dataset."""

    def true_parameter(self):
        """Return the ground-truth estimand when defined."""
        raise NotImplementedError
```

Important extension:

- `true_parameter()` or `truth()` should be part of the DGP contract for simulation studies. Without it, the evaluator cannot compute bias or coverage generically.

Possible return object for `sample()`:

```python
@dataclass
class SampledData:
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    metadata: dict
```

This is better than raw tuples because later models may include:

- instruments `Z`,
- censoring indicators,
- train/test partitions,
- latent truth for diagnostics.

### B. Estimator

Responsibilities:

- store algorithm hyperparameters,
- fit on a sampled dataset,
- expose fitted estimates and optional predictions,
- report diagnostics useful for evaluation.

Recommended interface:

```python
class Estimator(ABC):
    def __init__(self, name: str, hyper_parameters: dict):
        self.name = name
        self.hyper_parameters = hyper_parameters
        self.est_params = None

    @abstractmethod
    def fit(self, data):
        """Fit the estimator and populate est_params."""

    @abstractmethod
    def predict(self, data):
        """Return predictions on new data."""

    def summary(self):
        return self.est_params
```

Important extension:

`fit()` should return a structured estimate object rather than only mutating internal state. For example:

```python
@dataclass
class EstimateResult:
    target: str
    estimate: float
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    diagnostics: dict
```

This is useful because not every estimator is naturally prediction-oriented. For PLM, the central object is often the estimate of a scalar target like `theta_0`, not just predictions.

Suggested contract:

- `fit(data)` returns `EstimateResult` and stores it in `self.est_params`,
- `predict(data)` is optional for estimators where prediction is meaningful,
- `summary()` exposes the fitted object in a common shape.

### C. Evaluator

Responsibilities:

- define the simulation design,
- control seeds and repeated trials,
- instantiate DGPs and estimators,
- run experiments,
- compute metrics,
- save results and metadata.

Recommended interface:

```python
class Evaluator(ABC):
    def __init__(
        self,
        dgp,
        dgp_param_grid,
        estimators,
        n_trials,
        seed,
    ):
        ...

    @abstractmethod
    def run(self):
        """Execute the full study."""

    @abstractmethod
    def save(self, path):
        """Persist results and metadata."""
```

The evaluator should produce one row per:

- DGP setting,
- trial,
- estimator.

That makes aggregation straightforward later.

## Supporting Objects

To keep the base classes small, add a few lightweight records.

### DGPConfig

Stores model parameters such as:

- `n`,
- `p`,
- signal strength,
- treatment noise level,
- outcome noise level,
- sparsity or smoothness settings.

### EstimatorConfig

Stores:

- algorithm name,
- model hyperparameters,
- training hyperparameters,
- nuisance architecture settings,
- inference options.

### TrialRecord

One simulation run should log:

- trial id,
- seed,
- DGP name,
- DGP parameters,
- estimator name,
- estimator hyperparameters,
- true parameter,
- estimate,
- bias,
- squared error,
- confidence interval,
- covered or not,
- runtime,
- diagnostics.

### ExperimentResult

Container with:

- raw per-trial records,
- aggregated summaries,
- experiment metadata,
- software version info if needed later.

## The First Concrete DGP: Partial Linear Model

The PLM can be represented as:

```text
Y = theta_0 T + g_0(X) + epsilon
T = m_0(X) + v
```

with:

- `X in R^p`,
- `E[epsilon | X, T] = 0`,
- `E[v | X] = 0`.

The `PartialLinearDGP` should decide:

1. How `X` is generated.
2. How `m_0(X)` is specified.
3. How `g_0(X)` is specified.
4. What the true target `theta_0` is.
5. The noise distributions for `v` and `epsilon`.

Recommended constructor parameters:

- `theta0`,
- `x_dim`,
- `x_distribution`,
- `m_spec`,
- `g_spec`,
- `t_noise_scale`,
- `y_noise_scale`,
- `correlation_structure`,
- `standardize`.

Recommended methods:

- `sample(n, seed=None)`,
- `true_parameter() -> float`,
- optional helper methods `_sample_x`, `_m0`, `_g0`.

Important design choice:

`m_spec` and `g_spec` should not be hard-coded as one formula each. They should support either:

- string labels like `"linear"`, `"nonlinear_sparse"`, `"additive_sine"`,
- or callables.

That gives us a path to future studies without rewriting the class.

## The First Concrete Estimators

### 1. Neural-network DML estimator

The `DMLNeuralNetEstimator` should encapsulate:

- nuisance model classes for `m_0(X)` and `g_0(X)`,
- cross-fitting or sample splitting logic,
- final orthogonal-score regression,
- optional variance estimation and confidence intervals.

Important hyperparameters:

- number of folds,
- network width and depth,
- optimizer,
- epochs,
- batch size,
- learning rate,
- early stopping settings,
- device,
- random seed.

Implementation note:

Keep the nuisance learner separate from the causal estimator logic. For example, a helper wrapper around a neural net regressor can be reused later by other estimators.

### 2. Proposed estimator

The paper-specific estimator should be implemented behind the same `Estimator` interface. The class should expose:

- the scalar target estimate,
- standard error if available,
- diagnostics needed to compare against DML.

Important design goal:

The evaluator should not need special logic for the proposed estimator. If the estimator contract is clean, the evaluator can remain generic.

## Fair Comparison Principles for the Evaluator

To compare the proposed estimator against DML fairly, the evaluator should enforce:

### Shared data per trial

Within each trial and DGP setting, every estimator should use the exact same sampled dataset.

### Controlled randomness

Estimator-level randomness should be recorded and controlled. We should distinguish:

- the seed used to generate the dataset,
- the seed used inside estimator training.

### Comparable tuning budget

If one estimator uses hyperparameter tuning, the baseline should get a comparable tuning opportunity. Otherwise runtime and accuracy comparisons are hard to interpret.

### Common target and metrics

All estimators should be evaluated on the same estimand and report:

- point estimate,
- bias,
- variance,
- mean squared error,
- confidence interval coverage,
- average interval length,
- runtime.

### Consistent sample splitting

For methods that depend on sample splitting or cross-fitting, the split policy should be explicit and reproducible.

### Failure handling

If an estimator fails on a trial, the evaluator should:

- record the failure,
- keep the trial in the output,
- avoid silently dropping rows.

This matters when methods differ in stability.

## Experiment Runner Design

The evaluator can be factored into three stages:

### 1. Design expansion

Expand a grid of DGP settings into concrete experiment cells.

Example grid:

```python
{
    "n": [500, 1000, 2000],
    "x_dim": [10, 50],
    "m_spec": ["linear", "nonlinear_sparse"],
    "g_spec": ["linear", "additive_sine"],
}
```

### 2. Trial execution

For each experiment cell and trial:

1. derive a trial seed,
2. sample data from the DGP,
3. fit each estimator on the same sample,
4. collect estimate objects and diagnostics,
5. compute metrics against truth,
6. append one record per estimator.

### 3. Result persistence

Save:

- raw records as CSV or parquet,
- aggregate summaries as CSV or parquet,
- experiment metadata as JSON.

Recommended initial output layout:

```text
outputs/
  plm_study/
    raw_results.parquet
    summary_results.parquet
    metadata.json
```

Parquet is preferable once dependencies are in place because diagnostics columns and repeated runs become easier to manage.

## Suggested Result Schema

One row per estimator per trial:

```text
study_name
dgp_name
trial_id
data_seed
estimator_seed
n
x_dim
m_spec
g_spec
theta_true
theta_hat
std_error
ci_lower
ci_upper
bias
squared_error
covered
runtime_sec
fit_status
fit_message
estimator_name
estimator_config
dgp_config
diagnostics
```

This schema is generic enough for future simulation studies.

## Recommended Testing Strategy

The testing framework should distinguish contract tests, model-specific tests, and smoke tests.

### A. Contract tests for every new DGP

Every new DGP should satisfy:

1. `sample(n)` returns the required fields.
2. returned arrays have correct shapes.
3. repeated sampling with the same seed is deterministic.
4. different seeds give different samples.
5. `true_parameter()` returns the correct type and value.

For PLM specifically:

- verify `X.shape == (n, p)`,
- verify `T.shape == (n,)` or `(n, 1)` consistently,
- verify `Y.shape == (n,)` or `(n, 1)` consistently,
- verify `theta0` matches `true_parameter()`.

### B. Contract tests for every new estimator

Every estimator should satisfy:

1. `fit(data)` runs on a small valid sample,
2. `fit(data)` sets `est_params`,
3. returned estimate object has required fields,
4. `predict(data)` returns outputs in the expected shape when prediction is supported,
5. fixed seeds give reproducible results up to a chosen tolerance.

For inference-capable estimators:

- confidence interval fields should be present,
- `ci_lower <= estimate <= ci_upper` is not required in every run, but interval ordering should be valid.

### C. Integration smoke tests

Have a very small end-to-end test that runs:

- one PLM DGP setting,
- two estimators,
- two or three trials.

This test should verify:

- the evaluator runs without crashing,
- output rows have the expected count,
- both estimators appear in the results,
- metric columns are populated.

### D. Statistical sanity tests

These should be lightweight and tolerant, not strict asymptotic proofs.

Examples:

- under a very simple linear PLM with low noise, both estimators should recover `theta0` within a loose tolerance on average,
- increasing sample size should weakly reduce average squared error in a controlled benchmark,
- identical seeds should reproduce identical evaluator outputs.

These tests should avoid being flaky. Use moderate tolerances and simple DGP settings.

## How to Test a Newly Created DGP or Estimator

When we add a new DGP:

1. write a unit test for the sample contract,
2. write a determinism test for seed behavior,
3. write a truth test for `true_parameter()`,
4. add the DGP to one integration smoke test if it is intended for active use.

When we add a new estimator:

1. test `fit()` on a toy dataset,
2. test the shape and presence of its output fields,
3. test reproducibility under fixed seeds,
4. run one evaluator smoke test against a simple DGP,
5. if the estimator uses optimization, add a very small fast configuration for tests.

This means every new component should come with:

- one fast unit test file,
- at least one end-to-end smoke test path,
- a minimal example configuration.

## Practical Notes for Implementation

### Use dataclasses for records

Dataclasses make the interfaces readable and reduce dictionary-shaped bugs.

### Keep heavy ML dependencies behind estimator modules

The core framework should not depend on PyTorch or TensorFlow directly. Only the relevant estimator module should.

### Make experiment configs serializable

Avoid storing non-serializable objects in top-level configs if we want clean metadata output. For callable DGP components, store a human-readable label alongside the callable.

### Decide shape conventions early

Pick one convention and stick to it:

- vectors as `(n,)`, or
- vectors as `(n, 1)`.

A lot of estimator bugs come from inconsistent shape handling.

### Use a common metric function

Metrics like bias and coverage should be computed centrally in `evaluation/metrics.py`, not inside each estimator.

## Immediate Implementation Roadmap

I would build the package in this order:

1. core typed records: `SampledData`, `EstimateResult`, `TrialRecord`,
2. abstract base classes for DGP and estimator,
3. generic experiment runner and metric utilities,
4. `PartialLinearDGP`,
5. a simple reference estimator first, possibly a partially linear oracle or low-complexity plug-in baseline,
6. `DMLNeuralNetEstimator`,
7. the paper's proposed estimator,
8. smoke tests and a reproducible study script.

One practical reason to include a simple reference estimator before neural DML is that it helps debug the evaluator and DGP before introducing optimization-heavy learners.

## Recommended First Milestone

Before implementing the paper method, the first milestone should be:

- one working `PartialLinearDGP`,
- one very simple baseline estimator,
- one evaluator that runs a tiny study,
- one saved results table,
- one `pytest` smoke test.

If that path is stable, we can add DML and the proposed estimator with much lower integration risk.

## Open Design Choices To Resolve Before Coding

These do not block the outline, but we should settle them when we start implementation:

1. Should `params` and `hyper_parameters` be plain dictionaries or typed dataclasses?
2. What numeric stack do we want for the base package: numpy only, or numpy plus pandas from the start?
3. What deep-learning backend do we want for neural DML?
4. Do we want the evaluator to run serially first, or include parallel trial execution from the beginning?
5. What shape convention do we want for one-dimensional outputs?

My recommendation is:

- dataclasses for internal configs,
- numpy plus pandas initially,
- PyTorch only inside the neural estimator module,
- serial evaluator first,
- vector convention `(n,)` unless a backend strongly prefers `(n, 1)`.
