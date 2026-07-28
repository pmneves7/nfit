# Modeling and fitting pipeline

This page describes the developer-facing path from imported data to a fitted
model. User controls are documented in [Models and fitting](gui_fitting.md).

## Data flow

For each effective dataset, nfit:

1. loads or constructs a common data container;
2. applies unit and channel transforms;
3. applies file, group, and dataset masks;
4. performs any configured rebinning or compositing;
5. converts the result to fit points;
6. evaluates the shared model and dataset resolution; and
7. concatenates weighted residuals across datasets.

Import and plotting conventions do not belong inside physics kernels. A model
receives physical coordinates and metadata and returns a prediction in the
dataset's selected fit representation.

## Core objects

`PointData4D`
: Flattened fit points containing momentum, energy, observation, uncertainty,
  mask, optional temperature and field, and metadata.

`PointListData`
: Named coordinate and channel columns used for magnetization, heat capacity,
  powder diffraction, and other tabular data.

`MDHistoData`
: A regular multidimensional histogram with axes, signal, uncertainty, mask,
  event counts, and metadata.

`DatasetEntry`
: Project wrapper around a data container or lazy file reference. It stores the
  data type, parameters, transforms, scale, fit weight, and provenance. Install
  modified data with `replace_data`; its revision invalidates dependent caches.

`DataGroup`
: A workspace containing datasets, nested dataset groups, masks, backgrounds,
  model components, fit history, analyses, and plots.

`ParameterSpec`
: One optimizer parameter with a value, bounds, varied/fixed state, unit, and
  description.

`FitDataset`
: One prepared dataset in a global objective, including its statistical weight,
  parameter bindings, transforms, and optional resolution.

`ModelSpec`
: A callable measured-response model plus metadata and optional analytic
  Jacobian.

`FitProblem`
: The datasets, model, parameters, constraints, and optimizer metadata for one
  simultaneous fit.

`FitResult`
: Optimized values, residuals, predictions, goodness-of-fit values, covariance,
  and per-dataset contributions.

The project GUI compiles serializable `ModelComponentSpec` entries into these
lower-level objects. Project-aware scripts load the same component state and
use the same compilation path as the GUI. Developers can also construct a
`ModelSpec` and `FitProblem` directly for callable models that do not need GUI
or project serialization.

The common containers expose read-only numerical arrays. Use `with_updates` for
one replacement or `mutable_copy` followed by `DatasetEntry.replace_data` for
several edits. See [Data and extension conventions](data_philosophy.md#data-container-contract).

## Preparing data

### Masks

`PointData4D.mask` uses `True` for a point that remains eligible. Mask helpers
return new data objects, so transforms compose without mutating their inputs:

```python
from nfit import FitDataset, make_energy_q_mask_transform, make_mask_transform

dataset = FitDataset(
    "cut_50K",
    points,
    transforms=[
        make_mask_transform(E=(2.0, 40.0), H=(0.25, 0.75)),
        make_energy_q_mask_transform(energy=(8.0, 11.0)),
    ],
)
```

$|Q|$ masks require inverse-angstrom coordinates or lattice/UB metadata.
Projected boxes and ellipsoids use a complete coordinate basis. Phonon-cone
masks use physical HKL centers and an energy-versus-$|Q-Q_0|$ boundary.

The complete project-level semantics are in
[Importing and preparing data](data_import.md#masks).

### Rebinning

`rebin_nd` bins in-memory arrays. `rebin_nd_stream` accepts repeatable batches
from files or custom providers. Both support explicit bounds, bin counts or
step sizes, fractional overlap, symmetry expansion, and bounded temporary
memory.

Project rebinning applies masks first, then returns a density-valued normalized
histogram. Composite datasets additionally apply source scales and fit weights.
See [API reference](api.md#n-dimensional-rebinning).

## Simultaneous least squares

For dataset $d$, the residual is

$$
r_d=\sqrt{w_d}\,
\frac{a_dy_d-f_d(\mathbf p)}{a_d\sigma_d}.
$$

$y_d$ and $\sigma_d$ are the measured value and its one-sigma uncertainty,
$f_d(\mathbf p)$ is the model prediction at parameter vector $\mathbf p$, and
$w_d$ is the dataset fit weight. The positive dataset scale $a_d$ is an
experimental calibration based on quantities such as sample amount and
incident flux. A fixed scale is applied during data preparation; a fitted scale
is applied by the residual evaluator. Normalized data use $a_d=1$. Several
datasets may name one `FitDatasetInput.scale_group`, which compiles one shared
scale parameter and writes its result back to every member.

All $r_d$ are concatenated. Dataset weights therefore affect the objective but
not the stored observations or uncertainties. They may be used to give datasets
with very different point counts comparable influence. Covariance and posterior
uncertainties are conditional on these user-selected weights.

`OptimizationConfig(covariance_mode="absolute")` is the default and treats
$\sigma_d$ as absolute one-sigma uncertainty. Use
`covariance_mode="residual"` to multiply the local covariance by the reduced
chi-squared. Residual scaling estimates one missing global noise scale from all
residuals; it cannot separate underestimated counting errors from systematics,
correlations, outliers, or model inadequacy.

```python
from nfit import (
    FitDataset,
    FitProblem,
    ModelSpec,
    ParameterSpec,
    fit_problem_least_squares,
)

problem = FitProblem(
    datasets=[
        FitDataset("run_a", data_a, weight=1.0),
        FitDataset("run_b", data_b, weight=0.5),
    ],
    model=ModelSpec("response", predict_measured_intensity),
    parameter_specs=[
        ParameterSpec("amplitude", 1.0, min=0.0),
        ParameterSpec("width", 0.1, min=0.0),
        ParameterSpec("background", 0.0),
    ],
)

result = fit_problem_least_squares(problem)
```

An enabled project dataset with zero fit weight is omitted from preparation and
optimization. After the fit, the project pipeline evaluates it once for display.

### Loss functions

`OptimizationConfig.kwargs` is passed to SciPy least squares. The common loss
choices are:

| Loss | Behavior |
| --- | --- |
| `linear` | Ordinary chi-squared |
| `soft_l1` | Smoothly reduces the influence of large residuals |
| `huber` | Quadratic near zero and approximately linear beyond `f_scale` |
| `cauchy` | Strong outlier suppression |
| `arctan` | Aggressive, nearly capped outlier cost |

With uncertainty-normalized residuals, `f_scale=1` places the robust transition
near one standard deviation. Robust losses are diagnostic tools; ordinary
chi-squared is the clearest final objective when the uncertainty model is
credible.

### Initialization

Differential evolution can search finite parameter bounds before local least
squares:

```python
from nfit import OptimizationConfig

config = OptimizationConfig(
    kwargs={
        "initialization": {
            "method": "differential_evolution",
            "maxiter": 60,
            "popsize": 10,
            "seed": 123,
        }
    }
)
result = fit_problem_least_squares(problem, config=config)
```

It is useful when starting values are uncertain, but it requires finite bounds
for every varied parameter.

## Parameters and constraints

Project model components compile parameters using qualified names such as
`rpa.J1`. Sharing modes produce one global value, one value per dataset, or one
value per named group. Dataset-local parameter bindings map the compiled name
back to the unqualified key expected by the model callable.

Exact constraints remove a dependent parameter from the optimizer.
Inequalities introduce a nonnegative offset. Constraint compilation, cycle
detection, bounds compatibility, and degrees-of-freedom rules are documented in
[Fit constraints](fit_constraints.md).

Component factories may create parameters dynamically from configuration. For
example, generating Heisenberg bond orbits creates `J1`, `J2`, and related
tensor parameters. Dynamic parameters must still provide defaults, bounds,
units, descriptions, and stable serialized names.

## Compound models

A project model sums enabled components. At the lower level,
`compound_additive_model` combines callables that accept the same fit points
and parameter dictionary:

```python
from nfit import compound_additive_model, make_constant_intensity_model

model = compound_additive_model(
    physical_model,
    make_constant_intensity_model("background"),
)
```

The component compiler performs the same addition while preserving qualified
parameter names and component diagnostics.

## Analytic Jacobians

A `ModelSpec` may supply derivatives of its prediction with respect to model
parameters. When every active component and resolution step supplies compatible
gradients, nfit passes an analytic Jacobian to least squares. Otherwise it uses
finite differences.

Background and scalar Heisenberg-RPA components provide analytic derivatives.
Tensor RPA and general resolution convolution currently use finite differences.
See [Heisenberg RPA](heisenberg_rpa.md#analytic-derivatives).

## Resolution

Resolution belongs to the dataset rather than the shared physics model.
`ResolutionSpec` receives coordinates, unconvolved predictions, and parameters,
then returns the measured prediction.

`identity_resolution` is the no-op. `EnergyGaussianResolution` and
`polynomial_fwhm_energy_resolution` provide Gaussian energy broadening, with
either fixed or fitted width parameters. Convolution groups points by momentum,
constructs a dense energy grid, evaluates the unconvolved model there, and
interpolates the broadened result back to measured energies.

General multidimensional resolution convolution is listed in
[Planned features](planned_features.md).

## Posterior sampling

`sample_problem_parameters` evaluates the same weighted residual vector as a
Gaussian log likelihood and uses parameter bounds as uniform priors:

```python
from nfit import SamplerConfig, sample_problem_parameters

posterior = sample_problem_parameters(
    problem,
    SamplerConfig(
        n_walkers=48,
        n_steps=2000,
        burn_in=500,
        random_seed=123,
    ),
    initial_params=result.params,
)
```

An ensemble step evaluates the model once per walker. Analytic least-squares
Jacobians do not accelerate those likelihood calls. Inspect acceptance,
autocorrelation, walker traces, and parameter correlations before using
credible intervals.

Raw chains may be reinterpreted with different burn-in and thinning or extended
from their final walker positions. GUI behavior is described in
[Models and fitting](gui_fitting.md#posterior-sampling).

## Extension contract

A new model or transform should:

1. consume common data objects rather than instrument files;
2. preserve or explicitly declare units and quantity types;
3. avoid mutating its inputs;
4. serialize every scientific choice;
5. provide stable parameter names and validation;
6. expose the same operation to scripts and the GUI where applicable; and
7. include tests, tooltips, and user documentation.

Registry and serialization details are in
[Data and extension conventions](data_philosophy.md). Planned tight-binding
and itinerant models also follow the layered response, provenance, and result
contracts in
[Electronic-response design contract](electronic_response_contract.md).
