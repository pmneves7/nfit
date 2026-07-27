# Modeling and fitting pipeline

The fitting framework is organized around simultaneous fits to one or more
reduced datasets. The implementation is intentionally modular: importers adapt
instrument/file conventions into common containers, masking and rebinning are
dataset-local preprocessing steps, the physics model is shared across the fit,
each dataset may apply its own instrument resolution model, and the optimizer
minimizes one concatenated weighted residual vector.

## Core objects

`PointData4D`
: Flattened measured data containing `H`, `K`, `L`, energy transfer `E`,
  intensity, uncertainty `sigma`, an analysis mask, optional temperature, and
  metadata. It is not tied to a particular spectrometer; any reduced data source
  that can be expressed in these coordinates can use it.

`ParameterSpec`
: A named scalar parameter with an initial value, optional bounds, fixed/varying
  state, units, and a description. Fixed parameters are passed to models but are
  not included in the optimizer vector.

`FitDataset`
: One measured dataset in the global objective. It stores the data, a
  chi-squared weight, an ordered list of preprocessing transforms, optional
  dataset-specific resolution, and metadata such as instrument or scan settings.
  At the project-dataset level, an enabled dataset with zero fit weight is
  visualization-only: it is excluded before fit preparation and evaluated once
  after optimization for model/residual plotting.

`ModelSpec`
: A callable physics model plus descriptive metadata. The callable receives
  prepared point data and a parameter dictionary, and returns predicted measured
  intensity at each point. It can wrap a phenomenological formula, an ab initio
  calculation, a Monte Carlo model, or any later backend with the same interface.

`ResolutionSpec`
: Optional instrument-resolution hook. It receives data coordinates,
  unconvolved model values, and parameters, then returns resolution-corrected
  measured intensity. `identity_resolution` is the no-op used when
  resolution effects are disabled.

`FitProblem`
: The complete simultaneous-fitting definition: datasets, shared model,
  parameter specifications, and metadata.

`DatasetEntry`
: One raw dataset plus flexible experiment metadata. This wrapper is intentionally
  general: it can hold neutron point data, MDHisto data, susceptibility curves,
  powder data, or other point and histogram containers. Dataset entries may also
  carry dataset-local transforms such as masks, cuts, normalization corrections,
  or rebinning operations.

`DataGroup`
: A named collection of related datasets. A group can store sample-level context
  such as lattice parameters, space group, provenance, and any number of
  associated model sessions. This is the preferred container for field series,
  temperature series, multiple cuts from one experiment, or data collected on
  different instruments but fit together.

`FitModelSession`
: A model attached to a data group. It stores the physics model, current
  parameter specifications, optimizer settings, per-dataset resolution
  functions, parameter bindings, dataset weights, and an ordered history of
  completed fits. Each fit can update the current parameters, and previous
  history entries can be restored with `rollback`.

```python
from nfit import (
    DataGroup,
    DatasetEntry,
    FitModelSession,
    ParameterSpec,
    make_constant_intensity_model,
)

group = DataGroup(
    "field_series",
    lattice_parameters={"a": 3.8, "b": 3.8, "c": 12.0},
    spacegroup="P4/mmm",
    datasets=[
        DatasetEntry("0T", data_0t, kind="inelastic_neutron", parameters={"field": 0.0}),
        DatasetEntry("9T", data_9t, kind="inelastic_neutron", parameters={"field": 9.0}),
    ],
)

session = FitModelSession(
    "background_by_field",
    model=make_constant_intensity_model("constant"),
    parameter_specs=[
        ParameterSpec("background_0T", 0.1),
        ParameterSpec("background_9T", 0.1),
    ],
    parameter_bindings_by_dataset={
        "0T": {"constant": "background_0T"},
        "9T": {"constant": "background_9T"},
    },
)

group.add_model(session)
result = session.fit(group)
session.rollback(0)
```

## GUI and script parity

Scripts and the GUI use the same project, model, mask, fit, and resolution
objects. A workflow created in either interface can therefore be inspected,
saved, and reproduced in the other.

The current project explorer launches with `nfit` and lets a user:

- create workspaces and import reduced datasets from files,
- organize datasets into nested dataset groups,
- inspect file-provided axes, units, metadata, inferred axis roles, crystal
  information, source files, and imported sample-environment context,
- edit dataset configuration such as point-list coordinate/channel roles and
  rebinned views,
- edit and attach dataset or shared masks,
- attach one or more model components,
- configure optimizer settings, dataset weights, parameter bounds, fitted/fixed
  parameters, global/per-dataset parameter sharing, and hard parameter
  relationships,
- run fits, branch fit timelines, restore earlier fit states, and inspect
  structured fit metadata,
- compare data, model, and residual channels visually in the data viewer.

For workflows with a small symmetry-reduced fitting volume and a much larger
display volume, set the display dataset's fit weight to zero. Positive-weight
datasets alone define the objective and degrees of freedom. Once their fitted
parameters have been written back, the display dataset is prepared and its
model channel is calculated exactly once. The lower-level `FitModelSession`
likewise omits zero-weight entries from its `FitProblem`; post-fit visualization
channel generation is provided by the project fitting pipeline.

The data viewer remains driven by reusable data/model objects. It can display
current model channels and stored fit channels from GUI fit history, but
script-created datasets with attached fit comparisons produce the same viewer
behavior.
See [GUI workflows](gui_workflows.md) for current user-facing details.
See [Fit constraints](fit_constraints.md) for exact relationships,
inequalities, expression syntax, degrees of freedom, and backend examples.

## Masking

The analysis mask follows the `PointData4D` convention that `True` means a point
is still eligible for analysis. Masking helpers return new data objects and
combine with existing masks, so independent masks can be composed in a
`FitDataset` transform list.

```python
from nfit import FitDataset, make_energy_q_mask_transform, make_mask_transform

dataset = FitDataset(
    "hyspec_50K",
    data,
    transforms=[
        # Keep only this coarse analysis window.
        make_mask_transform(E=(2.0, 40.0), H=(0.25, 0.75)),
        # Exclude a known background band.
        make_energy_q_mask_transform(energy=(8.0, 11.0)),
    ],
)
```

### `|Q|` masks

`|Q|` masks are evaluated in inverse angstroms. They work directly if the point
data metadata says the momentum coordinates are already inverse angstroms, for
example `metadata={"coordinate_units": "angstrom^-1"}`. For RLU data, attach a
lattice or UB-style matrix first:

```python
from nfit import attach_lattice_parameters, make_energy_q_mask_transform

data = attach_lattice_parameters(
    data,
    a=3.8,
    b=3.8,
    c=12.4,
    alpha=90.0,
    beta=90.0,
    gamma=90.0,
)

mask_transform = make_energy_q_mask_transform(
    energy=(12.0, 18.0),
    q_modulus=(1.5, 2.2),
)
```

`attach_ub_matrix(data, ub_matrix)` is also available when a sample UB or other
HKL-to-`Q` matrix is known. The Mantid MDHisto NeXus importer looks for oriented
lattice metadata such as
`/MDHistoWorkspace/experiment99/sample/oriented_lattice/orientation_matrix` and
stores it under `metadata["oriented_lattice"]`.

### Projected box and ellipsoid masks

Use projected box or ellipsoid masks for regions such as phonon branches,
spurious sample-environment features, glue, or mount scattering. Dimensions may
be coordinate names (`"H"`, `"K"`, `"L"`, `"E"`) or projection vectors. A vector
with three entries projects `(H,K,L)`; a vector with four entries projects
`(H,K,L,E)`.

```python
from nfit import make_box_mask_transform, make_ellipsoid_mask_transform

box_mask = make_box_mask_transform(
    dimensions=[(1.0, 1.0, 0.0), "E"],
    center=[1.0, 5.0],
    half_widths=[0.08, 0.75],
)

ellipsoid_mask = make_ellipsoid_mask_transform(
    dimensions=["H", "K", "E"],
    center=[0.5, 0.5, 12.0],
    radii=[0.04, 0.04, 2.0],
)
```

A 1D, 2D, or 3D mask ignores the other coordinates. For example, the 2D box
above masks every point whose projected `(H+K, E)` coordinates fall inside the
box, regardless of `L`.

### Phonon cone masks

`make_phonon_mask_transform` masks a sphere in reciprocal space whose radius
grows linearly with energy, centered at a specified reciprocal-space point:

```python
from nfit import make_phonon_mask_transform

phonon_mask = make_phonon_mask_transform(
    center=[0.0, 0.0, 0.0],
    slope=35.0,              # meV per inverse angstrom
    energy_range=(0.0, 28.0),
    radius_offset=0.03,
)
```

By default the radius uses `abs(E)` and the energy range is applied to `abs(E)`,
which is convenient for symmetric positive/negative energy-transfer masks. Use
`use_absolute_energy=False` or `energy_range_uses_absolute=False` when signed
energy transfer should matter.

## Rebinning

`rebin_point_data` and `make_rebin_transform` provide regular 4D rebinning for
`(H,K,L,E)` point data:

```python
from nfit import make_rebin_transform

dataset = FitDataset(
    "coarse_grid",
    data,
    transforms=[
        make_rebin_transform(
            lower=[0.0, 0.0, -1.0, 0.0],
            upper=[1.0, 1.0, 1.0, 80.0],
            num_bins=[20, 20, 10, 40],
        ),
    ],
)
```

Specialized scripted operations can use a custom transform with the same
signature:

```python
def my_transform(data: PointData4D) -> PointData4D:
    ...
```

## Simultaneous fitting

The deterministic optimizer currently uses weighted least squares. For each
dataset, residuals are

```text
sqrt(dataset.weight) * (observed - model) / sigma
```

and all dataset residuals are concatenated into one objective. This allows high
quality datasets, low quality datasets, and data from different instruments to
contribute with explicitly chosen relative weights.

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
        FitDataset("instrument_a", data_a, weight=1.0),
        FitDataset("instrument_b", data_b, weight=0.4),
    ],
    model=ModelSpec("placeholder_model", predict_measured_intensity),
    parameter_specs=[
        ParameterSpec("amplitude", 1.0, min=0.0),
        ParameterSpec("width", 0.1, min=0.0),
        ParameterSpec("background", 0.0),
    ],
)

result = fit_problem_least_squares(problem)
```

`FitResult` reports optimized parameters, covariance/stderr when the local
Jacobian supports it, global chi-squared values, and per-dataset chi-squared
contributions, sizes, weights, residuals, and model values.

`OptimizationConfig.kwargs` is passed to SciPy least squares for deterministic
fits. Common options include `max_nfev`, tolerances such as `xtol`, and robust
loss controls:

```python
from nfit import OptimizationConfig

result = fit_problem_least_squares(
    problem,
    config=OptimizationConfig(
        kwargs={
            "loss": "soft_l1",
            "f_scale": 1.0,
        }
    ),
)
```

The `loss` option chooses the cost function: the rule that converts residuals
into the scalar objective minimized by least squares. Nfit residuals are
normally normalized by the data uncertainty, so a residual of `1` means the
model is about one standard deviation away from that point. `f_scale` sets the
residual size where robust losses begin treating a point as large; `f_scale=1.0`
means "start down-weighting around one sigma." Larger `f_scale` values make a
robust loss behave more like ordinary least squares, while smaller values make
down-weighting begin earlier.

| Loss | What it does | When to use it |
| --- | --- | --- |
| `linear` | Ordinary chi-squared minimization. Residuals are squared, so a point with residual `10` has one hundred times the cost of a point with residual `1`. | Use for clean data, reliable uncertainties, and final reported fits where standard least-squares errors should be interpretable. |
| `soft_l1` | A smooth robust loss. It behaves like chi-squared for small residuals, but large residuals grow more slowly. | A good first robust option for imperfect masks, detector artifacts, spurions, or a few outlier pixels. It reduces outlier leverage without being too abrupt. |
| `huber` | Chi-squared for small residuals, then roughly linear growth once residuals exceed `f_scale`. | Use when most points should still be trusted but occasional large residuals should not dominate the fit. This is a conservative, easy-to-explain robust choice. |
| `cauchy` | Strongly suppresses the influence of large residuals. Very bad points add only slowly increasing cost. | Useful for exploratory fits with obvious outliers or model-mismatch regions. Check residual maps carefully, because it can make a poor model look deceptively stable. |
| `arctan` | Very aggressive down-weighting; extremely large residuals contribute almost a capped cost. | Reserve for diagnostics or difficult convergence problems with severe outliers. If it materially changes the answer, improve the mask/model and repeat with a less aggressive loss. |

Robust losses are practical safety valves, not substitutes for understanding the
data. For publication-quality parameter values and uncertainty estimates,
ordinary chi-squared (`linear`) remains the clearest statistical objective when
the error model is credible.

For nonlinear fits with uncertain initial guesses, the least-squares start
point can be initialized by differential evolution:

```python
result = fit_problem_least_squares(
    problem,
    config=OptimizationConfig(
        kwargs={
            "initialization": {
                "method": "differential_evolution",
                "maxiter": 60,
                "popsize": 10,
                "seed": 123,
            }
        }
    ),
)
```

Differential evolution searches the bounded variable-parameter space before
least-squares polishing. It is opt-in, slower than a local fit, and requires
finite lower and upper bounds on every fitted parameter.

Posterior sampling uses `emcee` through `sample_problem_parameters`. The usual
workflow is to run least squares first, then initialize walkers around the
best-fit parameters:

```python
from nfit import SamplerConfig, sample_problem_parameters

fit = fit_problem_least_squares(problem)
posterior = sample_problem_parameters(
    problem,
    SamplerConfig(n_walkers=48, n_steps=2000, burn_in=500, random_seed=123),
    initial_params=fit.params,
)
```

The sampler uses the same weighted residual vector as a Gaussian log
likelihood and uniform priors implied by parameter bounds. It returns samples,
variable names, log probabilities, and compact diagnostics such as acceptance
fractions and estimated autocorrelation time when available.

### Why an emcee step is expensive

`emcee` is an *ensemble* sampler: every step advances all `n_walkers` walkers
at once, so a single step evaluates the model **once per walker** — `n_walkers`
likelihood evaluations. Compare this with one least-squares **residual
evaluation**, not one optimizer iteration: a least-squares iteration may need
multiple residual evaluations for a Jacobian or line search. The useful cost
estimate is therefore `n_walkers × n_steps` model evaluations, plus burn-in,
versus the optimizer's reported function-evaluation count.

When `Walkers` is left at 0 (or `SamplerConfig(n_walkers=None)`), the walker
count defaults to `max(32, 2 * n_parameters + 2)`, i.e. **32** for a typical
handful of parameters. Practical consequences:

- **Fewer walkers make each step faster but not the run cheaper.** The total
  work for a given effective sample size is roughly `n_walkers * n_steps`
  evaluations either way, and the stretch move mixes better with more walkers
  (the minimum is `2 * n_parameters`). Treat `Walkers` as a granularity knob,
  not a speed knob.
- **The analytic Jacobian does not help MCMC.** Sampling needs only the
  likelihood (the residual), not its gradient — so the Jacobian that accelerates
  least squares is unused here. The per-evaluation cost still benefits from the
  compute backend (see [Performance notes](performance.md)) and the cached
  per-dataset geometry.
- **Parallelizing across walkers is the one per-step lever, with a caveat.** The
  walker evaluations in a step are independent and can be spread over a pool
  (`SamplerConfig(kwargs={"workers": N})` / the GUI "emcee workers" field). But
  each evaluation already uses all cores through the numba/threaded backends, so
  raising emcee workers *and* leaving the backend multithreaded oversubscribes.
  To use it, pin the backend to one thread (`nfit.set_num_threads(1)`) and set
  emcee workers to the core count; on a single machine that is roughly a wash
  for large datasets, and it pays off mainly across cluster nodes.

MCMC is normally much slower than least squares. Start from the least-squares
solution, monitor convergence and autocorrelation, and choose burn-in and
production length from those diagnostics rather than a fixed wall-time rule.

## Primitive and compound models

The simplest measured-intensity model is constant in momentum and energy:

```python
from nfit import make_constant_intensity_model

background = make_constant_intensity_model("background")
```

Multiple primitive models can be added together with
`compound_additive_model`. Each component receives the same prepared data and
parameter dictionary, and the returned measured intensities are summed:

```python
from nfit import compound_additive_model, make_constant_intensity_model

model = compound_additive_model(
    make_constant_intensity_model("background"),
    more_complicated_model,
)
```

## Registered model components and dynamic parameters

GUI-facing model components (`ModelComponentSpec`) are compiled into a
`FitProblem` through `compile_fit_problem`, which looks each component type up
in `MODEL_TYPE_REGISTRY` (`nfit.fit_config`). A registration
(`ModelTypeInfo`) carries the static parameter names, the compatible dataset
data types, and a factory that closes over the component and returns the model
callable.

Some models derive additional parameter names from their configuration: the
`heisenberg_rpa` component emits one exchange parameter per bond orbit stored
in `config["orbits"]`. Registrations declare this with the optional
`dynamic_parameters` callable, and `component_parameter_names(component)`
returns the full static-plus-dynamic list. Dynamic parameters flow through
sharing modes, limits, and constraints exactly like static ones.

Components snapshot the structured configuration they need (crystal, expanded
`site_positions`, bond `orbits`) into `component.config` as plain JSON data.
The `DataGroup` remains the sample-level record (`lattice_parameters`,
`spacegroup`, `metadata["crystal"]`) used to stamp fit points with the
RLU-to-inverse-angstrom matrix, but a component stays self-contained so it
survives project save/load and can be scripted directly. See
[Spin-fluctuation models](spin_fluctuation_models.md) for the physics and the
orbit-generation API.

### Analytic Jacobians

A registration may also supply an optional `jacobian_factory`, mirroring
`factory`: it closes over the component and returns a callable
`(data, params) -> {qualified_name: d(model)/d(param)}` giving exact model
gradients keyed by the same qualified names the model reads. `compile_fit_problem`
attaches a combined `model_jacobian` to a `FitDataset` only when **every**
component applied to that dataset provides one (background models and
`heisenberg_rpa` do). `fit_problem_least_squares` then hands the assembled
residual Jacobian to SciPy (`problem_supports_analytic_jacobian` gates this,
additionally requiring no instrument resolution, whose convolution the bare
model gradient does not carry); otherwise it silently falls back to SciPy's
finite-difference Jacobian. The assembler maps each model column onto optimizer
variables through the same `parameter_bindings` and derived-parameter chain rule
the residual uses, so shared parameters and reparameterized inequality
constraints are handled automatically.

## Parameter bindings and constraints

`ParameterSpec` defines the global optimizer parameters. `FitDataset` can map
local model or resolution parameter names onto those global names using
`parameter_bindings`. This gives a simple way to express constraints such as
"datasets A and B share one constant background, while dataset C has another":

```python
problem = FitProblem(
    datasets=[
        FitDataset("A", data_a, parameter_bindings={"constant": "background_ab"}),
        FitDataset("B", data_b, parameter_bindings={"constant": "background_ab"}),
        FitDataset("C", data_c, parameter_bindings={"constant": "background_c"}),
    ],
    model=make_constant_intensity_model("constant"),
    parameter_specs=[
        ParameterSpec("background_ab", 0.1),
        ParameterSpec("background_c", 0.2),
    ],
)
```

Bindings may also be fixed scalar values, which is useful for holding a local
normalization or resolution setting fixed without adding a global parameter.

## Energy resolution

The first concrete resolution model is Gaussian broadening in energy. It
evaluates the physics model on an oversampled energy grid for each unique
`(H,K,L)` trajectory, then convolves the dense model back to the measured energy
points.

```python
from nfit import FitDataset, constant_fwhm_energy_resolution

dataset = FitDataset(
    "instrument_a",
    data,
    resolution=constant_fwhm_energy_resolution("resolution_fwhm", oversampling=5),
)
```

The FWHM may be fitted like any other parameter:

```python
ParameterSpec("resolution_fwhm", 1.2, min=0.05, unit="meV")
```

Polynomial FWHM is also available:

```python
from nfit import polynomial_fwhm_energy_resolution

resolution = polynomial_fwhm_energy_resolution(
    ["fwhm0", "fwhm1", "fwhm2"],
    oversampling=7,
)
```

The coefficients are ordered as `FWHM(E) = c0 + c1*E + c2*E**2 + ...`, and each
coefficient can be either a fixed number or a fitted parameter name. The FWHM is
required to remain finite and positive at all evaluated energies.

## Uncertainty sampling

`SamplerConfig`, `SamplingResult`, and `sample_problem_parameters` provide
posterior sampling with an `emcee` ensemble backend over the same `FitProblem`
and parameter specifications used by deterministic optimization. See the
posterior-sampling workflow above and, in particular,
[Why an emcee step is expensive](#why-an-emcee-step-is-expensive).
