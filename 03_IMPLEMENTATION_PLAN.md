# Implementation Plan and Milestones

## Milestone 0: Repository setup

Create a Python package with modern `pyproject.toml` packaging metadata and this
structure:

```text
chi4d/
  pyproject.toml
  README.md
  src/
    chi4d/
      __init__.py
      dataset.py
      coordinates.py
      models.py
      lattice_harmonics.py
      cross_section.py
      fitting.py
      uncertainty.py
      qfi.py
      visualization.py
  tests/
    test_models.py
    test_fitting.py
    test_cross_section.py
    test_qfi.py
  examples/
    00_synthetic_single_q_fit.ipynb
    01_synthetic_lattice_harmonic_fit.ipynb
    02_qfi_from_model.ipynb
    reference_cases/
      single_q_paramagnon/
      lattice_harmonic_shell/
      two_temperature_global_fit/
  docs/
    index.md
    getting_started.md
    physics_conventions.md
    model_reference.md
    approximation_notes.md
    references.md
```

Use pytest. Use ruff or black if desired. Do not require optional heavy dependencies for the minimal test suite.

Acceptance criteria:

- `pip install -e .` works.
- `pytest` runs.
- Package can be imported.
- Documentation source tree exists, even if initially minimal.
- Reference-example directory exists with a short README describing the intended validation role.
- Project metadata is compatible with an eventual PyPI release named `metallix`.

## Milestone 1: Core data containers

Implement:

- `PointData4D` dataclass for flattened fitting data.
- `from_arrays(...)` constructor.
- mask handling.
- validation of shapes.
- simple train/test or fit/holdout masking support.

Suggested dataclass:

```python
@dataclass
class PointData4D:
    H: np.ndarray
    K: np.ndarray
    L: np.ndarray
    E: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray | None = None
    temperature: float | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def valid(self) -> "PointData4D":
        ...
```

Acceptance criteria:

- bad shapes raise clear errors.
- `valid()` returns only masked valid points.
- sigma <= 0 is masked or raises a clear error depending on option.

## Milestone 2: Basic model functions

Implement:

- `single_q_paramagnon_chipp(H,K,L,E, params)`
- `multi_q_paramagnon_chipp(...)`
- `relaxational_chipp(chi_q, gamma_q, E)`
- `constant_background(...)`
- `linear_background(...)`

Use meV for E and Gamma.

Acceptance criteria:

- vectorized over arbitrary N.
- no NaN at E=0.
- tests verify limiting behavior.

## Milestone 3: Cross-section wrapper

Implement:

- Bose factor with stable behavior near zero energy.
- optional magnetic form factor placeholder.
- optional polarization factor placeholder.
- `intensity_from_chipp(...)`.

Formula:

I = scale * form_factor^2 * polarization * chipp / (1 - exp(-E/kBT)) + background

For E near 0, use numerically stable functions such as expm1.

Acceptance criteria:

- no NaN/inf near E=0 for physically meaningful inputs.
- can switch Bose correction on/off.
- unit tests for high-T and low-T limiting behavior.

## Milestone 4: Deterministic fitting

Implement:

- parameter vector packing/unpacking.
- residual function.
- `fit_least_squares(data, model, parameter_specs)`.
- fit result object containing best values, covariance if available, chi2, reduced chi2, residuals.

Acceptance criteria:

- can fit synthetic single-Q data.
- known parameters recovered within tolerance.
- covariance/errors reported when well-conditioned.
- parameter bounds respected.

## Milestone 5: Lattice harmonics

Implement a general lattice-harmonic system.

Start with a generic neighbor-shell Fourier transform:

Lambda(Q) = sum_n K_n gamma_n(Q)

gamma_n(Q) = sum_{d in shell n} cos(2 pi [H d_x + K d_y + L d_z])

where d is in fractional lattice-vector coordinates.

Provide helper functions for common lattices only when needed. Initially allow user-supplied shells.

Acceptance criteria:

- user can define shells manually.
- gamma_n(Q) has expected periodicity in RLU.
- synthetic harmonic model can be fit.

## Milestone 6: Global fitting across temperatures

Support multiple `PointData4D` datasets with shared and temperature-dependent parameters.

Examples:

- shared Q centers and harmonic coefficients.
- temperature-dependent r(T), amplitude(T), Gamma(T).
- shared background or dataset-specific background.

Acceptance criteria:

- global residual concatenates datasets correctly.
- parameters can be shared or separate.
- synthetic two-temperature dataset recovers known trends.

## Milestone 7: Uncertainty tools

Implement:

- covariance/correlation matrix reporting.
- bootstrap resampling.
- profile likelihood for one or two parameters.

Acceptance criteria:

- bootstrap returns distributions for selected parameters.
- profile likelihood identifies asymmetric uncertainty in a nonlinear synthetic example.
- fit reports warn about strong correlations.

## Milestone 8: QFI and moment tools

Implement:

- finite-window QFI integral.
- QFI integrand plotting helper.
- moment integral over measured window.
- normalization helper functions.

QFI finite-window lower bound:

F_Q_window = (4/pi) integral_{E_min}^{E_max} dE tanh(E/(2 kB T)) chi''(Q,E).

Moment finite-window estimate:

m2_window ~ integral_{Q window} dQ integral_{E window} dE S(Q,E).

Acceptance criteria:

- numerical integration works on gridded and point-cloud data.
- units and normalization assumptions are included in output object.
- no claim of rigorous entanglement depth unless normalization assumptions support it.

## Milestone 9: Documentation website

Build a polished documentation website for both scientific transparency and practical reuse.

Required tooling:

- Sphinx with a clean scientific theme.
- MyST Markdown for physics notes and equations.
- MyST-NB for Jupyter notebook examples.
- Read the Docs for hosted documentation builds.
- API reference generated from docstrings once the package API stabilizes.
- Local build command using `sphinx-build`.
- Modern `pyproject.toml` packaging configuration.

Core pages:

- landing page with package purpose, supported workflows, and current limitations.
- installation and environment setup.
- quickstart using synthetic data.
- data model guide for H,K,L,E, intensity, sigma, masks, metadata, and units.
- physics conventions for chi'', S(Q,E), Bose factor, form factor, polarization factor, reciprocal lattice units, and energy-transfer signs.
- model reference with equations, parameter definitions, units, and validity regimes.
- fitting guide with parameter bounds, shared parameters, uncertainty interpretation, and residual diagnostics.
- derived quantities guide for static susceptibility, local susceptibility, moments, and QFI-related integrals.
- approximations and caveats page explaining finite windows, background handling, normalization assumptions, phenomenological lattice harmonics, and when not to overinterpret fitted parameters.
- scientific references page with short notes on what each reference supports.
- examples gallery based on scripts or notebooks.
- distribution page explaining editable installs during development and the future `pip install metallix` path once released.

Acceptance criteria:

- docs build locally without warnings that hide broken links or missing pages.
- Read the Docs configuration is present once public hosting is desired.
- a new user can install the package, load or generate data, run a fit, plot results, and interpret key outputs by following the docs.
- every public model has a documented equation, parameter table, units, and convention notes.
- approximation pages distinguish implemented behavior from scientific interpretation.
- references are attached to the relevant equations or assumptions, not only listed at the end.
- packaging metadata is organized so the project can eventually be published as `metallix`.

## Milestone 10: Robust reference examples and validation gallery

Build a curated set of user-facing examples that also serve as known-reference
cases for human checking and regression validation.

Purpose:

- teach users how to use the package on realistic workflows.
- provide known inputs, expected fitted parameters, expected plots, and expected derived quantities.
- catch changes that alter scientific behavior even when low-level unit tests still pass.
- give collaborators a concrete way to inspect conventions, approximations, and output interpretation.

Recommended reference cases:

- clean synthetic single-Q paramagnon fit with known parameters.
- noisy single-Q fit with uncertainty estimates and residual diagnostics.
- lattice-harmonic susceptibility example with user-defined shells.
- two-temperature global fit with shared and temperature-dependent parameters.
- finite-window moment and QFI-related integral example with clearly stated lower-bound interpretation.
- background-misspecification example showing residuals and interpretation limits.
- optional realistic reduced-data example once a shareable dataset is available.

Each reference case should include:

- a script or notebook that runs from a clean install.
- a small deterministic input dataset or a fixed random seed for synthetic data.
- expected parameter values and tolerance windows.
- saved summary outputs, such as JSON or YAML, for automated comparison.
- representative plots for human visual inspection.
- a short explanation of the physics convention, assumptions, and what should not be overinterpreted.

Acceptance criteria:

- examples run in a predictable order and finish in reasonable time.
- automated checks compare key fitted parameters and derived quantities against stored reference values.
- generated plots are suitable for documentation and manual review.
- examples are linked from the documentation website.
- reference outputs are regenerated intentionally, with notes explaining any scientific or numerical change.

## Milestone 11: Resolution convolution, optional

Only after basic fitting works.

Possible approaches:

1. Gaussian convolution in gridded coordinates.
2. Monte Carlo resolution sampling per data point.
3. Interface to external resolution calculation if available.

Acceptance criteria:

- can turn resolution convolution on/off.
- tests with known Gaussian broadened model.
- performance remains acceptable for representative data subsets.

## Milestone 12: Performance tuning

Profile first. Then consider:

- vectorization improvements.
- numba for inner loops.
- jax for JIT and automatic differentiation.
- dask for chunked arrays.

Do not add performance dependencies until correctness tests exist.

## Milestone 13: Polarized neutron scattering support, far future

Only consider this after the base unpolarized chi'' analysis workflow is stable,
well tested, and has been used on representative datasets.

Possible capabilities:

- data containers that represent spin-flip and non-spin-flip channels explicitly.
- metadata for incident/analyzed polarization directions, guide field geometry, and polarization coordinate frames.
- correction tools for flipping ratios, finite polarization efficiency, and leakage between channels.
- channel-aware cross-section wrappers that can separate magnetic, nuclear, spin-incoherent, and background contributions when the measurement supports it.
- uncertainty propagation through polarization corrections.
- examples based on real or realistic polarized neutron datasets.

Acceptance criteria:

- unpolarized analysis behavior remains unchanged.
- channel conventions are documented with units and coordinate definitions.
- synthetic polarized-channel tests recover known magnetic and nonmagnetic components.
- polarization corrections expose assumptions instead of silently applying instrument-specific defaults.
