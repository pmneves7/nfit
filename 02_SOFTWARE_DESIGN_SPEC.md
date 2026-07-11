# Software Design Specification: 4D chi''(Q,E) Analysis Package

## 1. Package name

Working distribution name: `nfit`.

The package should eventually be installable with a command like:

```bash
pip install nfit
```

During early development, the import path may remain simple and provisional
while the public distribution name and module layout settle.

## 2. Data model

The package should support both gridded and flattened representations.

### 2.1 Flattened point cloud representation

This is best for fitting:

- H: array, shape (N,)
- K: array, shape (N,)
- L: array, shape (N,)
- E: array, shape (N,), meV
- intensity: array, shape (N,)
- sigma: array, shape (N,)
- mask: boolean array, shape (N,)
- temperature: scalar or array
- metadata: dict

The residual function should operate on the masked flattened arrays.

### 2.2 Gridded representation

This is best for visualization and integration:

Use xarray.DataArray or xarray.Dataset with dimensions such as:

- H
- K
- L
- E

Data variables:

- intensity
- sigma
- mask
- model, optional
- residual, optional

Metadata:

- sample name
- lattice constants
- reciprocal lattice vectors
- orientation matrix
- incident energy
- temperature
- absolute normalization status
- units

### 2.3 Storage formats

Support at least one simple internal format:

- HDF5 using h5py
- Zarr
- NetCDF via xarray

Avoid requiring Mantid or Horace to use the core package.

## 3. Coordinate handling

Default input coordinates: reciprocal lattice units H,K,L and energy transfer E in meV.

Functions needed:

- rlu_to_q_cart(H,K,L,lattice): convert to inverse Angstroms.
- q_cart_to_rlu(Qx,Qy,Qz,lattice)
- q_norm(H,K,L,lattice)
- fold_to_bz(H,K,L, lattice/symmetry), optional.
- apply_symmetry_equivalents(q, space_group or point_group), optional.
- distance_to_q0(H,K,L,q0, metric): isotropic or anisotropic distance in reciprocal space.

For fitting broad features, distance should allow either:

- Euclidean distance in absolute Q.
- RLU distance with user-defined metric.
- Quadratic form: dq^T A dq.

## 4. Physical model interface

Each model should be a pure function with signature similar to:

```python
def model_chi_double_prime(coords: Coordinates, params: Mapping[str, float]) -> np.ndarray:
    """Return chi'' at each input point."""
```

where coords contains vectorized H,K,L,E and optional temperature.

Do not tie model functions to lmfit Parameters. Write adapters that convert lmfit/scipy parameter vectors to dictionaries.

## 5. Initial models

### 5.1 Single-Q overdamped paramagnon

Parameters:

- amplitude: chi_delta or scale of chi
- q0_h, q0_k, q0_l
- kappa: inverse correlation length or width parameter
- omega_sf: spin fluctuation energy scale in meV
- background parameters

Formula:

a(Q) = 1 + |Q - Q0|^2/kappa^2

chi''(Q,E) = amplitude * (E/omega_sf) / [a(Q)^2 + (E/omega_sf)^2]

Optionally enforce oddness in energy:

chi''(Q,-E) = -chi''(Q,E)

For measured positive-energy loss data, use E > 0 only initially.

### 5.2 Multi-Q overdamped paramagnon

Sum over Q centers:

chi''(Q,E) = sum_delta amplitude_delta * (E/omega_sf_delta) /
[a_delta(Q)^2 + (E/omega_sf_delta)^2]

Allow symmetry-equivalent Q positions to share parameters.

### 5.3 Lattice-harmonic inverse susceptibility

Define:

chi^{-1}(Q,E) = r + Lambda(Q) - i E/gamma

Then:

chi''(Q,E) = [E/gamma] / { [r + Lambda(Q)]^2 + [E/gamma]^2 }

possibly multiplied by amplitude.

Equivalent relaxational form:

chi(Q) = amplitude / [r + Lambda(Q)]
Gamma_Q = gamma [r + Lambda(Q)]
chi''(Q,E) = chi(Q) * E Gamma_Q / [E^2 + Gamma_Q^2]

These are equivalent if amplitudes and gamma are defined consistently. Pick one convention and document it.

### 5.4 Background models

Start simple:

- constant background
- linear in E
- low-order polynomial in Q and E
- separable background B0 + B1 E + B2 |Q|^2

Later:

- phonon-inspired smooth background
- empirical background from high-Q or off-symmetry regions
- temperature-independent background shared across datasets

### 5.5 Neutron cross section wrapper

Separate chi'' from measured intensity:

I(Q,E) = scale * |f(Q)|^2 * P(Q) * chi''(Q,E) / [1 - exp(-E/kBT)] + B(Q,E)

where:

- f(Q): magnetic form factor; initially allow f=1 or tabulated approximate functions.
- P(Q): polarization factor; initially optional.
- Bose factor: handle carefully near E=0.
- scale: absolute or arbitrary normalization.

### 5.6 Far-future polarized neutron scattering support

Polarized neutron scattering is out of scope for the initial package. Treat it as a
far-future extension after the unpolarized 4D chi'' workflow, fitting, uncertainty
tools, normalization handling, and resolution treatment are reliable.

Eventual support should be designed around explicit polarization channels rather
than overloading the scalar intensity model. Likely requirements include:

- spin-flip and non-spin-flip channel labels.
- incident and analyzer polarization states.
- guide-field or XYZ polarization coordinate conventions.
- flipping-ratio corrections and polarization-efficiency matrices.
- leakage between channels and uncertainty propagation from polarization corrections.
- separation of magnetic, nuclear coherent, nuclear spin-incoherent, and background terms where the experiment supports it.
- clear metadata for instrument configuration and sample orientation.

Do not add this to the near-term model API until representative polarized datasets
and reduction conventions are available.

## 6. Fitting interface

### 6.1 Residuals

Use weighted residuals:

r_i(theta) = [I_obs_i - I_model_i(theta)] / sigma_i.

Handle invalid sigma values:

- mask sigma <= 0
- optionally impose sigma floor
- support Poisson-like sigma when raw counts are available

### 6.2 Optimizers

Initial deterministic optimizers:

- scipy.optimize.least_squares
- lmfit.Minimizer for named bounded parameters

Later:

- profile likelihood scans
- bootstrap resampling
- MCMC via emcee or PyMC

### 6.3 Parameter handling

Support:

- bounds
- fixed/free flags
- tied/shared parameters
- global parameters across temperatures
- temperature-dependent parameters such as r(T), Gamma(T), amplitude(T)

A suggested parameter object:

```python
@dataclass
class ParameterSpec:
    name: str
    value: float
    min: float | None = None
    max: float | None = None
    vary: bool = True
    unit: str = ""
    description: str = ""
```

## 7. Uncertainty estimates

Implement in stages.

### Stage 1: covariance from least squares

Use Jacobian from scipy/lmfit:

cov ~ s^2 (J^T J)^{-1}

Warn when covariance is singular or parameters are highly correlated.

### Stage 2: bootstrap

Options:

- residual bootstrap
- voxel/bootstrap over points
- block bootstrap over reciprocal-space/energy regions
- bootstrap over repeated scans, if scan identity exists

### Stage 3: profile likelihood

For each parameter of interest:

- fix parameter at grid of values
- reoptimize all others
- record delta chi^2

Useful for nonlinear correlated parameters.

### Stage 4: MCMC

Use deterministic fit as starting point. Define likelihood:

log L = -0.5 sum_i [ (I_obs_i - I_model_i)^2/sigma_i^2 + log(2 pi sigma_i^2) ].

Add priors for physically constrained parameters, e.g. positive widths and positive relaxation scales.

## 8. Derived quantities

### 8.1 Static susceptibility

For relaxational model, chi(Q) is an explicit parameter/function. For a general fitted chi'', estimate via Kramers-Kronig:

chi'(Q,0) = (2/pi) integral_0^infty [chi''(Q,E)/E] dE.

Only do this if the energy range and high-energy extrapolation are adequate.

### 8.2 Relaxation rate Gamma_Q

For the overdamped paramagnon form:

Gamma_Q = omega_sf * [1 + |Q-Q0|^2/kappa^2].

For inverse-susceptibility harmonic form:

Gamma_Q = gamma * [r + Lambda(Q)].

Report convention and units.

### 8.3 Local susceptibility

chi''_local(E) = integral_BZ d^3Q chi''(Q,E).

Implement numerical integration over measured grid with bin-volume weights if available.

### 8.4 Moment integral

Observed moment estimate:

m2_obs ~ integral dQ dE S(Q,E),

where S(Q,E) is corrected for Bose factor, form factor, polarization factor, and absolute normalization as available.

This is usually a lower bound if the measurement window is finite.

### 8.5 QFI integral

For a selected Q or Q region:

F_Q(Q,T) = (4/pi) integral_0^infty dE tanh[E/(2 kB T)] chi''(Q,E).

Implement finite-window lower-bound estimate. Also support Q-integrated or operator-defined versions if the user defines the operator.

Normalization options:

- absolute QFI density
- conservative S_max per ion/unit cell
- effective moment normalization using measured m2
- user-provided normalization constant

Always report the assumptions.

## 9. Visualization

Provide functions for:

- 1D cuts: intensity/model/residual vs E or along Q direction
- 2D slices: HK, HL, KL, Q-E maps
- parameter summary plots vs T
- residual maps
- corner plots or covariance heatmaps for uncertainty analysis
- chi''_local(E)
- QFI integrand vs E

Plotting should be optional and separate from fitting.

## 10. Documentation and website

The package should have a documentation website aimed at experimental condensed
matter physicists and collaborators who may not know the internals of the code.
The website is part of the scientific product, not just packaging polish.

Recommended structure:

- `docs/index.md`: purpose, scope, status, and links to the main workflows.
- `docs/getting_started.md`: install, import, create or load data, fit a first model.
- `docs/physics_conventions.md`: definitions of chi'', S(Q,E), E sign convention, Bose factor, form factor, polarization factor, RLU coordinates, absolute-unit assumptions, and temperature units.
- `docs/model_reference.md`: equations, parameter meanings, units, bounds, and limiting behavior for each model.
- `docs/fitting.md`: residual definitions, sigma handling, parameter sharing, uncertainty estimates, and diagnostics.
- `docs/derived_quantities.md`: static susceptibility, local susceptibility, moment integrals, and QFI-related quantities.
- `docs/approximation_notes.md`: finite measurement windows, background choices, resolution effects, phenomenological lattice harmonics, normalization caveats, and interpretation limits.
- `docs/references.md`: annotated scientific references with short explanations of what each reference supports.
- `docs/examples/`: rendered examples from notebooks or scripts.

Documentation rules:

- Use the same symbols in docs, docstrings, tests, and example notebooks.
- Put units next to every parameter and derived quantity.
- State whether a quantity is measured, corrected, model-derived, lower-bound, or extrapolated.
- Link references near the equations or approximations they justify.
- Keep examples executable against the current package API.
- Keep the README short and point readers to the full website.

Documentation and distribution stack:

- Sphinx for the documentation build.
- MyST Markdown for prose pages with equations and scientific notes.
- MyST-NB for executable Jupyter notebook examples in the documentation.
- Read the Docs for hosted documentation builds.
- Modern `pyproject.toml` packaging metadata and build configuration.
- A local `sphinx-build` check included in the developer workflow once docs are nontrivial.
- A future PyPI-ready distribution so users can install with `pip install nfit`.

## 11. Reference examples and validation cases

The package should include robust examples that are useful to users and also act
as reference validations for scientific behavior. These examples should sit
between lightweight unit tests and full analysis notebooks: readable enough for
humans, deterministic enough for automated comparison.

Design goals:

- every major workflow has at least one complete example.
- examples use fixed seeds or small committed datasets so outputs are reproducible.
- expected fitted parameters, uncertainties, derived quantities, and selected plot outputs are stored as reference artifacts.
- examples explain the physics assumptions, units, approximations, and interpretation limits.
- reference artifacts are updated only when an intentional scientific or numerical change is made.

Suggested examples:

- single-Q overdamped paramagnon recovery from synthetic data.
- noisy single-Q fit with covariance, bootstrap or profile-likelihood diagnostics.
- lattice-harmonic fit with user-supplied neighbor shells.
- global fit across temperatures with shared Q structure and temperature-dependent amplitudes or relaxation rates.
- moment and QFI finite-window lower-bound calculation.
- background or finite-window caveat example showing how residuals reveal model limitations.

Validation rules:

- keep examples executable by continuous checks where practical.
- compare numerical summaries against tolerances, not exact floating-point output.
- store compact machine-readable summaries in JSON or YAML.
- include human-readable plots for inspection in the documentation gallery.
- make any regenerated reference output an explicit review point.

## 12. Testing strategy

Tests should include:

1. Model returns correct shapes for vectorized inputs.
2. chi'' is nonnegative for E > 0 when amplitude and widths are positive.
3. Overdamped model peaks at expected energy for a given Q convention.
4. Gamma_Q formula matches the linewidth convention.
5. Synthetic-data fit recovers known parameters within tolerance.
6. Lattice harmonic functions are symmetry-consistent for simple lattices.
7. Bose factor handles E near zero without NaN/inf.
8. Masks and sigma handling work correctly.
9. QFI integral returns known analytic/numerical results for simple test functions.
10. Reference examples reproduce stored summary values within documented tolerances.
