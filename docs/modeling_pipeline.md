# Modeling and fitting pipeline

The fitting framework is organized around simultaneous fits to one or more
datasets. The implementation is intentionally modular: masking and rebinning
are dataset-local preprocessing steps, the physics model is shared across the
fit, each dataset may apply its own instrument resolution model, and the
optimizer minimizes one concatenated weighted residual vector.

## Core objects

`PointData4D`
: Flattened measured data containing `H`, `K`, `L`, energy transfer `E`,
  intensity, uncertainty `sigma`, an analysis mask, optional temperature, and
  metadata.

`ParameterSpec`
: A named scalar parameter with an initial value, optional bounds, fixed/varying
  state, units, and a description. Fixed parameters are passed to models but are
  not included in the optimizer vector.

`FitDataset`
: One measured dataset in the global objective. It stores the data, a
  chi-squared weight, an ordered list of preprocessing transforms, optional
  dataset-specific resolution, and metadata such as instrument or scan settings.

`ModelSpec`
: A callable physics model plus descriptive metadata. The callable receives
  prepared point data and a parameter dictionary, and returns predicted measured
  intensity at each point. It can wrap a phenomenological formula, an ab initio
  calculation, a Monte Carlo model, or any later backend with the same interface.

`ResolutionSpec`
: Optional instrument-resolution hook. It receives data coordinates,
  unconvolved model values, and parameters, then returns resolution-corrected
  measured intensity. `identity_resolution` is the placeholder used when
  resolution effects are disabled.

`FitProblem`
: The complete simultaneous-fitting definition: datasets, shared model,
  parameter specifications, and metadata.

## Masking

The analysis mask follows the `PointData4D` convention that `True` means a point
is still eligible for analysis. Masking helpers return new data objects and
combine with existing masks, so independent masks can be composed in a
`FitDataset` transform list.

```python
from metallix import FitDataset, make_energy_q_mask_transform, make_mask_transform

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
from metallix import attach_lattice_parameters, make_energy_q_mask_transform

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
from metallix import make_box_mask_transform, make_ellipsoid_mask_transform

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
from metallix import make_phonon_mask_transform

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

`rebin_point_data` and `make_rebin_transform` provide a regular 4D rebinning
placeholder for `(H,K,L,E)` point data:

```python
from metallix import make_rebin_transform

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

More specialized operations, such as projecting into a cut, integrating a
region, or applying instrument-specific normalization, should be written as
custom transforms with the same signature:

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
from metallix import (
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

## Primitive and compound models

The simplest measured-intensity model is constant in momentum and energy:

```python
from metallix import make_constant_intensity_model

background = make_constant_intensity_model("background")
```

Multiple primitive models can be added together with
`compound_additive_model`. Each component receives the same prepared data and
parameter dictionary, and the returned measured intensities are summed:

```python
from metallix import compound_additive_model, make_constant_intensity_model

model = compound_additive_model(
    make_constant_intensity_model("background"),
    more_complicated_model,
)
```

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
from metallix import FitDataset, constant_fwhm_energy_resolution

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
from metallix import polynomial_fwhm_energy_resolution

resolution = polynomial_fwhm_energy_resolution(
    ["fwhm0", "fwhm1", "fwhm2"],
    oversampling=7,
)
```

The coefficients are ordered as `FWHM(E) = c0 + c1*E + c2*E**2 + ...`, and each
coefficient can be either a fixed number or a fitted parameter name. The FWHM is
required to remain finite and positive at all evaluated energies.

## Uncertainty sampling

`SamplerConfig`, `SamplingResult`, and `sample_problem_parameters` define the
future MCMC/uncertainty-sampling entry point. The sampler itself is deliberately
not implemented yet; once a backend is chosen, it will consume the same
`FitProblem` and parameter specifications used by deterministic optimization.
