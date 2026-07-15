# nfit Data Playground and Analysis Implementation Plan

Status: implementation specification

Audience: implementation agents and the integration/review agent

Baseline: nfit 0.8.2, commit `d5cae92`

## 1. Objective

Add a first-class analysis layer for scientific operations on datasets that is
independent of the fitting pipeline. The user-facing name is **Data
Playground**. It must initially support:

1. Bragg-peak integration using a box, a fixed ellipsoid, or a fitted Gaussian
   ellipsoid, with optional local-background subtraction.
2. Quantitative spectral reductions, including total-moment integration over
   every measured Brillouin zone and quantum Fisher information (QFI) as a
   function of momentum.
3. A registry that can later host mode tracking, phonon/magnon peak extraction,
   dataset arithmetic, raw time-of-flight reduction, and general background
   subtraction.

Analysis operations must consume ordinary nfit datasets and return ordinary
nfit datasets, tables, or scalar results. They must not create a second hidden
scientific state system and must not depend on fit models, parameter sharing,
fit timelines, or optimizer configuration.

## 2. Non-negotiable design decisions

Implementation agents must follow these decisions unless Paul explicitly
changes the plan:

- Build one Data Playground window with registry-driven, tool-specific editors.
  Do not build an unrelated top-level application for each operation.
- Put an `Analyses` node beside `Datasets`, `Models`, and `Fits` in each
  workspace tree.
- Preserve input datasets. Every operation is non-destructive.
- Put dataset-like outputs back into the workspace under a `Derived data`
  dataset group. Keep the analysis entry as the authoritative recipe and
  provenance record.
- Use `PointListData` for Bragg tables, per-zone tables, and one-dimensional
  momentum curves. Use `MDHistoData` for momentum-resolved outputs with two or
  more remaining axes. Use scalar result records when no axes remain.
- Add stable dataset IDs before adding analysis references. Names are editable
  and cannot be used as foreign keys.
- Store operation parameters as plain JSON-compatible data and register their
  defaults, validation, labels, and tooltips in one registry.
- Keep numerical kernels free of Qt and project-GUI imports.
- Quantitative integrations must report measured coverage. Missing or masked
  volume must never silently count as zero.
- Quantitative spectral operations must require an explicit intensity
  convention. They must refuse ambiguous count data rather than labeling an
  arbitrary-unit integral as a total moment or QFI.
- Initial analysis operations accept reduced physical-coordinate data. Future
  raw-TOF reduction will produce reduced datasets through the same operation
  graph; Bragg and spectral kernels do not read instrument event files directly.
- All new GUI controls need useful hover text and tooltip regression tests, as
  required by `AGENTS.md`.

## 3. Scope boundaries

### Required in the first complete implementation

- Stable IDs and project-file migration.
- Analysis recipe/result objects and operation registry.
- Derived-output artifact persistence and stale-result detection.
- Data Playground window and project-tree integration.
- Box and fixed-ellipsoid Bragg summation.
- Gaussian-ellipsoid Bragg profile fitting.
- Optional local shell background for every Bragg method.
- Imported/manual and crystal-predicted peak lists.
- Spectral conversion to a declared `chi''` convention.
- QFI energy reduction producing a viewer-ready momentum map.
- Total-moment energy reduction and per-Brillouin-zone integration.
- Coverage, uncertainty, provenance, export, progress, and cancellation.
- Script APIs, documentation, and synthetic tests.

### Explicitly deferred

- Blind peak finding. The first release integrates predicted or imported peak
  positions. A later `find_bragg_peaks` operation can create a peak-list input.
- Simultaneous fitting of overlapping Bragg peaks.
- Full nuclear structure-factor refinement and absolute-scale refinement.
  Preserve enough output and crystal metadata to add this cleanly later.
- Mode/ridge tracking and dispersion fitting.
- Arbitrary Python expressions entered in the GUI. Weighted sums use registered
  kernels and typed parameters; never call `eval` on user text.
- Direct analysis of unreduced detector events.
- General covariance matrices between rebinned bins. Initial uncertainties
  assume independent bins and must record that assumption.
- Analysis-run history. Version 1 stores the latest result only. Rerunning an
  analysis replaces its artifacts after confirmation and preserves stable
  output dataset IDs.

## 4. Current repository seams to reuse

- `src/nfit/dataset.py`: `PointData4D` and `PointListData`.
- `src/nfit/mdhisto.py`: `MDHistoAxis`, `MDHistoData`, and measured-bin logic.
- `src/nfit/project_gui.py`: project serialization, dataset preparation,
  project tree, task workers, viewers, and portable `.npz` dataset archives.
- `src/nfit/cross_section.py`: Bose factor and forward `chi''` to intensity
  conversion.
- `src/nfit/form_factors.py`: magnetic form factors.
- `src/nfit/crystal.py`: CIF import, symmetry operations, and lattice geometry.
- `src/nfit/sum_rules.py`: model-side total-moment and Kramers-Kronig reference
  implementations.
- `src/nfit/rebin.py`: bounded-memory accumulation and fractional binning.
- `src/nfit/plotting.py` and `src/nfit/qt_slice_viewer.py`: viewer and channel
  selection.
- `src/nfit/pipeline.py`: project-domain dataclasses.

Do not make the analysis package import `project_gui.py`. Shared data-preparation
and coordinate helpers currently trapped in that module must move to non-Qt
modules as described below.

## 5. Target architecture

```text
NfitProject
  DataGroup (workspace)
    DatasetEntry[]                  source and derived datasets
    AnalysisEntry[]                 recipes and latest result manifests
    ModelComponentSpec[]            fitting remains independent
    FitTimelineEntry[]

AnalysisEntry
  -> references input DatasetEntry IDs
  -> selects AnalysisOperationDefinition from registry
  -> AnalysisRunner resolves prepared inputs and context
  -> pure operation returns AnalysisExecution
  -> ArtifactStore writes dataset/table arrays
  -> output DatasetEntry objects reference artifacts
  -> recipe/input hashes determine fresh versus stale
```

Create this package:

```text
src/nfit/analysis/
  __init__.py
  core.py             dataclasses, enums, exceptions, progress protocol
  registry.py         operation and parameter registry
  runner.py           project-aware input resolution and execution
  artifacts.py        atomic artifact persistence and loading
  fingerprint.py      canonical recipe and input hashing
  coordinates.py      physical coordinate grids, edges, Jacobians, Q matrices
  corrections.py      intensity/S/chi'' conversions and unit conventions
  zones.py            reciprocal-lattice basis and Brillouin-zone assignment
  bragg.py            peak-list generation and Bragg integration
  spectral.py         energy kernels and spectral reduction
```

Qt code belongs in:

```text
src/nfit/analysis_gui.py
```

If the generic progress worker is extracted from `project_gui.py`, put it in:

```text
src/nfit/qt_tasks.py
```

## 6. Domain model

Add stable IDs with lowercase 32-character UUID hex strings. IDs are generated
with `uuid.uuid4().hex` and are never derived from names.

Add `id` as the final defaulted field of `DatasetEntry` so existing positional
constructors remain compatible. Copying a dataset creates a new ID; moving or
renaming preserves it. Loading a version-1 project without IDs generates IDs,
and the next save persists them.

Add these dataclasses to `src/nfit/analysis/core.py`:

```python
@dataclass
class AnalysisEntry:
    name: str
    type: str
    input_dataset_ids: list[str]
    parameters: dict[str, Any]
    id: str = field(default_factory=new_analysis_id)
    operation_version: int = 1
    enabled: bool = True
    result: AnalysisResultRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResultRecord:
    recipe_hash: str
    input_fingerprints: dict[str, str]
    outputs: list[AnalysisOutputRef]
    status: str                 # success, failed, cancelled
    created_at: str
    duration_seconds: float | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

@dataclass
class AnalysisOutputRef:
    key: str                   # stable within an operation type
    label: str
    kind: str                  # dataset, table, scalar
    artifact_path: str | None = None
    dataset_id: str | None = None
    scalar_value: float | None = None
    scalar_uncertainty: float | None = None
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisInput:
    dataset_id: str
    dataset_name: str
    data: MDHistoData | PointListData | PointData4D
    context: AnalysisContext
    fingerprint: str

@dataclass
class AnalysisContext:
    data_group_name: str
    lattice_parameters: dict[str, float] | None
    spacegroup: str | None
    crystal: dict[str, Any] | None
    temperature_K: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisExecution:
    outputs: dict[str, AnalysisOutput]
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

`AnalysisOutput` is a tagged union of:

- `DatasetOutput(data, label, data_type, metadata)`
- `TableOutput(data: PointListData, label, metadata)`
- `ScalarOutput(value, uncertainty, unit, label, metadata)`

Add `analyses: list[AnalysisEntry]` as the final field of `DataGroup`.

Do not put large arrays inside `AnalysisResultRecord`, project JSON, or generic
metadata. Arrays live in artifacts or in runtime data objects.

## 7. Operation registry contract

Follow the registry style already used for masks and models. Define:

```python
@dataclass(frozen=True)
class AnalysisParameterDefinition:
    name: str
    label: str
    type: str
    default: Any
    description: str
    allowed: str
    example: str
    choices: tuple[tuple[str, str], ...] = ()
    required: bool = False

@dataclass(frozen=True)
class AnalysisOperationDefinition:
    key: str
    label: str
    version: int
    description: str
    min_inputs: int
    max_inputs: int | None
    accepted_containers: tuple[str, ...]
    parameters: tuple[AnalysisParameterDefinition, ...]
    validate: Callable[[Sequence[AnalysisInput], Mapping[str, Any]], None]
    execute: Callable[..., AnalysisExecution]
```

Required registry keys:

- `bragg_integration`
- `spectral_integration`

Reserve, but do not implement yet:

- `dataset_subtraction`
- `find_bragg_peaks`
- `mode_tracking`
- `raw_tof_reduction`

Provide these public functions:

```python
available_analysis_types()
analysis_definition(type_name)
default_analysis_parameters(type_name)
analysis_parameter_tooltip(type_name, parameter_name)
validate_analysis(type_name, inputs, parameters)
run_analysis_operation(type_name, inputs, parameters, progress_callback=None,
                       cancel_callback=None)
```

Every parameter must be registry-defined. GUI-only scientific parameters are
forbidden.

## 8. Project serialization and migration

Increment the nfit project JSON schema from version 1 to version 2. The loader
must accept versions 1 and 2.

Version 2 adds:

- `id` on every dataset.
- `analyses` on every data group.
- Serialized `AnalysisEntry`, `AnalysisResultRecord`, and output manifests.
- Relative artifact references.

Unknown analysis types must load without data loss. Display them as unavailable
and preserve their JSON so a project can still be resaved.

### Artifact layout

For a saved project `/path/sample.nfit`, use:

```text
/path/sample.nfit
/path/sample.nfit-assets/
  analyses/
    <analysis-id>/
      <output-key>.npz
```

Rules:

- The GUI must ask the user to save an unsaved project before running an
  operation that produces persisted output.
- Store artifact paths relative to the `.nfit` file.
- Write to a temporary file in the destination directory, close it, then use
  atomic `Path.replace`.
- Rerunning writes new temporary artifacts and replaces all successful outputs
  only after every output has been written. A failed/cancelled run leaves the
  previous successful artifacts and result record untouched.
- `Save As` copies the asset directory and rewrites no IDs.
- Missing artifacts do not prevent project loading. Mark the analysis stale and
  its derived dataset unavailable, with a rerun action.
- Deleting an analysis removes its derived dataset entries. Artifact deletion is
  allowed only after confirmation and only for paths under that project's asset
  root.
- Never delete arbitrary paths found in project JSON.

Extend the nfit dataset archive format to version 2 only if auxiliary histogram
channels are added (Section 13). The loader must continue to read version 1.

Centralize source-path handling. Add helpers that resolve either absolute source
files or project-relative analysis artifacts. Do not scatter `Path(source_file)`
through new code.

## 9. Provenance, fingerprints, and stale state

Canonical JSON uses sorted keys, compact separators, finite numeric values, and
no object `repr`. Hash with SHA-256.

The recipe hash includes:

- operation key and operation version;
- normalized parameters after defaults and validation;
- ordered input dataset IDs.

Each input fingerprint includes analysis-relevant state:

- dataset ID, kind, and data type;
- source file path, size, and `mtime_ns`, when file-backed;
- source artifact SHA-256, when derived;
- in-memory array content hash when no source exists;
- dataset scale and kinematic settings;
- dataset rebin configuration;
- enabled dataset and inherited group masks in effective order;
- group lattice, space group, crystal, and UB context;
- temperature and any correction metadata consumed by the operation.

Do not hash display-only GUI state or fit configuration.

An analysis is fresh only when its current recipe hash and every current input
fingerprint exactly match its latest successful result. Compute this on demand;
initial implementation does not need a mutation-event bus.

Derived dataset entries include:

```json
{
  "derived_from_analysis": {
    "analysis_id": "...",
    "output_key": "qfi_map",
    "recipe_hash": "..."
  }
}
```

Downstream analyses fingerprint the derived artifact, so changing an upstream
analysis naturally marks downstream results stale.

## 10. Quantitative signal conventions

Introduce a JSON-serializable `SpectralConvention` in
`analysis/corrections.py`. Do not infer these fields from unit strings alone.

Required fields:

```text
representation:
  measured_intensity | cross_section | s_qw | chi_double_prime
unit:
  explicit display/physics unit
normalization_basis:
  per_magnetic_ion | per_formula_unit | per_unit_cell | unknown
magnetic_ions_per_basis:
  positive float when conversion is needed
moment_unit:
  spin_squared | mu_B_squared
g_factor:
  positive float or null
form_factor_state:
  included | removed
polarization_state:
  included | removed
bose_state:
  included | removed
kf_ki_state:
  included | removed
absolute_scale:
  true | false
```

For a published total moment or QFI, `absolute_scale` must be true and the
normalization basis must be known. The GUI may still offer a clearly labeled
`weighted integral (arbitrary units)` kernel for non-absolute data, but it must
not label that result total moment or QFI.

Add an inverse of `intensity_from_chipp` to `cross_section.py`. The forward and
inverse functions must round-trip arrays under the same convention:

```text
I - background = scale * gamma_0^2 / pi * f(Q)^2 * P(Q)
                 * chi'' / [1 - exp(-E/k_B T)]
```

The inverse must reject zero/negative scale, nonpositive form factor or
polarization, and ambiguous `k_f/k_i` state. The current dataset preparation
already normalizes `k_f/k_i` when requested; the analysis result must record
whether that path was used.

When `chi''` is in `mu_B^2/meV` and QFI is requested for dimensionless spin
operators, divide by `(g * mu_B)^2`, represented numerically as division by
`g^2` after stripping the declared `mu_B^2` unit. Never silently set `g=2`.

Unpolarized neutron data measure components transverse to Q. Offer these modes:

- `already_corrected`
- `isotropic_single_component`
- `isotropic_trace`
- `custom_scalar`

Require the user to select an assumption unless the dataset convention already
states it. Record the selection in provenance and output labels.

## 11. Shared coordinate and integration primitives

Move/refactor physical coordinate helpers from `project_gui.py` into
`analysis/coordinates.py` or a more general `coordinates.py` module. Preserve
existing behavior with tests before switching GUI callers.

Required primitives:

```python
physical_coordinate_arrays(data) -> dict[str, np.ndarray]
physical_axis_vectors(data) -> np.ndarray
rlu_to_q_matrix(data_metadata, group_context) -> np.ndarray
bin_edges(axis, size) -> np.ndarray
bin_widths(axis, size) -> np.ndarray
q_bin_volume(data, context) -> float | np.ndarray
energy_bin_widths(data) -> np.ndarray
measured_mask(data) -> np.ndarray
```

Quantitative integration must distinguish signal density from per-bin integral.
Add explicit metadata:

```text
signal_semantics: density | bin_integral | unknown
```

- For `density`, multiply signal and error by physical overlap volume.
- For `bin_integral`, multiply by fractional bin overlap only.
- For `unknown`, quantitative operations fail validation with a useful message.

For linearly projected MDHisto axes, compute physical Q volume using the
determinant/Jacobian of the three momentum-axis vectors after conversion to
inverse angstrom. Reject non-independent momentum axes for 3D integrals.

Avoid materializing complete H/K/L coordinate grids for large 4D datasets.
Use axis vectors, bounding slices, broadcasting, and chunking.

## 12. Bragg integration operation

### Inputs

- Exactly one `MDHistoData` input in three momentum dimensions, or four
  dimensions with one energy-transfer axis and an explicit elastic energy
  window.
- Crystal/lattice/UB context sufficient to map HKL to Q.
- Peak positions from either:
  - `peak_source = crystal`: generate integer HKL candidates from space group
    and measured bounds; or
  - `peak_source = table`: a `PointListData`/parameter payload with H, K, L.

Blind peak discovery is not part of this operation.

### Parameters

Registry parameters must include:

```text
peak_source                 crystal | table
peak_table_dataset_id       dataset ID or blank
include_systematic_absences bool, default false
d_min_angstrom              optional
d_max_angstrom              optional
energy_min_meV              required for 4D input
energy_max_meV              required for 4D input
method                      box_sum | ellipsoid_sum | gaussian_fit
coordinate_frame            hkl | q_angstrom_inverse
box_half_widths             three positive floats
ellipsoid_semiaxes          three positive floats
ellipsoid_rotation          3x3 orthonormal matrix, identity default
center_mode                 nominal | centroid
centroid_search_radius      positive float
background_mode             none | shell
background_inner_scale      >= 1
background_outer_scale      > inner scale
exclude_neighbor_regions    bool, default true
minimum_peak_coverage       [0, 1], default 0.9
minimum_background_coverage [0, 1], default 0.7
edge_policy                 reject | report_partial
gaussian_background         constant | linear
gaussian_max_nfev           positive integer
gaussian_fallback           none | ellipsoid_sum
subvoxel_samples            odd integer, default 3
```

The GUI may provide convenient scalar linking for three semiaxes, but the
stored parameter is always an explicit three-vector.

### Peak generation

For crystal-generated positions:

1. Determine conservative integer HKL bounds from the measured momentum volume
   and the outer background extent.
2. Generate integer triples in those bounds.
3. Use space-group symmetry/centering to reject systematic absences unless the
   user opts in.
4. Apply d-spacing limits.
5. Retain peaks whose outer analysis region intersects the measured volume.
6. Sort deterministically by `|Q|`, then H, K, L.

The output must retain absent/rejected candidates only when a diagnostic option
is enabled; normal output contains attempted physical peaks.

### Box summation

For an HKL-aligned box, calculate exact 1D overlap fractions from bin edges and
take their tensor product. For a rotated Q-space box, use deterministic
subvoxel sampling, as for ellipsoids.

### Fixed-ellipsoid summation

Define membership in Q coordinates by:

```text
r = R^T (Q - Q0)
sum_i (r_i / a_i)^2 <= 1
```

Estimate fractional boundary-bin overlap with a deterministic Cartesian
subvoxel grid. `subvoxel_samples=3` means 27 samples per Q voxel. Do not use
random Monte Carlo sampling. Tests must show convergence when increasing from
3 to 5 samples.

### Local shell background

The inner and outer shell are homothetic copies of the peak region. Compute
effective measured volume, not only ideal geometric volume:

```text
rho_bg = I_shell / V_shell_measured
B      = rho_bg * V_peak_measured
var(B) = var(I_shell) * (V_peak_measured / V_shell_measured)^2
I_corr = I_peak - B
var(I_corr) = var(I_peak) + var(B)
```

This matches the established shell-density method while correctly accounting
for nfit masks and detector coverage. Exclude the peak regions of neighboring
predicted reflections from the shell when requested. Track the excluded volume.

Only the arithmetic mean background is required initially. Robust clipping or
median background estimation is deferred because its uncertainty is not the
same quadrature formula.

### Gaussian ellipsoid fit

This is a local profile operation and does not use nfit's global fitting
pipeline. Use `scipy.optimize.least_squares` directly in `analysis/bragg.py`.

Fit in Q coordinates:

```text
I(Q) = A exp[-0.5 (Q-Qc)^T Sigma^-1 (Q-Qc)] + background(Q)
```

Rules:

- Parameterize `Sigma` through a Cholesky factor so it stays positive definite.
- Constrain `A >= 0` and constrain `Qc` to the centroid search region.
- Enforce configurable lower/upper eigenvalue bounds based on bin resolution
  and outer region size.
- Support constant or linear background.
- Weight residuals by one-sigma bin uncertainties.
- Use only measured, unmasked bins.
- The infinite-volume integrated peak is
  `A * (2*pi)^(3/2) * sqrt(det(Sigma))` when signal is a Q-density.
- Propagate uncertainty from the fitted parameter covariance using the gradient
  of the integrated intensity. Record covariance rank and reduced chi-squared.
- If covariance is singular, return the intensity with `dI=NaN` and a diagnostic
  status; do not invent an uncertainty.
- If a configured fallback runs, record both the fit failure and fallback
  method. Never silently replace a failed fit.

### Bragg output

Return a `PointListData` table with coordinate columns H, K, L and at least:

```text
H, K, L, Q, d,
I_raw, dI_raw,
Background, dBackground,
I, dI, I_over_dI,
peak_coverage, background_coverage,
center_H, center_K, center_L,
status_code
```

Add method-specific numeric columns such as Gaussian widths, reduced
chi-squared, and fit iteration count. Units must be populated.

`PointListData` currently supports float columns only. Store row status strings,
messages, peak labels, and the numeric status-code mapping in metadata:

```text
row_status: list[str]
row_messages: list[str]
status_codes: dict[str, int]
```

Required statuses:

- `ok`
- `partial_coverage`
- `edge_rejected`
- `insufficient_background`
- `overlap`
- `fit_failed`
- `no_measured_bins`

The principal channel is `Integrated intensity` with value `I` and error `dI`.
Add `I/dI` and `Background` as additional channels where useful.

## 13. Spectral integration operation

### Scientific reference convention

nfit uses energy transfer `E = hbar*omega` in meV. With `chi''` expressed per
energy, registered positive-energy kernels are:

```text
total moment:
  K_m(E,T) = coth[E/(2 k_B T)] / pi

QFI density:
  K_QFI(E,T) = 4 tanh[E/(2 k_B T)] / pi

static susceptibility:
  K_KK(E) = 2 / (pi E)
```

The QFI kernel is the finite-temperature dynamic-susceptibility relation of
Hauke et al., Nature Physics 12, 778-782 (2016), used for neutron data by
Scheie et al., Phys. Rev. B 103, 224434 (2021). Those papers use frequency,
energy, and natural-unit notation in slightly different ways. nfit removes the
ambiguity by defining `chi''(Q,E)` per unit energy and integrating with `dE`,
where `E = hbar*omega` is stored in meV. Under that declared convention the
kernel above contains no additional hbar factor. Tests must verify units and a
direct numerical quadrature, not merely copy a typeset prefactor.

The total-moment positive-energy form matches `trace_moment_quadrature` in
`sum_rules.py`. Its full-zone normalization and component basis must remain
explicit in the result metadata.

### Kernel registry

Define `SpectralKernelDefinition` with:

```text
key, label, formula, requires_temperature, required_representation,
output_kind, output_unit, parameters, evaluator
```

Initial kernels:

- `total_moment`
- `qfi`
- `static_susceptibility`
- `energy_moment` with nonnegative integer power
- `weighted_integral_arbitrary_units`

The first three require absolute, declared units. The arbitrary-unit kernel is
the only one permitted for uncalibrated data.

### Parameters

```text
kernel                       registered kernel key
input_channel                signal by default
spectral_convention          complete convention payload
temperature_source           dataset | fixed
temperature_K                used when fixed
energy_min_meV               >= 0
energy_max_meV               > minimum
elastic_exclusion_meV        >= 0
positive_energy_policy       positive_only | fold_two_sided_S
form_factor_ion              registry ion or custom
custom_form_factor           optional coefficients
polarization_mode            explicit mode from Section 10
polarization_scalar          when custom_scalar
g_factor                     when needed for spin-unit QFI
spin_S                       optional, required only for normalized QFI
zone_mode                    none | automatic | explicit_centers
zone_centers_hkl             list of HKL centers when explicit
zone_basis_hkl               optional magnetic/override reciprocal basis
zone_subvoxel_samples        odd integer, default 3
minimum_energy_coverage      default 0.9
minimum_zone_coverage        default 0.8
partial_zone_policy          report | reject
```

### Stable spectral conversion

Normalize each valid bin to one declared `chi''` representation before applying
a physical kernel. Use the inverse cross-section path when starting from
measured intensity. Use:

```text
chi'' = pi * [1 - exp(-E/k_B T)] * S(Q,E)
```

when starting from nfit's `S(Q,E)` convention. Handle the small-E limit with
stable `expm1` expressions. Apply form-factor and polarization corrections only
when their convention fields say they remain included.

Refuse divisions through form-factor values below a configurable numerical
threshold. Mask those bins, report their coverage loss, and show a warning.

### Energy reduction

Use exact overlap widths between energy bins and the requested energy window.
For every remaining momentum bin:

```text
value = sum_i K(E_i,T) * chi''_i * DeltaE_i
var   = sum_i [K(E_i,T) * sigma_chi_i * DeltaE_i]^2
```

Use bin-center quadrature initially. Add trapezoid mode only after it has tests
for irregular bins and masked gaps.

Report both:

- geometric energy coverage fraction;
- weighted coverage fraction, when the kernel has a finite positive weight.

If coverage is below threshold, mask the output bin. Do not rescale missing
energy by default.

Select the dataset/result type after removing the energy axis:

- no remaining coordinate axes: store a scalar result with value, uncertainty,
  coverage, and unit;
- one remaining coordinate axis: store `PointListData` with the coordinate,
  QFI, `dQFI`, energy coverage, and weighted coverage columns/channels;
- two or more remaining coordinate axes: store `MDHistoData` whose `signal` is
  QFI and whose `errors` is the propagated one-sigma uncertainty.

This rule is required because the current histogram viewer rejects
one-dimensional `MDHistoData`, while its point-list path supports curves. All
remaining coordinate and projection metadata must be preserved so every output
opens directly in the existing data viewer. A later, separately tested viewer
enhancement may add native one-dimensional histogram support; the analysis
operation must not depend on that enhancement.

### Auxiliary histogram channels

Generalize viewer metadata channels rather than misusing `num_events`. Preferred
implementation:

```python
@dataclass
class MDHistoChannel:
    values: np.ndarray
    errors: np.ndarray | None = None
    label: str = ""
    unit: str = ""

MDHistoData.auxiliary_channels: dict[str, MDHistoChannel] = field(default_factory=dict)
```

All channel arrays must match `signal.shape`. Extend NPZ save/load and the Qt
viewer channel selector. QFI histogram maps add:

- `energy_coverage`
- `weighted_coverage`
- optionally `uncorrected_integral`

The viewer's built-in signal/error/mask/event channels remain unchanged.

### Brillouin-zone geometry

Automatic mode must identify the reciprocal lattice of the Bravais lattice,
not merely every integer conventional-cell HKL point.

Implementation algorithm:

1. Obtain centering translations from the resolved space group.
2. A conventional integer HKL vector is a reciprocal-lattice vector when
   `H dot t` is integral for every centering translation `t`.
3. Enumerate short allowed integer vectors and select a right-handed independent
   triple whose absolute integer determinant equals the centering multiplicity.
   Prefer the triple with the smallest Q-metric norm and deterministic
   lexicographic tie-breaking.
4. Verify the selected basis generates every allowed vector in a bounded test
   neighborhood.
5. Generate reciprocal-lattice centers covering the measured Q bounds plus one
   neighbor shell.
6. Assign each Q sample to the nearest center in inverse-angstrom Euclidean
   metric. This is the Wigner-Seitz Brillouin-zone partition. Resolve exact ties
   deterministically by center HKL order.

Tests are mandatory for P, I, F, A/B/C, and rhombohedral centerings. An explicit
`zone_basis_hkl` overrides chemical-lattice periodicity for magnetic Brillouin
zones or unusual conventions.

### Zone integration

First compute the energy-reduced total-moment density at every Q bin. Then use
deterministic Q-subvoxel sampling to estimate each voxel's fractional membership
in each Brillouin zone.

For every zone report:

```text
center_H, center_K, center_L,
moment_partial, dmoment_partial,
moment_coverage_corrected, dmoment_coverage_corrected,
coverage, measured_Q_volume, ideal_BZ_volume,
accepted, status_code
```

Definitions:

- `moment_partial` divides measured accumulated weight by the full ideal BZ
  volume; missing volume contributes no invented signal.
- `moment_coverage_corrected` divides by measured volume and is explicitly an
  estimate assuming measured and missing volume have the same mean response.
- The primary `moment` alias equals the full-zone value only when coverage meets
  threshold. Partial zones remain labeled partial.

Return the per-zone table as `PointListData`. Also return a scalar weighted mean
and uncertainty over accepted zones. Between-zone scatter is a separate
diagnostic and must not be folded into counting uncertainty without labeling it.

### QFI normalization

Return raw `f_Q(Q)` in the declared operator convention. If `spin_S` is given,
also provide normalized QFI following the chosen published convention. Record
the exact normalization formula in metadata. Do not infer spin from the ion
label and do not emit multipartite-entanglement claims unless all required
absolute/component conventions are satisfied.

## 14. Generic background subtraction and raw-TOF compatibility

The operation model must allow multiple inputs from day one. This is the seam
for later `dataset_subtraction`:

```text
sample reduced dataset
- scale * background reduced dataset
= derived background-subtracted dataset
```

That future operation must align/rebin axes, propagate variance by addition,
combine masks/coverage, and store its scale provenance. Bragg shell subtraction
remains local to the Bragg measurement and is not a replacement for this
general operation.

Future raw TOF processing should be represented as upstream operations:

```text
raw files
  -> detector correction / normalization
  -> coordinate conversion
  -> reduced MDHisto or point/event dataset
  -> general background subtraction
  -> Bragg or spectral analysis
  -> optional fitting
```

The analysis runner therefore needs progress, cancellation, artifact outputs,
multi-input recipes, and fingerprints now. No initial Bragg/spectral code should
know an instrument name or import Mantid.

## 15. Data Playground GUI

Implement one non-modal `QMainWindow` owned by `NfitProjectExplorer`. Reuse the
main application and close it when its workspace is deleted or the project is
closed.

### Project-tree integration

Each workspace renders:

```text
Workspace
  Datasets
  Analyses
    Bragg integration 1
      Peak table
    Spectral integration 1
      QFI curve or map
      Total moment by zone
  Models
  Fits
```

Dataset-like outputs also appear in `Datasets / Derived data`. The output child
under `Analyses` and the derived dataset node refer to the same dataset ID.

Context actions:

- Dataset/workspace: `Open in Data Playground`.
- Analyses folder: `New analysis`.
- Analysis: `Run`, `Open in Data Playground`, `Duplicate`, `Rename`, `Delete`.
- Analysis output: `Open in data viewer`, `Export`, `Show provenance`.

Tree status styling must distinguish never run, running, fresh, stale, failed,
and unavailable artifact. Do not encode status by color alone; include status
text/icon and an accessible tooltip.

### Window layout

- Left: analysis selector/list and input-dataset list.
- Center: tabs for `Results`, `Diagnostics`, and `Provenance`.
- Right: scrollable registry-driven parameter editor.
- Bottom/right command row: Run/cancel, open output, export.

Do not put cards inside cards. Tool-specific editors are stacked widgets inside
the same window.

### Bragg editor

- Peak source and optional peak-table dataset selector.
- Method segmented control.
- Coordinate frame and widths/semiaxes editors.
- Energy window when needed.
- Background enable toggle and shell controls.
- Coverage/edge policy.
- Run button.
- Sortable results table.
- Selecting a result row enables `View peak`, which opens the existing data
  viewer centered on that HKL. ROI overlays may be a later GUI increment; the
  numerical result must not depend on them.

### Spectral editor

- Kernel selector.
- Intensity convention section.
- Temperature, energy window, form factor, polarization, and unit controls.
- Zone controls shown only for zone-reducing kernels.
- Run button.
- Scalar/per-zone summary and derived-output list.
- `Open output` launches curves and maps in the existing viewer.

### Task execution

Extract or generalize the current background worker/progress dialog rather than
copying the fit worker. The analysis worker must:

- run outside the GUI thread;
- report named stage, completed/total work, message, and optional diagnostics;
- support cooperative cancellation;
- never mutate the project until execution and artifact writes succeed;
- restore all controls on success, failure, or cancellation.

## 16. Script API

Export these from `nfit`:

```python
AnalysisEntry
AnalysisResultRecord
AnalysisOutputRef
AnalysisContext
available_analysis_types
default_analysis_parameters
run_analysis_operation
run_project_analysis
integrate_bragg_peaks
spectral_energy_reduce
integrate_total_moment_by_zone
qfi_from_chipp
```

Pure numerical functions accept containers/arrays and explicit context. Project
functions resolve IDs, prepared dataset views, artifacts, and stale state.

Provide examples in `examples/` for synthetic Bragg integration and synthetic
QFI/total-moment integration. Examples must run with the explicit nfit conda
interpreter and use no local SEQUOIA files.

## 17. Performance requirements

- Bragg integration must slice a bounding region per peak. It must not scan or
  allocate a full 4D coordinate grid for every peak.
- Cache coordinate transforms, axis edges, measured masks, and bin-volume terms
  once per operation run.
- Chunk peak batches and Brillouin-zone subvoxel assignment.
- Energy reduction should reduce along the energy axis using broadcasting and
  bounded blocks; peak temporary memory should scale with output Q size, not an
  additional full set of H/K/L grids.
- Expose progress frequently enough for the GUI to update at least several times
  per second on long jobs, but do not call progress once per voxel.
- Check cancellation between peak fits and between numerical blocks.
- Add `benchmarks/benchmark_analysis.py` with synthetic inputs and peak/BZ
  workloads. Benchmarks are descriptive, not timing assertions in pytest.
- Add memory estimates and a GUI warning before an operation expected to exceed
  70% of available RAM, following current MDEvent behavior.

## 18. Required tests

### Core and persistence

- Dataset IDs survive rename, move, save/load, and project migration.
- Copying creates a new dataset ID.
- Duplicate IDs are rejected or repaired deterministically with a warning.
- Analysis recipes round-trip through project JSON.
- Version-1 projects load and receive IDs without losing fields.
- Unknown analysis types round-trip unchanged.
- Artifacts save/load every supported scalar and dataset output type.
- Atomic failure leaves the previous successful result intact.
- Missing artifact marks stale/unavailable without blocking project load.
- Recipe/input changes mark stale; display-only changes do not.
- Save As copies assets and preserves IDs/references.

### Bragg numerics

- Uniform-density box integral equals density times exact volume.
- Constant shell background subtracts to zero with analytic uncertainty.
- A peak plus constant background recovers known box/ellipsoid intensity.
- Synthetic rotated 3D Gaussian recovers center, covariance, integrated
  intensity, and uncertainty within tolerances.
- Singular/undersampled Gaussian returns a diagnostic failure.
- Configured Gaussian fallback is explicit in status/metadata.
- Non-orthogonal lattice gives correct Q, d spacing, and volume Jacobian.
- Masked bins reduce coverage and do not count as zero.
- Covered zero-event bins remain valid measurements.
- Edge policy reject/report behavior is locked.
- Neighbor peak regions are excluded from background.
- 4D input respects fractional energy-window overlap.
- Results are deterministic across runs and peak ordering.

### Spectral numerics

- Forward intensity and inverse correction round-trip.
- `S` to `chi''` conversion satisfies detailed balance and small-E limits.
- Form factor and polarization corrections are applied exactly once.
- Invalid/near-zero form factors are masked and reported.
- Numerical total moment matches `trace_moment_quadrature` on known Lorentzians.
- Numerical QFI matches direct high-accuracy quadrature on known Lorentzians.
- QFI tends to the correct zero-temperature kernel limit.
- Static-susceptibility kernel matches `kk_static_chi`.
- Irregular energy-bin widths and partial boundary bins are correct.
- Masked energy gaps lower coverage and are not interpolated.
- QFI output removes only the energy axis, preserves projection metadata, and
  chooses scalar, `PointListData`, or `MDHistoData` according to the number of
  remaining axes.
- Auxiliary coverage channels save/load and appear in the viewer.
- P/I/F/A/B/C/R reciprocal bases and nearest-zone assignment are correct.
- Full synthetic zones reproduce a known constant moment.
- Partial-zone raw and coverage-corrected values remain distinct.
- Per-zone uncertainty and accepted-zone scalar combination are analytic.

### GUI

- Analyses folder and statuses render correctly.
- New/duplicate/delete/run actions update project state.
- Input selectors use dataset IDs and survive renames.
- Tool-specific controls appear only when relevant.
- Every interactable control has a tooltip.
- Running is asynchronous; cancellation and errors restore controls.
- Success creates/replaces stable derived dataset entries.
- Stale results are visibly stale and can be rerun.
- Bragg tables and QFI curves/maps open in the existing viewer.
- Save/reopen restores analysis entries and output availability.

### Full regression

Run:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pytest -q
/Users/pmneves/anaconda3/envs/nfit/bin/python -m compileall -q src/nfit tests examples
```

## 19. Implementation work packages

These packages are ordered by dependency. Agents should not begin a package
whose prerequisites have not landed.

### Package A: stable IDs and analysis domain model

Owner files:

- `src/nfit/pipeline.py`
- new `src/nfit/analysis/core.py`
- new `src/nfit/analysis/registry.py`
- focused tests in new `tests/test_analysis_core.py`

Deliver:

- Stable dataset IDs and copy semantics.
- Analysis dataclasses and registry.
- Default/tooltip/validation APIs.
- No GUI.

Acceptance: core tests pass and existing constructors remain compatible.

### Package B: project schema, artifacts, and fingerprints

Prerequisite: A.

Owner files:

- new `analysis/artifacts.py`, `analysis/fingerprint.py`, `analysis/runner.py`
- project serialization sections of `project_gui.py`
- dataset NPZ save/load helpers, preferably moved to a non-Qt module
- `tests/test_analysis_persistence.py`

Deliver:

- Project schema v2 and v1 migration.
- Relative asset store and Save As behavior.
- Latest-result transaction semantics.
- Fresh/stale calculation.
- Derived dataset creation with stable IDs.

Acceptance: persistence tests pass, old project tests pass, missing artifacts are
nonfatal.

### Package C: coordinate and correction primitives

Prerequisite: A. May run in parallel with B if it does not edit
`project_gui.py`; integration refactoring happens after both land.

Owner files:

- new `analysis/coordinates.py`, `analysis/corrections.py`
- `cross_section.py`
- new `tests/test_analysis_coordinates.py`, `tests/test_analysis_corrections.py`

Deliver:

- Physical-axis/Jacobian/bin semantics utilities.
- Spectral convention validation.
- Forward/inverse cross-section round trip.
- Form-factor, polarization, and unit conversion helpers.

Acceptance: analytic tests pass; no Qt/project import.

### Package D: reciprocal zones

Prerequisite: C.

Owner files:

- new `analysis/zones.py`
- new `tests/test_analysis_zones.py`

Deliver:

- Centering-derived reciprocal basis.
- BZ center generation and nearest-center assignment.
- Subvoxel zone fractions and coverage.

Acceptance: all centering and constant-zone tests pass.

### Package E: Bragg engine

Prerequisites: C; registry contract from A. Artifact integration may be mocked
until B lands.

Owner files:

- new `analysis/bragg.py`
- new `tests/test_analysis_bragg.py`
- synthetic Bragg example

Deliver:

- Peak-list validation/generation.
- Box, fixed ellipsoid, shell background, Gaussian fit.
- Complete table/status metadata.
- Progress and cancellation.

Acceptance: all Bragg numerical tests pass without GUI.

### Package F: spectral engine

Prerequisites: C and D; registry contract from A.

Owner files:

- new `analysis/spectral.py`
- new `tests/test_analysis_spectral.py`
- synthetic spectral example

Deliver:

- Kernel registry.
- Stable conversion to `chi''`.
- Dimension-aware QFI scalar/curve/map, total-moment map, and per-zone
  table/scalar.
- Coverage channels and uncertainty propagation.

Acceptance: reference quadrature, BZ, coverage, and output-shape tests pass.

### Package G: viewer auxiliary channels

Prerequisite: F's output contract, but can begin from its agreed dataclass.

Owner files:

- `mdhisto.py`
- `plotting.py`
- `qt_slice_viewer.py`
- dataset archive helpers
- relevant plotting/MDHisto tests

Deliver:

- `MDHistoChannel` and auxiliary channel storage.
- NPZ round trip.
- Dynamic viewer selector and labels/units/errors.

Acceptance: existing viewer behavior remains unchanged; QFI coverage channels
are selectable.

### Package H: Data Playground GUI and tree integration

Prerequisites: A-G. This package is intentionally last because
`project_gui.py` is the conflict hotspot.

Owner files:

- new `analysis_gui.py`
- optional new `qt_tasks.py`
- `project_gui.py`
- `tests/test_analysis_gui.py` and focused project-GUI tests

Deliver:

- Analyses tree, window, registry form, status states, task execution.
- Derived-output links, tables, viewer launch, export, provenance.
- Complete tooltips.

Acceptance: all GUI tests and end-to-end synthetic workflows pass offscreen.

### Package I: integration, documentation, and performance

Prerequisites: all previous packages.

Owner files:

- `__init__.py`
- `docs/gui_workflows.md`, `docs/data_philosophy.md`, `docs/api.md`
- new user-facing `docs/data_playground.md`
- `docs/index.md`
- examples and benchmark
- packaging/version files

Deliver:

- Public exports and docs.
- End-to-end examples.
- Benchmark and memory review.
- Full test/compile run.

Acceptance: Definition of Done below.

## 20. Multi-agent coordination

Recommended execution:

```text
Wave 1: A
Wave 2: B and C in parallel
Wave 3: D and E in parallel after C; finish B
Wave 4: F, then G
Wave 5: H
Wave 6: I and final review
```

Do not let two workers edit `project_gui.py` concurrently. Do not let operation
workers invent alternative dataclasses or parameter names; `core.py` and this
document are the contract.

For parallel worktrees, implementation workers should preferably leave changes
uncommitted for the integration worker. If workers create commits, every commit
must follow `AGENTS.md` and bump `pyproject.toml` plus `docs/conf.py`; coordinate
versions to avoid conflicts. The integration worker owns final versioning,
documentation reconciliation, and the full suite.

Each worker handoff must contain:

- files changed;
- public APIs added;
- tests run and exact results;
- assumptions or deviations;
- unresolved failures;
- whether project schema or artifact layout changed.

## 21. Definition of Done

The analysis implementation is complete only when:

- A saved/reopened project preserves analysis recipes, results, stable IDs, and
  derived outputs.
- Editing an input or recipe visibly marks results stale.
- A synthetic Bragg dataset can produce the requested H, K, L, I, dI,
  Background, and I/dI table with correct uncertainty and coverage.
- Synthetic absolute spectra can produce scalar, one-dimensional curve, and
  multidimensional-map QFI outputs; the curve and map open in the existing
  viewer.
- The same spectrum can produce a per-BZ total-moment table and accepted-zone
  scalar with coverage diagnostics.
- All computation works from script APIs without constructing Qt objects.
- Long work runs asynchronously and can be cancelled without partial mutation.
- User-facing docs state every correction and normalization assumption.
- Every new GUI control has a tooltip test.
- Existing fitting, rebinning, MDEvent, report, and viewer workflows remain
  green.
- Full pytest and compileall commands pass in the local nfit conda environment.

## 22. Scientific and algorithm references

- P. Hauke et al., "Measuring multipartite entanglement through dynamic
  susceptibilities," Nature Physics 12, 778-782 (2016), author manuscript:
  https://strathprints.strath.ac.uk/55904/1/Hauke_etal_NP2016_measuring_multipartite_entanglement_dynamic_susceptibilities.pdf
- A. Scheie et al., "Witnessing entanglement in quantum magnets using neutron
  scattering," Phys. Rev. B 103, 224434 (2021):
  https://doi.org/10.1103/PhysRevB.103.224434
- Author manuscript with QFI formula and data-processing discussion:
  https://arxiv.org/pdf/2102.08376
- Mantid `IntegratePeaksMD` documentation for established ellipsoid/shell
  integration and background-error propagation:
  https://docs.mantidproject.org/v6.5.0/algorithms/IntegratePeaksMD-v2.html
- nfit internal total-moment convention:
  `docs/theory_notes.md` and `src/nfit/sum_rules.py`.

These references establish formulas and validation targets. nfit's
implementation remains native and must not add Mantid as a runtime dependency.
