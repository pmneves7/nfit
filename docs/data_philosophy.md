# Data and extension conventions

nfit separates instrument-specific import and reduction from analysis in
physical coordinates. Once a dataset contains axes such as `H`, `K`, `L`,
`|Q|`, or energy transfer, the fitting layer does not depend on the originating
instrument or file format.

## Architecture

- **Importers are adapters.** They translate file metadata and arrays into
  common data containers. Instrument-specific rules stay in the importer.
- **Detection uses evidence.** Instrument adapters probe internal format and
  instrument metadata with higher confidence than a filename extension. A
  multi-stream acquisition may declare named outputs that become separate
  datasets with shared source provenance; incompatible observables are never
  silently added together.
- **Axes retain their source meaning.** Importers preserve names and units when
  possible, then assign broad coordinate roles used by plotting and fitting.
- **Analysis is non-destructive.** Operations create derived datasets and
  record recipe hashes, input fingerprints, and relative artifact manifests.
  Clone and histogram-arithmetic datasets remain virtual source-linked recipes
  so their independent masks, bins, and symmetry can be applied to underlying
  data without materializing an intermediate source histogram.
- **Project state is portable and inspectable.** A `.nfit` file is a ZIP
  container whose `project.json` manifest records datasets, model sessions, fit
  results, plots, and analysis provenance. Array-valued analysis outputs and
  explicitly materialized composites are stored under `assets/` in the same
  file; large composite materializations load lazily.
- **Dataset identity is unique.** Importing or copying a dataset assigns a new
  ID, even when the source file is the same. Project loading rejects duplicate
  IDs because analysis and background references would otherwise be ambiguous.
- **Scientific actions belong in public APIs.** Dataset preparation, fit, and
  saved-plot workflows export editable scripts. Other GUI workflows should call
  the same package functions as Python code and must not hide scientific
  choices only in widget state. Current coverage and remaining gaps are listed
  in [Workflow scripts](workflow_scripts.md) and
  [Planned features](planned_features.md).
- **Resolution belongs to a dataset.** A resolution model may depend on the
  instrument and settings, but the optimizer receives only coordinates,
  observations, uncertainties, masks, weights, and predictions.
- **External model engines are adapters.** Project state records the source,
  adapter, version, conventions, and portable dependency closure rather than
  serializing an external Python object. The planned electronic-model contract
  is defined in
  [Electronic-response design contract](electronic_response_contract.md).

For example, the GUI's data-group workflow corresponds to ordinary package
calls:

```python
from nfit import NfitProject, create_data_group, import_dataset_paths, save_project

project = NfitProject()
group = create_data_group(project, "field_series")
import_dataset_paths(group, ["low_field.nxs", "high_field.nxs"])
save_project(project, "field_series.nfit")
```

For an iterative edit of an existing project, use the transactional project
editor rather than loading and saving the archive independently:

```python
from nfit import edit_project_file

with edit_project_file("field_series.nfit") as project:
    background = project.data_groups[0].models["Background"]
    background.parameters["constant"] = 1.25
```

On successful exit, `edit_project_file` reconciles the live workspace with fit
history, validates that the active snapshot reproduces the live state, and
atomically saves the archive. A lone **Initial** snapshot and a **Current
state** snapshot are updated in place. Editing from a completed result creates
a current state or branch and never rewrites the result. Use
`project_state_issues(project)` for a non-raising consistency report or
`validate_project_state(project)` when a mismatch should stop a script.

The GUI treats the serialized live workspace as authoritative on open; a
remembered fit path selects its tree entry without restoring the historical
snapshot. It also detects external file replacement and requires an explicit
reload, Save As, keep, or overwrite decision before either version is lost.

## Shared registry metadata

Masks, model components, and other registry-backed features declare defaults,
types, allowed values, examples, and short descriptions in one place. Scripts,
project files, validation, and GUI tooltips therefore use the same contract.
Values remain JSON-compatible unless the common project format is extended
deliberately.

Models use the public `ModelDefinition` registry. In addition to fields and
compatibility, a definition may provide numerical and Jacobian factories,
dynamic parameter discovery, diagnostics, report sections, model-owned plots,
citations, and separate project/workflow serializers. See
[Modeling and fitting pipeline](modeling_pipeline.md#shared-model-registry).

Every interactive control needs hover text that explains its effect and any
important constraint. A behavior change must update the relevant source page
and tests at the same time. The pages linked from
[GUI workflows](gui_workflows.md) describe the project explorer and data viewer.

## Data-container contract

The numerical payloads in `PointData4D`, `PointListData`, and `MDHistoData` are
read-only. This prevents a view, analysis, or script from changing an array
without invalidating cached results and provenance.

Use `with_updates(...)` for a direct replacement. For several in-place edits,
make an isolated writable copy and install the result:

```python
editable = dataset.data.mutable_copy()
editable.mask[bad_bins] = True
dataset.replace_data(editable)
```

`replace_data` converts the result back to the canonical read-only form and
advances the dataset's process-local data revision. Viewer, composite, model,
and analysis caches use this revision. Do not assign `dataset.data` directly.
Importer and lazy-loading code uses `source_backed=True` only when the loaded
arrays exactly represent `metadata["source_file"]`.

Project files store source references rather than arbitrary in-memory arrays.
If a script replaces source-backed data, save the result as a portable `.npz`
dataset and re-import it before saving the project. This makes the replacement
reproducible instead of silently reloading the old source on the next session.

Discrete metadata axes store exact coordinates in
`MDHistoAxis.metadata["discrete_centers"]`; `axis.centers` returns those values.
Their `axis.values` boundaries are display cells, not physical interpolation
or integration intervals. The originating `MetadataDimension` recipe is kept
with each axis and with the collection. Composite caches include both the
dataset content token and resolved metadata-coordinate signatures.

An optional `MetadataBinning` recipe groups assigned coordinates into histogram
bins using whole-bin assignment. These axes retain `metadata_dimension` and
`interpolation="none"` metadata, but use physical bin edges and their midpoints
instead of `discrete_centers`. Source points are partitioned before spatial
reduction so coarse metadata bins preserve source weights and uncertainties.

Per-point metadata coordinates must already align with the measured payload.
The MACS adapter records its scan/detector shape so aligned scan columns can be
broadcast over detector channels. Future event adapters must explicitly align
sample logs to events before using this coordinate-assignment API.

## Mask contract

A dataset mask is a `MaskSpec` with:

- `name`: an identifier unique within the dataset;
- `type`: a registered kind such as `coordinate_range`, `box`, `ellipsoid`, or
  `phonon_cone`;
- `parameters`: the values needed to reconstruct the mask;
- `enabled`: whether the mask participates in plotting and fitting;
- `metadata`: optional provenance or interface annotations.

GUI masks use exclusion semantics: their parameters describe the region to
remove. Defaults are inert. An interval `[0, 0]`, zero geometric extent, or zero
phonon-cone slope masks nothing. `invert` and `additive` explicitly modify this
baseline, and disabled masks are ignored. A phonon-cone center may be one
`[H, K, L]` vector or a list; multiple centers mask the union of the cones.

## Model-component contract

A data group stores model components as `ModelComponentSpec` entries:

- `name`: an identifier unique within the group;
- `type`: a registered model kind;
- `parameters`: current parameter values;
- `fit_parameters`: which parameters are varied;
- `sharing`: global, per-dataset, or grouped parameter sharing;
- `enabled`: whether the component contributes to the prediction;
- `config` and `metadata`: model configuration and provenance.

The prediction is the sum of enabled components. Parameter links, expressions,
and priors use the common fit specification described in
[Fit constraints](fit_constraints.md), not GUI-only state.

The component type resolves through `model_definition(type)`. Model-specific
post-fit diagnostics are stored with the fit result; report and plotting
providers consume that stored or reproducible scientific state. Custom
serializers must return JSON-compatible component dictionaries and preserve the
registered type key.

## File-backed event data

MDEvent imports do not require Mantid at runtime. A group owns shared
crystallographic orientation, detector mask, normalization, and optional
incident-energy or time-zero overrides. Run entries contain run-specific
metadata and references to event files. Reduction streams event chunks and
converts stored `Q_sample` vectors with $(2\pi UB)^{-1}$, avoiding duplicate
copies of event and instrument data.

Here UB is the sample orientation times the crystallographic reciprocal
basis without $2\pi$, `Q_sample` has physical Å$^{-1}$ coordinates, and the
result of $(2\pi UB)^{-1}\mathbf Q_{\rm sample}$ is dimensionless HKL.
See [UB matrices](data_import.md#ub-matrices) for the convention.

Normalized output is a ratio. Corrected event signal and error-squared form the
numerator; detector trajectories, proton charge, and energy coverage form the
denominator. Raw direct-geometry events receive the instrument-defined
wavelength-dependent detector-efficiency correction and the $k_i/k_f$
kinematic correction. In SHIVER-compatible imports, vanadium supplies a binary
detector mask rather than signal weights. nfit masks are applied afterward
through the ordinary group mask path.

The native MDEvent reducer can accumulate either a projected four-dimensional
HKLE grid or a radial $|\mathbf Q|,E$ grid. Radial reduction uses each event's
momentum magnitude and integrates the same detector trajectories through
powder bins; sample-goniometer rotations do not affect $|\mathbf Q|$.

Detector coverage and event count are distinct. A covered bin with no events
is a measured zero; a bin without detector coverage is masked. Because an empty
bin has no event variance, nfit assigns it 1.29 times the estimated uncertainty
of one representative event in that bin. This is the 68.27% Feldman–Cousins
upper endpoint for zero observed events and zero known background. See
[Measured-zero uncertainties](data_import.md#measured-zero-uncertainties)
for the calculation and [References](references.md) for the citation.

## Adding an importer or operation

An extension should:

1. preserve source units and provenance;
2. return a common dataset or analysis-result type;
3. propagate one-sigma uncertainties;
4. keep instrument-specific assumptions at the adapter boundary;
5. provide the same callable operation to scripts and the GUI;
6. serialize all scientific choices needed to reproduce the result; and
7. include validation, tests, tooltips, and user documentation.

Register a content probe when the format has a reliable internal signature.
Filename extensions are only a compatibility fallback because `.nxs` can hold
several distinct NeXus application definitions and instrument layouts. Declare
named streams when one source contains scientifically different measurements;
each stream must specify its data type and serialize the option that selects it.

Unimplemented directions are listed separately in
[Planned features](planned_features.md).
