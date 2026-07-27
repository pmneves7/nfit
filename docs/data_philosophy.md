# Data and extension conventions

nfit separates instrument-specific import and reduction from analysis in
physical coordinates. Once a dataset contains axes such as `H`, `K`, `L`,
`|Q|`, or energy transfer, the fitting layer does not depend on the originating
instrument or file format.

## Architecture

- **Importers are adapters.** They translate file metadata and arrays into
  common data containers. Instrument-specific rules stay in the importer.
- **Axes retain their source meaning.** Importers preserve names and units when
  possible, then assign broad coordinate roles used by plotting and fitting.
- **Analysis is non-destructive.** Operations create derived datasets and
  record recipe hashes, input fingerprints, and relative artifact manifests.
- **Project state is inspectable.** `.nfit` files are JSON documents containing
  datasets, model sessions, fit results, plots, and analysis provenance.
- **GUI actions are scriptable.** Scientific operations use the same package
  functions from the GUI and Python, rather than storing behavior only in
  widget state.
- **Resolution belongs to a dataset.** A resolution model may depend on the
  instrument and settings, but the optimizer receives only coordinates,
  observations, uncertainties, masks, weights, and predictions.

For example, the GUI's data-group workflow corresponds to ordinary package
calls:

```python
from nfit import NfitProject, create_data_group, import_dataset_paths, save_project

project = NfitProject()
group = create_data_group(project, "field_series")
import_dataset_paths(group, ["low_field.nxs", "high_field.nxs"])
save_project(project, "field_series.nfit")
```

## Shared registry metadata

Masks, model components, and other registry-backed features declare defaults,
types, allowed values, examples, and short descriptions in one place. Scripts,
project files, validation, and GUI tooltips therefore use the same contract.
Values remain JSON-compatible unless the common project format is extended
deliberately.

Every interactive control needs hover text that explains its effect and any
important constraint. A behavior change must update the relevant source page
and tests at the same time. [GUI workflows](gui_workflows.md) is the canonical
description of the project explorer and data viewer.

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

## File-backed event data

MDEvent imports do not require Mantid at runtime. A group owns shared
crystallographic orientation, detector mask, normalization, and optional
incident-energy or time-zero overrides. Run entries contain run-specific
metadata and references to event files. Reduction streams event chunks and
converts stored `Q_sample` vectors with $(2\pi UB)^{-1}$, avoiding duplicate
copies of event and instrument data.

Normalized output is a ratio. Corrected event signal and error-squared form the
numerator; detector trajectories, proton charge, and energy coverage form the
denominator. Raw direct-geometry events receive the instrument-defined
wavelength-dependent detector-efficiency correction and the $k_i/k_f$
kinematic correction. In SHIVER-compatible imports, vanadium supplies a binary
detector mask rather than signal weights. nfit masks are applied afterward
through the ordinary group mask path.

Detector coverage and event count are distinct. A covered bin with no events
is a measured zero; a bin without detector coverage is masked. Because an empty
bin has no event variance, nfit assigns it 1.29 times the estimated uncertainty
of one representative event in that bin. This is the 68.27% Feldman–Cousins
upper endpoint for zero observed events and zero known background. See
[Measured-zero uncertainties](gui_workflows.md#measured-zero-uncertainties)
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

Unimplemented directions are listed separately in
[Planned features](planned_features.md).
