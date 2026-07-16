# Reduced-data philosophy

`nfit` should be organized around reduced experimental coordinates and
models, not around any one instrument, file format, or Mantid workspace type.

Analysis operations are immutable recipes over stable dataset IDs. They do not
overwrite inputs; successful outputs return as derived datasets with recipe
hashes, input fingerprints, and relative artifact manifests. The same
provenance graph is intended to receive future raw-TOF reduction and general
background-subtraction outputs.

The common assumption is that data reduction has already converted raw detector
events into physically meaningful axes such as `H`, `K`, `L`, `|Q|`, energy
transfer `E`, intensity, and uncertainty. Once data are in those coordinates,
the fitting pipeline should not care whether the measurement came from HYSPEC,
another direct-geometry spectrometer, a triple-axis spectrometer, a powder
diffractometer, or a text export from an external workflow.

## Design principles

- Importers are adapters. A Mantid `MDHistoWorkspace` importer, a triple-axis
  text importer, and a powder diffraction importer should all translate file
  metadata into common data containers rather than forcing the rest of the
  package to know about the instrument.
- Axis names and units should be read from the file whenever possible. If a file
  labels an axis as `DeltaE`, `|Q|`, `H`, `[H,H,0]`, or similar, the importer
  should preserve that label and infer a broad axis role from it.
- Instrument-specific conventions should stay near the importer. For example,
  HYSPEC HHL data need a projection-specific mapping from `[H,H,0]` and
  `[H,-H,0]` to `(H,K,L)`. That belongs in a HYSPEC adapter/helper, not in the
  general fitting framework.
- `DataGroup`, `DatasetEntry`, and `FitModelSession` are the generic fitting
  layer. They may hold MDHisto data, point data, susceptibility curves, powder
  cuts, or later polarized-neutron containers.
- Resolution functions are per-dataset model components. They can depend on
  instrument and settings, but the optimizer sees only model predictions,
  observed data, uncertainties, masks, and weights.
- GUI workflows should be backed by scriptable operations. Any task a user can
  do through the interface should have a human-readable Python equivalent that
  can be edited, repeated, and version controlled. This is especially important
  for specialized analyses, repetitive imports, parameter sweeps, and shared
  fitting recipes. GUI code should call the same small functions a script would
  call rather than hiding scientific state changes inside widgets.
- Saved project files should be readable and inspectable whenever practical.
  Opaque binary state can be useful as a cache, but the primary project
  description should preserve enough text structure for users to understand and
  adjust data groups, source files, model settings, and fitting choices.
- Every user-interactable GUI control should have useful hover text. The tip
  should explain what the control changes, when the change takes effect, and
  any important constraints or examples. For registry-backed controls such as
  mask parameters, model parameters, model configuration, optimizers, and future
  resolution models, the GUI should render tooltip text from the same metadata
  that defines scriptable defaults and validation. New GUI features should add
  tests or audit coverage so controls are not introduced silently without
  hover help.
- User-facing behavior changes should update documentation at the same time as
  code. The Sphinx source pages in `docs/` are the canonical in-repository
  documentation, and `docs/gui_workflows.md` is the wiki-style page for the
  current project explorer and data-viewer workflows.

For example, the project explorer's current data-group workflow is intentionally
mirrored by scriptable calls:

```python
from nfit import NfitProject, create_data_group, import_dataset_paths, save_project

project = NfitProject()
group = create_data_group(project, "field_series")
import_dataset_paths(group, ["low_field.nxs", "high_field.nxs"])
save_project(project, "field_series.nfit")
```

## Current state

The current examples use Mantid MDHisto NeXus files because those are the first
local data available in the repository. That is a starting adapter, not a
package boundary. The same fitting framework is intended to support reduced
text tables, one-dimensional cuts, powder averages, and other reduced neutron
or bulk-measurement data as additional importers are added.

The first project explorer GUI follows this direction by using package-level
project functions for creating data groups, importing dataset paths, deleting
entries, and saving/loading projects. Its `.nfit` files are JSON documents for
the currently supported project state. Model sessions and embedded fit results
will need matching text-oriented serialization as those GUI features are added.
Slice-viewer integration should follow the same pattern: the GUI can open or
refresh a viewer from a selected `DataGroup`, but the viewable datasets and
fit-result overlays should come from scriptable data structures such as
`MDHistoData` plus attached `FitComparisonModelView` metadata. A script that
fits data and calls `attach_fit_comparisons(...)` should therefore produce the
same slice-viewer comparison controls that the GUI exposes.

## Mask Standard

Masks are first-class analysis specifications, not ad hoc GUI annotations. A
dataset mask is represented by a `MaskSpec` with:

- `name`: user-facing identifier, unique within one dataset.
- `type`: registered mask kind, such as `coordinate_range`, `energy_q_range`,
  `box`, `ellipsoid`, or `phonon_cone`.
- `parameters`: JSON-like values needed to rebuild the mask from a script.
- `enabled`: whether the mask participates in downstream analysis.
- `metadata`: optional provenance or UI annotations that do not change mask
  semantics.

Each mask type must declare its parameter names in the shared mask registry so
scripts, project files, and the GUI editor agree on the same fields. Parameter
values should stay human-readable and serializable: numbers, strings, booleans,
lists, and dictionaries are preferred over opaque objects. If a new mask type
needs a richer representation, update the common standard deliberately and keep
all existing mask types consistent with it.

Registry-backed GUI masks use exclusion semantics: parameters identify the
region to remove from viewing and fitting. Their defaults must be inert. Range
parameters use `[0, 0]` for an inactive dimension; zero geometric extent or
zero phonon-cone slope similarly masks nothing. `invert` and `additive` are
explicit modifiers to this baseline convention, and disabled masks are ignored.
The phonon-cone `center` parameter may be one `[H, K, L]` vector or a list of
such vectors; multiple centers mask the union of equivalent cones around a
series of Bragg peaks.

Each mask parameter should also declare a default value and standard hover text
metadata: a short description, allowed values, data type, and example. Defaults
should be safe starter values, ideally masking no data or almost no data, so the
user can inspect a baked-in example before narrowing it. The GUI should render
tooltips from the same registry that scripts use for defaults, and model or
resolution-model parameter editors should follow the same pattern when they are
added.

The same restriction should apply to future datasets, models, optimizers, and
resolution models: GUI controls edit documented, scriptable specifications;
scientific behavior should not depend on hidden widget state.

## File-backed MDEvent groups

MDEvent imports do not make Mantid a runtime dependency. A group owns shared
crystallographic orientation, detector mask, vanadium normalization, and
optional Ei/T0 overrides. Its run entries contain only run-specific metadata
and references into one or more event files. Reduction streams event chunks
and converts stored `Q_sample` vectors with `(2*pi*UB)^-1`, avoiding redundant
copies of the event table and instrument description.

The normalized result is a ratio. Event signal and error-squared accumulate in
the numerator after raw direct-geometry events receive the IDF-defined,
wavelength-dependent He-3 tube-efficiency correction and the kinematic
`ki/kf` correction. Detector trajectories contribute proton charge and energy
coverage to the denominator. In Shiver-compatible imports, vanadium supplies a
binary detector mask rather than a detector-value weight. Bad detectors
contribute to neither side. nfit masks are then applied through the normal
dataset-group mask pathway.

Detector coverage and event count remain separate. In particular, a covered
bin with zero events is a measurement of zero rather than an unmeasured bin.
An empty bin has no MDEvent row and therefore has no stored event variance of
its own. Its signal is retained as zero, while its uncertainty is set to `1.29`
times the uncertainty one representative event would have in that bin. This
factor is the 68.27% Feldman--Cousins upper endpoint for observing zero events
with zero known background (Table II of [Feldman and Cousins](https://arxiv.org/pdf/physics/9711021)).
The representative event scale is estimated from accepted events elsewhere in
the requested volume. Only bins without detector coverage are masked. The
complete formula and a numerical example are given in
[Measured-zero uncertainties](gui_workflows.md#measured-zero-uncertainties).

## Model Component Standard

Project-level model setup should use `ModelComponentSpec` entries attached to a
`DataGroup`. A data group may contain multiple model components; the intended
prediction is the sum of enabled components until a later model-composition
standard says otherwise. Each component records:

- `name`: user-facing identifier, unique within one data group.
- `type`: registered model kind, such as `constant_background`,
  `linear_background`, or `single_q_paramagnon`.
- `parameters`: JSON-like default/current parameter values.
- `global_fit`: per-parameter booleans indicating whether one shared value is
  fitted across all datasets (`True`) or whether each dataset may fit that
  parameter independently (`False`).
- `enabled`: whether the component participates in downstream fitting.
- `metadata`: optional provenance or UI annotations that do not change model
  semantics.

Like masks, each model parameter must declare a default value, description,
allowed values, data type, example, and default `global_fit` behavior in the
shared registry. Defaults should be reasonable starter values rather than hidden
scientific assumptions. More granular constraints, links, and priors should be
added by extending this common specification instead of inventing widget-local
state.
