# Workflow scripts

nfit workflow scripts are editable Python programs that reproduce scientific
state without opening the GUI. They describe operations—loading, preparing,
analyzing, fitting, and plotting—rather than replaying mouse clicks.

An export represents the state that is active when it is created. Fit-timeline
branches are convenient checkpoints for interactive work; a complete project
export does not rerun every historical branch.

## Dataset workflows

Right-click a dataset and choose **Copy workflow script** or **Save workflow
script...**. The generated script currently supports an ordinary source-backed
dataset and its:

- source importer and importer options;
- inherited and dataset masks;
- rebin settings;
- attached dataset backgrounds;
- fixed, fitted, or dataset-group-shared scale; and
- spectral-channel conversion.

The script separates `load_sources()` from `prepare_datasets()`. Configuration
is stored in ordinary dictionaries near the top of the file, so paths and
scientific settings can be changed directly. `build_workflow()` returns a
`WorkflowResult` with `source_datasets`, `prepared_datasets`, and
`analysis_results` attributes.

`SOURCE_ROOT` is the common source directory when possible. Relative source
paths are resolved from that location. File size and modification time are
recorded as a lightweight fingerprint; a changed source produces a warning
rather than silently claiming an exact reproduction.

## Native reduction workflows

`reduction_workflow_script(collection, binning_config=...)` exports an isolated
raw-DGS, CORELLI or MDEvent collection from original sources. The editable
`REDUCTION_RECIPE` and independent `REBIN_CONFIG` reproduce run membership,
shared defaults, per-run overrides, coordinate transforms, local masks and
scaling without a saved project or Qt. See
[Source and reduction recipes](reduction_recipes.md).

## Composite workflows

The collection's **Copy complete workflow** action calls
`composite_workflow_script(project, group_name, node_id=...)`. The script
reconstructs ordinary file-backed datasets and native DGS, CORELLI and MDEvent
collections from original sources. It retains nested collection topology,
ancestor masks and lattice, source order, disabled inputs, scales, resolution,
metadata axes and linked backgrounds, including sources in other workspaces.
Only the selected tree, its ancestors and background dependencies are included;
a saved project and Qt are not required.

The editable dictionaries separate the scientific operations:

- `COMPOSITE_RECIPE` contains workspace/collection structure, ordinary source
  descriptors and inherited settings. Each collection retains its own binning.
- `REDUCTION_RECIPES` contains native source membership, shared reduction
  defaults, per-run overrides, UB matrices, and native run/collection masks and
  background links.
- `METADATA_DIMENSIONS`, `REBIN_CONFIG` and `SCALING` control the selected output.
  Edit these to change its coordinate assignment, grid, symmetry, physical
  channel or scale independently of source reduction.

`build_group()` returns the reconstructed workspace and optional selected
collection; `run()` returns its histogram. The lower-level public
`export_composite_recipe(...)` and `replay_composite_recipe(...)` APIs expose
the same versioned recipe for batch programs. Original data and calibration
paths must remain accessible. Numerical caches, model state and historical
fit results are not embedded.

Source-less data, replaced numerical payloads, callable dataset transforms and
live derived-analysis inputs are rejected explicitly. Save such results as
portable dataset files and import those files to make a source-based composite
workflow. Standalone analysis and fit graph support has separate boundaries
described below.

A workspace-wide background is applied once to the requested result. Child
composites retain their own background links without reapplying the workspace
link during parent assembly. References use their own workspace lattice and
ancestor masks; a reference does not inherit the sample-wide subtraction that
it supplies. Genuine cycles through reference-local links are rejected.
Reference collection result scales and link scales are each applied once.

For an already loaded workspace and nested dataset collection:

```python
from nfit import MetadataDimension, composite_dataset_data, set_metadata_dimensions

set_metadata_dimensions(collection, [MetadataDimension(
    name="Temperature",
    source="entry/data/temp/average_value",
    units="K",
    sampling="dataset_mean",
    centers=[5, 10, 20, 30, 40, 50],
    tolerance=0.5,
)])
data = composite_dataset_data(workspace, node=collection)
```

The tolerance is in kelvin here. `metadata_channels(dataset)` lists available
numeric channels; `metadata_channel_values(dataset, source)` reads a channel
and its recorded units. With no explicit centers, exact unique coordinates
are retained. See [Metadata dimensions](data_import.md#metadata-dimensions)
for pointwise alignment rules and supported sources.

To rebin temperature, add `binning={"lower": 5, "upper": 50, "step": 5}` to
the dimension recipe, or `binning={"bin_edges": [0, 15, 25, 55]}` for explicit
edges. Add `"fractional": True` to a Step, Bins, or Edges recipe to distribute
coordinates between adjacent bins; the default is discrete assignment. The
script captures this optional binning along with the channel and nominal-coordinate
assignment. Leaving `binning=None` retains discrete values.

## Analysis workflows

Right-click a saved analysis and use the same copy or save actions. The script
includes every source and preparation step required by its input datasets,
then calls the public analysis registry. Analysis, viewing, and fitting share
one prepared-data convention: masks, rebinning, backgrounds, scale, and the
selected physical channel are applied before the analysis runs.

Grouped raw direct-geometry and MDEvent reductions, analyses that depend on
derived analysis datasets, workspace composites, and group backgrounds are not
yet expanded into standalone workflow-graph nodes. Saved live-composite
analyses and source-linked clone or histogram-arithmetic recipes instead export
a readable project-backed script. That script loads the saved recipe and calls
`derived_analysis_dataset_data(...)`, which applies the derived output grid to
the underlying sources. Other unsupported cases are reported explicitly rather
than producing an incomplete script.

## Fit workflows

Fit script actions export the workspace's live datasets, model components,
parameter sharing, bounds, constraints, and active optimizer configuration.
Dataset scale settings, including shared fitted scales, are part of that
scientific configuration. Running the script performs one fit. Stored results
and timeline branches are not replayed.

## Python API

Use `dataset_workflow_plan(project, dataset_id)` or
`analysis_workflow_plan(project, analysis_id)` to inspect a dependency graph.
`render_workflow_script(plan)` renders a supported graph; the corresponding
`*_workflow_script(...)` functions provide the common one-call paths.

The graph consists of versioned `WorkflowNode` operations with stable
dependencies and outputs. Only the dependency closure of the selected target
is rendered. This contract allows future plot and whole-project exports to
share the same reconstruction path.

Saved plots currently use their existing project-backed exporter. See
[Models and fitting](gui_fitting.md#scripts) and [Saved plots](plotting.md).

## Numbered-source import

**Copy source import script** exports an editable `SourceSelection` and
`import_source_selection` call. The script re-resolves the expression, and can
retain expression groups as ordinary dataset subfolders. Saved native reduction
recipes instead replay their recorded logical run membership. See
[Selecting numbered sources](source_selection.md) for syntax and shared-source
uncertainty restrictions.
