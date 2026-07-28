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
- scale factor; and
- spectral-channel conversion.

The script separates `load_sources()` from `prepare_datasets()`. Configuration
is stored in ordinary dictionaries near the top of the file, so paths and
scientific settings can be changed directly. `build_workflow()` returns both
the loaded `DatasetEntry` objects and the prepared numerical data.

`SOURCE_ROOT` is the common source directory when possible. Relative source
paths are resolved from that location. File size and modification time are
recorded as a lightweight fingerprint; a changed source produces a warning
rather than silently claiming an exact reproduction.

Grouped raw direct-geometry and MDEvent reductions, derived analysis datasets,
fits, and plots are not yet included in this exporter. nfit reports these cases
explicitly instead of producing an incomplete script.

## Python API

Use `dataset_workflow_plan(project, dataset_id)` to inspect the dependency
graph, `render_workflow_script(plan)` to render a supported graph, or
`dataset_workflow_script(project, dataset_id)` for the common one-call path.

The graph consists of versioned `WorkflowNode` operations with stable
dependencies and outputs. Only the dependency closure of the selected target
is rendered. This contract allows future analysis, fit, posterior, and plot
exports to share the same reconstruction path.

Fit and saved-plot scripts currently use their existing project-backed
exporters. See [Models and fitting](gui_fitting.md#scripts) and
[Saved plots](plotting.md).
