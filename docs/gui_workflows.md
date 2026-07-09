# GUI workflows

The graphical interface is a project explorer plus a data viewer. It is meant
to make common exploratory work easier without replacing scripts: every GUI
operation should correspond to a readable project file or a Python operation
that can be repeated later.

## Launch

After installing the package in the `metallix` environment, launch the project
explorer with:

```bash
metallix
```

During local development, the explicit environment interpreter is:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python -m metallix.project_gui
```

## Project explorer

The left tree organizes a project into top-level workspaces. A workspace holds
datasets, models, and fit history. Datasets may also contain nested dataset
groups and masks, so large experiments can be kept close to the physical or
sample-series structure the user thinks in.

Common actions are available from the bottom-left buttons, the right-side action
buttons, keyboard shortcuts, drag and drop, and right-click context menus:

- create, rename, reorder, and delete workspaces, datasets, dataset groups,
  masks, and models where the item type allows it,
- import datasets from files or drag files into the tree,
- copy and paste datasets between workspaces and masks between datasets,
- save datasets or whole projects,
- open or refresh the data viewer for a selected workspace or dataset,
- add masks and model components,
- run fits and keep the resulting history in the `Fits` tree.

The `File` menu supports New, Open, Recent projects, Save, Save As, Close, and
Quit. On platforms with standard shortcuts, these use the expected New/Open/Save
bindings, with Command as the default modifier on macOS and Control elsewhere.
Close and Quit prompt before discarding unsaved project changes.

## Dataset details

Selecting a dataset shows structured details in the right panel. The sections
include the dataset type, axes, crystal information, data summary, source file,
and imported metadata. Metadata is shown as an expandable tree so nested fields,
instrument logs, sample-environment information, timestamps, temperature, field,
and similar provenance remain inspectable without flattening everything into a
long text block.

Point-list datasets such as magnetization or powder elastic data expose editable
coordinate/channel configuration. MDHisto datasets expose rebin settings in the
Axes panel when rebinning is supported. Rebinning can be enabled for viewing and
fitting, and the current rebin can be materialized as a new independent dataset.

## Masks and models

Selecting a mask or model shows its type selector and parameter editor at the
top of the right panel. Parameter tooltips come from the same registry that
defines defaults and examples, so a user can inspect what each field means
without guessing from the widget label alone.

Masks are applied together. File-provided masks remain separate from metallix
masks; metallix masks are applied as their own analysis mask layer. The data
viewer's `Apply Masks` checkbox controls whether masked regions are hidden in
the viewer, but fitting still excludes masked data.

Models can have multiple components in one workspace. Model parameters include
controls for whether they are fitted and whether they are shared globally across
datasets. Each parameter can also define an optional plot label used as the
default nickname in fit diagnostics; plain text and Matplotlib mathtext/LaTeX
style labels such as `$\\Gamma$` are accepted. More granular constraints and
linking should continue to be expressed as explicit, scriptable model
configuration rather than hidden widget state.

## Fit timeline

The `Fits` tree stores initial conditions, fit results, current states, and
nested timelines. Selecting a fit state restores the dataset/mask/model
configuration stored with that fit. The active fit state is shown with a
different tree style from the ordinary cursor selection, so users can tell which
fit state is applied even after selecting a model or dataset elsewhere.

Running `Fit now` from the current state or from the end of a timeline appends a
new fit result at that level. Running from an earlier result, or explicitly
checking `Branch timeline`, creates a nested timeline so alternative fitting
attempts remain organized.

Changing scientific state creates or updates the current timeline state:
datasets, masks, model components, parameter values, fitted/fixed flags,
sharing, limits, constraints, scale factors, and dataset weights are all part of
the fit snapshot. Changing only fit-engine settings does not create an edit
branch by itself. This lets users rerun the same scientific state with a
different loss, initializer, or posterior sampler and compare the resulting fit
entries side by side.

The fit editor follows a simple opt-in pipeline:

1. `Least squares` always runs and is the default fast workflow.
2. `Differential evolution initialization` may be enabled when starting values
   are uncertain. It searches bounded parameter space before least-squares
   polishing and therefore requires finite bounds on every fitted parameter.
3. `Sample posterior with emcee` may be enabled after least squares to estimate
   posterior intervals and correlations around the fitted solution.

The `Loss` control selects the least-squares residual penalty. `linear` is the
ordinary chi-squared objective. Robust choices such as `soft_l1`, `huber`,
`cauchy`, and `arctan` down-weight large residuals and are useful as a safety
valve for imperfect masks, spurions, detector artifacts, or model-mismatch
regions. `Loss scale` sets the residual scale where robust losses begin to
down-weight points; with normalized residuals, `1.0` is approximately one
standard deviation.

Longer fit runs open or reuse a progress window. It reports the active stage,
iteration or residual-evaluation count, current cost when available, and current
parameter values in a table, with a short stage log below. Starting another fit
resets the same progress window instead of opening duplicates. A compact
progress log is also stored in the fit metadata.

Fit metadata is displayed in structured panels for optimizer configuration,
goodness-of-fit values, stored fit channels, arbitrary metadata, and the saved
snapshot. A `Fit results` panel presents best-fit parameter values in a compact
table with covariance-derived standard errors and, when emcee samples are
available, posterior median and 16/84 percentiles. Least-squares results also
include chi-squared, reduced chi-squared, covariance/correlation summaries, and
per-dataset contributions in the structured goodness-of-fit panel. Posterior
summaries include emcee settings, acceptance fractions, credible intervals, and
posterior correlations. Stored fit and residual channels are available in the
data viewer when their shapes still match the current dataset view. Fit results
with covariance estimates or stored emcee samples expose a `Fit diagnostics`
button. The diagnostics window includes a covariance heatmap when a covariance
matrix is available, or a correlation heatmap for older fit entries that only
stored correlations. Diagnostic plots use compact labels such as `p1`, `p2`,
... by default. A label table at the bottom of the diagnostics window shows the
full parameter names and editable plot labels; edits immediately redraw the
covariance, trace, and corner plots for that fit result. When emcee samples are
stored, the same window also includes trace and corner-style posterior tabs.
Corner histograms are drawn as unfilled step histograms with best-fit and
uncertainty marker lines plus value/error titles using a proper plus-minus
symbol with stacked asymmetric bounds. The lower-triangle panels combine density
shading, iso-density contour lines, faint individual samples, and solid best-fit
reference crosshairs so overplotted posterior pile-ups remain visible.

## Data viewer

The data viewer supports dataset switching, channel selection, mask toggling,
axis selection, hidden-axis slicing/integration, color scale and limit controls,
cursor readouts, histogram box cuts, 1D line styling, fit overlays, figure copy,
and script export.

For powder and magnetization-style datasets, cursor readouts hide coordinate
quantities that are not meaningful for that data type. `|Q|` readout is shown
when valid `q` information is present or can be computed from `2theta` and
wavelength.

The exported script is part of the package philosophy: GUI-produced figures
should be reproducible from editable Python code. When a GUI feature changes the
plot state, the script-export path should be updated at the same time.

## Hover text standard

Every user-interactable control in the project explorer and data viewer should
have useful hover text. A good tooltip explains what the control changes, when
the change takes effect, important constraints, and an example when the input is
not obvious. Registry-backed controls, such as mask parameters, model
parameters, model configuration, optimizers, and future resolution models,
should render hover text from the same metadata that defines defaults and
scriptable validation.

Tests audit the main GUI surfaces for missing tooltips. When adding a new
control, update the tooltip and the relevant documentation or wiki page in the
same change.
