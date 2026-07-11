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

When launched from a terminal, Ctrl+C sends an interrupt that closes the Qt
event loop cleanly with the standard interrupt exit status.

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
long text block. For binned MDHisto data, the data summary reports the total
bin count and a `Fit bins` count computed from the same prepared view used by
the optimizer, so file masks, inherited group masks, dataset masks, empty bins,
and invalid uncertainties are reflected in the number of bins that actually
participate in a fit.

Point-list datasets such as magnetization or powder elastic data expose editable
coordinate/channel configuration. MDHisto datasets expose rebin settings in the
Axes panel when rebinning is supported. Rebinning can be enabled for viewing and
fitting, and the current rebin can be materialized as a new independent dataset.
The materialized dataset keeps the source dataset's temperature override.

The dataset title row includes a `T (K)` control that sets a per-dataset sample
temperature override, stored in `dataset.parameters["temperature"]`. Spin down
to the minimum ("(from data)") to defer to any temperature imported with the
data. Physics models that need the Bose factor read the temperature from the
fit points and raise a clear error when no valid temperature is available.
When the data group defines lattice parameters, they are attached to the fit
points automatically so form factors and |Q|-dependent models work without
per-dataset setup.

## Masks and models

Selecting a mask or model shows its type selector and parameter editor at the
top of the right panel. Parameter tooltips come from the same registry that
defines defaults and examples, so a user can inspect what each field means
without guessing from the widget label alone.

Masks are applied together. File-provided masks remain separate from metallix
masks; metallix masks are applied as their own analysis mask layer. The data
viewer exposes `combined_mask`, `file_mask`, and `metallix_mask` channels:
`combined_mask` is the effective file-or-metallix exclusion mask used for
display and fitting, while the other two channels show provenance. The data
viewer's `Apply Masks` checkbox controls whether masked regions are hidden in
the viewer, but fitting still excludes masked data.

Models can have multiple components in one workspace. Model parameters include
controls for whether they are fitted and whether they are shared globally across
datasets. Each parameter can also define an optional plot label used as the
default nickname in fit diagnostics; plain text and Matplotlib mathtext/LaTeX
style labels such as `$\\Gamma$` are accepted. More granular constraints and
linking should continue to be expressed as explicit, scriptable model
configuration rather than hidden widget state. The model editor scrolls when a
component has many sections or parameters, so fit-parameter rows keep normal
editor height as generated Heisenberg RPA exchange orbits are added.

The spin-fluctuation models (`local_relaxational`, `mmp_relaxational`,
`heisenberg_rpa`; see [Spin-fluctuation models](spin_fluctuation_models.md))
add a form-factor picker with all tabulated magnetic-ion entries plus a
`Custom...` choice; the custom coefficient field appears only when that choice
is selected. The Heisenberg RPA model additionally shows a structured crystal
editor:

- **Crystal** — lattice parameters and space group, with `Import CIF...`
  (loads lattice, space group, and atomic sites from a CIF file, and copies
  them onto the data group) and `Use group crystal` (copies the group's stored
  crystal into the model). Bare numeric or Hermann-Mauguin space groups use
  gemmi's reference setting; append an explicit setting such as `:1` or `:2`
  when a non-reference origin choice is intended.
- **Atomic Sites** — an editable table of Wyckoff sites with fractional
  coordinates, a magnetic-ion selector per site, and a `Magnetic` checkbox
  marking which sites carry spins.
- **Exchange Bonds** — a bond-length cutoff and a `Generate symmetry orbits`
  button. Generation expands the magnetic sites through the space group,
  enumerates bonds up to the cutoff, groups them into symmetry-distinct orbits
  (`J1`, `J2`, `J3a`, `J3b`, ...), and adds one exchange fit parameter per
  orbit to the Fit Parameters grid. Values of orbits whose labels persist are
  kept across regeneration; a read-only table grows with the orbit count, up
  to a capped visible height, and summarizes each orbit's distance and
  multiplicity.

All crystal and bond state is plain data in the model component's
configuration, so it serializes with the project file and can equally be set
from a script.

## Fit timeline

The `Fits` tree stores initial conditions, fit results, current states, and
nested timelines. Selecting a fit state restores the dataset/mask/model
configuration stored with that fit. Running a fit creates only the new fit
result; a sibling `Current state` is created lazily only when the user edits the
latest result at that timeline level. The active fit state is shown with a
different tree style from the ordinary cursor selection, so users can tell which
fit state is applied even after selecting a model or dataset elsewhere.

Running `Fit now` from the current state or from the end of a timeline appends a
new fit result at that level. Running from an earlier result, or explicitly
checking `Branch timeline`, creates a nested timeline so alternative fitting
attempts remain organized. When a fit succeeds, globally shared fitted
parameters are written back into the live model component and the result
snapshot, so selecting the model after the fit shows the best-fit values in the
parameter editor. Parameters fitted separately per dataset or group are stored
on the component as scoped fitted values because one editor field cannot
represent several fitted numbers.

If a fit fails, the failed result is kept for diagnostics but the mutable
scientific state remains available as `Current state`: a failed run from an
existing fit state creates or reuses the following `Current state`, while a
failed run from `Current state` keeps that state active for immediate edits.

When a fit state is selected, `Branch timeline` stays above the scrollable
details area. Optimizer/loss controls, differential-evolution settings, and
posterior settings appear as panels inside the same scrollable details area as
the fit state summary. Fit-result and initial-state parameter tables sit in an
adjustable pane so the user can drag the divider to give more room to the
parameter list or to the metadata/details below it.

Changing scientific state creates or updates the current timeline state:
datasets, masks, model components, parameter values, fitted/fixed flags,
sharing, limits, constraints, scale factors, and dataset weights are all part of
the fit snapshot. Editing the latest fit result at a level moves the active fit
state to that level's `Current state` and records the edit there; editing an
earlier result creates a nested timeline. Changing only fit-engine settings does
not create an edit branch by itself. This lets users rerun the same scientific
state with a different loss, initializer, or posterior sampler and compare the
resulting fit entries side by side.

The fit editor follows a simple opt-in pipeline:

1. `Least squares` always runs and is the default fast workflow.
2. `Differential evolution initialization` may be enabled when starting values
   are uncertain. It searches bounded parameter space before least-squares
   polishing and therefore requires finite bounds on every fitted parameter.
3. `Sample posterior with emcee` may be enabled after least squares to estimate
   posterior intervals and correlations around the fitted solution.

Initialization and posterior sampling are optional. For simple fits with good
starting values, users can leave differential evolution and emcee disabled and
just run least squares. For more complex fits, emcee can be run as part of the
fit pipeline or later from an existing fit result.

The `Loss` control selects the cost function: the rule that turns residuals
into a number the optimizer tries to make small. A residual is the difference
between data and model divided by the data uncertainty, so a residual of `1`
means the model is about one standard deviation away from the data point. The
available losses are:

| Loss | What it does | When to use it |
| --- | --- | --- |
| `linear` | Ordinary chi-squared. Every residual is squared, so points with residual `10` count one hundred times more than points with residual `1`. | Use this for clean data, well-understood error bars, and final reported fits where the usual least-squares uncertainties should be easy to interpret. |
| `soft_l1` | A smooth robust loss. Small residuals behave like ordinary chi-squared, while large residuals grow more slowly. | A good first robust choice when a few pixels, spurions, or imperfectly masked regions are pulling the fit away from the main signal. It is gentler than the more aggressive robust losses. |
| `huber` | Chi-squared near zero, then roughly linear for residuals above the loss scale. | Use when you want a simple compromise: trust most points, but stop very large residuals from dominating. It is often easy to explain and less aggressive than `cauchy` or `arctan`. |
| `cauchy` | Strongly down-weights large residuals. Very bad points add only slowly increasing cost. | Use for exploratory fits when there are clear outliers or model-mismatch regions that should have little leverage. It can hide systematic problems, so inspect residuals before trusting the result. |
| `arctan` | The most aggressive option here. Very large residuals contribute almost a capped amount. | Use only as a last-resort diagnostic when severe outliers otherwise prevent convergence. If this changes the scientific conclusion, improve masks/modeling and refit with a less aggressive loss. |

`Loss scale` sets where robust losses begin to treat a point as "large." With
normalized residuals, `1.0` is approximately one standard deviation. Increasing
it makes robust losses behave more like ordinary least squares; decreasing it
makes down-weighting begin earlier.

Longer fit runs open or reuse a progress window and run in a background Qt
worker so the GUI can keep repainting and responding while initialization,
least squares, or emcee is active. The progress window reports the active
stage, iteration or residual-evaluation count, current cost when available, and
current time per step plus current parameter values in a table, with a short
stage log below; the divider between the table and log is draggable. The
time-per-step value is reported for differential-evolution
initialization, least-squares residual evaluations, and emcee posterior
sampling. The `Cancel` button requests cancellation at the next optimizer or
sampler progress update. Starting another fit resets the same progress window
instead of opening duplicates. When a fit pipeline finishes, the progress window
stays open so the final stage, parameters, and log remain available until the
user closes it. A compact progress log is also stored in the fit metadata.

`DE workers` and `emcee workers` control optional parallel worker threads for
differential-evolution objective evaluations and emcee log-probability
evaluations. The default is `1`, which keeps execution serial and predictable.
Use `-1` to let metallix choose a conservative CPU-based value, currently one
less than the available processor count capped at eight workers. Values greater
than one use an internal thread pool; this is most useful when model evaluation
spends substantial time in NumPy/SciPy code.

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
with best-fit parameters expose a `Posterior sampler` panel. That panel can
rerun emcee from the best-fit parameters, append additional steps to a stored
raw chain, promote the best stored emcee sample when it has a better likelihood
than the fit result, or change burn-in/thinning after the fact. These posterior-only
operations update the selected fit result's posterior summaries and diagnostics
without running least squares again and without creating a new timeline point.
Changing burn-in or thinning simply reinterprets the stored raw chain; rerun
replaces the stored posterior; append continues from the final walker positions
and extends the stored chain. Promoting a best sample is different: it restores
the fit result's snapshot, writes that sample into the editable model parameters,
and records the result through the same `Current state` path as a manual
parameter edit. Posterior rerun and append also use the background worker/progress
window and expose their own emcee worker-thread control.

Fit results with covariance estimates or stored emcee samples expose a `Fit
diagnostics` button. The diagnostics window includes a covariance heatmap when
a covariance matrix is available, or a correlation heatmap for older fit entries
that only stored correlations. Diagnostic plots use compact labels such as `p1`, `p2`,
... by default. A label table at the bottom of the diagnostics window shows the
full parameter names and editable plot labels; edits immediately redraw the
covariance, trace, and corner plots for that fit result. When emcee samples are
stored, the same window also includes trace and corner-style posterior tabs. The
trace tab uses the raw emcee chain when available, plotting each walker as its
own colored line against `MCMC step` and marking the selected burn-in step.
Older entries that only stored flattened samples fall back to a single `sample`
axis. Corner histograms are drawn as unfilled step histograms with best-fit and
uncertainty marker lines plus value/error titles using a proper plus-minus
symbol with stacked asymmetric bounds. The lower-triangle panels combine
density shading, iso-density contour lines, faint individual samples, and solid
best-fit reference crosshairs so overplotted posterior pile-ups remain visible.

## Data viewer

The data viewer supports dataset switching, channel selection, mask toggling,
axis selection, hidden-axis slicing/integration, color scale and limit controls,
cursor readouts, histogram box cuts, 1D line styling, model overlays, figure copy,
and script export.
When enabled model components have a complete parameter set, the viewer can
calculate and show the current model and residual channels from the `Show
model` control even before an optimization has been run. Stored fit-result
channels are still reused when no current model can be evaluated and the stored
channels remain compatible with the current dataset view.
For 2D fit comparisons, histogram box cuts show integrated data+fit cuts along
both plotted axes; when residuals are enabled, residual cuts are shown below
the residual panel and at the far right.

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
