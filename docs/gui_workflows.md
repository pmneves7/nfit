# GUI workflows

The project explorer and data viewer use the neutron “n” icon in their windows
and the OS Dock or taskbar. Restart nfit after upgrading to refresh the running
application icon.

The nfit interface has two main window types:

- the **project explorer**, which owns datasets, masks, models, fits, analyses,
  and saved plots; and
- the **data viewer**, which displays cuts, maps, fit comparisons, and volumes.

Scientific settings are stored in the `.nfit` project and use the same package
functions available through the Python API.

Click **Help** beside **File** in the project explorer toolbar to open the local
documentation home page (`docs/_build/html/index.html`) in your default web
browser. If it has not been built, Help shows the build command instead.

Open **File → Preferences…** for application-wide preferences. The **Colormaps**
page sets the defaults for newly opened continuous and waterfall plots, shows
the shared custom-palette folder, opens it in your file manager, and explains
the [RGB file format](data_viewer.md#custom-colormap-files). Preferences are
local to this nfit installation and are never stored in a project. Custom
palettes are shared across projects; restart nfit after adding or editing files.
The **Performance** page stores defaults for new rebin configurations and offers
machine calibration. Dataset and composite rebin panels also offer **Benchmark
this rebin…**; see [performance settings and benchmarks](performance.md#performance-preferences-and-benchmarks).

Each `.nfit` project is one portable file containing its JSON manifest and
generated analysis artifacts. Original imported measurement files remain
external source references.

**View in data viewer** opens a new, independent window each time, so several
datasets or views of the same dataset can remain visible side by side. Use
**Open new viewer** beside the visualization selector to duplicate the current
view before adjusting its dataset, axes, ranges, or styling independently.
Use **Store plot** beside it to snapshot the view and its source rebin settings
as an editable recipe in the workspace's **Plots** branch.
The viewer's **Figure** section can place the current hidden-axis binning above
the plot, including projected-axis labels and units.

The project tree supports Shift- and Ctrl/Command-selection of multiple items
of the same kind. Right-clicking any selected dataset preserves the whole
selection; **Enable**, **Disable**, **Delete**, and **Copy** apply to the batch,
and a copied dataset batch can be pasted into another workspace or nested
dataset group. Dataset groups can also be copied and pasted, including their
children and group-level configuration.

## Launch

Activate the nfit environment and run:

```bash
nfit
```

For local development, `python -m nfit.project_gui` is equivalent.

The Project Explorer initially requests a 1560 by 1000 pixel workspace. On a
smaller display, nfit reduces each dimension to fit within the available
desktop while leaving a small margin for window controls.

## First fit

1. Create or select a workspace.
2. Import data and verify its units, normalization, temperature, crystal
   information, and masks.
3. Add model components and choose which parameters vary.
4. Set parameter bounds, sharing, constraints, dataset weights, and scales.
5. Select **Fit now**.
6. Inspect the fitted curve and residuals in the data viewer.
7. Save the project, then export plots, scripts, or a fit report.

The detailed workflows are divided by task:

- [Importing and preparing data](data_import.md) covers file import, MACS,
  raw direct-geometry and MDEvent reduction, UB matrices, rebinning,
  conditions, masks, and backgrounds.
- [Models and fitting](gui_fitting.md) covers model components, constraints,
  fit timelines, optimizers, posterior sampling, diagnostics, and scripts.
- [Data viewer and saved plots](data_viewer.md) covers cuts, maps, waterfall
  plots, tiled 2D slices, model overlays, volumetric rendering, and plot export.
- [Analysis Window](data_playground.md) covers non-fitting operations such as
  Bragg integration, Curie-Weiss fitting, and spectral integrals.

Rebin controls use separate settings, metadata-dimension, and bin-information
tabs where applicable. The bin-information tab expands to show resolved centers
and edges. Rebin progress distinguishes numerical output construction from the
subsequent viewer and control refresh, so a slow plot redraw is not reported as
continued bin accumulation.

## Project organization

A project contains one or more workspaces. Each workspace owns:

- a dataset tree, including optional nested dataset groups;
- masks and backgrounds associated with datasets or groups;
- model components and fit history;
- analysis recipes and their derived outputs; and
- saved plot recipes.

Most actions are available from buttons and right-click menus. Files may also be
dragged into a workspace or dataset group. Dropping a `.nfit` file opens that
project after the usual unsaved-change check.

Multiple objects of the same kind can be selected with Shift-click or
Ctrl/Command-click and deleted together. Drag and drop reorders compatible
objects; copy and paste can transfer datasets between workspaces and masks
between datasets.

The **File** menu provides New, Open, Recent projects, Reload from Disk, Save,
Save As, Close, and Quit. Closing or reloading a modified project asks before
discarding changes. While the GUI is running, nfit checks whether another
process replaced the open project file. It then offers to reload the external
version, save the in-memory version under another name, or keep the current
state. An explicit Save never overwrites a detected external change without a
separate confirmation.

## Reproducibility

Project files store scientific state rather than screenshots of the GUI. This
includes dataset sources and conventions, masks, model configuration,
parameters, constraints, fit settings, fit results, analyses, and saved plot
recipes.

The live workspace stored in the project manifest is authoritative when a
project opens. The remembered fit-history path selects its tree entry but does
not restore that snapshot over the live model. Selecting a historical fit entry
after opening remains the explicit action that restores it.

Right-click an ordinary source-backed dataset to copy or save readable Python
that reloads and prepares it without Qt. Analysis and fit targets include their
dataset dependency closure; fit exports use the live state and do not replay
history branches. Saved-plot scripts still require a saved project path. See
[Workflow scripts](workflow_scripts.md) for scope and limitations.

For the underlying objects and extension contracts, see
[Modeling and fitting pipeline](modeling_pipeline.md),
[Data and extension conventions](data_philosophy.md), and
[API reference](api.md).
