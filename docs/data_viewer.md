# Data viewer and saved plots

The data viewer displays point lists and gridded datasets without changing
their stored values. It supports channel selection, masks, cuts, maps,
waterfalls, model comparisons, and volumetric rendering.

Every **View in data viewer** action opens a new window. Multiple viewers may
display the same workspace or different workspaces at once; dataset selection,
axes, ranges, visualization mode, styling, and other controls are independent
in each window. Project changes refresh every open viewer without merging their
view settings. **Open new viewer**, beside the visualization selector, creates
an independent viewer initialized from the current dataset and view settings.
Closing one viewer does not affect the others.

## Slices, lines, and maps

Choose the displayed axes and use the remaining-axis controls to select one bin
or integrate a range. Channel labels and units come from the dataset's declared
physical quantities. **Apply masks** hides the combined file and nfit mask in
the figure; fitting always excludes masked data.

In **Figure**, enable **Show other-axis binning above plot** to add the current
non-displayed-axis selections to the title. Projected labels and physical units
are retained, for example `[K,-K,0]=[0.9,1.1] r.l.u., ΔE=[1,2] meV`. The option
is preserved independently for each dataset view and included in saved or
copied figure scripts.

**Minimum coverage** applies independently to viewer reductions. A displayed
pixel is masked when the measured fraction of its requested hidden-axis
integration volume is below the cutoff. The same cutoff is recomputed over
each box selected by the histogram tool before its horizontal and vertical
profiles are drawn. Coverage is available as its own channel and in the cursor
readout; it does not alter pixel opacity or the intensity colormap.

For a histogram box cut, data are combined with inverse-variance weights:

$$
\bar y=\frac{\sum_i y_i/\sigma_i^2}{\sum_i1/\sigma_i^2},
\qquad
\sigma_{\bar y}=\frac{1}{\sqrt{\sum_i1/\sigma_i^2}}.
$$

$y_i$ is a contributing bin value, $\sigma_i$ is its one-sigma uncertainty,
and $\bar y$ is the reduced value with uncertainty $\sigma_{\bar y}$.
Masked bins and invalid uncertainties are excluded.

Plot smoothing is specified in displayed-bin widths. It affects only the
rendered figure and exported figure recipe, not fitting, rebinning, or numerical
data exports.

## Model and residual channels

**Show model** evaluates the current compatible model or reuses stored
fit-result channels. **Unmask model** extends model evaluation over finite
coordinates excluded from the fit while leaving the data masked. This is useful
for inspecting interpolation or extrapolation, but can be expensive for large
volumes.

For multidimensional data, the live model channel has the same full shape as
the prepared dataset. Slice selection, integration ranges, and reductions are
then applied identically to data and model in the viewer, so changing a viewed
or integrated axis does not require a new model evaluation. A model failure is
isolated to its dataset; compatible datasets retain their overlays, and the
failure text is recorded in the **Current state** metadata.

Two-dimensional comparisons show linked data, fit, and optional residual
panels. Box cuts through a comparison apply the same weighted reduction to the
data and model.

After fitting, nfit also attempts display-only model evaluation for disabled
and zero-weight datasets. Failure of this optional evaluation is recorded but
does not invalidate the fit or discard successful overlays from other
datasets.

## Waterfall plots

Choose **Visualization > Waterfall** to stack one-dimensional traces.

- For multidimensional data, choose the horizontal and waterfall axes; other
  dimensions retain their ordinary point/range controls.
- For a one-dimensional dataset, compatible datasets in the same immediate
  project group provide the traces. Choosing any sibling dataset switches to
  that sibling's immediate group.
- **Bin width** controls coarsening along the waterfall axis; **Auto** targets
  roughly ten traces.
- **Minimum coverage** is specific to waterfall trace bins. It is independent
  of the slice and histogram cutoff.

Trace offset, colors, marker fill, zero references, labels, and model overlays
are configurable. Saved plot recipes and generated scripts retain these
settings and all contributing dataset references. Trace markers are unfilled by
default.

The axis **Reset** buttons and Matplotlib **Home** button use the current trace
extent. That extent is recomputed when the dataset group, displayed axes, or
waterfall binning changes.

## Tiled 2D slices

Choose **Visualization > Tiled slices** for gridded data with at least three
dimensions containing more than one bin. Choose the horizontal and vertical
dimensions as for a normal map, then choose a distinct **Third dimension** in
the tiled-slice controls. Any further dimensions retain the ordinary
point/range integration controls.

**Range low** and **Range high** select the third-axis values included in the
figure. **Step size** groups adjacent values into panels; its horizontal slider
spans one native bin through the selected range. **Auto** makes nine panels
when at least nine third-axis values are available, and otherwise makes one
panel per available value.

By default, every panel is labeled at its bottom-right corner with its
representative third-axis value, backed by a translucent box so that it remains
legible over the data. Clear **Show tiled-slice value labels** in **Figure** to
hide these annotations.

All panels share their x and y limits. The default global color scale also
shares one normalization and one colorbar at the far right. With **Autoscale**
enabled, **Local scale per tiled plot** gives every panel independently computed
limits and its own adjacent colorbar. Local scales are unavailable with manual
color limits; turning off **Autoscale** returns the view to the global scale.
The mode intentionally displays data only; model and residual panels and
histogram box cuts are not shown. Saved plot recipes and generated scripts
preserve the three dimensions, range, step, label visibility, hidden-axis
selections, and global/local color setting.

## Volumetric mode

Gridded datasets with at least three dimensions can use **Volumetric** mode.
Point-list data must first be rebinned.

Select three distinct displayed axes. Remaining dimensions use the same bin or
range integration controls as the slice viewer. Axis limits crop by bin center;
**Equal data units** preserves physical axis ratios, while **Custom** enables
visual stretching without changing coordinates or scalar values.

**Volume** performs direct volume rendering. **Isosurface** extracts a surface
at the selected opacity level. Color and opacity channels may be linked or
controlled independently with editable transfer curves. Masked and non-finite
cells remain transparent.

Exports include:

- PNG for the current camera view;
- VTK PolyData, PLY, STL, or glTF for an isosurface;
- VTK rectilinear grid for volumetric scalar data; and
- MP4 for a camera orbit around a selected displayed axis.

Smoothing and axis scaling affect rendered images and movies. Numerical volume
and surface exports use unsmoothed channel values.

## Saved plots

Each workspace has a **Plots** branch. In the viewer, **Save plot** stores the
current figure as an editable recipe. A saved plot can be:

- reopened in a presentation window;
- restored into the data viewer;
- copied as a script; or
- saved as a standalone script.

Scripts require a saved project so dataset references have a stable path. They
use the public plotting API and can run without constructing the project
explorer.

See [Saved plots](plotting.md) for the serialization and scripting interface.
