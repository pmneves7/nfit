# Data viewer and saved plots

The data viewer displays point lists and gridded datasets without changing
their stored values. It supports channel selection, masks, cuts, maps,
waterfalls, model comparisons, and volumetric rendering.

Every **View in data viewer** action opens a new window. Multiple viewers may
display the same workspace or different workspaces at once; dataset selection,
axes, ranges, visualization mode, styling, and other controls are independent
in each window. Project changes refresh every open viewer without merging their
view settings. Closing one viewer does not affect the others.

## Slices, lines, and maps

Choose the displayed axes and use the remaining-axis controls to select one bin
or integrate a range. Channel labels and units come from the dataset's declared
physical quantities. **Apply masks** hides the combined file and nfit mask in
the figure; fitting always excludes masked data.

For a histogram box cut, data are combined with inverse-variance weights:

$$
\bar y=\frac{\sum_i y_i/\sigma_i^2}{\sum_i1/\sigma_i^2},
\qquad
\sigma_{\bar y}=\frac{1}{\sqrt{\sum_i1/\sigma_i^2}}.
$$

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

Two-dimensional comparisons show linked data, fit, and optional residual
panels. Box cuts through a comparison apply the same weighted reduction to the
data and model.

After fitting, nfit also attempts display-only model evaluation for disabled
and zero-weight datasets. Failure of this optional evaluation is recorded but
does not invalidate the fit.

## Waterfall plots

Choose **Visualization > Waterfall** to stack one-dimensional traces.

- For multidimensional data, choose the horizontal and waterfall axes; other
  dimensions retain their ordinary point/range controls.
- For a one-dimensional dataset, compatible datasets in the same immediate
  project group provide the traces.
- **Bin width** controls coarsening along the waterfall axis; **Auto** targets
  roughly ten traces.

Trace offset, colors, marker fill, zero references, labels, and model overlays
are configurable. Saved plot recipes and generated scripts retain these
settings and all contributing dataset references.

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
