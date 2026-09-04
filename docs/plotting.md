# Saved plots

Each workspace's **Plots** branch stores editable figure recipes. A recipe
contains stable dataset references, displayed axes and channels, integrations,
styles, limits, smoothing, figure size, and a complete rebin-configuration
snapshot for each source. It does not duplicate the numerical data or embed a
rendered image. A composite recipe additionally records its workspace or nested
dataset-group scope and composite rebin configuration.

Use **Store plot** beside **Open new viewer** to create or update a recipe.
Changing the live dataset rebin settings later does not change existing plot
recipes, so plots from one source may use different coordinate bases, bounds,
resolutions, symmetry expansions, or weighting settings. Opening a
saved plot shows a presentation window; **Edit in data viewer** restores its
interactive controls. Duplicate or rename a recipe before making a variant.
If a referenced dataset is missing, nfit preserves the recipe and reports the
missing source.

## Waterfalls and volumes

Waterfall recipes retain the horizontal and stacking axes, bin width, trace
offset, colors, labels, references, error bars, and model-overlay style. For a
one-dimensional source, compatible sibling datasets provide the individual
traces. For multidimensional data, coarse stacking bins are inverse-variance
weighted means with propagated one-sigma uncertainties.

Volumetric recipes retain the three displayed axes, remaining-axis selections,
camera, transfer functions, opacity, and axis scaling. Masks remain part of the
view; a fully masked selection is reported rather than rendered as an empty
opaque volume.

The interactive controls are described in
[Data viewer and saved plots](data_viewer.md).

## Scripts and fit diagnostics

Saved plots can copy or save a GUI-free Python script. The script loads the
project and renders through the public plot-recipe API, so it can use a
headless Matplotlib backend. A project must be saved before script generation
so every dataset has a stable path.

Fit results may also create covariance or correlation plots. These recipes
refer directly to the fit result and reopen in the presentation window rather
than the dataset viewer.
