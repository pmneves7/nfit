# Saved plots

The **Plots** section of each workspace stores editable figure recipes. A plot
records its dataset source by stable project ID and its displayed axes, channel,
integrations, color settings, limits, smoothing, figure size, and labels. It
does not duplicate the dataset or save an image in the project file.

The data viewer's **Waterfall** visualization stores the same kind of recipe.
For one multidimensional MDHisto dataset, choose the horizontal axis and
waterfall axis; hidden axes keep their normal point/range controls. The
waterfall-axis bin width defaults to approximately ten traces. Each coarse bin
is an inverse-variance weighted mean with propagated one-sigma errors.

When the selected dataset is 1D, every sibling in its immediate project data
group with the same horizontal-axis name, unit, and selected channel contributes
one trace. The saved recipe retains every contributing dataset by stable project
ID. Trace offset defaults to half the largest absolute intensity; its slider
spans zero through that absolute maximum. The bin-width slider spans one native
waterfall-axis bin through the complete axis. Continuous colormaps have a
two-handle sampling-range control, and marker interiors may be empty, match
each trace outline, or use one named color. The controls also provide optional
per-trace zero references, reference style, trace labels, data error bars, and
model-line color behavior. Multidimensional energy-bin labels use `ΔE` rather
than the internal `DeltaE` axis name. A custom suffix, font size, and common or
per-trace font color can be stored with the waterfall recipe.

Create a plot with **Save plot** beside **Copy figure** in the data viewer's
**Figure** panel. Open a saved plot from the tree to see a clean figure window.
Its only visible command is the **Plot** menu, which can copy or save the
figure, open the hidden controls dock, and copy or save a GUI-free generating
script. **Ctrl+S** (**Command+S** on macOS) in the interactive data viewer saves
the owning nfit project.

Choose **Edit in data viewer** on a saved plot to restore its recipe into the
interactive viewer. Use **Save plot** there to update that same plot;
rename or duplicate a tree entry before making a variant. Missing sources leave
the recipe intact and report an error instead of silently changing its dataset.

Generated scripts load the project and render through the backend-only plot
recipe API. They do not create Qt windows, so they can be run with a headless
Matplotlib backend for batch figure generation.

Fit results with stored covariance or correlation diagnostics also offer
**Create covariance plot**. These plots retain a stable fit-result reference
and reopen in the clean plot window; they are not sent to the data viewer,
whose controls describe dataset slices rather than parameter matrices.
