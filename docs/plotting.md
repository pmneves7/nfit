# Saved plots

The **Plots** section of each workspace stores editable figure recipes. A plot
records its dataset source by stable project ID and its displayed axes, channel,
integrations, color settings, limits, smoothing, figure size, and labels. It
does not duplicate the dataset or save an image in the project file.

Create a plot from the data viewer's **Plot** menu. Open a saved plot from the
tree to see a clean figure window. Its only visible command is the **Plot**
menu, which can copy or save the figure, open the hidden controls dock, and
copy or save a GUI-free generating script.

Choose **Edit in data viewer** on a saved plot to restore its recipe into the
interactive viewer. Use **Create saved plot** there to update that same plot;
rename or duplicate a tree entry before making a variant. Missing sources leave
the recipe intact and report an error instead of silently changing its dataset.

Generated scripts load the project and render through the backend-only plot
recipe API. They do not create Qt windows, so they can be run with a headless
Matplotlib backend for batch figure generation.

Fit results with stored covariance or correlation diagnostics also offer
**Create covariance plot**. These plots retain a stable fit-result reference
and reopen in the clean plot window; they are not sent to the data viewer,
whose controls describe dataset slices rather than parameter matrices.
