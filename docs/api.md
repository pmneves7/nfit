# API reference

## Reduced data, import adapters, and viewing

The package centers on reduced experimental coordinates. Importers should read
axis names, units, masks, intensities, uncertainties, and metadata from their
source files, then translate them into common containers. Mantid MDHisto NeXus
support is currently the most developed adapter, but it is not intended to be
the only supported input.

Mantid `SaveMD` / `MDHistoWorkspace` NeXus files can be loaded with
`load_mantid_mdhisto_nxs`. The returned `MDHistoData` stores axis metadata plus
same-shaped `signal`, propagated `errors`, `mask`, and `num_events` arrays. By
default, the importer avoids copying bulky ancillary NeXus groups such as
`experiment0`; pass `copy_metadata=True` when a metadata tree copy is needed.
Axes expose broad roles inferred from file labels and units, such as `h`, `k`,
`l`, `q_modulus`, `momentum_projection`, and `energy_transfer`.

Use `slice_viewer(data)` for the PySide6 interactive viewer. The viewer supports
choosing displayed x/y axes, integrating hidden axes, switching the displayed
channel (`signal`, `errors`, `num_events`/multiplicity, `combined_mask`,
`file_mask`, or `nfit_mask`), color-scale controls, cursor readout,
histogram box cuts, mask toggling, model overlays,
figure font sizing, dataset switching, clipboard copy, and script export. The
Qt viewer accepts either one `MDHistoData` object or a sequence of datasets plus
optional `dataset_names`. Independent displayed-X/Y Gaussian smoothing is
specified in bin-width units and is visual only; source arrays and fitting data
are never modified.

For gridded data with at least three dimensions, the same viewer exposes a
PyVista 3D mode. It supports independent X/Y/Z selection, slicing or integrating
specified ranges on remaining dimensions, viewed-axis limits, equal-data-unit
or custom visual axis scaling, volume and isosurface rendering, separate color
and opacity channels, editable color/opacity transfer curves, and mask-aware
rendering. Independent X/Y/Z Gaussian smoothing applies only to rendering and
image/movie output; numerical grids and surface-model exports remain
unsmoothed. It can export volume data (`.vtr`), surface meshes/scenes (`.vtp`,
`.ply`, `.stl`, or `.gltf`), still PNG images, and a full-orbit MP4 movie around
displayed X/Y/Z (Z by default) or the camera's current vertical direction.
PyVista, PyVistaQt, imageio, and the bundled imageio FFmpeg backend are installed
as nfit application dependencies.

Use `plot_mdhisto_slice(data, ...)` when a non-interactive Matplotlib colormap
figure is preferred, for example in notebooks or batch scripts. Use
`plot_mdhisto_line(data, ...)` for one-dimensional MDHisto outputs such as
shape `(1, 1, 1, N)`, and `plot_mdhisto_auto(data, ...)` to dispatch to a line
plot for exactly-one-non-singleton data or a slice figure otherwise. The
exported scripts from the GUI call `plot_mdhisto_slice` and include the current
display settings.

Fit-comparison views can be attached to `MDHistoData` with
`attach_fit_comparisons`. Project GUI fit results can also store fit and
residual channels directly. When the Qt viewer sees compatible stored channels,
it can render linked data/fit or data/fit/residual panels while preserving the
current plotting settings.

Launch the project explorer with `nfit`. The explorer manages saved
projects, workspaces, datasets, dataset groups, masks, models, rebinned dataset
views, and fit timelines. The GUI is documented in
[GUI workflows](gui_workflows.md); its controls are expected to stay backed by
scriptable project state rather than hidden widget-only state.

## N-dimensional rebinning

Use `rebin_nd` or `NDRebin` to bin point values onto regular N-dimensional
grids. Fractional binning is enabled by default, so a source point can be
distributed to neighboring bins according to its geometric overlap; pass
`fractional=False` for single-bin assignment. With `normalize=True`, each
output bin is an average of all source points that contribute to that bin. The
default averaging mode is
`mean_weighting="inverse_variance"`: when `data_errs` are supplied, each source
point receives a `1 / sigma**2` weight, and fractional binning multiplies that
statistical weight by the point's fractional spatial contribution. The reported
bin error follows the accumulated inverse-variance weight, reducing to
`1 / sqrt(sum(1 / sigma**2))` for non-fractional inverse-variance averages.
Points with non-finite or non-positive uncertainties are skipped in this mode.
Optional `data_weights` multiply each point's statistical weight; the GUI uses
this for workspace composites so dataset fit weights enter as
`fit_weight / sigma**2`.

Set `mean_weighting="uniform"` to keep the legacy simple mean behavior. In that
mode, each point has equal statistical weight and fractional binning contributes
only the spatial fraction. If `normalize=False`, rebinning returns weighted sums
instead of means and `mean_weighting` is ignored.

Fractional binning is accumulated in batches. Earlier implementations expanded
every point into its full list of neighboring-bin contributions at once, which
can be prohibitive for large 4D datasets. The current rebinner accumulates each
batch directly into the output arrays; set `batch_size` explicitly for
benchmarking or use `max_batch_bytes` to let `NDRebin` choose an approximate
working-memory target. The default target is 192 MB of per-batch working arrays.
This value is not a cap on total rebinner memory use: the rebinner still holds
the source data, coordinates, errors, output arrays, and other bookkeeping. A
smaller target usually reduces temporary memory at the cost of more CPU time,
while a larger target can reduce batching overhead but raises peak memory. The
best value depends on the dataset size, output grid size, dimensionality, and
available memory.

## Modeling and fitting

The fitting API supports both the original single-dataset
`fit_least_squares(data, model, specs)` convenience function and the
simultaneous-fit `FitProblem` workflow. `FitDataset` carries dataset-local
weights, preprocessing transforms, and optional instrument resolution, while
`ModelSpec` wraps the shared physics model. `DataGroup` and `FitModelSession`
package related datasets, per-dataset model overrides, optimizer settings, and
fit history for iterative analysis. Deterministic fits support robust
least-squares losses and optional differential-evolution initialization;
posterior checks use `emcee` through `sample_problem_parameters`. See the
[modeling pipeline](modeling_pipeline.md) page for the recommended structure.

```{eval-rst}
.. automodule:: nfit.axes
   :members:

.. automodule:: nfit.dataset
   :members:

.. automodule:: nfit.mdhisto
   :members:

.. automodule:: nfit.models
   :members:

.. automodule:: nfit.cross_section
   :members:

.. automodule:: nfit.spin_fluctuations
   :members:

.. automodule:: nfit.form_factors
   :members:

.. automodule:: nfit.crystal
   :members:

.. automodule:: nfit.fitting
   :members:

.. automodule:: nfit.fit_views
   :members:

.. automodule:: nfit.pipeline
   :members:

.. automodule:: nfit.resolution
   :members:

.. automodule:: nfit.rebin
   :members:

.. automodule:: nfit.plotting
   :members:
```
