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
`file_mask`, or `metallix_mask`), color-scale controls, cursor readout,
histogram box cuts, mask toggling, model overlays,
figure font sizing, dataset switching, clipboard copy, and script export. The
Qt viewer accepts either one `MDHistoData` object or a sequence of datasets plus
optional `dataset_names`.

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

Launch the project explorer with `metallix`. The explorer manages saved
projects, workspaces, datasets, dataset groups, masks, models, rebinned dataset
views, and fit timelines. The GUI is documented in
[GUI workflows](gui_workflows.md); its controls are expected to stay backed by
scriptable project state rather than hidden widget-only state.

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
.. automodule:: metallix.axes
   :members:

.. automodule:: metallix.dataset
   :members:

.. automodule:: metallix.mdhisto
   :members:

.. automodule:: metallix.models
   :members:

.. automodule:: metallix.cross_section
   :members:

.. automodule:: metallix.spin_fluctuations
   :members:

.. automodule:: metallix.form_factors
   :members:

.. automodule:: metallix.crystal
   :members:

.. automodule:: metallix.fitting
   :members:

.. automodule:: metallix.fit_views
   :members:

.. automodule:: metallix.pipeline
   :members:

.. automodule:: metallix.resolution
   :members:

.. automodule:: metallix.rebin
   :members:

.. automodule:: metallix.plotting
   :members:
```
