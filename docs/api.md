# API reference

## MDHisto import and viewing

Mantid `SaveMD` / `MDHistoWorkspace` NeXus files can be loaded with
`load_mantid_mdhisto_nxs`. The returned `MDHistoData` stores axis metadata plus
same-shaped `signal`, propagated `errors`, `mask`, and `num_events` arrays. By
default, the importer avoids copying bulky ancillary NeXus groups such as
`experiment0`; pass `copy_metadata=True` when a metadata tree copy is needed.

Use `slice_viewer(data)` for the PySide6 interactive viewer. The viewer supports
choosing displayed x/y axes, integrating hidden axes, switching the displayed
channel (`signal`, `errors`, `num_events`/multiplicity, or `mask`), color-scale
controls, cursor readout, histogram box cuts, figure font sizing, dataset
switching, clipboard copy, and script export. The Qt viewer accepts either one
`MDHistoData` object or a sequence of datasets plus optional `dataset_names`.

Use `plot_mdhisto_slice(data, ...)` when a non-interactive Matplotlib colormap
figure is preferred, for example in notebooks or batch scripts. Use
`plot_mdhisto_line(data, ...)` for one-dimensional MDHisto outputs such as
shape `(1, 1, 1, N)`, and `plot_mdhisto_auto(data, ...)` to dispatch to a line
plot for exactly-one-non-singleton data or a slice figure otherwise. The
exported scripts from the GUI call `plot_mdhisto_slice` and include the current
display settings.

## Modeling and fitting

The fitting API supports both the original single-dataset
`fit_least_squares(data, model, specs)` convenience function and the
simultaneous-fit `FitProblem` workflow. `FitDataset` carries dataset-local
weights, preprocessing transforms, and optional instrument resolution, while
`ModelSpec` wraps the shared physics model. See the
[modeling pipeline](modeling_pipeline.md) page for the recommended structure.

```{eval-rst}
.. automodule:: metallix.dataset
   :members:

.. automodule:: metallix.models
   :members:

.. automodule:: metallix.cross_section
   :members:

.. automodule:: metallix.fitting
   :members:

.. automodule:: metallix.resolution
   :members:

.. automodule:: metallix.rebin
   :members:

.. automodule:: metallix.plotting
   :members:
```
