# API reference

## MDHisto import and viewing

Mantid `SaveMD` / `MDHistoWorkspace` NeXus files can be loaded with
`load_mantid_mdhisto_nxs`. The returned `MDHistoData` stores axis metadata plus
same-shaped `signal`, propagated `errors`, `mask`, and `num_events` arrays.

Use `slice_viewer(data)` for the PySide6 interactive viewer. The viewer supports
choosing displayed x/y axes, integrating hidden axes, switching the displayed
channel (`signal`, `errors`, `num_events`/multiplicity, or `mask`), color-scale
controls, cursor readout, histogram box cuts, figure font sizing, clipboard
copy, and script export.

Use `plot_mdhisto_slice(data, ...)` when a non-interactive Matplotlib figure is
preferred, for example in notebooks or batch scripts. The exported scripts from
the GUI call this helper and include the current display settings.

```{eval-rst}
.. automodule:: metallix.dataset
   :members:

.. automodule:: metallix.models
   :members:

.. automodule:: metallix.cross_section
   :members:

.. automodule:: metallix.fitting
   :members:

.. automodule:: metallix.rebin
   :members:

.. automodule:: metallix.plotting
   :members:
```
