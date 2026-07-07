# Reduced-data philosophy

`metallix` should be organized around reduced experimental coordinates and
models, not around any one instrument, file format, or Mantid workspace type.

The common assumption is that data reduction has already converted raw detector
events into physically meaningful axes such as `H`, `K`, `L`, `|Q|`, energy
transfer `E`, intensity, and uncertainty. Once data are in those coordinates,
the fitting pipeline should not care whether the measurement came from HYSPEC,
another direct-geometry spectrometer, a triple-axis spectrometer, a powder
diffractometer, or a text export from an external workflow.

## Design principles

- Importers are adapters. A Mantid `MDHistoWorkspace` importer, a triple-axis
  text importer, and a powder diffraction importer should all translate file
  metadata into common data containers rather than forcing the rest of the
  package to know about the instrument.
- Axis names and units should be read from the file whenever possible. If a file
  labels an axis as `DeltaE`, `|Q|`, `H`, `[H,H,0]`, or similar, the importer
  should preserve that label and infer a broad axis role from it.
- Instrument-specific conventions should stay near the importer. For example,
  HYSPEC HHL data need a projection-specific mapping from `[H,H,0]` and
  `[H,-H,0]` to `(H,K,L)`. That belongs in a HYSPEC adapter/helper, not in the
  general fitting framework.
- `DataGroup`, `DatasetEntry`, and `FitModelSession` are the generic fitting
  layer. They may hold MDHisto data, point data, susceptibility curves, powder
  cuts, or later polarized-neutron containers.
- Resolution functions are per-dataset model components. They can depend on
  instrument and settings, but the optimizer sees only model predictions,
  observed data, uncertainties, masks, and weights.

## Current state

The current examples use Mantid MDHisto NeXus files because those are the first
local data available in the repository. That is a starting adapter, not a
package boundary. The same fitting framework is intended to support reduced
text tables, one-dimensional cuts, powder averages, and other reduced neutron
or bulk-measurement data as additional importers are added.
