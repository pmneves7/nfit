# API reference

## Reduced data, import adapters, and viewing

nfit importers translate source axes, units, masks, intensities, uncertainties,
and metadata into common containers. Supported paths include Mantid MDHisto,
file-backed MDEvent and raw direct-geometry events, powder cuts, MPMS, PPMS,
and general point tables.

Mantid `SaveMD` / `MDHistoWorkspace` NeXus files can be loaded with
`load_mantid_mdhisto_nxs`. The returned `MDHistoData` stores axis metadata plus
same-shaped `signal`, propagated `errors`, `mask`, and `num_events` arrays. By
default, the importer avoids copying bulky ancillary NeXus groups such as
`experiment0`; pass `copy_metadata=True` when a metadata tree copy is needed.
Axes expose broad roles inferred from file labels and units, such as `h`, `k`,
`l`, `q_modulus`, `momentum_projection`, and `energy_transfer`.

Common-container numerical arrays are read-only. `with_updates(...)` returns a
new container; `mutable_copy()` returns an isolated writable working copy.
When the container belongs to a `DatasetEntry`, finish an edit with
`dataset.replace_data(...)` so dependent caches and fingerprints are
invalidated.

MDEvent NeXus files are supported without Mantid through `nfit.mdevent`:

- `inspect_mdevent_workspace(path)` reads run and orientation metadata without
  loading the event table.
- `mdevent_dataset_group(...)` creates file-backed run entries sharing one
  reduction configuration.
- `append_mdevent_file(group, path)` adds a compatible event file without
  duplicating shared metadata.
- `bin_mdevent_group(...)` streams events, converts `Q_sample` to HKL, and
  performs native proton-charge and detector-trajectory normalization. Its
  optional `progress_callback` uses the standard rebinner event dictionary.
- `load_detector_normalization(path)` reads processed detector values for
  vanadium efficiency and bad-detector masking.

This pathway requires `h5py`. Numba accelerates detector trajectories when
available and has a NumPy fallback; Mantid is neither imported nor launched.
The returned `MDHistoData` carries `normalization_denominator` and
`zero_event_bins_are_measured` metadata. Covered zero-event bins have signal
zero and a finite 68.27% Feldman--Cousins upper-limit uncertainty; uncovered
bins remain masked and non-finite. A zero-count bin has no event row and
therefore has no stored `errorSquared` value of its own. nfit uses `1.29` times
the uncertainty that one representative event would have in that bin. This is
the `[0, 1.29]` Feldman--Cousins interval for zero observed counts and zero
known background (Table II of [Feldman and Cousins](https://arxiv.org/pdf/physics/9711021)). It
estimates the representative one-event scale from the nonzero accepted events
in the requested volume as `sqrt(sum(errorSquared_i) / N)`, then divides by the
bin's normalization. This scale is not an nfit fit weight or dataset scale
factor. See
[Measured-zero uncertainties](data_import.md#measured-zero-uncertainties) for
the three coverage/count cases and a numerical example.

The GUI module also provides `read_isaw_ub`, `write_isaw_ub`, and
`ub_from_lattice_orientation` for scripting the same UB workflow. ISAW matrices
are transposed on disk, and orientation construction uses the IPNS frame with
beam `+x` and vertical `+z`.

Use `slice_viewer(data)` for the PySide6 interactive viewer. It accepts one
`MDHistoData` object or a sequence with optional `dataset_names` and supports
axis selection, hidden-axis integration, channel selection, masks, model
overlays, smoothing, and script export. Smoothing is visual only and never
changes source or fitting arrays. See [Data viewer](data_viewer.md)
for the complete control reference.

For data with at least three dimensions, Volumetric mode provides PyVista
volume and isosurface rendering. It exports `.vtr`, `.vtp`, `.ply`, `.stl`,
`.gltf`, PNG, and orbit movies. Rendering-time smoothing does not affect
numerical grids or surface-model exports.

Use `plot_mdhisto_slice(data, ...)` when a non-interactive Matplotlib colormap
figure is preferred, for example in notebooks or batch scripts. Use
`plot_mdhisto_line(data, ...)` for one-dimensional MDHisto outputs such as
shape `(1, 1, 1, N)`, and `plot_mdhisto_auto(data, ...)` to dispatch to a line
plot for exactly-one-non-singleton data or a slice figure otherwise. The
`plot_mdhisto_waterfall(data, ...)` backend accepts one multidimensional
MDHisto dataset or a sequence of compatible 1D datasets and reproduces the
interactive waterfall controls without Qt. `prepare_mdhisto_waterfall`
returns the reduced traces and propagated errors for custom plotting.
`waterfall_step_bounds` gives the native-bin/full-span bin-width limits and
`waterfall_absolute_max` gives the data-dependent offset limit. The
`color_range` and `marker_face="outline"` options reproduce the interactive
colormap-range and per-trace marker-fill controls. Exported
scripts from the GUI call the matching slice, line, or waterfall backend and
include the current display settings.

Saved workspace plots use `PlotEntry`, `render_plot`, and `render_project_plot`.
They return Matplotlib figures and never construct Qt widgets, so generated plot
scripts run in batch or headless environments as well as interactive Python.

Project GUI fit results store fit and residual channels directly. When the Qt
viewer sees compatible stored channels, it can render linked data/fit or
data/fit/residual panels while preserving the current plotting settings.

`dataset_workflow_script(project, dataset_id)` generates GUI-free Python that
rebuilds a supported source-backed dataset through preparation.
`dataset_workflow_plan(...)` returns its versioned dependency graph, and
`render_workflow_script(...)` renders a supported graph. See
[Workflow scripts](workflow_scripts.md) for the current node coverage.
Analysis equivalents are `analysis_workflow_plan(...)` and
`analysis_workflow_script(...)`. `prepare_analysis_inputs(...)` and
`run_project_analysis(...)` use the same prepared data as the viewer.
`fit_workflow_plan(...)` and `fit_workflow_script(...)` export the live
workspace model and active optimizer configuration.

Launch the project explorer with `nfit`. The explorer manages saved
projects, workspaces, datasets, dataset groups, masks, models, rebinned dataset
views, and fit timelines. The GUI is documented in
[GUI workflows](gui_workflows.md).

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

`rebin_nd_symmetry` accepts reciprocal-HKL operation matrices and streams each
transformed image through the same accumulator. The project GUI resolves
space-group, point-group, Jones-faithful operation-list, and geometric-generator
syntax through `SymmetrySpec` and `resolve_symmetry`. A space group contributes
only its point-group rotations: translations are not applied to reciprocal-space
coordinates, and energy transfer remains unchanged.

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

`NDRebin` adaptively selects its CPU implementation. Small jobs use the NumPy
backend to avoid JIT startup overhead. When optional Numba support is installed,
jobs with at least 500,000 source points use a fused kernel that streams
each batch directly into the four output accumulators. This avoids both the
full point-by-dimension bin-index array and the repeated output-grid-sized
`bincount` temporaries used by the NumPy implementation. Pass
`backend="numpy"` or `backend="numba"` to force a backend for testing and
benchmarking; requesting unavailable Numba falls back to NumPy. The result's
`resolved_backend` records the implementation used, and `timings` reports
preparation, accumulation, normalization, and total wall time.

Run `python benchmarks/benchmark_rebin.py` from a source checkout for the
100-point and 500,000-point benchmark tiers. The memory-intensive 50-million
point tier is opt-in with `--sizes 100 500000 50000000`.

Large in-memory jobs also use adaptive threaded reduction. Dense reduction gives
each worker private output accumulators and is selected only when those arrays
fit within `max_parallel_bytes` (512 MB by default). If dense copies do not fit,
auto mode uses sparse touched-bin maps only when estimated occupancy is at most
5% and the maps fit the same budget; otherwise it runs the fused kernel on one
worker. With `workers=None`, the shared `NFIT_NUM_THREADS` setting, Linux CPU
affinity, cgroups, and SLURM allocations set the worker ceiling. Explicit worker
counts and `parallel_strategy="serial"`, `"dense"`, or `"sparse"` are useful
for controlled benchmarks. Results report `resolved_workers` and
`resolved_parallel_strategy`.

For data larger than memory, use a rewindable batch source with
`rebin_nd_stream`. `ArrayRebinSource` wraps arrays and memory maps; custom
HDF5, NeXus, or Zarr adapters can expose the same `ndim`, `n_points`, and
`iter_batches()` contract and yield `RebinBatch` objects:

```python
from nfit import ArrayRebinSource, rebin_nd_stream

source = ArrayRebinSource(signal, coordinates, data_errs=sigma, batch_size=1_000_000)
result = rebin_nd_stream(source, num_bins=[40, 40, 80, 120])
```

Explicit limits require one source pass. Missing or non-finite limits trigger a
first pass over projected coordinate minima and maxima, followed by the
accumulation pass, so custom sources must be rewindable. Projection, validity
checks, and accumulation operate only on the current batch. Temporary source
memory therefore scales with batch size, although output accumulators still
scale with the complete output grid. Streaming batches use the same `workers`,
`parallel_strategy`, and `max_parallel_bytes` policy as in-memory rebinning.

## Modeling and fitting

The fitting API supports both the original single-dataset
`fit_least_squares(data, model, specs)` convenience function and the
simultaneous-fit `FitProblem` workflow. `FitDataset` carries dataset-local
weights, preprocessing transforms, and optional instrument resolution, while
`ModelSpec` wraps the shared physics model. Serializable GUI models use
`ModelComponentSpec`; `compile_fit_problem` converts those components and their
dataset sharing rules into a `FitProblem`. Deterministic fits support robust
least-squares losses and optional differential-evolution initialization.
Posterior checks use `emcee` through `sample_problem_parameters`. See the
[modeling pipeline](modeling_pipeline.md) page for the recommended structure.

Serializable component types are described by `ModelDefinition` entries in
`MODEL_TYPE_REGISTRY`. `register_model_definition`, `model_definition`, and
`model_plot_definitions` are the public extension points for numerical
factories, GUI metadata, diagnostics, reports, plots, and serialization.

```{eval-rst}
.. automodule:: nfit.model_registry
   :members:

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

.. automodule:: nfit.spectral_channels
   :members:

.. automodule:: nfit.quantities
   :members:

.. automodule:: nfit.spin_fluctuations
   :members:

.. automodule:: nfit.form_factors
   :members:

.. automodule:: nfit.crystal
   :members:

.. automodule:: nfit.fitting
   :members:

.. automodule:: nfit.pipeline
   :members:

.. automodule:: nfit.resolution
   :members:

.. automodule:: nfit.rebin
   :members:

.. automodule:: nfit.plotting
   :members:

.. automodule:: nfit.analysis.core
   :members:

.. automodule:: nfit.analysis.registry
   :members:

.. automodule:: nfit.analysis.bragg
   :members:

.. automodule:: nfit.analysis.spectral
   :members:

.. automodule:: nfit.analysis.curie_weiss

.. automodule:: nfit.analysis.heat_capacity
   :members:
```

## Compatible direct-geometry spectrometer TOF reduction

Compatible direct-geometry spectrometer event NeXus files are supported through
`nfit.raw_dgs`. This adapter expects compatible event banks and run logs plus an
embedded Mantid instrument definition; it is not a generic importer for every
direct-geometry instrument.
`inspect_raw_dgs_run(path)` reads run metadata without reading event arrays;
`raw_dgs_dataset_group(paths, ...)` creates lightweight entries sharing one raw
reduction setup; and `bin_raw_dgs_group(...)` resolves detector positions from
the embedded IDF and streams banks into an HKLE histogram. It accepts the same
coordinate-basis and progress-callback conventions as the MDEvent reducer.

This native reducer integrates retained raw proton-pulse charge in
microampere-hours and builds an MDNorm-style detector-trajectory denominator.
For Shiver-compatible raw imports, processed vanadium and explicit mask files
are binary detector masks rather than detector-value weights. By default it
applies Mantid's wavelength-dependent He-3 tube-efficiency correction, when
the embedded IDF supplies tube geometry and pressure, thickness, and
temperature parameters, followed by the `ki/kf` direct-geometry correction.
Both corrections multiply the event uncertainty by the same factor and are
recorded in returned metadata. Incident energy and
T0 follow Mantid GetEi v2 for each run from monitor locations in its embedded
instrument definition. For parameter-defined paths such as CNCS and HYSPEC,
nfit applies Mantid's published `t0_formula` using the requested incident
energy; it does not use empirical per-instrument timing offsets.
Covered bins with zero accepted events retain a zero signal and use nfit's
normalization-scaled 68% Feldman-Cousins upper-limit uncertainty. It does not
invoke Mantid. The full ordered raw-event reduction, including detector masking,
bad-pulse charge selection, TOF-to-HKLE conversion, He-3 and `ki/kf` event
weights, trajectory normalization, and measured-zero handling, is documented
in [Raw TOF reduction sequence](data_import.md#raw-tof-reduction-sequence).
