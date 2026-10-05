# Performance notes

nfit selects conservative CPU and memory strategies automatically. Most users
only need to limit grid sizes and install an optional acceleration backend for
large jobs.

## Rebinning and event reduction

Native DGS raw reconstruction and cached-event accumulation currently process
runs and chunks sequentially. Detector-trajectory normalization uses parallel
workers when the compiled backend is available. The CPU limit is a ceiling,
not a request to parallelize every stage. In the main window, 1.6% Project CPU
on a 64-CPU machine corresponds to approximately one fully occupied core.

Manual DGS trials compare saved-project wall time as well as event-kernel time.
Independent producers are not enabled solely to increase CPU utilization:
scheduling, cache reads, HDF5 serialization and worker setup can erase their
compute gains. The benchmark tooling records raw-event and whole-histogram
parity separately; see `benchmarks/README.md` for the reproducible trial route.

Small rebinning jobs use NumPy. Large jobs can use a fused Numba kernel with
memory-bounded private accumulators. Extremely sparse output grids may use
touched-bin maps when their estimated occupancy is at most 5%; otherwise nfit
falls back to a serial fused kernel if threaded copies exceed the memory
budget.

`rebin_nd_stream` and MDEvent reduction process source data in bounded batches,
so source files may exceed RAM. Explicit output limits permit one pass;
automatic limits require a discovery pass. The batch target bounds temporary
event storage but cannot reduce the persistent output arrays.

Large regular MDHisto rebins also generate coordinates and geometric coverage
in chunks, avoiding full-grid coordinate meshes and filtered copies. Automatic
coordinate limits use the extrema of the original grid, including symmetry
images. Histograms with fewer than 250,000 source bins retain the direct path;
this is about 10 MB for five float64 channels plus a boolean mask. The
threshold reflects coordinate-processing cost, so it uses bin count rather
than a fixed file size. Discrete and
tolerance-based axis clustering still uses the direct path because it needs
the complete coordinate distribution. These strategies preserve float64
signals, errors, normalization, masks, and coverage; they do not compress or
reduce the precision of active numerical arrays.

Streaming also uses fused accumulation for explicit output edges, including
nonuniform grids and fractional coverage. Edge membership is resolved against
the supplied edges before accumulation, preserving values exactly on a bin
boundary. Invalid, non-finite coordinates are ignored.

File-backed datasets are loaded on first use and retained by their
`DatasetEntry`; viewers, analyses, and composites share that single loading
path. nfit does not keep a second raw-array cache.

Before reduction or binning, nfit checks the planned allocation against the
remaining process RAM budget and available system RAM. Resolved-grid checks run
before output buffers are allocated, including automatically determined grids.
See [Resource Manager](resources.md) for explicit loading and unloading.

Use `benchmarks/benchmark_rebin.py` to measure representative grids on the
target machine.

### Native DGS event projection

Uniform HKLE grids with the Mantid precision policy combine coordinate
projection and ordered event accumulation in one compiled pass. The transform
coefficients and float32 multiply/add rounding are shared with the reference
projector. No projected-coordinate table is allocated for each symmetry copy.
The optional same-event variance treatment and detector-trajectory exposure
calculation are unchanged. Nonuniform grids, high precision, and unavailable
compiled acceleration retain the reference path.

Saved detector masks reuse an immutable parsed result for exactly identical
instrument-parameter text. The cache holds at most 16 texts, accepts texts up
to 1 MiB, and reads source metadata on every request. Changed masks and distinct
instrument text miss the cache. Source checks and numerical run settings still
apply on every operation.

Raw reduction reuses successful scalar run inspection between import and
binning after checking complete file identity and the monitor policy. It retains
no monitor or event arrays. Detector distances and directions are calculated
once per immutable resolved geometry and static detector selection; changed
run-log geometry, masks, and mixed instruments retain separate snapshots.

Compiled HKLE normalization pools charges for literally identical trajectory
tasks before batching. Detector identities, directions, normalization and masks
must first match completely. Tasks also require identical affine-transform bits,
effective incident energy and the same source-energy interval after clipping
to the requested energy grid. Raw collections with mixed geometry bypass this
pooling; MDE inputs pool separately within each checked geometry. Positive finite
charges add linearly; event numerators, event variances and contribution counts
are accumulated from every original source as before. The Python reference path
is unchanged. This reuse preserves exposure up to floating-point summation
rounding and does not create a shared cache between different binnings.

Named raw-DGS HKLE binnings can reuse a compatible unsubtracted histogram when
its symmetry operations are an exact multiset subset of the requested operations.
Only missing operations are reduced and binned; retained event numerator,
variance, counts and exposure are added before recalculating intensity,
uncertainty, coverage and masks. Complete source, geometry, calibration, weight,
mask, grid and policy signatures must still match. This uses the existing lazy
project histogram caches; it adds no saved format or application preference.

Cache validation and joining scan the whole output grid. For grids of at least
250,000 cells, nfit uses incremental symmetry only when known source event counts
and cached operations imply at least eight saved event projections per output
cell. Unknown counts or smaller workloads use full recalculation without loading
the partial cache. This conservative cost estimate prevents cache scans from
slowing down a few-run job on a very large grid; it changes no scientific setting.

This reuse applies to independent-copy variance on fixed HKLE grids. Powder,
metadata axes, automatic bounds, reduced MDE inputs, source dependencies and
copy-covariance treatments retain full replay. Partial masks never remove retained
numerator contributions. Grouping floating-point additions can change their last
bits, while counts and exposure support retain the full-replay result. Direct
stateless reduction calls have no project histogram cache to reuse.

`benchmarks/profile_dgs_workflow.py` measures a real MDE source, including reads,
run selection, trajectory normalization, and finalization. It reports first and
repeated calls separately and can compare every output cell with the reference
path. `benchmarks/profile_dgs_event_kernels.py` isolates projection and accumulation.

For raw-DGS comparisons, `benchmarks/benchmark_dgs_nfit_workflow.py` measures
import, reduction/cache construction, binning, native project saving, lazy
reopening, and later rebins. The separate manual
`benchmark_dgs_mantid_workflow.py` uses an installed Shiver environment and
actual MDE/histogram files. Run both sequentially on the same hardware with
one shared configuration. First saved-dataset time and later reuse workflows
are reported separately; diagnostic array scans and exports are excluded.
Kernel speedups should not be interpreted as reduction or whole-project speedups;
raw reconstruction, background replay, and archive I/O require separate timings.
For source-candidate acceptance, `NFIT_DGS_BENCHMARK_SOURCE` may select a staged
directory containing `nfit`. The harness records that runtime explicitly and
hashes its numerical modules; without this variable it uses the installed bundle.

### Measured-background replay

Measured-event background replay uses a compiled parallel Numba kernel when
available. Each original event is transformed at the sample angles and symmetry
operations; copies landing in the same output bin are combined before their
variance is accumulated. Worker-local lookup tables avoid repeatedly searching
all earlier angle copies. A deterministic compiled scatter updates the shared
output arrays without allocating a full four-dimensional histogram per worker.
The energy-bin lookup is performed once per source event and reused across all
angles and momentum symmetry operations; edge handling and accumulation order
are unchanged.

For near-uniform momentum axes, replay predicts a bin index arithmetically,
then verifies membership against the original edges. A failed prediction uses
binary search; nonuniform axes use binary search directly. This changes neither
edge conventions nor floating-point accumulation order.

Both temporary memory and event-transform work per batch are bounded. Progress
and cancellation are checked before replay and between batches, and the dialog
reports the backend and CPU count. The first call may need JIT compilation;
compiled kernels are cached for later sessions. Without Numba, or with JIT
compilation disabled, the reference
NumPy implementation remains available and is explicitly identified in progress.
The work still scales with the number of source events, angles, and symmetry
operations, and must be repeated for different output grids.

Use `benchmarks/benchmark_measured_background_replay.py` to compare the warmed
event kernel with the NumPy reference at 722 synthetic sample angles. This
benchmark excludes file loading and detector-trajectory normalization and is
not an end-to-end runtime prediction for a project.

`benchmarks/replay_candidates.py` compares the binary-search baseline, an
intermediate-array transpose, disjoint-energy parallel scatter, and the production
index predictor. It checks bitwise equality and includes allocation and transpose
costs. The transpose and parallel-scatter variants are experiments only.

A local four-worker test (20,000 source events, warmed kernels, median of 5–7
interleaved repetitions) measured:

| Workload | Binary lookup | Checked predictor | Speedup |
| --- | ---: | ---: | ---: |
| 722 transforms, 884,736 output bins | 146 ms | 79 ms | 1.84× |
| 722 transforms, 3,538,944 output bins | 185 ms | 100 ms | 1.85× |
| 2,888 transforms, 12,000 source events | 290 ms | 156 ms | 1.86× |
| 12 transforms | 3.7 ms | 2.7 ms | 1.35× |
| 722 transforms, nonuniform axes | 154 ms | 151 ms | 1.02× |

These are replay-kernel measurements, not end-to-end project speedups; they
exclude file I/O, normalization and JIT compilation. Numerical outputs matched
bitwise in every trial. The transpose was slower on the main workload; parallel
scatter improved total replay by only about 9%, so neither is enabled. Recorded
workloads and per-stage timings are in
`benchmarks/results/replay_candidates_20260928.json`.

Raw direct-geometry event reduction also processes symmetry coordinate copies
one at a time, so their temporary coordinate storage does not grow with the
number of symmetry operations.

## Performance preferences and benchmarks

**File → Preferences → Performance** provides one **CPU limit** and one **RAM
limit** in GB (1 GB = 1,000,000,000 bytes). The GUI accepts three decimal
places and saves the nearest MiB (1 MiB = 1,048,576 bytes), preserving existing
budgets and the scripting API's `ram_limit_mb` units. The CPU limit bounds nfit's shared
worker allocation, including large-array archive compression and loading.
Admission checks charge current process RAM, managed array storage, and pending
numerical work against the RAM limit. This is not an operating-system memory
cap; third-party allocations can occur outside managed entry points.
Small operations run serially, and large operations may use fewer workers to
stay within the memory allowance. Batch sizes and archive thresholds are
chosen internally. Existing scientific rebin and saved-plot recipes remain
unchanged.

Preferences are machine-local in `~/.config/nfit/performance.json`; set
`NFIT_PERFORMANCE_FILE` to use another file. The headless equivalents are
`load_resource_limits()` and `save_resource_limits(cpu_limit=4,
ram_limit_mb=8192)` in `nfit.performance`. Zero selects Auto: CPU allocation
follows the process configuration, and managed RAM normally uses 25% of
currently available memory. Older percentage-based RAM preferences remain
effective until changed. The legacy `load_performance_settings()` and
`save_performance_settings()` APIs remain available for scripts. Legacy per-rebin batch and worker fields remain readable in projects and
scripts, but project workflows choose their resources from the central limits.
Direct numerical APIs still accept explicit worker ceilings. Saving Auto in
Preferences resets any migrated RAM percentage to the normal automatic policy.

Use **Calibrate this machine…** for representative 3-D hard-binning and 4-D
fractional-binning workloads. Use **Benchmark this rebin…** in dataset or
composite rebin controls for the full selected configuration, including axes,
symmetry, masks, and output-grid settings. Candidate ranges scale to the CPU
allocation and currently available memory detected for the process. Trials
respect the central limits and report the effective CPU and batch ceilings. Small
machines retain low worker and 32, 192, and 512 MiB batch candidates. Larger
machines add intermediate and full CPU-allocation trials and batch targets up
to one eighth of available RAM, capped at 64 GiB. Linux cgroup memory limits
are honored. Calibration does not test every possible workload or worker count
and cannot establish a universal optimum.

Each candidate runs in a fresh process, warms up once, and reports the median
of two subsequent runtimes. Peak memory is the **trial process's peak resident
memory**, including imported libraries, input data, and warmup; it is neither
the batch target nor the total memory used by nfit and the trial together.
Compilation and source loading are warmed rather than measured as cold-start
costs. Real-data trials may need substantial additional memory and time, since
they run the full rebin rather than a sample. **Cancel** kills the active trial;
no live dataset, viewer cache, or preference is changed by running a benchmark.

The recommendation prefers fewer workers, then smaller batches, among timings
within 5% of the fastest. Machine calibration offers **Apply CPU
recommendation**, which updates the central CPU ceiling while retaining the
RAM limit. Batch sizes remain automatic. A real-data benchmark is informational
and does not change a binning configuration.

Both dialogs provide **Save benchmark script…**. Real-data exports write an
adjacent `.nfit` project snapshot using the normal project-save rules; source
files must remain available, and unsaved replacement data must first be saved
to a portable dataset file. Existing snapshot files are not overwritten.
The script contains an editable candidate list and uses the public API without
constructing Qt widgets:

```python
from nfit.performance_benchmark import benchmark_rebin

result = benchmark_rebin()  # machine calibration
# Or: benchmark_rebin(project, dataset_id="saved-dataset-id")
# Or: benchmark_rebin(project, group_name="Workspace1", node_id="nested-group-id")
print(result["recommendation"])
```

`benchmark_rebin` also accepts `candidates`, a `cancel()` predicate, and a
`progress(row)` callback. It never applies its result automatically.

## GUI and analysis caches

nfit keeps separate bounded caches for prepared point tables, viewer-ready
data, group composites, compiled model overlays, and analysis fingerprints.
They cache different stages and are not duplicate copies of one result.
Entries use stable dataset IDs plus a data revision and the relevant masks,
rebin settings, backgrounds, or model structure. `DatasetEntry.replace_data`
advances the revision; changing only a model parameter can still reuse compiled
geometry.

Completed viewer and composite binnings stay resident until explicitly unloaded
or invalidated by changed data/settings. There is no automatic compression,
oldest-result eviction, or separate disk-cache prompt for these project caches.
Repeated preparations share immutable arrays, including scaled and spectral
views. Multiple viewers can hold the same cube without copying it.
[Resource Manager](resources.md) lists the owners, shared storage, saved project
caches, and estimated memory freed by an unload.

Archive readers inspect NumPy headers before allocating the expanded arrays.
Process-wide reservations prevent concurrent operations from spending the same
available RAM. The logical cache footprint also counts untouched output arrays
and retained mapped storage, even when their pages are not resident. Checks
include temporary working arrays and resolved output grids; the process RSS
includes sources, viewers, models, analysis, libraries and other allocations.
These safeguards cover managed entry points, rather than imposing an OS memory
cap on Qt or third-party code.

Whole histogram cubes load into RAM for responsive viewing. Low-level scripts
can still request `read_dataset_artifact(..., memory_map=True, temp_dir=...)`;
mapping failure raises rather than falling back to an unplanned heap allocation.
Project viewers do not automatically select that storage mode.

Scientific temporary storage is allocated when an operation needs it. By
default, reduced-event and analysis staging lives beside the owning project or
source; save staging lives beside its destination. Resource Manager can select
an existing writable temporary directory for future allocations. Existing files
stay where their current owners placed them. The choice is recorded in the
project but is checked only when storage is used, so an unavailable old path
does not prevent opening an empty application or inspecting a project.

File-backed operations never silently move scientific staging into another
filesystem. On clusters requiring data in experiment directories, keep the
project and selected temporary directory there. Save replacement space is
reserved on the destination filesystem while the old archive remains intact;
filesystem quotas can still reject a write. Archive copy and decode work runs
outside the GUI thread, with bounded buffers and cooperative cancellation.
Unchanged saved NPZ members are streamed without recompression.

Closing a project releases its process-local binning and model cache references.
Explicitly unloading a cube preserves its saved backing and recipe. Closing a
viewer releases its references; arrays still owned by another viewer or cache
remain live. Python or the OS may retain freed pages for reuse.

Single-crystal MDEvent detector normalization groups runs with identical
detector geometry and evaluates all requested symmetry operations in the
compiled trajectory kernel. Each trajectory is clipped to the requested HKLE
box before internal bin boundaries are examined. These optimizations change
neither the event histogram nor the integrated normalization denominator; the
Python reference path is retained for numerical-equivalence testing. nfit uses
the central CPU ceiling and lowers the normalization worker count when the
thread-private output accumulators would exceed the managed RAM allowance.
The HKLE reducer reserves space for its three live event grids and its final
normalization grid before assigning private worker buffers. It allocates those
buffers once across detector geometries and progress batches, then sums them
once. Progress updates and cancellation checks remain between batches.
Peak-memory estimates include these worker buffers; existing project results
and source arrays remain additional memory.
For speed, small worker buffers are zeroed up front when their combined size
is at most 8 GiB and one eighth of the central RAM allowance. Larger buffers
use zero-filled NumPy storage. On systems that defer physical
allocation of untouched zero pages, resident memory depends on the bins
visited by the trajectories; an initially sparse grid can grow as more runs
are processed. The estimate therefore counts the full buffer capacity, even
when observed resident memory is lower. Reducing the CPU limit can reduce
private-buffer storage, but may increase integration time.

MDEvent HKLE and powder finalization reuses the owned event-sum arrays for
normalized signal and errors. The immutable result adopts those arrays, and
its normalization channel shares storage with the normalization metadata.
This avoids full-grid copies without reducing precision. Already-created
results retain their original storage until released or reloaded.

Large-memory nodes can therefore use multiple CPUs for large
four-dimensional grids, while memory-constrained machines still fall back to a
smaller worker count. **Benchmark this rebin…** measures the
complete saved configuration on the current machine; its exported script can
test a custom candidate list when a cluster node warrants a broader sweep. The
default benchmark sweep includes the machine's full detected CPU allowance in
addition to conservative smaller ceilings.

**Preferences → Performance → RAM limit** controls the shared managed memory
allowance. Resource Manager edits the same machine-local limits in decimal GB.
Auto normally uses 25% of currently available memory. A fixed ceiling is checked
against both current process usage and available RAM before additional
allocations. Thread-private accumulators and batch planning use the same
resource policy; no per-binning CPU or batch controls are needed.

An over-budget request stops before its managed allocation and offers Resource
Manager. Increase the budget or unload selected objects, then retry. Pending
results and peak working arrays are estimates; checks for automatic grids run
again once their actual dimensions are known. The rebin panel reports the
persistent payload estimate, or the actual resident bytes when available, and
the compressed saved artifact size without decoding the binning.

On macOS, available memory includes inactive and speculative pages that the
operating system can reclaim; this prevents the worker planner from falling to
one CPU merely because the file cache has consumed most completely free pages.

Process caches normally disappear when nfit exits. For projects whose raw data
are expensive to load or rebin, enable **File → Cache binnings** before saving.
The project then embeds current dataset and composite binnings. On the next
open, current cache entries are registered from their metadata and decoded only
when requested. Unchanged saved binnings are copied as compressed archive
members on Save and Save As; they are neither decoded nor recompressed.
Missing or signature-stale binnings are recomputed. This project-specific option defaults off
so ordinary project files remain small. Embedded caches carry a numerical format
version and a signature of the source, rebin settings, masks, and backgrounds.
nfit reuses compatible cache formats and discards a cache when its saved
signature no longer matches the live recipe. Incompatible older formats are
recomputed from their source data.

Large changed arrays use bounded parallel DEFLATE compression, and large NPZ
members can be decoded concurrently when the RAM allowance permits. Small
arrays keep the ordinary NumPy writer. Both paths produce standard, lossless
NPZ artifacts readable by existing nfit versions and NumPy. Opening a nested
artifact reads directly from its project member instead of first allocating a
copy of the entire compressed artifact. Decoded histogram arrays transfer to
the immutable container without a second full-array copy.

On systems with positional file reads, large channels use independent cursors
on one validated project snapshot. This avoids serializing their reads through
a shared seek lock and keeps the original file open if another operation
atomically replaces the project. Member checksums and NumPy header validation
remain active. Other systems and older compressed outer project members use
the shared reader.

Saving still writes and atomically replaces the complete project archive. A
large unchanged project therefore still incurs sequential file I/O, but no
array compression work. Loading an individual requested binning still decodes
its full arrays; the archive is not a chunk-addressable HDF5 or Zarr store.
These distinctions matter when estimating performance on a shared filesystem.

Histogram slice calculations select the requested region before sanitizing
coverage or allocating absent masks. Ordinary slice appearance edits reuse
the displayed slice; changes to scientific selections recompute it. These
optimizations retain the same integration, errors, masks, and coverage rules.

For an isolated comparison of artifact write/read times and numerical slice
extraction, run `python benchmarks/benchmark_large_arrays.py --workers 1,4`
from the source checkout. `--shape 96,80,64,24` selects the four-dimensional
test grid. The script uses temporary files, verifies array round trips, and
leaves project files and preferences unchanged. The CPU preference still caps
the requested worker counts; these measurements exclude GUI rendering and
are not estimates of shared-filesystem or remote-desktop latency.
MDEvent powder reductions integrate detector-trajectory normalization in
compiled parallel batches. Progress and cancellation are checked between
batches, including while a save is refreshing several cached powder binnings.
Source-backed cache signatures use the source path, size, and modification time,
so lazily loading an unchanged source after opening a project does not invalidate
its restored binning. A cached binning of a live derived recipe records the
recipe identity and its complete resolved dependency signature, so it can be
restored after restarting nfit without loading source data or rebinning. Older
saved live-recipe cache entries with a process-local memory token are accepted
only when every other signature field matches, then rewritten with the stable
identity at the next save. Editing one named binning invalidates only that result;
other fit or visualization binnings remain available when their own inputs and
settings are unchanged. When an operation processes multiple named binnings,
the progress dialog reports how many binnings are complete and identifies the
current binning in the existing status area. Dataset and composite icons carry
a green dot while all of their enabled named binnings have current cache
signatures.

## Heisenberg RPA

For $M$ fitted momentum points and $N$ magnetic sublattices, the scalar model's
batched eigendecomposition costs approximately $O(MN^3)$. nfit reduces this
cost without changing the result:

- masked data are prepared once per fit;
- datasets with zero fit weight are evaluated only after optimization;
- phase geometry, form factors, the Bose factor, and the channel conversion's
  linear factor are cached per dataset, so nothing that depends only on the
  fitted points is recomputed per optimizer iteration;
- the fitted momentum grid is deduplicated with a single lexicographic sort;
- powder datasets orient only the distinct $|Q|$ values, so a map with one
  momentum bin per energy row builds its geometry from a grid smaller by the
  number of energy bins;
- centered cells are reduced to translationally distinct magnetic sites when
  the interaction permits it;
- exchange-independent geometry is reused between value and Jacobian calls;
- mode weights that depend only on $Q$ are computed once per
  eigendecomposition rather than once per fitted point;
- analytic scalar-model derivatives avoid one model evaluation per parameter;
- gradients and tensor-path susceptibilities stream in bounded point blocks;
  and
- GUI views, overlays, and analysis fingerprints use bounded caches.

Primitive-cell reduction is often the largest exact saving because the
eigendecomposition cost is cubic in $N$. See
[Heisenberg RPA](heisenberg_rpa.md#primitive-cell-reduction).

For many small matrices, a fused Numba Jacobi eigensolver removes repeated
LAPACK call overhead. Larger matrices use LAPACK with batch-level threading,
which pays from roughly $N=12$ upwards; below that the per-call cost is too
short to amortize dispatch and the batch runs on one thread.

## Compute backends

The resolvent contractions support:

| backend | use | installation |
| --- | --- | --- |
| `numpy` | default and small jobs | included |
| `numba` | large CPU jobs | `pip install nfit[accel]` |
| `cupy` | large GPU jobs | `pip install nfit[gpu]` with a matching CuPy wheel |

`nfit.set_rpa_backend("auto")` is the default. It uses NumPy below roughly
50,000 points, then selects an installed accelerated backend when worthwhile.
Set `NFIT_RPA_BACKEND` or call `set_rpa_backend()` to override it.
`nfit.available_rpa_backends()` reports what is available. An unavailable
forced backend falls back to NumPy.

The selection also governs the anisotropic (tensor) evaluator. Above the same
threshold its unpolarized channel uses a fused Numba kernel that forms each
point's $3\times3$ susceptibility and contracts it with the polarization
projector in one pass, instead of materializing per-point mode and tensor
intermediates. The NumPy path stays the reference and the only path where Numba
is absent; the two agree to floating-point precision.

## Tight-binding electronic structure

Bands, density of states, and constant-energy surfaces share a bounded
eigensystem service. It applies the same float64/complex128 Fourier
Hamiltonian and Hermitian NumPy solver in serial or across independent
wavevector batches. Unprojected calculations use `eigvalsh`, avoiding the
eigenvectors needed only for orbital projections.

Three exact reductions cut the sampled wavevector count before any
diagonalization, none of which changes a returned value:

- Gaussian and tetrahedron density of states integrate on the irreducible
  mesh, using orbit multiplicities as weights. Gaussian DOS additionally saves
  the broadening kernel, because it never needs the full ordered grid.
- Constant-energy surfaces drop the repeated zone face, then reduce to the
  irreducible wedge, and gather eigenvalues back onto the full grid.
- The orbit decomposition itself depends only on the certified rotations and
  the mesh shape and shift, never on parameter values, so it is memoized. That
  matters because reducing a large mesh costs an order of magnitude more than
  diagonalizing the resulting irreducible set, and a fit or a convergence scan
  repeats the same reduction many times.

A projector that does not cover the whole basis is not certified as symmetry
invariant, so requesting one declines the reduction under `auto` and raises
under `reduced`.

The component settings are:

- `electronic_backend`: `auto`, `numpy`, `threaded`, or explicit optional
  `cupy`;
- `electronic_workers`: zero for the nfit allocation or a positive limit; and
- `electronic_max_batch_mb`: the temporary Hamiltonian/eigensystem target.

`auto` does not select a GPU. A compatible CuPy installation can be selected
explicitly and uses the same precision. An unavailable requested GPU falls
back to NumPy. `NFIT_ELECTRONIC_BACKEND` sets the process default, while
`NFIT_NUM_THREADS` supplies the automatic CPU allocation. Execution details
and the absence of numerical approximations are stored in result provenance.

Immutable electronic models cache their reciprocal lattice and scientific
structure digest. For a repeated path or mesh, nfit also caches the
Fourier-transformed basis

\[
H(\mathbf k;\mathbf p)=H_0(\mathbf k)+
\sum_a p_a H_a(\mathbf k).
\]

Changing a fitted onsite, hopping, or spin--orbit coefficient then combines
the cached matrices before diagonalization instead of repeating their Fourier
transforms. The cache is bounded and automatically uses the direct
real-space path when the component basis would exceed its memory limit.

For total DOS, `k_mesh(..., symmetry="auto")` can use exact orbit
multiplicities from a GUI-built model's certified reciprocal symmetry.
`symmetry="full"` is the default, while `"reduced"` fails if reduction cannot
be certified. Wannier90 models, reduced-dimensional meshes, incompatible
shifts, and component-level projected DOS retain full sampling in automatic
mode.
Fermi-surface extraction also retains a regular full grid because marching
contours require its topology.

## Electronic response

The Lindhard and electronic-RPA path separates three memory controls:

- `response_max_batch_mb` bounds temporary Hamiltonian/eigensystem waves;
- `response_transition_max_batch_mb` bounds particle--hole matrix elements and
  denominators; and
- `response_cache_mb` plus `response_cache_entries` bound reusable immutable
  eigensystems, completed bare responses, and CuPy-resident Hamiltonian
  components.

Cache keys include the electronic-model digest, sampling, operators,
thermodynamic state, broadening, and execution inputs. Varying only an
interaction vertex can reuse the completed bare susceptibility; varying only
broadening can still reuse eigensystems. Changing an onsite, hopping, or SOC
coefficient creates a different parameter-point digest while retaining the
same structural digest and reusable process-level CPU Hamiltonian components.
A compiled fit constructs one response evaluator per observable component, so
all compatible datasets and dataset groups share its bounded response cache.
Each insertion records its retained array bytes once, so eviction remains
constant-time as a fit visits new parameter points. Host and device entries
have independent byte accounting. Cache contents are never serialized as
scientific state.

On a complete uniform mesh, commensurate transferred wavevectors use an exact
periodic permutation of the base eigenvalues and eigenvectors. No shifted
Hamiltonian is assembled or diagonalized. Whether a wavevector is commensurate
is decided once per distinct wavevector rather than once per response point,
which matters because a constant-Q cut repeats one wavevector at every energy.
Full-volume response evaluation likewise groups all points by distinct
transferred wavevector with one stable sort, rather than rescanning the entire
point array for every group. The resulting susceptibility is still returned in
the original full-volume order for viewer-side binning and integration.
Off-mesh points remain direct unless
a positive interpolation tolerance is configured. Validated interpolation
uses only required periodic-linear stencils, refines through commensurate mesh
divisors, and falls back to the direct reference when `auto` cannot meet the
tolerance.

Hamiltonian evaluation caches parameter-independent Fourier coefficients for
repeated paths and meshes. Lindhard energy points and electronic-RPA linear
systems are processed in bounded batches.

On CPU, `response_transition_backend="auto"` first recognizes ordered
orbital-pair operators and applies an exact factorized tensor contraction.
This lowers the arithmetic scaling of the Hubbard--Hund bubble and avoids
dense one-hot operator matrices. Other sufficiently large operator problems
use the fused Numba contraction when their batch size can amortize compilation
and dispatch. The NumPy and Numba reference paths can always be forced; the
resolved implementation is recorded in susceptibility provenance.

For long constant-Q energy scans, explicitly selecting
`response_transition_backend="numba"` with several `response_workers` can
also accelerate the implicit scalar-spin response. Automatic selection keeps
that case on NumPy. Local 32 x 32 mesh benchmarks with 6-12 bands and 401
energies measured about 5.5x speedup with four Numba workers; short requests
showed little benefit. These timings exclude JIT startup and are not full-fit
runtime predictions.

The repository's `benchmarks/benchmark_lindhard_spectral.py` evaluates an
experimental approximation: transition masses are deposited on a gap-energy
grid and convolved with the complex retarded denominator using an FFT. On
tested long scans it was 2.7-8.5x faster than the fastest measured exact CPU
path, including preparation, with peak-normalized complex errors of
0.004-0.015% at grid spacing one sixteenth of the broadening. It was slower
for three-energy requests. This experiment is not a production backend:
large operator bases, narrow broadening, and interaction poles need additional
memory and error controls. Workloads, convergence data, and limitations are
recorded in `benchmarks/results/lindhard-acceleration.md` and the adjacent
JSON files.

For sufficiently large calculations with several distinct transferred
wavevectors, nfit partitions the total `response_workers` allocation across
the independent q groups and assigns the remaining workers within each group.
The base mesh eigensystem is shared rather than recomputed. Off-mesh shifted
eigensystems also use the bounded response cache, including in parallel q
batches. Changing only broadening or occupations can reuse them; a changed
Hamiltonian or mesh invalidates reuse. Cold misses are still diagonalized in
batches, and each retained q slice owns its arrays so eviction can release
its accounted storage independently. Fourier Hamiltonian
assembly uses thread-safe direct contractions while those q groups run in
parallel. Small jobs and single-q energy scans stay on the lower-overhead
serial-q path.

With explicit `response_backend="cupy"`, Fourier Hamiltonian components,
eigensystems, occupations, magnetic matrix elements, denominators, and the
Lindhard contraction remain on the GPU. nfit copies only the completed complex
susceptibility to host memory. Reusable device arrays obey the same configured
cache limits; an oversized Hamiltonian-component basis falls back to bounded
on-device assembly without changing the calculation.

Threaded response execution first constructs a bounded wave of Hamiltonians
serially and then diagonalizes those completed matrices in parallel. This
prevents Apple Accelerate or another LAPACK implementation from overlapping
with complex Hamiltonian assembly. Results retain input order and use the same
NumPy eigensolver as the serial reference.

`response_validate_backend=true` compares a deterministic eigensystem probe
with serial NumPy before a new model digest uses threaded or CuPy execution.
Requested CuPy response execution therefore fails clearly when CuPy is
unavailable or its eigenvalues exceed the configured tolerance, even though
the lower-level electronic service retains its documented NumPy fallback.

`response_symmetry="auto"` uses only the certified little group that fixes all
requested wavevectors. It currently applies to nfit-built implicit isotropic
spin responses. Imported models, explicit-spin operators, orbital-pair
matrices, and generic wavevector sets retain the full mesh. `"reduced"`
requires certification and `"full"` disables the attempt.

The convergence plot varies mesh and broadening independently. Its mesh metric
compares meshes at fixed $\eta$; its broadening metric compares $\eta$ values
at fixed mesh. This distinction is required because extra broadening can hide
an underconverged mesh.

Automatic density selection is observable-specific. DOS certification compares
total and projected energy-resolved curves, including an integrated error.
Lindhard certification compares the full complex Cartesian tensor at declared
$(\mathbf Q,E)$ points. Both require two successive passing refinements and
store the tested middle mesh from the final two passing comparisons before
fitting; they never change the objective's sampling during optimization.
Fermi-surface topology and geometric-distance certification remain future
work. See
[Automatic Brillouin-zone sampling](electronic_sampling.md).

For jobs larger than one process, `partition_response_points` and
`merge_response_chunks` define deterministic scheduler-neutral work units.
`response_slurm_array_script` supplies an editable Slurm wrapper, but nfit
does not submit jobs or prescribe a shared-filesystem policy.

## Threads

nfit respects the CPUs available through affinity, cgroups, or a SLURM
allocation. **Preferences → Performance → CPU limit** provides the common
ceiling. Scripts can request fewer workers with `NFIT_NUM_THREADS`,
`nfit.set_num_threads(n)`, or a scoped thread budget; these requests cannot
exceed the machine allocation or the preference ceiling.

Batch-level eigendecomposition pins BLAS/LAPACK to one thread to avoid nested
oversubscription. On Apple silicon, Accelerate's shared AMX unit can make a
smaller worker count faster for large matrices; lower `NFIT_NUM_THREADS` if
benchmarking shows this behavior. Small primitive-cell models do not use the
threaded LAPACK path.

## Tensor interactions

Tensor interactions promote the scalar $N\times N$ exchange matrix to
$3N\times3N$:

- **Field off:** one Hermitian eigendecomposition is reused for every energy
  at a momentum. Cubic scaling makes the matrix work roughly 27 times the
  scalar cost at fixed $N$.
- **Field on:** the gyrotropic local response requires a batched linear solve
  at each fitted point.
- **Dipoles:** the Ewald tensor is cached densely and disables primitive-cell
  reduction. Its complex array has shape $(n_Q,N,N,3,3)$, where $n_Q$ is the
  number of distinct momentum points.
- **Gradients:** tensor mode uses central differences; scalar mode retains its
  analytic Jacobian.

The scalar path is unchanged when tensor interactions are disabled. See
[Heisenberg RPA](heisenberg_rpa.md#tensor-interactions) for the model
definitions.

## Closures

Field-free Onsager, SCR, and TAC calculations reuse one Brillouin-zone
eigendecomposition and cache exact closure results across a temperature
series. Field-on closures require numerical energy integration and batched
linear solves, so their cost scales with both Brillouin-zone and energy-grid
sizes.

Use a coarse grid for exploration, then check convergence of fitted and derived
quantities before reporting quantitative results. The closure equations and
grid controls are in [Sum rules and self-consistency](theory_notes.md).

## Current limitations

- Tensor mode uses finite-difference gradients.
- RPA GPU execution does not include eigendecomposition or field-on tensor
  solves. Tight-binding eigendecomposition has a separate explicit CuPy
  backend.
- Dipolar tensor mode cannot use primitive-cell reduction.

Broader directions are listed in [Planned features](planned_features.md).
