# Performance notes

nfit selects conservative CPU and memory strategies automatically. Most users
only need to limit grid sizes and install an optional acceleration backend for
large jobs.

## Rebinning and event reduction

Small rebinning jobs use NumPy. Large jobs can use a fused Numba kernel with
memory-bounded private accumulators. Extremely sparse output grids may use
touched-bin maps when their estimated occupancy is at most 5%; otherwise nfit
falls back to a serial fused kernel if threaded copies exceed the memory
budget.

`rebin_nd_stream` and MDEvent reduction process source data in bounded batches,
so source files may exceed RAM. Explicit output limits permit one pass;
automatic limits require a discovery pass. The batch target bounds temporary
event storage but cannot reduce the persistent output arrays.

File-backed datasets are loaded on first use and retained by their
`DatasetEntry`; viewers, analyses, and composites share that single loading
path. nfit does not keep a second raw-array cache.

Before a 4D MDEvent reduction, the GUI estimates peak memory from the output
bin counts and normalization arrays. It warns above 70% of available RAM. The
lower-level API rejects an over-budget allocation unless the caller explicitly
disables enforcement.

Use `benchmarks/benchmark_rebin.py` to measure representative grids on the
target machine.

## Performance preferences and benchmarks

**File → Preferences → Performance** provides a default rebin batch target in
MiB (1 MiB = 1,048,576 bytes) and a worker ceiling. **Auto** uses 192 MiB and
the nfit CPU allocation, respectively. Choose **Save defaults** to persist
changes. These values are copied into newly initialized dataset/composite rebin
configurations; existing values, including independent saved-plot recipes, are
not overwritten. These are rebin defaults, not controls for fitting or
electronic-response parallelism. The CPU setting is a ceiling, not a promise
that all workers will be used.

Preferences are machine-local in `~/.config/nfit/performance.json`; set
`NFIT_PERFORMANCE_FILE` to use another file. The headless equivalents are
`load_performance_settings()` and `save_performance_settings(max_batch_mb=64,
workers=4)` in `nfit.performance`. Zero selects Auto. Explicit saved rebin
values remain part of project and plot configurations and workflow scripts.

Use **Calibrate this machine…** for representative 3-D hard-binning and 4-D
fractional-binning workloads. Use **Benchmark this rebin…** in dataset or
composite rebin controls for the full selected configuration, including axes,
symmetry, masks, and output-grid settings. Candidate ranges scale to the CPU
allocation and currently available memory detected for the process. Small
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
within 5% of the fastest. **Apply recommendation** is required: calibration
updates machine defaults, whereas a real-data benchmark updates only the
selected rebin configuration. As with other edits, Auto rebin may then refresh
that configuration's view.

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

Cache entries are process-local and evicted by least-recent use. Each GUI data
cache has both an entry-count limit and an estimated numerical-array memory
budget. Completed viewer and composite binnings first use resident arrays,
then a bounded, chunk-compressed RAM tier. nfit does not write automatic cache
spill files to `/tmp` or scratch. When the compressed tier fills, the GUI asks
whether to discard the oldest compressed binning or save it as a compressed
`.npz` file at a path you choose. A discarded binning can be recomputed from
its sources. Saving a compressed NPZ exports the numerical result; it does not
add that file to the project automatically. A binning too large for the
compressed tier is not retained there. Scripting workflows evict old results
without a GUI prompt. The budgets count distinct NumPy array payloads and
compressed bytes, with small Python-object overhead excluded.
The prepared-table cache defaults to 128 MiB and the model-overlay cache to
256 MiB. Viewer and composite cache capacities use one sixteenth of physical
memory, bounded between 768 MiB and 4 GiB for each cache. Three quarters of
each allowance holds resident arrays and one quarter holds compressed results.
This capacity is not allocated in
advance: small projects retain only the arrays they produce. The adaptive
budget can retain a practical four-dimensional reduction that exceeded the old
256 MiB limit, avoiding immediate eviction and duplicate work while preparing
derived datasets or reopening a viewer.
Caches also have defensive entry-count limits. The composite cache accepts up
to 64 entries within its memory budget so projects with several dataset-group
composites do not repeatedly evict and rebuild one another while reopening a
viewer.
Parent composites also retain child reductions on matching grids; saving the
children later reuses those results. Changed source data, masks, backgrounds,
or numerical bin settings invalidate the corresponding cached results.

Single-crystal MDEvent detector normalization groups runs with identical
detector geometry and evaluates all requested symmetry operations in the
compiled trajectory kernel. Each trajectory is clipped to the requested HKLE
box before internal bin boundaries are examined. These optimizations change
neither the event histogram nor the integrated normalization denominator; the
Python reference path is retained for numerical-equivalence testing. The saved
per-rebin **Workers** value is a ceiling: nfit uses it on a workstation or
cluster node but automatically lowers the normalization worker count when the
thread-private output accumulators would exceed one quarter of currently
available memory. Large-memory nodes can therefore use multiple CPUs for large
four-dimensional grids, while memory-constrained machines still fall back to a
smaller worker count. **Benchmark this rebin…** measures the
complete saved configuration on the current machine; its exported script can
test a custom candidate list when a cluster node warrants a broader sweep. The
default benchmark sweep includes the machine's full detected CPU allowance in
addition to conservative smaller ceilings.

**Preferences → Performance → Transient memory ceiling** controls that share
of currently available memory. The automatic value is 25%, and an explicit
value may range up to 80%. The same ceiling limits thread-private rebin
accumulators and clamps each saved per-rebin batch target, so increasing a
batch target cannot silently exceed the machine policy. The setting applies to
existing projects because it describes the current machine rather than project
state. The separate MDEvent output-grid preflight continues to warn when the
estimated complete reduction exceeds 50% of currently available memory. Other
dataset and composite rebins use an estimate of their output-array workspace
plus the batch target and warn at the same threshold. The native reducer still
blocks an unapproved estimate above 70%. Opening a data viewer also warns
before starting pending large rebins. A memory estimate is advisory: source
data, other caches, and the operating system can change the actual peak.
On macOS, available memory includes inactive and speculative pages that the
operating system can reclaim; this prevents the worker planner from falling to
one CPU merely because the file cache has consumed most completely free pages.

Process caches normally disappear when nfit exits. For projects whose raw data
are expensive to load or rebin, enable **File → Cache binnings** before saving.
The project then embeds current dataset and composite binnings and restores them
on the next open. Saving does not recompute current entries; it updates only
missing or signature-stale binnings. This project-specific option defaults off
so ordinary project files remain small. Embedded caches carry a numerical format
version and a signature of the source, rebin settings, masks, and backgrounds.
nfit reuses compatible cache formats and discards a cache when its saved
signature no longer matches the live recipe. Incompatible older formats are
recomputed from their source data.
Source-backed cache signatures use the source path, size, and modification time,
so lazily loading an unchanged source after opening a project does not invalidate
its restored binning. Editing one named binning invalidates only that result;
other fit or visualization binnings remain available when their own inputs and
settings are unchanged. When an operation processes multiple named binnings,
the progress dialog reports how many binnings are complete and identifies the
current binning in the existing status area.

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

For sufficiently large calculations with several distinct transferred
wavevectors, nfit partitions the total `response_workers` allocation across
the independent q groups and assigns the remaining workers within each group.
The base mesh eigensystem is shared rather than recomputed. Fourier Hamiltonian
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
allocation. Override the detected worker count with `NFIT_NUM_THREADS` or
`nfit.set_num_threads(n)`.

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
