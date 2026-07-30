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
budget; an entry larger than its cache budget is used but not retained. The
budgets count distinct NumPy array payloads, not small Python-object overhead.
The prepared-table cache defaults to 128 MiB; viewer, composite, and model
overlay caches each default to 256 MiB.
Caches are performance aids, not stored scientific state: recomputation after
eviction produces the same result.

## Heisenberg RPA

For $M$ fitted momentum points and $N$ magnetic sublattices, the scalar model's
batched eigendecomposition costs approximately $O(MN^3)$. nfit reduces this
cost without changing the result:

- masked data are prepared once per fit;
- datasets with zero fit weight are evaluated only after optimization;
- phase geometry and form factors are cached per dataset;
- centered cells are reduced to translationally distinct magnetic sites when
  the interaction permits it;
- exchange-independent geometry is reused between value and Jacobian calls;
- analytic scalar-model derivatives avoid one model evaluation per parameter;
- gradients stream in bounded point blocks; and
- GUI views, overlays, and analysis fingerprints use bounded caches.

Primitive-cell reduction is often the largest exact saving because the
eigendecomposition cost is cubic in $N$. See
[Heisenberg RPA](heisenberg_rpa.md#primitive-cell-reduction).

For many small matrices, a fused Numba Jacobi eigensolver removes repeated
LAPACK call overhead. Larger matrices use LAPACK, with batch-level threading
only when the workload can amortize dispatch.

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

## Tight-binding electronic structure

Bands, density of states, and constant-energy surfaces share a bounded
eigensystem service. It applies the same float64/complex128 Fourier
Hamiltonian and Hermitian NumPy solver in serial or across independent
wavevector batches. Unprojected calculations use `eigvalsh`, avoiding the
eigenvectors needed only for orbital projections.

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

Immutable electronic models cache their reciprocal lattice, parameter-resolved
real-space blocks, and scientific digest. Updating a named coefficient creates
a new model and therefore a new cache.

For total DOS, `k_mesh(..., symmetry="auto")` can use exact orbit
multiplicities from a GUI-built model's certified reciprocal symmetry.
`symmetry="full"` is the default, while `"reduced"` fails if reduction cannot
be certified. Wannier90 models, reduced-dimensional meshes, incompatible
shifts, and component-level projected DOS retain full sampling in automatic
mode.
Fermi-surface extraction also retains a regular full grid because marching
contours require its topology.

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
