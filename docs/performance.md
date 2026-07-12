# Performance notes

Fitting `heisenberg_rpa` (and future coupled models) evaluates a batched
Hermitian eigendecomposition of $J(\mathbf{Q})$ at every fitted point, so the
cost scales with the number of valid points and the cube of the number of
magnetic sublattices. The package applies several exact optimizations
automatically — none of them changes the fit result or requires configuration.

## What is automatic

- **Prepared-data memoization.** Each dataset's masked, validity-filtered
  points are computed once per fit and reused across every optimizer iteration
  (`FitDataset.prepared_valid`), rather than re-running the transform chain each
  time.
- **Per-dataset geometry cache.** The $\mathbf{Q}$-dependent, exchange-
  independent phase arrays and the magnetic form factor are built once per
  dataset and cached (identity-checked against the data object).
- **Primitive-cell reduction.** Centered lattices are folded onto their
  translationally distinct sublattices before evaluation — exact, and a large
  win for high-symmetry crystals (pyrochlore in the conventional cubic cell:
  16 → 4 sites, ~64× less eigendecomposition work). All inputs and outputs stay
  in the user's specified cell. See
  [Spin-fluctuation models](spin_fluctuation_models.md).
- **Analytic Jacobian.** When every model component on a dataset can supply
  exact gradients (backgrounds and `heisenberg_rpa` do, absent instrument
  resolution), the optimizer uses them instead of finite differences, cutting a
  least-squares iteration from $1 + n_{\text{param}}$ model evaluations to about
  two and improving convergence.
- **Shared eigendecomposition.** The eigendecomposition of $J(\mathbf{Q})$
  depends only on the exchange values, so it is cached on the geometry and the
  back-to-back value and Jacobian evaluations of one least-squares iteration
  reuse a single decomposition instead of computing it twice.
- **Fused small-matrix eigensolver.** For datasets with little
  energy-per-$\mathbf{Q}$ deduplication (2D maps), the cost is dominated by
  decomposing many tiny $J(\mathbf{Q})$ matrices, where LAPACK's per-call
  overhead is the bottleneck. A fused numba Jacobi eigensolver decomposes the
  whole batch in one parallel kernel — ~7× faster than batched LAPACK for
  $4\times4$ matrices, and it scales across all cores. It is used for small
  sublattice counts ($N \le 16$) and large batches; larger matrices use LAPACK.
  Combined with the shared decomposition, this took a 393k-point 2D-map
  iteration from ~1.3 s to ~0.26 s.
- **Work-gated threaded eigendecomposition.** When the LAPACK path is used, the
  batched `eigh` is chunked across a thread pool only when the total work
  $M N^3$ is large enough to amortize the dispatch (roughly $N \ge 8$), with the
  underlying BLAS pinned to one thread to avoid nested oversubscription.
- **Memory-bounded gradients.** The analytic Jacobian streams over points in
  blocks, so its $(\text{block}, N, N)$ temporaries stay within a fixed budget
  regardless of dataset size.
- **Live overlay caching.** In the GUI, the model overlay is evaluated only over
  valid (unmasked) points and its bundles and phase geometry are cached across
  parameter edits, so changing a fitted value re-evaluates in ~0.1 s rather than
  rebuilding from scratch. Rapid edits are debounced.
- **Viewer-view and details caching.** The masked, rebinned viewer view is
  cached per dataset (keyed on data identity, masks, and rebin — not the
  selection), so merely selecting a fit result or "Current state" node, or
  re-selecting a large dataset, no longer re-masks the whole volume. Dataset
  detail counts are computed directly from the mask/event/intensity arrays
  instead of materializing coordinate grids. On the 4D reference dataset (59M
  bins) this takes selecting a fit result from several seconds to ~0.4 s and the
  dataset-details panel from ~5 s to well under 0.1 s.

Together these took the reference pyrochlore project (162k fitted points) from
~1.8 s per model evaluation with per-iteration finite differences to well under
0.3 s per evaluation with exact gradients — roughly an order of magnitude on
end-to-end fit time.

## Compute backends (large datasets)

The per-point resolvent contractions — the dominant cost once the problem has
millions of points — run through a selectable backend:

| backend | when | dependency |
| --- | --- | --- |
| `numpy` | always available; fastest for small problems | — |
| `numba` | large problems on CPU (fused, parallel, streaming kernel) | `pip install nfit[accel]` |
| `cupy` | large problems on an NVIDIA/AMD GPU | `pip install nfit[gpu]` (CuPy wheel matching your CUDA/ROCm) |

`nfit.set_rpa_backend("auto")` (the default, also via the
`NFIT_RPA_BACKEND` environment variable) picks `numpy` below ~50k points so
a 100-point dataset pays no JIT or host↔device overhead, `numba` for larger CPU
problems, and `cupy` for the largest when a GPU is present. `numpy` /
`numba` / `cupy` force a specific backend (falling back to `numpy` if the
requested one is unavailable); `nfit.available_rpa_backends()` reports what
is installed. All backends produce identical results (locked to the numpy path
to floating-point precision by the test suite).

On the reference 4D pyrochlore fit (4.5M valid points, 156k unique Q), the numba
backend cut a least-squares iteration from ~2.8 s (numpy) to ~0.8 s, with the
gradient itself dropping ~4×.

## Threads and many-core / cluster nodes

The parallel worker budget for the RPA kernels (the batched-eigendecomposition
thread pool and the numba kernel) auto-detects the CPUs the process is actually
*allowed* to run on: on Linux that respects cgroup / cpuset / SLURM allocations
via `os.sched_getaffinity`, so a 16-core allocation on a 128-core node uses 16
workers, not 128. Override with `nfit.set_num_threads(n)` or the
`NFIT_NUM_THREADS` environment variable.

The batched eigendecomposition pins the underlying BLAS/LAPACK to a single
thread per call while our thread pool provides the batch-level parallelism —
otherwise a multithreaded BLAS (MKL/OpenBLAS/Accelerate) would nest
`workers × BLAS_threads` threads, which oversubscribes badly on many-core
nodes. (This is what `threadpoolctl` is for; it is a hard dependency.)

One hardware caveat: on Apple silicon, `eigh` runs through Accelerate's
AMX-backed LAPACK, whose throughput is bounded by the shared AMX unit rather
than by core count, so for large magnetic cells ($N \gtrsim 8$ sublattices)
fewer eigh workers can be faster there — set `NFIT_NUM_THREADS` lower if
you hit it. On non-AMX platforms (Linux/Windows with OpenBLAS/MKL) the
pinned-BLAS pool scales the eigendecomposition across all allocated cores. The
primitive-cell-reduced common case ($N \le 6$) does not thread the eigh at all,
so it is unaffected either way.

## Tensor (anisotropic) interactions

Enabling anisotropic exchange, single-ion anisotropy, dipole–dipole, or Zeeman
promotes $J(\mathbf{Q})$ to a $3N\times3N$ matrix
([Spin-fluctuation models](spin_fluctuation_models.md#tensor-anisotropic-interactions)).
Cost notes:

- **Tier A (field off).** One Hermitian $3N\times3N$ eigendecomposition per
  unique $\mathbf{Q}$: roughly $3^3\approx 27\times$ the scalar per-$\mathbf{Q}$
  work at fixed $N$, though $3N\le 16$ (e.g. pyrochlore $N=4\Rightarrow 3N=12$)
  still fits the fused numba Jacobi eigensolver. Structure matrices are stored
  **bond-resolved** (per-bond phase rows + $3\times3$ Cartesian tensors) and
  assembled per iteration, so no $(n_Q, N, N, 3, 3)$ array is held per parameter.
- **Tier B (field on).** A batched LU solve of
  $\mathbb 1 - X_0(\omega)\mathbb{J}(\mathbf{Q})$ per fitted point (chunked at
  200k points), since the gyrotropic $X_0(\omega)$ breaks the eigenbasis reuse.
  Budget a few seconds per evaluation on the 4D job.
- **Dipole cache.** The Ewald tensor is built once per dataset geometry and
  cached densely at shape $(n_Q, N, N, 3, 3)$ complex — ≈360 MB for
  $n_Q=156\text{k}$, $N=4$. It also disables primitive-cell reduction (the Ewald
  sum needs the primitive lattice), so the full cell is used when dipole is on.
- **Gradients.** Tensor mode currently uses central-difference gradients
  ($1+n_{\text{param}}$ evaluations per iteration); the scalar path keeps its
  analytic Jacobian.

The scalar path is untouched when no tensor section is configured, so existing
projects see no change.

## Hardware and libraries

nfit computes through NumPy/SciPy, so it inherits whatever LAPACK/BLAS
those were built against. This is an **environment choice, not a code change**,
and the package stays portable across macOS, Linux, Windows, and clusters with
no configuration:

- On Apple silicon, conda-forge can link Apple's Accelerate
  (`libblas=*=*accelerate`), which uses the AMX coprocessor and is often faster
  for the small dense decompositions here. On other platforms OpenBLAS or MKL
  are the usual defaults.
- The batched eigendecomposition and per-point contractions are isolated behind
  a small set of seams (`nfit.spin_fluctuations._rpa_modes`,
  `_rpa_numba`, `_rpa_cupy`). The GPU backend currently accelerates the
  per-point contractions; keeping the eigendecomposition on the GPU
  (`cupy.linalg.eigh`) to avoid the host↔device transfer is a natural next step.

## Deferred

An analytic Jacobian for the tensor path (the resolvent sandwich generalizes,
but tensor mode currently falls back to finite differences), a fully on-GPU
pipeline (GPU eigendecomposition and the Tier-B batched solve), and
tensor-carrying primitive-cell reduction *with* dipoles (needs the primitive
lattice vectors for the Ewald sum) are left for when they are needed. The
backend seams keep the hot path swappable so those can be added without
disturbing the physics or the portable numpy default.
