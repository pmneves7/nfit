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
- **Work-gated threaded eigendecomposition.** The batched `eigh` over the
  unique-$\mathbf{Q}$ grid is chunked across a thread pool only when the total
  work $M N^3$ is large enough to amortize the dispatch (roughly $N \ge 8$).
  NumPy releases the GIL per small decomposition, so this parallelizes cleanly;
  for the tiny matrices left after primitive-cell reduction it correctly stays
  single-threaded.
- **Memory-bounded gradients.** The analytic Jacobian streams over points in
  blocks, so its $(\text{block}, N, N)$ temporaries stay within a fixed budget
  regardless of dataset size.
- **Live overlay caching.** In the GUI, the model overlay is evaluated only over
  valid (unmasked) points and its bundles and phase geometry are cached across
  parameter edits, so changing a fitted value re-evaluates in ~0.1 s rather than
  rebuilding from scratch. Rapid edits are debounced.

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
| `numba` | large problems on CPU (fused, parallel, streaming kernel) | `pip install metallix[accel]` |
| `cupy` | large problems on an NVIDIA/AMD GPU | `pip install metallix[gpu]` (CuPy wheel matching your CUDA/ROCm) |

`metallix.set_rpa_backend("auto")` (the default, also via the
`METALLIX_RPA_BACKEND` environment variable) picks `numpy` below ~50k points so
a 100-point dataset pays no JIT or host↔device overhead, `numba` for larger CPU
problems, and `cupy` for the largest when a GPU is present. `numpy` /
`numba` / `cupy` force a specific backend (falling back to `numpy` if the
requested one is unavailable); `metallix.available_rpa_backends()` reports what
is installed. All backends produce identical results (locked to the numpy path
to floating-point precision by the test suite).

On the reference 4D pyrochlore fit (4.5M valid points, 156k unique Q), the numba
backend cut a least-squares iteration from ~2.8 s (numpy) to ~0.8 s, with the
gradient itself dropping ~4×.

## Hardware and libraries

metallix computes through NumPy/SciPy, so it inherits whatever LAPACK/BLAS
those were built against. This is an **environment choice, not a code change**,
and the package stays portable across macOS, Linux, Windows, and clusters with
no configuration:

- On Apple silicon, conda-forge can link Apple's Accelerate
  (`libblas=*=*accelerate`), which uses the AMX coprocessor and is often faster
  for the small dense decompositions here. On other platforms OpenBLAS or MKL
  are the usual defaults.
- The batched eigendecomposition and per-point contractions are isolated behind
  a small set of seams (`metallix.spin_fluctuations._rpa_modes`,
  `_rpa_numba`, `_rpa_cupy`). The GPU backend currently accelerates the
  per-point contractions; keeping the eigendecomposition on the GPU
  (`cupy.linalg.eigh`) to avoid the host↔device transfer is a natural next step.

## Deferred

A batched-LU resolvent kernel (needed once single-ion anisotropy makes the
local propagator a per-site tensor and the eigenbasis of $J$ no longer
diagonalizes the RPA denominator) and a fully on-GPU pipeline (GPU
eigendecomposition) are left for when they are needed. The backend seams keep
the hot path swappable so those can be added without disturbing the physics or
the portable numpy default.
