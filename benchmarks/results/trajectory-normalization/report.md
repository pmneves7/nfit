# SEQUOIA detector-normalization memory investigation

## Workload and diagnosis

The reported nfit 0.97.0 run on analysis-node23 was integrating detector
trajectories for **YMn2 40 meV sample / Fd-3m symmetry**, not rebinning an
already-gridded MDHisto. The screenshot showed 464,011,821 output bins,
36 workers, 3,467,722,752 trajectories, 254.7 GB process memory, and
56 minutes elapsed at 45.2% of this normalization stage. These screenshots
cannot establish the source of every resident byte or a timing regression.
The earlier large-array MDHisto benchmarks did not exercise this path.

The shared project metadata confirms a 161 × 161 × 221 × 81 grid and 603
sample runs. Shared preferences specify 64 CPUs and 128,000 MiB managed RAM.
A float64 grid occupies 3,712,094,568 bytes. The previous worker planner
assigned essentially the entire RAM allowance to 36 private normalization
grids (133.64 GB), in addition to event grids, the accumulated result,
returned batch grids, and earlier project results. It allocated and reduced
those private grids again at each 32-million-trajectory checkpoint: at least
109 batches for the screenshot workload. The peak-memory estimate omitted
this worker-dependent storage.

SSH reached **analysis-node21**, so this investigation did not inspect the
live node23 process or restart the user's application. Authentication expired
during the temporary source upload, before the real geometry fixture or
full-grid trial could run.

## Implemented changes

- Allocate the HKLE normalization worker buffers once across run batches and
  detector geometries; reduce once at the end. Progress and cancellation
  checkpoints remain between batches. A single worker returns its sole grid
  without another allocation.
- Reserve three event grids and the final normalization grid before selecting
  the HKLE worker count. For the reported settings, this chooses 32 rather
  than 36 workers. The estimate now includes worker-private buffers.
- Normalize the owned signal and variance accumulation arrays in place; reuse
  a boolean temporary for the final mask. Immutable result containers adopt
  these owned arrays without copies. Metadata and the auxiliary normalization
  channel share one read-only denominator.

The finalization changes remove approximately **50 bytes per output bin** of
avoidable simultaneous storage in that stage: **23.20 GB** at the reported
size. One denominator copy, **3.71 GB**, was also retained in each old finished
result. These are allocation-based estimates, not measured whole-project RSS
savings. The normalization and finalization stage savings must not be added
as if they were simultaneous peaks.

The RAM preference remains a managed allocation budget, not a hard process
RSS cap. Earlier project results are additional live storage, and the new
normalization still has one full private grid per worker. Saved-cache memory
mapping does not automatically map newly computed bins. This patch therefore
does not claim to solve the entire 254.7 GB footprint.

## Local measurements

The reproducible synthetic fixture uses the project's UB and coordinate basis,
12,000 synthetic detector directions, 16 synthetic goniometer transforms,
and a 81 × 81 × 111 × 41 grid (29,859,111 bins; 238.87 MB per float64 grid).
Four batches contain 192,000 trajectories in total. Four workers were used.
Compilation was warmed on a small grid before timing. Each timing/RSS trial
ran in a separate local process using the nfit conda interpreter. Peak RSS
includes imports and compilation. GB here means decimal gigabytes.

| Algorithm | Time (s) | Peak process memory (GB) |
| --- | ---: | ---: |
| Previous per-batch private grids | 0.8059 | 2.0504 |
| Persistent private grids (implemented) | 0.3412 | 1.5253 |
| Disjoint first-axis slabs (experimental) | 0.2183 | 0.5323 |

These initial single trials were taken on a laptop during development, with
other validation work potentially active. They demonstrate the allocation
improvement but are not a controlled remote throughput comparison or an
estimate of the full SEQUOIA runtime. All 29,859,111 output bins were compared
separately: both alternatives satisfy `rtol=2e-13, atol=0`; maximum absolute
differences were 1.14e-13 (persistent) and 6.82e-13 (slabs).

The slab prototype assigns workers disjoint regions of one output array and
clips trajectories against each region. It avoids full private grids but
repeats geometry work. Separate small synthetic trials found slowdowns for
geometry-heavy grids, so **production slab dispatch is disabled** pending
real-data testing. Shared slab boundaries require half-open ownership; tests
of stationary trajectories on these boundaries informed the prototype.

An isolated sorting experiment also found a 1.20× improvement for intersection
generation/sorting using Numba's in-place sort instead of insertion sort.
That is not a complete normalization timing; production sorting is unchanged.

## Reproduction and remaining work

Local fixture and per-process trials:

```bash
python benchmarks/benchmark_trajectory.py --synthetic --fixture geometry.npz
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_trajectory.py \
  --fixture geometry.npz --mode baseline --workers 4
# Repeat with --mode persistent and --mode slab.
```

Use the repository's nfit conda interpreter locally. On ORNL, prepare a fixture
from four actual runs spread across the angular scan:

```bash
python benchmarks/benchmark_trajectory.py --prepare /path/to/YMn2_SEQ_IPTS-32969.nfit \
  --fixture geometry.npz
```

Then compare the full 464-million-bin grid using all 192 transforms, with both
one large batch and multiple progress batches (`--transforms 192 --batches 1`
and `--transforms 48 --batches 4`). Sweep CPU allocations, repeat warmed trials,
and check complete arrays or bounded block comparisons. Include smaller grids
and geometry-heavy workloads before selecting a slab crossover. The source
transform in `trajectory_candidates.py` reconstructs the prior fused Numba
allocation/reduction lifecycle from the shared, unchanged geometry kernel.

Remote testing and cleanup remain pending authentication. The task-owned
shared directory is `/SNS/users/paulneves/.cache/nfit-trajectory-7RuHYl`; upload
ended with SSH exit 255 and its contents must be verified and removed after
results are collected. The saved user project and performance preferences
were read only.

## Validation

The full local suite passed: **1,860 passed, 1 optional CuPy test skipped**.
The final focused run passed 60 tests; the last ownership/planning checks
passed four additional targeted tests. Ruff, byte compilation, Sphinx with
warnings treated as errors, and staged diff checks passed. An outdated
Project Explorer menu assertion was updated to include the already-existing
"Rebin stale binnings" shortcut; no menu behavior changed in this patch.
