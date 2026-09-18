# SEQUOIA detector-normalization memory investigation

## Workload and diagnosis

The reported nfit 0.97.0 run on analysis-node23 was integrating detector
trajectories for **YMn2 40 meV sample / Fd-3m symmetry**, rather than rebinning an
already-gridded MDHisto. The screenshot showed 464,011,821 output bins,
36 workers, 3,467,722,752 trajectories, 254.7 GB process memory, and
56 minutes elapsed at 45.2% of this normalization stage. These screenshots
cannot establish the source of every resident byte or a timing regression.
The earlier 10 GB MDHisto benchmarks did not exercise this path.

The shared project metadata confirms a 161 × 161 × 221 × 81 grid and 603
sample runs. Shared preferences specify 64 CPUs and 128,000 MiB managed RAM.
A float64 grid occupies 3,712,094,568 bytes. The previous worker planner
assigned essentially the entire RAM allowance to 36 private normalization
grids (133.64 GB), in addition to event grids, the accumulated result,
returned batch grids, and earlier project results. It allocated and reduced
those private grids again at each 32-million-trajectory checkpoint: at least
109 batches for the screenshot workload. The peak-memory estimate omitted
this worker-dependent storage.

## Implemented changes

In 0.97.4:

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

## Measurements on analysis-node23

Measurements were completed after SSH reconnection on September 18, 2026,
on **analysis-node23.sns.gov**. Each trial ran in a fresh isolated process;
matching kernels were warmed on a tiny grid before timing. Timing excludes
compilation, fixture preparation, output saving and numerical comparisons.
Peak RSS includes imports and compilation, but is recorded before output
saving or comparison with a memory-mapped reference. GB means decimal
gigabytes throughout the result tables.

Fixtures use actual detector directions, normalization weights, proton
charges, run transformations, UB matrix, symmetry and output edges from the
saved SEQUOIA project. They do not load neutron event payloads. Each run has
119,808 detector directions and 48 symmetry transforms. Runs were selected
at evenly spaced indices across the 603-run collection. This is a benchmark
of the expensive normalization stage, **not an end-to-end 603-run rebin**.

The old lifecycle is reconstructed from the unchanged production geometry
kernel with the original fused Numba allocation and reduction. Persistent
and slab candidates use the same geometry rules. Full output arrays were
compared in bounded blocks with `rtol=2e-13, atol=0`; all alternatives passed.
No reduced-precision arrays or approximate integration were introduced.

### Full output grid, representative progress batches

Sixteen real runs produce 92,012,544 trajectories in three batches of
30,670,848, close to the production checkpoint of 32 million trajectories.
The output has all **464,011,821 bins**. These are single trials on a shared
node, so small timing differences should not be treated as precise rankings.

| Algorithm | Workers | Time (s) | Peak process RAM (GB) |
| --- | ---: | ---: | ---: |
| Previous per-batch grids | 36 | 78.52 | 145.12 |
| Persistent grids | 32 | 69.93 | 32.82 |
| Final production path, automatic allocation | 32 | 70.93 | 32.83 |
| Disjoint first-axis slabs, experimental | 36 | 76.42 | 1.70 |
| Disjoint slabs, experimental | 16 | 125.56 | 1.70 |
| Disjoint slabs, experimental | 8 | 227.66 | 1.71 |
| Persistent grids | 8 | 121.42 | 13.50 |

The normal planner's persistent path was about **10–11% faster** and used
**77% less peak resident memory** than the old lifecycle in this fixture.
The largest absolute difference from the baseline was 5.32e-10; every bin
passed the relative tolerance above, including exact agreement on zero bins.

NumPy's zero-filled allocation permits the operating system to defer physical
pages until touched. Numba's old fused allocation eagerly initialized all
worker grids. This accounts for much of the RSS difference beyond eliminating
repeated allocations. More run angles touch more bins: **32.82 GB is not an
upper bound for the full 603-run workload**. Full private-grid capacity is
still included in memory planning. These isolated RSS measurements also
exclude live project results and event histogram arrays.

### Sparser sample and shorter batches

Four real runs, four batches of 48 transforms, produce 23,003,136 trajectories
on the same full grid. This emphasizes repeated-allocation costs more strongly
than production-sized batches and covers fewer angles.

| Algorithm | Workers | Time (s) | Peak process RAM (GB) |
| --- | ---: | ---: | ---: |
| Previous per-batch grids | 36 | 50.43 | 145.12 |
| Persistent grids | 36 | 22.62 | 16.48 |
| Persistent grids | 32 | 22.33 | 15.99 |
| Disjoint slabs, experimental | 36 | 19.35 | 1.56 |

Slabs win here, but lose against persistent grids in the denser sixteen-run
fixture. Output array size alone cannot choose the faster algorithm.

### Smaller grids and allocation overhead

Four real runs were also tested as one batch of 192 transforms, with 32
workers and coarser edges retaining the complete original domain. The old,
persistent and eager-persistent timings below are medians of three trials;
slab timings are single trials. The allocation policy is the only difference
between the two persistent candidates.

| Float64 output grid | Algorithm | Time (s) | Peak process RAM (GB) |
| --- | --- | ---: | ---: |
| 238.87 MB, 29,859,111 bins | Previous per-batch grids | 4.302 | 8.235 |
| 238.87 MB | Deferred persistent grids | 4.632 | 1.828 |
| 238.87 MB | Eager persistent grids | 4.249 | 8.213 |
| 238.87 MB | Slabs | 9.665 | 0.395 |
| 15.81 MB, 1,976,856 bins | Previous per-batch grids | 1.482 | 0.875 |
| 15.81 MB | Deferred persistent grids | 1.513 | 0.475 |
| 15.81 MB | Eager persistent grids | 1.396 | 0.851 |
| 15.81 MB | Slabs | 5.311 | 0.309 |

Small-grid old/persistent/eager results were bitwise identical. Deferred
allocation has a repeatable roughly 0.33-second overhead on the 239 MB grid;
the smaller 16 MB difference overlaps trial variability. Eager initialization
recovers small-grid speed, at the cost of making the private buffers resident.

## Dispatch decision

For 0.97.5, small persistent worker buffers are eagerly initialized only when
their **combined** size is at most **8 GiB and one eighth of the central
managed RAM allowance**. Larger workspaces keep deferred NumPy allocation.
This preserves the faster small-grid path when there is ample RAM, without
pre-touching the enormous full-grid worker workspace. No additional preference
is exposed. The cutoff is a conservative allocation bound, not a claim of a
universal timing crossover; output size alone misses multiplication by the
worker count and the available RAM budget.

A final fresh-process check exercised the production accumulator and its real
allocation policy: **4.122 s / 8.213 GB** for the 239 MB grid and
**1.417 s / 0.853 GB** for the 16 MB grid, both selecting eager allocation.
The full 464-million-bin grid selected deferred allocation and took
**70.930 s / 32.830 GB**. All output bins passed comparison with the old
baseline; the smaller results were identical. The code hashes are recorded in
`node23/production-source.json`. The benchmark driver subsequently gained a
cold-start option and worker-clamped allocation reporting; production code
and the candidate helper retain those recorded hashes.

Two additional fresh-process checks included first-call compilation on the
16 MB grid: the old path took **5.67 and 5.61 s**, while production took
**4.59 and 4.31 s**. Both production outputs were identical to the baseline.
Thus the extra allocator dispatcher did not create a cold-start regression
against the old lifecycle in this test. Import/startup time remains outside
these timers. Use `--cold` to repeat this check.

**Slab dispatch remains disabled.** Slabs save much more memory but were
2–3.5 times slower on smaller grids and about 9% slower than persistent
32-worker accumulation in the representative full-grid fixture. Reducing slab
workers made the latter markedly slower. They may merit a future explicit
memory-pressure strategy: 36 slabs beat an eight-worker dense calculation,
but safely selecting that tradeoff needs more workloads, boundary coverage
and production-quality cancellation. There is no evidence for automatically
switching every array above a few gigabytes to slabs.

## Reproduction

Use the nfit conda interpreter locally. The remote environment and fixture
array hashes are recorded in [node23/environment.json](node23/environment.json).
Raw per-trial measurements are in the adjacent `node23/*-trials.log` files.
Production geometry during the initial trials matched commit `1457c8a`.
The node has 64 logical CPUs; measurements used Python 3.14.7, NumPy 2.5.3
and Numba 0.67.0. OpenBLAS used one thread, Numba allowed up to 36. Other
users were active on the node; these are not idle-host throughput guarantees.

```bash
python benchmarks/benchmark_trajectory.py --prepare /path/to/YMn2_SEQ_IPTS-32969.nfit \
  --fixture geometry16.npz --runs 16
NUMBA_NUM_THREADS=36 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_trajectory.py \
  --fixture geometry16.npz --mode baseline --workers 36 --transforms 256 --batches 3 \
  --save-result baseline.npy
NUMBA_NUM_THREADS=36 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_trajectory.py \
  --fixture geometry16.npz --mode production --workers 32 --transforms 256 --batches 3 \
  --reference baseline.npy
```

Prepare four runs for the small-grid comparisons, then use `--transforms 192
--batches 1 --stride 2` or `--stride 4`. Candidate modes `persistent`, `eager`
and `slab` isolate the allocation strategies from automatic production
selection. `--synthetic` reproduces the earlier laptop-only fixture; its
measurements should not be mixed with these actual-geometry remote results.

## Validation and cleanup

The 0.97.5 full local suite passed **1,862 tests**, with one optional CuPy skip.
Ruff, byte compilation, Sphinx with warnings treated as errors, and staged
diff checks passed. All four version declarations agree. Scientific geometry,
sorting, project schemas and installed application files are unchanged by
this follow-up.

The earlier `/tmp/nfit-mapped-artifacts-Cz62Ea` directory was found on node23,
verified against the saved 10 GB archive hash, removed and verified absent.
After collecting all **37 measurements**, including **33 full-array comparison
checks**, the task-owned source/fixture/result directory
`/SNS/users/paulneves/.cache/nfit-trajectory-7RuHYl` was removed. All seven runner
completion markers reported success, no process had its working directory
inside the scratch root, and both remote scratch paths were verified absent
on node23 at **2026-09-18 15:40:38 UTC**. The local upload tarballs were also
removed. Pre-existing dependency caches were retained. User project files and
performance preferences were read only; no running nfit process was stopped
or installed application updated.
