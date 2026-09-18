# Large histogram pipeline and storage measurements

These follow-up measurements compare the direct and streamed MDHisto paths
using the same numerical implementation and input values. They supplement the
[real SEQUOIA measurements](memory-report.md), which measured a 1.62 GiB crop
from the saved project. The large fixtures here are generated test data, not a
measurement of the complete SEQUOIA application session.

## Method

Trials ran on the ORNL analysis cluster. The initial connection reported
`analysis-node23.sns.gov`, but the load-balanced SSH alias can route later
connections elsewhere. The original main/tiny runners did not record a
per-process hostname, so that node attribution is not guaranteed. Each runner
executes its paired trials locally on the same node; their recorded platform,
CPU count, and library versions are preserved in the raw results. Rebin trials use four CPUs and an isolated
16 GiB managed RAM preference, without changing the user's preferences or
project. Each path runs in a fresh process and warms its own numerical kernels
before timing. Source construction, hashing, and result serialization are
outside rebin timing. Process peak RSS includes imports, warmup, source data,
and the operation. The direct path can exceed the managed RAM setting because
its full coordinate meshes are not bounded working batches.

The largest fixture has shape `(256, 192, 128, 40)`: 251,658,240 source bins,
five float64 grids and one boolean mask, totaling **10,317,987,840 bytes**
(10.318 GB, 9.609 GiB). Sparse and dense cases contain approximately 15% and
70% valid signal bins. Both retain the full dense array payload. Output axes
are coarsened approximately twofold per dimension using the existing center
and automatic-limit conventions. Fractional and inversion-symmetry cases
exercise the more expensive contribution and coverage calculations.

Small tiers are repeated three times. Large tiers are single trials: they show
the measured effect on this machine, not a universal hardware crossover.
Floating-point reductions may differ in summation order. Saved large outputs
are compared with `rtol=atol=1e-12`, including masks, axes, events, errors,
coverage, and normalization; source digests must match.

## Rebin results

| 10.318 GB operation | Direct time | Streamed time | Direct peak RSS | Streamed peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Regular, 15% valid | 29.03 s | 18.60 s | 45.93 GB | 12.71 GB |
| Regular, 70% valid | 58.97 s | 33.71 s | 58.11 GB | 12.75 GB |
| Fractional, 15% valid | 82.43 s | 21.21 s | 45.93 GB | 13.31 GB |
| Fractional with P-1 symmetry, 15% valid | 161.40 s | 122.19 s | 67.08 GB | 14.85 GB |

For sparse regular rebins, measured direct/streamed times were 0.0367/0.0202 s
at 10.7 MB, 0.246/0.160 s at 86 MB, 1.444/0.897 s at 516 MB,
2.860/1.830 s at 1.032 GB, and 8.665/5.407 s at 3.095 GB.

The smaller regular-rebin sweep brackets the crossover:

| Source payload | Valid bins | Direct median | Streamed median |
| --- | ---: | ---: | ---: |
| 0.105 MB | 15% | 4.788 ms | 5.306 ms |
| 0.105 MB | 70% | 4.822 ms | 5.638 ms |
| 1.343 MB | 15% | 7.034 ms | 7.637 ms |
| 1.343 MB | 70% | 9.971 ms | 8.178 ms |
| 5.374 MB | 15% | 18.097 ms | 11.968 ms |
| 5.374 MB | 70% | 32.432 ms | 18.283 ms |

A further three-repeat confirmation at 131,072 bins (5.374 MB), with each
process recording `analysis-node21.sns.gov`, found:

| Operation | Valid bins | Direct median | Streamed median |
| --- | ---: | ---: | ---: |
| Fractional | 15% | 31.667 ms | 14.857 ms |
| Fractional | 70% | 92.985 ms | 22.617 ms |
| Fractional with P-1 symmetry | 15% | 68.869 ms | 69.217 ms |
| Fractional with P-1 symmetry | 70% | 118.385 ms | 105.508 ms |

Sparse symmetry is effectively tied at this smaller size, so the switch is
not lowered to 131,072 bins merely on the regular-rebin timing advantage.

All four 10.318 GB output pairs pass the strict numerical comparison. The
largest observed absolute difference is approximately `1.42e-14` in symmetry
coverage. Symmetry raw summary records retain `NaN` for the coverage sum where
masked output bins contain NaNs; their full coverage arrays were still checked.

The existing **250,000-source-bin** threshold is retained. For the tested
payload that is approximately 10 MB, well below several GB. Coordinate
generation costs scale with bin count, so optional extra channels should not
determine the algorithm switch. Small histograms and discrete/tolerance
clustering keep their previous direct path.

## Storage policy

Final parallel-reader measurements ran on `analysis-node21.sns.gov`, with
exact source/array hash agreement and an unchanged source archive. The 10.318
GB generated payload compressed to approximately 849 MB. Resident and mapped
readers used the same four-CPU limit. These are post-write, OS-cache-dependent
measurements; no global page-cache flush was performed.

| Expanded payload | Resident open | Mapped open | Resident RSS after open | Mapped RSS after open | Resident warmed signal scan | Mapped warmed signal scan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.032 GB | 0.531 s | 0.762 s | 1.211 GB | 0.208 GB | 0.08775 s | 0.08880 s |
| 10.318 GB | 5.247 s | 7.098 s | 10.497 GB | 0.177 GB | 0.89586 s | 0.90317 s |

The warmed scan is the median of three `np.nansum(signal)` calls. On the large
case the first scan took 0.895 s resident and 0.983 s mapped, about 10% slower
from first-touch page faults. Warmed scans differed by 0.8%. Initial mapped
opening cost an additional 1.85 seconds (35%) while anonymous memory immediately
after open fell from 10.445 GB to 0.124 GB. These opening RSS values are not
the eventual high-water mark: touching all channels brings file-backed pages
into RSS, and NumPy reductions can allocate working copies. Mapped pages can
subsequently be reclaimed without discarding the array object.

An earlier 1 GB sequential-inflation trial took 1.674 s versus 0.523 s resident,
which was too slow to enable broadly. The final reader therefore decompresses
large members concurrently under the existing central CPU/RAM limits. Tests
cover cleanup even when another worker finishes after the first failure.

Saved histogram caches can use temporary read-only NumPy memory maps on macOS
and Linux. Automatic selection requires expanded artifact size of at least
2 GiB and at least one quarter of the central managed RAM allowance. Thus a
16 GiB allowance selects mapping from approximately 4 GiB. This is a memory
pressure policy, distinct from the much smaller rebin crossover.

The mapping reader preserves ordinary NumPy arrays and float64 precision. It
expands compressed members into private temporary files and unlinks them once
mapped. Array views retain the backing storage until their last reference is
released. Temporary storage failure falls back to the existing resident reader;
corruption is not retried. The source project is never modified.

Mapping removes the permanent heap allocation for expanded cached payloads.
Hot mapped pages still count toward RSS, and the operating system may retain
them until it needs memory. It can reclaim disk-backed pages; later access can
then incur disk I/O. The measurements do not claim cold-cache performance or
zero paging cost. Newly computed results remain in NumPy RAM arrays.

There is no new library dependency or performance preference. In-memory
compression is not used for active arithmetic: the earlier real-data storage
comparison found ordinary NumPy substantially faster for repeated full-array
access. The existing compressed cache tier remains available for inactive data.

## Reproduction

`benchmarks/benchmark_rebin_crossover.py` accepts `--shape`, `--path dense|stream`,
`--operation regular|fractional|symmetry`, `--occupancy sparse|dense`, and
`--workers`. `benchmarks/benchmark_mapped_artifacts.py` separates fixture
preparation from fresh-process resident/mapped loading and records array hashes,
per-operation times, anonymous memory, file-backed RSS, and peak RSS.

Benchmark source copies and generated data on ORNL are temporary; raw JSON
measurements and provenance are retained in this repository.

Validation: a fixed source snapshot of `46daf6d` plus the memory changes passed
1,836 tests, with one optional CuPy test skipped. Focused mapping/cache,
packaging, and numerical-boundary tests also passed, as did Ruff,
byte-compilation, and the Sphinx warning-as-error build.

Final shared rebin and storage scratch was removed and its absence verified.
The earlier node-local directory `/tmp/nfit-mapped-artifacts-Cz62Ea` was found
on node23 after reconnection on 2026-09-18. Its matching fixture provenance was
checked, the directory was removed, and its absence verified. See the storage
cleanup audit for the original limitation and the follow-up evidence.
