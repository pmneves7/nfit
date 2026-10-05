# DGS serial reduction and event-binning audit

## Scope and observed workload

Read-only process sampling on analysis-node23, October 5, 2026, and an audit of
the native SEQUOIA/HYSPEC execution paths. No parallel numerical implementation
was installed, no active job was interrupted, and no project data was changed.
The GUI workflow is the large NiO/Al2O3 project with multiple named binnings.
It is not an isolated benchmark: other work was running on the node and
filesystem caches were not flushed. CPU percentages below use one core = 100%.

| Running build and phase | CPU observed | Application reads | Application writes |
| --- | ---: | ---: | ---: |
| 0.114.5, cached-event accumulation, six two-second samples | 96.5–99.9% | about 48–87 MiB/s | not measured in this interval |
| 0.116.17, fresh raw-event processing, five two-second samples | 91.4–96.4% | 94.3–155.4 MiB/s | 70.6–106.1 MiB/s |

The older process also reached approximately 15–20 core equivalents in a
different interval. The active cached-event NPZ inspected through its open file
descriptor used ZIP_STORED, with 38,020,048 event bytes and no compression.
Fresh processing had both raw NeXus and newly written reduced-event files open.
Linux reported zero accounted physical read bytes in the sampled intervals.
That counter is not a complete measurement of GPFS server traffic; the stronger
evidence against dominant I/O waiting is the almost continuously occupied core.
These observations do not distinguish array copying, reconstruction, archive
checksums, and histogram arithmetic from one another.

The GUI's machine-normalized 1.5–1.6% corresponds to one core out of 64.
The progress counters count examined raw events, including discarded events,
and cannot be converted into accepted-event kernel throughput.

## Execution map and constraints

| Stage | Present implementation | Parallelism assessment |
| --- | --- | --- |
| Run inspection and geometry | Per-run preparation; immutable resolved geometry reused only for identical complete definitions | Independent runs can be prepared concurrently. Python XML traversal and shared caches need separate consideration from native array arithmetic. HYSPEC tank position and instrument identity must remain run-specific. |
| Raw event reading and reconstruction | Sequential banks/chunks in `raw_dgs._iter_reduced_event_chunks`; vectorized detector lookup, pulse filtering, TOF conversion, efficiency and kinematic corrections | Reconstruction has independent event work and no scientific requirement to be serial. Reader/compute pipelining or bounded independent run producers are candidates. Preserve each floating-point stage and correction order. |
| Reduced-event cache construction | One writer per run, bounded coalesced uncompressed column blocks | Independent run caches can be produced separately. Never share the writer or reusable buffer between workers; publish only complete validated caches. |
| Cached-event reading, rotation and scaling | One run/block at a time; then symmetry copies in order | Prefetching can overlap block loading with compiled work, but introduces another live block and may not help a mostly CPU-bound job. Rotation/scale work can be distributed with bounded inputs. |
| Event projection and histogram updates | Fused Numba loops with `nogil=True`, ordinary `range`, and ordered updates to shared C/V/count arrays | The kernel releases the Python GIL but does not create workers. Concurrent shared-bin writes are unsafe. Independent output ownership or private partial sums are necessary. |
| Detector-trajectory exposure | Compiled parallel accumulator with memory-aware worker selection and geometry-checked task pooling | Already parallel; it explains the observed CPU bursts. Its exposure array and resource ownership remain independent from event parallelism. |

Here C is the weighted event numerator, V its variance numerator, and N the
trajectory exposure. Final intensity is C/N and its stored standard uncertainty
is sqrt(V)/N. C and V have the corresponding event-weight units and squared
units; N carries the configured calibration/exposure convention. Counts are
dimensionless. Parallelism must preserve all these channels, not just intensity.

[h5py documents](https://docs.h5py.org/en/stable/threads.html) a process-wide
lock around HDF5 calls, including calls against different files. Threads can
overlap HDF5 reading with non-HDF5 computation, but cannot make simultaneous
h5py reads execute concurrently. Independent processes avoid that particular
lock; they add startup, geometry duplication, communication, and filesystem
contention. A process design must use safe spawning in a Qt application and
avoid transferring large arrays through unrestricted pickle queues.

## Why directly adding threads is insufficient

Two events or symmetry copies may target the same bin. Concurrent `+=` updates
to C, V and counts would race, lose contributions, and corrupt uncertainties.
Atomic updates would change floating-point accumulation order and still face
contention around strong elastic peaks. They are not a drop-in parity path.

This project's 91,584,578-cell example requires about 2.05 GiB for just three
float64 accumulator arrays per worker. At 64 workers that is about 131 GiB,
before exposure, finalized channels, existing viewers, inputs, and scratch.
Scanning and merging all those mostly empty grids can cost more than the
original event pass. The generic point-rebin private-accumulator strategy
therefore cannot simply be applied to every DGS cube.

Disjoint flat-bin ownership can avoid both races and complete private grids.
To retain literal sums, events must be stably partitioned and consumed in their
original event, symmetry-copy and chunk order for each bin. A dominant bin still
belongs to one worker; the partitioning itself and extra array passes cost time.
With optional same-event covariance, copy assignments and variance correction
must retain their established relationships and run exactly once.

## Existing experiments and recommendation

The [earlier ownership experiment](dgs-remaining-performance.md#real-cached-event-ownership-tests)
already tested this approach on all 617 cached SEQUOIA sources: 72.780 to
58.317 s for the twelve-copy event interval, with literal C/V/counts and event
support. That saved about 14.5 s, or roughly 5% of the then measured complete
saved-project workflow. The extra projection/partition paths were deferred
because that whole-workflow gain did not justify their complexity. Single-core
utilization alone is not new evidence that this tradeoff has changed.

The existing 24-run instrumented raw profile reports 2,784 raw iterator resumes
and 16,728 projection dispatch calls. That identifies bank-sized delivery as a
candidate, not a measured current-build speedup; cumulative profile components
overlap and must not be summed. See the
[fresh-reduction audit](dgs-remaining-performance.md#remaining-fresh-reduction-audit).

Recommended next experiment:

1. Use a separate bounded real-run pilot after the active GUI job finishes.
   Record non-overlapping time in inspection/geometry, HDF reads, reconstruction,
   cache writes, rotation/scaling, projection/scatter, normalization, and saving.
   Compare complete initial and cached-rebin workflows on the same hardware;
   retain worker counts, module hashes and filesystem-cache conditions.
2. First test the previously deferred bounded coalesced delivery of fresh
   detector-bank chunks. It could reduce allocations and dispatch overhead
   while retaining one ordered consumer and existing cache storage.
3. If reconstruction remains dominant, test two and four independent producers
   with ordered consumption. Restrict total in-flight bytes and cache files,
   rather than selecting workers from CPU count alone. Use threads only where
   native compute overlap pays; test processes separately if HDF reads dominate.
4. Revisit cached-event bin ownership only if current complete-workflow timings
   establish enough remaining event cost to make a substantial total saving.

Every candidate needs identical event membership/counts/edges, C/V agreement
with an explicitly stated rounding tolerance, exact exposure support, fringe
signal/error comparisons, and unchanged masks and reduction signatures. Include
both precision modes, changed detector geometry, optional covariance, empty
chunks, cancellation, failure cleanup, and cache publication. Worker code must
not call GUI widgets or concurrently mutate dataset metadata. Keep scientific
staging and outputs in the experiment IPTS directory.

Conclusion: raw reconstruction can safely be parallelized with an appropriate
producer/consumer design. Event accumulation is serial for valid race and
memory reasons, but can be partitioned safely. Neither finding establishes a
worthwhile whole-workflow speedup without the proposed current-build pilot.
