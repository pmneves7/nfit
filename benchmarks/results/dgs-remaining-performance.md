# Remaining DGS performance opportunities

This code audit follows the accepted event fusion, geometry/inspection reuse and
parallel histogram loading changes. Timings below are the unprofiled 0.116.1
full 617-run SEQUOIA workflow on node19. The current angle-corrected build and
accepted trajectory-pooling build have separate acceptance receipts. Pooling
reduces full617 normalization intervals by 31–32% and the complete saved
workflow by 9.96%, while preserving the strict every-cell numerical gate. The
follow-up measurements below distinguish accepted reuse from rejected prototypes
and remaining profiling candidates.

## Candidates worth profiling

| Priority | Candidate | Evidence and acceptance requirements |
| --- | --- | --- |
| 1 | Bounded concurrent raw-run reduction, ordered consumption | The 283 s event interval mixes raw reads, reconstruction, reduced-cache writes and scatter. Profile those components first. Independent producers must keep each run's geometry and consume original run/chunk order, preserving literal reduced-event payloads and C/V/counts, support, bounded memory and cancellation cleanup. |
| 2 | Disjoint groups of flat bins for ordered event accumulation | Cached twelve-copy events take about 82 s. Stable partitioning by actual projected bins gives each output bin one worker without atomics or reordered additions. Flat blocks can spread elastic-heavy events spatially; energy-only partitions concentrate them in one worker. Measure partitioning overhead, memory bandwidth and false sharing; require literal C/V/counts and identical membership. |
| 3 | Independent lazy exposure reuse | Efficiency corrections can change event C/V while leaving grid, geometry, charge, energy window, UB and symmetry unchanged. A complete exposure signature could avoid N recalculation for those edits. New grids, masks, weights, energies, geometry and windows must invalidate it. Existing scale/background reuse remains separate. |
| 4 | Trajectory memory locality | Dense worker grids receive scattered writes. Pursue tiling only if profiling establishes bandwidth or merging as the dominant cost. Changes in addition order still need the full exposure-support and every-cell tolerance gates. |

Raw producers and event-bin ownership remain prototypes to investigate, rather
than promised multipliers. HDF library serialization and shared-filesystem
bandwidth may limit concurrent reads. Event partitioning may cost more than it
saves. Worker count alone does not establish usable parallelism.

## Proposals already rejected or substantially addressed

- Cached reduced-event blocks are already uncompressed and streamed once per
  run in bounded chunks. Another decompression cache is not justified.
- Large saved histograms already decode with independent positional readers and
  parallel workers, retaining their validated original descriptor and checksums.
  Actual full-workflow access improved from 39.7 to 7.9 s; immutable containers
  adopt the decoded arrays without another full copy.
- Unchanged histogram members save by compressed-member copying. Changed arrays
  already compress in parallel. About 36 s saving within a 504 s initial
  workflow does not justify an archive redesign by itself.
- Finalization takes about 5–7 s. Even eliminating it cannot provide a major
  whole-workflow improvement.
- Successful scalar inspection and resolved geometry arithmetic are already
  reused. Fresh 617-source inspection takes about 78 s; investigate it with raw
  production before adding another persistent metadata cache.
- Clipped crossing scans, loop rearrangement, invariant hoisting and eager
  allocation prototypes were slower, approximately 0.88–0.98 times baseline
  throughput. They were not adopted.
- A synthetic locality diagnostic ran about 1.8 times faster by deliberately
  changing spatial assignments. It failed numerical support/parity and is not
  an admissible optimization; it only motivates measuring memory cost.

The guarded duplicate-trajectory pooling gate is complete. The next useful
profile should separate raw reconstruction, cache writing, reads and event
scatter so that a new implementation addresses a measured limiting cost.
No production algorithm changes follow from this audit alone.

## Current worker policy and next event prototype

The real full617 normalization sweep favors the existing 64-worker policy:
median 127.526 s at 64 versus 142.320 s at 32. Every exposure cell/support passes
and memory stays bounded. There is no evidence for reducing the worker ceiling
on this host. These are normalization-only trials, not project-workflow times.

A bounded local synthetic prototype uses the authoritative projection followed
by stable ownership partitioning and original-order scatter within each bin.
With 600,000 events × six copies, eight workers and either 64 or 101 energy bins,
4096-bin flat blocks improve kernel throughput about 3.0–3.7× for spatially
spread ordinary/elastic-heavy data, with literal C/V/counts in alternating trials.
Partitioning and intermediate allocations are included. A single dominant Q/E
cell remains serial and improves only about 1.5–1.7×. Eight-bin modulo ownership
can accidentally collapse to energy bands when the energy dimension is 64;
flat blocks avoid that structural imbalance in these controls. Scratch is
16 bytes per event, with no worker-private full histogram grids.

### Real cached-event ownership tests

The manual prototype derives the bin projector from the loaded authoritative
kernel, keeps its scientific arithmetic unchanged, and partitions events into
disjoint bin owners in original order. Its preserved source and hashes are in
each experiment output. This AST mechanism is confined to the experiment.

On the 24-run selection, alternating paired trials at eight workers measure
3.857 → 3.211 s for six copies and 4.160 → 3.597 s for twelve. At 32 workers,
they measure 3.368 → 2.703 s and 4.016 → 3.063 s. The real gains are smaller
than the synthetic result.

All 617 cached sources and twelve copies at 32 workers take median
**72.780 → 58.317 s**: 14.463 s, or 19.9%, saved in this event-only pass.
Cache reads take 11.293 / 11.540 s and rotation/scaling 13.391 / 14.013 s.
The old ordered kernel takes 40.734 s; the new projection, stable partition and
scatter take 8.401 + 10.182 + 6.979 s. All four trials preserve literal C/V/counts
and event support over the full 91,584,578-cell grid. They process 178,344,943
reduced events in 754 chunks, with all 617 cache hits and no raw fallback.
Maximum scratch is 11,185,080 bytes; inputs and scientific code remain unchanged.

These timings include event-cache reads, rotation and histogramming, but exclude
reduction, normalization and saving. The saved-project workflow currently takes
about 266 s, so this reduction in event work would remove only about 5% of it.
The production kernel remains unchanged: extra projection/partition/JIT paths
are disproportionate to this complete-workflow gain. Optional precision and
same-bin covariance were also outside the prototype scope.

Receipts under SEQ/IPTS-37189's `shared/nfit/benchmarks/6A-P-node19/nfit`:

- `seq-pilot24-ordered-owners-v011604-{8,32}workers/receipt.json`
- `seq-full617-ordered-owners-v011604-32workers/receipt.json`

### Accepted incremental symmetry reuse

The 0.116.5 candidate reuses six compatible cached operations and calculates the
six missing operations. Complete cache signatures and exact operation multisets
guard native raw-DGS independent-copy statistics. C/V/counts/exposure are added
before recomputing intensity, uncertainty, coverage and thresholds. Covariance
and reduced-MDE user-mask partitions retain full replay.

Matched saved-project jobs improve SEQUOIA histogram calculation
215.565 → 160.899 s and the complete load/rebin/save workflow
260.527 → 209.932 s (19.4% saved). HYSPEC improves 38.765 → 30.316 s and
49.665 → 41.626 s respectively (16.2% saved). Counts, C/V, masks and exposure
support are exact in both full grids; exposure differs only through addition
rounding. See [the symmetry-cache report](dgs-symmetry-cache-node19.md) for
scope, all-cell gates and scalar receipts. The current worker policy is retained.

A whole-grid scan was slower than full replay for a 24-run selection on the
91.6-million-cell grid. The final implementation therefore checks estimated
saved event work before loading a partial cache. The guarded small job uses
full replay; the guarded full617 job retains six cached plus six computed
operations and takes 153.954 s binning / 200.654 s load, bin and save. This
cost guard avoids the measured small-workload regression.

### Remaining fresh-reduction audit

The fresh raw event interval still combines HDF reads, event reconstruction,
cache writing and six-copy accumulation. A cached twelve-copy interval uses a
different chunk layout and copy count; subtracting it does not measure raw
reduction alone.

A separate instrumented 24-run, two-stage workflow confirms that geometry is
already reused: the first resolved geometry takes 10.494 s and all 24 take
10.781 s together. The fresh raw iterator resumes 2,784 times and accounts for
9.251 s cumulative time; HDF dataset reads across the workflow account for
4.572 s. Projection dispatch makes 16,728 calls and accounts for 5.516 s.
These overlapping cProfile measurements include instrumentation and both
workflow stages; they cannot be added or extrapolated into full-job speedups.
The scalar profile summary is in `dgs-raw-profile-node19.json`.

The raw iterator performs detector lookup, pulse selection, successive TOF/final
energy/momentum arrays, and detector/kinematic corrections. Its final projection
and scatter are already fused. Geometry arithmetic, calibrations and successful
inspection are already reused; reduced-event writes are coalesced and uncompressed.

The cache writer currently emits original bank chunks for fresh accumulation,
whereas saved rebins consume larger column blocks. Bounded coalesced delivery
could reduce repeated rotations, allocations and dispatches without another
producer framework. It must preserve cache bytes, event/raw-count order,
consumer memory limits, empty-bank progress, cancellation and publication after
final consumption. Its existing pass-through default is a tested contract.

Bounded independent raw-run producers feeding one ordered accumulator remain a
larger candidate if reconstruction dominates. Each producer must retain its own
resolved geometry. HDF library serialization may erase thread gains if reads
dominate. A fused reconstruction kernel has greater numerical risk and needs
array-pass dominance evidence first. Neither proposal is adopted solely from
the fresh/cached gap.
