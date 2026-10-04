# Remaining DGS performance opportunities

This code audit follows the accepted event fusion, geometry/inspection reuse and
parallel histogram loading changes. Timings below are the unprofiled 0.116.1
full 617-run SEQUOIA workflow on node19. The current angle-corrected build and
accepted trajectory-pooling build have separate acceptance receipts. Pooling
reduces full617 normalization intervals by 31–32% and the complete saved
workflow by 9.96%, while preserving the strict every-cell numerical gate. None of the
proposals below is a measured additional speedup.

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

This is synthetic evidence, not a measured scientific-workflow speedup. A manual
cached-event prototype must next check actual policy-owned transforms, masks,
source ordering, boundaries, cancellation, optional variance policies, full-grid
literal C/V/counts and whole-workflow time. No ownership algorithm is adopted
from this audit alone.
