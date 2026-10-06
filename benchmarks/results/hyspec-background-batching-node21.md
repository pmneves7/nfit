# Native HYSPEC background batching

## Matched real-data trial

Measured October 6, 2026 on analysis-node21, sequentially with 16 workers,
100,000 MiB RAM allowance and 192 MiB batch allowance. The native project's
361-run low-temperature bank-70 sample contains 105,836,355 raw events. One
565,707-event dummy run is replayed over those angles and eight symmetry
operations, onto the saved in-plane grid (161 × 221 × 12 × 49 cells).
Both variants load the same unchanged source project and save separate projects.
Scientific outputs, temporary files and logs remain in the HYSPEC IPTS.

| Timed scientific stage | Baseline seconds | Candidate seconds |
| --- | ---: | ---: |
| Sample raw reads, reduction, event-cache writes and histogram | 72.01 | 71.24 |
| First background preparation | 69.61 | 57.40 |
| Second background preparation | 63.27 | 44.15 |
| Project save, including reduced-event caches | 14.05 | 14.44 |

The background stages improve by **18% and 30%**. Their combined time improves
by 24%; the sum of these four timed stages improves by 15%. These are two
background preparations in one matched workflow, not repeated complete-project
benchmarks or a prediction for all 27 configured work items. Imports, source
project metadata loading and untimed diagnostic HDF writes are excluded.
Filesystem caches and other users' activity were uncontrolled. Sample reduction
and saving show no established improvement. No new Mantid timing was measured.

Peak process RSS was 5.64 GiB for baseline and 7.15 GiB for candidate. This
includes the full sample/background setup and save, not just replay scratch.
Both remain within the configured RAM allowance; batch scratch still obeys the
existing byte limit. Larger permitted batches and retaining source caches do
have a memory cost.

## Adopted changes

1. Background preparation retains original dataset objects and passes ancestor
   masks separately. Previously `dataclasses.replace` discarded the runtime
   reduction-cache field. Dummy caches were published on temporary objects;
   subsequent background binnings reduced the file again, and saved metadata
   could reference an absent cache asset. The baseline did not retain the dummy
   cache after either pass. The candidate retained it on the original run.
   Regression tests cover ordered inherited/additive masks, reuse and lazy
   reopening after saving.
2. The replay work ceiling increases from one million to four million
   event-transform tasks per batch. The existing scratch-byte ceiling and
   100,000-row ceiling remain effective. This amortizes compiled parallel
   dispatch, allocations and small native reads. It does not change arithmetic,
   event order or the same-event collision/variance algorithm. Tests check byte
   and work ceilings and cancellation between batches.

The first change covers native and imported reduced-event background workflows;
the second depends on transform count and resource limits. Neither selects an
instrument, a run count or a particular grid. Ordinary point-measurement
binning follows its existing path. No worker-private full histogram grids or
instrument-geometry assumptions were added.

## Numerical gate

Every one of the 20,921,628 cells was compared with bounded reads. Counts, masks,
all edges and exposure support match literally, including coverage fringes.
Maximum relative differences are 1.94e-15 for exposure, 1.94e-15 for signal and
1.92e-15 for uncertainty. These pass the existing 1e-12 relative gate with zero
absolute tolerance. Parallel trajectory sums retain floating-point roundoff;
the results are not claimed to be bitwise identical normalized arrays.
Both original-project identity checks passed. Diagnostics contain normalized
signal, uncertainty, exposure, counts and masks, rather than raw C/V channels.

The [receipt](hyspec-background-batching-node21.json) includes configuration,
source module hashes, stage times and full numerical summaries. The candidate
snapshot uses the cache fix and an explicit four-million-task benchmark override;
the retained production change sets that same ceiling directly.

## Remaining audit conclusions

The separate component profile confirmed that native replay reconstruction was
small in this case (about 0.56 seconds of a 62-second background pass). Geometry
preparation, trajectory integration and replay dominate. Decoupling input and
replay batches further is therefore deferred until evidence warrants it.
Serial ordered histogram updates protect shared-bin sums and avoid private
large grids. Low CPU use alone is not evidence of a worthwhile parallelization.

Cross-entry duplicate reduction, automatic bounds and arbitrary nested workflows
still require complete-project profiling before adopting further changes. The
older node23 build log stopped without a completion receipt; its process status
could not be established from node21. This trial does not certify that rebuild.
