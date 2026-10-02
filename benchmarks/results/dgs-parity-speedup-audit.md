# Remaining DGS performance opportunities (read-only audit)

Code inspected: nfit 0.107.0 scientific pipeline. No production files modified.
Local prototypes use the mandated nfit conda environment and preserve Mantid
float32 multiply/add order, symmetry-copy order, event order, and C/V/count sums.
No ORNL scientific projects were opened or changed by this audit; no competing
heavy remote benchmark was started during the authorized project rebuild.

## Highest priority: fused event projection and ordered accumulation

Current `prepare_dgs_event_projector` builds the N×4 float32 input independently
for every symmetry copy, then creates several N×4 multiply/add temporaries.
`accumulate_discrete_event_coordinates` reads that projected array in another
compiled pass. A uniform-grid Mantid-specialized scalar Numba prototype projects
one event and immediately updates C,V,count, retaining the arithmetic and order.
It falls back to the established projector for nonuniform/native policies; no
scientific defaults change.

Warm local Apple ARM measurements, 600,000 events × six symmetries, five repeats:

| Synthetic workload | Current median | Fused median | Kernel speedup | Contributions |
|---|---:|---:|---:|---:|
| Fine slab 61×7×5×61 |0.167446 s|0.040178 s|4.168×|3,497|
| Coarsened cube 33×67×50×50 |0.252266 s|0.060524 s|4.168×|2,799,624|

Both C,V,count arrays were bitwise identical. An additional 150,000
boundary-adjacent physical-coordinate probes produced bitwise identical
projected float32 coordinates (with NaNs compared as equal).

The prototypes are diagnostic experiments, not installed runtime code.

These are kernel-only synthetic results, not end-to-end speedups. Current real
receipts put ~160/170 s of fine-cut MDE time in the event phase, and ~112/244 s
of full-cube time there. Event reading/filtering and raw reduction remain.
If half the fine-cut event phase is accelerated at the measured kernel ratio,
end-to-end speedup would be ~1.55×. Eliminating the entire full-cube event phase
would cap end-to-end improvement at ~1.85×; a translated fourfold event-phase
improvement would yield ~1.54×. These bounds do not establish actual CPU speedups.

Adoption gate: benchmark the existing real reduced-event cache on ORNL after the
current rebuild, comparing every C/V/count cell literally and N unchanged;
verify Q_sample source-order transformation remains unchanged; include weighted
runs, masks, explicit-edge fallback, invalid coordinates, exact boundaries,
6/12 symmetries, and streamed multiple chunks. Keep fastmath off. Do not collapse
matrix products or change event/symmetry ordering. A focused shared kernel can
serve both raw DGS and MDE without a second independent transform definition.

## Other structural candidates

1. Bounded independent-run raw reduction with ordered histogram consumption.
   Per-run reduction is serial today. Prefetching reduced chunks/caches can
   overlap reduction/I/O while consuming completed runs in their original order.
   Preserve atomically published per-run caches and cancellation cleanup. HDF5
   global locking limits threads; process workers duplicate geometry/JIT/startup
   and require bounded disk/RAM use. Need isolated reduction/I/O profile before
   choosing the design. Up to 4 workers is a plausible first experiment, not a
   proven production setting. Account CPU quota and avoid nested oversubscription.

2. Exposure reuse when the exact exposure inputs are unchanged.
   A changed event efficiency correction or intensity scale can invalidate C/V
   without changing N. Current full histogram caches do not independently reuse
   N. A separate complete exposure signature can reference the old lazy N array
   without duplicating its disk payload. Include geometry, detectors, vanadium,
   masks, charge, fit weights, Ei, energy limits, UB/goniometer, symmetry,
   grid/edges, precision, and the algorithm version. Changing fit_weight or
   detector masks does change N. This can avoid most normalization for applicable
   repeated reductions but does not accelerate a first new grid. Existing scale
   and background caches already handle some such workflows: avoid duplication.

3. Immutable detector lookup payload.
   Raw reduction recomputes detector distance and direction for every event.
   Distances/directions can be cached with each checked immutable geometry,
   using identical NumPy operation order; compact detector-ID lookup can also
   avoid repeated searchsorted when the ID span is suitably bounded. Very sparse
   or large IDs need the current safe fallback. Gains unmeasured, likely secondary
   compared to the measured projection opportunity. Compiling exp/sqrt reduction
   wholesale risks libm rounding differences in literal Mantid weights.

4. Respect actual CPU quota in worker planning.
   `_parallel.detect_cpu_budget` reads affinity but not `cpu.max`, despite the
   nearby affinity/cgroup comment. Quota-aware ceilings can avoid 64 private
   normalization grids under a 20-core user quota. This improves memory and may
   reduce throttling/merge cost. Normalization addition partition changes can
   change last-bit N; compare coverage exactly and N within the established
   parity tolerance rather than promise literal exposure equality.

## Lower priority / rejected without new evidence

- Exact-identical trajectory aggregation: geometry/angles/Ei/limits/masks must
  all agree. Per-run energy windows vary even under first-run trajectory Ei, so
  the present NiO data is unlikely to expose many exact duplicates. Charge
  pooling changes floating-point addition order; do not approximate grouping.
- Event compression, projector caching, generic batching, GPU changes and
  fractional/discrete switching have already been investigated or change the
  scientific result. Avoid repeating those experiments without a new profile.
- Fine-cut normalization is only ~5–10 s; optimizing that kernel cannot provide
  a major fine-cut end-to-end gain. Full-cube normalization remains substantial.

Conclusion: major remaining speedups have not been exhausted. There is a simple,
measured fourfold local kernel opportunity in event projection/accumulation.
Real cache benchmarks, not the synthetic result, should decide adoption.
