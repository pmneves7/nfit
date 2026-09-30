# SEQUOIA reduction and binning

Measurements on 2026-09-29 use NiO runs 392985–393469 and 393500–393631
(617 runs, 373,966,474 raw detector events) and sapphire runs 409259–409996
(738 runs, 447,189,564 raw events). Both use the processed vanadium and mask
`van382236_D` and their respective saved UB matrices from IPTS-37189.

The cube follows (1,1,1), (1,-1,0), and (1,1,-2), with center limits and steps
−1…1 / 0.03 r.l.u., −2…2 / 0.03 r.l.u., −1…1 / 0.02 r.l.u., and
0…50 / 0.5 meV. Its shape is 67 × 134 × 101 × 101 (91,584,578 bins).

## Environment and timing boundaries

The cluster node has two AMD EPYC 9334 processors, 64 physical CPU cores,
about 1.5 TiB of physical RAM, and an NVIDIA RTX A6000 with 48 GiB of memory.
nfit preferences limit work to 64 CPUs and 200,000 MiB of RAM. The runtime
uses Python 3.14.7 and NumPy 2.5.3; the Shiver environment uses Mantid 6.16.0.1.
Native BLAS was limited to one thread during the nfit candidate measurements.
Candidate modules were run with the existing installed nfit runtime; these
experiments did not update the cluster desktop application.

Raw reduction times include reading detector events, calibration, coordinate
conversion, histogram accumulation, trajectory normalization, and output-array
finalization. Lightweight collection import and cache serialization are reported
separately. No MDE intermediate is produced by the native nfit route.

## Raw reduction

| NiO workflow | Time | Status |
| --- | ---: | --- |
| Original nfit raw job, including project save | 3,022 s (50.4 min) | Measured |
| Geometry cache candidate with full Python profiling enabled | 392.05 s (6.53 min) | Measured; profiling overhead included |
| Geometry cache, static masks, sorted detector lookup, and run metadata reuse | 240.69 s (4.01 min) | Measured; reduction and histogram |
| Lightweight raw collection import for the optimized job | 44.58 s | Measured separately |
| Standard single Shiver job reduction | About 8,120 s (135 min), plus final merge/save | Estimated |
| Eight concurrent Shiver jobs, recovery, merge, and MDE save | 1,627 s (27.1 min) | Measured |

The standard Shiver estimate comes from 25 completed `ConvertDGSToSingleMDE`
calls in the original single-job run: total 328.91 s, mean 13.1564 s per run,
median 13.09 s, range 12.53–14.44 s. Multiplying the mean by 617 yields 135.3
minutes. This is an extrapolation, not a completed standard-job measurement;
later event loads and filesystem behavior can change it. Allowing for final
merge/save and the histogram workflow below suggests roughly 2¼–2½ hours for
the usual complete Shiver/Mantid workflow. “Single job” still permits internal
Mantid threading; it means one normal Shiver invocation rather than eight
independent reduction jobs.

The native NiO result retains exactly the original event counts (58,545,675
accepted events) and masks. Signal and errors agree with the saved baseline to
float64 roundoff; the largest absolute difference is 3.55 × 10⁻¹⁵. The old cache
writer omitted the large normalization array, so there is no saved full
denominator against which to assert byte equality. The repaired caches retain
that array as an auxiliary channel and use the corrected serializer.

Geometry reuse is keyed by the complete instrument XML bytes, with at most
eight entries. Each lookup reads the current definition. Distinct instruments,
positions, or efficiency parameters therefore produce distinct entries. Static
vanadium/mask selection is evaluated once per run; its sorted detector lookup
is reused across event chunks. Run metadata is refreshed once per reduction
and reused for normalization within that operation.

The corresponding optimized sapphire reduction took 317.61 s (5.29 min),
with 54.86 s of collection import reported separately. Its 117,417,178 accepted
events, masks, signal, and errors agree with the previous saved result; arrays
were compared at float64 tolerance (1e-12), with exact counts and masks.

## Histogram rebin modes, workers, and batches

These measurements read the repaired, normalization-weighted NiO histogram
and coarsen all axes by two to 33 × 67 × 50 × 50. Loading its 3.50 GiB numerical
payload took about 9.5 s. Each fresh process measured three repetitions per
mode; the table reports medians including the first dispatch.

| Workers | Batch target | Discrete | Fractional momentum only | Fractional all axes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 192 MiB | 5.68 s | 7.10 s | 8.60 s |
| 8 | 192 MiB | 5.68 s | 6.38 s | 6.79 s |
| 64 | 32 MiB | 5.63 s | 6.89 s | 7.07 s |
| 64 | 192 MiB | 5.75 s | 6.97 s | 7.14 s |
| 64 | 1024 MiB | 5.62 s | 6.90 s | 7.08 s |

Discrete assignment is about 16–20% faster than fractional assignment with
parallel workers for this complete histogram operation. Eight workers slightly
outperform 64 on the fractional workload. Batch changes give small differences;
these results do not justify new worker or batching policies. Geometric coverage
and source preparation are included, so a reduction in accumulation work does
not translate directly into the same end-to-end gain. Within each mode the
reported output summaries are identical across configurations. Different modes
change the interpolation and therefore the scientific result.

## Binning the same MDE file

These times use the complete 617-run Shiver MDE file and the same cube.

| Operation | nfit | Shiver/Mantid |
| --- | ---: | ---: |
| Import/metadata inspection or full load with vanadium | 3.66 s | 108.69 s |
| Histogram and normalization | 106.68 s | 131.59 s |
| Mantid full load, histogram, and saved histogram | — | 331.12 s |
| Process peak RSS | 16.49 GiB | 42.48 GiB |

nfit's MDE bin time comprises about 59.37 s of event accumulation, 9.06 s of
normalization setup, 35.00 s of trajectory integration, and 3.25 s of
finalization. This measured binning gain is about 19%, substantially smaller
than the raw reduction gain. Both native raw-DGS and MDE event binners already
use discrete event assignment.

## Sparse worker merge

A separate controlled CPU experiment used 1,000,000 uniformly distributed 4D
points in a coordinate span of 0.8, a 32⁴ output grid, discrete assignment,
four sparse workers, and a 192 MiB batch target. On the local 16-core Apple ARM
machine, a warm full rebin took 2.273 s with the Python map merge and 0.125 s
with the compiled merge: an 18.1-fold improvement. Signal, propagated errors,
and sample-count arrays were exactly equal. The first compiled run took 0.164 s.

This change benefits sparse threaded reduction. Dense and serial reduction
already merge arrays without Python iteration and do not receive this gain.

## GPU prospects

The GPU has enough memory for these histogram accumulators. Event coordinate
conversion, efficiency weights, scatter accumulation, and detector trajectories
are plausible GPU targets. XML expansion, Python/HDF5 metadata work, and file
reads remain CPU/I/O work. A useful prototype should keep histogram arrays on
the device across batches and measure transfers, startup, and final output
copies along with kernel time. Histogram atomic contention and the A6000's
FP64 capability also need measurement. No neutron reduction GPU backend was
implemented or timed here.
