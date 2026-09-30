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
Candidates were benchmarked with the existing installed nfit runtime before
updating its loose source modules and offline documentation. The desktop
shortcut continues to use the existing installation. No GUI launch check was
performed. Measurements include normal filesystem/OS caching; they are not
cold-disk measurements.

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

## Dataset event caches and symmetry binnings

The single project contains both datasets, all 1,355 per-run event caches, both
basic histograms, and the NiO `3bar` and `3barm` histograms. Each cached event
retains laboratory Q in Å⁻¹, energy transfer in meV, and corrected weight as five
float64 values. Output axes, UB, symmetry, and histogram bounds are applied
when binning; changes to the raw/calibration inputs or reduction settings
invalidate the relevant run's cache.

| Operation with initial event-cache implementation | Time |
| --- | ---: |
| NiO reduction, basic binning, and cache construction | 257.74 s |
| Sapphire reduction, basic binning, and cache construction | 341.15 s |
| Save both datasets' event caches and basic binnings | 26.27 s |
| NiO basic binning from saved event caches | 110.41 s |
| NiO `3bar` binning, six symmetry operations | 529.60 s |
| NiO `3barm` binning, twelve symmetry operations | 1,005.38 s |
| Save both additional symmetry binnings | 40.96 s |

All three cached NiO computations reported 617 cache hits and zero misses.
Raw metadata, instrument geometry, and calibration loaders were replaced by
functions that raise during these runs. Counts and masks match exactly;
signal, errors, and full normalization arrays agree at 1e-12 tolerance.
Project opening took 0.129 s with `numpy.load` replaced by a function that
raises, demonstrating metadata-only loading. Binnings and reduced events
remain lazy.

Before event compression, per-run cache assets occupied 10,129,415,838 bytes
for NiO and 13,956,879,112 bytes for sapphire. The project with both symmetry
histograms occupied 27,142,435,999 bytes (25.28 GiB). Event caches retain events
outside the first output grid so later grids can be constructed without raw
reduction.

`3bar` uses `x,y,z; y,z,x; z,x,y; -x,-y,-z; -y,-z,-x; -z,-x,-y`.
`3barm` uses `x,y,z; y,z,x; z,x,y; y,x,z; x,z,y; z,y,x; -x,-y,-z;
-y,-z,-x; -z,-x,-y; -y,-x,-z; -x,-z,-y; -z,-y,-x`.

## Ordered event accumulation and trajectory normalization

A compiled event pass combines bin lookup, range rejection, and the three
ordered updates for signal, variance, and counts. Coordinate transforms retain
their existing float64 operation order. Detector trajectory normalization
reuses its private worker grids across batches and merges the four monotone
momentum/energy crossing lists instead of insertion-sorting all crossings.
Every geometry still supplies its own detector angles and solid-angle weights.

Controlled warm measurements on the local Apple ARM machine:

- One million events on a 64 × 64 × 64 × 48 grid: ordered NumPy accumulation
  0.165 s versus compiled accumulation 0.0303 s (5.45×), for a mostly
  out-of-grid sample; 0.1992 s versus 0.05384 s (3.70×) for a denser sample.
  Signal, variance, and count arrays are exactly equal. Five runs were timed
  after compilation.
- 320,000 trajectories (64 runs, 5,000 detectors), four workers,
  32³ × 101 bins: insertion sort median 0.3127 s versus ordered crossing merge
  0.1524 s (2.05×). Coverage is exact; maximum normalization difference is
  5.68 × 10⁻¹⁴, consistent with float64 roundoff. Five warm runs were timed.

On the complete saved NiO event caches, these changes reduced basic binning
from 110.41 s to 72.69 s (1.52×). The six-operation `3bar` computation fell
from 529.60 s to 275.24 s (1.92×); the latter includes Python profiling
and is therefore conservative. It spent 118.66 s in the event phase and
151.02 s in normalization. Both outputs passed full-array comparison with
the saved binnings: exact counts/masks and 1e-12 signal/error/normalization
tolerance. Both used 617 cache hits and zero misses. Peak process RSS was
23.01 GiB for basic; the subsequent profiled `3bar` process high-water mark
was 41.47 GiB. No project bytes were changed by this benchmark.

The profile identifies additional archive-directory/chunk overhead and a
14.42 s serial merge of normalization worker grids. Subsequent changes
coalesce bounded event blocks, open each cached run once, and merge independent
normalization output
tiles in parallel with the worker addition order preserved.

## Event-cache compression experiment

The storage experiment used the shared NPZ/DEFLATE writer used by rebin
artifacts, retaining float64 bits and standard NumPy compatibility. Small bank
chunks are buffered into approximately 32 MiB blocks in original event order.
Columns use standard Fortran-order NPY storage, which compresses the NiO sample
better than interleaved rows. This is ordinary NPZ storage, not a custom codec.
Each block remains independently and lazily readable; older uncompressed caches
are also supported.

Conversion of all 1,355 saved caches, including reading them and verifying
SHA-256 hashes of the original and decoded event bytes, took 338.08 s using
eight bounded run workers. Every run's raw-event progress count, run metadata,
and normalization payload also matched. Compressed cache assets occupy
18,708,588,627 bytes, versus 24,086,294,950 bytes previously (22.3% smaller).
The compressed trial project occupied 21,764,702,616 bytes (20.27 GiB), down from 25.28 GiB.
Saving took 26.84 s; opening with numerical loads forbidden took 0.346 s.
Existing basic and symmetry histogram caches remained in that same project.
The final storage policy uses uncompressed event blocks, following the user's
preference for repeated-binning speed over these storage savings. Histogram
rebins retain their existing compression; both event formats remain readable.

Before the uniform-grid lookup change, complete binning from compressed
caches took 87.63 s for basic and 253.89 s for `3bar`, with exact counts/masks
and signal/errors/normalization agreement at 1e-12. Both reported 617 hits and
zero misses. Compression adds CPU decoding work: the uncompressed basic
candidate took 72.69 s, so storage savings are not a free speed increase.
These are complete operation measurements, including archive reads, projection,
accumulation, normalization, and finalization.

Additional controlled tests of the uniform-grid event lookup compare the
existing fused binary-search kernel with arithmetic lookup, corrected against
actual bin edges. For one million events and a 64 × 64 × 64 × 48 grid, the
five-run warm medians were 0.05096 s versus 0.03750 s for the denser coordinate
span ±1.35 (1.36×), and 0.03061 s versus 0.01126 s for span ±4 (2.72×).
Outputs were exactly equal. Nonuniform grids retain binary search; edge and
adjacent-float tests include large coordinate offsets and tiny steps.

With the uniform-grid lookup enabled, complete compressed-cache timings were
84.46 s for basic, 231.04 s for `3bar`, and 400.53 s for `3barm`. All three
passed full numerical comparisons with their saved binnings, with exact
counts/masks and 1e-12 signal/errors/normalization tolerance. Each reported
617 cache hits and zero misses. Peak RSS high-water marks were 23.04, 42.28,
and 43.51 GiB, respectively, in the sequential benchmark process.

A separate experiment precomputed detector directions and each run's energy
crossings outside the trajectory loop, with angle snapshots checked for exact
agreement across batches. For 320,000 controlled trajectories its warm median
changed from approximately 0.148 s to 0.144 s. This small gain did not justify
the additional preparation/cache machinery, and it was not adopted.

## Final uncompressed event policy

The final project uses uncompressed, coalesced event blocks. Conversion from
the compressed trial, including per-run event-byte SHA-256 checks and matching
normalization/run metadata/progress counts, took 40.98 s with eight bounded
workers. Cache assets occupy 24,006,909,616 bytes; the complete project is
27,063,023,605 bytes (25.20 GiB). The smaller header count saves approximately
79 MB even without compression. Saving took 31.72 s, and project opening with
numerical loads forbidden took 0.384 s. All 1,355 event caches and all requested
histograms remain inside the single project.

Event data remains on disk and is read one approximately 32 MiB block at a time.
Project opening does not hold the 25.20 GiB numerical payload in process RAM.
Binning still requires histogram arrays, trajectory worker grids, and the small
per-run normalization snapshots. The OS may cache file pages independently of
nfit's numerical heap.

The final complete uncompressed-cache benchmark measured:

| NiO binning | Initial cache implementation | Final CPU/block implementation | Speedup |
| --- | ---: | ---: | ---: |
| basic | 110.41 s | 56.69 s | 1.95× |
| `3bar` | 529.60 s | 202.11 s | 2.62× |
| `3barm` | 1,005.38 s | 373.96 s | 2.69× |

All three report 617 cache hits and zero misses, with raw metadata, geometry,
and calibration loaders set to fail if called. Counts and masks match exactly;
signal, errors, and full normalization arrays agree with the saved histograms
at 1e-12 tolerance. The project file's size and modification time remained
unchanged. Peak RSS high-water marks were 22.98, 42.16, and 43.53 GiB in the
sequential process. Basic's event phase finished at 27.07 s and its normalization
phase at 54.59 s; corresponding `3bar` phases finished at 63.49 and 195.58 s.
For `3barm` normalization finished at 366.16 s. These timings include archive
reads, coordinate projection, event accumulation, normalization and finalization;
comparison/loading of the saved reference is excluded.

Validation used the local nfit conda interpreter: full suite 2,177 passed,
one optional GPU test skipped, plus the final uncompressed-cache/Numba/archive
focused suite of 116 passing tests and a final three-test packaging check.
Ruff, byte-compilation, Sphinx with warnings treated as errors, and diff checks
passed. The desktop GUI launch check is left to the user as requested.

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
