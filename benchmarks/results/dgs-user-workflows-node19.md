# DGS workflows on ORNL node19

These measurements use analysis-node19, a Xeon Gold 6246R system with 32
physical cores, 64 logical CPUs, and approximately 754 GiB RAM. Both engines
have a 64-thread ceiling; nfit has a 200000 MiB scientific RAM ceiling. BLAS
uses one thread. Scientific jobs run sequentially. Filesystem caches are
uncontrolled, so these are fresh application workflows rather than cold-disk
measurements.

Native timings use the existing desktop installation. The reference invokes
the installed Shiver `GenerateDGSMDE` in one ordinary sequential job, followed
by Mantid `MDNorm`. Its own internal threading remains enabled. Neither
production nfit nor its unit tests invokes Mantid. Instrumentation records
children without replacing their algorithms. Profiling is separate from the
timed workflows.

## Matched SEQUOIA pilot

The 24 angle-spaced raw runs cover the complete 617-run NiO membership. Both
engines use the same UB source, vanadium/mask, reduction conventions, and
67 × 134 × 101 × 101 cube. The axes are (1,1,1), (1,−1,0), (1,1,−2), and
energy; energy spacing is 0.5 meV. Six and twelve symmetry operations reproduce
the requested 3bar and 3barm definitions. Original projects are untouched.

| Operation | nfit 0.116.0 | Shiver/Mantid |
| --- | ---: | ---: |
| Initial saved dataset, including loading, reduction, six-copy binning and saving | 51.066 s | 565.660 s |
| Reopen saved data and access the histogram | 30.658 s | 14.269 s |
| Later twelve-copy bin and save | 23.646 s | 159.489 s |

The native initial workflow is 11.1× faster; the later workflow is 6.7× faster.
Native saving persists reduced events and binnings in one project; the reference
saves merged MDE and histogram files. The reference reopens both files, whereas
native event-cache payloads remain lazy until the later rebin. Native initial
setup includes an unnecessary approximately 11-second inspection of the UB
reference workspace; the subsequent full-job harness reads only its 3×3 matrix.

The reference splits into 460.743 s for fresh sequential raw loading/reduction,
4.648 s saving MDE, 60.803 s for six-copy MDNorm, and 34.731 s saving the
histogram, plus 4.734 s setup. Its later MDNorm takes 117.055 s and saving takes
42.434 s. Native raw reduction, reduced-cache writing, and event accumulation
are interleaved: the 33.216 s combined interval is not a reduction-only time.
Native primary saving takes 6.433 s; later binning takes 15.874 s and saving
both binnings takes 7.761 s.

### Numerical comparison

Event numerator C, event variance V, and contribution counts are identical in
every cell for both symmetry choices. Exposure support is identical, including
the same 27/26 event-containing cells without positive exposure. These cells
remain masked. Exposure N has relative L2 differences of 4.06×10⁻¹⁰ and
3.53×10⁻¹⁰. Maximum differences relative to the peak N are 5.70×10⁻⁸ and
3.79×10⁻⁸; worst relative differences among individual nonzero N cells are
32.4 and 88.3 ppm. Full-cube exposure is numerically close, not bitwise equal.

For the HHH/energy cut at K=0±0.03 and L=0.33±0.02 r.l.u., signal and
uncertainty relative L2 differences are approximately 10⁻¹³, including coverage
fringes and the low-exposure region. Reported diagnostic Q-edge differences
up to 2.99×10⁻⁷ r.l.u. are exactly reproduced by native float32 step
multiplication versus diagnostic float64 interpolation between Mantid's stored
float32 limits; they did not change event assignments in this pilot. The small
whole-cube exposure residual has not been localized to one arithmetic operation.

## Full 617-run native baseline

The unprofiled 0.116.0 baseline imports 617 raw sources, constructs and saves
the six-copy histogram and reduced-event caches, reopens that project, makes
the twelve-copy histogram from saved events, and saves both.

| Stage | Seconds |
| --- | ---: |
| Import and setup | 73.266 |
| Raw reduction, cache construction and six-copy histogram | 458.509 |
| Primary project save | 35.320 |
| Initial saved dataset total | 567.152 |
| Metadata reopening | 0.685 |
| Saved histogram access | 39.706 |
| Saved-event twelve-copy binning | 257.952 |
| Save both binnings | 39.435 |

All 617 initial caches miss and all 617 later caches hit. The saved project
contains 617 reduced-event members and is 13.624 GB; reduced-cache members
occupy 11.520 GB. Peak resident memory is approximately 41.2 GiB for initial
construction and 53.8 GiB later. The normalization intervals take 91.926 s and
173.038 s respectively; they are observational progress intervals inside the
public workflow, not independently rerun kernels.

## Candidate changes and acceptance

The candidate uses source patches to the existing 0.116.0 bundle, with exact
module hashes recorded in its receipts. Release metadata is updated to 0.116.1
after scientific acceptance.

The candidate reuses successful scalar inspection after complete source-file
identity and monitor-policy checks, and calculates detector distances/directions
once per immutable resolved geometry and mask. Different instrument definitions,
run-log geometry and detector selection keep their own snapshots. Failed
calibration retries and warns; exceptional detector positions retain the original
selected-event arithmetic. There is no scientific precision or cache-format
change.

Large cached histograms decode with independent positional readers on the
original validated open file descriptor. An atomic project replacement cannot
mix old and new payloads. CPU/RAM limits, checksums, NumPy header validation,
immutable arrays, and existing platform/older-format fallbacks remain active.
Two alternating real-project prototype pairs measured median loading
27.764 → 2.537 s (10.9×), with byte-identical arrays.

The full candidate's initial saved-dataset workflow takes 503.554 s (8m24s),
versus 567.152 s (9m27s), a 1.13× speedup. Import/setup takes 77.801 s, the
combined raw/cache/six-copy histogram interval 389.663 s, and saving 36.051 s.
The event interval falls from 350.956 to 283.063 s (1.24×); normalization remains
approximately 93 s. Metadata reopen takes 1.063 s and cached histogram access
7.926 s, a 5.01× improvement in the actual full workflow. This measured load
gain is smaller than the alternating prototype's 10.9× result.

Saved-event twelve-copy binning takes 262.590 s, versus 257.952 s in the
baseline; its normalization interval is 172.280 s versus 173.038 s. Including
the later save, that workflow takes 298.990 s versus 297.402 s. No speedup is
claimed for this unchanged binning/normalization work. The complete two-stage
native workflow takes 811.540 s versus 904.951 s, excluding untimed diagnostics.
The final project retains exactly the same reduced-cache byte count; final
peak RSS is approximately 53.8 GiB.

Whole-array candidate acceptance passes for both six and twelve copies:
C, V, counts and edges are identical, exposure support is identical, and N
relative L2 differences are 2.92×10⁻¹⁶ and 3.07×10⁻¹⁶. Maximum absolute N
differences are 6.98×10⁻¹⁰ and 1.40×10⁻⁹. The strict comparison requires
every nonzero N cell within 10⁻¹² relative, including fringes. Cache usage
remains 617 initial misses and 617 later hits.

The full ordinary Shiver reduction and one/six/twelve-copy binnings were launched
after native acceptance, with real saves/reopens and process/resource logs.
The 2026-10-03 status check finds the job still reducing raw runs, without a
reported error; histogramming and saving have not finished.
Remembered two-hour reduction and thirty-minute binning times are informal
context, not measurements or benchmark estimates. The full reference remains
pending until its final receipt records successful completion.

## Evidence locations

Scientific outputs and receipts remain in the experiment's
`shared/nfit/benchmarks/6A-P-node19` folder:

- `nfit/seq-full617-v011600-baseline/receipt.json`
- `nfit/seq-full617-v011601-candidate/receipt.json`
- `mantid/seq-pilot24-v011600-02/receipt_summary.json`
- `comparisons/seq-pilot24-{6,12}copies-nfit-mantid.json`
- `reader-prototype/`

The full reference job uses tag `seq-full617-shiver-reference-01`. Its console
and timing stages go to `logs/mantid-seq-full617-shiver-reference-01.log`,
with process wall/CPU/RSS in the adjacent `-resources.txt` file. Its
`mantid/seq-full617-shiver-reference-01/receipt.partial.json` records progress;
successful completion replaces it with `receipt.json`.

The HYSPEC benchmark uses IPTS-36860 and separate twelve-run selections for
each measured detector-bank position, with explicit matching detector-tip masks
and the respective UB source. Its 34° source contains all 361 current runs;
the historical 337-run histogram has narrower membership. Native candidate
acceptance compares original event cells and exposure support before pooling.
Existing Shiver observed-extrema event clipping is a separate documented HYSPEC
policy difference and must not be attributed to these optimizations.

| HYSPEC bank pilot | Baseline initial save | Candidate initial save | Baseline cached twelve-copy bin | Candidate cached twelve-copy bin |
| --- | ---: | ---: | ---: | ---: |
| 34°, twelve runs | 9.664 s | 9.576 s | 1.719 s | 1.478 s |
| 70°, twelve runs | 9.012 s | 9.056 s | 1.452 s | 1.340 s |

There is no material initial-workflow gain in these small HYSPEC pilots.
Both banks and both symmetry choices preserve literal C/V/counts and exposure
support. N relative L2 differences are at most 7.38×10⁻¹⁷ and maximum absolute
differences at most 2.91×10⁻¹¹. Initial caches miss for all twelve runs and later
rebins hit all twelve. These bounded pilots establish candidate correctness;
they do not measure full 361-run raw reduction or whole-project rebuilding.

The matched ordinary Shiver pilots take 47.220 s at 34° and 49.820 s at 70°
for the initial saved dataset, versus native 9.576 s and 9.056 s. Their saved-MDE
twelve-copy bin/save intervals are 3.593 s and 4.259 s. These are bounded
measurements, not full-collection extrapolations.

Cross-engine HYSPEC numerical parity is incomplete. At 34°, six-copy counts
differ in 319 cells (net native excess 26 contributions), with exposure relative
L2 difference 2.985×10⁻⁴ and 23 exposure-support differences. Twelve copies
have a net excess 52 and 32 support differences. At 70°, six-copy counts differ
in 30 cells (net excess 30) while exposure agrees near 3.45×10⁻¹⁴ with identical
support. Twelve-copy counts have a net excess 60, exposure relative L2 difference
1.198×10⁻⁵, and two support differences. Event extrema clipping can explain
some excess events, but it does not establish the cause of the 34° migrations
or all exposure differences. A metadata audit pinpoints one pre-existing
discrepancy in run 505784: nfit uses the arithmetic omega average
48.5016271525°, while Mantid's duration-weighted mean is 48.5043924634°.
Reconstructing the mean from its two timestamps reproduces Mantid's angle.
Other 23 pilot goniometer matrices match to rounding; Ei, T0 and UB agree.
This is a concrete source of 34° coordinate differences, but its complete
histogram contribution has not been isolated by replay. Correcting that recipe
and diagnosing remaining boundary differences form a separate scientific
checkpoint; the optimizations pass the strict unchanged-native gate.
