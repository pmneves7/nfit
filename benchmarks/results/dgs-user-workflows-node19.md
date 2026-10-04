# DGS workflows on ORNL node19

These measurements use analysis-node19, a Xeon Gold 6246R system with 32
physical cores, 64 logical CPUs, and approximately 754 GiB RAM. Both engines
have a 64-thread ceiling; nfit has a 200000 MiB scientific RAM ceiling. BLAS
uses one thread. Scientific jobs run sequentially. Filesystem caches are
uncontrolled, so these are fresh application workflows rather than cold-disk
measurements.

Native timings use the existing desktop runtime. Source-candidate jobs explicitly
select staged code and record its module hashes. The reference invokes
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
up to 2.99×10⁻⁷ r.l.u. are exactly reproduced by native float 32 step
multiplication versus diagnostic float 64 interpolation between Mantid's stored
float 32 limits; they did not change event assignments in this pilot. The small
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

## Completed full 617-run ordinary Shiver reference

The successful reference contains 617 experiments and 178,343,999 merged events.
It runs one sequential `GenerateDGSMDE` job with ordinary internal Mantid
threading, followed by resident-MDE `MDNorm`. These are measured times on the
same node, rather than extrapolations from the pilot or from Paul's remembered
two-hour reduction and thirty-minute binning.

| Stage | Shiver/Mantid seconds |
| --- | ---: |
| Imports/setup | 4.537 |
| Sequential raw loading/reduction | 12947.186 |
| Save merged MDE | 259.797 |
| Primary resident six-copy MDNorm | 2473.180 |
| Save primary histogram | 231.354 |
| Initial saved-dataset workflow | 15916.054 |
| Reopen saved MDE | 161.318 |
| Reopen primary histogram | 123.301 |
| Saved-MDE one-copy MDNorm | 352.619 |
| Save one-copy histogram | 116.522 |
| Saved-MDE six-copy MDNorm | 1813.511 |
| Save six-copy histogram | 149.083 |
| Saved-MDE twelve-copy MDNorm | 2996.235 |
| Save twelve-copy histogram | 149.371 |

The initial saved dataset takes **4h25m16s**, compared with the earlier accepted
nfit 0.116.1 workflow's **8m24s** (31.6×). The later twelve-copy calculation and
save take **52m26s**, compared with **4m59s** (10.5×), with reopens reported
separately. Native reduced-event writing and first histogramming are interleaved;
there is no measured native reduction-only interval to compare with 12947 s.
The post-angle-correction 0.116.3 workflow takes 546.518 s (9m07s) initially
and 309.367 s (5m09s) for the later rebin/save. Its metadata reopen and primary
histogram access total 4.584 s. Both full-cube comparisons still have literal
C/V/counts, exact exposure support and the same boundary-scale N residuals.
These current timings give 29.1× and 10.2× against the measured reference.
The accepted trajectory-pooling candidate is measured below.

Peak reference RSS is 177884.734 MiB (173.7 GiB). The complete process lasts
21855.061 s (6h04m15s) because it also performs all three independent cached-MDE
rebins, save/reopen/cleanup operations and diagnostics. That total is not the
initial saved-dataset time. Child-algorithm and enclosing-workflow timers overlap
and must not be added together. Normal buffered saves are measured; filesystem
caches are uncontrolled.

### Full-cube and low-coverage acceptance

Untimed streamed comparisons check all 91,584,578 cells of each six/twelve-copy
cube against nfit 0.116.1. C, V and event-contribution counts agree literally.
Exposure support agrees exactly; the same 13/9 event-containing cells without
positive exposure remain masked. N relative L2 differences are 1.950×10⁻¹⁰
and 1.625×10⁻¹⁰. The worst individual nonzero-cell differences are 12.028 and
6.907 ppm; maximum differences relative to peak N are 5.251×10⁻⁹ and
3.497×10⁻⁹. This retains the known trajectory-boundary arithmetic residual,
accepted as immaterial rounding, rather than claiming bitwise N parity.

In the HHH/energy cut at K=0±0.03 and L=0.33±0.02 r.l.u., event statistics and
exposure support also agree literally. For six copies, signal relative L2
differences are 3.23×10⁻¹³ at coverage fringes, 1.08×10⁻¹³ in the row-wise
lowest-exposure decile and 4.81×10⁻¹⁴ in the central low-coverage ROI. Observed
uncertainty differences have corresponding relative L2 values 3.01×10⁻¹³,
2.48×10⁻¹³ and 5.93×10⁻¹⁴. Every diagnostic input's size and modification
time is unchanged.

## Guarded trajectory pooling on the full 617-run job

The 0.116.4 candidate combines charge only for literally identical affine
transforms, effective incident energy and clipped energy windows, after detector
identity, geometry, mask and normalization equivalence checks. Mixed raw geometry
keeps its existing path; MDE sources pool separately within each checked geometry.
Event accumulation and source membership are unchanged.

| Operation | Time-weighted 0.116.3 | Pooling candidate | Improvement |
| --- | ---: | ---: | ---: |
| Initial saved dataset | 546.518 s | 504.488 s | 1.083× |
| Reopen and access primary histogram | 4.584 s | 4.383 s | 1.046× |
| Cached twelve-copy bin | 272.298 s | 225.373 s | 1.208× |
| Later bin and save | 309.367 s | 265.919 s | 1.163× |
| Complete two-stage workflow | 860.469 s | 774.791 s | 1.111× |

The complete workflow saves 85.678 s (9.96%). Six/twelve-copy normalization
processes 243,187,488 / 486,374,976 detector trajectories instead of
373,250,448 / 746,500,896: **34.8% less trajectory work**. Observed normalization intervals improve 98.659 → 66.997 s (32.1%) and
180.280 → 124.205 s (31.1%); these are progress
intervals within public operations, not independent kernel timings. The event
intervals are 309.317 s and 86.801 s. Raw loading, reduction, event-cache writing
and accumulation remain interleaved. Cache usage stays 617 misses initially and
617 hits later. Peak RSS is 55,128.28 MiB (53.8 GiB), and the project remains
13.624 GB with 11.520 GB of reduced-event cache payloads.

Every cell of both 91,584,578-cell cubes passes the strict native gate:
C/V/counts/edges are literal, exposure support is identical, and maximum relative
nonzero N differences are 1.835×10⁻¹⁵ / 1.619×10⁻¹⁵. Relative L2 N differences
are 2.883×10⁻¹⁶ / 3.052×10⁻¹⁶. The same 13/9 event-containing cells without
positive exposure remain masked. This also preserves the accepted cross-engine
boundary residuals and low-coverage cut. Source inputs remain unchanged.

Against the measured ordinary Shiver workflow, the new initial saved dataset is
**31.5× faster** (8m24s versus 4h25m16s), and the later twelve-copy bin and save
is **11.8× faster** (4m26s versus 52m26s). These ratios include each engine's
normal persistence workflow. They do not compare a native reduction-only timer
or claim that a resident histogram has been loaded from cold disk.

### Normalization worker count

A separate metadata-only MDE sweep uses the same 617 runs, full 91,584,578-cell
grid and pooled twelve-copy tasks, without event reads or saves. Two alternating
pairs measure 32/64/64/32 workers. Times are 137.490 / 130.067 / 124.984 /
147.150 s, giving medians **142.320 s at 32** and **127.526 s at 64**. The current
64-worker setting is 1.116× faster and remains unchanged. Peak RSS is about
21.3 GiB at 32 and 40.1 GiB at 64; managed RAM still bounds available workers.
All cells pass the exposure tolerance and support gate; maximum relative
nonzero difference is 2.24×10⁻¹⁵. This sweep uses the saved MDE detector payloads,
including masked entries, so its task count and timings are not the raw project's
normalization progress interval. Inputs and code hashes remain unchanged.
Evidence: `nfit/seq-full617-normalization-workers-v011604/receipt.json`.

## Evidence locations

Scientific outputs and receipts remain in the experiment's
`shared/nfit/benchmarks/6A-P-node19` folder:

- `nfit/seq-full617-v011600-baseline/receipt.json`
- `nfit/seq-full617-v011601-candidate/receipt.json`
- `nfit/seq-full617-v011603-timeweighted-baseline/receipt.json`
- `nfit/seq-full617-v011604-pooling-candidate/receipt.json`
- `mantid/seq-full617-shiver-reference-01/receipt.json`
- `diagnostics/full617-v011601-vs-shiver-{6,12}.json`
- `diagnostics/full617-v011603-vs-shiver-{6,12}.json`
- `diagnostics/full617-v011604-pooling-vs-timeweighted-{6,12}.json`
- `mantid/seq-pilot24-v011600-02/receipt_summary.json`
- `comparisons/seq-pilot24-{6,12}copies-nfit-mantid.json`
- `reader-prototype/`

The full reference job uses tag `seq-full617-shiver-reference-01`. Its console
and timing stages go to `logs/mantid-seq-full617-shiver-reference-01.log`,
with process wall/CPU/RSS in the adjacent `-resources.txt` file. Its
`mantid/seq-full617-shiver-reference-01/receipt.json` records successful
completion. Reference diagnostic arrays are `arrays_primary_6copies.h5` and
`arrays_rebin12_12copies.h5` in that job directory.

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

Before the time-weighted correction, the matched HYSPEC pilots had these
cross-engine discrepancies. At 34°, six-copy counts
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
These measurements prompted the time-weighted correction and replay below;
the performance optimizations separately passed the unchanged-native gate.

### Corrected time-weighted HYSPEC replay

The fresh 0.116.3 twelve-run replay fixes run 505784's accepted-time mean.
Primary six-copy exposure now agrees with resident Mantid at relative L2
6.11×10⁻¹⁵ (34°) and 3.45×10⁻¹⁴ (70°), with identical support. The 34°
event migrations disappear: only 32 additional boundary contributions remain,
versus 30 at 70°, all with one extra contribution per differing cell. These
tiny observed-extrema differences remain explicitly accepted rounding.

Twelve-copy reference normalization is calculated after saving and reloading
MDE. It differs by relative L2 9.06×10⁻⁷ and 1.198×10⁻⁵, with two 70°
support differences. A control using only Mantid shows the same issue:
its resident-versus-reloaded **six-copy** N differs by 1.100×10⁻⁶ and
1.495×10⁻⁵, with four support changes at 70°. The remaining comparison
therefore mixes a storage round trip with the symmetry change. Fresh resident twelve-copy controls remove that confound: N relative L2
is 5.18×10⁻¹⁵ (34°) and 2.51×10⁻¹⁴ (70°), with identical exposure support.
Only the accepted energy-edge count excess remains (64/60 contributions).
Loaded UB, goniometer matrices and charge are unchanged; substituting actual
post-LoadMD detector directions into native normalization reproduces the
loaded Mantid residual. The instrument parameter serialization/reconstruction
changes those directions. See [the round-trip attribution](hyspec-mde-roundtrip.md).
Native preserves the physical directions rather than emulating that loss.

The guarded pooling candidate preserves both corrected native banks and both
symmetry choices: C/V/counts/edges are literal, support is identical, and maximum
relative nonzero N difference is 5.54×10⁻¹⁶. Original projects remain unchanged.

### Full 361-run HYSPEC workflow

The corrected 34° workflow includes all raw runs 505555–505915, matching masks,
UB, Ei/T0, accepted-time angles and the 81 × 81 × 9 × 57 grid. Jobs run
sequentially on node19 with the same resource ceilings used above.

| Operation | nfit pooling candidate | Ordinary Shiver/Mantid | Ratio |
| --- | ---: | ---: | ---: |
| Initial reduction, six-copy binning and saving | 88.976 s | 1434.725 s | 16.1× |
| Reopen and access saved native histogram | 0.704 s | Reported in reference receipt | — |
| Later twelve-copy bin and save | 41.731 s | 101.513 s | 2.43× |

The initial native workflow is **1m29s**, versus **23m55s**. Its combined raw
reduction/cache construction/six-copy histogram interval takes 62.301 s;
event processing takes 54.196 s and normalization 5.289 s. The later bin takes
31.616 s, including 26.530 s for events and 4.056 s for normalization, followed
by 10.106 s saving both binnings. All 361 initial caches miss and all 361 later
caches hit. The saved project is 4.010 GB; native peak RSS is 1355.8 MiB.

The ordinary reference spends 1315.667 s in sequential raw loading/reduction
and 48.751 s in resident six-copy MDNorm. It contains 361 experiments and
76,874,136 merged events; peak RSS is 30674.4 MiB. Its later twelve-copy MDNorm
takes 89.656 s and saving takes 11.857 s. These latter times exclude MDE
reopening, which is recorded separately. The reference's extra independent
one/six-copy calculations and diagnostics are excluded from the headline times.

The resident six-copy comparison checks every one of 3,365,793 cells.
Exposure support is identical (415,120 exposed cells), with N relative L2
difference **3.77×10⁻¹⁵** and maximum difference relative to peak N
1.14×10⁻¹⁴. Event C/V/counts differ in 660 cells, with 944 extra native symmetry
contributions; maximum per-cell count difference is seven. This is the accepted
observed energy-limit difference, separate from normalization parity. All count
deltas are positive and confined to the two endpoint energy bins, consistent
with the reference's observed-extrema clipping; no interior count deltas occur.

The saved/reloaded twelve-copy reference has identical support (417,186 exposed
cells), but N relative L2 difference 5.93×10⁻⁷ and worst individual nonzero-cell
difference 0.630%. It retains the separately attributed instrument serialization
loss. Event C/V/counts differ in 1072 cells; maximum count difference is ten.
Native pooling acceptance uses the corrected native baseline and the fresh
resident pilot controls, rather than claiming equality with altered loaded
geometry. All original sources and projects remain unchanged.
The twelve-copy count deltas are likewise positive and confined to endpoint bins.

Evidence under HYS/IPTS-36860's `shared/nfit/benchmarks/6A-P-node19`:

- `nfit/hys34-full361-v011604-pooling/receipt.json`
- `mantid/hys34-full361-shiver-reference-01/receipt.json`
- `diagnostics/hys34-full361-v011604-pooling-vs-shiver-6.json`
- `diagnostics/hys34-full361-v011604-pooling-vs-shiver-loaded12.json`

The aggregate endpoint-location audit is in SEQ/IPTS-37189's corresponding
`diagnostics/hys34-full361-count-delta-location-audit.json`.
Evidence is under HYSPEC's `diagnostics/hys{34,70}-v011603-vs-shiver-{6,12}.json`
and `hys{34,70}-v011604-pooling-vs-timeweighted-{6,12}.json`.

Fresh twelve-copy resident control evidence is
`mantid/hys{34,70}-pilot12-resident12-reference-01/receipt.json` and
`diagnostics/hys{34,70}-v011604-vs-resident-shiver-12.json`. The metadata-only
round-trip probe and scalar replay receipts stay in the same HYSPEC IPTS
benchmark folder. Its original scientific inputs are unchanged.
