# DGS reference and current native measurements

Measured on 2026-10-01 on `analysis-node03.sns.gov`. The
[aggregate JSON receipts](dgs-reference-current.json) retain numerical results
and source hashes. The original NiO project was never saved; its size and
modification time were checked unchanged.

## Timings

| Operation | nfit | Mantid/Shiver | Scope |
|---|---:|---:|---|
| Shared 617-run MDE → thin 3bar histogram | 71.88–72.94 s | 118.14–160.26 s | Two runs each; native streams events, Mantid starts with resident MDE. |
| Same MDE → full 3bar cube | 188.78 s (3.15 min) | 986.43 s (16.44 min) | One full cube each; **5.23×** wall-time ratio. |
| Final fresh raw 617 runs → thin 3bar histogram | 306.85 s (5.11 min) | — | Independent fresh validation pass, no reduced cache. |
| Final fresh raw 617 runs → full 3bar cube | **418.88 s (6.98 min)** | — | No reduced-event cache reads or writes; includes reduction, binning, normalization and first-dispatch JIT. |
| Three raw runs → thin 3bar, first / warm fresh dispatch | 14.62 / 5.00 s | — | Preliminary monitor arithmetic; first includes JIT and shared geometry setup. |
| Same three runs, cache construction / true cache hit | 4.91 / 0.146 s | — | Preliminary reduction version 2; misses 3 then hits 3, no project save. |
| Three raw runs, load/filter/reduce/convert to MDE | — | 44.06 s total | Sequential single job; diagnostic extraction/export excluded. |
| Standard 617-run raw reduction estimate | — | ~151 min; illustrative 143–157 min range | Three-run extrapolation; final merge/save additional. |

The three selected runs (392985, 393089, 393387) span different run sizes but
are not an unbiased random sample. The extrapolated range is not a confidence
interval or a measured full reduction. Shared geometry and vanadium setup are
not multiplied by 617. Current cache version 3 invalidates the preliminary
reduced caches; their timings describe the tested mechanism, not a new final
version 3 cache benchmark.

Mantid MDE loading took 151.94 s and shared vanadium loading 3.56 s. Native MDE
metadata loading took 4.83 s; streamed event reading is included in its binning
times. The estimated ordinary sequential Mantid/Shiver reduction plus measured
full-cube binning is approximately **167.5 min**, with final merge/save
additional. Reopening a saved MDE adds the measured 2.59 min load/setup cost;
that reload is unnecessary when binning the merged in-memory workspace. These
pipeline totals are estimates.

For the final native raw cube, the reduction/event accumulation and trajectory
setup phase lasted 321.53 s; normalization and finalization lasted 97.35 s.
The earlier full raw pass took 479.09 s, before the monitor fix. Uncontrolled
page caches and first-dispatch work prevent interpreting that change as a
measured speedup from the calibration correction.

### Environment and limits

Experiments ran sequentially, one process per engine, allowing up to 64 internal threads
and no external parallel jobs. The host has two AMD EPYC 7343 processors,
32 physical / 64 logical cores. Native resource limits were 64 CPUs and
200000 MiB; OpenBLAS used 1 thread and OpenMP 64. Mantid manual raw runs
recorded OpenBLAS 64 and OpenMP 64. No node-wide CPU policy was changed.

The existing nfit frozen runtime supplied Python 3.14.7, NumPy 2.5.3 and
Numba 0.67.0. Existing Mantid 6.16.0.1 used Python 3.12.13 and NumPy 2.1.3.
OS page caches were not flushed. Native full-MDE binning used compiled kernels
warmed by the subset experiment; it still reads the source MDE on every call.
No runtime was installed in the data directory. Peak RSS is a process-lifetime
high-water mark: native MDE cube 38.12 GiB, Mantid cube 41.18 GiB, final native
fresh raw cube 41.18 GiB. Historical node23 timings used a different CPU.

## Matched inputs and grids

Runs 392985–393469 and 393500–393631 contribute 617 experiments and
177778751 source MDE events. Both engines used `van382236_D` as normalization
and mask, the same UB, discrete event assignment, and the six 3bar operations.
Momentum axes were (1,1,1), (1,−1,0), (1,1,−2), followed by energy.

| Grid | Shape | First edges | Steps |
|---|---|---|---|
| Thin | 21 × 3 × 3 × 101 | −0.025, −0.035, 0.29, −0.25 | 0.03, 0.03, 0.02 r.l.u.; 0.5 meV |
| Cube | 67 × 134 × 101 × 101 | −1.015, −2.015, −1.01, −0.25 | 0.03, 0.03, 0.02 r.l.u.; 0.5 meV |

These are the saved centre-grid edges, including half-bin offsets. Mantid
stores axis limits as float32; full-grid endpoint differences reach
1.14 × 10⁻⁷ r.l.u.

### Binning the same saved MDE

Thin pooled H/E counts and weighted numerators agree exactly. Four fine cells
redistribute one event at 1 and 29 meV and cancel under hidden K/L pooling;
maximum fine-cell count difference is 1. Fine-cell equality is not exact
everywhere. No coverage disagreement or events in unexposed cells occurs.
Fringe, low-exposure and selected-ROI fine-cell counts/numerators agree exactly;
variance agrees to roundoff. Whole-thin normalization relative L2 difference
is 1.73 × 10⁻⁶.

Full-cube counts are 352458271 native versus 352458245 Mantid (26 difference).
Native corrects 386440 same-source symmetry-copy pairs landing in the same bin,
adding 1022278.65 to count-space variance. Subtracting that aggregate correction
gives native diagonal variance sum 554570344.29 versus Mantid 554570221.98.
A scalar subtraction does not establish per-bin variance parity.

875289452 cross-bin copy pairs remain unrepresented. Their covariance can
matter when later integrating bins; copied events are not independent
measurements. Primary native variance includes the within-bin correction;
Mantid MDNorm uses the independent-copy diagonal convention.

## Raw conversion and calibration audit

### Preliminary representative parity

Before the monitor stability change, fresh representative runs 392985, 393089,
393387 matched Mantid event counts 25569, 7878, 421781 and detector-ID sequences
exactly. Mapped pulse-retained event counts agreed exactly; all-bank file totals
also contain unmapped detector IDs and are not comparable to Mantid workspace
counts. Geometry IDs agreed and positions differed by at most 8.9 × 10⁻¹⁶ m.
Energy agreed within 6.5 × 10⁻¹³ meV; corrected weights and variances agreed
at about 10⁻⁷ relative L2, consistent with float32 weighted-event storage.
Charge and angles matched roundoff. Saved MDE extrema clipping removed 1/2/1
boundary events from these runs, outside the 0–50 meV cube.

These results validate the earlier converter path, **not complete final
converter parity**. Final stable monitor arithmetic also changes run 393089.

### Monitor failure found by the full-dataset audit

The initial full fresh raw cube contained 21506 more events than native
binning of the shared MDE. An audit of all 617 runs found four native Ei=60 meV,
T0=0 µs fallbacks and one smaller calibration difference. Manual fresh Mantid
GetEi matched saved MDE logs for all five problem runs.

The derivative-error formula subtracted nearly equal positive terms, producing
negative roundoff in empty monitor tails. Python raised a square-root exception
that previously triggered a silent fallback. The correction propagates the
centred derivative's shared central measurement through a nonnegative squared
coefficient. This uses mathematically correct variance arithmetic rather than
reproducing NaN-dependent stopping behavior in the expanded Mantid expression.

Reduction cache version 3 invalidates earlier raw reductions. Imported run
metadata and histogram provenance record calibration source and warning.
Missing monitor data retains the compatible requested-energy fallback; failure
to fit available measured monitors warns explicitly, including coincident
flight times.

### Final full-dataset residual

All 617 final runs used monitor calibration with no fallback warnings.
**26/617** final Ei/T0 estimates differ from saved Mantid logs. Maximum absolute
differences are **0.0110355 meV** and **0.8270435 µs**. These remaining peak-tail
differences are a pending reference-parity item.

The final raw cube has **352457563 events**, 708 fewer than native binning of
the shared MDE (about 2.0 ppm). Weighted numerator differs by −841.85 and
primary variance by −1006.50; normalization agrees to roundoff. This residual
is separate from the 26-event discrepancy between engines binning the same
saved MDE. Global agreement does not establish local fringe agreement or
complete covariance propagation.

### Final five-run comparison

Fresh manual Mantid and final native reduction have identical accepted counts
and detector-ID sequences for all five problem runs:

| Run | Accepted events | Maximum energy difference | Weight / variance relative L2 differences |
|---|---:|---:|---:|
| 393069 | 36399 | 2.8 × 10⁻¹³ meV | 8.1 × 10⁻⁸ / 1.3 × 10⁻⁷ |
| 393085 | 50831 | 0.00368 meV | 2.0 × 10⁻⁵ / 3.8 × 10⁻⁵ |
| 393197 | 90111 | 0.00283 meV | 2.0 × 10⁻⁵ / 4.2 × 10⁻⁵ |
| 392991 | 29652 | 4.1 × 10⁻¹³ meV | 7.3 × 10⁻⁸ / 1.2 × 10⁻⁷ |
| 393083 | 83343 | 0.00308 meV | 1.7 × 10⁻⁵ / 3.4 × 10⁻⁵ |

393069 and 392991 match Ei/T0 to roundoff after fixing the fallback. The other
three retain small calibrated peak-tail differences. These tests confirm the
identified failure and its correction without claiming all-run parity.

### Final fresh raw fringe comparison

A separate fresh 617-run thin reduction took 306.85 s with no cache reads or
writes and no project save. It produced 86498 events versus Mantid 86499.
Only four fine cells differ by one event; pooled H/E counts differ by at most
one. Coverage matches for all 9700 covered cells, with no events in unexposed
cells and no empty-cell variance floor.

| Region | Fine-cell count differences | Numerator relative L2 | Variance relative L2 | Standard-error ratio range at common exposure |
|---|---:|---:|---:|---:|
| Geometric fringe | 0 | 3.33 × 10⁻⁷ | 6.76 × 10⁻⁷ | 0.99999948–1.00000213 |
| Bottom exposure decile | 0 | 7.72 × 10⁻⁸ | 1.59 × 10⁻⁷ | 0.99999982–1.00000003 |
| Selected ROI | 0 | 3.09 × 10⁻⁷ | 6.13 × 10⁻⁷ | 0.99999945–1.00000059 |

The fringe comprises pixels within two 8-neighbor steps of actual zero-exposure
pixels, excluding crop-window exterior. Low exposure is the bottom decile in
each energy row above 3 meV. The ROI is H=0.13–0.23 r.l.u., E=3.5–25 meV,
with hidden K/L bins collapsed. These definitions match the earlier diagnostic.

Whole-slab fine-cell numerator/variance relative L2 differences are
0.00523/0.00564, localized to interior redistribution. Map standard-error
ratios range 0.99868–1.00108. The local fringe and low-exposure agreement is
therefore stronger than a claim based only on global counts. Remaining
calibration and boundary discrepancies still require review; this thin wedge
has no symmetry-copy collisions and does not establish cross-bin covariance.

## Current thin uncertainty

Let C be summed corrected event weights, N the exposure/trajectory denominator,
and V count-space variance. Intensity I=C/N and standard error √V/N carry the
chosen intensity units. Covered empty cells have observed V=0; this is distinct
from a confidence interval for an unknown Poisson rate. Uncovered cells are
masked.

The same-MDE supplied-slab diagnostic uses `--event-statistics` with auxiliary
normalization, holding exposure fixed while substituting Mantid raw variance.
Fringe empty-cell fraction is 41.1%, low-exposure 90.0%, selected ROI 56.5%.
None adds an empty-cell error floor. Map standard errors agree to roundoff in
these regions. Fringe exposure ratios have p95=1.000032 and max=1.000733.
This thin wedge has no copy collisions and does not validate cross-bin
covariance after integration.

Current/reference standard errors agree through hidden-axis pooling and
additive display coarsening. Equal variance-prescription ratios for regular or
rotated cuts do **not** establish the estimator is appropriate. Ordinary
inverse-variance cut errors divided by count/exposure pooled errors range
0.630–5.478, median 0.984. These cuts exclude zero-error cells and target a
different quantity. ROI sums of normalized pixel intensities are not pooled
count rates or integrals with bin-width factors. Cut estimators and cross-bin
covariance remain separate review gates.

## Reproduction and provenance

Manual engine scripts and full diagnostic slabs remain under
`/SNS/users/paulneves/.cache/nfit-converter-1d`. The repository contains aggregate
statistics and source checksums only. Existing nfit uses `--run-script` with its
scientific package completely reloaded from the temporary source snapshot.
Critical module paths/hashes are checked, including `_mdevent_numba`. Mantid
scripts use the existing Shiver pixi Python. Run engines sequentially to avoid
contention. JSON records exact grids, timings, phase transitions, caches,
versions, module hashes and slab/receipt hashes.

The base is commit `72cf2b6` (nfit 0.106.0). The final raw pass includes the
monitor stability/provenance patch; unchanged MDE measurements use the base
science. The snapshot manifest predates that patch; recorded critical-module
hashes identify measured source. A coincident-flight-time guard was added after
the final cube process started and does not affect any successful fit here.
No Mantid/Shiver import or invocation was added to nfit or its unit tests;
reference engine measurements ran manually on the cluster.
