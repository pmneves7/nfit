# Ordered DGS event projection and accumulation

The optimized path uses the existing Mantid affine coefficients and performs
projection, uniform-bin assignment and accumulation in one compiled event pass.
It preserves float32 conversion, individual multiply/add rounding, upper-edge
exclusion, source-event order, symmetry-copy order and chunk order. Fast-math is
disabled. It removes the full projected-coordinate arrays and their multiply/add
temporaries. Nonuniform grids, high-precision projection and unavailable Numba
retain the existing projector and accumulator.

The measured kernel updates weighted event numerator C, its accumulated variance
V, and dimensionless event-contribution counts. Optional same-event copy
covariance still uses the exact assignments from that pass and the established
correction service. Detector trajectories and their exposure N are unchanged.

## Bounded real-event measurement

On the local nfit interpreter, the first 600,000 events from
`Ei15meV_50K_240Hz_s2_70.nxs` were read without changing the file. The real saved
UB matrix and event weights/variances were used. Each timing is a median of five
alternating reference/candidate repetitions after warming the actual strided
array layout. BLAS and OpenMP were limited to one thread. The scalar receipt
`dgs-kernel-profile-candidate.json` records source hashes, script hash, grids,
timings and literal equality assertions. No scientific arrays were exported.

| Grid | Symmetry copies | Reference event kernel | Fused event kernel | Kernel speedup |
| --- | ---: | ---: | ---: | ---: |
| 80 × 80 × 1 × 57 slab | 1 | 0.02537 s | 0.001429 s | 17.75× |
| Slab | 6 | 0.12542 s | 0.005691 s | 22.04× |
| Slab | 12 | 0.26071 s | 0.011817 s | 22.06× |
| 40 × 50 × 25 × 57 cube | 1 | 0.02366 s | 0.001387 s | 17.06× |
| Cube | 6 | 0.12326 s | 0.005870 s | 21.00× |
| Cube | 12 | 0.25513 s | 0.011946 s | 21.36× |

These are **kernel timings**, excluding event reads, trajectory normalization,
archive/project preparation, histogram allocation and covariance correction.
The contiguous source subset has low acceptance in these grids: the identity
slab accepts 1,887 contributions and the identity cube accepts 12,304. Early
rejection consequently contributes to the large speedup. This is not an estimate
of whole-project speedup or other instruments' throughput.

Within this workload, projection took approximately 85% of the default event
kernel time. The optional same-event covariance correction took approximately
0.008–0.009 s for six copies and 0.026–0.027 s for twelve copies; it is unchanged.
All C, V, counts, bin assignments and covariance summaries agreed literally in
every measured repetition. Complete replay timing, including reads and
normalization, is measured separately by `profile_dgs_workflow.py`.

## Complete real-source histogram

The same local interpreter processed the complete 49,415,626-event 50 K / 70°
source and all 361 sample runs. Baseline source was frozen from commit
`5ddd972` (0.114.5); candidate runs used the fused dispatcher. Each ran in a
fresh process with eight normalization workers, the actual UB, and the same
80 × 20 × 8 × 48 grid in H,H,0; 0,0,L; H,−H,0; and energy. There were no
concurrent benchmark jobs. Both repeats streamed all events; the second reused
metadata/geometry and operating-system caches. Timings exclude source inspection,
Python import, archive persistence, and raw reconstruction.

| Symmetry copies | Baseline first call | Fused first call | Baseline repeated call | Fused repeated call | Repeated-call speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.386 s | 5.552 s | 6.001 s | 4.169 s | 1.44× |
| 6 | 18.241 s | 6.607 s | 16.485 s | 5.225 s | 3.16× |

In the six-copy repeated call, the event phase, including file reads and filtering,
fell from 14.077 to 2.863 s. Normalization setup remained approximately 1.6 s and
integration 0.75 s. The identity case spends 2.316 s in its remaining event phase,
1.636 s in normalization setup, and 0.194 s integrating trajectories. Peak resident
memory for the six-copy process fell from approximately 1499 to 1254 MiB.

The baseline and candidate receipts have identical full-array hashes for C, V,
event counts and masks. A further candidate/reference-dispatch comparison checked
every cell. Exposure, signal and uncertainty differ by at most 1.15 × 10⁻¹⁴
relative on nonzero cells, and 1.28 × 10⁻¹⁵ in the lowest-exposure decile. The
unchanged parallel normalization already has last-bit differences between its
own repeated calls. This optimization changes neither that algorithm nor its
coverage support. Original source size and modification time were unchanged.

Receipts are `hyspec-workflow-baseline-0.114.5.json`,
`hyspec-workflow-candidate.json`, and the corresponding `identity` receipts.
These grids are bounded compared with the project's full cached volume; the
results are not a full-project rebuild benchmark. Current SEQUOIA/raw-reduction
and matched Mantid measurements require the ORNL experiment mount, which was
unavailable on node23 during this validation. Older timings in
`sequoia-performance.md` do not establish a current-build regression or speedup.

After adding the exact-text mask parser cache described below, clean complete
repeats took **2.899 s without symmetry** and **3.913 s with six copies**.
Against 0.114.5's 6.001 and 16.485 s this is **2.07× and 4.21×** respectively.
First calls took 4.612 and 5.455 s, including normalization kernel startup.
Normalization setup is now 0.332–0.342 s. The final receipts are
`hyspec-workflow-optimized-identity.json` and `hyspec-workflow-optimized.json`.
C/V/count/mask hashes still match the frozen baseline. Every-cell comparisons
with reference event dispatch passed, including the lowest-exposure decile.

## Repeated normalization setup

The remaining setup cost came primarily from repeatedly interpreting the saved
detector mask parameter map. All 361 runs in this source contain the same exact
100,866-byte serialized map. A metadata-only profile found 924,160 boolean mask
records parsed per setup call, accounting for approximately 90% of its profiled
time. Geometry and reduction settings still need to be checked for each run.

The saved-mask reader now reuses only the pure text interpretation, keyed by
the complete serialized text. Its least-recently-used cache holds at most 16
entries and admits text of at most 1 MiB encoded size per entry. Cached values
are immutable tuples; public results remain fresh read-only integer arrays.
Every call still reads the source map, so edited masks or different instrument
maps naturally miss. Detector coordinates, calibration, charge, UB, goniometer,
energy limits and current reduction settings are rebuilt through their existing
paths. Large maps are parsed without retention.

Five alternating cached/uncached metadata setup calls on the same real 361-run
source measured median **1.646 s → 0.337 s**, a **4.89× setup speedup**, saving
1.31 s per new histogram. Every detector and run payload, including ordering and
dtype, agreed literally. The original source was unchanged. These timings
exclude events and trajectory integration; the combined event-kernel plus
mask-cache workflow is timed separately. Source inspection with the cache took
0.821 s and required one parse followed by 360 cache hits. The reusable script
`profile_dgs_normalization_setup.py` and scalar receipt
`dgs-normalization-setup-candidate.json` record the comparison and provenance.

## Correctness scope

Focused tests cover contiguous, Fortran, reversed and sliced inputs; unequal and
signed weights; explicit variances and weight-squared variances; enabled-event
masks; streamed chunks; exact/nextafter boundaries; nonfinite coordinates;
one/six/twelve symmetry copies; both supported copy-variance treatments; and
established high-precision/nonuniform/unavailable-backend fallbacks. The
prepared affine remains owned by `dgs_reduction_policy.py`; the focused service
imports no reduction facade or GUI. Input arrays and configuration are not
modified.
