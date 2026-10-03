# Representative measurement acceptance

Stage 6A survey on the nfit 0.114.0 scientific build (`a6f10f8`), with a
metadata-only CORELLI provenance disclosure added for 0.114.1. The survey found
adapter gaps, so full cross-instrument acceptance remains incomplete. No
scientific default or original project was changed. Unit tests use synthetic
references and never import or run Mantid/Shiver.

## Scope and evidence

| Measurement | Check | Result and boundary |
|---|---|---|
| SEQUOIA DGS | Current native binning against stored Mantid output; fresh raw subset | Matched inputs and conditional diagonal uncertainty; see the SEQUOIA receipt |
| HYSPEC reduced DGS | Two real logical runs, event statistics, final energy cuts and source replay | Downstream checks pass; raw conversion needs dynamic geometry |
| CORELLI | Compiled/Python reconstruction, native recipe replay, original full run | Numerical checks pass; physical final-cut covariance remains incomplete |
| MACS SPEC and DIFF | Original counts/denominator, native replay, discrete/fractional deposition, final cuts | Direct pooling passes; native histogram/profile target propagation fails |
| Reduced CW diffraction | Synthetic HB2A table import, angle/q preparation and supplied errors | Analytic checks pass; real CW example is MACS DIFF |
| Susceptibility versus temperature | Analytic targets and repeated acquisitions with Gaussian node errors | Declared common-response/coordinate targets and represented dependencies pass |

Manual validators are `benchmarks/validate_sequoia_measurements.py`,
`validate_local_instruments.py`, `validate_corelli_measurements.py` and
`validate_hyspec_raw.py`. Scalar JSON receipts accompany this report. Large
scientific products stay in IPTS directories on ORNL. Local projects and remote
originals are read-only; receipts retain before/after size and modification-time
checks. Timings below are acceptance-run timings with uncontrolled JIT and OS
caches; they are not Mantid performance comparisons.

## SEQUOIA

The current native MDE binner uses all 617 runs, the same UB, processed vanadium
and mask, the twelve 3barm operations, and the stored Mantid comparison grid:
61×7×5×61 cells with 0.01 r.l.u. momentum steps and 0.5 meV energy steps.
Count numerators, event variances and event counts match the stored reference
exactly. Exposure support matches, with exposure differences within the
established relative tolerance of $3\times10^{-8}$; the observed maximum is
$1.002\times10^{-8}$ (about 0.010 ppm). The checks include 10,563 cells in the
lowest-exposure decile and 15,372 cells beside coverage boundaries. Both subsets
have maximum relative exposure difference $4.75\times10^{-10}$.

The viewer integrates the K,-K,0 and L,L,-2L axes by pooling the retained
numerator, variance and exposure. A box cut at H,H,H = 0.13–0.23 r.l.u. and
energy transfer 3.5–25 meV verifies the same statistics in the low-coverage
region. Fresh raw runs 392985, 393089 and 393387 are compared with those same
logical runs from the authoritative merged MDE: count numerators, variances and
event counts are exact, with maximum relative exposure difference
$6.86\times10^{-10}$. Reduction-recipe replay is exact for those three event
fields; exposure differs by at most $4.25\times10^{-16}$ relatively.

These comparisons validate the recorded Mantid-compatible diagonal convention.
They do not establish full symmetry/cross-bin covariance or calibration
uncertainty. No current Mantid process is used, and the two main NiO projects
are unchanged. The [SEQUOIA receipt](measurement-acceptance-sequoia.json) records
the differences and source hashes.

## Continuous measurements and reduced CW tables

The synthetic susceptibility is $\chi(T)=2+0.3T$ in emu/mol with temperature
$T$ in K, deliberately clustered observations and known Gaussian variances.
Its coordinate mean over 0–20 K is 5 emu/mol. Uniform and precision-weighted
observation means are approximately 2.553 and 2.464 emu/mol: they estimate
different targets. Coordinate weights, not observation density, decide an
interval mean.

For 2,048 repeated acquisitions, the coordinate estimate's expected variance is
0.014336 (emu/mol)² and empirical variance is 0.014929; known-variance Gaussian
95% coverage is 94.29%. A separate common-response experiment gives expected
variance 0.0075294 (emu/mol)², empirical 0.0074858, and coverage 94.87%.
The tests bound Monte Carlo sampling error; these are not empirical uncertainty
calibration claims for private susceptibility data.

Shared boundary nodes retain their covariance across neighboring coordinate
intervals. Shared-background final variance is 0.26875 (emu/mol)² versus
0.08125 for separately measured independent backgrounds; direct, staged and
final-region estimates agree with primitive coefficients. Masked coordinate
nodes remove adjacent interpolation support. Repeated temperatures require an
explicit common-response combination before coordinate interpolation.

Supplied node variances do not include interpolation-model error. A controlled
quadratic example with a large sampling gap demonstrates that a piecewise-linear
estimate can be badly biased despite tiny node errors. This is an explicit
model limitation, not improved precision from more densely sampling another
part of the interval.

Synthetic HB2A tables are already reduced intensity/uncertainty observations,
not raw counts. Uniform and precision means, measured zeros with supplied
positive uncertainty, repeats, masked fringes and final cuts preserve that
distinction. A separate explicitly declared count example verifies covered-zero
exposure pooling and exact final-bin Poisson bounds. The current public
point-list preparation facade is fit-oriented and requires positive supplied
uncertainty; it is not a general raw CW reduction API.

## MACS: a native final-profile target gap

The real 900-point source `Ef3p7_et_0.1_261.nxs.ng0` exercises SPEC and DIFF.
DIFF supplies the requested real CW diffraction example, under its recorded
elastic coordinate approximation. The native importer reconstructs raw counts
and its legacy count-space variance `max(counts, 1)` from the stored intensity,
uncertainty and monitor/efficiency denominator. Direct one-bin native pooling,
portable recipe replay and fractional count/exposure conservation pass.

However, native MACS points have no explicit measurement contract or retained
C,V,N payload. Here C is the raw count numerator, V its declared variance and
N the monitor/efficiency exposure, all in the experiment's arbitrary
normalization convention. Native point histograms discard N. A later legacy
precision profile therefore changes the target:

| Stream | Direct exposure pool, intensity ± standard uncertainty | Native fine histogram → legacy final profile |
|---|---:|---:|
| SPEC | 231312.7485 ± 73.5583 | 28665.0902 ± 25.8946 |
| DIFF | 660081.6889 ± 121.1135 | 90014.2862 ± 44.7249 |

Values retain the source's monitor-target intensity units. These differences
are not roundoff or evidence that smaller error bars are better. The explicit
raw-count contract constructed only for validation pools C,V,N correctly
through final cuts. The adapter must retain that model/target before native
profile acceptance can pass. Monitor/efficiency uncertainty and the legacy
zero-count floor require separate treatment; defaults were not changed.

## HYSPEC

The reduced QSample source `Ei15meV_50K_240Hz_s2_70.nxs` contains 361 logical
runs. The first two contribute 855,993 accepted MDE events. Independent source
reads reproduce numerator and variance sums exactly. A 120×120×1×40 histogram
and direct source replay on the final energy grid agree: maximum signal
difference is $7.70\times10^{-11}$ and variance difference
$8.41\times10^{-17}$ in their respective source units. There are 1,376 exposed
cells, 410 covered-zero cells and 574,624 unexposed cells. Of 138 cells in the
lowest exposure decile, 134 have zero counts. No empty-cell variance floor is
added. QLab background input correctly rejects HKLE interpretation.

Identity/inversion tests reproduce the saved independent-copy convention;
they do not establish physical copy covariance. Native exposure is conditional
on known calibration. Raw runs 506277 and 506278 are also available on ORNL.
Their instrument definition uses log-dependent moderator distance and detector
tank rotation, which the current static raw geometry parser does not resolve.
The inspected source distance is zero instead of 40.783 m, so the raw converter
rejects the runs. Static detector directions differ from the stored physical
directions by 69.30–69.99°. The HYSPEC T0 formula also uses `^3` while the safe
evaluator accepts `**`; failed evaluation silently returns 0 instead of the
stored 100.111590 µs. Raw-converter acceptance therefore requires geometry and
timing fixes. The [raw receipt](measurement-acceptance-hyspec-raw.json) records
source Ei/T0 and geometry evidence. Goniometer omega agrees for these two runs.

## CORELLI

The 315,392-event first-4096-events-per-bank probe actually exercises compiled
dispatch. Discrete, fractional momentum assignment, inversion symmetry and
portable recipe replay agree with the independent Python implementation.
Maximum signal differences are of order $10^{-13}$ in source units and masks
are identical. The original run 403634 contains 6,713,729 events; its warmed
8³×3 reconstruction takes 7.89 s. This checks only three requested energy
hypotheses, not production-volume throughput. Timing offset is 18,000 ns,
retained charge is 63.28769 µAh and duty cycle is 0.498475.

Physical uncertainty acceptance fails for shared-event operations. Two
overlapping inversion copies double signal and report twice the single-copy
variance; shared-source arithmetic requires four times that variance. Energy
hypotheses also reuse measured neutrons. In an independent 200,000-acquisition
reference, a final reconstructed sum has true variance 34 versus naive diagonal
20.1; a cancellation example has true variance 3 versus diagonal 20.1.
Missing covariance can increase or decrease the correct uncertainty.

CORELLI currently retains no complete native primitive payload for these cuts.
An explicit shared-source region request rejects missing dependencies, but the
legacy viewer/profile fallback can still use diagonal Gaussian errors.
Statistics and provenance now exposes the recorded covariance and normalization
limits. Charge/duty and pointwise solid-angle/flux correction are not a full
four-dimensional MDNorm trajectory denominator. Finite-energy reconstruction is
not established as equivalent to Mantid's elastic correlation output.

## Additional source formats and follow-ups

`measurement-source-inventory.json` records the user-supplied DMC, D33, SANS-I,
GP-SANS and WAND² examples. DMC/D33/SANS-I have no registered native importer;
raw GP-SANS and WAND² do not have native reducers. Event-bank layout alone does
not identify a DGS acquisition. Reduced WAND² histogram import is available.
New raw adapters need their own geometry, normalization and uncertainty contracts.
WAND² is CW diffraction, CORELLI uses correlation-chopper reconstruction, and
MACS uses measured-point normalization. None should use the DGS converter.
The current raw-DGS event-bank probe is too broad; source classification needs
to enforce these boundaries before new event-format adapters are added.

The plan splits unresolved acceptance into 6A-R (dynamic DGS geometry and
HYSPEC), 6A-M (MACS statistical payloads), and 6A-C (CORELLI shared-event
dependencies). Optional numerical/statistical alternatives remain 6A1, and
scientific default adoption remains 6A2. No universal parity/accuracy claim is
made by this survey.
