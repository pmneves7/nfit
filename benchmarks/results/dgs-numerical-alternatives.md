# Optional DGS numerical algorithms: independent truth checks

The manual
[`validation script`](../validate_dgs_numerical_alternatives.py) and
[`scalar receipt`](dgs-numerical-alternatives.json) compare existing optional
paths with independent numerical references and controlled calibration truth.
No production/default changes, instrument files, original projects, cluster
jobs, or Mantid are involved. Source hashes and the fixed seed are recorded.

An 80-digit Decimal reference treats the supplied binary64 values as exact.
It solves coordinate transforms by independent elimination and intersects each
trajectory with each voxel analytically. This establishes numerical fidelity
to fixed inputs; it does **not** establish that those inputs describe an
instrument more accurately. Shared kinematic constants are held fixed to
isolate arithmetic, rather than claiming an independent physical calibration.
Units and coordinate conventions follow
[`physics_conventions.md`](../../docs/physics_conventions.md).

## Coordinate projection

Momentum in Å⁻¹ is transformed using a supplied orientation matrix UB, the
`(1,1,1)`, `(1,-1,0)`, `(1,1,-2)` axes in reciprocal-lattice units (r.l.u.),
a cyclic symmetry operation, and energy in meV. The independent reference
solves the defining linear equations; it does not call production inversion,
affine construction or bin-index helpers.

| Controlled sample | High-precision bin differences from requested grid | Compatibility bin differences from requested grid |
| --- | ---: | ---: |
| 400 ordinary interior events | 0 | 0 |
| 400 events deliberately 10⁻⁹–10⁻⁶ r.l.u. from internal boundaries | 0 | 115 |

High precision agrees with the Decimal projection within 5.6 × 10⁻¹⁶ r.l.u.
Compatibility projection differs by up to 2.91 × 10⁻⁶ r.l.u. and
1.90 × 10⁻⁶ meV in these cases. Compatibility includes six-significant-digit
serialization and float32 transforms; its effective rounded conventions can
differ from the original requested coordinates/grid.

The boundary challenge deliberately magnifies differences and is not an
estimate of their frequency in real data. Ordinary-event membership is unchanged.
These displacements are often far smaller than instrument/UB calibration
tolerances. Higher precision cannot restore precision already lost in a saved
float32 MDE or make geometry, energy calibration or resolution exact. Tiny
observed-extrema clipping remains accepted compatibility rounding; this script
tests internal boundaries and does not emulate that clipping.

## Trajectory integration and coverage fringes

For a single fixed detector, the oracle independently bounds the final momentum
`k_f` interval admitted by each momentum/energy voxel and integrates final
energy. Exposure units here are meV times a fixed charge/detector weight; the
weight is one. Each policy is checked against its **own effective edges**, so
midpoint assignment errors are distinguished from changed grid definitions.
Both the scalar and compiled production trajectory paths are checked.

- On an ordinary grid over 0–40 meV, both policies agree with the oracle to
  about 2 × 10⁻¹⁴ absolute exposure and have identical covered-bin membership.
- A controlled near-coincidence between a momentum boundary and a 20-meV
  energy boundary leaves a nonzero exposure sliver of 10⁻⁷ meV-weight. Float32
  midpoint assignment loses that voxel's exposure, assigning it to its neighbor.
  High precision preserves the covered fringe with about 5 × 10⁻¹⁵ absolute
  error. Whole-window exposure remains 2 under both policies: the sliver is only
  5 × 10⁻⁸ of the total. Its local relative error is nevertheless 100%.
- An artificial 10⁻⁸-Å⁻¹-wide grid cannot be represented by the compatibility
  float32 edges and correctly raises the existing precision guard. High precision
  preserves its coverage. Residual relative exposure error reaches about
  1.5 × 10⁻⁸, demonstrating that binary64 integration is not arbitrary precision.

These examples establish a numerical benefit at representable internal fringes,
not an accuracy claim for physically uncertain detector trajectories. They
do not validate continuous sample rotation, uncertainty in geometry/Ei/UB,
or every alternate raw correction bundled with a precision policy. In particular,
the full optional raw-converter path's He-3 geometry/efficiency assumptions need
separate validation. Performance must also be measured independently before
choosing a general default.

## Stable monitor derivative variance

For independent histogram observations `y_-`, `y_0`, `y_+` with errors `s`,
the derivative is
`0.5[(y_+ - y_0)/f + (y_0 - y_-)/b]`, where `b` and `f` are adjacent spacings
in µs. Direct differentiation gives coefficients
`[-1/(2b), (f-b)/(2fb), 1/(2f)]`; variance is their squared, error-weighted sum.
The Decimal reference uses these coefficients directly.

The stable path's sum of squares avoids the expanded reference subtraction
becoming negative by roundoff. It matches the independent derivative variance
for the ordinary unequal-spacing and uniform-spacing cases. In the existing
sparse-tail cancellation case, compatibility produces NaN whereas stable stays
finite and nonnegative. This is a demonstrated arithmetic improvement.

The stable expression still subtracts two reciprocals for the central
coefficient. In that extreme case the exact sigma is 1.94939 × 10⁻¹⁵ in
histogram-ordinate units per µs; stable returns 2.06368 × 10⁻¹⁵, a 5.86% relative
but 1.14 × 10⁻¹⁶ absolute discrepancy. No production change is warranted by this
tiny example alone. Nor does a nonnegative derivative variance certify the
independence/noise model of fractionally rebinned monitor observations.

## Controlled Ei/T0 calibration

The two-monitor simulation uses distances 18 and 20 m, true incident energy
`Ei = 60 meV`, true source timing offset `T0 = 15 µs`, and 2,000 events per
monitor, for 120 independent realizations. Known mean flight times obey the
kinematic relation used by the calibration. This targets the two-monitor branch;
HYSPEC currently resolves an instrument T0 formula before that branch, so this
does not establish better HYSPEC calibration.
The native peak routine is exercised directly, with Ei/T0 independently solved
from its two peak estimates; raw-file monitor discovery and IDF interpretation
are outside this controlled truth check.

| Pulse model / treatment | Ei RMSE, meV | T0 RMSE, µs |
| --- | ---: | ---: |
| Symmetric Gaussian / compatibility | 0.19487 | 9.11556 |
| Symmetric Gaussian / stable | 0.19495 | 9.11991 |
| Symmetric Gaussian / known-population sample mean | 0.14324 | 6.68468 |
| Unequal asymmetric tails / compatibility | 1.59258 | 68.12366 |
| Unequal asymmetric tails / stable | 1.59258 | 68.12366 |
| Unequal asymmetric tails / known-population sample mean | 0.15560 | 7.00666 |

The stable and compatibility calibrations are bit-identical in 119/120 Gaussian
realizations and all asymmetric realizations. The single changed Gaussian case
does not improve squared error; its paired error difference is only one Monte
Carlo standard error. Thus these controlled data do not demonstrate better
Ei/T0 calibration from the stable variance expression.

For the asymmetric model, each Gaussian-plus-exponential pulse is shifted to
retain its known mean flight time, but monitor shapes differ. Both peak-region
recipes then have the same large bias: approximately +1.57047 meV in Ei and
+67.15414 µs in T0. Stable derivative arithmetic does not correct peak-region
truncation or define the physically appropriate arrival-time reference.
The sample-mean comparator knows that every simulated event belongs to the
signal pulse and that the calibrated reference is its full mean; it is not a
proposed general importer for monitors with background or multiple pulses.

The peak code does not return Ei/T0 confidence intervals. Empirical RMSE and
Monte Carlo standard errors above therefore do not certify reported calibration
uncertainties or coverage.

## Decisions supported by this evidence

Keep both optional paths available and retain current defaults. Higher precision
improves fidelity to specified coordinates and can retain extremely small
internal fringe segments; its typical physical significance and full-converter
cost/assumptions are not certified here. Stable variance avoids invalid arithmetic
but does not establish unbiased Ei/T0 or better HYSPEC calibration.

Before adopting a new calibration default, obtain actual monitor pulse shapes,
background/multiple-pulse identification, reliable source-to-monitor distances,
an instrument-specific definition/reference of the calibrated arrival time, and
independent Ei/T0 calibration measurements with uncertainties. Validate monitor
rebinning/dependence assumptions and jointly propagate calibration uncertainty
to final intensity/coverage fringes. Repeatability of an existing recipe alone
does not supply that physical reference.

Manual acceptance, Ruff and compilation pass. Relevant existing optional-policy,
monitor and raw-precision tests pass (25 tests) and remain independent of Mantid. Run:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/validate_dgs_numerical_alternatives.py
```
