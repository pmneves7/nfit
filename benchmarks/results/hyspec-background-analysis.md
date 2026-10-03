# HYSPEC directional background analysis: checkpoint 6A-R2

The intended reference is the successful directional subtraction in
`HYSPEC_all.nfit`. Its four sample collections under `Workspace1/Group1` use
`measured_events`; legacy names containing “powder” do not change that mode.
The separate `powder averages` branch uses center projection. Original projects,
MDE files and saved histograms were read only; no project was rebuilt.

## What was validated

The complete measured background from each temperature/bank configuration was
replayed at nine evenly spaced sample angles onto a bounded HHL/energy slab.
An independent event oracle reads QLab coordinates, detector identities, source
variances, masks and goniometers, explicitly transforms events and adds pairwise
covariance. It does not use the production replay accumulation or bin-index
helpers. Exposure additivity uses the shared trajectory integrator, whose
Mantid agreement was established separately in [6A-R](hyspec-dgs-parity.md).

| Sample temperature | Bank | Background events | Maximum final-bin numerator discrepancy / peak |
| --- | --- | ---: | ---: |
| 50 K | 34° | 4,744,544 | 3.48 × 10⁻¹⁴ |
| 50 K | 70° | 4,624,140 | 1.18 × 10⁻¹⁴ |
| 1.8 K | 34° | 5,080,328 | 9.03 × 10⁻¹⁴ |
| 1.5 K | 70° | 4,978,029 | 1.44 × 10⁻¹⁴ |

Numerators, marginal variances and source-event counts agree in covered cells,
the lowest-exposure decile and coverage fringes. Covered zero-count cells remain
measured; their display confidence intervals are separate from accumulated event
variance. Direct replay onto an energy-only final grid agrees with the oracle's
full covariance calculation to below 9.1 × 10⁻¹⁴ relative to the variance peak.
Exposure sums agree to below 6.1 × 10⁻¹⁴ relative to the exposure peak. Nine-angle
sampling makes the selected slab sparse: every covered cell is within two cells
of missing coverage, so the fringe check covers the entire measured support.
This is a bounded acceptance calculation, not a rebuild of every cached binning
or a validation of shared calibration uncertainty.

Synthetic tests independently verify 10:1 directional contrast at equal momentum
modulus and energy, unequal sample charge and fit weights, background calibration,
repeated angles and same-event covariance. Both NumPy and Numba replay backends
are exercised without Mantid imports.

## Historical Mantid discrepancy

The saved background is directional QLab MDE data, not a radial map. Its upstream
`GenerateDGSMDE` recipe combines background sample angles into one laboratory-frame
experiment while preserving detector directions. The 50 K sources retain 17,920
detector IDs: twelve raw background runs at 34° (506944–506955) and eleven at 70°
(506933–506943). Their total proton charges are 1734.730895389 and
1713.504429619 µAh respectively.

The previous comparison included 24 extra 34° sample runs. The corrected
comparator selects the saved histogram's actual 337 runs, 505555–505891, and
361 runs, 506277–506637, at 70°. Correcting membership does not remove the main
outliers. The original remote 337-run file and extended local source have
bit-identical serialized physical detector geometry; selected event-payload
identity across those two files was not established.

Historical `MDNorm` replays measured background QLab events into the sample frame,
but constructs the background denominator with **sample detector geometry**.
Native measured-event replay uses **background detector geometry**, matching the
actual directions of its numerator. The bank settings differ slightly: sample
minus background Tank angle is 0.0260925293° at 34° and 0.0013732910° at 70°.
These small differences move trajectory intersections across coverage fringes.

At HH = −0.375, L = 1.025, H−H ≈ −0.01, energy transfer = 0.25 meV, the sample
numerator and variance are zero:

| Treatment | Background-subtracted intensity, saved arbitrary units |
| --- | ---: |
| Native consistent background geometry | −0.100043765 |
| Saved Mantid histogram | −6.233133007 |
| Diagnostic using historical sample-geometry denominator | −6.293544791 |

Changing only the denominator geometry after matching historical angle weighting
reduces overall signal RMS difference from 0.004968718 to 0.001266771, and maximum
difference from 6.133089 to 1.255083. This isolates the principal cause of the
largest background artifact. It does **not** establish complete historical
parity. Residuals remain, including possible event projection/edge precision and
unverified equivalence of the selected original and extended event payloads.
Uniform angle weighting and charge-times-angle-count pooling alone leave the
largest pixel unchanged. Alternate bank pooling is diagnostic only; no defaults
were changed.

Keep the current consistent directional treatment. A future detector-pixel
transfer between calibrations should remap detector ID and energy onto sample
directions for both numerator and denominator, then be validated independently.
The mixed historical geometry is a compatibility convention to review at the
later alternatives checkpoint, not an automatically preferred physical model.

## Uncertainty boundary and the fix

The replay service correctly combines copies of a source event that meet in one
voxel. It does not retain their covariance across different voxels or a certified
additive count-statistics payload. Later cached-grid profiles therefore cannot
reconstruct the exact background uncertainty or necessarily preserve an
exposure-pooling target. Replay directly onto the requested final cut grid for
quantitative uncertainty under the current model.

For an analytic source count with variance 8 and angle fractions 0.25 and 0.75,
final normalized variance is 2. Summing only diagonal numerator variances gives
1.25, and an inverse-variance mean of the two normalized pixels gives 1. Neither
creates a second independent background measurement. In the real nine-angle
50 K / 34° energy profile, retaining cross-voxel covariance increases uncertainty
by a median factor 1.80 and a maximum 2.12 relative to diagonal pooling. The
lowest-exposure decile reaches 1.50. These are ratios for this diagnostic's
specific pooled target, not correction factors to apply to arbitrary cuts.

A separate bug was fixed: singleton trajectory preparation could silently choose
per-run incident energies when first-run Ei was selected. References are now
resolved before sample trajectory preparation and background mask partitioning.
The regression gives exposure 2.00 for first-run Ei and 2.01 for the optional
per-run convention. Existing background/composite caches are invalidated on
requested recomputation; reduced event caches remain reusable. Original saved
projects were not changed.

## Receipts and remaining work

- `hyspec-background-directional-replay.json` and
  `../validate_hyspec_background_replay.py`: all four real background sources,
  event oracle, fringe and final-grid covariance checks.
- `hyspec-background-matched-history.json` and
  `../validate_hyspec_saved_histogram.py`: selected historical membership,
  angle/bank conventions, geometry isolation and the twelve largest residuals.
- `tests/test_mdevent_background_statistics.py`: analytic regression references.

6A-R2 analysis is complete. Required implementation follow-up 6A-R3 is automatic
final-cut replay or retained source dependencies/explicit target for cached
background cuts and exports. Strict historical compatibility and calibration
transfer alternatives remain for 6A1/6A2. Full HYSPEC histogram/uncertainty parity
with the saved historical subtraction is not claimed.

Validation: 2,905 tests passed, one unavailable-CuPy test skipped. Production and
unit tests remain independent of Mantid/Shiver. Ruff, byte-compilation, diff
checks and warning-as-error Sphinx build passed.
