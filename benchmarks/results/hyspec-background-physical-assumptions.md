# HYSPEC dummy-background assumptions and statistical target

This audit concerns `HYSPEC_all.nfit`, specifically `Workspace1/Group1`.
Original projects and MDEs were read only; no defaults or production code were
changed. Arithmetic acceptance is distinct from validation of the physical
background transfer.

## Intended directional average

Paul clarified that the dummy is an identical copy of the sample mount, with
matching amounts of glue and aluminum but without crystals; there is no can.
It is nominally isotropic. Aluminum texture, multiple scattering and beam
inhomogeneity can introduce small orientation-dependent effects, so its coarse
approximately 180-degree sweep was intentionally collected to average them.
The intended prediction is this **orientation-averaged dummy background**, with
detector direction, out-of-plane variation and energy dependence retained.

QLab merging realizes a count/exposure-weighted average over dummy rotations:
counts from a detector and energy remain on its laboratory trajectory while
counts and charge are summed. Replay rotates that already averaged directional
background at sample angles. It does not retain each dummy orientation's small
individual variation, and it does not perform radial `|Q|, energy` averaging.
This matches the stated strategy, conditional on a representative coarse sweep
and approximate transfer of the dummy environment to the crystal assembly.
Crystals can change attenuation or multiple scattering; this approximation is
not resolved merely by using identical mount/glue/aluminum quantities.

### Exposure-weighted versus equal-angle averaging

For dummy-angle rate `B_r` and charge `Q_r` in µAh, merging estimates
`sum_r Q_r B_r / sum_r Q_r`. Equal measured-angle averaging instead estimates
`sum_r B_r / R` after normalizing the `R` acquisitions separately. They agree
for equal charges or angle-independent rates. Pooling counts/exposure is the
Poisson estimate for a constant rate; when orientation effects are deliberately
averaged, angular weighting defines the target.

Pulse-charge and omega logs resolve eleven nominal angles, 0–180° in 18° steps:

| Dummy source | Largest relative departure from equal `1/11` charge weight | Total variation from equal weights |
| --- | ---: | ---: |
| Low temperature, 34° | 0.0461% | 0.00951% |
| Low temperature, 70° | 0.0507% | 0.01005% |
| 50 K, 34° | 12.3642% | 1.12402% |
| 50 K, 70° | 0.0300% | 0.00627% |

For 50 K / 34°, 180° receives approximately 10.2149% of charge instead of
9.0909%; other angles each receive about 8.974–8.983%. If the orientation rates
range from `B_min` to `B_max`, the difference between weighted and equal-angle
averages is bounded by `0.0112402 * (B_max - B_min)`, before counting uncertainty.
This is a target-difference bound, not a measured bias of that size.

These pulse-log exposure proxies sum to within 0.00039–0.00234% of stored good
charge. They are not exact pause-filtered per-run charges. Merged event rows no
longer identify original dummy angles, so exact equal-angle rate reconstruction
requires original acquisitions. A continuous uniform angular integral would
also require explicit endpoint/quadrature weights; it is a different target
from the mean of measured orientations. No averaging default was changed.

## Actual saved settings and bank combination

All four sample collections use enabled `measured_events` links at scale 1,
despite legacy names containing “powder”. Private background powder binnings do
not enter that mode. Extra 34° center-projection features are disabled. All run
fit weights and calibration scales are 1; no user masks are saved on these
collections/runs. Detector masks still apply. Background groups are excluded
as ordinary sample contributors, with run entries enabled for explicit links.

| Collection | Sample runs | Sample charge, µAh | Dummy charge, µAh |
| --- | ---: | ---: | ---: |
| Low temperature, 34° | 722 | 42517.260861 | 1836.407184 |
| Low temperature, 70° | 361 | 33173.666196 | 1836.428649 |
| 50 K, 34° | 361 | 26119.089861 | 1734.730895 |
| 50 K, 70° | 361 | 16145.311756 | 1713.504430 |

The low-temperature banks are approximately 1.8 K and 1.5 K; their parent name
explicitly acknowledges the range. Each sample run identity is unique.

Parent `mean_weighting="uniform"` omits an inverse-variance factor, but
`project_composites._composite_mdhisto_data` still passes each child's sample
trajectory denominator as `data_weights`. For aligned grids the result is
`sum_k N_sk * (I_sk - B_k) / sum_k N_sk`, not an equal-bank mean. Here `N_sk`
is sample exposure for bank `k`, `I_sk` its rate and `B_k` its reconstructed
dummy rate. The native MDE reduction pools counts/exposure irrespective of its
legacy per-bank composite `inverse_variance` setting.

This target is unbiased for a common within-bin signal if background predictions
are unbiased for each sample acceptance. For nonconstant rates within finite
voxels, it is an acceptance-weighted average. Exposure weighting is not
necessarily minimum variance with very unequal finite-background uncertainty;
alternatives require a declared target and likelihood, rather than merely
smaller reported errors. Pooling background counts at dummy-charge weights
instead of sample-exposure weights would estimate another target.

## Geometry transfer: what is and is not established

At 50 K, sample minus dummy tank angle differs by 0.0260925293° at 34° and
0.0013732910° at 70°. The larger offset at zero energy transfer and 15 meV
incident energy displaces momentum by about 0.0012 Å⁻¹, or 0.0016 in cubic
reciprocal-lattice coordinates for lattice constant 8.227 Å. Although small
relative to a 0.05-r.l.u. bin, fringes can magnify small acceptance changes.

Current nfit keeps measured dummy QLab coordinates and normalizes their actual
dummy trajectories. Historical MDNorm uses those coordinates with sample
trajectories in the denominator. R2 isolated that mixed-geometry inconsistency
as the main large fringe artifact. Consistency alone does not prove that the
source-geometry rate is exactly the sample-geometry background prediction.

- **Detector-fixed rate:** if each detector's corrected energy-dependent dummy
  rate transfers unchanged, remap `(detector ID, energy)` onto sample detector
  directions and use sample geometry for both numerator and denominator.
  Directional and out-of-plane contrast remain intact.
- **Laboratory angular field:** if rates vary with laboratory direction, the
  bank move probes a different value. Pixel remapping assumes local constancy;
  original QLab replay measures source trajectories but cannot determine an
  unmeasured target direction without a field model/additional observations.
- **Averaged dummy environment:** actual small orientation effects are part of
  the averaging design. Replay predicts the chosen average rather than each
  sample orientation's environment. Geometry transfer and crystal-induced
  attenuation/multiple scattering are separate approximations.

The 0.1° compatibility tolerance is not an error bound for these assumptions.
Pixel transfer requires stable detector IDs, compatible energy reconstruction,
efficiency/solid-angle correction and masks. Changed flight paths, calibration,
grouping or acquisition state can require raw re-reduction. Paul confirmed that
the bank was moved repeatedly between temperatures and sample/dummy acquisitions. The encoder reaches each target within the instrument's
positioning tolerance; these offsets are expected physical acquisition differences,
not evidence of faulty alignment. The accepted treatment retains each acquisition's
own geometry for both events and normalization and replays the averaged dummy at
sample rotations. Detector remapping is not adopted.

### Controlled truth, rather than appearance

The independent detector/energy-cell calculation shifts the bank by the actual
34° offset magnitude. A bin boundary moves across one of two detector cells
with fixed rates 10 and 50; true sample signal is 5. For 200,000 independent
Poisson replicates with known exposures:

| Treatment | Mean signal | Bias | Empirical sigma | 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Consistent original QLab geometry | −15.00298 | −20.00298 | 0.73677 | 0% |
| Source numerator / sample denominator | 9.99842 | 4.99842 | 0.41037 | 0% |
| Detector-ID/energy remapping | 4.99846 | −0.00154 | 0.43919 | 95.075% |

Remapping mean's Monte Carlo standard error is 0.000982 and its reported sigma
matches empirical scatter. The mixed recipe has smaller sigma but large bias.
Current QLab treatment is also biased for this particular detector-fixed truth.
This is a discrete energy-cell limit, not a continuous-event implementation test.
For a different truth with lab-field logarithmic derivative 1 per degree, the
same move changes rates by 2.64% and unadjusted pixel transfer leaves background
bias about 0.793 in these arbitrary rate units. No universal policy follows.

## Counting covariance, gaps and calibration uncertainty

Let `a_i = Q_i f_i / sum_j Q_j f_j` be sample-angle fractions. A source event's
coefficient in output bin `b` is `c_eb = sum_i a_i h_ib(e)`, with `h` selecting
transformed bin membership and valid acceptance. For corrected weight `w_e`
and counting variance `v_e`, replay gives `C_b = sum_e w_e c_eb`,
`V_b = sum_e v_e c_eb²`, and rate `B_b = C_b/N_b`. Cross-bin covariance is
`sum_e v_e c_eb c_ed / (N_b N_d)`. Replaying one event does not produce new
independent counts. R2 verifies these formulas conditionally; cached cuts
require original-source replay or retained dependencies to recover cross terms.

Sample and dummy raw runs are disjoint, supporting conditional count independence.
For a known link scale `alpha`, subtraction variance is
`Var(I_s) + alpha² Var(B) - 2 alpha Cov(I_s,B)`. Common observations or nuisance
parameters need the covariance term; disjoint runs do not establish calibration
independence. Actual bank samples and their distinct dummy sources do not share
counting primitives, but may share calibration parameters.

Replay intersects detector masks and common energy coverage. Native background
gaps remain unavailable; aligned subtraction needs both operands measured.
Explicit user background exclusions are treated as zero contributions, a
separate declared decision (none exist in inspected settings). A covered zero
count has zero observed variance but a positive confidence limit. The current
Feldman–Cousins upper display limit, scaled by global event-weight RMS, is not
a symmetric Gaussian sigma or an additive count variance. Keep it separate
from final-bin variance and likelihoods.

Current errors condition on fixed charge, geometry, incident energy, correction
factors and link scale. Their finite uncertainties are not provided in saved
settings. An independent uncertain link scale contributes `B² Var(alpha)` at
first order. Shared calibration must perturb the entire subtraction jointly:
a common 2% factor applied to rates 35 and 30 contributes variance
`(0.02 * 5)² = 0.01`, whereas incorrect independent operand perturbations give
`0.02² * (35²+30²) = 0.85`. At coverage discontinuities, paired nuisance replay
is more informative than local derivatives or independent per-bin jitter.

## Acceptance recommendations and receipts

### Assessment of DGS alternatives

| Treatment | Statistical or numerical reason | Decision boundary |
| --- | --- | --- |
| Pool event numerators and exposure | Estimates a common rate under independent counting acquisitions; covered zeros add exposure. Observed-count inverse-variance weights can bias sparse rates and omit zeros. | Keep the DGS target explicit; a finite-bin acceptance average is not a uniform spatial integral. |
| Track copies of one background event | Coefficients of a reused observation must combine before variance propagation. This is a dependency requirement, rather than a preference for larger errors. | Recover background covariance for the requested final bins; sample symmetry and calibration dependencies remain separately declared. |
| Each run's trajectory Ei | Uses the measured run energy instead of the first experiment's energy when energies actually differ. | Validate energy calibration and acceptance fringes before changing the Mantid-compatible default. This is scalar per-run Ei, not a pulse-resolved inference. |
| Stable monitor variance | Avoids numerical cancellation in monitor variance construction and resulting unstable calibration fits. | Numerical stability alone does not validate the monitor noise model or guarantee a more accurate Ei/T0. Compare with calibration truth. |
| High-precision event projection | Reduces storage/projection rounding and avoids artificial boundary loss. | It does not improve instrumental resolution or replace geometry/Ei uncertainty. Keep an exact Mantid compatibility path. |
| Physical converter bounds | Retains valid events rejected only by float32 rounding of observed extrema. Fresh HYSPEC references recover exactly the native event set with physical bounds. | Document the original recipe's one/two-event omissions separately; do not call different event membership parity. |
| Detector-ID/energy background transfer | Predicts sample acceptance consistently when the dummy's rate transfers by detector identity. | Validate the meaning of the bank-angle offsets and the detector/energy response. The controlled truth establishes the assumption, not that it universally holds. |

Keep the existing resolved conventions reproducible. Assess these alternatives
against bias and interval coverage, especially in the fringes, before selecting
future defaults. Smoother plots and smaller marginal error bars are insufficient.

Retain averaging all dummy rotations together in detector direction and energy,
then reconstruct at sample rotations. Preserve out-of-plane/detector anisotropy;
do not replace it with radial powder averaging or demand zero dummy orientation
variation. Record the averaging target, especially the 50 K / 34° weighting.
A per-detector/energy comparison of original dummy angles can bound residual
variation; merged events alone cannot provide it. Exact equal-angle reweighting
and pixel transfer remain reviewed alternatives. Keep numerator/denominator
geometry consistent. Finish declared final-cut target/dependency handling before
claiming complete quantitative uncertainty. Original settings remain unchanged.

Receipts and reproducible manual diagnostics:

- `hyspec-background-angle-exposure.json` and
  `../validate_hyspec_background_angle_exposure.py`: merged-log angle weights.
- `hyspec-background-transfer-assumptions.json` and
  `../validate_hyspec_background_transfer_assumptions.py`: controlled statistical
  truths, bias/coverage and shared-calibration example.
- `hyspec-background-analysis.md`: existing R2 event-oracle and historical
  geometry isolation evidence.

Controlled assertions, Ruff, compilation and diff checks pass. No CORELLI or
MACS work was included.
