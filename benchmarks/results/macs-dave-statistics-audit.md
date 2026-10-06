# MACS counting and DAVE reference audit

## Inputs and reference

The manual reference uses the public NCNR MACS August 2026 files from
[202608/15/data](https://ncnr.nist.gov/pub/ncnrdata/macs/202608/15/data/), runs
260, 250, 261, 266, 248, 249, 267, 268, 247, 243, 373, 271, 244, 273, 245 and 246.
The supplied `test_dave.txt` was exported without smoothing. Its saved session
still contains the earlier smoothing setting and must not be used to infer the
current export's smoothing.

Reference settings are a cubic 8.24 Å cell, orientation [1,1,0]/[0,0,1], A3
offset 66.5°, axes (H+K)/2, L and energy transfer, steps 0.05 r.l.u., 0.05 r.l.u.
and 0.1 meV, target monitor 10⁶, monitor-weighted averaging, no ki/kf adjustment,
and the historical lambda/2 monitor correction. The installed DAVE application
reports version 2.6 Beta. Public DAVE source and the
[mslice manual](https://www.ncnr.nist.gov/dave/documentation/dcs_mslice.pdf)
provide independent algorithm evidence; the installed compiled application is
not assumed identical to the public source snapshot.

## Numerical evidence

An independent reconstruction includes 290,240 detector observations. It has
68,008 occupied bins; the supplied export has 68,007. Matching bins by the
reported coordinates gives:

| Channel | Positive matched bins | 99th percentile absolute relative discrepancy | Bins differing by more than 10⁻⁴ |
| --- | ---: | ---: | ---: |
| Signal | 59,990 | 4.25 × 10⁻⁷ | 179 |
| Standard uncertainty | 59,990 | 4.37 × 10⁻⁷ | 180 |

The 179 signal differences represent approximately 0.30% of positive matched
bins, rather than a uniform 0.30% intensity-scale error. One reference bin has
no corresponding occupied reconstructed bin. The largest relative signal
discrepancy is approximately 0.5. The source trace below accounts for these
differences.

With identical coordinates, corrected observations and monitor weights, the
production nfit accumulator agrees with the independent pooled calculation to
3.92 × 10⁻⁸ absolute signal and 5.92 × 10⁻¹² absolute standard uncertainty.
This verifies accumulator arithmetic, not the full DAVE reduction chain.
The legacy saved inverse-variance recipe gives a median relative signal
difference of approximately −14.3% from the DAVE export under this matched
input comparison. DAVE's MACS weighted-mean path uses supplied monitor weights;
it is not the inverse-variance estimator for normalized measurements.

## Cause of the sparse DAVE discrepancies

The inspected [public DAVE source](https://www.ncnr.nist.gov/dave/download.html)
contains a preliminary `dm_combine_macs` step in `dm_filetools.pro`, called when
loading multiple files. It groups scan points using rounded keys for incident
energy Ei (meV), final energy Ef (meV), sample rotation A3 (degrees), detector-bank
rotation kidney (degrees), incident-filter state and analyzer selection. The
supplied session uses tolerances 0.01 meV, 0.01° for A3 and 0.1° for kidney.
The key for a coordinate x and positive tolerance t is round(x/t + 0.1), not a
pairwise distance test. DAVE sums detector counts, variances and monitor counts
within each group, then replaces Ei, Ef, A3 and kidney by monitor-weighted
averages. Momentum reconstruction and final histogramming follow this merge.

Independent reproduction of that step combines 14,512 scan points into 13,490:
12,556 singleton groups, 846 pairs and 88 triples. Each has twenty detector
channels. This reduces the 179 signal mismatches to two; a further observation
at a rounding-sensitive boundary explains the remaining occupancy difference.
There are two affected observations overall, both within 3.9 × 10⁻⁷ r.l.u. of
a momentum edge. Diagnostic replay of their exported bin memberships gives
all 68,007 occupied reference bins, no zero-channel mismatches and the following
maximum absolute relative differences:

| Channel | Positive reference bins | Maximum relative difference after boundary replay |
| --- | ---: | ---: |
| Signal | 59,991 | 5.98 × 10⁻⁷ |
| Standard uncertainty | 59,991 | 6.17 × 10⁻⁷ |
| Monitor exposure | 68,007 | 5.33 × 10⁻⁷ |

The two membership changes are diagnostic accounting checks, not changes to
nfit's physical edge rules or a certificate of bit-for-bit IDL arithmetic.
They involve run 266 scan index 601, detector 20 (1,785 counts), and run 245
scan index 731, detector 19 (zero counts). Scan indices here start at zero;
detector numbers start at one.

A concrete non-rounding example is detector 18 at scan index 748 in runs 248
and 249. Before merging, their L coordinates are −0.71215832 and −0.71177351
r.l.u., with 64 and 4 counts respectively. They straddle the edge near
−0.7120946 r.l.u. DAVE's weighted geometry gives L = −0.71209437 r.l.u.,
placing all 68 counts together in the upper bin. nfit's separate-coordinate
histogram places the two measurements in their respective bins. The change
in exposure matters even when a moved observation has zero counts.

The user's fresh `new_test_dave.txt`, exported after clearing and reloading,
is numerically identical to `test_dave.txt`. A separate run-248 export has
4,748 occupied bins and only two signal discrepancies, explained by one
3,072-count observation 2.6 × 10⁻⁷ r.l.u. from an edge. Replaying that membership
makes all channels agree within 0.64 ppm. Native nfit run-248 coordinates differ
from the independent float32 DAVE formulas by at most 6 × 10⁻⁷ r.l.u. and
1.01 × 10⁻⁶ meV; corrected monitor exposure agrees within 0.103 ppm.

[The numerical receipt](macs-dave-merge-audit.json) records reference and input
hashes, source hashes, settings, metrics and boundary probes. The source archive
is the September 16, 2026 snapshot; the installed application reports 2.6 Beta
and IDL 9.2.0. This is a manual comparison and introduces no DAVE or Mantid
dependency into nfit or its unit tests.

**Policy:** retain each measured coordinate and pool statistics after assignment
to the requested physical grid. Preliminary coordinate averaging loses the
individual measurement locations and can change a finely binned response.
DAVE's final monitor-weighted estimator agrees with the counting model below;
its preliminary coordinate merge is not required for that estimator. For
smoother assignment near edges, fractional binning remains a separate explicit
choice with its documented diagonal uncertainty approximation.

## Adopted counting model

For detector observation j, retain raw counts Cⱼ, observed Poisson variance
Vⱼ = Cⱼ and calibrated exposure Dⱼ. A known multiplicative intensity correction
cⱼ belongs in exposure: Dⱼ = Mⱼ/(T cⱼ), where Mⱼ is the corrected monitor count
and T is the dimensionless requested monitor target. A hard bin estimates
sum(Cⱼ)/sum(Dⱼ), with standard uncertainty sqrt(sum(Vⱼ))/sum(Dⱼ). This is the
Poisson maximum-likelihood estimate of a common response when exposures are
known and observations independent. It weights an exposed zero-count
observation through its exposure, without using its measured error as a weight.
The same retained quantities determine subsequent aligned histogram profiles.

Fractional assignment αⱼ contributes αⱼ Cⱼ, αⱼ² Vⱼ and αⱼ Dⱼ. Only diagonal
uncertainty is retained; local inter-bin covariance and covariance from shared
symmetry copies remain separate work. Zero observed variance is retained as a
sampling statistic. It does not imply a known zero rate. Choice of zero-count
intervals and fitting objectives remains open; no one-count floor is added.

For a 1/v incident monitor, correct monitor response and apply neutron ki/kf
separately. Together these give the dimensionless intensity multiplier k₀/kf,
where k₀ is the declared reference wavevector (default 1 Å⁻¹). This avoids an
unwanted incident-energy dependence. Equal detector sensitivity is the default
until an experiment-specific calibration exists. Sparse nonzero occupancy is
an optional detector-health heuristic, not an automatic sensitivity estimate.

## DAVE choices that are not adopted as defaults

- The historical lambda/2 model is empirical and optional. The reference export
  is consistent with applying it to all selected observations, but the NeXus
  incident-filter logs read IN. The intended configuration changed the incident
  Be filter above 1.3 meV transfer. Those conflicting records do not justify an
  automatic all-run correction. The nfit option checks the filter state and its
  2–20 meV calibration range; it can be overridden per run.
- A post-sample BeO filter does not establish the incident-monitor spectrum.
  Unfiltered beam contamination needs a justified incident-flux calibration.
- DAVE's optional neighborhood zero-error estimate and histogram smoothing are
  not copied into the reduction. In the inspected source, smoothing applies
  kernel coefficients directly to variance rather than squaring them, which
  does not propagate independent-input variance. The reference comparison uses
  no smoothing. nfit display smoothing changes only the viewer presentation.

The automated tests use synthetic files, saved native project round trips and
independent formulas. They do not invoke DAVE or Mantid.
