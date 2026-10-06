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

One reference bin has no corresponding occupied reconstructed bin. The largest
relative signal discrepancy is approximately 0.5. These sparse differences are
not an export-rounding certificate: full importer/histogram parity remains
open. Changing only arithmetic precision in bin membership does not resolve
them. Further diagnosis must trace the geometry and individual observations.

With identical coordinates, corrected observations and monitor weights, the
production nfit accumulator agrees with the independent pooled calculation to
3.92 × 10⁻⁸ absolute signal and 5.92 × 10⁻¹² absolute standard uncertainty.
This verifies accumulator arithmetic, not the full DAVE reduction chain.
The legacy saved inverse-variance recipe gives a median relative signal
difference of approximately −14.3% from the DAVE export under this matched
input comparison. DAVE's MACS weighted-mean path uses supplied monitor weights;
it is not the inverse-variance estimator for normalized measurements.

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
