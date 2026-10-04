# Exposed zero-count bins: uncertainty and likelihood checks

The reproducible calculation is
[`validate_dgs_zero_count_uncertainty.py`](../validate_dgs_zero_count_uncertainty.py),
with scalar results in
[`dgs-zero-count-uncertainty.json`](dgs-zero-count-uncertainty.json).
It uses independent Poisson-tail inversion, exact sums over sampling
probabilities, and 120,000 fixed-seed replicates per simulated case. Public nfit
support, interval, likelihood and source-sensitivity APIs are checked directly.
No instrument data, scientific project or Mantid is opened. Production code and
defaults are unchanged.

In these controlled models, counts are dimensionless, exposure `N` is in seconds,
rate `I` is counts/s and event weights are dimensionless. A rate therefore has
variance in counts²/s². Actual DGS normalization exposure has the convention
documented in [`physics_conventions.md`](../../docs/physics_conventions.md);
this report does not identify it with elapsed time or establish its calibration.

## 1. A measured zero is informative, but its observed variance is not an interval

For independent unit-weight source counts `C ~ Poisson(N I)` with known positive
exposure, the rate estimate is `C/N` and its accumulated observed variance is
`C/N²`. At `C=0`, both are zero. This observed variance estimates sampling variance;
it is not a claim that the unknown rate is exactly zero. The zero-count likelihood
is `exp(-N I)`, so a positive trial rate remains constrained by the measurement.
Adding exposure from a measured zero also changes a pooled estimate with other
nonzero observations.

For a central confidence level `1-alpha`, a zero has Garwood interval
`[0, -log(alpha/2)/N]`. At 68.27% the upper bound is `1.84102/N`; at 95% it is
`3.68888/N`. A **one-sided** 95% upper bound is `2.99573/N`, a different requested
interval. An upper bound is not a symmetric standard error and should not be
inserted into a Gaussian fit as though it were one.
Resampling an empty observed event set, or simulating Poisson counts at its
zero maximum-likelihood rate, also generates only zeros. That bootstrap cannot
recover an upper limit without additional model-based inference.

The interval implementation agrees with independent inversion of Poisson tail
probabilities for observed counts 0–64 at both confidence levels. Exact coverage
is at least the requested confidence throughout the tested means. Conservative
coverage is expected for discrete observations.

| True expected count | Exact central 95% coverage | Observed-error Gaussian 95% coverage, simulation |
| ---: | ---: | ---: |
| 0.05 | 99.879% | 4.971% |
| 0.2 | 98.248% | 18.283% |
| 0.8 | 99.092% | 55.088% |
| 2 | 98.344% | 86.286% |
| 10 | 97.539% | 92.417% |

This is a confidence-interval failure of the Gaussian approximation at sparse
counts, not an error in summing event variances. The already validated
[estimator comparison](dgs-estimator-alternatives.md) separately demonstrates
bias from omitting zero counts to construct observed inverse-variance weights.

### Current nfit behavior

The controlled public-API check confirms:

- An unmasked bin with positive retained exposure and zero counts remains
  measured. An unexposed bin and a masked bin are excluded.
- Audited Poisson-deviance fitting retains the exposed zero. For a predicted mean
  of two counts, its signed deviance residual is `-2`, even though its stored
  standard error is zero.
- Gaussian fitting requires positive standard errors. Its historical selection
  excludes zero-error points; passing an exposed zero directly to the Gaussian
  residual API raises rather than assigning infinite precision. This objective
  is unsuitable for sparse counting data without a justified alternative model.
- A final-bin audited Poisson confidence channel gives a positive upper bound
  for the exposed zero and NaN for unexposed or masked bins.
- The interval API rejects undeclared event weights, heterogeneous variance,
  symmetry copies, background differences and uncertain exposure. Integer-looking
  corrected numerators alone do not certify independent Poisson counts.

No production correctness bug was found in these paths. Source-aware counting
containers also install their propagated primary dependencies during
construction, so fit preparation does not silently drop represented exposure
uncertainty.

## 2. Weighted zeros need the source response model

Let source counts in event classes be independent Poisson variables `n_j`, with
known correction weights `w_j`. The corrected numerator is `C=sum(w_j n_j)` and
its observed variance is `V=sum(w_j² n_j)`. These remain valid additive statistics.
The weighted numerator generally is not an integer Poisson variable. When all
source counts are zero, `C=V=0` contains no information about the weights of
unobserved events.

Even the constant-weight cases demonstrate this limitation: at `C=V=0` and
`N=1 s`, known weight 1 gives a central 95% corrected-rate upper bound 3.68888,
whereas known weight 4 gives 14.75552 counts/s. The same three observed statistics
need different bounds under different declared source responses.

The compound-count test uses weights `[1,4]` with externally known, independent
mark probabilities `[0.9,0.1]` and a source-event rate of 2/s. The corrected target
rate is 2.6/s. Mean accumulated variance agrees with the analytic variance, while
Gaussian 95% coverage is only 18.17%, 86.58% and 91.66% for exposures 0.1, 1 and
10 s respectively. Correct event variance alone does not establish low-count
interval coverage.

If that mark distribution is truly known independently and has no additional
dependence on the unknown signal, the integer total source count can estimate
the event rate. Scaling its interval by the known mean weight 1.3 gives calibrated
bounds for the **expected corrected rate**. This also uses the estimator
`1.3*sum(n_j)/N`, rather than the realized weighted estimator `C/N`. It is a
different model and estimator. In the zero-count, 0.1 s case the former upper
bound is 47.9554/s; replacing the known mean weight with the RMS weight gives
58.3263/s. No universal RMS-weight floor is inferred from this example.

For DGS data, correction weights, detector/energy responses and the signal often
vary together. A known independent mark law cannot be assumed from the weights
of detected events, neighboring pixels or an empty bin. A valid model may instead
require class-specific source counts and trajectories with the physical response
of each class. Fractional interpolation also requires the original source
membership and its covariance; it does not create independent counts or
measurement exposure in an unobserved physical region.

## 3. Exposure calibration is a separate nuisance measurement

The controlled calibration example observes two independent counts:
`K ~ Poisson(q I)` and `M ~ Poisson(q kappa)`, where `q` is unknown exposure in
seconds and the monitor response `kappa=5/s` is known. Conditional on `K+M`,
`K` is binomial with success probability `p=I/(I+kappa)`. Exact conditional
binomial intervals, transformed by `I=kappa*p/(1-p)`, eliminate `q` without
treating the noisy monitor count as known exposure.

For expected monitor counts 0.5, 2.5 and 10, exact interval coverage is 99.930%,
99.156% and 97.577%. Simulated coverage agrees. At expected monitor count 10,
plugging `M/kappa` into a known-exposure Garwood interval gives only 93.014%
coverage among observations with resolved positive monitor count. At expected
monitor count 0.5, that plug-in construction has undefined exposure in about
60.9% of repetitions. This is a missing normalization measurement, not evidence
that the sample region was never measured.

For the particular observed pair `K=0, M=1`, the central conditional 95% upper
rate bound is 195/s, versus 18.4444/s from treating the monitor estimate as exact.
For `K=M=0`, the conditional interval is unbounded above. These deliberately
sparse synthetic monitors illustrate a model boundary, not the uncertainty of
an actual SNS monitor or vanadium calibration.

First-order propagation of `C/N` gives exposure derivative `-C/N²`. At observed
`C=0`, the exposure term is zero, and the public delta-method estimator correctly
returns zero observed variance in the tested case. That approximation is not a
confidence interval for an unknown rate. Calibration covariance can still affect
a pooled nonzero estimate and must be propagated before evaluating the pooled
ratio. Current DGS caches do not retain a complete shared monitor/vanadium
likelihood, so this conditional model cannot be automatically enabled for them.

## 4. Background zeros and shared sources need their joint model

With signal rate `S`, background rate `B`, known sample exposure `N_s` and
background exposure `N_b`, independent counts have means
`N_s(S+B)` and `N_b B`. The signed rate estimator is
`K/N_s-L/N_b`, with observed variance `K/N_s²+L/N_b²`.
Both zero counts again give zero estimate and zero observed variance without
fixing `S` or `B` exactly.

As an independent conservative reference, central 97.5% intervals for each
component give simultaneous coverage at least 95% by the union bound. Subtracting
the endpoints gives a valid interval for the signed difference. This remains a
distinct confidence calculation, not an additive error-bar rule or a preferred
background fitting objective.

For `S=1/s`, `B=2/s`, `N_s=0.2 s` and `N_b=0.05 s`, both counts are zero in
49.66% of measurements. Observed-error Gaussian 95% coverage is about 50.29%,
whereas the conservative difference interval covers in essentially every tested
repetition. At two zeros its signed bounds are `[-87.6405,21.9101]/s`.
With larger exposures `N_s=1 s`, `N_b=4 s`, Gaussian coverage is 90.42% and the
conservative interval coverage is 99.954%. The conservatism is substantial;
appropriate joint source likelihoods could give more useful bounds under their
stated assumptions.

Replaying one background measurement at six sample angles still supplies only
one background observation. In the shared-background test, the correct final
variance is 22.5 counts²/s² and the empirical variance is 22.7485. Treating the
six background copies as independent reports only 5.85924 on average. The public
source-sensitivity estimator reproduces 22.5 by merging the shared background
coefficients before squaring. It is an observed covariance calculation, not a
new Poisson likelihood for the subtracted pixels. Virtual trajectory exposure
can be remapped or repeated consistently with replayed numerators; it is not
additional physical counting exposure.

An all-zero source realization still has zero observed propagated variance.
Shared-source identity must be retained for a joint count model: repeated copies
must not multiply independent background likelihood contributions. Current cached
background profile replay fixes represented covariance of the existing field
target; it does not by itself supply low-count nuisance-parameter intervals.

## 5. Future direction: retain physical count exposure from the response model

A concrete weighted-zero interval is possible if a forward model is validated.
Let `x` identify a unique physical detector/energy/acquisition coordinate, with
independent Poisson event mean `I dR(x)` for a common intensity `I`. For a known,
deterministic, strictly positive correction `w(x)`, require the actual corrected
normalization identity `N=integral(w dR)`. Then physical count exposure is
`B=integral(dR)=integral(dN/w)` over the accepted source region. `B` has units of
count per intensity, so `I B` is a dimensionless expected count. The unique raw
count is Poisson with mean `I B`; at zero its likelihood is `exp(-I B)` and the
central confidence upper bound is `-log(alpha/2)/B`. For constant weight this
reduces to the existing `w*(-log(alpha/2))/N`. A retained, response-derived `B`
would supply the missing information without guessing an RMS event weight or
changing accumulated variance. Under this common-intensity model the count
likelihood also uses the unique raw count for nonzero observations; it must not
reinterpret a heterogeneous weighted numerator as an integer count.

Current native DGS normalization accumulates charge times detector normalization
times accepted trajectory energy length; raw event weights can include helium-3
efficiency and `ki/kf` corrections. A future implementation could integrate their
known inverse response over the same clipped trajectories, but first needs to
validate the normalization identity, calibration scale, acceptance filters and
detector/energy response. The existing aggregate `C,V,N` and weights of realized
events do not certify it. Generic corrected MDE data also need complete response
provenance. This is a conditional design derivation, **not an instrument-validated
DGS interval or an adopted default**.

`B` must count each physical source acceptance region once. Symmetry copies
select a union of original-source regions; overlaps cannot multiply count
exposure, and the common-intensity/symmetry assumption must hold on that union.
The independent-copy variance convention does not certify Poisson intervals.
Replayed background counts require their separate positive-rate source model
and joint likelihood. Calibration uncertainty needs nuisance observations.
Fractional kernels constrain their source footprint rather than automatically
measuring an unobserved requested cell. If intensity varies within that footprint
or bin, zero counts constrain `integral(I(x) dR)`: its count-response-weighted
average differs from the corrected `w dR`-weighted field target. Applying the
same bound to that different target requires additional shape assumptions or a
separately justified conservative construction.

## Recommended implementation boundaries

1. Preserve `C`, `V`, physical normalization exposure, masks and source identity.
   Keep measured zeros in counting aggregation; do not give absent regions a
   zero measurement or fill their support through interpolation implicitly.
2. Keep standard uncertainty and confidence bounds separate in channels,
   exports and fits. Display interval method, level, tail convention, target and
   normalization assumptions. Do not replace `V` with an upper-limit square.
3. Continue using the audited known-exposure, constant-weight Poisson path for
   its admissible data. For sparse counting fits, choose an explicit count
   likelihood that retains zeros; Gaussian selection with positive sigma is a
   different objective and should remain visibly declared.
4. Before adding DGS weighted/background interval channels or likelihoods,
   retain a bounded primitive model: unique source counts, response/exposure by
   class, actual copy memberships and any shared calibration observations.
   Evaluate the likelihood once per independent primitive. Test final cuts and
   low-exposure fringes with exact or repeated-sampling coverage.
5. Treat an externally known mark law, calibration likelihood, nonnegative
   signal constraint, Bayesian prior or conservative nuisance envelope as a
   separately named option with explicit assumptions. None is established by
   the three additive histogram statistics alone. No default change or universal
   positive-error floor is supported by these tests.

The related fractional-binning study must declare whether it estimates a
physical bin average or an interpolation target. Smoother images and propagated
kernel variance alone cannot establish either missing physical coverage or a
calibrated interval for an exposed zero.
