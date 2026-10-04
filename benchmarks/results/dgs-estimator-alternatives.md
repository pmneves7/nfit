# DGS estimator alternatives: conditional truth checks

The reproducible calculation is
[`validate_dgs_estimator_alternatives.py`](../validate_dgs_estimator_alternatives.py),
with scalar results in [`dgs-estimator-alternatives.json`](dgs-estimator-alternatives.json).
It uses 120,000 independent replicates per case, fixed seed 20261003, independent
analytic expectations, and public nfit estimator, covariance and interval APIs.
It reads only the existing scalar dummy-angle receipt; no scientific project,
instrument data, or Mantid is opened. Application defaults are unchanged.

The accepted time-weighted sample-angle correction is the acquisition convention
for follow-up real-data checks. Continuous sample rotation remains deferred.
These calculations isolate statistical assumptions; they do not certify unknown
instrument calibration, angular stationarity, or nuisance-parameter uncertainty.
Physics quantities and the general estimator contracts are defined in
[`physics_conventions.md`](../../docs/physics_conventions.md).

## 1. Pool counts and exposure for independent counting measurements

Let `C_i` be independent Poisson counts and `N_i` known exposure in arbitrary
exposure units. For a common rate `I`, `E[C_i] = I N_i`. The pooled estimator
`sum(C_i)/sum(N_i)` is unbiased, with variance `I/sum(N_i)`. Its normalized
weights are `N_i/sum(N_i)`. An inverse-variance mean using the **expected**
variance `I/N_i` gives exactly the same weights. Using the fluctuating observed
variance `C_i/N_i²` changes the estimator.

For exposures proportional to `[0.1, 0.3, 1, 3]`, the uniform mean of the four
normalized rates has 4.0333 times the pooled variance. The analytic ratio is
unchanged when all exposures are scaled; the simulations recover it at low,
moderate and high counts. Both methods estimate the same common rate here.

| Expected total counts | Pooled mean, true rate 2 | Uniform mean | Observed inverse variance, zeros omitted |
| --- | ---: | ---: | ---: |
| 0.88, fringe | 1.99945 | 1.99706 | 8.38931, conditional on a nonempty selection |
| 8.8 | 1.99674 | 2.00438 | 1.95918 |
| 880 | 1.99963 | 1.99973 | 1.99250 |

Omitting zeros excludes about 41.5% of the lowest-exposure replicates entirely
and strongly selects upward fluctuations. Even the high-count observed-weight
case is measurably biased. This diagnostic workaround is **not** nfit's current
reference estimator: the public Gaussian inverse-variance API explicitly rejects
zero-error observations, and the counting API includes covered zeros in exposure.

If true rates vary across the bin, weights define different targets. With rates
`[1, 1.5, 3, 5]` and the same relative exposures, pooling estimates 4.21591,
whereas an equal measured-observation mean estimates 2.625. Neither becomes an
estimate of a uniform coordinate integral without a sampling/response model.

The general application should keep its measurement-specific choices. For
independent Gaussian observations with **supplied**, known variances
`[1, 4, 25, 100]`, inverse variance reduces variance by 10.5625 times relative to
uniform averaging and has calibrated 95% coverage. It is appropriate for a
common continuous response, rather than count-dependent Poisson weighting.

Finite background precision adds another qualification. In a controlled
two-bank Gaussian limit with sample exposures `[1, 10]`, sample variances
`[2, 0.2]`, and background variances `[0.01, 4]`, known full inverse variance
reduces common-signal variance by 2.5656 times relative to sample-exposure
weighting. Thus sample exposure is not universally minimum variance for a
background-subtracted common signal. This assumes common signal, independent
bank errors, and known variances. Shared background/calibration requires a
covariance-aware estimator; sparse counts call for a joint count likelihood.
It does not justify silently replacing the existing exposure-weighted field
target or weighting a noisy observed difference by its own estimated variance.

## 2. Symmetry copies are the same observation

One source event with variance `v` copied `m` times into a final bin contributes
numerator variance `m² v`, not `m v`. Copying its exposure along with its signal
does not produce additional counting information. The signal estimator is
unchanged; treating copies as independent understates sigma by `sqrt(m)` in
the complete-overlap case: 2.4495 for six copies and 3.4641 for twelve.

The independent Poisson simulation confirms this analytically known variance.
At a mean of 30 primitive counts, treating six copies as independent gives
58.97% coverage for a nominal observed-error Gaussian 95% interval; twelve give
41.22%. Primitive-count Garwood coverage is 95.60%. Correct covariance alone
does not make the Gaussian approximation exact: even with one copy its coverage
is about 93.2% at this count level.

Overlap is not necessarily complete, especially at fringes. For independent
source populations with exposures `[0.1, 0.3, 1]` and accepted copy multiplicities
`[1, 2, 6]`, empirical variance is 1.66149. The source-aware reported mean
variance is 1.66312, versus 0.29873 when copies are treated as independent.
The analytic variance ratio is 5.5672. This is a weighted compound-count sum;
integer-Poisson intervals for its accumulated numerator are not justified.

Correct covariance does not necessarily make the duplicated estimator minimum
variance. Using exactly these same independent source counts once gives the
pooled Poisson maximum-likelihood estimator `sum(C_i)/sum(N_i)`, whose analytic
variance is 1.42857. The covariance-correct duplicated ratio has 1.1633 times
that variance because unequal multiplicities change the weights. Both are
unbiased for the common rate. The receipt compares them on the same simulated
counts: count-once empirical variance is 1.42868 and the empirical duplicated/
count-once variance ratio is 1.16296. It checks the count-once result against
the public counting reference.

This count-once result requires independent **unweighted** Poisson counts,
known exposures, a genuinely common response, and exactly the same source
populations. It is not a proposed weighted-DGS or symmetrized-field default.
A physical final bin would require the union of original-source trajectory
regions admitted by its symmetry operations, with counts and exposure included
once over that same union. Simply deduplicating event rows while retaining the
duplicated denominator would be wrong. Within-bin variation changes the target
when multiplicities change its weights; detector/energy corrections and shared
calibration need a separate statistical and response model.

The production same-bin correction and public shared-source reference pass an
additional exact final-cut check: two copies in each of two cached bins have
correct marginal numerator variances, but a cut combining both bins still needs
their cross covariance. Normalized variance is 8 with full dependencies, 4 with
only the within-bin correction, and 2 with independent-copy accumulation.

**Recommendation:** retain event identity and merge source coefficients before
squaring for the actual requested final bin. The existing optional
`within_bin_covariance` is statistically warranted for that bin, conditional on
the symmetry model. It is not a certificate that cached sample-bin diagonals
can later be added independently. Direct original-event binning onto the final
grid can recover the variance of that final bin; subsequent coarsening or
correlated fitting still requires dependencies. Actual symmetry of the sample
response is a separate physical assumption. Independently acquired events
related by symmetry remain independent; only transformed copies share identity.

## 3. Per-run Ei improves exposure only when it describes the acquisition

The independent one-detector model uses scattering angle 60°, energy transfer
0–40 meV, and `q_parallel = k_i - cos(60°) k_f` in Å⁻¹. It solves trajectory/bin
intersections analytically and integrates final energy, using the explicit toy
kinematic coefficient `E = 2.0721 k²` in meV. It does not use the production
integrator or its precision conventions.

For true run energies 59.7 and 60.3 meV, a 0.01-Å⁻¹ fringe bin at 2.68–2.69 Å⁻¹
is covered by only one of the trajectories. Reusing the first run's energy for
both gives exposure 4 instead of the correct 2 in the chosen arbitrary units.
At true rate 2, the per-run calculation averages 2.00386, while shared first-Ei
averages 1.00193. The analytic shared-energy bias is −1. Smaller errors from the
incorrect larger denominator do not indicate a better estimate. This is a
controlled boundary example, **not** a measured HYSPEC or SEQUOIA bias.

A counterexample uses the same measured Ei values but true common Ei of 60 meV.
Treating noisy calibration deviations as physical run changes then causes up to
32.85% exposure error over the true covered bins of this toy. A known correct
common calibrated energy has no such error. Shared-first Ei is not that
calibrated common estimate, either.

**Recommendation:** prefer per-run trajectories when genuine run-energy changes
are established and the measurements are sufficiently calibrated. When the
variation is primarily calibration noise, a shared or hierarchical physical
energy estimate and its uncertainty can be better. Validate monitor fitting,
Ei/T0 uncertainty and acceptance fringes together before choosing a universal
default. Per-run scalar Ei is distinct from pulse-resolved Ei. These simulations
do not establish which model describes the current instruments; existing
numerical per-run/first-run acceptance checks remain separate.

## 4. Dummy-angle averaging specifies a physical target

The existing 50 K/34° HYSPEC pulse-charge proxy has the largest recorded imbalance:
the last of eleven nominal angles receives 10.2149% instead of 9.0909% exposure.
Its source receipt and qualifications are retained by hash. The original angular
count populations are unavailable in merged QLab events; this calculation uses
only measured relative exposures and synthetic detector/energy rates.

For orientation-independent dummy rate, charge pooling is the independent
Poisson common-rate estimate. Equal-angle averaging raises variance by only
0.1378% with these actual relative charges. There is no material statistical
gain from changing this dataset's weighting under stationary-background truth.

If the heavily weighted angle has rate 4 and all others rate 1, charge averaging
targets 1.30645 and equal measured-angle averaging targets 1.27273. Both are
unbiased for their own target. Their difference 0.03372 saturates the existing
bound `0.0112402 × (max_rate - min_rate)` because this controlled variation is
aligned with the exposure imbalance. It is an intentionally adverse example,
not evidence that the actual orientation dependence has this size or form.
Statistical variance alone cannot select between these physical targets.

An equal average of measured angles is also distinct from a continuous uniform
orientation integral. For `1 + cos(angle)²` over 0–180°, the eleven endpoint-
including sample values average 1.54545, whereas the continuous mean is 1.5.
Endpoint/quadrature choices must therefore be explicit if that is the intended
dummy-average target; this does not implement continuous sample-rotation replay.

**Recommendation:** keep directional detector/energy replay. Offer original-angle
weighting only with a declared angular target and retained per-acquisition
provenance. The stationary-rate assumption supports charge pooling; intentional
uniform orientation averaging supports explicit angular weights. Real angular
variation and geometry/calibration transfer need measurements, not smaller
synthetic error bars. The earlier
[`physical-assumptions report`](hyspec-background-physical-assumptions.md)
already covers detector-versus-laboratory transfer and shared calibration; it
was not repeated or overridden.

## Low coverage needs more than an additive variance

At 0.88 expected independent unweighted counts, the pooled observed variance
correctly averages to the true variance. Nevertheless its Gaussian 95% interval
covers only 58.44%, principally because zero counts yield zero observed variance.
The final-bin Garwood interval covers 98.71% in simulation, versus exact Poisson
coverage 98.75%; discreteness makes it conservative. A covered zero remains
measured exposure with a positive rate upper limit, not an exactly known zero.

Use separate channels for numerator/counts, exposure, propagated variance and
model-qualified intervals. Do not turn an upper confidence limit into an
additive Gaussian sigma. The integer-count interval used here requires known
exposure and independent constant-weight counts. Heterogeneous corrections,
reused sources, background differences, or uncertain normalization need an
appropriate joint likelihood or primitive-source model; applying this interval
to those accumulated counts would be another false precision claim.

## Acceptance and follow-up

All analytic equalities, six-standard-error mean checks, bounded variance checks,
exact Poisson coverage enumeration, public-reference checks and zero-count
guards passed. Source/script hashes are recorded. The manual calculation takes
about three seconds after environment import caches are warm; this is not a
scientific-workflow performance benchmark.

Run:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/validate_dgs_estimator_alternatives.py
```

The strongest supported follow-up is complete primitive covariance for final
sample-symmetry cuts, together with model-qualified low-count intervals or
likelihoods. Pooling remains the natural independent counting estimator, while
Gaussian precision and coordinate integration retain their separate roles.
Per-run Ei and dummy-angle weighting need declared physical models and real
acquisition validation before default adoption. No default changes, project
rebuilds, calibration claims, continuous-rotation implementation or speedup
claims are made by this acceptance.
