# Fractional versus discrete counting bins

## Recommendation

Keep discrete event assignment for native DGS histograms. Treat fractional
assignment as an optional change of resolution and statistical target, with
matched event/exposure kernels and retained source covariance. It can improve
mean squared error for smooth fields; it is not uniformly more accurate for
the requested geometric bin average and cannot reconstruct an unmeasured
region without an additional model.

The native SEQUOIA/HYSPEC reducer currently deposits discrete events and
integrates discrete detector trajectories. Its `fractional_axes` argument is
used by the CORELLI delegate, not by the DGS implementation. Changing a general
fractional control therefore does not constitute a native DGS comparison.
No production DGS path, project, or default was changed by this study.

## Reproduce

```bash
MPLCONFIGDIR=/private/tmp/nfit-audit-matplotlib \
  /Users/pmneves/anaconda3/envs/nfit/bin/python \
  benchmarks/validate_dgs_fractional_binning.py
```

[Scalar results](dgs-fractional-binning.json) contain source hashes, seed,
targets, every output-bin bias/coverage, and public API acceptance checks.
The study uses 12,000 independent realizations per case, 640 fine cells,
16 output bins, and flat, sloping, narrow-peak, step, and hidden-peak fields.
The coordinate is dimensionless, ranges from -2 to 2, and has requested bin
width 0.25 and fine-cell width 0.00625.
The intensity is constant within each fine cell, so its geometric bin average
is known without a reconstruction or quadrature ambiguity. Exposure is known
and fixed. Counts are uncorrected independent Poisson observations; these are
not instrument measurements or a timing comparison.

The unequal-exposure fixture includes variation within individual bins, low
exposure at fringes and through the central feature, and one entirely
unobserved requested bin. Monte Carlo standard error for a single 95% coverage
estimate is about 0.2 percentage points. Source primitive independence and
known exposure are assumptions of this experiment.

## What is being estimated

Let `C_i` be the observed count in fine cell `i`, `N_i` its known exposure
(exposure units), and `I_i` its intensity (counts per exposure unit).
`C_i` has mean `N_i I_i`. For dimensionless assignment coefficients `K_bi`,
the estimate in output bin `b` is

```text
I_hat_b = sum_i K_bi C_i / sum_i K_bi N_i
target_b = sum_i K_bi N_i I_i / sum_i K_bi N_i
Var_hat(I_hat_b) = sum_i K_bi² C_i / (sum_i K_bi N_i)²
Cov_hat(I_hat_b, I_hat_c) = sum_i K_bi K_ci C_i / (N_b N_c)
N_b = sum_i K_bi N_i
```

Discrete `K` is membership inside the requested edges. Fractional `K`
interpolates between neighboring output centers, saturating at the outermost
centers. It has support beyond the requested edges of an interior bin. Both
methods apply their coefficients to numerator and exposure consistently;
variance uses their squares. Units of variance/covariance are squared intensity
units. The sum of fractional coefficients for each accepted source cell is one.

For uniform exposure, discrete estimates have the requested geometric-bin
average as their target. For unequal exposure, discrete pooling targets the
covered exposure-weighted mean; a geometric mean is a different estimand.
Fractional estimates target a different covered kernel average. A zero-exposure
fine cell supplies no observation to either estimator. Interpolated support is
not directly measured coverage in the originally requested cell.

## Bias, mean squared error, and coverage

Results below compare against the requested geometric bin average. Mean
squared error has squared intensity units; coverage is the frequency with which
the nominal `estimate ± 1.96 propagated_sigma` contains that average.

| Uniform exposure field | Discrete MSE | Fractional MSE | Discrete coverage | Fractional coverage |
| --- | ---: | ---: | ---: | ---: |
| Flat | 0.00794 | 0.00546 | 94.88% | 95.08% |
| Slope | 0.00806 | 0.00557 | 94.93% | 94.85% |
| Narrow peak | 0.00559 | 0.06004 | 94.91% | 77.09% |
| Step | 0.01167 | 0.00851 | 95.02% | 93.72% |

The flat-field fractional MSE is 31% lower. The narrow-peak fractional MSE is
10.7 times larger: its peak-bin target drops from 6.292 to 5.529 while adjacent
bins rise. Fractional intervals cover their *own kernel target* about 95% of
the time in this high-count case; they under-cover the requested bin average
because of target bias, not an error in the squared-coefficient variance.
Smoother appearance is therefore not itself evidence of better bin estimates.

In the unequal-exposure flat-field case, fractional MSE falls from 0.536 to
0.261 and fringe Wald coverage improves from 87.47% to 92.17%. It still does
not establish 95% interval coverage. For the unweighted discrete Poisson sums,
exact Garwood intervals cover their exposure-weighted target 97.10% on average
over exposed bins. Fractional weighted sums do not have an ordinary Poisson
count distribution, so this exact Garwood assertion cannot be transferred to
them. Corrected DGS event weights likewise require the appropriate compound
counting model rather than this unweighted-count shortcut.

The hidden peak is a direct coverage counterexample. The missing requested bin
has true geometric mean **4.590**. Discrete binning leaves it unmeasured;
fractional interpolation supplies a target of **1.042** from nearby measured
cells. Neither method has observed the hidden peak. Displaying the interpolated
value as a directly measured estimate would conceal that limitation. Comparing
MSE over different sets of available bins must also retain this abstention.

## Covariance matters after fractional assignment

With uniform exposure and a flat field, summing the entire fractional histogram
while ignoring cross-bin covariance retains only **68.74%** of the correct
variance. The nominal 95% region interval then covers **89.96%** of realizations,
versus **95.27%** with source covariance. The complete region numerator and
exposure are otherwise unchanged from discrete assignment. There is no gain
in its information: nearby fractional bins share the same observations.

The unequal-exposure flat-field fixture retains only **58.07%** of the correct
region variance if correlations are discarded; region coverage falls from
95.10% to 86.89%. The script verifies that `project_source_dependencies`
reproduces the full covariance-aware region variance and that NumPy/Numba
`NDRebin` match the candidate kernel's means and diagonal errors.

An observed zero count has zero accumulated event variance. It does not imply
a noiseless true rate. At exposure 0.5, zero unweighted counts have a 95% Garwood
interval from 0 to 7.378 counts per exposure unit. Missing exposure and covered
zero counts must remain distinct.

## Implementation audit and focused fixes

- `NDRebin` propagates squared fractional and averaging coefficients correctly
  within each bin, assuming independent input observations. Its legacy result
  does not retain cross-bin primitive dependencies. Subsequent independent-bin
  integrations can therefore understate uncertainty after fractional splitting.
- Explicit measurement-contract point binning rejects fractional assignment
  requiring replay rather than silently discarding covariance.
- Native DGS fractional deposition would require an exposure integral of the
  same kernel along detector trajectories. Splitting events while keeping the
  current discrete denominator, or interpolating final normalized intensities,
  does not implement the estimator above.
- `smooth_count_histogram` already provides an explicit kernel-target operation
  with numerator/exposure dependency propagation for bounded selected
  histograms. Its source-replay requirements and missing-support controls are
  appropriate foundations for optional scientific smoothing. It is not a
  replacement for a full-volume fractional DGS trajectory integrator.
- A demonstrated general-rebin boundary bug was fixed: a point exactly at the
  final uniform edge formerly split equally into the last two bins, whereas
  identical explicit edges correctly saturated the last bin. NumPy and both
  compiled kernels now agree at exact and neighboring representable endpoints.
- A demonstrated NumPy streaming sum bug was fixed: masking an individual
  batch's missing bins modified its retained numerator sum to NaN. Final display
  masking now uses a separate sum payload, so disjoint batches combine correctly
  while covered zeros and absent bins remain distinct.
- General numerical cache signatures are versioned so existing affected
  results are replayed. Native raw-DGS/MDE/CORELLI group identities and reduced
  event caches are unaffected by that general-rebin rule version.

## Future optional implementation gates

1. Declare whether the requested output is a geometric bin average, an
   exposure-weighted measured response, or a kernel-weighted response. Use that
   declared target in accuracy checks and model predictions.
2. If fractional native DGS is added, share one coefficient definition between
   event deposition and trajectory integration. Preserve true measured support
   separately from interpolation support.
3. Retain primitive covariance or replay original events for final cuts, wider
   bins, symmetry copies, and fits. A diagonal-only cache cannot reconstruct
   correlations introduced by prior fractional assignment.
4. Test bias and interval coverage on narrow features, gradients, coverage
   holes, corrected event weights, and shared calibrations/backgrounds before
   selecting defaults. This study establishes conditional truths, not actual
   instrument calibration or a universally optimal estimator.
