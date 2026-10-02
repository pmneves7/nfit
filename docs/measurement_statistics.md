# Measurement statistics

Choose the quantity a bin should estimate before choosing its weights. The
instrument adapter provides calibrated observations, units, support, and source
dependencies; the instrument name does not determine the statistical estimator.

`MeasurementContract` and `estimate_measurement_bin` provide public, instrument-
independent declarations and scalar reference estimates. Existing project
binnings, viewers, fits, and exports retain their current behavior until the
workflow integration described in [Planned features](planned_features.md#statistical-binning-and-reduction-recipes).
The API does not automatically attach contracts to existing projects. Regular
and rotated box profiles already use these declarations for validated count
statistics and legacy independent precision means; other workflow integration
remains staged.

## Choose a target

| Kind and estimator | Target and assumptions | Inputs |
| --- | --- | --- |
| `counting`, `exposure_pool` | Common intensity with known exposure; under heterogeneity, an exposure-weighted intensity | Corrected numerators, numerator variances, exposures |
| `continuous`, `inverse_variance_mean` | Common value measured independently with supplied Gaussian variances | Observations and strictly positive variances |
| `continuous`, `uniform_mean` | Equal-weight mean with propagated supplied variances | Observations and nonnegative variances |
| `sampled_function`, `coordinate_mean` | Mean over a coordinate interval under explicit linear interpolation | Ordered coordinates, observations, variances, interval |
| `sampled_function`, `coordinate_integral` | Integral over that interval | Same inputs as the coordinate mean |
| `linear_reconstruction`, `linear_sum` | Stated linear combination; signed values and coefficients are allowed | Observations, variances, coefficients, represented dependencies |

For example, repeated measurements at one temperature can estimate a common
susceptibility with precision weights. An average susceptibility over a
temperature interval is a different quantity: dense sampling of one part of
the interval must not give that part extra coordinate weight. A bin width is
not automatically an exposure, integration weight, or instrument resolution.

## Counts and measured zeros

```python
from nfit import MeasurementContract, estimate_measurement_bin

counts = MeasurementContract(
    kind="counting", estimator="exposure_pool",
    quantity="count rate", value_units="counts/s", exposure_units="s",
)
result = estimate_measurement_bin(
    counts, values=[4, 0], variances=[4, 0], exposure=[1, 9],
)
print(result.value, result.standard_error)  # 0.4, 0.2
```

Here `values` are count numerators rather than already divided intensities.
Corrected or weighted events require their independently propagated numerator
variance; it must not be reconstructed from corrected counts. Positive exposure
with zero counts contributes exposure and zero observed variance. Zero exposure
means unmeasured support and is omitted by default. It does not establish a
geometric coverage fraction. `result.numerator`, `numerator_variance`, and
`exposure` retain pooled statistics.

Zero observed variance is not certainty about an unknown rate. The separate
Poisson confidence-interval API has narrower independent, constant-weight
assumptions; see [Covered empty cells and confidence intervals](physics_conventions.md#covered-empty-cells-and-confidence-intervals).
The count estimator does not impose Poisson errors on signed background
differences or reconstructed measurements.

## Continuous measurements and coordinate averages

```python
common = MeasurementContract(
    kind="continuous", estimator="inverse_variance_mean",
    quantity="susceptibility", value_units="emu/mol",
)
estimate = estimate_measurement_bin(common, [2, 10], [1, 9])
print(estimate.value, estimate.variance)  # 2.8, 0.9

interval_mean = MeasurementContract(
    kind="sampled_function", estimator="coordinate_mean",
    quantity="susceptibility", value_units="emu/mol", coordinate_units="K",
    interpolation="linear",
)
estimate = estimate_measurement_bin(
    interval_mean, [0, 1, 4], [1, 1, 1],
    coordinates=[0, 1, 4], interval=(0, 4),
)
print(estimate.value, estimate.variance)  # 2.0, 0.40625
```

The inverse-variance result reports `chi2` and `degrees_of_freedom` for the
common-value model. Large disagreement can indicate a varying response,
underestimated errors, or missing calibration dependencies. The API does not
automatically inflate variance or fit a random-effects model. Correlated
observations need generalized least squares rather than diagonal precision
weights; that estimator is not implemented here.

For interval estimates, coordinates must be finite and strictly increasing.
Repeated coordinates must first be combined under their measurement model.
Linear interpolation is explicit, including between widely spaced valid nodes;
no interpolation-model error is estimated. Missing or masked nodes remove both
adjacent segments, and extrapolation is prohibited. The same node used by
neighboring segments remains one observation: its integral coefficients are
added before propagating variance.

The default `partial_support="reject"` returns a missing estimate when the
requested interval is not fully covered. `partial_support="covered_only"`
estimates the covered interval only and reports `coverage_fraction` and its
coordinate length in `support`. Uncovered regions are never filled with zero.
An integral has units `value_units * coordinate_units`; a mean retains the
observed quantity's units.

## Shared sources and reconstructed measurements

`dependence="shared_sources"` requires a list of `SourceTerm` objects for each
observation. Each primitive source has a stable ID, known variance, and a
coefficient converting its fluctuation into that observation's units. Primitive
sources with different IDs are declared independent; identical IDs refer to
the same source. Terms must reproduce each included observation's supplied
variance. Inconsistent source variances are rejected.

```python
from nfit import SourceTerm

reconstruction = MeasurementContract(
    kind="linear_reconstruction", estimator="linear_sum",
    quantity="background difference", value_units="U",
    dependence="shared_sources",
)
background = [SourceTerm("same background", coefficient=1, variance=9)]
estimate = estimate_measurement_bin(
    reconstruction, [-2, -2], [9, 9], coefficients=[1, -1],
    source_terms=[background, background],
)
print(estimate.value, estimate.variance)  # 0.0, 0.0
```

The same-source coefficients cancel. Independent-source errors would not.
Partially shared backgrounds and calibration sources can be represented by
multiple terms per observation. Signed signals do not require negative
variances, a variance floor, or a Poisson model.

Linear reconstructions default to `missing="reject"`: a missing required
background must not silently produce the sample alone. Explicitly choosing
`missing="omit"` changes the target to the available terms. Other kinds default
to omission; masks and missing observations remove their corresponding support.
All-missing results have NaN value/variance and `measured=False`.

## Uncertain normalization

For counts, `normalizer="uncertain"` requires an `exposure_variance` per
observation. Independent declarations propagate independent exposure errors.
Shared declarations require `exposure_source_terms` as well as numerator terms;
reuse IDs to retain both shared calibration errors and numerator/denominator
covariance. A common calibration error must not shrink as if independently
remeasured in every run.

The returned uncertainty is labeled `delta_method`: it is first-order ratio
propagation and can fail for poorly constrained denominators. It is not an exact
likelihood, confidence interval, or proof that the plug-in ratio is unbiased.
Known-normalizer contracts reject uncertainty inputs instead of silently
ignoring them. Canonical equations are in
[Measurement estimator contracts](physics_conventions.md#measurement-estimator-contracts).

## Serialization and workflow boundaries

`MeasurementContract.to_dict()` and `from_dict()` preserve every scientific
choice in a versioned JSON-compatible declaration. Unknown versions and fields
fail explicitly. Units are nonempty labels (`"1"` for dimensionless quantities);
the API does not convert units or infer compatibility from matching text alone.

The returned immutable `MeasurementEstimate` contains value, propagated variance,
standard error, support, included/excluded observation counts, and relevant
pooled statistics. The scalar reference routines use sparse source terms, not a
dense multidimensional covariance matrix. Project integration must retain or
replay these dependencies through binning, cuts, fits and exports before claiming
end-to-end equivalence under changes of intermediate grids.


## Histogram box profiles

`histogram_box_profiles(view, values, errors, extents, angle=0, ...)` accepts a
prepared two-dimensional slice from `MDHistoSliceViewer.slice_arrays()`.
`extents` is `(xmin, xmax, ymin, ymax)` in the displayed coordinate units;
rotation uses that coordinate plane. Regular cuts retain native coordinates;
rotated cuts assign selected pixel centers to projected bins. No partial-pixel
redistribution or missing source covariance is inferred.

```python
from nfit import histogram_box_profiles, save_measurement_profile_csv

# view is an existing prepared slice; these calls do not construct Qt widgets.
profiles = histogram_box_profiles(
    view, view["signal"], view["errors"], (0, 0.6, 0, 30),
    coverage_threshold=0.1,
)
if profiles.x_measurement is not None:
    save_measurement_profile_csv(
        "x_cut.csv", profiles.x_measurement, coordinate_name="x",
        coordinate_unit="r.l.u.",
    )
```

`profiles.x` and `.y` retain the `(coordinate, value, standard_error)` tuple API.
`.x_measurement` and `.y_measurement` additionally contain an immutable
one-dimensional `MDHistoData` and its `MeasurementContract`. Count payloads retain
C,V,N, source contribution semantics, masks and geometric coverage. The explicit
normalization exposure does not substitute for geometric coverage. Count statistics
are used only for the validated primary signal; other channels use their separate
legacy precision-mean treatment. A saved compatible contract is preserved.
Unsupported shared-source, uncertain-normalizer or other estimator declarations
fail explicitly instead of silently choosing an independent model.

A `reference_values` argument projects model predictions using observation
weights. The result records `profile_role="model_prediction"` and an observation
error for overlay; it does not establish model uncertainty or observed event counts.
Only diagonal uncertainty is propagated here. Full dependency propagation, region
estimation and fitting likelihoods remain separate workflow requirements.
