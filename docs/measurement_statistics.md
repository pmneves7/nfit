# Measurement statistics

Choose the quantity a bin should estimate before choosing its weights. The
instrument adapter provides calibrated observations, units, support, and source
dependencies; the instrument name does not determine the statistical estimator.

`MeasurementContract` and `estimate_measurement_bin` provide public, instrument-
independent declarations and scalar reference estimates. Explicit declarations
also drive source-point binning, aligned histogram coarsening, slice reduction,
box profiles, waterfalls, fit preparation, and statistical CSV export. Existing
projects are not assigned an acquisition model from their instrument name.
Unmarked legacy data retain their saved behavior and record that provenance.

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
weights. A bounded GLS objective is available for fitting tracked observations;
the scalar inverse-variance estimator continues to require independence.

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
dense multidimensional covariance matrix. Histogram and point-data archives
retain optional numerical statistics and sparse source dependencies as immutable
array payloads. Project loading remains lazy: factors are read with their owning
dataset. Replacing primary values clears stale statistical declarations;
deliberate calibration propagates them through the dedicated scaling path.


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
Represented shared-source and uncertain-normalizer payloads propagate through
profiles. Other unsupported declarations require source replay explicitly.

A `reference_values` argument projects model predictions using observation
weights. The result records `profile_role="model_prediction"` and an observation
error for overlay; it does not establish model uncertainty or observed event counts.
Primary source factors are projected with the observation weights. Model overlays
do not acquire observed count statistics or model-parameter covariance payloads.
Smoothing is a plot preview with a diagonal error approximation. For declared
measurements it invalidates statistical payloads and requires source replay or
an explicit new uncertainty model before further scientific aggregation.

## Binning and source replay

`bin_measurement_points(data, bin_edges, contract=...)` bins original
`PointData4D` observations with a declared target. Count observations require
validated numerator, numerator variance, and exposure. Continuous means retain
four additive arrays: weighted value sum, propagated variance of that sum,
weight sum, and number of observations. Precision weights carry a shared numerical
scale; independently prepared sources reconcile that scale before combination.
Counts of observations are not neutron event counts.

`coarsen_measurement_histogram(data, bin_edges)` combines complete source cells.
Output edges must coincide with source edges in the same physical basis.
`combine_measurement_histograms(datasets)` pools compatible aligned payloads;
contracts and units must agree. Marked project rebinning uses these services.
Masks and measured zeros affect every additive channel consistently. A composite's
maximum source coverage is recorded explicitly; overlapping fractional support
geometry is not reconstructed from scalar coverage fractions.

Refinement, crossing existing cells, changed bases, and fractional/symmetry
redistribution require original measurements rather than a guess about a cell's
internal distribution. `replay_measurement_histogram(group, **binning)` bins a
raw-DGS or MDEvent group directly on the final grid and reuses valid reduced-event
caches. Give integrated dimensions a single final bin. Its provenance records
source identities, source configuration, binning and numerical policies.
Same-event variance follows the recorded copy policy; replay does not invent
unrecorded detector-calibration dependencies.

A measured-event background needs the same final-grid treatment. First obtain
`sample_final = replay_measurement_histogram(sample_group, **final_binning)`,
then call
`project_measured_background_mdevent(sample_group, background_group, sample_final)`.
Copies of each background event that meet in a final voxel combine before their
variance is calculated. This accounts for their within-final-bin covariance.
The replay output does not retain cross-voxel dependencies or certified additive
count statistics: a later profile of its cached pixels cannot reconstruct that
covariance. Subtract the background on the requested final grid, keeping sample
and background uncertainty separate; arbitrary later cuts require a new replay.
Shared calibration uncertainty remains outside this replay model.

For a sampled function, bin the original ordered nodes over explicit coordinate
intervals. The supported point workflow has one varying coordinate and constant
remaining coordinates; multidimensional interpolation needs a separate model.
An interval result retains nodal sensitivities, so adjacent intervals can share
uncertainty. Rebinning those interval estimates requires the original nodes.
Supply a stable `source_namespace` or metadata `measurement_source_id` when
constructing independent node factors. Reuse it only for the same original
observable. This preserves identity across repeated calls and selections;
project source-file identity is a fallback for an original file-backed series.

## Source dependencies in derived data

`SourceDependencies` stores bounded sparse sensitivities from stable independent
primitive IDs to the primary estimate. IDs shared by different datasets refer to
the same fluctuation and must have the same variance. Projection and signed
combination merge coefficients before squaring. The default payload budget is
256 MiB; exceeding it raises `SourceReplayRequired` instead of dropping covariance.
No dense covariance matrix is allocated for an entire project histogram.
`scale_measurement_data(data, factor)` propagates a known calibration through
statistics and source factors. Spectral conversions also update declared quantity
and units; they do not establish a count likelihood for signed responses.

`CountingDependencies` retains separate numerator and exposure sensitivities.
These primitives must be pooled before the ratio Jacobian is evaluated. A
zero-count cell's uncertain exposure can affect a pooled nonzero rate even when
its own first-order rate variance is zero. The result remains explicitly a
delta-method approximation. Primary-rate sensitivities alone cannot recover this
information.

Shared backgrounds propagate through aligned subtraction, interpolation and
signed region/Bragg integration. A transformed explicitly declared measurement
needs a new statistical target; it must not inherit a count or mean contract
describing the unsubtracted signal. Historical unmarked subtractions retain
their compatibility provenance. `estimate_measurement_region(...)` evaluates a
stated linear sum, with optional signed or volume coefficients and source factors.
It is a sum/integral target, not the box-profile mean.

Current DGS caches do not retain the source model needed to reconstruct shared
vanadium/monitor calibration uncertainty. Bragg tables record individual reflection
variances but flag unretained covariance between reflections; fitting such a table
requires primitive-source replay. These boundaries are explicit in the APIs.
Metadata-axis stacking currently requires original-source replay for declared
payloads; existing unmarked stacks retain their compatibility path.

## Fit objectives

`prepare_histogram_fit_points(data)` retains contracts, validated statistics,
exposure and represented primary source sensitivities. `FitDataset(...,
likelihood="gaussian")` preserves the default independent Gaussian objective;
positive standard errors are required. It rejects tracked shared observations
rather than counting them as independent. `likelihood="gaussian_gls"` uses the
represented covariance of a small selection, with a 128 MiB work budget. A
singular covariance requires fitting primitive sources; deterministic constraints
are not discarded through a pseudoinverse.

`likelihood="poisson_deviance"` includes measured count zeros, but requires an
explicit `PoissonCountModel(constant_weight=..., provenance=...).to_dict()` in
`data.metadata["poisson_count_model"]`. This certifies independent integer
primitive counts, known exposure and a constant event weight. Integer-looking
corrected intensities are insufficient. Heterogeneous weights, symmetry-expanded
copies, uncertain exposure and signed/background reconstructions are rejected.
Count parameter covariance uses expected Fisher information and is an
asymptotic estimate, not a low-count confidence interval.

For an audited independent constant-weight Poisson histogram, the slice viewer
also exposes 68.27% Garwood rate bounds. `poisson_interval_channels(view,
confidence=...)` accepts a final prepared slice mapping with the same model and
retained count numerator, numerator variance, and exposure. It validates that
those statistics reproduce the signal and observed error. Pool counts and
exposure first; bounds are not additive statistics and must not be averaged.

Saved dataset parameter `fit_likelihood` selects these objectives through the
project compiler and script API. Reports distinguish deviance from Gaussian
chi-squared. Aggregated GLS residual components are display diagnostics, not a new
fit or a spatial map of independent standard-normal residuals.

## Scientific count smoothing

`smooth_count_histogram(data, sigma, truncate=4, fill_missing=False)` is an
optional Python operation for a declared counting histogram with represented
`CountingDependencies`. Gaussian widths `sigma` are in native bin widths, one
per axis; a scalar applies to every axis. The finite kernel uses zero extension
at grid edges and omits masked or unexposed inputs. A work budget rejects kernels
that would require excessive sparse assignments. It is intended for selected
histograms, rather than materializing dependencies for an entire DGS volume.

For kernel coefficient $K_{ji}$ from input bin $i$ to output bin $j$, numerator
$C_i$ and exposure $N_i$ produce

$$I_j = \frac{\sum_i K_{ji} C_i}{\sum_i K_{ji} N_i}.$$

The kernel is dimensionless. $C_i/N_i$ and $I_j$ have the declared signal units;
the numerator and exposure retain their original units. This estimates a kernel-
weighted response with a changed resolution. It differs from blurring already
divided intensities when exposure varies. Numerator and exposure sensitivities
are projected through the same kernel; their represented covariance propagates
through the division and subsequent cuts. Adjacent output bins are correlated.
Unknown cross-bin event or calibration dependencies cannot be recovered from
diagonal errors; missing dependency payloads require source replay.

The returned histogram is immutable and the input remains available unchanged.
With `fill_missing=False`, bins lacking original support remain masked even if
neighbors supply a kernel estimate. Opting into `fill_missing=True` exposes those
estimates. The output coverage channel reports the measured fraction of the
in-grid smoothing kernel, rather than original geometric coverage; original
coverage remains a separate channel. Constant-weight Poisson declarations and
stale fitted channels are removed. No GUI display-smoothing setting invokes this
operation, and no existing scientific default changes.

## Statistical exports

`save_measurement_profile_csv`, `save_measurement_grid_csv` and prepared waterfall
export retain available additive statistics, masks, support and declaration
sidecars. `.csv.sources.npz` stores sparse factors when present; `.csv.json`
records their layout and assumptions. Legacy three-column profile and simple
map exports remain available. Diagnostic channels do not inherit a count contract
merely because their source dataset contains counts.
