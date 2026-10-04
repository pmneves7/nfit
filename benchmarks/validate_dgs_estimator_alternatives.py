"""Bounded, independent statistical truths for optional DGS estimators.

Run manually with the nfit interpreter; no instrument files or Mantid are used.
These are model-conditional acceptance calculations, not real-data calibration
claims. The incident-energy example uses a one-detector analytic trajectory,
not the production trajectory integration. No application defaults are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, poisson

from nfit.event_covariance import accumulate_copy_covariance
from nfit.histogram_statistics import poisson_rate_interval
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_statistics import SourceTerm, estimate_measurement_bin

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261003
REPLICATES = 120_000
CONFIDENCE = 0.95
Z = float(norm.ppf((1 + CONFIDENCE) / 2))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(values, variances, target):
    values, variances = np.asarray(values), np.asarray(variances)
    valid = np.isfinite(values) & np.isfinite(variances)
    values, variances = values[valid], variances[valid]
    sigma = np.sqrt(variances)
    return {
        "target": float(target),
        "mean": float(values.mean()),
        "bias": float(values.mean() - target),
        "mean_monte_carlo_standard_error": float(values.std(ddof=1) / np.sqrt(values.size)),
        "empirical_variance": float(values.var(ddof=1)),
        "mean_reported_variance": float(variances.mean()),
        "wald_95_percent_coverage": float(np.mean(np.abs(values - target) <= Z * sigma)),
        "included_replicates": int(values.size),
        "excluded_replicates": int(valid.size - values.size),
    }


def _assert_unbiased(summary, *, analytic_variance=None):
    assert abs(summary["bias"]) <= 6 * summary["mean_monte_carlo_standard_error"] + 1e-12, summary
    if analytic_variance is not None:
        # Monte Carlo checks have intentionally bounded, fixed-seed tolerance.
        assert abs(summary["empirical_variance"] / analytic_variance - 1) < 0.035, summary
        assert abs(summary["mean_reported_variance"] / analytic_variance - 1) < 0.035, summary


def _count_contract(dependence="independent"):
    return MeasurementContract(kind="counting", estimator="exposure_pool",
        quantity="constant corrected rate", value_units="counts/exposure",
        exposure_units="exposure", dependence=dependence)


def _continuous_contract(estimator):
    return MeasurementContract(kind="continuous", estimator=estimator,
        quantity="common response", value_units="response")


def _exact_interval_coverage(mean_count, copies=1):
    counts = np.arange(int(poisson.ppf(1 - 1e-12, mean_count)) + 1)
    low, high = poisson_rate_interval(copies * counts, copies, confidence=CONFIDENCE)
    return float(np.sum(poisson.pmf(counts, mean_count) * ((low <= mean_count) & (mean_count <= high))))


def _pooling(rng):
    result = {}
    for label, scale in (("fringe", 0.1), ("uneven_coverage", 1.0), ("high_counts", 100.0)):
        exposure = scale * np.array([0.1, 0.3, 1.0, 3.0])
        rate = 2.0
        counts = rng.poisson(rate * exposure, size=(REPLICATES, len(exposure)))
        rates = counts / exposure
        total = counts.sum(axis=1)
        pooled = total / exposure.sum()
        pooled_variance = total / exposure.sum()**2
        uniform = rates.mean(axis=1)
        uniform_variance = (counts / exposure**2).sum(axis=1) / len(exposure)**2
        # This commonly attempted workaround is deliberately NOT nfit's current
        # reference API, which rejects zero-error inverse-variance measurements.
        # It illustrates the consequences of omitting zeros to define precision.
        precision = np.zeros_like(rates)
        np.divide(exposure**2, counts, out=precision, where=counts > 0)
        denominator = precision.sum(axis=1)
        observed_iv = np.full(REPLICATES, np.nan)
        observed_iv_variance = np.full(REPLICATES, np.nan)
        np.divide((precision * rates).sum(axis=1), denominator, out=observed_iv, where=denominator > 0)
        np.divide(1, denominator, out=observed_iv_variance, where=denominator > 0)
        # Expected Poisson precision is N/rate, so its normalized weights are
        # exactly N/sum(N); this equality does not require knowing the rate.
        np.testing.assert_allclose(rates @ (exposure / exposure.sum()), pooled, rtol=1e-14, atol=1e-14)
        p = _summary(pooled, pooled_variance, rate)
        u = _summary(uniform, uniform_variance, rate)
        _assert_unbiased(p, analytic_variance=rate / exposure.sum())
        _assert_unbiased(u, analytic_variance=rate * np.sum(1 / exposure) / len(exposure)**2)
        lower, upper = poisson_rate_interval(total, exposure.sum(), confidence=CONFIDENCE)
        expected_coverage = _exact_interval_coverage(rate * exposure.sum())
        observed_coverage = float(np.mean((lower <= rate) & (rate <= upper)))
        assert abs(expected_coverage - observed_coverage) < 0.006
        p["garwood_95_percent_coverage"] = observed_coverage
        p["analytic_garwood_95_percent_coverage"] = expected_coverage
        result[label] = {
            "exposure": exposure.tolist(), "expected_total_counts": float(rate * exposure.sum()),
            "pooled": p, "uniform": u,
            "observed_inverse_variance_omitting_zeros": _summary(observed_iv, observed_iv_variance, rate),
            "uniform_to_pooled_analytic_variance_ratio": float(exposure.sum() * np.sum(1 / exposure) / len(exposure)**2),
            "any_zero_observation_fraction": float(np.mean(np.any(counts == 0, axis=1))),
            "zero_total_fraction": float(np.mean(total == 0)),
        }
        # Public scientific reference checked against independent direct sums.
        public = estimate_measurement_bin(_count_contract(), counts[0], counts[0], exposure=exposure)
        np.testing.assert_allclose([public.value, public.variance], [pooled[0], pooled_variance[0]], rtol=1e-14)
        zeros = estimate_measurement_bin(_count_contract(), np.zeros(4), np.zeros(4), exposure=exposure)
        assert zeros.included == 4 and zeros.support == exposure.sum() and zeros.variance == 0
        zero_lower, zero_upper = poisson_rate_interval(0, exposure.sum(), confidence=CONFIDENCE)
        assert zero_lower == 0 and zero_upper > 0
        result[label]["covered_zero_count_95_percent_upper_limit"] = float(zero_upper)
        try:
            estimate_measurement_bin(_continuous_contract("inverse_variance_mean"), [0, 1], [0, 1])
        except ValueError as error:
            assert "positive variances" in str(error)
        else:
            raise AssertionError("Zero observed count error must not become infinite Gaussian precision")

    # Varying rates change the target rather than simply changing efficiency.
    exposure = np.array([0.1, 0.3, 1.0, 3.0])
    rates = np.array([1.0, 1.5, 3.0, 5.0])
    counts = rng.poisson(exposure * rates, size=(REPLICATES, 4))
    target_pool = float(exposure @ rates / exposure.sum())
    target_uniform = float(rates.mean())
    pool = _summary(counts.sum(axis=1) / exposure.sum(), counts.sum(axis=1) / exposure.sum()**2, target_pool)
    uniform = _summary((counts / exposure).mean(axis=1), (counts / exposure**2).sum(axis=1) / 16, target_uniform)
    _assert_unbiased(pool)
    _assert_unbiased(uniform)
    result["varying_true_rate"] = {"rates": rates.tolist(), "exposure": exposure.tolist(),
        "pooled": pool, "uniform": uniform,
        "pooled_bias_if_uniform_target_requested": target_pool - target_uniform}

    # A continuous measurement with supplied Gaussian uncertainty is a different
    # model: known inverse variance, not Poisson counts used as their own weights.
    variance = np.array([1.0, 4.0, 25.0, 100.0])
    response = rng.normal(2, np.sqrt(variance), size=(REPLICATES, 4))
    weights = (1 / variance) / np.sum(1 / variance)
    weighted_variance = float(np.sum(weights**2 * variance))
    inverse = _summary(response @ weights, np.full(REPLICATES, weighted_variance), 2)
    uniform = _summary(response.mean(axis=1), np.full(REPLICATES, variance.sum() / 16), 2)
    _assert_unbiased(inverse, analytic_variance=weighted_variance)
    _assert_unbiased(uniform, analytic_variance=variance.sum() / 16)
    public = estimate_measurement_bin(_continuous_contract("inverse_variance_mean"), response[0], variance)
    np.testing.assert_allclose([public.value, public.variance], [response[0] @ weights, weighted_variance])
    result["known_variance_gaussian"] = {"inverse_variance": inverse, "uniform": uniform,
        "uniform_to_inverse_variance_ratio": float((variance.sum() / 16) / weighted_variance)}
    # Finite background noise changes precision, even when the sample exposure
    # is known. This is a supplied-variance Gaussian limit, not an endorsement
    # of plugging noisy observed Poisson differences into precision weights.
    exposure = np.array([1.0, 10.0])
    sample_variance = 2 / exposure
    background_variance = np.array([0.01, 4.0])
    total_variance = sample_variance + background_variance
    response = rng.normal(2, np.sqrt(total_variance), size=(REPLICATES, 2))
    exposure_weights = exposure / exposure.sum()
    precision_weights = (1 / total_variance) / np.sum(1 / total_variance)
    exposure_var = float(exposure_weights**2 @ total_variance)
    precision_var = float(precision_weights**2 @ total_variance)
    exposure_summary = _summary(response @ exposure_weights, np.full(REPLICATES, exposure_var), 2)
    precision_summary = _summary(response @ precision_weights, np.full(REPLICATES, precision_var), 2)
    _assert_unbiased(exposure_summary, analytic_variance=exposure_var)
    _assert_unbiased(precision_summary, analytic_variance=precision_var)
    result["known_variance_background_limited_gaussian"] = {
        "sample_exposure": exposure.tolist(), "sample_variance": sample_variance.tolist(),
        "background_variance": background_variance.tolist(),
        "sample_exposure_weighted": exposure_summary, "known_precision_weighted": precision_summary,
        "exposure_to_precision_analytic_variance_ratio": exposure_var / precision_var,
        "scope": "Common signal, independent Gaussian bank errors and known full variances. Shared background/calibration needs GLS or a joint count model.",
    }
    return result


def _symmetry(rng):
    result = {}
    for label, mean in (("fringe", 0.8), ("central", 30.0)):
        count = rng.poisson(mean, REPLICATES)
        result[label] = {}
        for copies in (1, 6, 12):
            # Signal and exposure copy together; duplication cannot add counts.
            independent = _summary(count.astype(float), count / copies, mean)
            correlated = _summary(count.astype(float), count.astype(float), mean)
            _assert_unbiased(correlated, analytic_variance=mean)
            independent["misapplied_garwood_coverage_using_duplicated_counts"] = _exact_interval_coverage(mean, copies)
            correlated["primitive_count_garwood_coverage"] = _exact_interval_coverage(mean)
            result[label][str(copies)] = {"independent_copies": independent,
                "source_correlated": correlated, "analytic_sigma_understatement": float(np.sqrt(copies))}

    # Two copies per cached bin. A final profile includes both bins, so fixing
    # only their marginal variances is still insufficient for the final cut.
    source_variance = 8.0
    cached_variance = np.array([2 * source_variance, 2 * source_variance])
    stats = accumulate_copy_covariance(np.array([[0], [0], [1], [1]]), [source_variance], cached_variance)
    np.testing.assert_equal(cached_variance, [4 * source_variance, 4 * source_variance])
    public = estimate_measurement_bin(_count_contract("shared_sources"), [16, 16], cached_variance,
        exposure=[2, 2], source_terms=[[SourceTerm("primitive", 2, source_variance)]] * 2)
    assert public.value == 8 and public.variance == source_variance
    result["final_cut_across_cached_bins"] = {
        "signal": public.value,
        "correct_normalized_variance": public.variance,
        "independent_copy_normalized_variance": 4 * source_variance / 4**2,
        "within_bin_only_normalized_variance": float(cached_variance.sum() / 4**2),
        "production_same_bin_diagnostics": stats,
        "exact_public_reference_merges_primitive_coefficients": True,
    }
    multiplicities = np.array([1, 2, 6])
    exposures = np.array([0.1, 0.3, 1.0])
    rate = 2.0
    counts = rng.poisson(rate * exposures, size=(REPLICATES, 3))
    denominator = float(exposures @ multiplicities)
    estimate = counts @ multiplicities / denominator
    source_variance = counts @ multiplicities**2 / denominator**2
    copy_variance = counts @ multiplicities / denominator**2
    correct = _summary(estimate, source_variance, rate)
    expected = float(rate * (exposures @ multiplicities**2) / denominator**2)
    _assert_unbiased(correct, analytic_variance=expected)
    # Same independent source populations and same simulated counts, now each
    # counted once. With known exposures and a common unweighted Poisson rate,
    # this is the pooled MLE. A physical DGS final bin would additionally need
    # the union of accepted original-source trajectory regions/exposures; this
    # calculation does not deduplicate a weighted or spatially varying field.
    unique_exposure = float(exposures.sum())
    unique_counts = counts.sum(axis=1)
    unique = _summary(unique_counts / unique_exposure, unique_counts / unique_exposure**2, rate)
    unique_expected_variance = rate / unique_exposure
    _assert_unbiased(unique, analytic_variance=unique_expected_variance)
    public_unique = estimate_measurement_bin(_count_contract(), counts[0], counts[0], exposure=exposures)
    np.testing.assert_allclose([public_unique.value, public_unique.variance],
        [unique_counts[0] / unique_exposure, unique_counts[0] / unique_exposure**2], rtol=1e-14)
    analytic_variance_ratio = expected / unique_expected_variance
    assert analytic_variance_ratio >= 1  # Cauchy-Schwarz for positive exposure.
    result["mixed_fringe_overlap"] = {
        "exposures": exposures.tolist(), "accepted_copy_multiplicity": multiplicities.tolist(),
        "correlated": correct, "independent_copies": _summary(estimate, copy_variance, rate),
        "analytic_correlated_to_independent_variance_ratio": float((exposures @ multiplicities**2) / (exposures @ multiplicities)),
        "interval_scope": "Unequal effective source weights; integer-Poisson Garwood intervals do not apply to this sum.",
        "unique_source_pooled_mle": unique,
        "analytic_duplicated_to_unique_source_variance_ratio": float(analytic_variance_ratio),
        "empirical_duplicated_to_unique_source_variance_ratio": correct["empirical_variance"] / unique["empirical_variance"],
        "unique_source_scope": "Same independent unweighted Poisson source populations/counts, known exposure and true common response. Not a weighted-DGS or varying-field default. Physical final-bin source selection/trajectory-union exposure, varying targets and calibration require a separate model.",
    }
    return result


def _trajectory_exposure(ei, edges, *, theta_degrees=60.0, energy_window=(0.0, 40.0)):
    """Exact toy path acceptance in meV, flat efficiency and no flux correction.

    q_parallel = k_i - cos(theta) k_f [angstrom^-1], E = 2.0721 k^2
    [meV]. For positive cos(theta), solve bin intersections analytically in k_f
    and integrate dE_f. No midpoint quadrature/production helper is used.
    """
    constant = 2.0721  # Explicit toy kinematic coefficient, not a parity constant.
    cosine = np.cos(np.deg2rad(theta_degrees))
    ki = np.sqrt(ei / constant)
    physical_low = np.sqrt((ei - energy_window[1]) / constant)
    physical_high = np.sqrt((ei - energy_window[0]) / constant)
    low = np.maximum(physical_low, (ki - np.asarray(edges[1:])) / cosine)
    high = np.minimum(physical_high, (ki - np.asarray(edges[:-1])) / cosine)
    return np.where(high > low, constant * (high**2 - low**2), 0.0)


def _energy(rng):
    energies = np.array([59.7, 60.3])
    edges = np.arange(2.64, 3.66, 0.01)
    exposures = np.stack([_trajectory_exposure(ei, edges) for ei in energies])
    correct = exposures.sum(axis=0)
    shared = 2 * exposures[0]
    supported = correct > 1e-12
    both = supported & (shared > 1e-12)
    fringe = supported & (correct <= np.quantile(correct[supported], 0.1))
    index = int(np.flatnonzero(both & fringe)[np.argmax(np.abs(correct[both & fringe] / shared[both & fringe] - 1))])
    # Scale exposure to an expected four counts, keeping model bias prominent.
    charge = 2 / correct[index]
    true_rate = 2.0
    true_exposure = charge * correct[index]
    false_exposure = charge * shared[index]
    count = rng.poisson(true_rate * true_exposure, REPLICATES)
    per_run = _summary(count / true_exposure, count / true_exposure**2, true_rate)
    shared_rate = _summary(count / false_exposure, count / false_exposure**2, true_rate)
    _assert_unbiased(per_run, analytic_variance=true_rate / true_exposure)
    deterministic_bias = true_rate * (true_exposure / false_exposure - 1)
    assert abs(shared_rate["bias"] - deterministic_bias) < 6 * shared_rate["mean_monte_carlo_standard_error"]
    # Counterexample: measured energy varies but actual energy does not. Then
    # neither nominal noisy estimate is the physical exposure truth; averaging
    # calibrated energy is not the same as averaging nonlinear acceptances.
    common_truth = 2 * _trajectory_exposure(60.0, edges)
    usable = common_truth > 1e-12
    noise_mismatch = np.abs(correct[usable] / common_truth[usable] - 1)
    return {
        "scope": "Independent analytic one-detector acceptance model; not a production trajectory test or real-data calibration claim.",
        "theta_degrees": 60.0, "energy_transfer_window_meV": [0, 40],
        "true_incident_energies_meV": energies.tolist(), "q_parallel_edges_inverse_angstrom": edges.tolist(),
        "selected_fringe_bin": [float(edges[index]), float(edges[index + 1])],
        "true_exposure": true_exposure, "shared_first_energy_exposure": false_exposure,
        "per_run": per_run, "shared_first_energy": shared_rate,
        "analytic_shared_first_energy_bias": deterministic_bias,
        "bins_physically_covered_but_first_energy_uncovered": int(np.count_nonzero(supported & ~both)),
        "fringe_bins": int(np.count_nonzero(fringe)),
        "calibration_noise_counterexample": {
            "true_common_Ei_meV": 60.0, "measured_Ei_meV": energies.tolist(),
            "maximum_relative_exposure_error_per_run_over_true_covered_bins": float(noise_mismatch.max()),
            "using_known_common_calibrated_energy_relative_exposure_error": 0.0,
        },
    }


def _angle_averaging(rng):
    receipt = ROOT / "benchmarks/results/hyspec-background-angle-exposure.json"
    old = json.loads(receipt.read_text())
    source = old["sources"]["Ei15meV_50K_240Hz_s2_34_bkg.nxs"]
    weights = np.asarray(source["charge_fractions"])
    equal = np.full(len(weights), 1 / len(weights))
    # Use already measured *relative* angle exposures; absolute low-count toy
    # exposure is deliberately synthetic, not counts observed in this source.
    exposures = 22 * weights
    cases = {
        "orientation_independent": np.full(len(weights), 2.0),
        "orientation_varies": np.where(weights > equal, 4.0, 1.0),
    }
    result = {"real_charge_proxy_receipt_sha256": _sha(receipt),
        "relative_exposure_source": "Ei15meV_50K_240Hz_s2_34_bkg.nxs",
        "qualification": old["qualification"], "synthetic_total_exposure": float(exposures.sum()),
        "cases": {}}
    for label, rates in cases.items():
        counts = rng.poisson(exposures * rates, size=(REPLICATES, len(weights)))
        charge_target = float(weights @ rates)
        equal_target = float(equal @ rates)
        charge_average = counts.sum(axis=1) / exposures.sum()
        charge_variance = counts.sum(axis=1) / exposures.sum()**2
        equal_average = (counts / exposures).mean(axis=1)
        equal_variance = (counts / exposures**2).sum(axis=1) / len(weights)**2
        charge = _summary(charge_average, charge_variance, charge_target)
        uniform = _summary(equal_average, equal_variance, equal_target)
        _assert_unbiased(charge, analytic_variance=float(weights @ rates / exposures.sum()))
        _assert_unbiased(uniform, analytic_variance=float(np.sum(rates / exposures) / len(weights)**2))
        bound = float(0.5 * np.sum(np.abs(weights - equal)) * np.ptp(rates))
        assert abs(charge_target - equal_target) <= bound + 1e-14
        result["cases"][label] = {"true_rates": rates.tolist(),
            "charge_weighted": charge, "equal_measured_angle": uniform,
            "charge_weighted_bias_if_equal_angle_target_requested": charge_target - equal_target,
            "target_difference_bound_from_total_variation": bound,
            "equal_to_charge_analytic_variance_ratio": float((np.sum(rates / exposures) / len(weights)**2) / (weights @ rates / exposures.sum()))}
    # Equal-angle sample points are not automatically an angular integral.
    angles = np.linspace(0, np.pi, len(weights))
    smooth_field = 1 + np.cos(angles)**2
    trapezoid = np.trapezoid(smooth_field, angles) / np.pi
    np.testing.assert_allclose(trapezoid, 1.5, rtol=1e-14)
    result["measured_angle_mean_vs_continuous_angle_mean"] = {
        "field": "1 + cos(angle)^2", "angle_range_degrees": [0, 180],
        "mean_of_11_endpoint_including_angles": float(smooth_field.mean()),
        "uniform_continuous_integral": 1.5,
        "trapezoid_mean": float(trapezoid),
        "scope": "Illustrates a distinct integration target; continuous sample rotation remains deferred.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/dgs-estimator-alternatives.json")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    result = {
        "seed": SEED, "replicates_per_case": REPLICATES,
        "defaults_changed": False, "original_projects_accessed": False,
        "scope": "Conditional Poisson/Gaussian truths, analytic acceptance and existing scalar angle-exposure receipt; no instrument files or Mantid.",
        "goniometer_scope": "Time-weighted stationary sample-angle convention accepted; continuous sample rotation is deferred.",
        "provenance": {str(path.relative_to(ROOT)): _sha(path) for path in (
            Path(__file__), ROOT / "src/nfit/measurement_statistics.py", ROOT / "src/nfit/measurement_contracts.py",
            ROOT / "src/nfit/event_covariance.py", ROOT / "src/nfit/histogram_statistics.py")},
        "pooling": _pooling(rng), "symmetry": _symmetry(rng),
        "trajectory_energy": _energy(rng), "dummy_angle_averaging": _angle_averaging(rng),
        "acceptance": "All analytic equalities, public-reference checks and bounded fixed-seed simulation checks passed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "acceptance": result["acceptance"]}))


if __name__ == "__main__":
    main()
