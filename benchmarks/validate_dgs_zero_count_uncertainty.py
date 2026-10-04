"""Controlled zero-count uncertainty checks; no instrument files or Mantid.

Run with the nfit interpreter. Independent Poisson-CDF inversions and exact
probability sums check interval coverage; fixed-seed repeated sampling checks
weighted-event and shared-background variance. Calibration/background interval
examples are statistical references, not implemented DGS likelihoods or defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, norm, poisson

from nfit.histogram_statistics import (
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
    normalized_event_statistics,
    poisson_rate_interval,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData, mdhisto_measured_bins
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_diagnostics import (
    available_confidence_channels,
    poisson_interval_channels,
)
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.measurement_likelihoods import (
    PoissonCountModel,
    fit_measurement_residuals,
    poisson_count_statistics,
    poisson_deviance_residuals,
)
from nfit.measurement_statistics import SourceTerm, estimate_measurement_bin

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
REPLICATES = 120_000
CONFIDENCE = 0.95


def _reference_interval(count, confidence=CONFIDENCE):
    """Invert Poisson tail probabilities, independently of chi-square formulas."""
    tail = (1 - confidence) / 2
    high_bracket = max(10.0, 4.0 * (count + 1))
    while poisson.cdf(count, high_bracket) > tail:
        high_bracket *= 2
    upper = brentq(lambda mean: poisson.cdf(count, mean) - tail, 0.0, high_bracket)
    lower = (
        0.0
        if count == 0
        else brentq(lambda mean: poisson.sf(count - 1, mean) - tail, 0.0, high_bracket)
    )
    return lower, upper


def _coverage(bounds, target, probabilities=None):
    inside = (bounds[0] <= target) & (target <= bounds[1])
    return float(np.mean(inside) if probabilities is None else np.sum(inside * probabilities))


def _known_exposure(rng):
    result = {}
    for confidence in (0.682689492137, CONFIDENCE):
        reference = np.array([_reference_interval(n, confidence) for n in range(65)]).T
        actual = poisson_rate_interval(np.arange(65), 1.0, confidence=confidence)
        np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-12)
        z = norm.ppf((1 + confidence) / 2)
        cases = []
        for mean in (0.0, 0.05, 0.2, 0.8, 2.0, 10.0):
            counts = rng.poisson(mean, size=REPLICATES)
            support = np.arange(int(poisson.ppf(1 - 1e-13, mean)) + 1)
            exact = _coverage(
                poisson_rate_interval(support, 1.0, confidence=confidence),
                mean,
                poisson.pmf(support, mean),
            )
            simulated = _coverage(poisson_rate_interval(counts, 1.0, confidence=confidence), mean)
            assert exact >= confidence - 2e-12
            assert abs(simulated - exact) < 0.006
            wald = np.abs(counts - mean) <= z * np.sqrt(counts)
            cases.append(
                {
                    "true_mean_count": mean,
                    "zero_probability": float(np.exp(-mean)),
                    "exact_interval_coverage": exact,
                    "simulated_interval_coverage": simulated,
                    "observed_variance_gaussian_coverage": float(wald.mean()),
                }
            )
        result[str(confidence)] = {
            "central_zero_upper_per_unit_exposure": float(-np.log((1 - confidence) / 2)),
            "one_sided_zero_upper_per_unit_exposure": float(-np.log(1 - confidence)),
            "cases": cases,
        }
    return result


def _public_contract_checks():
    c, n = np.array([0.0, 4.0, 0.0, 0.0]), np.array([1.0, 9.0, 0.0, 2.0])
    signal, variance = normalized_event_statistics(c, c, n)
    metadata = {
        "event_statistics": dict(EVENT_STATISTICS_METADATA),
        "zero_event_bins_are_measured": True,
        "normalization_denominator": n,
        "measurement_contract": MeasurementContract(
            kind="counting",
            estimator="exposure_pool",
            quantity="rate",
            value_units="1/s",
            exposure_units="s",
        ).to_dict(),
        "poisson_count_model": PoissonCountModel(
            constant_weight=1.0, provenance="controlled independent unit-weight source counts"
        ).to_dict(),
    }
    data = MDHistoData(
        (MDHistoAxis("DeltaE", np.arange(5.0), "meV", "energy"),),
        signal,
        np.sqrt(variance),
        np.array([False, False, False, True]),
        c,
        metadata=metadata,
        auxiliary_channels=event_statistics_channels(c, c, n),
    )
    np.testing.assert_array_equal(mdhisto_measured_bins(data), [True, True, False, False])
    points = prepare_histogram_fit_points(data)
    np.testing.assert_array_equal(points.mask, [True, True, False, False])
    selected = points.subset(points.mask)
    residual = fit_measurement_residuals(selected, np.array([2.0, 2.0]), "poisson_deviance")
    assert residual[0] == -2.0
    np.testing.assert_allclose(poisson_deviance_residuals([0.0], [2.0]), [-2.0])
    try:
        fit_measurement_residuals(selected, np.array([2.0, 2.0]), "gaussian")
    except ValueError:
        pass
    else:
        raise AssertionError("zero sigma must not become infinite Gaussian precision")
    view = {
        **metadata,
        "signal": signal,
        "errors": np.sqrt(variance),
        "mask": data.mask,
        "event_signal_numerator": c,
        "event_variance_numerator": c,
        "normalization_denominator": n,
    }
    bounds = poisson_interval_channels(view, confidence=CONFIDENCE)
    np.testing.assert_allclose(bounds["confidence_upper"][0], -np.log(0.025))
    assert np.isnan(bounds["confidence_upper"][[2, 3]]).all()
    assert available_confidence_channels(data)
    rejected = []
    for label, changes in (
        ("undeclared_weights", {"poisson_count_model": None}),
        (
            "symmetry_copies",
            {"symmetry_operations_hkl": [np.eye(3).tolist(), (-np.eye(3)).tolist()]},
        ),
        ("background_difference", {"background_subtractions": ["shared dummy"]}),
        (
            "uncertain_exposure",
            {"event_statistics": {**EVENT_STATISTICS_METADATA, "exposure": "uncertain"}},
        ),
    ):
        try:
            poisson_interval_channels({**view, **changes})
        except (ValueError, TypeError):
            rejected.append(label)
        else:
            raise AssertionError(f"invalid Poisson interval accepted: {label}")
    # Integer corrected C does not certify Poisson weights: C=4, V=8 violates w=1.
    payload = dict(selected.measurement_payload)
    payload["event_variance_numerator"] = np.array([0.0, 8.0])
    try:
        poisson_count_statistics(selected.with_updates(measurement_payload=payload))
    except ValueError:
        rejected.append("heterogeneous_weight_variance")
    else:
        raise AssertionError("heterogeneous event variance accepted as integer counts")
    contract = MeasurementContract(
        kind="counting",
        estimator="exposure_pool",
        quantity="rate",
        value_units="1/s",
        exposure_units="s",
    )
    pooled = estimate_measurement_bin(contract, [0.0, 4.0], [0.0, 4.0], exposure=[1.0, 9.0])
    np.testing.assert_allclose([pooled.value, pooled.variance, pooled.exposure], [0.4, 0.04, 10.0])
    # Known zero and absent zero have different support and interval behavior.
    return {
        "covered_zero_measured": True,
        "absent_and_masked_excluded": True,
        "zero_deviance_for_predicted_mean_2": float(residual[0]),
        "covered_zero_gaussian_positive_sigma_guard": True,
        "rejected_interval_models": rejected,
        "pooled_with_zero_rate": pooled.value,
        "pooled_with_zero_variance": pooled.variance,
    }


def _weighted_events(rng):
    weights = np.array([1.0, 4.0])
    probabilities = np.array([0.9, 0.1])
    mean_weight, mean_square = float(probabilities @ weights), float(probabilities @ weights**2)
    cases = []
    for exposure in (0.1, 1.0, 10.0):
        event_rate = 2.0
        counts = rng.poisson(exposure * event_rate * probabilities, size=(REPLICATES, 2))
        c, v = counts @ weights, counts @ weights**2
        true_rate, true_variance = event_rate * mean_weight, event_rate * mean_square / exposure
        observed_rate = c / exposure
        observed_variance = v / exposure**2
        assert abs(observed_rate.mean() - true_rate) < 6 * observed_rate.std(ddof=1) / np.sqrt(
            REPLICATES
        )
        assert abs(observed_rate.var(ddof=1) / true_variance - 1) < 0.045
        assert abs(observed_variance.mean() / true_variance - 1) < 0.045
        # If the independent mark law is externally known, total integer source
        # counts estimate its event rate. This differs from corrected C/N.
        marked_model_bounds = poisson_rate_interval(
            counts.sum(axis=1), exposure, confidence=CONFIDENCE, constant_weight=mean_weight
        )
        coverage = _coverage(marked_model_bounds, true_rate)
        support = np.arange(int(poisson.ppf(1 - 1e-13, exposure * event_rate)) + 1)
        exact = _coverage(
            poisson_rate_interval(
                support, exposure, confidence=CONFIDENCE, constant_weight=mean_weight
            ),
            true_rate,
            poisson.pmf(support, exposure * event_rate),
        )
        assert abs(coverage - exact) < 0.006
        zero = counts.sum(axis=1) == 0
        assert np.all(observed_variance[zero] == 0)
        cases.append(
            {
                "exposure": exposure,
                "true_corrected_rate": true_rate,
                "analytic_corrected_rate_variance": true_variance,
                "empirical_corrected_rate_variance": float(observed_rate.var(ddof=1)),
                "mean_observed_rate_variance": float(observed_variance.mean()),
                "zero_observed_variance_frequency": float(zero.mean()),
                "gaussian_95_percent_coverage": float(
                    np.mean(
                        np.abs(observed_rate - true_rate)
                        <= norm.ppf(0.975) * np.sqrt(observed_variance)
                    )
                ),
                "known_mark_law_count_interval_coverage": coverage,
                "known_mark_law_zero_upper": float(-np.log(0.025) * mean_weight / exposure),
                "rms_weight_zero_upper_if_imposed": float(
                    -np.log(0.025) * np.sqrt(mean_square) / exposure
                ),
            }
        )
    return {
        "weights": weights.tolist(),
        "known_mark_probabilities": probabilities.tolist(),
        "scope": "The known mark-law estimator uses source integer counts and mean weight, not the realized corrected C/N estimator; no universal weighted interval is inferred.",
        "identical_observed_zero_C_V_N_different_known_constant_weights": {
            "C": 0.0,
            "V": 0.0,
            "N": 1.0,
            "weight_1_upper_95": float(-np.log(0.025)),
            "weight_4_upper_95": float(-4 * np.log(0.025)),
            "conclusion": "C=V=0 contains no event-weight information; a source response/mark model is required.",
        },
        "cases": cases,
    }


def _conditional_monitor_interval(counts, monitor, response, confidence=CONFIDENCE):
    """Exact conditional binomial interval for two independent Poisson counts.

    K~Pois(q*I), M~Pois(q*response). Conditional on K+M, q cancels and
    p=I/(I+response). Clopper-Pearson bounds are inverted to rate I.
    """
    counts, monitor = np.broadcast_arrays(counts, monitor)
    tail = (1 - confidence) / 2
    low = np.zeros(counts.shape)
    high = np.ones(counts.shape)
    positive_counts, positive_monitor = counts > 0, monitor > 0
    low[positive_counts] = beta.ppf(tail, counts[positive_counts], monitor[positive_counts] + 1)
    high[positive_monitor] = beta.ppf(
        1 - tail, counts[positive_monitor] + 1, monitor[positive_monitor]
    )
    with np.errstate(divide="ignore"):
        return response * low / (1 - low), response * high / (1 - high)


def _calibration_uncertainty(rng):
    cases = []
    rate, response = 2.0, 5.0
    for true_exposure in (0.1, 0.5, 2.0):
        k = rng.poisson(true_exposure * rate, REPLICATES)
        m = rng.poisson(true_exposure * response, REPLICATES)
        exact_bounds = _conditional_monitor_interval(k, m, response)
        ks = np.arange(int(poisson.ppf(1 - 1e-13, true_exposure * rate)) + 1)[:, None]
        ms = np.arange(int(poisson.ppf(1 - 1e-13, true_exposure * response)) + 1)[None, :]
        probabilities = poisson.pmf(ks, true_exposure * rate) * poisson.pmf(
            ms, true_exposure * response
        )
        exact = _coverage(_conditional_monitor_interval(ks, ms, response), rate, probabilities)
        simulated = _coverage(exact_bounds, rate)
        assert exact >= CONFIDENCE - 2e-12
        assert abs(simulated - exact) < 0.006
        plug_in = poisson_rate_interval(k, m / response, confidence=CONFIDENCE)
        resolved = m > 0
        cases.append(
            {
                "true_exposure": true_exposure,
                "true_rate": rate,
                "mean_monitor_counts": true_exposure * response,
                "exact_conditional_interval_coverage": exact,
                "simulated_conditional_interval_coverage": simulated,
                "unresolved_plugin_exposure_frequency": float(np.mean(~resolved)),
                "plugin_interval_coverage_among_resolved_exposures": _coverage(
                    (plug_in[0][resolved], plug_in[1][resolved]), rate
                ),
                "conditional_zero_upper_when_sample_0_monitor_1": float(
                    _conditional_monitor_interval(np.array([0]), np.array([1]), response)[1][0]
                ),
                "plugin_zero_upper_when_sample_0_monitor_1": float(-np.log(0.025) * response),
            }
        )
    uncertain = MeasurementContract(
        kind="counting",
        estimator="exposure_pool",
        quantity="rate",
        value_units="1/s",
        exposure_units="s",
        normalizer="uncertain",
    )
    zero = estimate_measurement_bin(
        uncertain, [0.0], [0.0], exposure=[0.2], exposure_variance=[0.04]
    )
    assert zero.variance == 0.0
    return {
        "model": "independent sample K~Poisson(q*I) and monitor M~Poisson(q*5); monitor response 5 known",
        "delta_method_observed_variance_at_zero": zero.variance,
        "scope": "Conditional nuisance-exposure reference only; present DGS caches do not retain this complete sample/monitor/calibration likelihood.",
        "cases": cases,
    }


def _background_intervals(k, l, sample_exposure, background_exposure):
    # Each component interval has 97.5% coverage. Union bound guarantees
    # simultaneous >=95%, hence an interval for A-B; independence is unnecessary.
    component_confidence = 1 - (1 - CONFIDENCE) / 2
    sample = poisson_rate_interval(k, sample_exposure, confidence=component_confidence)
    background = poisson_rate_interval(l, background_exposure, confidence=component_confidence)
    return sample[0] - background[1], sample[1] - background[0]


def _backgrounds(rng):
    rate, background = 1.0, 2.0
    cases = []
    for sample_exposure, background_exposure in ((0.2, 0.05), (1.0, 4.0)):
        k = rng.poisson(sample_exposure * (rate + background), REPLICATES)
        l = rng.poisson(background_exposure * background, REPLICATES)
        value = k / sample_exposure - l / background_exposure
        variance = k / sample_exposure**2 + l / background_exposure**2
        ks = np.arange(int(poisson.ppf(1 - 1e-13, sample_exposure * (rate + background))) + 1)[
            :, None
        ]
        ls = np.arange(int(poisson.ppf(1 - 1e-13, background_exposure * background)) + 1)[None, :]
        probabilities = poisson.pmf(ks, sample_exposure * (rate + background)) * poisson.pmf(
            ls, background_exposure * background
        )
        exact = _coverage(
            _background_intervals(ks, ls, sample_exposure, background_exposure), rate, probabilities
        )
        simulated = _coverage(
            _background_intervals(k, l, sample_exposure, background_exposure), rate
        )
        assert exact >= CONFIDENCE - 2e-12
        assert abs(simulated - exact) < 0.006
        zero_bounds = _background_intervals(
            np.array([0]), np.array([0]), sample_exposure, background_exposure
        )
        cases.append(
            {
                "sample_exposure": sample_exposure,
                "background_exposure": background_exposure,
                "both_zero_probability": float(
                    np.exp(
                        -sample_exposure * (rate + background) - background_exposure * background
                    )
                ),
                "exact_simultaneous_difference_coverage": exact,
                "simulated_simultaneous_difference_coverage": simulated,
                "gaussian_95_percent_coverage": float(
                    np.mean(np.abs(value - rate) <= norm.ppf(0.975) * np.sqrt(variance))
                ),
                "both_zero_signed_difference_interval_95": [
                    float(zero_bounds[0][0]),
                    float(zero_bounds[1][0]),
                ],
            }
        )
    copies, ns, nb = 6, 0.2, 0.1
    k = rng.poisson(ns * (rate + background), size=(REPLICATES, copies))
    l = rng.poisson(nb * background, REPLICATES)
    value = k.mean(axis=1) / ns - l / nb
    variance = k.sum(axis=1) / (copies * ns) ** 2 + l / nb**2
    false_independent = k.sum(axis=1) / (copies * ns) ** 2 + l / (copies * nb**2)
    analytic_variance = (rate + background) / (copies * ns) + background / nb
    assert abs(value.var(ddof=1) / analytic_variance - 1) < 0.04
    assert abs(variance.mean() / analytic_variance - 1) < 0.04
    # Public primitive sensitivities combine reused source coefficients before
    # squaring; they are conditional observed variances, not count likelihoods.
    contract = MeasurementContract(
        kind="linear_reconstruction",
        estimator="linear_sum",
        quantity="sample minus shared background",
        value_units="1/s",
        dependence="shared_sources",
    )
    terms = [
        [
            SourceTerm(f"sample{i}", 1 / ns, ns * (rate + background)),
            SourceTerm("background", -1 / nb, nb * background),
        ]
        for i in range(copies)
    ]
    reference = estimate_measurement_bin(
        contract,
        np.full(copies, rate),
        np.full(copies, (rate + background) / ns + background / nb),
        coefficients=np.full(copies, 1 / copies),
        source_terms=terms,
    )
    np.testing.assert_allclose(reference.variance, analytic_variance)
    return {
        "independent_sample_background_cases": cases,
        "shared_background": {
            "sample_angles": copies,
            "analytic_final_variance": analytic_variance,
            "empirical_final_variance": float(value.var(ddof=1)),
            "mean_reported_correct_variance": float(variance.mean()),
            "mean_false_independent_variance": float(false_independent.mean()),
            "all_sources_zero_observed_variance": 0.0,
            "joint_all_zero_probability": float(
                np.exp(-copies * ns * (rate + background) - nb * background)
            ),
            "scope": "Six sample observations and one background observation; background replay creates neither six independent counts nor additional physical counting exposure.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/results/dgs-zero-count-uncertainty.json"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    paths = (
        Path(__file__),
        ROOT / "src/nfit/histogram_statistics.py",
        ROOT / "src/nfit/mdhisto.py",
        ROOT / "src/nfit/measurement_diagnostics.py",
        ROOT / "src/nfit/measurement_likelihoods.py",
        ROOT / "src/nfit/measurement_fit_data.py",
        ROOT / "src/nfit/measurement_statistics.py",
    )
    result = {
        "seed": SEED,
        "replicates_per_case": REPLICATES,
        "defaults_changed": False,
        "original_projects_accessed": False,
        "scope": "Controlled exact Poisson/compound-event/monitor/background models; no instrument files or Mantid.",
        "provenance": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "public_contract_checks": _public_contract_checks(),
        "known_exposure": _known_exposure(rng),
        "weighted_events": _weighted_events(rng),
        "calibration_uncertainty": _calibration_uncertainty(rng),
        "backgrounds": _backgrounds(rng),
        "acceptance": "Independent tail inversions, exact probability sums, model rejection/support/likelihood guards and bounded simulation checks passed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "acceptance": result["acceptance"]}))


if __name__ == "__main__":
    main()
