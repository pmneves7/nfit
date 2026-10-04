"""Independent fractional/discrete Poisson binning study; no instrument files.

Fine cells have independent Poisson counts and known exposure. Their rates
define a piecewise-constant intensity field. Fractional coefficients interpolate
between output centers and clamp at outer centers. This is a candidate kernel,
not a production DGS implementation or a claim of instrument-data parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from nfit.histogram_statistics import poisson_rate_interval
from nfit.measurement_dependencies import (
    independent_source_dependencies,
    project_source_dependencies,
)
from nfit.rebin import rebin_nd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
Z = float(norm.ppf(.975))


def _kernels(x, edges):
    centers = .5 * (edges[:-1] + edges[1:])
    discrete = ((x[None, :] >= edges[:-1, None]) & (x[None, :] < edges[1:, None])).astype(float)
    fractional = np.maximum(0., 1. - np.abs(centers[:, None] - x[None, :]) / (centers[1] - centers[0]))
    # Domain-edge half-hats retain all contributions in the outermost bin.
    fractional[0, x < centers[0]] = 1.
    fractional[-1, x > centers[-1]] = 1.
    np.testing.assert_allclose(discrete.sum(axis=0), 1., rtol=0, atol=0)
    np.testing.assert_allclose(fractional.sum(axis=0), 1., rtol=0, atol=1e-14)
    return {"discrete": discrete, "fractional": fractional}


def _public_checks(x, edges, kernels, exposure):
    positive = exposure > 0
    counts = (np.arange(len(x)) % 7).astype(float)
    checks = {}
    for name, matrix in kernels.items():
        kernel = matrix[:, positive]
        n = kernel @ exposure[positive]
        c = kernel @ counts[positive]
        v = kernel**2 @ counts[positive]
        checks[name] = {}
        for backend in ("numpy", "numba"):
            result = rebin_nd(
                counts[positive] / exposure[positive], x[positive],
                data_errs=np.sqrt(counts[positive]) / exposure[positive],
                data_weights=exposure[positive], bin_edges=[edges],
                fractional=name == "fractional", normalize=True, backend=backend, workers=1)
            np.testing.assert_allclose(result.binned_data, c / n, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(result.binned_data_errs**2, v / n**2, rtol=1e-13, atol=1e-13)
            checks[name][backend] = "Mean and diagonal squared-coefficient variance agree"
        primitives = independent_source_dependencies(counts[positive], "fractional-study")
        outputs, inputs = np.nonzero(kernel)
        dependencies = project_source_dependencies(primitives, inputs, outputs, kernel[outputs, inputs], (len(n),))
        np.testing.assert_allclose(dependencies.variance(), v, rtol=1e-13, atol=1e-13)
        combined = project_source_dependencies(dependencies, np.arange(len(n)), np.zeros(len(n), dtype=int),
                                               np.ones(len(n)), (1,))
        full_variance = float((kernel.sum(axis=0)**2) @ counts[positive])
        np.testing.assert_allclose(combined.variance().item(), full_variance, rtol=1e-13)
        checks[name]["public_source_dependencies"] = "Retained cross-bin covariance reproduces whole-region variance"
    # Upper-edge regression: this was a 50/50 split before the focused fix.
    endpoint = rebin_nd([4.], [2.], data_errs=[1.], lower=.5, upper=1.5, num_bins=2,
                        fractional=True, normalize=False, backend="numpy")
    np.testing.assert_allclose(endpoint.binned_data, [np.nan, 4.], equal_nan=True)
    np.testing.assert_allclose(endpoint.n_samples, [0., 1.])
    checks["uniform_upper_endpoint"] = "Entire value/error/sample goes to final bin, matching explicit edges"
    return checks


def _metrics(values, variances, target):
    good = np.isfinite(target) & np.all(np.isfinite(values), axis=0)
    error = values[:, good] - target[good]
    bias = values[:, good].mean(axis=0) - target[good]
    return {
        "rms_bias": float(np.sqrt(np.mean(bias**2))),
        "mean_squared_error": float(np.mean(error**2)),
        "wald_95_percent_coverage": float(np.mean(np.abs(error) <= Z * np.sqrt(variances[:, good]))),
        "per_bin_bias": bias.tolist(),
        "included_bins": int(np.count_nonzero(good)),
        "missing_bins": np.flatnonzero(~good).tolist(),
    }


def _simulate(rng, rate, exposure, kernels, replicates):
    expected_counts = exposure * rate
    geometric_target = kernels["discrete"] @ rate / kernels["discrete"].sum(axis=1)
    stores = {}
    for name, matrix in kernels.items():
        n = matrix @ exposure
        with np.errstate(divide="ignore", invalid="ignore"):
            target = matrix @ expected_counts / n
            analytic_variance = matrix**2 @ expected_counts / n**2
        stores[name] = {"matrix": matrix, "exposure": n, "target": target,
                        "analytic_variance": analytic_variance,
                        "values": np.full((replicates, len(n)), np.nan),
                        "variances": np.full((replicates, len(n)), np.nan),
                        "region_values": np.empty(replicates), "region_variances": np.empty(replicates),
                        "region_diagonal_variances": np.empty(replicates)}
    discrete_interval_coverage = np.zeros(len(geometric_target))
    for start in range(0, replicates, 512):
        stop = min(start + 512, replicates)
        counts = rng.poisson(expected_counts, size=(stop - start, len(rate)))
        for name, data in stores.items():
            matrix, n = data["matrix"], data["exposure"]
            c, v = counts @ matrix.T, counts @ (matrix**2).T
            with np.errstate(divide="ignore", invalid="ignore"):
                data["values"][start:stop] = c / n
                data["variances"][start:stop] = v / n**2
            if name == "discrete":
                lower, upper = poisson_rate_interval(c, n, confidence=.95)
                discrete_interval_coverage += np.sum((lower <= data["target"]) & (data["target"] <= upper), axis=0)
            coefficients = matrix.sum(axis=0)
            data["region_values"][start:stop] = counts @ coefficients / n.sum()
            data["region_variances"][start:stop] = counts @ (coefficients**2) / n.sum()**2
            data["region_diagonal_variances"][start:stop] = v.sum(axis=1) / n.sum()**2
    results = {"geometric_bin_average": geometric_target.tolist(), "estimators": {}}
    direct_n = stores["discrete"]["exposure"]
    for name, data in stores.items():
        target, values, variances = data["target"], data["values"], data["variances"]
        good = np.isfinite(target)
        empirical = np.var(values[:, good], axis=0, ddof=1)
        expected = data["analytic_variance"][good]
        # Fixed-seed bounded simulation gate, separate from actual CI coverage.
        np.testing.assert_allclose(empirical, expected, rtol=.065, atol=1e-12)
        np.testing.assert_allclose(np.mean(variances[:, good], axis=0), expected, rtol=.065, atol=1e-12)
        assert np.all(np.abs(values[:, good].mean(axis=0) - target[good])
                      <= 7 * np.sqrt(expected / replicates) + 1e-12)
        region_target = float(exposure @ rate / exposure.sum())
        region_error = np.abs(data["region_values"] - region_target)
        fringe = good & (direct_n < 20)
        result = {
            "covered_kernel_target": target.tolist(), "exposure": data["exposure"].tolist(),
            "analytic_kernel_minus_geometric_bias": (target - geometric_target).tolist(),
            "analytic_rms_target_difference": float(np.sqrt(np.mean((target[good] - geometric_target[good])**2))),
            "own_kernel_target": _metrics(values, variances, target),
            "requested_geometric_bin_target": _metrics(values, variances, geometric_target),
            "per_bin_wald_95_percent_coverage_own_target": np.mean(
                np.abs(values - target) <= Z * np.sqrt(variances), axis=0).tolist(),
            "empirical_to_analytic_variance_ratio": (empirical / expected).tolist(),
            "mean_reported_to_analytic_variance_ratio": (np.mean(variances[:, good], axis=0) / expected).tolist(),
            "newly_interpolated_bins_without_direct_exposure": np.flatnonzero((direct_n == 0) & (data["exposure"] > 0)).tolist(),
            "whole_region_correct_variance": float(np.mean(data["region_variances"])),
            "whole_region_diagonal_only_variance": float(np.mean(data["region_diagonal_variances"])),
            "whole_region_wald_coverage_correct_covariance": float(np.mean(region_error <= Z * np.sqrt(data["region_variances"]))),
            "whole_region_wald_coverage_diagonal_only": float(np.mean(region_error <= Z * np.sqrt(data["region_diagonal_variances"]))),
            "whole_region_diagonal_to_correct_variance_ratio": float(np.mean(data["region_diagonal_variances"]) / np.mean(data["region_variances"])),
        }
        if np.any(fringe):
            result["fringe_wald_coverage_own_target"] = float(np.mean(
                np.abs(values[:, fringe] - target[fringe]) <= Z * np.sqrt(variances[:, fringe])))
        if name == "discrete":
            result["per_bin_garwood_95_percent_coverage_own_target"] = (discrete_interval_coverage / replicates).tolist()
        results["estimators"][name] = result
    # Whole-region source sum is unchanged, although per-bin targets differ.
    np.testing.assert_allclose(stores["discrete"]["region_values"], stores["fractional"]["region_values"], rtol=1e-13)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=12000)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/dgs-fractional-binning.json")
    args = parser.parse_args()
    if args.replicates < 10000:
        parser.error("At least 10000 replicates are required for the fixed variance gate")
    edges = np.linspace(-2., 2., 17)
    x = np.linspace(-2., 2., 641)
    x = .5 * (x[:-1] + x[1:])
    dx = x[1] - x[0]
    kernels = _kernels(x, edges)
    fields = {"flat": np.full(len(x), 2.), "slope": 2. + .6 * x,
              "sharp_peak": 1. + 8. * np.exp(-.5 * ((x - .17) / .08)**2),
              "unobserved_peak": 1. + 8. * np.exp(-.5 * ((x + .625) / .045)**2),
              "step": np.where(x < .08, 1., 5.)}
    uniform = np.full(len(x), 1000. * dx)
    uneven = uniform * np.where(np.mod(x + 2., .25) < .18, .1, 1.)
    uneven[np.abs(x) > 1.25] *= .02
    uneven[(x > -.4) & (x < .2)] *= .03
    uneven[(x >= -.75) & (x < -.5)] = 0.
    exposure_cases = {"uniform": uniform, "unequal_fringe_and_hole": uneven}
    rng = np.random.default_rng(SEED)
    result = {
        "seed": SEED, "replicates": args.replicates, "edges": edges.tolist(),
        "scope": "Independent 1D fine-cell Poisson truth; no DGS instrument files or trajectory reconstruction",
        "defaults_changed": False, "original_projects_accessed": False,
        "exposure_assumption": "Known, fixed, nonnegative; corrected/uncertain exposure is outside this study",
        "targets": {"geometric": "Uniform fine-cell mean inside requested edges",
                    "covered_kernel": "sum_i K_bi N_i rate_i / sum_i K_bi N_i",
                    "variance": "sum_i K_bi^2 C_i / (sum_i K_bi N_i)^2",
                    "cross_bin_covariance": "sum_i K_bi K_ci C_i / (N_b N_c)"},
        "provenance": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT / "src/nfit/rebin.py", ROOT / "src/nfit/_rebin_numba.py",
                                    ROOT / "src/nfit/raw_dgs.py", ROOT / "src/nfit/measurement_dependencies.py")},
        "cases": {},
    }
    with threadpool_limits(limits=2):
        result["public_checks"] = _public_checks(x, edges, kernels, uniform)
        for coverage, exposure in exposure_cases.items():
            for field, rate in fields.items():
                result["cases"][f"{coverage}/{field}"] = _simulate(rng, rate, exposure, kernels, args.replicates)
    zero_lower, zero_upper = poisson_rate_interval(0., .5, confidence=.95)
    result["covered_zero_counts"] = {"observed_diagonal_variance": 0., "true_rate": 2.,
                                    "exposure": .5, "garwood_lower": float(zero_lower), "garwood_upper": float(zero_upper),
                                    "interpretation": "Zero observed propagated variance is not a noiseless rate or a confidence interval"}
    result["acceptance"] = "Public kernel/covariance equalities and fixed-seed analytic mean/variance checks passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value

    args.output.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "acceptance": result["acceptance"]}))


if __name__ == "__main__":
    main()
