"""Independent controlled numerical/calibration checks of optional DGS paths.

No Mantid, instrument data or project files are opened. Decimal references
describe exact arithmetic on the supplied binary64 inputs, not a higher-accuracy
instrument model. Monitor pulse means are known by construction; their shape
and noise are controlled assumptions, not measured calibration evidence.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from nfit import mdevent, raw_dgs_monitors
from nfit.dgs_reduction_policy import ENERGY_TO_K, dgs_event_coordinates, dgs_histogram_edges

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004


def _d(value):
    return Decimal.from_float(float(value))


def _solve(matrix, rhs):
    """Independent pivoted elimination over Decimal, not NumPy inversion."""
    rows = [list(row) + [value] for row, value in zip(matrix, rhs, strict=True)]
    size = len(rows)
    for i in range(size):
        pivot = max(range(i, size), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        scale = rows[i][i]
        assert scale != 0
        rows[i] = [x / scale for x in rows[i]]
        for j in range(size):
            if i != j:
                factor = rows[j][i]
                rows[j] = [x - factor * y for x, y in zip(rows[j], rows[i], strict=True)]
    return [row[-1] for row in rows]


def _project_decimal(q, energy, ub, basis, operation):
    momentum = [[_d(2 * np.pi) * _d(x) for x in row] for row in ub]
    hkl = _solve(momentum, [_d(x) for x in q])
    transformed = [sum(_d(x) * y for x, y in zip(row, hkl, strict=True)) for row in operation]
    return _solve([[_d(x) for x in row] for row in basis.T], [*transformed, _d(energy)])


def _indices_decimal(values, edges):
    result = []
    for coordinate in values:
        indices = [bisect.bisect_right([_d(x) for x in edge], value) - 1
                   for value, edge in zip(coordinate, edges, strict=True)]
        result.append(indices if all(0 <= i < len(e) - 1 for i, e in zip(indices, edges, strict=True)) else [-1] * len(edges))
    return np.asarray(result)


def _indices_float(values, edges):
    indices = np.column_stack([np.searchsorted(e, values[:, j], side="right") - 1 for j, e in enumerate(edges)])
    valid = np.all(np.isfinite(values), axis=1)
    for j, e in enumerate(edges):
        valid &= (indices[:, j] >= 0) & (indices[:, j] < len(e) - 1)
    indices[~valid] = -1
    return indices


def _coordinates(rng):
    ub = np.array([[.1013533711408154, .0950501976190635, -.1954505169221155],
                   [.20010161787729, -.13733358962545242, -.13891933146859048],
                   [.015841341021, .244182312581, .03445771238]])
    basis = np.eye(4)
    basis[:3, :3] = [[1, 1, 1], [1, -1, 0], [1, 1, -2]]
    operation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    edges = (np.linspace(-1.015, .995, 68), np.linspace(-2.015, 2.005, 135),
             np.linspace(-1.01, 1.01, 102), np.linspace(-.25, 50.25, 102))
    cloud = rng.uniform([-.8, -1.8, -.8, 1], [.8, 1.8, .8, 49], size=(400, 4))
    # Challenge internal boundaries only; tiny observed-extrema clipping is an
    # accepted compatibility rounding effect and is deliberately not emulated.
    boundary = cloud.copy()
    boundary[:, 0] = edges[0][rng.integers(2, len(edges[0]) - 2, len(boundary))]
    shifts = rng.choice([-1, 1], len(boundary)) * rng.uniform(1e-9, 1e-6, len(boundary))
    boundary[:, 0] += shifts
    result = {}
    for label, requested in (("ordinary_cloud", cloud), ("internal_boundary_challenge", boundary)):
        hkl = (requested[:, :3] @ basis[:3, :3]) @ np.linalg.inv(operation).T
        q = hkl @ (2 * np.pi * ub).T
        oracle = [_project_decimal(point, e, ub, basis, operation) for point, e in zip(q, requested[:, 3], strict=True)]
        expected_indices = _indices_decimal(oracle, edges)
        expected_coordinates = np.asarray(oracle, dtype=float)
        result[label] = {"events": len(requested), "policies": {}}
        for policy in ("mantid", "high_precision"):
            coordinates, actual_edges = dgs_event_coordinates(q, requested[:, 3], ub, basis, operation, edges, policy)
            indices = _indices_float(coordinates, actual_edges)
            physical = coordinates.copy()
            if policy == "mantid":
                for j, edge in enumerate(edges):
                    physical[:, j] = edge[0] + coordinates[:, j] * (edge[-1] - edge[0]) / (len(edge) - 1)
            error = np.abs(physical - expected_coordinates)
            mismatch = int(np.count_nonzero(np.any(indices != expected_indices, axis=1)))
            if policy == "high_precision":
                assert mismatch == 0
                assert error.max() < 2e-13
            result[label]["policies"][policy] = {
                "wrong_requested_grid_membership": mismatch,
                "max_coordinate_absolute_error": error.max(axis=0).tolist(),
                "rms_coordinate_error": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            }
    assert result["internal_boundary_challenge"]["policies"]["mantid"]["wrong_requested_grid_membership"] > 0
    result["definition"] = "Exact binary64 UB/basis/input values, 80-digit Decimal solving; Å^-1 momentum projects to r.l.u. axes and meV. Compatibility policies can define rounded grids rather than the originally requested grid."
    return result


def _trajectory_decimal(edges, inverse, direction, energy, limits):
    """Intersect each voxel independently with k_f and integrate final energy."""
    shape = tuple(len(e) - 1 for e in edges)
    result = np.zeros(shape)
    factor, ei = _d(ENERGY_TO_K), _d(energy)
    ki = (ei * factor).sqrt()
    incoming = [sum(_d(inverse[i, j]) * (ki if j == 2 else 0) for j in range(3)) for i in range(3)]
    outgoing = [sum(_d(inverse[i, j]) * _d(direction[j]) for j in range(3)) for i in range(3)]
    physical_lo = ((ei - _d(limits[1])) * factor).sqrt()
    physical_hi = ((ei - _d(limits[0])) * factor).sqrt()
    for index in np.ndindex(shape):
        lo, hi = physical_lo, physical_hi
        for j in range(3):
            lower, upper = map(_d, edges[j][index[j]:index[j] + 2])
            if outgoing[j] == 0:
                if not lower <= incoming[j] < upper:
                    hi = lo
                    break
            else:
                crossings = ((incoming[j] - lower) / outgoing[j], (incoming[j] - upper) / outgoing[j])
                lo, hi = max(lo, min(crossings)), min(hi, max(crossings))
        e_lo, e_hi = map(_d, edges[3][index[3]:index[3] + 2])
        lo = max(lo, (max(ei - e_hi, Decimal(0)) * factor).sqrt())
        hi = min(hi, (max(ei - e_lo, Decimal(0)) * factor).sqrt())
        if hi > lo:
            result[index] = float((hi**2 - lo**2) / factor)
    return result


def _trajectories():
    inverse = np.eye(3)
    theta = float(np.arcsin(.5))
    direction = np.array([np.sin(theta), 0, np.cos(theta)])
    crossing_edges = (np.linspace(-2.5, -1.5, 11), np.array([-.1, .1]),
                      np.array([.5, 3]), np.linspace(19, 21, 5))
    rounded_crossing = dgs_histogram_edges(crossing_edges, "mantid")[0][3]
    crossing_energy = 20 + (2 * rounded_crossing)**2 / ENERGY_TO_K + 1e-7
    cases = {
        "ordinary_grid": ((np.linspace(-3, -1, 9), np.array([-.1, .1]),
                           np.linspace(.5, 3, 11), np.linspace(0, 40, 17)), 60., [0., 40.]),
        "near_coincident_crossings": (crossing_edges, crossing_energy, [19., 21.]),
        "narrow_fringe_grid": ((np.linspace(-2.2 - 3e-8, -2.2 + 3e-8, 7),
                               np.array([-.1, .1]), np.array([.5, 3]), np.array([0., 40.])), 60., [0., 40.]),
    }
    result = {}
    for label, (requested_edges, energy, limits) in cases.items():
        shape = tuple(len(edge) - 1 for edge in requested_edges)
        result[label] = {"shape": list(shape), "Ei_meV": energy,
            "energy_transfer_window_meV": limits, "policies": {}}
        for policy in ("mantid", "high_precision"):
            try:
                edges = dgs_histogram_edges(requested_edges, policy)
            except ValueError as error:
                assert policy == "mantid" and label == "narrow_fringe_grid"
                result[label]["policies"][policy] = {"unsupported_grid_guard": str(error)}
                continue
            expected = _trajectory_decimal(edges, inverse, direction, energy, limits)
            actual = np.zeros(shape)
            mdevent._accumulate_detector_trajectory(actual, edges, inverse, direction, energy, limits, 1.,
                mantid_precision=policy == "mantid")
            difference = np.abs(actual - expected)
            covered = expected > 0
            fringe = covered & (expected <= np.quantile(expected[covered], .1))
            peak = float(expected.max())
            relative = difference[covered] / expected[covered]
            if policy == "high_precision":
                assert difference.max() / peak < 3e-7
                assert np.array_equal(actual > 0, covered)
            result[label]["policies"][policy] = {
                "exposure_units": "meV times fixed detector/charge weight",
                "oracle_total": float(expected.sum()), "actual_total": float(actual.sum()),
                "max_absolute_error": float(difference.max()),
                "max_error_divided_by_peak": float(difference.max() / peak),
                "max_relative_error_in_covered_bins": float(relative.max()),
                "fringe_max_relative_error": float((difference[fringe] / expected[fringe]).max()),
                "coverage_membership_disagreements": int(np.count_nonzero((actual > 0) != covered)),
            }
            if mdevent._MDEVENT_NUMBA is not None:
                compiled = mdevent._MDEVENT_NUMBA.run_trajectory_normalization(
                    np.array([theta]), np.array([0.]), np.array([1.]), np.array([inverse]),
                    np.array([energy]), np.array([limits]), np.array([1.]), *edges,
                    np.array(shape, dtype=np.int64), policy == "mantid", workers=1,
                ).reshape(shape)
                compiled_difference = np.abs(compiled - expected)
                if policy == "high_precision":
                    assert compiled_difference.max() / peak < 3e-7
                    assert np.array_equal(compiled > 0, covered)
                result[label]["policies"][policy]["compiled_backend"] = {
                    "max_absolute_error": float(compiled_difference.max()),
                    "max_error_divided_by_peak": float(compiled_difference.max() / peak),
                    "fringe_max_relative_error": float((compiled_difference[fringe] / expected[fringe]).max()),
                    "coverage_membership_disagreements": int(np.count_nonzero((compiled > 0) != covered)),
                }
    result["definition"] = "Per-voxel exact analytic path intersections on each policy's own effective edges; no production crossing enumeration or midpoint membership in the oracle."
    return result


def _derivative_reference(errors, backward, forward):
    # Differentiate the three independent observations directly. This avoids
    # both the expanded variance and the reciprocal difference used in prod.
    b, f = _d(backward), _d(forward)
    coefficients = (-1 / (2 * b), (f - b) / (2 * f * b), 1 / (2 * f))
    return sum((_d(error) * coefficient)**2 for error, coefficient in zip(errors, coefficients, strict=True)).sqrt()


def _derivatives():
    cases = {
        "sparse_near_equal_spacing": ([0., 18.587976302691597, 0.], 1.4550783633161066, 1.455078363316107),
        "uniform_independent_observations": ([2., 1000., 3.], 1.5, 1.5),
        "unequal_independent_observations": ([2., 10., 3.], 1.3, 1.7),
        "sparse_exact_equal_spacing": ([0., 1000., 0.], 1.5, 1.5),
    }
    result = {}
    for label, (errors, backward, forward) in cases.items():
        expected = float(_derivative_reference(errors, backward, forward))
        result[label] = {"errors": errors, "backward_spacing_us": backward,
            "forward_spacing_us": forward, "oracle_sigma": expected, "policies": {}}
        for policy in ("mantid", "stable"):
            values = [raw_dgs_monitors._monitor_derivative_uncertainty(*errors, backward, forward,
                trailing=trailing, variance_policy=policy) for trailing in (False, True)]
            result[label]["policies"][policy] = {
                "leading_sigma": float(values[0]) if math.isfinite(values[0]) else None,
                "trailing_sigma": float(values[1]) if math.isfinite(values[1]) else None,
                "absolute_errors": [abs(v - expected) if math.isfinite(v) else None for v in values],
                "relative_errors": [v / expected - 1 if expected > 0 and math.isfinite(v) else None for v in values],
            }
            if policy == "stable":
                assert all(math.isfinite(v) and v >= 0 for v in values)
                if label != "sparse_near_equal_spacing":
                    np.testing.assert_allclose(values, expected, rtol=2e-14, atol=1e-15)
    assert result["sparse_near_equal_spacing"]["policies"]["mantid"]["trailing_sigma"] is None
    return result


def _calibration(rng):
    replicates, count = 120, 2000
    energy, t0 = 60.0, 15.0
    distances = np.array([18., 20.])
    coefficient = raw_dgs_monitors.TOF_US_PER_M_SQRT_MEV
    centres = coefficient * distances / np.sqrt(energy) + t0
    result = {"replicates": replicates, "peak_events_per_monitor": count,
        "true_Ei_meV": energy, "true_T0_us": t0, "distances_m": distances.tolist(), "cases": {}}
    for shape in ("symmetric_gaussian", "unequal_asymmetric_tails"):
        outputs = {policy: [] for policy in ("mantid", "stable", "known_population_sample_mean")}
        failures = {policy: 0 for policy in outputs}
        successful_ids = {policy: [] for policy in outputs}
        for replicate in range(replicates):
            if shape == "symmetric_gaussian":
                values = [rng.normal(centre, 20, count) for centre in centres]
            else:
                # Mean delay zero by construction; pulse shapes differ but their
                # physical mean flight times still satisfy the known calibration.
                values = [centre + rng.normal(0, 9, count) + rng.exponential(tail, count) - tail
                          for centre, tail in zip(centres, [8, 30], strict=True)]
            for policy in outputs:
                if policy == "known_population_sample_mean":
                    peaks = [float(value.mean()) for value in values]
                else:
                    peaks = [raw_dgs_monitors._mantid_getei_v2_peak(value, distance, energy,
                        grid_start=0., variance_policy=policy) for value, distance in zip(values, distances, strict=True)]
                if None in peaks:
                    failures[policy] += 1
                    continue
                elapsed = peaks[1] - peaks[0]
                estimated_energy = (coefficient * (distances[1] - distances[0]) / elapsed)**2
                estimated_t0 = peaks[0] - distances[0] * elapsed / (distances[1] - distances[0])
                outputs[policy].append([estimated_energy, estimated_t0])
                successful_ids[policy].append(replicate)
        case = {}
        for policy, rows in outputs.items():
            rows = np.asarray(rows)
            assert rows.size > 0
            bias = rows.mean(axis=0) - [energy, t0]
            case[policy] = {"successful_replicates": len(rows), "failures": failures[policy],
                "mean_Ei_meV": float(rows[:, 0].mean()), "mean_T0_us": float(rows[:, 1].mean()),
                "bias_Ei_meV": float(bias[0]), "bias_T0_us": float(bias[1]),
                "rmse_Ei_meV": float(np.sqrt(np.mean((rows[:, 0] - energy)**2))),
                "rmse_T0_us": float(np.sqrt(np.mean((rows[:, 1] - t0)**2))),
                "mean_Ei_monte_carlo_standard_error": float(rows[:, 0].std(ddof=1) / np.sqrt(len(rows))),
                "mean_T0_monte_carlo_standard_error": float(rows[:, 1].std(ddof=1) / np.sqrt(len(rows)))}
        reference = dict(zip(successful_ids["mantid"], outputs["mantid"], strict=True))
        stable = dict(zip(successful_ids["stable"], outputs["stable"], strict=True))
        common_ids = sorted(reference.keys() & stable.keys())
        paired_reference = np.asarray([reference[i] for i in common_ids])
        paired_stable = np.asarray([stable[i] for i in common_ids])
        differences = paired_stable - paired_reference
        squared_error_difference = ((paired_stable - [energy, t0])**2 - (paired_reference - [energy, t0])**2)
        case["paired_stable_minus_reference"] = {
            "common_successful_replicates": len(common_ids),
            "bit_identical_calibration_fraction": float(np.mean(np.all(differences == 0, axis=1))),
            "maximum_absolute_Ei_difference_meV": float(np.abs(differences[:, 0]).max()),
            "maximum_absolute_T0_difference_us": float(np.abs(differences[:, 1]).max()),
            "mean_squared_Ei_error_difference": float(squared_error_difference[:, 0].mean()),
            "mean_squared_Ei_error_difference_monte_carlo_standard_error": float(squared_error_difference[:, 0].std(ddof=1) / np.sqrt(len(common_ids))),
            "mean_squared_T0_error_difference": float(squared_error_difference[:, 1].mean()),
            "mean_squared_T0_error_difference_monte_carlo_standard_error": float(squared_error_difference[:, 1].std(ddof=1) / np.sqrt(len(common_ids))),
            "sign_convention": "Negative squared-error difference would favor stable; uncertainties are Monte Carlo standard errors, not calibration errors.",
        }
        comparator = case["known_population_sample_mean"]
        assert abs(comparator["bias_Ei_meV"]) < 6 * comparator["mean_Ei_monte_carlo_standard_error"]
        assert abs(comparator["bias_T0_us"]) < 6 * comparator["mean_T0_monte_carlo_standard_error"]
        if shape == "unequal_asymmetric_tails":
            for policy in ("mantid", "stable"):
                assert case[policy]["bias_Ei_meV"] > 10 * case[policy]["mean_Ei_monte_carlo_standard_error"]
                assert case[policy]["bias_T0_us"] > 10 * case[policy]["mean_T0_monte_carlo_standard_error"]
        result["cases"][shape] = case
    result["qualification"] = "Controlled independent pulses with known mean flight times and no monitor background. Sample-mean comparator knows event membership; not proposed as a general importer. Peak-region uncertainty controls stopping; no Ei/T0 confidence interval is produced."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/dgs-numerical-alternatives.json")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    with localcontext() as context:
        context.prec = 80
        result = {"seed": SEED, "decimal_precision": context.prec,
            "defaults_changed": False, "original_projects_accessed": False,
            "coordinates": _coordinates(rng), "trajectory": _trajectories(),
            "monitor_derivative": _derivatives(), "monitor_calibration": _calibration(rng),
            "acceptance": "All independent numerical-reference checks and controlled calibration diagnostics completed.",
            "provenance": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__), ROOT / "src/nfit/dgs_reduction_policy.py", ROOT / "src/nfit/raw_dgs_monitors.py",
                ROOT / "src/nfit/mdevent.py", ROOT / "src/nfit/_mdevent_numba.py",
                ROOT / "src/nfit/event_bin_indices.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "acceptance": result["acceptance"]}))


if __name__ == "__main__":
    main()
