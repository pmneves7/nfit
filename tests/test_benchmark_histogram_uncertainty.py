"""Validate analytic references without locking in the diagnosed floor policy."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_histogram_uncertainty.py"
    spec = importlib.util.spec_from_file_location("histogram_uncertainty_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_poisson_pooling_is_invariant_to_cell_subdivision(benchmark):
    report = benchmark.diagnostic_report(trials=2)
    sparse = report["sparse_cells"]
    fine = sparse["independent_poisson_reference"]
    coarse = sparse["same_observation_in_one_cell"]
    assert fine == coarse
    assert fine["intensity"] == pytest.approx(0.25)
    assert fine["variance"] == pytest.approx(sparse["analytic_reference_variance"])
    # The floor result is reported, not asserted as correct or permanent.
    assert np.isfinite(sparse["modeled_dgs_cell_floor_then_nfit_pooling"]["standard_error"])


def test_uneven_exposure_and_covered_zero_keep_their_exposure(benchmark):
    report = benchmark.diagnostic_report(trials=2)
    for name in ("uneven_exposure", "missing_exposure"):
        case = report[name]
        assert case["nfit_pooled"]["intensity"] == pytest.approx(case["analytic_intensity"])
        assert case["nfit_pooled"]["variance"] == pytest.approx(case["analytic_variance"])
    assert report["uneven_exposure"]["nfit_pooled"]["exposure"] == 20.0
    assert report["missing_exposure"]["nfit_pooled"]["exposure"] == 5.0
    assert report["uneven_exposure"]["unweighted_cell_intensity_mean"] != pytest.approx(1.55)
    masked = benchmark.pool_counts([8.0, 999.0, 0.0], [2.0, 100.0, 3.0], mask=[False, True, False])
    assert masked == report["missing_exposure"]["nfit_pooled"]


def test_independent_background_variance_adds_even_for_negative_signal(benchmark):
    case = benchmark.diagnostic_report(trials=2)["independent_background"]
    assert case["net_intensity"] < 0
    assert case["net_intensity"] == pytest.approx(case["analytic_intensity"])
    assert case["net_variance"] == pytest.approx(case["analytic_variance"])
    assert case["net_standard_error"]**2 == pytest.approx(case["analytic_variance"])


def test_repeated_poisson_sampling_matches_reference_variance_and_coverage(benchmark):
    case = benchmark.sampling_reference()
    expected = case["expected_variance"]
    mean_standard_error = np.sqrt(expected / case["trials"])
    assert abs(case["empirical_mean"] - case["true_rate"]) < 5 * mean_standard_error
    assert case["empirical_variance"] == pytest.approx(expected, rel=0.04)
    assert case["known_sigma_coverage"] == pytest.approx(0.6827, abs=0.02)


def test_cli_emits_and_saves_same_json(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "uncertainty.json"
    result = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "benchmark_histogram_uncertainty.py"),
         "--trials", "10", "--output", str(output)],
        check=True, capture_output=True, text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
    )
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["repeated_sampling"]["trials"] == 10
    assert report["assumptions"]["mantid_measured"] is False


def test_invalid_sampling_size_is_rejected(benchmark):
    with pytest.raises(ValueError, match="at least two"):
        benchmark.sampling_reference(trials=1)


@pytest.fixture
def matched_slabs(tmp_path):
    shape = (2, 1, 1, 2)
    counts = np.array([4.0, 999.0, 0.0, 9.0]).reshape(shape)
    nfit_norm = np.array([2.0, 0.0, 2.0, 2.0]).reshape(shape)
    mantid_norm = np.full(shape, 4.0)
    numerator_variance = np.array([4.0, np.nan, 1.0, 9.0]).reshape(shape)
    edges = {
        "edges0": np.array([0.0, 1.0, 2.0]),
        "edges1": np.array([-0.05, 0.05]),
        "edges2": np.array([0.0, 1.0]),
        "edges3": np.array([6.0, 10.0, 14.0]),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        nfit = dict(signal=counts / nfit_norm, norm=nfit_norm, counts=counts,
                    errors=np.sqrt(numerator_variance) / nfit_norm, numerator=counts, **edges)
    mantid = dict(signal=counts / mantid_norm, norm=mantid_norm, counts=counts,
                  errors=np.sqrt(counts) / mantid_norm, numerator=counts, **edges)
    data = dict(numerator=counts, variance=counts)
    paths = tuple(tmp_path / name for name in ("nfit.npz", "mantid.npz", "mantid_data.npz"))
    for path, arrays in zip(paths, (nfit, mantid, data), strict=True):
        np.savez(path, **arrays)
    return paths, (nfit, mantid, data)


def test_slab_comparison_recovers_variance_before_exposure_division(benchmark, matched_slabs):
    paths, _ = matched_slabs
    result = benchmark.compare_supplied_slabs(*paths)
    metrics = result["all_common_support"]
    assert metrics["common_covered_cells"] == 3
    assert metrics["count_max_abs_difference"] == 0
    assert metrics["numerator_relative_l2"] == 0
    assert metrics["nonzero_numerator_variance_relative_l2"] == 0
    assert metrics["normalization_relative_l2"] == pytest.approx(0.5)
    assert metrics["covered_empty_cells"] == 1
    assert metrics["covered_empty_nfit_numerator_variance_sum"] == 1
    assert metrics["covered_empty_mantid_numerator_variance_sum"] == 0
    # Supplied synthetic empty-cell variance is one; no production policy is asserted.
    assert metrics["aggregate_standard_error_ratio_common_exposure"]["max"] == pytest.approx(np.sqrt(5 / 4))
    assert metrics["aggregate_standard_error_ratio_own_exposure"]["max"] == pytest.approx(2 * np.sqrt(5 / 4))
    assert result["central_roi"]["common_covered_cells"] == 3
    assert all(len(item["sha256"]) == 64 for item in result["provenance"].values())


@pytest.mark.parametrize("invalid", ["shape", "edges", "variance", "missing"])
def test_slab_comparison_rejects_invalid_inputs(benchmark, matched_slabs, invalid):
    paths, arrays = matched_slabs
    nfit, mantid, data = arrays
    if invalid == "shape":
        nfit["counts"] = np.zeros(3)
    elif invalid == "edges":
        mantid["edges1"] = np.array([-0.04, 0.05])
    elif invalid == "variance":
        data["variance"] = np.full_like(data["variance"], -1)
    else:
        del nfit["errors"]
    for path, payload in zip(paths, arrays, strict=True):
        np.savez(path, **payload)
    with pytest.raises(ValueError):
        benchmark.compare_supplied_slabs(*paths)


def test_slab_comparison_rejects_duplicate_aggregation_axes(benchmark, matched_slabs):
    paths, _ = matched_slabs
    with pytest.raises(ValueError, match="distinct"):
        benchmark.compare_supplied_slabs(*paths, axes=(0, 0))


def test_slab_comparison_records_float32_edge_rounding(benchmark, matched_slabs):
    paths, arrays = matched_slabs
    mantid = arrays[1]
    for i in range(4):
        mantid[f"edges{i}"] = mantid[f"edges{i}"].astype(np.float32)
    np.savez(paths[1], **mantid)
    result = benchmark.compare_supplied_slabs(*paths)
    assert result["edge_max_abs_differences"]["edges1"] > 0


def test_cli_slab_inputs_are_required_together(tmp_path):
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "benchmark_histogram_uncertainty.py"),
         "--nfit-slab", str(tmp_path / "not_read.npz")],
        capture_output=True, text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 2
    assert "must be supplied together" in result.stderr
