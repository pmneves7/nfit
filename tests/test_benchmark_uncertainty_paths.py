"""Native uncertainty-path diagnostics use synthetic NPZ fixtures only."""

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
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_uncertainty_paths.py"
    spec = importlib.util.spec_from_file_location("uncertainty_paths_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def slabs(tmp_path):
    shape = (4, 2, 1, 4)
    norm = np.broadcast_to(np.array([1.0, 2.0, 8.0, 4.0])[:, None, None, None] * np.array([1.0, 3.0])[None, :, None, None], shape).copy()
    counts = norm * np.array([1.0, 2.0, 1.0, 3.0])[:, None, None, None]
    # One partially empty and one fully empty measured map pixel.
    counts[1, 0, 0, 0] = 0
    counts[1, :, 0, 1] = 0
    # Geometrical gaps, distinct from observed zero counts.
    norm[0, :, 0, 2] = 0
    norm[3, :, 0, 3] = 0
    counts[norm == 0] = 0
    current_variance = counts.copy()
    current_variance[(counts == 0) & (norm > 0)] = 4.0
    mantid_norm = norm * 1.01
    edges = {
        "edges0": np.array([0.075, 0.125, 0.175, 0.225, 0.275]),
        "edges1": np.array([-0.03, 0.0, 0.03]),
        "edges2": np.array([0.31, 0.35]),
        "edges3": np.array([3.0, 5.0, 7.0, 9.0, 11.0]),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        nfit = dict(signal=counts / norm, norm=norm, counts=counts,
                    errors=np.sqrt(current_variance) / norm, numerator=counts, **edges)
        mantid = dict(signal=counts / mantid_norm, norm=mantid_norm, counts=counts,
                      errors=np.sqrt(counts) / mantid_norm, numerator=counts, **edges)
    nfit["mask"] = norm == 0
    mantid["mask"] = norm == 0
    data = dict(numerator=counts, variance=counts)
    paths = tuple(tmp_path / name for name in ("nfit.npz", "mantid.npz", "reference.npz"))
    for path, arrays in zip(paths, (nfit, mantid, data), strict=True):
        np.savez(path, **arrays)
    return paths, (nfit, mantid, data)


def test_viewer_hidden_pooling_and_full_energy_cut_match_analytic_sums(benchmark, slabs):
    paths, (nfit, _, data) = slabs
    report = benchmark.diagnose_paths(*paths)
    maps = report["maps"]
    x = np.array([False, True, True, False])
    reference = report["energy_profiles"]["analytic"]
    expected_n = np.sum(nfit["norm"][x], axis=(0, 1, 2))
    expected_c = np.sum(nfit["numerator"][x], axis=(0, 1, 2))
    expected_v = np.sum(data["variance"][x], axis=(0, 1, 2))
    np.testing.assert_allclose(reference["nfit_exposure"], expected_n)
    np.testing.assert_allclose(reference["numerator"], expected_c)
    np.testing.assert_allclose(reference["analytic_reference_error"], np.sqrt(expected_v) / expected_n)
    valid = maps["nfit_exposure"] > 0
    np.testing.assert_allclose(maps["reference_same_exposure_signal"][valid], maps["numerator"][valid] / maps["nfit_exposure"][valid])
    np.testing.assert_allclose(maps["reference_same_exposure_error"][valid], np.sqrt(maps["reference_numerator_variance"][valid]) / maps["nfit_exposure"][valid])
    assert report["viewer_has_normalization_channel"] is False
    assert report["regions"]["selected_roi"]["map_error_ratio_same_exposure"]["zero_reference_nonzero_current_bins"] == 1
    np.testing.assert_allclose(maps["floor_removed_numerator_variance"], maps["reference_numerator_variance"])


def test_native_regular_and_rotated_paths_are_reported_separately(benchmark, slabs):
    paths, _ = slabs
    report = benchmark.diagnose_paths(*paths)
    profiles = report["energy_profiles"]["native"]["current"]
    assert profiles["regular_inverse_variance"]["energy"].size == 4
    assert profiles["rotated_zero_degree"]["energy"].size > 0
    assert profiles["roi_sum"]["pixels"] == 8
    assert profiles["roi_sum"]["error"] > 0
    coarsened = report["display_coarsened"]["current"]
    assert coarsened["signal"].shape == (2, 2)
    assert coarsened["profiles"]["regular_inverse_variance"]["energy"].size == 2


def test_auxiliary_storage_is_explicit_and_preserves_hidden_pooling(benchmark, slabs):
    paths, _ = slabs
    metadata = benchmark.diagnose_paths(*paths)
    auxiliary = benchmark.diagnose_paths(*paths, normalization_storage="auxiliary")
    assert auxiliary["viewer_has_normalization_channel"] is True
    np.testing.assert_equal(auxiliary["maps"]["current_signal"], metadata["maps"]["current_signal"])
    np.testing.assert_equal(auxiliary["maps"]["current_error"], metadata["maps"]["current_error"])


def test_fringe_uses_real_gaps_and_never_crop_exterior(benchmark):
    full = np.ones((7, 7), dtype=bool)
    assert not np.any(benchmark.fringe_mask(full))
    full[3, 3] = False
    fringe = benchmark.fringe_mask(full)
    assert fringe.sum() == 24
    assert not fringe[3, 3]
    assert not fringe[0, 0]
    assert fringe[1, 1]


def test_low_exposure_classifies_each_energy_row_and_excludes_missing(benchmark):
    exposure = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=float)
    result = benchmark.low_exposure_mask(exposure, exposure > 0, [2.0, 4.0])
    assert not result[0].any()
    assert not result[1, 0]
    assert result[1, 1]
    assert result[1].sum() == 1


def test_ratio_summary_reports_small_as_well_as_large_errors(benchmark):
    ratios = benchmark._ratio_summary(np.array([0.5, 1.0, 2.0]), np.ones(3), np.ones(3, dtype=bool))
    assert ratios["min"] == 0.5
    assert ratios["p5"] == pytest.approx(0.55)


def test_coverage_disagreement_retains_events_filtered_from_common_support(benchmark, slabs):
    paths, arrays = slabs
    nfit, mantid, _ = arrays
    # A real event in an nfit unexposed cell, while Mantid exposes that cell.
    index = (0, 0, 0, 2)
    nfit["counts"][index] = 7
    nfit["numerator"][index] = 7
    mantid["norm"][index] = 2
    mantid["counts"][index] = 7
    mantid["numerator"][index] = 7
    mantid["signal"][index] = 3.5
    mantid["errors"][index] = np.sqrt(7) / 2
    mantid["mask"][index] = False
    # A positive-count cell excluded by a user/file mask despite positive N.
    masked_index = (3, 0, 0, 0)
    nfit["mask"][masked_index] = True
    for path, payload in zip(paths, arrays, strict=True):
        np.savez(path, **payload)
    report = benchmark.diagnose_paths(*paths)
    stats = report["fine_cell_coverage_stats"]
    assert stats["nfit"]["geometrically_unexposed_cells_with_positive_events"] == 1
    assert stats["nfit"]["events_in_geometrically_unexposed_cells"] == 7
    assert stats["nfit"]["masked_cells_with_positive_events"] == 2
    assert stats["mantid_covered_nfit_uncovered"]["cells"] == 2
    assert stats["mantid_covered_nfit_uncovered"]["nfit_event_sum"] == 19


def test_fine_count_redistribution_is_visible_when_map_counts_match(benchmark, slabs):
    paths, arrays = slabs
    nfit, _, _ = arrays
    nfit["counts"][2, 0, 0, 0] += 1
    nfit["counts"][2, 1, 0, 0] -= 1
    nfit["numerator"] = nfit["counts"].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        nfit["signal"] = nfit["numerator"] / nfit["norm"]
    np.savez(paths[0], **nfit)
    region = benchmark.diagnose_paths(*paths)["regions"]["selected_roi"]
    assert region["count_max_abs_difference"] == 0
    assert region["fine_cell_count_max_abs_difference"] == 1
    assert region["fine_cell_count_mismatch_cells"] == 2
    assert region["fine_cell_numerator_relative_l2"] > 0


@pytest.mark.parametrize("invalid", ["edge", "shape", "variance", "mask"])
def test_invalid_slab_configuration_is_rejected(benchmark, slabs, invalid):
    paths, arrays = slabs
    if invalid == "edge":
        arrays[1]["edges0"] = arrays[1]["edges0"] + 0.01
    elif invalid == "shape":
        arrays[0]["counts"] = np.zeros(2)
    elif invalid == "variance":
        arrays[2]["variance"] = np.full_like(arrays[2]["variance"], -1)
    else:
        arrays[0]["mask"] = np.zeros(2, dtype=bool)
    for path, payload in zip(paths, arrays, strict=True):
        np.savez(path, **payload)
    with pytest.raises(ValueError):
        benchmark.diagnose_paths(*paths)


def test_cli_outputs_strict_json_with_supplied_data_provenance(slabs, tmp_path):
    paths, _ = slabs
    root = Path(__file__).parents[1]
    output = tmp_path / "diagnostic.json"
    completed = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "benchmark_uncertainty_paths.py"),
         "--nfit-slab", str(paths[0]), "--mantid-slab", str(paths[1]),
         "--mantid-data", str(paths[2]), "--output", str(output)],
        check=True, capture_output=True, text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
    )
    report = json.loads(completed.stdout)
    assert report == json.loads(output.read_text())
    assert "no fresh Mantid" in report["source"]
    assert report["normalization_storage"] == "metadata"
    assert all(len(value["sha256"]) == 64 for value in report["provenance"].values())
