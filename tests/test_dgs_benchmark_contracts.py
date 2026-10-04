"""Safety and comparison contracts for manual DGS benchmark tooling.

These tests load only engine-independent helpers and the native benchmark's
function definitions. They never import or invoke an external reduction engine.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

BENCHMARKS = Path(__file__).parents[1] / "benchmarks"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"nfit_manual_{name}", BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def settings_helper():
    return _load_script("benchmark_dgs_settings")


@pytest.fixture(scope="module")
def native_helper():
    return _load_script("benchmark_dgs_nfit_workflow")


@pytest.fixture(scope="module")
def comparator():
    return _load_script("benchmark_dgs_compare")


def test_shared_settings_helper_has_only_standard_library_imports():
    tree = ast.parse((BENCHMARKS / "benchmark_dgs_settings.py").read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    imports.update((node.module or "").split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom))
    assert imports <= {"__future__", "copy", "json", "math", "re", "pathlib"}


def test_benchmark_default_and_invalid_source_preserve_loaded_package(native_helper, monkeypatch, tmp_path):
    monkeypatch.delenv("NFIT_DGS_BENCHMARK_SOURCE", raising=False)
    original = importlib.import_module("nfit")
    assert native_helper._select_source_tree() == "installed nfit; no source injection"
    assert sys.modules["nfit"] is original
    monkeypatch.setenv("NFIT_DGS_BENCHMARK_SOURCE", str(tmp_path))
    with pytest.raises(ValueError, match="contain the nfit package"):
        native_helper._select_source_tree()
    assert sys.modules["nfit"] is original


def test_benchmark_explicit_source_replaces_module_family_in_isolated_process(tmp_path):
    package = tmp_path / "nfit"
    package.mkdir()
    (package / "__init__.py").write_text("candidate_marker = 42\n")
    script = f"""
import importlib.util, os, sys, types
spec = importlib.util.spec_from_file_location('manual', {str(BENCHMARKS / 'benchmark_dgs_nfit_workflow.py')!r})
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sys.modules['nfit'] = types.ModuleType('nfit')
sys.modules['nfit.old'] = types.ModuleType('nfit.old')
os.environ['NFIT_DGS_BENCHMARK_SOURCE'] = {str(tmp_path)!r}
assert {str(tmp_path)!r} in helper._select_source_tree()
assert 'nfit.old' not in sys.modules
import nfit
assert nfit.candidate_marker == 42
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_shared_sequoia_membership_and_original_center_grid(settings_helper):
    config = settings_helper.benchmark_settings()
    expected_runs = [*range(392985, 393470), *range(393500, 393632)]
    assert config["run_numbers"] == expected_runs
    assert len(expected_runs) == 617
    assert len(set(config["raw_files"])) == 617
    assert [Path(path).name for path in config["raw_files"]] == [f"SEQ_{run}.nxs.h5" for run in expected_runs]
    edges = config["grid"]["bin_edges"]
    assert [len(axis)-1 for axis in edges] == [67, 134, 101, 101]
    # Independent original-bin check: requested limits describe centers, so
    # the energy zero center must be present rather than an edge at zero.
    for axis, low, step in zip(edges, [-1., -2., -1., 0.], [.03, .03, .02, .5], strict=True):
        assert axis[0] == low - step/2
        np.testing.assert_allclose(np.diff(axis), step, rtol=0., atol=3e-15)
        np.testing.assert_allclose(.5*(np.asarray(axis[:-1])+axis[1:]),
                                   low+np.arange(len(axis)-1)*step, rtol=0., atol=3e-15)
    assert edges[-1][0] == -.25
    assert edges[-1][-1] == 50.25
    assert config["symmetry_operations"]["6"] == "x,y,z;y,z,x;z,x,y;-x,-y,-z;-y,-z,-x;-z,-x,-y"
    assert len(config["symmetry_operations"]["12"].split(";")) == 12


@pytest.mark.parametrize("preset, first, last", [
    ("hyspec-50k34", 505555, 505915),
    ("hyspec-50k70", 506277, 506637),
])
def test_hyspec_full_membership_uses_explicit_shiver_tip_mask(settings_helper, preset, first, last):
    config = settings_helper.benchmark_settings(preset)
    assert config["run_numbers"] == list(range(first, last + 1))
    assert len(config["run_numbers"]) == 361
    assert config["normalization_file"] is None
    assert config["mask_file"] == (
        f"/SNS/HYS/IPTS-36860/shared/nfit/.nfit-diagnostics/6ar/hyspec_tip_mask_{first}.nxs"
    )
    assert config["reduction"]["ei_override"] == 15.
    assert config["reduction"]["t0_override"] is None
    assert config["reduction"]["bad_pulses_threshold"] == 0.
    assert config["reduction"]["time_independent_background"] == ""


@pytest.mark.parametrize("corruption", ["duplicate", "missing_source", "empty", "wrong_frame", "schema"])
def test_shared_configuration_rejects_invalid_membership(settings_helper, tmp_path, corruption):
    config = settings_helper.benchmark_settings()
    if corruption == "duplicate":
        config["run_numbers"][1] = config["run_numbers"][0]
    elif corruption == "missing_source":
        config["raw_files"].pop()
    elif corruption == "empty":
        config.update(run_numbers=[], raw_files=[])
    elif corruption == "wrong_frame":
        config["reduction"]["q_frame"] = "Q_lab"
    else:
        config["schema_version"] = 999
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        settings_helper.load_settings(path)


def test_pilot_selection_preserves_run_source_pairing_and_original(settings_helper):
    config = settings_helper.benchmark_settings()
    before = copy.deepcopy(config)
    selected = [393631, 392985, 393500]
    pilot = settings_helper.select_runs(config, run_numbers=selected)
    assert pilot["run_numbers"] == selected
    assert [Path(path).stem for path in pilot["raw_files"]] == [f"SEQ_{run}.nxs" for run in selected]
    assert settings_helper.select_runs(config, first_runs=24)["run_numbers"] == config["run_numbers"][:24]
    pilot["grid"]["bin_edges"][0][0] = 123.
    assert config == before


@pytest.mark.parametrize("options", [{"run_numbers": [392985, 392985]}, {"run_numbers": [0]},
                                      {"run_numbers": []}, {"first_runs": 0}, {"first_runs": -1},
                                      {"run_numbers": [392985], "first_runs": 1}])
def test_pilot_selection_rejects_ambiguous_or_invalid_inputs(settings_helper, options):
    with pytest.raises(ValueError):
        settings_helper.select_runs(settings_helper.benchmark_settings(), **options)


def test_receipt_describes_huge_arrays_without_reading_or_serializing(native_helper):
    class MetadataOnlyArray(np.ndarray):
        def tolist(self):
            raise AssertionError("Receipt tried to materialize a numerical array")

        def __getitem__(self, key):
            raise AssertionError("Receipt tried to scan a numerical array")

        def __iter__(self):
            raise AssertionError("Receipt tried to iterate a numerical array")

    huge = np.broadcast_to(np.zeros(1), (67, 134, 101, 101)).view(MetadataOnlyArray)
    assert huge.size == 91_584_578
    safe = native_helper._bounded_metadata({"normalization_denominator": huge,
                                           "nested": {"other_array": huge}, "scalar": np.float64(3.)})
    serialized = json.dumps(safe, default=native_helper._plain)
    assert len(serialized) < 1000
    assert safe["normalization_denominator"] == {
        "numerical_payload": "omitted_from_scalar_receipt", "shape": [67, 134, 101, 101],
        "dtype": "float64", "diagnostic_dataset": "normalization",
    }
    assert safe["scalar"] == 3.
    with pytest.raises(TypeError, match="never in JSON"):
        json.dumps({"unhandled_array": huge}, default=native_helper._plain)


def _write_channels(path, *, normalization, counts=None, numerator=None):
    normalization = np.asarray(normalization, dtype=float)
    with h5py.File(path, "w") as handle:
        for key, value in (
            ("normalization", normalization),
            ("counts", np.zeros(normalization.shape) if counts is None else counts),
            ("numerator", normalization.copy() if numerator is None else numerator),
            ("variance", normalization.copy()),
        ):
            handle.create_dataset(key, data=value)


def test_strict_native_comparator_rejects_single_count_mismatch(comparator, tmp_path):
    actual_path, reference_path = tmp_path / "actual.h5", tmp_path / "reference.h5"
    norm = np.ones((3, 2))
    reference_counts = np.full(norm.shape, 1_000_000.)
    actual_counts = reference_counts.copy()
    actual_counts[2, 1] += 1
    _write_channels(actual_path, normalization=norm, counts=actual_counts)
    _write_channels(reference_path, normalization=norm, counts=reference_counts)
    with h5py.File(actual_path) as actual, h5py.File(reference_path) as reference:
        with pytest.raises(AssertionError):
            comparator._whole_cells(actual, reference, require_native_parity=True, block_cells=2)
        report = comparator._whole_cells(actual, reference, require_native_parity=False, block_cells=2)
    assert report["channels"]["counts"]["different_cells"] == 1
    assert report["first_count_differences"] == [{"index": (2, 1), "actual": 1_000_001., "reference": 1_000_000.}]


def test_comparator_reports_exposure_support_and_events_without_exposure(comparator, tmp_path):
    actual_path, reference_path = tmp_path / "actual.h5", tmp_path / "reference.h5"
    actual_norm = np.array([[0., 1., 0.], [2., 0., 3.]])
    reference_norm = np.array([[0., 1., 1.], [2., 0., 3.]])
    counts = np.array([[0., 0., 2.], [0., 1., 0.]])
    _write_channels(actual_path, normalization=actual_norm, counts=counts)
    _write_channels(reference_path, normalization=reference_norm, counts=counts)
    with h5py.File(actual_path) as actual, h5py.File(reference_path) as reference:
        report = comparator._whole_cells(actual, reference, require_native_parity=False, block_cells=3)
        with pytest.raises(AssertionError):
            comparator._whole_cells(actual, reference, require_native_parity=True, block_cells=3)
    assert report["exposure_support"] == {
        "actual_exposed": 3, "reference_exposed": 4, "differing_exposure_support": 1,
        "actual_events_without_exposure": 2, "reference_events_without_exposure": 1,
    }


def test_native_comparator_does_not_hide_nonfinite_support_mismatch(comparator, tmp_path):
    actual_path, reference_path = tmp_path / "actual.h5", tmp_path / "reference.h5"
    norm = np.ones((2, 2))
    numerator = norm.copy()
    numerator[1, 1] = np.nan
    _write_channels(actual_path, normalization=norm, numerator=numerator)
    _write_channels(reference_path, normalization=norm)
    with h5py.File(actual_path) as actual, h5py.File(reference_path) as reference:
        report = comparator._whole_cells(actual, reference, require_native_parity=False)
        with pytest.raises(AssertionError):
            comparator._whole_cells(actual, reference, require_native_parity=True)
    assert report["channels"]["numerator"]["finite_support_equal"] is False


def test_nio_cut_support_uses_exposure_and_keeps_covered_zeros(comparator, tmp_path):
    path = tmp_path / "covered_zero_cut.h5"
    # Rows below are energies; columns are HHH. Hidden-axis cells each hold
    # half the exposure, so the cut has this independently specified exposure.
    exposure = np.array([[0., 1., 2., 3.], [1., 0., 2., 3.], [1., 2., 0., 3.]])
    norm = np.broadcast_to(exposure.T[:, None, None, :]/2, (4, 2, 1, 3)).copy()
    _write_channels(path, normalization=norm)  # Zero observed counts at every cell.
    with h5py.File(path, "a") as handle:
        for i, edges in enumerate(([.125, .175, .225, .275, .325], [-.05, 0., .05], [.31, .35], [0., 10., 20., 30.])):
            handle.create_dataset(f"edges/{i}", data=edges)
    with h5py.File(path) as handle:
        report = comparator._nio_cut(handle, handle)
    assert report["hidden_cell_indices"] == [[0, 2], [0, 1]]
    assert json.loads(json.dumps(report))["hidden_cell_indices"] == [[0, 2], [0, 1]]
    regions = report["regions"]
    assert regions["all"]["normalization"]["cells"] == 12
    # The upper-right cell is three face-neighbor steps from an uncovered
    # cell; the two-step fringe excludes that cell and includes the other eight.
    assert regions["coverage_fringe"]["normalization"]["cells"] == 8
    assert regions["row_lowest_exposure_decile"]["normalization"]["cells"] == 3
    assert regions["central_low_coverage_roi"]["normalization"]["cells"] == 4
    assert regions["coverage_fringe"]["counts"]["actual_sum"] == 0.
    assert regions["coverage_fringe"]["normalization"]["actual_sum"] == pytest.approx(exposure.sum()-exposure[0, 3])
