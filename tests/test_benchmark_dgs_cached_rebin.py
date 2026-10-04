"""Safety and saved-cache-only execution of the manual native rebin harness."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks/benchmark_dgs_cached_rebin.py"


@pytest.fixture(scope="module")
def helper():
    spec = importlib.util.spec_from_file_location("cached_rebin_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings(root):
    return {"schema_version": 1, "preset": "synthetic", "instrument": "SEQUOIA",
            "run_numbers": [42], "raw_files": [str(root / "nexus/SEQ_42.nxs.h5")],
            "normalization_file": None, "mask_file": None, "ub_source": str(root / "shared/ub.nxs"),
            "output_root": str(root / "shared/nfit/benchmarks"), "threads": 1, "ram_limit_mib": 2048,
            "primary_copies": 6, "grid": {"vectors": [[1, 1, 1], [1, -1, 0], [1, 1, -2]],
                "bin_edges": [[-10., 0., 10.]] * 3 + [[-10., 0., 15.]]},
            "symmetry_operations": {"1": "x,y,z", "6": "x,y,z;y,z,x;z,x,y;-x,-y,-z;-y,-z,-x;-z,-x,-y",
                "12": "x,y,z;y,z,x;z,x,y;y,x,z;x,z,y;z,y,x;-x,-y,-z;-y,-z,-x;-z,-x,-y;-y,-x,-z;-x,-z,-y;-z,-y,-x"},
            "reduction": {"ei_override": None, "t0_override": None, "emin_fraction": -.95,
                "emax_fraction": .95, "bad_pulses_threshold": 0., "q_frame": "Q_sample"}}


def test_cached_harness_has_no_external_engine_dependency():
    tree = ast.parse(SCRIPT.read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports.update((node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not imports.intersection({"mantid", "shiver"})


def test_paths_reject_non_ipts_or_other_experiment(helper, tmp_path):
    root = tmp_path / "IPTS-42"
    source = root / "shared/nfit/input.nfit"
    source.parent.mkdir(parents=True)
    source.touch()
    settings = _settings(root)
    assert helper._validate_paths(settings, source) == source.resolve()
    settings["raw_files"] = [str(tmp_path / "IPTS-43/nexus/SEQ_42.nxs.h5")]
    with pytest.raises(ValueError, match="outside the configured IPTS"):
        helper._validate_paths(settings, source)
    with pytest.raises(ValueError, match="inside one IPTS"):
        helper._ipts_root(tmp_path / "home/data.nfit")


def test_matching_group_requires_unique_ordered_actual_membership(helper, tmp_path):
    settings = _settings(tmp_path / "IPTS-42")
    dataset = SimpleNamespace(metadata={"source_file": settings["raw_files"][0], "run_number": "42"})
    group = SimpleNamespace(metadata={"raw_dgs": {}}, datasets=[dataset], subgroups=[])
    project = SimpleNamespace(data_groups=[group])
    assert helper._matching_group(project, settings) == (group, group)
    dataset.metadata["run_number"] = "43"
    with pytest.raises(ValueError, match="run membership"):
        helper._matching_group(project, settings)
    dataset.metadata["run_number"] = "42"
    project.data_groups.append(group)
    with pytest.raises(ValueError, match="exactly one"):
        helper._matching_group(project, settings)


def test_saved_cache_precondition_rejects_miss_before_reduction(helper, tmp_path):
    from nfit import raw_dgs_dataset_group
    from tests.test_raw_dgs import _write_raw_dgs
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    with pytest.raises(AssertionError, match="compatible saved reduced cache"):
        helper._require_saved_caches(group)


def test_explicit_calculation_hook_is_scoped_and_restored_on_failure(helper, monkeypatch):
    from nfit import project_composites

    def lookup(group, signature, binning_id=None):
        return "existing histogram"

    monkeypatch.setattr(project_composites, "_matching_composite_base_key", lookup)
    with pytest.raises(RuntimeError, match="cancelled"):
        with helper._calculate_new_binning("new"):
            assert project_composites._matching_composite_base_key(None, "signature", binning_id="new") is None
            assert project_composites._matching_composite_base_key(None, "signature", "old") == "existing histogram"
            raise RuntimeError("cancelled")
    assert project_composites._matching_composite_base_key is lookup


def test_manifest_receipt_detects_replacement_and_grid_guard_checks_edges(helper, tmp_path):
    path = tmp_path / "input.nfit"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", '{"version":1}')
    first = helper._input_receipt(path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", '{"version":2}')
    assert helper._input_receipt(path)["manifest_sha256"] != first["manifest_sha256"]
    settings = _settings(tmp_path / "IPTS-42")
    histogram = SimpleNamespace(shape=(2,)*4, axes=[SimpleNamespace(values=np.asarray(edge)) for edge in settings["grid"]["bin_edges"]])
    helper._validate_histogram_grid(histogram, settings)
    histogram.axes[0].values[0] = np.nextafter(-10., -np.inf)
    with pytest.raises(AssertionError, match="exact shared grid"):
        helper._validate_histogram_grid(histogram, settings)


def test_real_cached_project_rebin_save_and_input_preservation(helper, tmp_path):
    from nfit import (
        DataGroup,
        NfitProject,
        add_data_group_composite_binning,
        raw_dgs_dataset_group,
        refresh_composite_dataset,
        save_project,
        set_reduction_settings,
    )
    from nfit.data_workspace import project_data_workspace
    from tests.test_raw_dgs import _write_raw_dgs

    root_path = tmp_path / "IPTS-42"
    settings = _settings(root_path)
    source = Path(settings["raw_files"][0])
    source.parent.mkdir(parents=True)
    _write_raw_dgs(source)
    ub_path = Path(settings["ub_source"])
    ub_path.parent.mkdir(parents=True)
    with h5py.File(ub_path, "w") as archive:
        archive.create_dataset("MDEventWorkspace/experiment0/sample/oriented_lattice/orientation_matrix", data=np.eye(3))
    spec = importlib.util.spec_from_file_location("native_cached_rebin_test", SCRIPT.parent / "benchmark_dgs_nfit_workflow.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    group = raw_dgs_dataset_group([source])
    set_reduction_settings(group, {"ub_matrix": np.eye(3).tolist(), "bad_pulse_threshold": 0., "cache_reduced_events": True})
    root = DataGroup(name="baseline", subgroups=[group])
    project = NfitProject(data_groups=[root], settings={"cache_binnings": True})
    native._configure(group, settings, 6)
    project_path = root_path / "shared/nfit/baseline.nfit"
    project_path.parent.mkdir(parents=True)
    with project_data_workspace(project, project_path):
        refresh_composite_dataset(root, node=group)
        existing_id = add_data_group_composite_binning(group, name="existing 12 copies")
        native._configure(group, settings, 12, existing_id)
        refresh_composite_dataset(root, node=group, binning_id=existing_id)
        save_project(project, project_path)
    before = helper._input_receipt(project_path)
    config = root_path / "shared/nfit/config.json"
    config.write_text(json.dumps(settings))
    env = {key: value for key, value in os.environ.items() if not key.startswith("NFIT_DGS_BENCHMARK_")}
    env["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--project", str(project_path),
                    "--tag", "cache-only-smoke"], env=env, capture_output=True, text=True, check=True)
    output = Path(settings["output_root"]) / "nfit/cache-only-smoke"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["raw_reduction_performed"] is False
    assert receipt["saved_event_rebin_cache_usage"] == {"hits": 1, "misses": 0}
    assert helper._input_receipt(project_path) == before
    assert (output / "DGS_cached_rebin.nfit").is_file()
    assert (output / "histogram-12.h5").is_file()
    assert receipt["cached_rebin_workflow_stage_labels"] == ["native_project_reopen_metadata",
        "saved_histogram_access_6", "additional_binning_setup", "saved_event_cache_bin_12", "native_project_save_12"]
    intervals = next(stage for stage in receipt["stages"] if stage["label"] == "saved_event_cache_bin_12")["progress_intervals_seconds"]
    assert "raw_dgs_events" in intervals and "raw_dgs_normalization" in intervals
    assert "primary_workflow_seconds" not in receipt
