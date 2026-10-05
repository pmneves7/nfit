from __future__ import annotations

import ast
import copy
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from nfit.analysis.artifacts import write_dataset_artifact
from nfit.measurement_dependencies import independent_source_dependencies
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup, ModelComponentSpec
from nfit.project_archive import read_project_manifest
from nfit.project_io import NfitProject, _project_to_dict
from nfit.project_paths import portable_project_manifest, resolve_project_manifest
from tests.plotting_test_data import tiny_mdhisto_data


def _tree(tmp_path, name):
    experiment = tmp_path / name / "experiment"
    source = experiment / "nexus" / "run.nxs"
    calibration = experiment / "shared" / "calibrations" / "vanadium.nxs"
    project = experiment / "shared" / "nfit" / "sample.nfit"
    for path in (source, calibration, project):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return experiment, source, calibration, project


def _manifest(source, calibration, project):
    identity = str(source) + "#experiment=1"
    dataset = DatasetEntry("run", None, kind="raw_dgs_nexus", metadata={
        "source_file": str(source),
        "source_selection_lineage": {"version": 1, "source_identity": identity},
        "source_lineage": {"version": 1, "source_ids": [identity, "named-acquisition"]},
        "reduction_overrides": {"mask_file": str(calibration)},
        "raw_dgs_reduction_cache": {"signature": json.dumps([8, [str(source), 7, 13, 19],
            [str(calibration), 7, 13, 19]])},
    })
    config = {"format": "raw-direct-geometry-nexus", "source_files": [str(source)],
        "normalization_file": str(calibration), "mask_file": str(calibration),
        "ub_file": str(calibration), "enabled": True}
    recipe = {"source_selection": {"directory": str(source.parent), "resolved_files": [str(source)],
        "appearances": [{"path": str(source), "source_identity": identity}]},
        "shared_defaults": {"normalization_file": str(calibration)},
        "runs": [{"source_identity": {"path": str(source), "experiment_index": 1},
                  "source_file": str(source)}],
        "provenance": {"calibrations_by_run": {dataset.id: {"mask_file": {
            "path": str(calibration), "available": True, "size_bytes": 7, "mtime_ns": 13}}}}}
    child = DatasetGroup("runs", [dataset], metadata={"raw_dgs": config, "reduction_recipe": recipe})
    artifact = DatasetEntry("embedded", None, metadata={"source_file": "assets/datasets/embedded/data.npz",
        "project_artifact_path": "assets/datasets/embedded/data.npz"})
    group = DataGroup("workspace", datasets=[artifact], subgroups=[child], metadata={
        "source_project": str(project.with_name("reference.nfit")),
        "comment": str(calibration), "unrelated": {"path": str(calibration)},
    })
    group.models["electronic"] = ModelComponentSpec("electronic", config={"source_path": str(calibration)})
    signature = json.dumps([{"normalization_file": str(calibration)},
        ["source", str(source), 7, 13], json.dumps({"mask_file": str(calibration)})])
    return _project_to_dict(NfitProject([group], settings={"binning_cache_entries": [{"signature": signature}]}))


def test_relative_paths_cover_recipes_provenance_models_and_leave_text_untouched(tmp_path):
    experiment, source, calibration, project = _tree(tmp_path, "old")
    project.with_name("reference.nfit").write_bytes(b"reference")
    original = _manifest(source, calibration, project)
    saved = portable_project_manifest(original, project)
    node = saved["data_groups"][0]["subgroups"][0]
    assert node["datasets"][0]["metadata"]["source_file"] == "../../nexus/run.nxs"
    assert node["metadata"]["raw_dgs"]["normalization_file"] == "../calibrations/vanadium.nxs"
    assert saved["data_groups"][0]["models"][0]["config"]["source_path"] == "../calibrations/vanadium.nxs"
    assert saved["data_groups"][0]["datasets"][0]["metadata"]["source_file"].startswith("assets/")
    assert saved["data_groups"][0]["metadata"]["comment"] == str(calibration)
    assert saved["data_groups"][0]["metadata"]["unrelated"]["path"] == str(calibration)
    assert original["data_groups"][0]["subgroups"][0]["metadata"]["raw_dgs"]["normalization_file"] == str(calibration)

    new_experiment = tmp_path / "new" / "experiment"
    shutil.copytree(experiment, new_experiment)
    shutil.rmtree(experiment)
    opened = new_experiment / "shared" / "nfit" / project.name
    resolved, resolution = resolve_project_manifest(saved, opened)
    node = resolved["data_groups"][0]["subgroups"][0]
    new_source = str(new_experiment / "nexus" / source.name)
    new_calibration = str(new_experiment / "shared" / "calibrations" / calibration.name)
    assert node["datasets"][0]["metadata"]["source_file"] == new_source
    assert node["metadata"]["raw_dgs"]["mask_file"] == new_calibration
    assert node["metadata"]["reduction_recipe"]["runs"][0]["source_identity"]["path"] == new_source
    assert node["datasets"][0]["metadata"]["source_lineage"]["source_ids"] == [new_source + "#experiment=1", "named-acquisition"]
    assert node["metadata"]["raw_dgs"]["enabled"] is True
    signature = json.loads(resolved["settings"]["binning_cache_entries"][0]["signature"])
    assert signature[0]["normalization_file"] == new_calibration
    assert signature[1] == ["source", new_source, 7, 13]
    assert json.loads(signature[2])["mask_file"] == new_calibration
    assert copy.deepcopy(resolution) is resolution


def test_legacy_aliases_resolve_from_shared_ancestry_without_searching_by_basename(tmp_path):
    experiment, source, calibration, project = _tree(tmp_path, "private")
    project.with_name("reference.nfit").write_bytes(b"reference")
    saved = _manifest(source, calibration, project)
    public = tmp_path / "public" / "experiment"
    shutil.copytree(experiment, public)
    shutil.rmtree(experiment)
    resolved, resolution = resolve_project_manifest(saved, public / "shared" / "nfit" / project.name)
    config = resolved["data_groups"][0]["subgroups"][0]["metadata"]["raw_dgs"]
    assert config["normalization_file"] == str(public / "shared" / "calibrations" / calibration.name)
    assert config["source_files"] == [str(public / "nexus" / source.name)]
    assert resolved["data_groups"][0]["metadata"]["source_project"] == str(public / "shared" / "nfit" / "reference.nfit")
    unrelated = tmp_path / "missing" / "vanadium.nxs"
    assert resolution.resolve(str(unrelated)) == str(unrelated)


def test_inaccessible_foreign_alias_is_unavailable_and_resolves_to_shared_tree(tmp_path, monkeypatch):
    experiment, source, calibration, project = _tree(tmp_path, "foreign-home")
    project.with_name("reference.nfit").write_bytes(b"reference")
    payload = _manifest(source, calibration, project)
    public = tmp_path / "shared-volume" / "experiment"
    shutil.copytree(experiment, public)
    foreign = tmp_path / "foreign-home"
    original_exists, original_resolve = Path.exists, Path.resolve

    def inaccessible_exists(path):
        if path.is_relative_to(foreign):
            raise PermissionError("foreign home cannot be traversed")
        return original_exists(path)

    def inaccessible_resolve(path, *args, **kwargs):
        if path.is_relative_to(foreign):
            raise PermissionError("foreign home cannot be traversed")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", inaccessible_exists)
    monkeypatch.setattr(Path, "resolve", inaccessible_resolve)
    resolved, resolution = resolve_project_manifest(payload, public / "shared" / "nfit" / project.name)
    config = resolved["data_groups"][0]["subgroups"][0]["metadata"]["raw_dgs"]
    assert config["source_files"] == [str(public / "nexus" / source.name)]
    assert config["normalization_file"] == str(public / "shared" / "calibrations" / calibration.name)
    assert resolved["data_groups"][0]["metadata"]["source_project"] == str(public / "shared" / "nfit" / "reference.nfit")
    assert resolution.resolve(str(foreign / "missing" / "run.nxs")) == str(foreign / "missing" / "run.nxs")


def test_external_roots_use_explicit_mapping_and_preserve_opaque_strings(tmp_path):
    _experiment, source, calibration, project = _tree(tmp_path, "workspace")
    external = tmp_path / "external-original" / "bank"
    external.mkdir(parents=True)
    path = external / "model.npz"
    path.write_bytes(b"model")
    saved = portable_project_manifest(_manifest(source, calibration, project), project)
    saved["data_groups"][0]["models"][0]["config"]["source_path"] = str(path)
    new_root = tmp_path / "replacement-bank"
    shutil.copytree(external, new_root)
    shutil.rmtree(external)
    resolved, _resolution = resolve_project_manifest(saved, project,
        path_mappings={str(external): str(new_root)})
    assert resolved["data_groups"][0]["models"][0]["config"]["source_path"] == str(new_root / path.name)
    assert resolved["data_groups"][0]["metadata"]["comment"] == str(calibration)


def test_cached_metadata_rebases_source_primitives_without_copying_arrays(tmp_path):
    experiment, source, calibration, project = _tree(tmp_path, "old")
    saved = portable_project_manifest(_manifest(source, calibration, project), project)
    public = tmp_path / "new" / "experiment"
    shutil.copytree(experiment, public)
    shutil.rmtree(experiment)
    _resolved, resolution = resolve_project_manifest(saved, public / "shared" / "nfit" / project.name)
    data = tiny_mdhisto_data()
    dependencies = independent_source_dependencies(data.errors**2, str(source))
    data = data.with_updates(metadata={"source_file": str(source), "source_lineage": {
        "version": 1, "source_ids": [str(source) + "#experiment=0"]}}, source_dependencies=dependencies)
    prepared = resolution.prepare_data(data)
    assert prepared.signal is data.signal
    assert prepared.errors is data.errors
    assert prepared.source_dependencies.coefficients is data.source_dependencies.coefficients
    assert prepared.source_dependencies.source_ids[0].startswith(str(public / "nexus" / source.name) + ":")
    assert prepared.metadata["source_lineage"]["source_ids"] == [str(public / "nexus" / source.name) + "#experiment=0"]
    dataset = DatasetEntry("loaded source", None)
    dataset._project_path_resolution = resolution
    loaded = dataset.replace_data(data, source_backed=True)
    assert loaded.signal is data.signal
    assert loaded.metadata["source_file"] == str(public / "nexus" / source.name)
    assert dataset.copy()._project_path_resolution is resolution
    assert copy.deepcopy(dataset)._project_path_resolution is resolution


def test_moved_project_retains_lazy_saved_histogram_and_source_lineage(tmp_path, monkeypatch):
    from nfit import project_data, project_gui

    experiment, source, _calibration, project_path = _tree(tmp_path, "old")
    data = tiny_mdhisto_data().with_updates(metadata={"source_file": str(source), "source_lineage": {
        "version": 1, "source_ids": [str(source)]}})
    write_dataset_artifact(data, source)
    dataset = DatasetEntry("scan", data, kind="mdhisto", metadata={"source_file": str(source)})
    project_data.dataset_rebin_config(dataset).update(enabled=True, minimum_coverage=0.)
    project = NfitProject([DataGroup("workspace", datasets=[dataset])], settings={"cache_binnings": True})
    project_gui.save_project(project, project_path)
    expected = project_data.dataset_for_slice_viewer(dataset)
    manifest = read_project_manifest(project_path)
    assert manifest["data_groups"][0]["datasets"][0]["metadata"]["source_file"] == "../../nexus/run.nxs"
    new_experiment = tmp_path / "new" / "experiment"
    shutil.copytree(experiment, new_experiment)
    shutil.rmtree(experiment)
    project_data._VIEWER_VIEW_CACHE.clear()
    new_project = new_experiment / "shared" / "nfit" / project_path.name
    reopened = project_gui.load_project(new_project)
    restored = reopened.data_groups[0].datasets[0]
    assert restored.data is None
    records = project_data._VIEWER_VIEW_CACHE.resource_records()
    assert any(record.key == dataset.id and record.data is None for record in records)
    monkeypatch.setattr(project_data, "_rebin_mdhisto_data", lambda *_a, **_k: pytest.fail("recomputed saved cube"))
    cached = project_data.dataset_for_slice_viewer(restored)
    assert cached.signal == pytest.approx(expected.signal)
    assert cached.metadata["source_lineage"]["source_ids"] == [str(new_experiment / "nexus" / source.name)]
    project_data._VIEWER_VIEW_CACHE.clear()


def test_save_as_reuses_cold_member_and_preserves_historical_metadata_aliases(tmp_path, monkeypatch):
    from nfit import project_data, project_gui, rebin_cache

    experiment, source, _calibration, original = _tree(tmp_path, "old")
    data = tiny_mdhisto_data().with_updates(metadata={"source_lineage": {
        "version": 1, "source_ids": [str(source)]}})
    dataset = DatasetEntry("scan", data, kind="mdhisto", metadata={"source_file": str(source)})
    project_data.dataset_rebin_config(dataset).update(enabled=True, minimum_coverage=0.)
    project = NfitProject([DataGroup("workspace", datasets=[dataset])], settings={"cache_binnings": True})
    project_gui.save_project(project, original)
    member = project.settings[project_gui.PROJECT_BINNING_CACHE_ENTRIES_KEY][0]["member"]
    with zipfile.ZipFile(original) as archive:
        crc = archive.getinfo(member).CRC
    relocated = tmp_path / "relocated" / "experiment"
    shutil.copytree(experiment, relocated)
    shutil.rmtree(experiment)
    moved_original = relocated / "shared" / "nfit" / original.name
    project_data._VIEWER_VIEW_CACHE.clear()
    reopened = project_gui.load_project(moved_original)
    destination = relocated / "reports" / "copied.nfit"

    with monkeypatch.context() as patch:
        patch.setattr(project_gui, "write_dataset_artifact", lambda *_a, **_k: pytest.fail("recompressed cold cube"))
        patch.setattr(rebin_cache, "read_project_dataset_artifact", lambda *_a, **_k: pytest.fail("decoded cold cube"))
        project_gui.save_project(reopened, destination, asset_source=moved_original)
        with zipfile.ZipFile(destination) as archive:
            assert archive.getinfo(member).CRC == crc
    moved_original.unlink()
    project_data._VIEWER_VIEW_CACHE.clear()
    restored = project_gui.load_project(destination)
    with monkeypatch.context() as patch:
        patch.setattr(project_data, "_rebin_mdhisto_data", lambda *_a, **_k: pytest.fail("recomputed saved cube"))
        cached = project_data.dataset_for_slice_viewer(restored.data_groups[0].datasets[0])
    assert cached.metadata["source_lineage"]["source_ids"] == [str(relocated / "nexus" / source.name)]
    project_data._VIEWER_VIEW_CACHE.clear()


def test_moved_raw_reduction_reuses_events_and_reports_current_source(tmp_path, monkeypatch):
    from nfit import bin_raw_dgs_group, load_project, raw_dgs, raw_dgs_dataset_group, save_project
    from tests.test_raw_dgs import _write_raw_dgs
    from tests.test_raw_dgs_cache import OPTIONS, assert_equal

    experiment, source, _calibration, project_path = _tree(tmp_path, "old")
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    expected = bin_raw_dgs_group(group, **OPTIONS)
    project = NfitProject([DataGroup("raw", subgroups=[group])])
    save_project(project, project_path)
    relocated = tmp_path / "relocated" / "experiment"
    relocated.parent.mkdir()
    experiment.rename(relocated)  # Preserve source fingerprints while changing its root.
    reopened = load_project(relocated / "shared" / "nfit" / project_path.name)
    restored = reopened.data_groups[0].subgroups[0]
    monkeypatch.setattr(raw_dgs, "inspect_raw_dgs_run", lambda *_a, **_k: pytest.fail("reread raw run"))
    monkeypatch.setattr(raw_dgs, "_detector_geometry", lambda *_a, **_k: pytest.fail("reread raw geometry"))
    cached_info, recorded_paths = raw_dgs._run_info_from_cache, []

    def record_cached_info(payload):
        info = cached_info(payload)
        recorded_paths.append(info.path)
        return info

    monkeypatch.setattr(raw_dgs, "_run_info_from_cache", record_cached_info)
    actual = bin_raw_dgs_group(restored, **OPTIONS)
    assert_equal(actual, expected)
    assert actual.metadata["reduced_event_cache"] == {"hits": 1, "misses": 0}
    assert actual.metadata["raw_dgs"]["source_files"] == [str(relocated / "nexus" / source.name)]
    assert recorded_paths == [relocated / "nexus" / source.name]


def test_path_service_is_gui_independent_and_uses_no_site_specific_paths():
    from nfit import project_paths

    text = Path(project_paths.__file__).read_text()
    tree = ast.parse(text)
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(module.startswith(("PySide", "PyQt")) or module.rsplit(".", 1)[-1]
                   in ("project_gui", "project_data", "project_io", "pipeline") for module in modules)
    assert "/SNS" not in text and "ORNL" not in text
