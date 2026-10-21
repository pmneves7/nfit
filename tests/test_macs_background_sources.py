"""Individual MACS background choices remain visible, portable source recipes."""

import copy

import pytest

from nfit import DataGroup, NfitProject, load_project, save_project
from nfit.macs_background_sources import (
    effective_macs_background_config,
    list_macs_background_sources,
    set_macs_background_source_settings,
)
from nfit.project_imports import import_dataset_paths
from nfit.project_paths import portable_project_manifest, resolve_project_manifest
from nfit.reduction_recipes import (
    apply_reduction_recipe,
    effective_reduction_config,
    export_reduction_recipe,
    reduction_recipe_script,
    set_reduction_settings,
)
from tests.test_macs import _write_macs_nexus


def _group(tmp_path):
    root = DataGroup("MACS")
    import_dataset_paths(root, [_write_macs_nexus(tmp_path / "sample.ng0")])
    group = root.subgroups[0]
    paths = [tmp_path / "background1.ng0", tmp_path / "background2.ng0"]
    set_reduction_settings(group, {"background_files": paths})
    return root, group, paths


def test_background_records_are_copies_and_inherit_sample_choices(tmp_path):
    _, group, paths = _group(tmp_path)
    source = str(paths[0])
    defaults = list_macs_background_sources(group)
    assert defaults[0] == {"source_file": source, "enabled": True,
                           "scale_factor": 1.0, "reduction_overrides": {}}
    defaults[0]["enabled"] = False
    assert list_macs_background_sources(group)[0]["enabled"]
    dataset = group.datasets[0]
    set_reduction_settings(group, {"a3_offset_deg": 3.0}, dataset_ids=[dataset.id])
    set_macs_background_source_settings(group, source, scale_factor=1.5,
                                       reduction_overrides={"masked_analyzer_channels": [2, 4]})
    result = effective_macs_background_config(group, source, sample_dataset=dataset)
    assert result["a3_offset_deg"] == 3.0
    assert result["masked_analyzer_channels"] == [2, 4]
    assert effective_reduction_config(group, dataset)["masked_analyzer_channels"] == []
    assert len(group.datasets) == 1


def test_background_edits_merge_and_inherit_individual_fields_without_raw_reload(tmp_path):
    _, group, paths = _group(tmp_path)
    dataset = group.datasets[0]
    points = dataset.data
    options = copy.deepcopy(dataset.metadata["import_options"])
    group.metadata["composite"]["stale"] = False
    edit = set_macs_background_source_settings(group, paths[0], enabled=False, scale_factor=2.0,
                                             reduction_overrides={"a3_offset_deg": 1.0})
    assert edit.settings_changed
    assert edit.changed_dataset_ids == (dataset.id,)
    assert group.metadata["composite"]["stale"]
    assert dataset.data is points
    assert dataset.metadata["import_options"] == options
    set_macs_background_source_settings(group, paths[0], reduction_overrides={"incident_filter": "out"})
    record = list_macs_background_sources(group)[0]
    assert record["reduction_overrides"] == {"a3_offset_deg": 1.0, "incident_filter": "out"}
    set_macs_background_source_settings(group, paths[0], reduction_overrides={"a3_offset_deg": None}, inherit=True)
    record = list_macs_background_sources(group)[0]
    assert record["reduction_overrides"] == {"incident_filter": "out"}
    assert not record["enabled"] and record["scale_factor"] == 2.0
    set_macs_background_source_settings(group, paths[0], inherit=True)
    assert "background_source_overrides" not in group.metadata["macs"]
    assert not set_macs_background_source_settings(group, paths[0], inherit=True).settings_changed


@pytest.mark.parametrize("updates", [
    {"enabled": 1}, {"scale_factor": float("nan")}, {"scale_factor": True},
    {"scale_factor": "bad"}, {"reduction_overrides": {"background_scale": 2.}},
    {"reduction_overrides": {"unknown": 1}}, {"reduction_overrides": {"monitor_target": 0}},
    {"reduction_overrides": {"masked_analyzer_channels": [21]}},
    {"reduction_overrides": {"background_files": []}, "inherit": True},
])
def test_background_source_invalid_updates_are_atomic(tmp_path, updates):
    _, group, paths = _group(tmp_path)
    metadata = copy.deepcopy(group.metadata)
    with pytest.raises((ValueError, TypeError)):
        set_macs_background_source_settings(group, paths[0], **updates)
    assert group.metadata == metadata


def test_background_source_membership_and_global_validation(tmp_path):
    _, group, paths = _group(tmp_path)
    with pytest.raises(ValueError, match="belong"):
        set_macs_background_source_settings(group, tmp_path / "other.ng0", enabled=False)
    set_macs_background_source_settings(group, paths[0], reduction_overrides={"apply_detector_efficiency": True})
    before = copy.deepcopy(group.metadata)
    with pytest.raises(ValueError, match="Choose vanadium"):
        set_reduction_settings(group, {"vanadium_files": [tmp_path / "vanadium.ng0"]})
    assert group.metadata == before
    set_reduction_settings(group, {"background_files": [paths[1]]})
    assert "background_source_overrides" not in group.metadata["macs"]
    assert list_macs_background_sources(group)[0]["source_file"] == str(paths[1])


def test_background_source_portability_and_replay(tmp_path):
    root, group, paths = _group(tmp_path)
    set_macs_background_source_settings(group, paths[0], enabled=False, scale_factor=.75,
                                       reduction_overrides={"masked_analyzer_channels": [2],
                                                            "vanadium_files": [tmp_path / "calibration.ng0"]})
    project = NfitProject(data_groups=[root])
    path = tmp_path / "example.nfit"
    save_project(project, path)
    restored = load_project(path).data_groups[0].subgroups[0]
    assert list_macs_background_sources(restored) == list_macs_background_sources(group)
    namespace = {}
    exec(reduction_recipe_script(group), namespace)
    assert list_macs_background_sources(namespace["group"]) == list_macs_background_sources(group)
    recipe = export_reduction_recipe(group)
    recipe["background_source_overrides"][0]["scale_factor"] = 3.0
    dataset = group.datasets[0]
    points = dataset.data
    group.metadata["composite"]["stale"] = False
    edit = apply_reduction_recipe(group, recipe)
    assert edit.settings_changed and group.metadata["composite"]["stale"]
    assert dataset.data is points
    assert list_macs_background_sources(group)[0]["scale_factor"] == 3.0


def test_background_records_and_override_calibrations_relocate_with_file_list(tmp_path):
    _, group, paths = _group(tmp_path)
    calibration = tmp_path / "calibration.ng0"
    set_macs_background_source_settings(group, paths[0], reduction_overrides={"vanadium_files": [calibration]})
    payload = {"data_groups": [{"metadata": {"macs": effective_reduction_config(group)}}]}
    project_path = tmp_path / "project.nfit"
    portable = portable_project_manifest(payload, project_path)
    record = portable["data_groups"][0]["metadata"]["macs"]["background_source_overrides"][0]
    assert record["source_file"] == paths[0].name
    assert record["reduction_overrides"]["vanadium_files"] == [calibration.name]
    destination = tmp_path / "moved"
    restored, _ = resolve_project_manifest(portable, destination / "project.nfit",
                                          path_mappings={tmp_path: destination})
    config = restored["data_groups"][0]["metadata"]["macs"]
    record = config["background_source_overrides"][0]
    assert record["source_file"] == config["background_files"][0] == str(destination / paths[0].name)
    assert record["reduction_overrides"]["vanadium_files"] == [str(destination / calibration.name)]
