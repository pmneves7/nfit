"""Disabled recipes survive project I/O without persisting numerical results."""

from __future__ import annotations

import json
from zipfile import ZipFile

import numpy as np
import pytest

from nfit import project_background_cache, project_data, project_gui
from nfit import project_composites as comp
from nfit.binning_lifecycle import set_binning_enabled
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from nfit.project_io import NfitProject
from nfit.project_transfer import read_project_selection
from tests.project_gui_test_support import _tiny_mdhisto_data


def _current_binning(tmp_path, kind):
    data = _tiny_mdhisto_data(4.0)
    source = tmp_path / "source.nxs"
    source.write_bytes(b"external source placeholder")
    dataset = DatasetEntry("sample", data, kind="mdhisto", metadata={"source_file": str(source)})
    dataset.replace_data(data, source_backed=True)
    group = DataGroup("workspace", datasets=[dataset])
    project = NfitProject([group], settings={"cache_binnings": True})
    if kind == "dataset":
        config = project_gui.dataset_rebin_config(dataset)
        config.update(enabled=True, minimum_coverage=0.0)
        signature = project_gui._viewer_view_signature(dataset, [], config)
        key, cache = dataset.id, project_gui._VIEWER_VIEW_CACHE
    else:
        config = comp.data_group_composite_config(group)
        config.update(enabled=True, minimum_coverage=0.0)
        signature = comp._composite_cache_signature(group)
        key, cache = comp._composite_cache_key(group), comp._COMPOSITE_DATA_CACHE
    cache[key] = signature, data
    return project, dataset, group, config, cache, key, data


def test_disabled_raw_rebin_never_substitutes_background_subtracted_viewer_data(tmp_path, monkeypatch):
    _project, dataset, _group, config, cache, _key, data = _current_binning(tmp_path, "dataset")
    try:
        dataset.replace_data(data.with_updates(signal=np.full(data.shape, 10.0)))
        background = DatasetEntry("background", data.with_updates(signal=np.full(data.shape, 3.0)), kind="mdhisto")
        dataset.backgrounds.append(BackgroundSpec("background", source_entry=background))
        config["resolution_mode"] = "bins"
        for axis_config, axis in zip(config["axes"], data.axes, strict=True):
            axis_config.update(lower=float(axis.values[0]), upper=float(axis.values[-1]),
                               num_bins=axis.values.size - 1, mode="bins",
                               auto_lower=False, auto_upper=False, auto_step_size=False)
        raw = project_data.rebinned_dataset_data(dataset)
        viewer = project_data.dataset_for_slice_viewer(dataset)
        np.testing.assert_allclose(raw.signal, 10.0)
        np.testing.assert_allclose(viewer.signal, 7.0)
        set_binning_enabled(config, False)
        monkeypatch.setattr(cache, "get", lambda *_a, **_k: pytest.fail("restored disabled result"))
        monkeypatch.setattr(project_data, "_rebin_mdhisto_data", lambda *_a, **_k: pytest.fail("rebinned disabled result"))
        assert project_data.dataset_for_slice_viewer(dataset) is viewer
        with pytest.raises(ValueError, match="disabled.*background subtraction.*Enable"):
            project_data.rebinned_dataset_data(dataset)
        with pytest.raises(ValueError, match="disabled.*background subtraction.*Enable"):
            project_data._viewer_data_before_scale_uncached(dataset)
        np.testing.assert_allclose(project_data.dataset_for_slice_viewer(dataset).signal, 7.0)
    finally:
        cache.clear()


def test_sample_owned_background_uses_sources_when_reference_output_binning_is_disabled(monkeypatch):
    sample = DatasetGroup("sample", datasets=[DatasetEntry("sample", _tiny_mdhisto_data(10.0), kind="mdhisto")])
    reference = DatasetGroup("reference", datasets=[DatasetEntry("reference", _tiny_mdhisto_data(3.0), kind="mdhisto")])
    root = DataGroup("workspace", subgroups=[sample, reference])
    sample_scope = comp._composite_scope(root, sample)
    reference_scope = comp._composite_scope(root, reference)
    sample.backgrounds.append(BackgroundSpec("reference", source_group_id=reference.id,
                                             source_group=reference, projection="center"))
    for scope in (sample_scope, reference_scope):
        config = comp.data_group_composite_config(scope)
        config.update(enabled=True, minimum_coverage=0.0, resolution_mode="bins")
        for settings, axis in zip(config["axes"], scope.node.datasets[0].data.axes, strict=True):
            settings.update(lower=float(axis.values[0]), upper=float(axis.values[-1]),
                            num_bins=axis.values.size - 1, mode="bins",
                            auto_lower=False, auto_upper=False, auto_step_size=False)
    set_binning_enabled(comp.data_group_composite_config(reference_scope), False)
    comp._COMPOSITE_DATA_CACHE.clear()
    monkeypatch.setattr(comp, "_cached_composite_dataset_data", lambda *_a, **_k: pytest.fail("requested standalone reference output"))
    original_get = comp._COMPOSITE_DATA_CACHE.get
    reference_key = comp._composite_cache_key(reference_scope)

    def get(key, *args, **kwargs):
        if key == reference_key:
            pytest.fail("restored standalone reference output")
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(comp._COMPOSITE_DATA_CACHE, "get", get)
    try:
        result = comp.composite_dataset_data(sample_scope)
        np.testing.assert_allclose(result.signal, 7.0)
        assert comp._COMPOSITE_DATA_CACHE.peek_resident(reference_key) is None
    finally:
        comp._COMPOSITE_DATA_CACHE.clear()


@pytest.mark.parametrize("tier", ["resident", "compressed", "disk"])
@pytest.mark.parametrize("stale", [False, True])
def test_disabled_composite_donor_reuse_requires_current_resident_data(tmp_path, monkeypatch, tier, stale):
    _project, _dataset, group, config, cache, key, data = _current_binning(tmp_path, "composite")
    named = comp.add_data_group_composite_binning(group)
    set_binning_enabled(config, False)
    config["stale"] = stale
    cache[key] = "donor signature", data
    if tier == "compressed":
        cache._compress_resident(key, max_bytes=1_000_000)
    elif tier == "disk":
        artifact = tmp_path / "donor.npz"
        project_background_cache.write_dataset_artifact(data, artifact)
        archive_path = tmp_path / "donor.nfit"
        with ZipFile(archive_path, "w") as archive:
            archive.write(artifact, "data.npz")
        cache.set_project_backing(key, signature="donor signature", project_path=archive_path,
                                  member="data.npz", lazy=True)
    monkeypatch.setattr(cache, "get", lambda *_a, **_k: pytest.fail("loaded a disabled donor"))
    try:
        match = comp._matching_composite_base_key(group, "donor signature", named)
        assert match == (key if tier == "resident" and not stale else None)
    finally:
        cache.clear()


@pytest.mark.parametrize("tier", ["resident", "compressed"])
def test_enabled_background_replay_does_not_restore_disabled_donor(tmp_path, monkeypatch, tier):
    _project, _dataset, group, config, cache, _key, data = _current_binning(tmp_path, "composite")
    background_data = _tiny_mdhisto_data(1.0)
    background = BackgroundSpec("reference", source_entry=DatasetEntry("reference", background_data, kind="mdhisto"))
    group.backgrounds.append(background)
    named = comp.add_data_group_composite_binning(group)
    donor_config = comp.data_group_composite_config_by_id(group, named)
    donor_config["stale"] = False
    set_binning_enabled(donor_config, False)
    key = comp._composite_background_cache_key(group, background, donor_config)
    signature = comp._composite_background_signature(group, background, config)
    cache[key] = signature, background_data
    if tier == "compressed":
        cache._compress_resident(key, max_bytes=1_000_000)
    original_get = cache.get

    def get(candidate, *args, **kwargs):
        if candidate == key and tier != "resident":
            pytest.fail("restored a disabled background donor")
        return original_get(candidate, *args, **kwargs)

    calls = []
    monkeypatch.setattr(cache, "get", get)
    monkeypatch.setattr(comp, "_composite_background_data", lambda *_a, **_k: (calls.append(True), background_data)[1])
    try:
        result = comp._apply_composite_backgrounds(group, data, config=config)
        np.testing.assert_allclose(result.signal, 3.0)
        assert calls == ([] if tier == "resident" else [True])
    finally:
        cache.clear()


@pytest.mark.parametrize("kind", ["dataset", "composite"])
def test_save_disabled_binning_keeps_recipe_and_resident_result_without_artifact(tmp_path, monkeypatch, kind):
    project, dataset, group, config, cache, key, data = _current_binning(tmp_path, kind)
    try:
        set_binning_enabled(config, False)
        monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", lambda *_a, **_k: pytest.fail("calculated disabled dataset"))
        monkeypatch.setattr(project_gui, "_cached_composite_dataset_data", lambda *_a, **_k: pytest.fail("calculated disabled composite"))
        path = tmp_path / "disabled.nfit"
        project_gui.save_project(project, path)
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("project.json"))
            assert not any(name.startswith("assets/binnings/") for name in archive.namelist())
        assert manifest["settings"]["binning_cache_entries"] == []
        assert cache.peek_resident(key)[1] is data
        loaded = project_gui.load_project(path)
        restored_config = (project_gui.dataset_rebin_config(loaded.data_groups[0].datasets[0]) if kind == "dataset"
                           else comp.data_group_composite_config(loaded.data_groups[0]))
        assert restored_config["binning_enabled"] is False
        assert restored_config["enabled"] is True
        assert restored_config["axes"] == config["axes"]
        assert cache.peek_resident(key)[1] is data
    finally:
        cache.clear()


@pytest.mark.parametrize("kind", ["dataset", "composite"])
@pytest.mark.parametrize("version", [4, 5, 6])
def test_disabled_binning_restore_skips_legacy_decode_and_lazy_backing(tmp_path, monkeypatch, kind, version):
    project, _dataset, _group, config, cache, _key, _data = _current_binning(tmp_path, kind)
    try:
        path = tmp_path / "previously-enabled.nfit"
        project_gui.save_project(project, path)
        assert project.settings["binning_cache_entries"]
        for entry in project.settings["binning_cache_entries"]:
            entry["format_version"] = version
        set_binning_enabled(config, False)
        cache.clear()
        monkeypatch.setattr(project_gui, "read_project_dataset_artifact", lambda *_a, **_k: pytest.fail("decoded disabled artifact"))
        monkeypatch.setattr(cache, "set_project_backing", lambda *_a, **_k: pytest.fail("restored disabled backing"))
        project_gui._restore_project_binning_cache(project, path)
        assert cache.resource_records() == ()
    finally:
        cache.clear()


def test_disabled_background_cache_is_not_saved_or_restored(tmp_path, monkeypatch):
    project, _dataset, group, config, cache, _key, data = _current_binning(tmp_path, "composite")
    background = BackgroundSpec("dummy", source_entry=DatasetEntry("dummy", data, kind="mdhisto"))
    group.backgrounds.append(background)
    key = comp._composite_background_cache_key(group, background, config)
    signature = comp._composite_background_signature(group, background, config)
    cache[key] = signature, data
    try:
        artifacts, entries = project_background_cache.background_binning_artifacts(project, tmp_path, 6)
        assert len(artifacts) == len(entries) == 1
        set_binning_enabled(config, False)
        monkeypatch.setattr(comp, "_composite_background_signature", lambda *_a, **_k: pytest.fail("prepared disabled background"))
        assert project_background_cache.background_binning_artifacts(project, tmp_path, 6) == ({}, [])
        cache.clear()
        monkeypatch.setattr(cache, "set_project_backing", lambda *_a, **_k: pytest.fail("restored disabled background"))
        project_background_cache.restore_background_binning_backing(project, tmp_path / "old.nfit", entries, lazy=True)
    finally:
        cache.clear()


@pytest.mark.parametrize("kind", ["dataset", "composite"])
def test_disabled_binning_selection_keeps_config_and_omits_artifacts(tmp_path, kind):
    project, _dataset, group, config, cache, key, data = _current_binning(tmp_path, kind)
    try:
        set_binning_enabled(config, False)
        path = tmp_path / "selection.nfit"
        project_gui.export_project_items(project, "group", [group], path)
        selection = read_project_selection(path)
        assert selection.project.settings["binning_cache_entries"] == []
        with ZipFile(path) as archive:
            assert not any(name.startswith("assets/binnings/") for name in archive.namelist())
        destination = NfitProject()
        copied = project_gui.import_project_items(destination, path).items[0]
        copied_config = (project_gui.dataset_rebin_config(copied.datasets[0]) if kind == "dataset"
                         else comp.data_group_composite_config(copied))
        assert copied_config["binning_enabled"] is False
        assert cache.peek_resident(key)[1] is data
    finally:
        cache.clear()


@pytest.mark.parametrize("kind", ["dataset", "composite"])
def test_old_selection_artifacts_are_not_installed_for_disabled_recipes(tmp_path, monkeypatch, kind):
    project, _dataset, group, config, cache, _key, _data = _current_binning(tmp_path, kind)
    try:
        set_binning_enabled(config, True)
        path = tmp_path / "old-selection.nfit"
        project_gui.export_project_items(project, "group", [group], path)
        with ZipFile(path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(contents["project.json"])
        assert manifest["settings"]["binning_cache_entries"]

        def disable(value):
            if isinstance(value, dict):
                if "binning_enabled" in value:
                    value["binning_enabled"] = False
                for child in value.values():
                    disable(child)
            elif isinstance(value, list):
                for child in value:
                    disable(child)

        disable(manifest)
        contents["project.json"] = json.dumps(manifest).encode()
        with ZipFile(path, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
        monkeypatch.setattr(cache, "set_project_backing", lambda *_a, **_k: pytest.fail("installed disabled selection cache"))
        copied = project_gui.import_project_items(NfitProject(), path).items[0]
        copied_config = (project_gui.dataset_rebin_config(copied.datasets[0]) if kind == "dataset"
                         else comp.data_group_composite_config(copied))
        assert copied_config["binning_enabled"] is False
    finally:
        cache.clear()


@pytest.mark.parametrize("standalone", [False, True])
def test_disabled_binning_workflow_script_preserves_flag_and_refuses_calculation(tmp_path, monkeypatch, standalone):
    from nfit import mdevent_dataset_group
    from nfit.workflow import composite_workflow_script, reduction_workflow_script
    from tests.test_mdevent import _write_mdevent

    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    source = mdevent_dataset_group(path)
    root = DataGroup("workspace", subgroups=[source])
    scope = comp._composite_scope(root, source)
    config = comp.data_group_composite_config(scope)
    config.update(enabled=True, binning_enabled=False)
    # Dataset recipes and named visualization recipes must retain their flags too.
    dataset_config = project_gui.dataset_rebin_config(source.datasets[0])
    dataset_config.update(enabled=True, binning_enabled=False)
    named = comp.add_data_group_composite_binning(scope, name="Disabled overview")
    script = (reduction_workflow_script(source) if standalone else
              composite_workflow_script(NfitProject([root]), root.name, node_id=source.id))
    namespace = {"__name__": "disabled_replay"}
    exec(compile(script, "disabled_replay.py", "exec"), namespace)
    assert namespace["REBIN_CONFIG"]["binning_enabled"] is False
    replay_root, replay_source = namespace["build_group"]()
    assert project_gui.dataset_rebin_config(replay_source.datasets[0])["binning_enabled"] is False
    if not standalone:
        replay_scope = comp._composite_scope(replay_root, replay_source)
        assert comp.data_group_composite_config_by_id(replay_scope, named)["binning_enabled"] is False
    monkeypatch.setattr(comp, "_composite_dataset_data", lambda *_a, **_k: pytest.fail("calculated disabled script result"))
    with pytest.raises(ValueError, match="disabled"):
        namespace["run"]()
