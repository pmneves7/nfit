from __future__ import annotations

import json

import numpy as np
import pytest

from nfit import project_gui
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from nfit.project_gui import export_project_items, import_project_items, load_project, save_project
from nfit.project_io import NfitProject
from nfit.project_transfer import read_project_selection
from tests.project_gui_test_support import _tiny_mdhisto_data


def test_unsaved_histogram_copy_is_lazy_independent_and_saved_in_destination(tmp_path):
    data = _tiny_mdhisto_data(7.)
    source = DatasetEntry("measured", data, kind="mdhisto")
    group = DataGroup("Source", datasets=[source])
    packet = tmp_path / "selection.nfit"
    export_project_items(NfitProject([group]), "dataset", [source], packet, source_group=group)
    loaded = read_project_selection(packet)
    assert loaded.payload.items[0].data is None
    destination_group = DataGroup("Destination")
    destination = NfitProject([destination_group])
    pasted = import_project_items(destination, packet, target_role="datasets", data_group=destination_group).items[0]
    assert pasted.id != source.id
    assert pasted.data is None
    packet.unlink()  # Simulate source GUI exit and destruction of its private file.
    result = project_gui.dataset_for_slice_viewer(pasted)
    np.testing.assert_array_equal(result.signal, data.signal)
    np.testing.assert_array_equal(result.errors, data.errors)
    np.testing.assert_array_equal(result.num_events, data.num_events)
    output = tmp_path / "destination.nfit"
    save_project(destination, output)
    assert pasted._project_artifact_source.path == output
    destination._project_transfer_owners[0].cleanup()
    reopened = load_project(output)
    copied = reopened.data_groups[0].datasets[0]
    assert copied.data is None
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(copied).signal, data.signal)
    assert source.data is data and not source.metadata.get("project_artifact_path")


def test_whole_workspace_retains_cached_binning_without_reduction_or_decode(tmp_path, monkeypatch):
    from nfit import project_composites
    from nfit.analysis import artifacts
    from nfit.cache_utils import lru_store

    data = _tiny_mdhisto_data(4.)
    source_file = tmp_path / "source.nxs"
    source_file.write_bytes(b"external source")
    source = DatasetEntry("run", data, kind="mdhisto", metadata={"source_file": str(source_file)})
    source.replace_data(data, source_backed=True)
    group = DataGroup("Workspace", datasets=[source])
    project = NfitProject([group])
    config = project_gui.data_group_composite_config(group)
    config.update(enabled=True, minimum_coverage=0.)
    signature = project_gui._composite_cache_signature(group)
    key = project_gui._composite_cache_key(group)
    cache = project_composites._COMPOSITE_DATA_CACHE
    lru_store(cache, key, (signature, data), None, None)
    packet = tmp_path / "workspace.nfit"
    export_project_items(project, "group", [group], packet)
    selection = read_project_selection(packet)
    assert len(selection.project.settings["binning_cache_entries"]) == 1
    monkeypatch.setattr(artifacts, "read_dataset_artifact", lambda *_args, **_kwargs: pytest.fail("decoded during paste"))
    destination = NfitProject()
    pasted = import_project_items(destination, packet).items[0]
    new_key = project_gui._composite_cache_key(pasted)
    new_signature = project_gui._composite_cache_signature(pasted)
    backing = cache.archive_backing(new_key, new_signature)
    assert backing is not None
    assert backing[0] != packet
    assert pasted.datasets[0].data is None
    monkeypatch.undo()
    packet.unlink()
    assert backing[0].exists()
    np.testing.assert_array_equal(cache.get(new_key)[1].signal, data.signal)
    output = tmp_path / "cached-destination.nfit"
    destination.settings["cache_binnings"] = True
    monkeypatch.setattr(project_gui, "prepare_project_binning_cache", lambda *_args, **_kwargs: None)
    save_project(destination, output)
    assert cache.archive_backing(new_key, new_signature)[0] == output
    cache.clear()


def test_subgroup_copy_includes_measured_background_sources(tmp_path):
    sample = DatasetEntry("sample", _tiny_mdhisto_data(5.), kind="mdhisto")
    dummy = DatasetEntry("dummy", _tiny_mdhisto_data(2.), kind="mdhisto")
    background = DatasetGroup("Dummy", datasets=[dummy])
    selected = DatasetGroup("Sample", datasets=[sample],
                            backgrounds=[BackgroundSpec("empty mount", source_group_id=background.id, source_group=background)])
    root = DataGroup("Workspace", subgroups=[selected, background])
    packet = tmp_path / "subgroup.nfit"
    export_project_items(NfitProject([root]), "dataset_group", [selected], packet, source_group=root)
    target = DataGroup("Target")
    destination = NfitProject([target])
    pasted = import_project_items(destination, packet, target_role="datasets", data_group=target)
    assert len(pasted.items) == 2
    sample_group, dummy_group = target.subgroups
    assert sample_group.backgrounds[0].source_group is dummy_group
    assert sample_group.backgrounds[0].source_group_id == dummy_group.id != background.id
    packet.unlink()
    assert sample_group.datasets[0].data is None
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(dummy_group.datasets[0]).signal,
                                  dummy.data.signal)


def test_cached_data_rejected_when_destination_adds_masks(tmp_path):
    data = _tiny_mdhisto_data(4.)
    source_file = tmp_path / "source.nxs"
    source_file.write_bytes(b"source")
    source = DatasetEntry("run", data, kind="mdhisto", metadata={"source_file": str(source_file)})
    group = DataGroup("source", datasets=[source])
    config = project_gui.dataset_rebin_config(source)
    config.update(enabled=True, minimum_coverage=0.)
    signature = project_gui._viewer_view_signature(source, [], config)
    project_gui._VIEWER_VIEW_CACHE[source.id] = (signature, data)
    packet = tmp_path / "masked.nfit"
    export_project_items(NfitProject([group]), "dataset", [source], packet, source_group=group)
    target = DataGroup("target", masks=[MaskSpec("changed mask", "box", {"H": [-1., 1.]})])
    copied = import_project_items(NfitProject([target]), packet, target_role="datasets", data_group=target).items[0]
    assert not project_gui._VIEWER_VIEW_CACHE.has_signature(copied.id, project_gui._viewer_view_signature(copied, target.masks, config))
    project_gui._VIEWER_VIEW_CACHE.clear()


def test_changed_external_source_does_not_restore_old_binning(tmp_path):
    from nfit import project_composites
    data = _tiny_mdhisto_data(3.)
    source_file = tmp_path / "source.nxs"
    source_file.write_bytes(b"old")
    source = DatasetEntry("run", data, kind="mdhisto", metadata={"source_file": str(source_file)})
    group = DataGroup("source", datasets=[source])
    config = project_gui.data_group_composite_config(group)
    config.update(enabled=True, minimum_coverage=0.)
    cache = project_composites._COMPOSITE_DATA_CACHE
    cache[project_gui._composite_cache_key(group)] = (project_gui._composite_cache_signature(group), data)
    packet = tmp_path / "changed.nfit"
    export_project_items(NfitProject([group]), "group", [group], packet)
    source_file.write_bytes(b"different source contents")
    copied = import_project_items(NfitProject(), packet).items[0]
    assert cache.archive_backing(project_gui._composite_cache_key(copied), project_gui._composite_cache_signature(copied)) is None
    cache.clear()


def test_transfer_manifest_is_data_only_and_validated(tmp_path):
    path = tmp_path / "invalid.nfit"
    import zipfile
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", json.dumps({"project_selection": {"schema": "unsupported"}}))
    with pytest.raises(ValueError, match="Unsupported"):
        read_project_selection(path)


def test_stored_viewer_plot_transfers_settings_and_reconnects_copied_source(tmp_path):
    from nfit.plot_recipes import new_plot_entry

    dataset = DatasetEntry("measured", _tiny_mdhisto_data(5.), kind="mdhisto")
    settings = {"x_dim": "H", "y_dim": "E", "xlim": [-.5, .5],
                "ylim": [0., 30.], "selections": {1: [-.03, .03]},
                "integrate_checks": {1: True}, "cmap": "magma",
                "color_scale": "symlog", "smoothing_x": .7, "font_size": 19.}
    plot = new_plot_entry("My slice", dataset.id, settings, plot_type="mdhisto_slice")
    source = DataGroup("Source", datasets=[dataset], plots=[plot])
    project = NfitProject([source])
    workspace_packet = tmp_path / "workspace.nfit"
    export_project_items(project, "group", [source], workspace_packet)
    destination = NfitProject()
    copied = import_project_items(destination, workspace_packet).items[0]
    serialized_settings = json.loads(json.dumps(settings))
    assert copied.plots[0].settings == serialized_settings
    assert copied.plots[0].sources[0].dataset_id == copied.datasets[0].id != dataset.id
    # Also support copying a stored plot after its run was pasted separately.
    packet = tmp_path / "plot.nfit"
    export_project_items(project, "plot", [plot], packet, source_group=source)
    pasted = import_project_items(destination, packet, target_role="plots", data_group=copied).items[0]
    assert pasted.settings == serialized_settings
    assert pasted.sources[0].dataset_id == copied.datasets[0].id
    pasted.settings["smoothing_x"] = 1.2
    assert copied.plots[0].settings["smoothing_x"] == .7
    output = tmp_path / "saved.nfit"
    save_project(destination, output)
    assert load_project(output).data_groups[0].plots[0].settings["xlim"] == [-.5, .5]


def test_reduced_events_are_copied_lazily_and_survive_source_shutdown(tmp_path, monkeypatch):
    from nfit.raw_dgs_cache import cache_event_chunks, iter_cached_event_chunks

    raw = tmp_path / "raw.nxs.h5"
    raw.touch()
    dataset = DatasetEntry("run", None, kind="raw_dgs", metadata={"source_file": str(raw)})
    events = np.arange(114.).reshape(19, 6)
    list(cache_event_chunks(dataset, "reduction settings", {}, {}, iter([(events, 23)])))
    original = dataset._raw_dgs_reduction_cache
    group = DataGroup("runs", datasets=[dataset])
    packet = tmp_path / "reduced.nfit"
    export_project_items(NfitProject([group]), "group", [group], packet)
    destination = NfitProject()
    copied = import_project_items(destination, packet).items[0].datasets[0]
    assert copied.data is None
    assert copied._raw_dgs_reduction_cache.signature == original.signature
    original.staging.cleanup()
    packet.unlink()
    with copied._raw_dgs_reduction_cache.open() as archive:
        chunks = list(iter_cached_event_chunks(archive, 1024))
    np.testing.assert_array_equal(np.concatenate([item[0] for item in chunks]), events)
    assert sum(item[1] for item in chunks) == 23
    output = tmp_path / "saved.nfit"
    save_project(destination, output)
    destination._project_transfer_owners[0].cleanup()
    reopened = load_project(output).data_groups[0].datasets[0]
    with reopened._raw_dgs_reduction_cache.open() as archive:
        np.testing.assert_array_equal(archive["events_0"], events)


def test_partial_copy_preserves_inherited_masks_and_collection_contents(tmp_path):
    outer = MaskSpec("root mask", "box", {"H": [0., 1.]})
    inner = MaskSpec("parent mask", "box", {"K": [0., 1.]})
    dataset = DatasetEntry("run", _tiny_mdhisto_data(5.))
    subgroup = DatasetGroup("selected", datasets=[dataset])
    parent = DatasetGroup("parent", subgroups=[subgroup], masks=[inner])
    root = DataGroup("Source", subgroups=[parent], masks=[outer])
    packet = tmp_path / "subgroup.nfit"
    export_project_items(NfitProject([root]), "dataset_group", [subgroup], packet, source_group=root)
    target = DataGroup("target")
    destination = NfitProject([target])
    copied = import_project_items(destination, packet, target_role="datasets", data_group=target).items[0]
    assert [item.name for item in copied.masks] == ["root mask", "parent mask"]
    collection = tmp_path / "collection.nfit"
    export_project_items(NfitProject([root]), "datasets", [parent, outer], collection, source_group=root)
    other = DataGroup("other")
    project = NfitProject([other])
    import_project_items(project, collection, target_role="datasets", data_group=other)
    assert len(other.subgroups) == 1
    assert [item.name for item in other.masks] == ["root mask"]


def test_saved_copy_script_replays_after_original_packet_is_removed(tmp_path, monkeypatch):
    from nfit.project_transfer import project_selection_script, save_project_selection

    group = DataGroup("Workspace", datasets=[DatasetEntry("run", _tiny_mdhisto_data(5.))])
    packet = tmp_path / "transient.nfit"
    export_project_items(NfitProject([group]), "group", [group], packet)
    permanent = tmp_path / "copied.nfit"
    save_project_selection(read_project_selection(packet), permanent)
    packet.unlink()
    monkeypatch.chdir(tmp_path)
    exec(project_selection_script(permanent), {})
    reopened = load_project(tmp_path / "destination.nfit")
    copied = next(item for root in reopened.data_groups for item in root.datasets)
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(copied).signal, group.datasets[0].data.signal)


def test_cancelled_save_preserves_pasted_backing_and_existing_destination(tmp_path):
    from nfit.operation_control import operation_progress

    packet = tmp_path / "copied.nfit"
    group = DataGroup("source", datasets=[DatasetEntry("run", _tiny_mdhisto_data(3.))])
    export_project_items(NfitProject([group]), "group", [group], packet)
    destination = NfitProject()
    copied = import_project_items(destination, packet).items[0].datasets[0]
    target = tmp_path / "existing.nfit"
    save_project(NfitProject(), target)
    before = target.read_bytes()

    def cancel(_event):
        raise RuntimeError("cancelled")

    with operation_progress(cancel), pytest.raises(RuntimeError, match="cancelled"):
        save_project(destination, target)
    assert target.read_bytes() == before
    assert copied._project_artifact_source.path != target
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(copied).signal, group.datasets[0].data.signal)


def test_recopied_edited_data_use_current_arrays_not_previous_artifact(tmp_path):
    group = DataGroup("source", datasets=[DatasetEntry("run", _tiny_mdhisto_data(3.))])
    first = tmp_path / "first.nfit"
    export_project_items(NfitProject([group]), "group", [group], first)
    project = NfitProject()
    copied_group = import_project_items(project, first).items[0]
    copied = copied_group.datasets[0]
    copied.replace_data(_tiny_mdhisto_data(8.))
    second = tmp_path / "edited.nfit"
    export_project_items(project, "dataset", [copied], second, source_group=copied_group)
    target = DataGroup("target")
    destination = NfitProject([target])
    new = import_project_items(destination, second, target_role="datasets", data_group=target).items[0]
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(new).signal, copied.data.signal)


def test_generic_point_payloads_transfer_without_instrument_assumptions(tmp_path):
    from nfit.dataset import PointData4D, PointListData

    point_list = PointListData({"T": [1., 2., 3.], "chi": [.1, .2, .3], "error": [.01, .02, .03]},
                              units={"T": "K"}, coordinate_names=["T"],
                              channels=[{"label": "chi", "value": "chi", "error": "error"}])
    point4 = PointData4D([1., 2.], [0., 0.], [0., 0.], [0., 1.], [3., 4.], [.3, .4], temperature=5.)
    root = DataGroup("measurements", datasets=[DatasetEntry("chi(T)", point_list), DatasetEntry("scattering", point4)])
    packet = tmp_path / "points.nfit"
    export_project_items(NfitProject([root]), "group", [root], packet)
    destination = NfitProject()
    copied = import_project_items(destination, packet).items[0]
    assert all(item.data is None for item in copied.datasets)
    for item in copied.datasets:
        project_gui._ensure_dataset_data_loaded(item)
    for column in point_list.columns:
        np.testing.assert_array_equal(copied.datasets[0].data.columns[column], point_list.columns[column])
    assert copied.datasets[0].data.units == point_list.units
    np.testing.assert_array_equal(copied.datasets[1].data.intensity, point4.intensity)
    np.testing.assert_array_equal(copied.datasets[1].data.sigma, point4.sigma)


def test_repeated_workspace_copies_preserve_independent_analysis_assets(tmp_path):
    from nfit.analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
    from nfit.project_archive import open_project_artifact, write_project_manifest
    from nfit.project_io import _project_manifest_for_save

    analysis = AnalysisEntry("table", "dataset_clone", [], {})
    member = f"assets/analyses/{analysis.id}/table.csv"
    analysis.result = AnalysisResultRecord("recipe", {}, [AnalysisOutputRef("table", "table", "table", artifact_path=member)], "success", "now")
    root = DataGroup("Workspace", analyses=[analysis])
    source = NfitProject([root])
    source_path = tmp_path / "source.nfit"
    write_project_manifest(source_path, _project_manifest_for_save(source, source_path), binning_artifacts={member: b"first"})
    source._project_path = source_path
    packet = tmp_path / "first.nfit"
    export_project_items(source, "group", [root], packet)
    target = NfitProject()
    first = import_project_items(target, packet).items[0]
    write_project_manifest(source_path, _project_manifest_for_save(source, source_path), binning_artifacts={member: b"second"})
    export_project_items(source, "group", [root], tmp_path / "second.nfit")
    second = import_project_items(target, tmp_path / "second.nfit").items[0]
    paths = [group.analyses[0].result.outputs[0].artifact_path for group in [first, second]]
    assert paths[0] != paths[1]
    saved = tmp_path / "saved.nfit"
    save_project(target, saved)
    for path, expected in zip(paths, [b"first", b"second"], strict=True):
        with open_project_artifact(saved, path) as reader:
            assert reader.read() == expected
    # Copy a pasted workspace again before saving it: pending assets still work.
    next_packet = tmp_path / "again.nfit"
    export_project_items(target, "group", [first], next_packet)
    another = NfitProject()
    third = import_project_items(another, next_packet).items[0]
    save_project(another, tmp_path / "third.nfit")
    with open_project_artifact(tmp_path / "third.nfit", third.analyses[0].result.outputs[0].artifact_path) as reader:
        assert reader.read() == b"first"
