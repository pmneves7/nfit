"""Portable incremental archives preserve opaque scientific payloads."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nfit import compact_project, project_storage_info
from nfit.project_archive import (
    ArchiveMember,
    _project_artifact_reader_factory,
    open_project_artifact,
    read_project_artifact,
    read_project_manifest,
    replace_analysis_artifacts,
    replace_dataset_artifact,
    write_project_manifest,
)
from nfit.project_store import StoreConflictError, inspect_project_store, open_project_zip

MEMBER = "assets/binnings/cache/data.npz"
RAW = "assets/reduced_events/run/events.npz"


def test_metadata_save_reuses_artifacts_without_reading_them(tmp_path, monkeypatch):
    import nfit.project_archive as archive_api

    path = tmp_path / "project.nfit"
    payload = b"scientific data" * 100_000
    write_project_manifest(path, {"name": "before"}, storage_mode="incremental",
                           binning_artifacts={MEMBER: payload}, reduced_event_artifacts={RAW: payload})
    with open_project_zip(path) as archive:
        offsets = {name: archive.getinfo(name).header_offset for name in (MEMBER, RAW)}
    before = path.stat().st_size
    references = {name: ArchiveMember(path, name) for name in (MEMBER, RAW)}
    monkeypatch.setattr(archive_api, "_copy_archive_stream", lambda *_: pytest.fail("Read unchanged data"))
    write_project_manifest(path, {"name": "after"}, binning_artifacts={MEMBER: references[MEMBER]},
                           reduced_event_artifacts={RAW: references[RAW]})
    with open_project_zip(path) as archive:
        assert archive.nfit_revision.generation == 2
        assert {name: archive.getinfo(name).header_offset for name in offsets} == offsets
    assert 0 < path.stat().st_size - before < 4096
    assert read_project_manifest(path)["name"] == "after"
    assert read_project_artifact(path, MEMBER) == payload


def test_incremental_mutations_portable_save_as_and_compaction(tmp_path):
    path = tmp_path / "project.nfit"
    write_project_manifest(path, {"name": "sample"}, storage_mode="incremental",
                           binning_artifacts={MEMBER: b"old" * 10000})
    replace_analysis_artifacts(path, "a", {"result.csv": b"x,y\n1,2\n"})
    replace_dataset_artifact(path, "d", b"dataset")
    write_project_manifest(path, {"name": "changed"}, binning_artifacts={MEMBER: b"new"})
    old_revision = inspect_project_store(path).revision
    before = path.stat().st_size
    with open_project_zip(path) as old:
        compact_project(path)
        # Already-open viewers/readers keep their original committed file.
        assert old.read(MEMBER) == b"new"
    assert path.stat().st_size < before
    assert inspect_project_store(path).revision.file_uuid != old_revision.file_uuid
    assert read_project_artifact(path, "assets/datasets/d/data.npz") == b"dataset"
    assert read_project_artifact(path, "assets/analyses/a/result.csv") == b"x,y\n1,2\n"
    legacy = tmp_path / "portable.nfit"
    write_project_manifest(legacy, {"name": "changed"}, asset_source=path)
    assert project_storage_info(legacy)["format"] == "legacy"
    assert read_project_artifact(legacy, MEMBER) == b"new"
    copied = tmp_path / "copy.nfit"
    shutil.copyfile(path, copied)
    assert read_project_manifest(copied)["name"] == "changed"
    assert read_project_artifact(copied, MEMBER) == b"new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["copy.nfit", "portable.nfit", "project.nfit"]


def test_legacy_migration_requires_new_destination_and_streams_identical_bytes(tmp_path):
    source = tmp_path / "legacy.nfit"
    target = tmp_path / "incremental.nfit"
    write_project_manifest(source, {"a": 1}, binning_artifacts={MEMBER: b"unchanged"})
    original = source.read_bytes()
    with pytest.raises(ValueError, match="legacy"):
        write_project_manifest(source, {"a": 2}, storage_mode="incremental")
    assert source.read_bytes() == original
    write_project_manifest(target, {"a": 1}, asset_source=source, storage_mode="incremental")
    assert read_project_artifact(target, MEMBER) == b"unchanged"
    assert source.read_bytes() == original


def test_migration_preserves_original_and_renamed_alias_from_same_source(tmp_path):
    source = tmp_path / "source.nfit"
    target = tmp_path / "target.nfit"
    write_project_manifest(source, {}, reduced_event_artifacts={RAW: b"original"})
    alias = "assets/binnings/alias/data.npz"
    write_project_manifest(target, {}, asset_source=source, storage_mode="incremental",
                           binning_artifacts={alias: ArchiveMember(source, RAW)})
    assert read_project_artifact(target, RAW) == b"original"
    assert read_project_artifact(target, alias) == b"original"


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_deferred_source_replacement_is_rejected(tmp_path, mode):
    source = tmp_path / "source.nfit"
    target = tmp_path / "target.nfit"
    write_project_manifest(source, {}, storage_mode=mode, binning_artifacts={MEMBER: b"original"})
    ref = ArchiveMember(source, MEMBER)
    write_project_manifest(source, {}, binning_artifacts={MEMBER: b"replacement"})
    with pytest.raises(ValueError, match="Source project changed"):
        write_project_manifest(target, {}, storage_mode=mode, binning_artifacts={MEMBER: ref})
    assert not target.exists()


def test_expected_revision_conflict_preserves_newer_save(tmp_path):
    path = tmp_path / "project.nfit"
    write_project_manifest(path, {"a": 1}, storage_mode="incremental")
    revision = inspect_project_store(path).revision
    write_project_manifest(path, {"a": 2})
    with pytest.raises(StoreConflictError):
        write_project_manifest(path, {"a": 3}, expected_revision=revision)
    assert read_project_manifest(path) == {"a": 2}


def test_parallel_nested_npz_reads_pin_the_committed_descriptor(tmp_path):
    path = tmp_path / "project.nfit"
    npz = tmp_path / "arrays.npz"
    values = np.arange(10000, dtype=np.float64)
    np.savez_compressed(npz, values=values)
    write_project_manifest(path, {}, storage_mode="incremental", binning_artifacts={MEMBER: npz})
    with open_project_artifact(path, MEMBER) as reader:
        factory = _project_artifact_reader_factory(reader)
        assert factory is not None
        compact_project(path)

        def load(_):
            with factory() as independent, np.load(independent) as arrays:
                return arrays["values"]

        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(load, range(8)))
        for result in results:
            np.testing.assert_array_equal(result, values)


def test_public_project_save_load_and_lazy_cross_window_transfer(tmp_path, monkeypatch):
    from nfit import DataGroup, DatasetEntry, NfitProject, load_project, project_gui, save_project
    from tests.project_gui_test_support import _tiny_mdhisto_data

    data = _tiny_mdhisto_data(7.)
    source = DatasetEntry("measured", data, kind="mdhisto")
    group = DataGroup("Source", datasets=[source])
    packet = tmp_path / "selection.nfit"
    project_gui.export_project_items(NfitProject([group]), "dataset", [source], packet, source_group=group)
    destination_group = DataGroup("Destination")
    project = NfitProject([destination_group])
    pasted = project_gui.import_project_items(project, packet, target_role="datasets",
                                              data_group=destination_group).items[0]
    path = tmp_path / "project.nfit"
    save_project(project, path, storage_mode="incremental")
    assert pasted.data is None
    packet.unlink()
    project.name = "renamed"
    save_project(project, path)
    assert inspect_project_store(path).revision.generation == 2
    loaded = load_project(path)
    dataset = loaded.data_groups[0].datasets[0]
    assert dataset.data is None
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(dataset).signal, data.signal)
    exported = tmp_path / "exported.nfit"
    project_gui.export_project_items(loaded, "group", [loaded.data_groups[0]], exported)
    with open_project_zip(exported) as archive:
        assert json.loads(archive.read("project.json"))["project_selection"]["kind"] == "group"


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_unrelated_owned_artifact_write_preserves_cached_references(tmp_path, mode):
    path = tmp_path / "project.nfit"
    write_project_manifest(path, {}, storage_mode=mode, reduced_event_artifacts={RAW: b"events"})
    ref = ArchiveMember(path, RAW)
    replace_analysis_artifacts(path, "a", {"result.csv": b"result"})
    with open_project_artifact(path, RAW, expected_identity=ref.identity) as stream:
        assert stream.read() == b"events"
    write_project_manifest(path, {}, reduced_event_artifacts={RAW: ref})
    assert read_project_artifact(path, RAW) == b"events"


def test_cancelled_append_keeps_references_valid_for_read_and_retry(tmp_path, monkeypatch):
    from nfit import project_store as store

    path = tmp_path / "project.nfit"
    write_project_manifest(path, {}, storage_mode="incremental", reduced_event_artifacts={RAW: b"events"})
    ref = ArchiveMember(path, RAW)

    def interrupt(stage):
        if stage == "after_body_fsync":
            raise OSError("interrupted")

    monkeypatch.setattr(store, "_checkpoint", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        write_project_manifest(path, {"changed": True})
    with open_project_artifact(path, RAW, expected_identity=ref.identity) as stream:
        assert stream.read() == b"events"
    monkeypatch.undo()
    write_project_manifest(path, {"retry": True}, reduced_event_artifacts={RAW: ref})
    assert read_project_artifact(path, RAW) == b"events"


def test_load_rejects_concurrent_commit_without_binding_the_wrong_snapshot(tmp_path, monkeypatch):
    from nfit import NfitProject, load_project, project_gui, save_project

    path = tmp_path / "project.nfit"
    save_project(NfitProject(), path, storage_mode="incremental")
    original = project_gui._bind_project_analysis_sources

    def race(project, filename, **kwargs):
        # Use a different context to simulate a second application process.
        from nfit import project_store as store
        token = store._PINNED_READERS.set(None)
        try:
            write_project_manifest(filename, read_project_manifest(filename))
        finally:
            store._PINNED_READERS.reset(token)
        original(project, filename, **kwargs)

    monkeypatch.setattr(project_gui, "_bind_project_analysis_sources", race)
    with pytest.raises(StoreConflictError, match="while loading"):
        load_project(path)


def test_recovered_load_warns_and_save_as_exports_valid_generation(tmp_path):
    from nfit import NfitProject, load_project, save_project
    from nfit import project_store as store

    path = tmp_path / "project.nfit"
    save_project(NfitProject(), path, storage_mode="incremental")
    save_project(load_project(path), path)
    with path.open("r+b") as stream:
        stream.seek(2 * store._PAGE)
        stream.write(b"broken")
    with pytest.warns(RuntimeWarning, match="older project generation"):
        recovered = load_project(path)
    assert recovered._storage_recovered
    target = tmp_path / "recovered.nfit"
    save_project(recovered, target, storage_mode="incremental")
    assert not inspect_project_store(target).recovered


def test_load_baseline_precedes_commit_selection(tmp_path, monkeypatch):
    from nfit import NfitProject, load_project, save_project
    from nfit import project_store as store

    path = tmp_path / "project.nfit"
    save_project(NfitProject(), path, storage_mode="incremental")
    original = store._inspect
    fired = False

    def race(stream):
        nonlocal fired
        result = original(stream)
        if not fired:
            fired = True
            write_project_manifest(path, read_project_manifest(path))
        return result

    monkeypatch.setattr(store, "_inspect", race)
    with pytest.raises(StoreConflictError, match="while loading"):
        load_project(path)


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_selection_load_rejects_concurrent_save(tmp_path, monkeypatch, mode):
    from nfit import DataGroup, NfitProject, project_gui, project_transfer
    from nfit import project_store as store

    path = tmp_path / "selection.nfit"
    group = DataGroup("workspace")
    project_gui.export_project_items(NfitProject([group]), "group", [group], path)
    if mode == "incremental":
        converted = tmp_path / "incremental.nfit"
        write_project_manifest(converted, read_project_manifest(path), asset_source=path,
                               storage_mode=mode)
        path = converted
    original = project_transfer.bind_project_reduced_event_caches

    def race(project, filename):
        token = store._PINNED_READERS.set(None)
        try:
            write_project_manifest(filename, read_project_manifest(filename))
        finally:
            store._PINNED_READERS.reset(token)
        original(project, filename)

    monkeypatch.setattr(project_transfer, "bind_project_reduced_event_caches", race)
    with pytest.raises(StoreConflictError, match="selection changed while loading"):
        project_transfer.read_project_selection(path)


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
@pytest.mark.parametrize("transferred", [False, True])
def test_canonical_lazy_load_rejects_replaced_artifact(tmp_path, mode, transferred):
    from nfit import (
        DataGroup,
        DatasetEntry,
        NfitProject,
        load_project,
        project_gui,
        project_transfer,
    )
    from nfit.analysis.artifacts import write_dataset_artifact
    from tests.project_gui_test_support import _tiny_mdhisto_data

    packet, path = tmp_path / "packet.nfit", tmp_path / "project.nfit"
    group = DataGroup("source", datasets=[DatasetEntry("measurement", _tiny_mdhisto_data(3.), kind="mdhisto")])
    project_gui.export_project_items(NfitProject([group]), "group", [group], packet)
    write_project_manifest(path, read_project_manifest(packet), asset_source=packet, storage_mode=mode)
    project = project_transfer.read_project_selection(path).project if transferred else load_project(path)
    dataset = project.data_groups[0].datasets[0]
    assert dataset.data is None
    artifact = tmp_path / "replacement.npz"
    write_dataset_artifact(_tiny_mdhisto_data(99.), artifact)
    replace_dataset_artifact(path, dataset.id, artifact)
    with pytest.raises(ValueError, match="changed"):
        project_gui.dataset_for_slice_viewer(dataset)
    assert dataset.data is None


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_canonical_lazy_read_survives_unrelated_analysis_save(tmp_path, mode):
    from nfit import DataGroup, DatasetEntry, NfitProject, load_project, project_gui
    from tests.project_gui_test_support import _tiny_mdhisto_data

    packet, path = tmp_path / "packet.nfit", tmp_path / "project.nfit"
    group = DataGroup("source", datasets=[DatasetEntry("measurement", _tiny_mdhisto_data(3.), kind="mdhisto")])
    project_gui.export_project_items(NfitProject([group]), "group", [group], packet)
    write_project_manifest(path, read_project_manifest(packet), asset_source=packet, storage_mode=mode)
    dataset = load_project(path).data_groups[0].datasets[0]
    replace_analysis_artifacts(path, "other", {"result.csv": b"x,y\n1,2\n"})
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(dataset).signal, [[3.]])


@pytest.mark.parametrize("cache_binnings", [False, True])
def test_save_rejects_newer_post_commit_generation_before_any_adoption(tmp_path, monkeypatch, cache_binnings):
    from nfit import NfitProject, project_gui, save_project

    path = tmp_path / "project.nfit"
    original = project_gui.write_project_manifest

    def competing_save(*args, **kwargs):
        written = original(*args, **kwargs)
        assert written is not None
        payload = read_project_manifest(path)
        payload["competitor"] = True
        write_project_manifest(path, payload)
        return written

    monkeypatch.setattr(project_gui, "write_project_manifest", competing_save)
    monkeypatch.setattr(project_gui, "_bind_project_analysis_sources", lambda *_args, **_kwargs: pytest.fail("Adopted newer artifacts"))
    monkeypatch.setattr(project_gui, "_adopt_saved_project_binning_backing", lambda *_args, **_kwargs: pytest.fail("Adopted newer histograms"))
    with pytest.raises(StoreConflictError, match="was saved, but changed afterward"):
        save_project(NfitProject(settings={"cache_binnings": cache_binnings}), path, storage_mode="incremental")
    assert read_project_manifest(path)["competitor"]


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
@pytest.mark.parametrize("export_selection", [False, True])
def test_canonical_save_and_copy_reject_replaced_source(tmp_path, mode, export_selection):
    from nfit import DataGroup, DatasetEntry, NfitProject, load_project, project_gui, save_project
    from nfit.analysis.artifacts import write_dataset_artifact
    from tests.project_gui_test_support import _tiny_mdhisto_data

    packet, path = tmp_path / "packet.nfit", tmp_path / "project.nfit"
    group = DataGroup("source", datasets=[DatasetEntry("measurement", _tiny_mdhisto_data(3.), kind="mdhisto")])
    project_gui.export_project_items(NfitProject([group]), "group", [group], packet)
    write_project_manifest(path, read_project_manifest(packet), asset_source=packet, storage_mode=mode)
    project = load_project(path)
    group = project.data_groups[0]
    # A resident source-backed cube must not be associated with a new payload.
    np.testing.assert_array_equal(project_gui.dataset_for_slice_viewer(group.datasets[0]).signal, [[3.]])
    artifact = tmp_path / "replacement.npz"
    write_dataset_artifact(_tiny_mdhisto_data(99.), artifact)
    replace_dataset_artifact(path, group.datasets[0].id, artifact)
    target = tmp_path / "destination.nfit"
    with pytest.raises(ValueError, match="Source project changed"):
        if export_selection:
            project_gui.export_project_items(project, "group", [group], target)
        else:
            save_project(project, target, storage_mode=mode)
    assert not target.exists()
