"""Immutable saved histograms survive changes to unrelated project bytes."""

from contextlib import contextmanager

import numpy as np
import pytest

from nfit import project_store
from nfit.analysis.artifacts import write_dataset_artifact
from nfit.project_archive import ArchiveMember, replace_analysis_artifacts, write_project_manifest
from nfit.rebin_cache import RebinCache, RebinCacheBudget, _DiskBinning, _file_identity
from tests.project_gui_test_support import _tiny_mdhisto_data

MEMBER = "assets/binnings/cache/data.npz"


def _saved_cache(tmp_path, mode="incremental", *, payload=b"opaque histogram"):
    path = tmp_path / "project.nfit"
    write_project_manifest(path, {}, storage_mode=mode, binning_artifacts={MEMBER: payload})
    cache = RebinCache(budget=RebinCacheBudget())
    cache.set_project_backing("cache", signature="same-science", project_path=path, member=MEMBER, lazy=True)
    return path, cache


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_unrelated_asset_write_retains_lazy_histogram_without_decoding(tmp_path, monkeypatch, mode):
    path, cache = _saved_cache(tmp_path, mode)
    monkeypatch.setattr(_DiskBinning, "restore", lambda *_: pytest.fail("Decoded unchanged histogram"))
    replace_analysis_artifacts(path, "analysis", {"result.csv": b"result"})
    assert cache.has_signature("cache", "same-science")
    assert cache.archive_backing("cache", "same-science") == (path, MEMBER)
    assert cache._disk["cache"].project_identity == _file_identity(path)


def test_cancelled_append_retains_lazy_histogram_for_save_retry(tmp_path, monkeypatch):
    path, cache = _saved_cache(tmp_path)

    def cancel(stage):
        if stage == "after_body_fsync":
            raise OSError("cancelled")

    monkeypatch.setattr(project_store, "_checkpoint", cancel)
    with pytest.raises(OSError, match="cancelled"):
        write_project_manifest(path, {"unpublished": True})
    monkeypatch.setattr(_DiskBinning, "restore", lambda *_: pytest.fail("Decoded unchanged histogram"))
    assert cache.has_signature("cache", "same-science")
    assert cache.archive_backing("cache", "same-science") == (path, MEMBER)
    monkeypatch.setattr(project_store, "_checkpoint", lambda _: None)
    write_project_manifest(path, {"retry": True})
    assert cache.archive_backing("cache", "same-science") == (path, MEMBER)


def test_available_rechecks_index_only_when_physical_file_changes(tmp_path, monkeypatch):
    path, cache = _saved_cache(tmp_path)
    original = project_store.open_project_zip
    opened = []

    @contextmanager
    def counted(filename):
        opened.append(filename)
        with original(filename) as archive:
            yield archive

    monkeypatch.setattr(project_store, "open_project_zip", counted)
    for _ in range(5):
        assert cache.archive_backing("cache", "same-science") == (path, MEMBER)
    assert opened == []
    replace_analysis_artifacts(path, "analysis", {"result.csv": b"result"})
    opened.clear()
    assert cache.archive_backing("cache", "same-science") == (path, MEMBER)
    assert len(opened) == 1
    for _ in range(5):
        assert cache.archive_backing("cache", "same-science") == (path, MEMBER)
    assert len(opened) == 1


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_replacing_histogram_member_invalidates_saved_backing(tmp_path, mode):
    path, cache = _saved_cache(tmp_path, mode)
    write_project_manifest(path, {}, binning_artifacts={MEMBER: b"different science"})
    assert cache.archive_backing("cache", "same-science") is None
    assert not cache.has_signature("cache", "same-science")


def test_standalone_session_file_keeps_physical_identity_guard(tmp_path):
    path = tmp_path / "session.npz"
    path.write_bytes(b"session data")
    backing = _DiskBinning("signature", path, project_identity=_file_identity(path))
    assert backing.available()
    path.write_bytes(b"changed session payload")
    assert not backing.available()


def test_pinned_load_cannot_bless_a_later_generation(tmp_path):
    path, _ = _saved_cache(tmp_path)
    with project_store.project_read_snapshot(path) as original:
        token = project_store._PINNED_READERS.set(None)
        try:
            write_project_manifest(path, {}, binning_artifacts={MEMBER: b"new science"})
        finally:
            project_store._PINNED_READERS.reset(token)
        cache = RebinCache(budget=RebinCacheBudget())
        cache.set_project_backing("cache", signature="old-science", project_path=path, member=MEMBER, lazy=True)
        assert cache._disk["cache"].project_identity == original.nfit_file_identity
        assert cache.archive_backing("cache", "old-science") == (path, MEMBER)
    assert cache.archive_backing("cache", "old-science") is None


def test_decoding_uses_the_snapshot_whose_member_was_verified(tmp_path, monkeypatch):
    import nfit.rebin_cache as rebin_cache

    artifact = tmp_path / "histogram.npz"
    expected = _tiny_mdhisto_data(3.0)
    write_dataset_artifact(expected, artifact)
    path, cache = _saved_cache(tmp_path, payload=artifact)
    read = rebin_cache.read_project_dataset_artifact

    def concurrent_replace(filename, member, **kwargs):
        token = project_store._PINNED_READERS.set(None)
        try:
            write_project_manifest(path, {}, binning_artifacts={MEMBER: b"not the histogram"})
        finally:
            project_store._PINNED_READERS.reset(token)
        return read(filename, member, **kwargs)

    monkeypatch.setattr(rebin_cache, "read_project_dataset_artifact", concurrent_replace)
    _, restored = cache.get("cache")
    np.testing.assert_array_equal(restored.signal, expected.signal)
    assert not restored.signal.flags.writeable
    assert not restored.errors.flags.writeable


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_archive_reference_keeps_registered_identity_if_member_changes_after_check(tmp_path, monkeypatch, mode):
    path, cache = _saved_cache(tmp_path, mode)
    backing = cache._project_backings["cache"]
    expected_identity = backing.project_member_identity
    available = _DiskBinning.available
    changed = False

    def replace_after_check(record):
        nonlocal changed
        result = available(record)
        if record is backing and result and not changed:
            changed = True
            write_project_manifest(path, {}, binning_artifacts={MEMBER: b"different science"})
        return result

    monkeypatch.setattr(_DiskBinning, "available", replace_after_check)
    reference = cache.archive_reference("cache", "same-science")
    assert isinstance(reference, ArchiveMember)
    assert reference.identity == expected_identity
    assert reference.identity != ArchiveMember(path, MEMBER).identity
    with pytest.raises(ValueError, match="Source project changed"):
        write_project_manifest(tmp_path / "copy.nfit", {}, binning_artifacts={MEMBER: reference})


def test_archive_reference_preserves_standalone_file_guard(tmp_path):
    path = tmp_path / "session.npz"
    path.write_bytes(b"session payload")
    cache = RebinCache(budget=RebinCacheBudget())
    cache._disk["cache"] = _DiskBinning("signature", path, project_identity=_file_identity(path))
    assert cache.archive_reference("cache", "signature") == path
    assert cache.archive_reference("cache", "other signature") is None
    path.write_bytes(b"changed session payload")
    assert cache.archive_reference("cache", "signature") is None
