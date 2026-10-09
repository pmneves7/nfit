"""Publication, recovery and snapshot isolation for incremental projects."""

import os
import subprocess
import sys
import zipfile

import pytest

from nfit import project_store as store
from nfit.operation_control import operation_progress


def _create(path, value=b"first"):
    return store.write_incremental(path, lambda destination, _: destination.writestr("project.json", value))


def _replace(path, value=b"second", **kwargs):
    return store.write_incremental(path, lambda destination, _: destination.writestr("project.json", value), **kwargs)


def _read(path):
    with store.open_project_zip(path) as archive:
        return archive.read("project.json")


def _copy(destination, current):
    for info in current.infolist():
        with current.open(info) as source, destination.open(info.filename, "w", force_zip64=True) as target:
            while block := source.read(4096):
                target.write(block)


def test_legacy_reader_and_explicit_migration(tmp_path):
    path = tmp_path / "legacy.nfit"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", b"legacy")
    assert store.inspect_project_store(path).format == "legacy"
    with store.open_project_zip(path) as archive:
        assert archive.nfit_revision is None
        assert archive.nfit_file_identity[2] == path.stat().st_size
        assert archive.read("project.json") == b"legacy"
    with pytest.raises(ValueError, match="Migrate"):
        _replace(path)
    assert _read(path) == b"legacy"


def test_append_reuses_member_offsets_and_pins_old_reader(tmp_path):
    path = tmp_path / "project.nfit"

    def initial(destination, current):
        assert current is None
        destination.writestr("project.json", b"first")
        destination.writestr("assets/μ-data.bin", bytes(range(256)) * 200, compress_type=zipfile.ZIP_DEFLATED)

    first = store.write_incremental(path, initial)
    with store.open_project_zip(path) as old:
        before = old.getinfo("assets/μ-data.bin")
        identity = old.nfit_file_identity

        def change(destination, current):
            assert current.nfit_revision == first
            assert current.nfit_file_identity == identity
            store.reuse_project_members(destination, current, ["assets/μ-data.bin"])
            destination.writestr("project.json", b"second")

        second = store.write_incremental(path, change, expected_revision=first)
        assert second.generation == 2
        assert second.file_uuid == first.file_uuid
        assert old.read("project.json") == b"first"
        assert old.fp.fileno() >= 0
        with store.open_project_zip(path) as new:
            after = new.getinfo("assets/μ-data.bin")
            assert (after.header_offset, after.compress_size, after.CRC) == (
                before.header_offset, before.compress_size, before.CRC)
            assert new.read("assets/μ-data.bin") == bytes(range(256)) * 200
            assert new.read("project.json") == b"second"
    assert store.inspect_project_store(path).revision == second


def test_reuse_requires_same_transaction_and_rejects_duplicates(tmp_path):
    path = tmp_path / "project.nfit"
    _create(path)
    with store.open_project_zip(path) as unrelated:
        with pytest.raises(ValueError, match="transaction"):
            store.write_incremental(path, lambda destination, _: store.reuse_project_members(destination, unrelated))

    def duplicate(destination, current):
        store.reuse_project_members(destination, current)
        store.reuse_project_members(destination, current)

    with pytest.raises(ValueError, match="Duplicate"):
        store.write_incremental(path, duplicate)
    assert _read(path) == b"first"


def test_expected_revision_conflict(tmp_path):
    path = tmp_path / "project.nfit"
    first = _create(path)
    second = _replace(path)
    with pytest.raises(store.StoreConflictError):
        _replace(path, b"stale", expected_revision=first)
    assert store.inspect_project_store(path).revision == second
    assert _read(path) == b"second"
    with pytest.raises(store.StoreConflictError):
        _replace(tmp_path / "missing.nfit", expected_revision=first)


def test_operation_cancellation_after_body_fsync_prevents_publication(tmp_path):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def cancel(progress):
        if progress["message"] == "Publishing the saved project…":
            raise InterruptedError("user cancelled")

    with operation_progress(cancel), pytest.raises(InterruptedError, match="cancelled"):
        _replace(path)
    assert store.inspect_project_store(path).revision == first
    assert _read(path) == b"first"


def test_compaction_checks_cancellation_before_replacing_original(tmp_path):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def cancel(progress):
        if progress["message"] == "Publishing the compacted project…":
            raise InterruptedError("user cancelled")

    with operation_progress(cancel), pytest.raises(InterruptedError, match="cancelled"):
        store.compact_incremental(path, _copy)
    assert store.inspect_project_store(path).revision == first


@pytest.mark.parametrize("stage", ["after_body_flush", "after_body_fsync", "before_slot"])
def test_precommit_failure_keeps_prior_generation_and_discards_tail_on_retry(tmp_path, monkeypatch, stage):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def fail(point):
        if point == stage:
            raise KeyboardInterrupt("cancelled")

    monkeypatch.setattr(store, "_checkpoint", fail)
    with pytest.raises(KeyboardInterrupt):
        _replace(path, b"uncommitted" * 10000)
    assert path.stat().st_size > first.committed_size
    assert store.inspect_project_store(path).revision == first
    assert _read(path) == b"first"
    monkeypatch.setattr(store, "_checkpoint", lambda _: None)
    second = _replace(path, b"retry")
    assert second.committed_size < first.committed_size + 10000
    assert _read(path) == b"retry"


@pytest.mark.parametrize("stage", ["after_slot_write", "after_slot_fsync"])
def test_publication_failure_never_truncates_observable_generation(tmp_path, monkeypatch, stage):
    path = tmp_path / "project.nfit"
    _create(path)

    def fail(point):
        if point == stage:
            raise OSError("simulated fsync failure")

    monkeypatch.setattr(store, "_checkpoint", fail)
    with pytest.raises(store.StoreCommitUncertainError):
        _replace(path)
    assert store.inspect_project_store(path).revision.generation == 2
    assert _read(path) == b"second"


def test_torn_slot_falls_back_read_only(tmp_path, monkeypatch):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def torn(stream, offset, page):
        stream.seek(offset)
        stream.write(page[:100])
        raise OSError("short device write")

    monkeypatch.setattr(store, "_write_slot", torn)
    with pytest.raises(store.StoreCommitUncertainError):
        _replace(path)
    info = store.inspect_project_store(path)
    assert info.recovered and info.revision == first
    with store.open_project_zip(path) as archive:
        assert archive.nfit_recovered
        assert archive.read("project.json") == b"first"
    with pytest.raises(store.StoreRecoveryError, match="read-only"):
        _replace(path)


@pytest.mark.parametrize("offset", [0, 12, 50, 4095])
def test_header_corruption_fails_closed(tmp_path, offset):
    path = tmp_path / "project.nfit"
    _create(path)
    with path.open("r+b") as stream:
        stream.seek(offset)
        byte = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(store.StoreRecoveryError):
        _read(path)


def test_corrupt_eocd_is_covered_by_commit_hash(tmp_path):
    path = tmp_path / "project.nfit"
    first = _create(path)
    second = _replace(path)
    with path.open("r+b") as stream:
        stream.seek(second.committed_size - 22)
        stream.write(b"BAD!")
    info = store.inspect_project_store(path)
    assert info.recovered and info.revision == first
    assert _read(path) == b"first"


def test_both_invalid_slots_fail_closed(tmp_path):
    path = tmp_path / "project.nfit"
    _create(path)
    _replace(path)
    with path.open("r+b") as stream:
        for offset in (store._PAGE, 2 * store._PAGE):
            stream.seek(offset)
            stream.write(b"BAD!")
    with pytest.raises(store.StoreRecoveryError, match="No valid"):
        _read(path)


def test_valid_slot_checksum_with_forged_directory_offset_is_rejected(tmp_path):
    path = tmp_path / "project.nfit"
    _create(path)
    with path.open("r+b") as stream:
        page = store._read_at(stream, store._PAGE, store._PAGE)
        marker, identity, generation, end, offset, size = store._SLOT.unpack(page[:store._SLOT.size])
        # Include one byte of preceding data in the trailer; the SHA is valid,
        # but ZIP's actual directory location disagrees with the slot.
        offset -= 1
        size += 1
        digest = store._digest_range(stream, offset, size)
        forged = store._SLOT.pack(marker, identity, generation, end, offset, size) + digest
        stream.seek(store._PAGE)
        stream.write(store._page(forged))
    with pytest.raises(store.StoreRecoveryError, match="No valid"):
        _read(path)


def test_large_uncommitted_tail_is_ignored(tmp_path):
    path = tmp_path / "project.nfit"
    first = _create(path)
    with path.open("ab") as stream:
        stream.write(bytes(200000))
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("project.json", b"uncommitted ZIP")
    assert store.inspect_project_store(path).revision == first
    assert _read(path) == b"first"


def test_compaction_preserves_old_reader_and_changes_identity(tmp_path):
    path = tmp_path / "project.nfit"
    _create(path, b"obsolete" * 10000)
    previous = _replace(path)
    old_size = path.stat().st_size
    os.chmod(path, 0o640)
    with store.open_project_zip(path) as old:
        fresh = store.compact_incremental(path, _copy, expected_revision=previous)
        assert old.read("project.json") == b"second"
        assert fresh.file_uuid != previous.file_uuid
        assert fresh.generation == 1
        assert path.stat().st_size < old_size
        assert path.stat().st_mode & 0o777 == 0o640
        assert _read(path) == b"second"
    with pytest.raises(store.StoreConflictError):
        _replace(path, expected_revision=previous)


def test_compaction_failure_before_replace_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def fail(point):
        if point == "before_replace":
            raise OSError("replace unavailable")

    monkeypatch.setattr(store, "_checkpoint", fail)
    with pytest.raises(OSError, match="replace unavailable"):
        store.compact_incremental(path, _copy)
    assert store.inspect_project_store(path).revision == first
    assert _read(path) == b"first"


def test_compaction_post_replace_failure_reports_uncertainty(tmp_path, monkeypatch):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def fail(point):
        if point == "after_replace":
            raise OSError("directory fsync failed")

    monkeypatch.setattr(store, "_checkpoint", fail)
    with pytest.raises(store.StoreCommitUncertainError):
        store.compact_incremental(path, _copy)
    assert store.inspect_project_store(path).revision.file_uuid != first.file_uuid
    assert _read(path) == b"first"


@pytest.mark.parametrize("stage,published", [("after_slot_write", False), ("before_publish", False), ("after_publish", True)])
def test_creation_failures_observe_publication_boundary(tmp_path, monkeypatch, stage, published):
    path = tmp_path / "project.nfit"

    def fail(point):
        if point == stage:
            raise OSError("injected failure")

    monkeypatch.setattr(store, "_checkpoint", fail)
    with pytest.raises(OSError) as error:
        _create(path)
    assert isinstance(error.value, store.StoreCommitUncertainError) == published
    assert path.exists() == published
    if published:
        assert _read(path) == b"first"


def test_racing_creators_do_not_overwrite_winner(tmp_path):
    path = tmp_path / "project.nfit"

    def race(destination, _):
        _create(path, b"winner")
        destination.writestr("project.json", b"loser")

    with pytest.raises(store.StoreConflictError):
        store.write_incremental(path, race)
    assert _read(path) == b"winner"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["project.nfit"]


def test_path_replacement_during_callback_is_rejected(tmp_path):
    path = tmp_path / "project.nfit"
    replacement = tmp_path / "replacement.nfit"
    _create(path)
    _create(replacement, b"external")

    def replace(destination, current):
        destination.writestr("project.json", b"lost")
        os.replace(replacement, path)

    with pytest.raises(store.StoreConflictError):
        store.write_incremental(path, replace)
    assert _read(path) == b"external"


@pytest.mark.skipif(os.name == "nt", reason="POSIX lock coordination test")
def test_other_process_writer_lock_is_nonblocking(tmp_path):
    path = tmp_path / "project.nfit"
    _create(path)
    script = ("import fcntl,sys; f=open(sys.argv[1],'r+b'); "
              "fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.read()")
    process = subprocess.Popen([sys.executable, "-c", script, str(path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(store.StoreBusyError):
            _replace(path)
        assert _read(path) == b"first"
    finally:
        process.communicate(timeout=10)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX crash injection")
@pytest.mark.parametrize("stage,expected", [("after_body_fsync", b"first"), ("after_slot_fsync", b"second")])
def test_process_exit_at_commit_boundary(tmp_path, stage, expected):
    path = tmp_path / "project.nfit"
    _create(path)
    pid = os.fork()
    if pid == 0:
        store._checkpoint = lambda point: os._exit(71) if point == stage else None
        _replace(path)
        os._exit(72)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 71
    assert _read(path) == expected


def test_forced_zip64_members_and_indices_can_be_reused(tmp_path, monkeypatch):
    path = tmp_path / "project.nfit"
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 64)
    _create(path, b"large" * 100)
    store.write_incremental(path, lambda destination, current: store.reuse_project_members(destination, current))
    assert _read(path) == b"large" * 100


def test_duplicate_written_names_do_not_commit(tmp_path):
    path = tmp_path / "project.nfit"
    first = _create(path)

    def duplicate(destination, _):
        destination.writestr("project.json", b"one")
        destination.writestr("project.json", b"two")

    with pytest.warns(UserWarning, match="Duplicate"), pytest.raises(store.StoreRecoveryError):
        store.write_incremental(path, duplicate)
    assert store.inspect_project_store(path).revision == first
