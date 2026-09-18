from __future__ import annotations

import gc
import io
import threading
from dataclasses import make_dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from nfit.analysis.artifacts import (
    _payload,
    dataset_artifact_from_payload,
    read_dataset_artifact,
)
from nfit.mapped_archive import (
    MappedArchiveError,
    MappedWorkspaceError,
    array_storage_nbytes,
    is_mapped_array,
    read_mapped_array_archive,
)
from tests.project_gui_test_support import _tiny_mdhisto_data


def _archive_bytes(payload: dict[str, object]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def test_large_members_are_readonly_mapped_and_small_members_stay_in_heap(tmp_path):
    large = np.arange(4096, dtype=np.float64)
    payload = read_mapped_array_archive(
        _archive_bytes({"large": large, "label": np.asarray("signal")}),
        mapped_min_bytes=1024,
        temp_dir=tmp_path,
    )

    np.testing.assert_equal(payload["large"], large)
    assert is_mapped_array(payload["large"])
    assert not payload["large"].flags.writeable
    assert not is_mapped_array(payload["label"])
    assert not payload["label"].flags.writeable
    assert list(tmp_path.iterdir()) == []


def test_decoded_array_view_survives_payload_eviction(tmp_path):
    source = _tiny_mdhisto_data(2.0)

    payload = read_mapped_array_archive(
        _archive_bytes(_payload(source)),
        mapped_min_bytes=1,
        temp_dir=tmp_path,
    )
    decoded = dataset_artifact_from_payload(payload)
    signal = decoded.signal
    assert is_mapped_array(signal)

    del decoded, payload
    gc.collect()

    assert is_mapped_array(signal)
    np.testing.assert_equal(signal, source.signal)
    assert list(tmp_path.iterdir()) == []


def test_unsupported_platform_falls_back_to_resident_artifact_read(monkeypatch):
    import nfit.analysis.artifacts as artifacts

    source = _tiny_mdhisto_data(2.0)
    archive = _archive_bytes(_payload(source))
    monkeypatch.setattr(
        "nfit.mapped_archive._supports_unlinked_mappings", lambda: False
    )
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 1)

    decoded = read_dataset_artifact(archive, memory_map=True)

    np.testing.assert_equal(decoded.signal, source.signal)
    assert not is_mapped_array(decoded.signal)


def test_storage_accounting_separates_heap_and_mapped_arrays(tmp_path):
    heap = np.arange(5, dtype=np.int16)
    payload = read_mapped_array_archive(
        _archive_bytes({"mapped": np.arange(100, dtype=np.float64)}),
        mapped_min_bytes=1,
        temp_dir=tmp_path,
    )
    mapped = payload["mapped"]
    storage = array_storage_nbytes(
        {"heap": heap, "mapped": mapped, "view": mapped[10:]}
    )

    assert storage.heap == heap.nbytes
    assert storage.mapped == mapped.nbytes
    assert storage.total == heap.nbytes + mapped.nbytes

    sliced_storage = array_storage_nbytes(mapped[10:20])
    assert sliced_storage.mapped == mapped.nbytes


def test_storage_accounting_preserves_cache_payload_special_cases():
    class FakeCudaArray:
        __cuda_array_interface__ = {"shape": (4,), "typestr": "<f8"}
        nbytes = 32

    source = np.arange(8, dtype=np.uint8)
    view = memoryview(source)
    FitDataBundle = make_dataclass("FitDataBundle", ["dataset", "retained"])
    bundle = FitDataBundle(
        dataset=np.arange(100, dtype=np.float64),
        retained=np.arange(3, dtype=np.float64),
    )

    storage = array_storage_nbytes((FakeCudaArray(), view, bundle))

    assert storage.mapped == 0
    assert storage.heap == 32 + view.nbytes + bundle.retained.nbytes


def test_disk_space_is_checked_before_any_member_is_inflated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nfit.mapped_archive.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )
    with pytest.raises(OSError, match="temporary bytes"):
        read_mapped_array_archive(
            _archive_bytes({"large": np.arange(1000, dtype=np.float64)}),
            mapped_min_bytes=1,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_disk_space_check_retains_temporary_filesystem_headroom(tmp_path, monkeypatch):
    gibibyte = 1024**3
    monkeypatch.setattr(
        "nfit.mapped_archive.shutil.disk_usage",
        lambda _path: SimpleNamespace(
            total=2 * gibibyte, used=gibibyte - 100, free=gibibyte + 100
        ),
    )

    with pytest.raises(MappedWorkspaceError, match="headroom"):
        read_mapped_array_archive(
            _archive_bytes({"large": np.arange(100, dtype=np.uint8)}),
            mapped_min_bytes=1,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_memory_backed_temporary_mount_is_rejected(monkeypatch):
    import nfit.mapped_archive as mapped_archive

    root = "/tmp/nfit space"
    monkeypatch.setattr(
        mapped_archive,
        "_mountinfo_text",
        lambda: "\n".join((
            "20 1 8:1 / / rw - ext4 /dev/root rw",
            "21 20 0:5 / /tmp/nfit\\040space rw - tmpfs tmpfs rw",
        )),
    )
    monkeypatch.setattr(
        mapped_archive, "_resolved_temp_path", lambda path: path
    )

    with pytest.raises(MappedWorkspaceError, match="memory-backed tmpfs"):
        mapped_archive._require_disk_backed_temp(mapped_archive.Path(root))


def test_mountinfo_uses_longest_disk_backed_mount_and_tolerates_absence(monkeypatch):
    import nfit.mapped_archive as mapped_archive

    root = mapped_archive.Path("/work/cache/nfit")
    monkeypatch.setattr(
        mapped_archive,
        "_mountinfo_text",
        lambda: "\n".join((
            "20 1 0:5 / / rw - tmpfs tmpfs rw",
            "21 20 8:1 / /work/cache rw - ext4 /dev/disk rw",
        )),
    )
    mapped_archive._require_disk_backed_temp(root)

    monkeypatch.setattr(mapped_archive, "_mountinfo_text", lambda: None)
    mapped_archive._require_disk_backed_temp(root)


def test_later_member_failure_closes_prior_unlinked_mappings(tmp_path, monkeypatch):
    import nfit.mapped_archive as mapped_archive

    original = mapped_archive._inflate_mapped_member
    created = []

    def fail_second(archive, info, *, temp_dir):
        if created:
            raise MappedArchiveError("second member failed")
        array = original(archive, info, temp_dir=temp_dir)
        created.append(mapped_archive.mapped_array_backing(array))
        return array

    monkeypatch.setattr(mapped_archive, "_inflate_mapped_member", fail_second)
    with pytest.raises(MappedArchiveError, match="second member"):
        read_mapped_array_archive(
            _archive_bytes({"first": np.arange(100), "second": np.arange(100)}),
            mapped_min_bytes=1,
            temp_dir=tmp_path,
        )

    assert len(created) == 1
    assert created[0].closed
    assert list(tmp_path.iterdir()) == []


def test_parallel_inflation_obeys_central_worker_count(tmp_path, monkeypatch):
    import nfit.mapped_archive as mapped_archive

    original = mapped_archive._inflate_mapped_member
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    peak = 0
    policy_calls = []

    def worker_count(total_bytes, **options):
        policy_calls.append((total_bytes, options))
        return 2

    def measured_inflate(archive, info, *, temp_dir):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        try:
            return original(archive, info, temp_dir=temp_dir)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(mapped_archive, "operation_worker_count", worker_count)
    monkeypatch.setattr(mapped_archive, "_inflate_mapped_member", measured_inflate)
    payload = read_mapped_array_archive(
        _archive_bytes({"first": np.arange(100), "second": np.arange(200)}),
        mapped_min_bytes=1,
        temp_dir=tmp_path,
    )

    assert peak == 2
    assert len(policy_calls) == 1
    assert policy_calls[0][1] == {
        "bytes_per_worker": 3 * mapped_archive._COPY_CHUNK_BYTES,
        "min_parallel_bytes": 32 * 1024**2,
    }
    assert list(payload) == ["first", "second"]


def test_parallel_failure_closes_mappings_completed_after_first_error(
    tmp_path, monkeypatch
):
    import nfit.mapped_archive as mapped_archive

    original = mapped_archive._inflate_mapped_member
    second_complete = threading.Event()
    created = []

    def fail_with_concurrent_success(archive, info, *, temp_dir):
        if info.filename == "first.npy":
            assert second_complete.wait(timeout=5)
            raise MappedArchiveError("parallel member failed")
        array = original(archive, info, temp_dir=temp_dir)
        created.append(mapped_archive.mapped_array_backing(array))
        second_complete.set()
        return array

    monkeypatch.setattr(mapped_archive, "operation_worker_count", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        mapped_archive, "_inflate_mapped_member", fail_with_concurrent_success
    )

    with pytest.raises(MappedArchiveError, match="parallel member failed"):
        read_mapped_array_archive(
            _archive_bytes({"first": np.arange(100), "second": np.arange(200)}),
            mapped_min_bytes=1,
            temp_dir=tmp_path,
        )

    assert len(created) == 1
    assert created[0].closed
    assert list(tmp_path.iterdir()) == []


def test_object_array_and_invalid_member_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="objects|Object"):
        read_mapped_array_archive(
            _archive_bytes({"object": np.asarray([{}], dtype=object)}),
            mapped_min_bytes=1,
            temp_dir=tmp_path,
        )
    stream = io.BytesIO()
    np.save(stream, np.arange(3))
    import zipfile

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("nested/value.npy", stream.getvalue())
    with pytest.raises(MappedArchiveError, match="invalid member"):
        read_mapped_array_archive(
            archive.getvalue(), mapped_min_bytes=1, temp_dir=tmp_path
        )


def test_source_archive_is_never_modified(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(_archive_bytes({"large": np.arange(1000)}))
    before = source.read_bytes()
    maps = tmp_path / "maps"
    maps.mkdir()
    read_mapped_array_archive(source, mapped_min_bytes=1, temp_dir=maps)
    assert source.read_bytes() == before
