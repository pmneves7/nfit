from __future__ import annotations

import io
import threading
from zipfile import ZipFile

import numpy as np
import pytest

from nfit import array_archive
from nfit.analysis import artifacts
from tests.project_gui_test_support import _tiny_mdhisto_data


@pytest.mark.parametrize("workers", [1, 3])
def test_archive_is_readable_by_numpy_and_zipfile(monkeypatch, workers):
    monkeypatch.setattr(array_archive, "operation_worker_count", lambda *a, **k: workers)
    monkeypatch.setattr(array_archive, "_CHUNK_BYTES", 1024)
    monkeypatch.setattr(array_archive, "_PARALLEL_MIN_BYTES", 2048)
    payload = {
        "c": np.arange(16_384, dtype=float).reshape(128, 128),
        "fortran": np.asfortranarray(np.arange(4096).reshape(64, 64)),
        "strided": np.arange(8192, dtype="<i4")[::3],
        "mask": np.arange(8192) % 3 == 0,
        "text": np.asarray("ΔE and μB"),
        "empty": np.empty((3, 0)),
        "scalar": np.asarray(np.nan),
    }
    output = io.BytesIO()
    array_archive.write_array_archive(output, payload)
    with ZipFile(io.BytesIO(output.getvalue())) as archive:
        assert archive.testzip() is None
    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as archive:
        assert set(archive.files) == set(payload)
        for key, expected in payload.items():
            np.testing.assert_equal(archive[key], expected)
        assert archive["fortran"].flags.f_contiguous


def test_large_archive_compression_uses_bounded_worker_pool(monkeypatch):
    monkeypatch.setattr(array_archive, "operation_worker_count", lambda *a, **k: 2)
    monkeypatch.setattr(array_archive, "_CHUNK_BYTES", 1024)
    monkeypatch.setattr(array_archive, "_PARALLEL_MIN_BYTES", 2048)
    original = array_archive._deflate_block
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)
    threads = set()
    calls = 0

    def deflate(data):
        nonlocal calls
        with lock:
            calls += 1
            first_pair = calls <= 2
            threads.add(threading.get_ident())
        if first_pair:
            barrier.wait()
        return original(data)

    monkeypatch.setattr(array_archive, "_deflate_block", deflate)
    output = io.BytesIO()
    expected = np.arange(16_384, dtype=float)
    array_archive.write_array_archive(output, {"signal": expected})
    assert len(threads) == 2
    with np.load(io.BytesIO(output.getvalue())) as archive:
        np.testing.assert_equal(archive["signal"], expected)


def test_small_archive_does_not_construct_executor(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("small arrays should use the ordinary NumPy writer")

    monkeypatch.setattr(array_archive, "ThreadPoolExecutor", unexpected)
    output = io.BytesIO()
    array_archive.write_array_archive(output, {"signal": np.arange(8)})
    with np.load(io.BytesIO(output.getvalue())) as archive:
        np.testing.assert_equal(archive["signal"], np.arange(8))


def test_object_arrays_are_rejected():
    with pytest.raises(ValueError, match="object arrays"):
        array_archive.write_array_archive(io.BytesIO(), {"bad": np.asarray([{}])})


def test_failed_artifact_write_preserves_destination_and_removes_temporary(tmp_path, monkeypatch):
    target = tmp_path / "data.npz"
    target.write_bytes(b"original")

    def fail(*args, **kwargs):
        raise OSError("compression failed")

    monkeypatch.setattr(artifacts, "write_array_archive", fail)
    with pytest.raises(OSError, match="compression failed"):
        artifacts.write_dataset_artifact(_tiny_mdhisto_data(2.0), target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_parallel_compressor_failure_preserves_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(array_archive, "operation_worker_count", lambda *a, **k: 2)
    monkeypatch.setattr(array_archive, "_CHUNK_BYTES", 4)
    monkeypatch.setattr(array_archive, "_PARALLEL_MIN_BYTES", 8)

    def fail(_block):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(array_archive, "_deflate_block", fail)
    target = tmp_path / "data.npz"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="worker failed"):
        artifacts.write_dataset_artifact(_tiny_mdhisto_data(2.0), target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_owned_decoder_avoids_second_copy_but_public_decoder_isolates_inputs(monkeypatch):
    source = _tiny_mdhisto_data(2.0)
    payload = artifacts._payload(source)
    payload["signal"] = source.signal.copy()
    decoded = artifacts.dataset_artifact_from_payload(payload)
    assert payload["signal"].flags.writeable
    assert not np.shares_memory(decoded.signal, payload["signal"])

    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    stream.seek(0)
    with np.load(stream) as archive:
        owned = artifacts._OwnedArchiveArrays(archive)
        signal = owned._read("signal")
        owned.loaded["signal"] = signal
        decoded = artifacts.dataset_artifact_from_payload(owned)
        assert decoded.signal is signal
        assert not decoded.signal.flags.writeable
        np.testing.assert_equal(decoded.signal, source.signal)


def test_parallel_artifact_decode_is_lossless(monkeypatch):
    monkeypatch.setattr(artifacts, "operation_worker_count", lambda *a, **k: 3)
    source = _tiny_mdhisto_data(2.0)
    decoded = artifacts.read_dataset_artifact(artifacts.dataset_artifact_bytes(source))
    np.testing.assert_equal(decoded.signal, source.signal)
    np.testing.assert_equal(decoded.errors, source.errors)
    np.testing.assert_equal(decoded.mask, source.mask)
    assert not decoded.signal.flags.writeable


def test_artifact_capacity_only_reads_headers_and_rejects_before_decode(monkeypatch):
    from nfit import resource_budget

    source = _tiny_mdhisto_data(2.0)
    encoded = artifacts.dataset_artifact_bytes(source)
    capacity = artifacts.dataset_artifact_capacity(encoded)
    assert capacity.expanded_bytes == sum(np.asarray(value).nbytes for value in artifacts._payload(source).values())
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: capacity.peak_bytes + 99)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: capacity.peak_bytes * 2)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(artifacts, "_OwnedArchiveArrays", lambda *args: (_ for _ in ()).throw(AssertionError("decoded")))
    with pytest.raises(resource_budget.ResourceLimitError):
        artifacts.read_dataset_artifact(encoded)


def test_cancellable_decode_releases_reservation_without_publishing_result():
    from nfit.operation_control import operation_progress
    from nfit.resource_budget import snapshot_memory

    source = _tiny_mdhisto_data(2.0)
    encoded = artifacts.dataset_artifact_bytes(source)
    before = snapshot_memory().reserved_bytes

    def cancel(progress):
        raise RuntimeError("cancel decode")

    with operation_progress(cancel), pytest.raises(RuntimeError, match="cancel decode"):
        artifacts.read_dataset_artifact(encoded)
    assert snapshot_memory().reserved_bytes == before


def test_bounded_member_decoder_handles_fortran_and_empty_arrays():
    payload = {"fortran": np.asfortranarray(np.arange(24).reshape(4, 6)), "empty": np.empty((0, 4))}
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    stream.seek(0)
    with np.load(stream) as archive:
        for name, expected in payload.items():
            actual = artifacts._decode_array_member(archive, name)
            np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("workers", [1, 2])
def test_compression_cancel_is_bounded_and_keeps_existing_artifact(tmp_path, monkeypatch, workers):
    from nfit.mdhisto import MDHistoAxis, MDHistoData
    from nfit.operation_control import operation_progress
    from nfit.resource_budget import snapshot_memory

    monkeypatch.setattr(array_archive, "_CHUNK_BYTES", 1024)
    monkeypatch.setattr(array_archive, "_PARALLEL_MIN_BYTES", 2048)
    monkeypatch.setattr(array_archive, "operation_worker_count", lambda *args, **kwargs: workers)
    signal = np.arange(4096.)
    data = MDHistoData((MDHistoAxis("E", np.arange(4097.), "meV", "energy"),),
                      signal, np.ones_like(signal), np.zeros(signal.size, bool), np.ones_like(signal))
    target = tmp_path / "original.npz"
    target.write_bytes(b"existing saved artifact")
    before = snapshot_memory().reserved_bytes
    completed = []

    def cancel(event):
        if event["message"] == "Saving array signal":
            completed.append(event["iteration"])
            if event["iteration"] >= 2048:
                raise RuntimeError("cancel compression")

    with operation_progress(cancel), pytest.raises(RuntimeError, match="cancel compression"):
        artifacts.write_dataset_artifact(data, target)
    assert completed[-1] < signal.nbytes
    assert target.read_bytes() == b"existing saved artifact"
    assert list(tmp_path.iterdir()) == [target]
    assert snapshot_memory().reserved_bytes == before


def test_compression_reserves_scratch_before_writing_member(monkeypatch):
    from nfit import resource_budget

    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: 1000)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 1000)
    monkeypatch.setattr(np.lib.format, "write_array", lambda *args, **kwargs: pytest.fail("member write"))
    output = io.BytesIO()
    with pytest.raises(resource_budget.ResourceLimitError):
        array_archive.write_array_archive(output, {"signal": np.arange(8192.)})
    assert output.getvalue() == b""


def test_in_memory_artifact_reserves_retained_output_before_serializing(monkeypatch):
    from nfit import resource_budget

    source = _tiny_mdhisto_data(2.0)
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: 1000)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 1000)
    monkeypatch.setattr(artifacts, "write_array_archive", lambda *args: pytest.fail("serialized"))
    with pytest.raises(resource_budget.ResourceLimitError):
        artifacts.dataset_artifact_bytes(source)


def test_integer_point_columns_reserve_cast_and_immutable_copy_before_decode(monkeypatch):
    from nfit import resource_budget
    from nfit.dataset import PointListData

    source = PointListData({"T": np.arange(4096.), "M": np.arange(4096.)},
                           coordinate_names=["T"], channels=[{"value": "M"}])
    payload = artifacts._payload(source)
    payload["column_0"] = np.arange(4096, dtype=np.int32)
    payload["column_1"] = np.arange(4096, dtype=np.uint8)
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    encoded = stream.getvalue()
    capacity = artifacts.dataset_artifact_capacity(encoded)
    converted_bytes = 2 * 2 * 4096 * np.dtype(float).itemsize
    assert capacity.peak_bytes == capacity.expanded_bytes + converted_bytes + 16 * 1024**2
    decoded = artifacts.read_dataset_artifact(encoded)
    for index, name in enumerate(("T", "M")):
        np.testing.assert_equal(decoded.column(name), payload[f"column_{index}"])
        assert decoded.column(name).dtype == np.dtype(float)
        assert not decoded.column(name).flags.writeable

    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: capacity.peak_bytes + 99)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 2 * capacity.peak_bytes)
    monkeypatch.setattr(artifacts, "_OwnedArchiveArrays", lambda *args: pytest.fail("decoded"))
    with pytest.raises(resource_budget.ResourceLimitError):
        artifacts.read_dataset_artifact(encoded)


def test_byte_mask_conversion_is_in_capacity_and_normal_mask_reuses_owned_storage():
    payload = artifacts._payload(_tiny_mdhisto_data(2.0))
    payload["mask"] = payload["mask"].astype(np.uint8)
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    encoded = stream.getvalue()
    capacity = artifacts.dataset_artifact_capacity(encoded)
    assert capacity.peak_bytes == capacity.expanded_bytes + 2 * payload["mask"].size + 16 * 1024**2
    decoded = artifacts.read_dataset_artifact(encoded)
    np.testing.assert_equal(decoded.mask, payload["mask"])
    assert decoded.mask.dtype == np.dtype(bool)
    assert not decoded.mask.flags.writeable

    payload["mask"] = payload["mask"].astype(bool)
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    stream.seek(0)
    with np.load(stream) as archive:
        owned = artifacts._OwnedArchiveArrays(archive)
        mask = owned._read("mask")
        owned.loaded["mask"] = mask
        decoded = artifacts.dataset_artifact_from_payload(owned)
        assert decoded.mask is mask


def test_streaming_writer_has_no_reservation_between_members_or_after_early_close():
    from nfit.resource_budget import reserve_memory, snapshot_memory

    output = io.BytesIO()
    writer = None
    before = snapshot_memory().reserved_bytes
    try:
        with reserve_memory(1024**2):
            writer = array_archive.array_archive_writer(output, max_member_bytes=8192,
                                                        compressed=False)
            write = writer.__enter__()
            write("signal", np.arange(1024.))
            assert snapshot_memory().reserved_bytes == before + 1024**2
        assert snapshot_memory().reserved_bytes == before
    finally:
        if writer is not None:
            writer.__exit__(None, None, None)
    assert snapshot_memory().reserved_bytes == before
    with np.load(io.BytesIO(output.getvalue())) as archive:
        np.testing.assert_equal(archive["signal"], np.arange(1024.))
