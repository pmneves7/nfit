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
