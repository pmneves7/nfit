"""Independent artifact cursors preserve validated snapshots and array contracts."""
from __future__ import annotations

import io
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nfit._parallel import thread_budget
from nfit.analysis import artifacts
from nfit.mapped_archive import is_mapped_array
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.project_archive import _project_artifact_reader_factory, open_project_artifact

MEMBER = "assets/binnings/cache/data.npz"


def _histogram(bias=0., size=16):
    value = np.arange(size, dtype=float) + bias
    value.setflags(write=False)
    error = np.full(size, .25)
    error.setflags(write=False)
    return MDHistoData(
        (MDHistoAxis("E", np.arange(size+1.), "meV", "energy"),),
        value, error, np.zeros(size, bool), value,
        metadata={"normalization_denominator": value},
        auxiliary_channels={
            "numerator": MDHistoChannel(value, error, label="Numerator"),
            "variance": MDHistoChannel(value, label="Variance"),
            "normalization_denominator": MDHistoChannel(value, label="Exposure"),
        },
    )


def _project(path, data=None, *, payload=None, compression=zipfile.ZIP_STORED):
    if payload is None:
        payload = artifacts.dataset_artifact_bytes(data)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(MEMBER, payload, compress_type=compression)
    return path


def _assert_same(actual, expected):
    for name in ("signal", "errors", "mask", "num_events"):
        array = getattr(actual, name)
        np.testing.assert_array_equal(array, getattr(expected, name))
        assert not array.flags.writeable
    for axis, expected_axis in zip(actual.axes, expected.axes, strict=True):
        np.testing.assert_array_equal(axis.values, expected_axis.values)
        assert not axis.values.flags.writeable
    for name, channel in actual.auxiliary_channels.items():
        np.testing.assert_array_equal(channel.values, expected.auxiliary_channels[name].values)
        assert not channel.values.flags.writeable
        if channel.errors is not None:
            np.testing.assert_array_equal(channel.errors, expected.auxiliary_channels[name].errors)
            assert not channel.errors.flags.writeable
    assert np.shares_memory(actual.metadata["normalization_denominator"],
                            actual.auxiliary_channels["normalization_denominator"].values)


@pytest.fixture
def force_parallel(monkeypatch):
    monkeypatch.setattr(artifacts, "operation_worker_count", lambda *_args, **_kwargs: 4)


@pytest.mark.skipif(not hasattr(os, "pread"), reason="Platform has no positional file reads")
def test_cloned_streams_have_independent_bounded_cursors(tmp_path):
    payload = bytes(range(256))*32
    project = _project(tmp_path / "project.nfit", payload=payload)
    with open_project_artifact(project, MEMBER) as owner:
        factory = _project_artifact_reader_factory(owner)
        assert factory is not None
        with factory() as first, factory() as second:
            first.seek(17)
            second.seek(600)
            owner.seek(33)
            with ThreadPoolExecutor(max_workers=2) as executor:
                a = executor.submit(first.read, 200)
                b = executor.submit(second.read, 300)
                assert a.result() == payload[17:217]
                assert b.result() == payload[600:900]
            assert owner.tell() == 33
            assert owner.read(20) == payload[33:53]
            first.seek(-4, os.SEEK_END)
            assert first.read(9999) == payload[-4:]
            assert first.read() == b""
            first.seek(10**6)
            assert first.tell() == len(payload)
            with pytest.raises(ValueError, match="negative"):
                second.seek(-1)


@pytest.mark.skipif(not hasattr(os, "pread"), reason="Platform has no positional file reads")
def test_parallel_decode_keeps_original_snapshot_across_atomic_replace(tmp_path, monkeypatch, force_parallel):
    original, replacement_data = _histogram(), _histogram(100.)
    project = _project(tmp_path / "project.nfit", original)
    replacement = _project(tmp_path / "replacement.nfit", replacement_data)
    original_factory = artifacts._project_artifact_reader_factory
    lock = threading.Lock()
    calls = []

    def clone_factory(owner):
        factory = original_factory(owner)
        assert factory is not None

        def clone():
            with lock:
                if not calls:
                    replacement.replace(project)
                calls.append(1)
            return factory()

        return clone

    monkeypatch.setattr(artifacts, "_project_artifact_reader_factory", clone_factory)
    decoded = artifacts.read_project_dataset_artifact(project, MEMBER)
    _assert_same(decoded, original)
    assert len(calls) == 8  # Four primaries and three values plus one channel error.
    monkeypatch.setattr(artifacts, "_project_artifact_reader_factory", original_factory)
    _assert_same(artifacts.read_project_dataset_artifact(project, MEMBER), replacement_data)


@pytest.mark.skipif(not hasattr(os, "pread"), reason="Platform has no positional file reads")
def test_header_validation_and_stored_view_use_one_original_descriptor(tmp_path, monkeypatch):
    original = _histogram()
    project = _project(tmp_path / "project.nfit", original)
    replacement = _project(tmp_path / "replacement.nfit", _histogram(200.))
    existing_open = zipfile.ZipFile.open
    replaced = False

    def open_and_replace(archive, name, *args, **kwargs):
        nonlocal replaced
        result = existing_open(archive, name, *args, **kwargs)
        if archive.filename == str(project) and not replaced:
            replacement.replace(project)
            replaced = True
        return result

    monkeypatch.setattr(zipfile.ZipFile, "open", open_and_replace)
    _assert_same(artifacts.read_project_dataset_artifact(project, MEMBER), original)
    assert replaced


@pytest.mark.parametrize("fallback", ["no_pread", "compressed_outer", "standalone", "bytes"])
def test_noncloneable_inputs_keep_shared_reader(tmp_path, monkeypatch, force_parallel, fallback):
    data = _histogram()
    payload = artifacts.dataset_artifact_bytes(data)
    if fallback == "no_pread":
        monkeypatch.delattr(os, "pread", raising=False)
    expected_calls = []
    factory_helper = artifacts._project_artifact_reader_factory

    def inspect_factory(stream):
        result = factory_helper(stream)
        expected_calls.append(result)
        return result

    monkeypatch.setattr(artifacts, "_project_artifact_reader_factory", inspect_factory)
    if fallback in {"no_pread", "compressed_outer"}:
        compression = zipfile.ZIP_DEFLATED if fallback == "compressed_outer" else zipfile.ZIP_STORED
        path = _project(tmp_path / "project.nfit", payload=payload, compression=compression)
        decoded = artifacts.read_project_dataset_artifact(path, MEMBER)
    elif fallback == "standalone":
        path = tmp_path / "data.npz"
        path.write_bytes(payload)
        decoded = artifacts.read_dataset_artifact(path)
    else:
        decoded = artifacts.read_dataset_artifact(payload)
    _assert_same(decoded, data)
    assert expected_calls == [None]


def test_mapped_reader_does_not_request_independent_resident_streams(tmp_path, monkeypatch):
    data = _histogram()
    project = _project(tmp_path / "project.nfit", data)
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 1)
    monkeypatch.setattr(artifacts, "_project_artifact_reader_factory",
                        lambda *_args: pytest.fail("Resident clone requested for mapped data"))
    decoded = artifacts.read_project_dataset_artifact(project, MEMBER, memory_map=True)
    _assert_same(decoded, data)
    assert is_mapped_array(decoded.signal)


def _replace_inner_member(payload, name, replacement):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(output, "w") as changed:
        for info in original.infolist():
            changed.writestr(info, replacement if info.filename == name else original.read(info.filename))
    return output.getvalue()


@pytest.mark.parametrize("damage", ["crc", "npy_header", "object_array", "outer_header"])
def test_parallel_reader_preserves_archive_error_checks(tmp_path, force_parallel, damage):
    payload = artifacts.dataset_artifact_bytes(_histogram())
    expected_exception, message = ValueError, None
    if damage == "crc":
        broken = bytearray(payload)
        start = 0
        while True:
            start = broken.find(b"PK\x01\x02", start)
            assert start >= 0
            if broken[start+46:start+56] == b"signal.npy":
                broken[start+16] ^= 1
                break
            start += 4
        payload = bytes(broken)
        expected_exception, message = zipfile.BadZipFile, "CRC"
    elif damage == "object_array":
        stream = io.BytesIO()
        np.save(stream, np.full(16, object(), dtype=object), allow_pickle=True)
        payload = _replace_inner_member(payload, "signal.npy", stream.getvalue())
        message = "object arrays are not supported"
    elif damage == "npy_header":
        stream = io.BytesIO()
        np.save(stream, np.arange(16.), allow_pickle=False)
        broken = stream.getvalue()
        payload = _replace_inner_member(payload, "signal.npy", broken[:8]+b"\xff\xff"+broken[10:])
    project = _project(tmp_path / "project.nfit", payload=payload)
    if damage == "outer_header":
        with zipfile.ZipFile(project) as archive:
            offset = archive.getinfo(MEMBER).header_offset
        broken = bytearray(project.read_bytes())
        broken[offset] = 0
        project.write_bytes(broken)
        expected_exception, message = zipfile.BadZipFile, "header"
    with pytest.raises(expected_exception, match=message):
        artifacts.read_project_dataset_artifact(project, MEMBER)


@pytest.mark.parametrize("budget_mib,expected_workers", [(16, 1), (32, 2)])
def test_parallel_restore_honors_cpu_and_memory_worker_budget(tmp_path, monkeypatch, budget_mib, expected_workers):
    from nfit import performance, resource_budget

    # 750k cells exceed the normal parallel threshold; each largest member is
    # 6 MB, so the unchanged 16 MiB per-worker floor gives this exact RAM cap.
    project = _project(tmp_path / "project.nfit", _histogram(size=750_000))
    # These tiny limits constrain worker scratch only. The independent whole-
    # process admission ceiling must cover the interpreter and decoded cube.
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: 1024**3)
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100 * 1024**2)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 1024**3)
    monkeypatch.setattr(performance, "scientific_memory_limit_bytes", lambda: budget_mib*1024**2)
    executor = artifacts.ThreadPoolExecutor
    workers = []

    def tracked_executor(*args, **kwargs):
        workers.append(kwargs["max_workers"])
        return executor(*args, **kwargs)

    monkeypatch.setattr(artifacts, "ThreadPoolExecutor", tracked_executor)
    with thread_budget(2):
        data = artifacts.read_project_dataset_artifact(project, MEMBER)
    assert data.signal.size == 750_000
    assert workers == ([] if expected_workers == 1 else [expected_workers])


def test_parallel_decoded_arrays_remain_save_compatible(tmp_path, force_parallel):
    original = _histogram()
    project = _project(tmp_path / "project.nfit", original)
    decoded = artifacts.read_project_dataset_artifact(project, MEMBER)
    destination = tmp_path / "saved_again.npz"
    artifacts.write_dataset_artifact(decoded, destination)
    _assert_same(artifacts.read_dataset_artifact(destination), original)


def test_independent_stream_factory_is_owned_by_project_archive():
    import ast
    from pathlib import Path

    from nfit import project_archive

    assert artifacts._project_artifact_reader_factory is project_archive._project_artifact_reader_factory
    tree = ast.parse(Path(project_archive.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.split(".")[-1] in {"artifacts", "project_gui", "project_data", "numpy"}
                   or name.startswith(("PySide", "PyQt")) for name in imports)
