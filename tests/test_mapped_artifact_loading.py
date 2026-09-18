"""Large lazy histogram artifacts share reclaimable read-only array storage."""

import gc
import io
import zipfile

import numpy as np
import pytest

from nfit.analysis import artifacts
from nfit.mapped_archive import array_storage_nbytes, is_mapped_array
from nfit.rebin_cache import RebinCache, RebinCacheBudget
from tests.project_gui_test_support import _tiny_mdhisto_data


@pytest.fixture
def mapped_policy(monkeypatch):
    monkeypatch.setattr(artifacts, "_MAPPED_ARTIFACT_MIN_BYTES", 1)
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 1)
    monkeypatch.setattr(artifacts, "scientific_memory_limit_bytes", lambda: 1)


def test_ordinary_reads_stay_resident_and_auto_reads_map(tmp_path, mapped_policy):
    original = _tiny_mdhisto_data(3.0)
    source = tmp_path / "data.npz"
    artifacts.write_dataset_artifact(original, source)
    resident = artifacts.read_dataset_artifact(source)
    mapped = artifacts.read_dataset_artifact(source, memory_map=None)
    assert not is_mapped_array(resident.signal)
    assert is_mapped_array(mapped.signal)
    for name in ("signal", "errors", "mask", "num_events"):
        np.testing.assert_equal(getattr(mapped, name), getattr(original, name))
        assert not getattr(mapped, name).flags.writeable


def test_auto_mapping_requires_both_size_and_memory_pressure(tmp_path, monkeypatch):
    source = tmp_path / "data.npz"
    artifacts.write_dataset_artifact(_tiny_mdhisto_data(3.0), source)
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 1)
    monkeypatch.setattr(artifacts, "_MAPPED_ARTIFACT_MIN_BYTES", 10**9)
    monkeypatch.setattr(artifacts, "scientific_memory_limit_bytes", lambda: 1)
    assert not is_mapped_array(artifacts.read_dataset_artifact(source, memory_map=None).signal)
    monkeypatch.setattr(artifacts, "_MAPPED_ARTIFACT_MIN_BYTES", 1)
    monkeypatch.setattr(artifacts, "scientific_memory_limit_bytes", lambda: 10**9)
    assert not is_mapped_array(artifacts.read_dataset_artifact(source, memory_map=None).signal)


def test_point_list_artifacts_keep_the_existing_reader(tmp_path, mapped_policy):
    from nfit.dataset import PointListData

    source = tmp_path / "points.npz"
    original = PointListData({"x": np.arange(4.0), "signal": np.arange(4.0)})
    artifacts.write_dataset_artifact(original, source)
    result = artifacts.read_dataset_artifact(source, memory_map=True)
    np.testing.assert_equal(result.column("signal"), original.column("signal"))
    assert not is_mapped_array(result.column("signal"))


@pytest.mark.parametrize("outer_compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_lazy_project_cache_mapping_and_escaped_view_lifetime(
    tmp_path, mapped_policy, outer_compression
):
    original = _tiny_mdhisto_data(5.0)
    project = tmp_path / "project.nfit"
    member = "assets/binnings/test/data.npz"
    with zipfile.ZipFile(project, "w", compression=outer_compression) as archive:
        archive.writestr(member, artifacts.dataset_artifact_bytes(original))
    before = project.read_bytes()
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    cache.set_project_backing(
        "key", signature="sig", project_path=project, member=member, lazy=True
    )
    decoded = cache.get("key")[1]
    assert is_mapped_array(decoded.signal)
    assert cache.get("key")[1] is decoded
    storage = array_storage_nbytes(decoded)
    assert budget.total_bytes() == storage.heap
    assert storage.mapped > 0
    view = decoded.signal.reshape(-1)[1:]
    cache.clear()
    del decoded
    gc.collect()
    np.testing.assert_equal(view, original.signal.reshape(-1)[1:])
    assert project.read_bytes() == before


def test_workspace_failure_rewinds_before_resident_fallback(monkeypatch, mapped_policy):
    from nfit.mapped_archive import MappedWorkspaceError

    original = _tiny_mdhisto_data(2.0)
    stream = io.BytesIO(artifacts.dataset_artifact_bytes(original))

    def unavailable(source, **kwargs):
        source.seek(17)
        raise MappedWorkspaceError("no temporary disk space")

    monkeypatch.setattr(artifacts, "read_mapped_array_archive", unavailable)
    decoded = artifacts.read_dataset_artifact(stream, memory_map=None)
    assert not is_mapped_array(decoded.signal)
    np.testing.assert_equal(decoded.signal, original.signal)


def test_corrupt_mapped_payload_is_not_retried_as_resident(monkeypatch, mapped_policy):
    stream = io.BytesIO(artifacts.dataset_artifact_bytes(_tiny_mdhisto_data(2.0)))

    def corrupt(source, **kwargs):
        raise ValueError("invalid NPY payload")

    monkeypatch.setattr(artifacts, "read_mapped_array_archive", corrupt)
    with pytest.raises(ValueError, match="invalid NPY"):
        artifacts.read_dataset_artifact(stream, memory_map=None)


def test_mapped_service_does_not_import_gui_or_facade():
    import ast
    from pathlib import Path

    import nfit.mapped_archive

    tree = ast.parse(Path(nfit.mapped_archive.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(
        name.startswith(("PySide", "PyQt"))
        or name.split(".")[-1] in {"project_gui", "project_data", "artifacts", "rebin_cache"}
        for name in imports
    )
