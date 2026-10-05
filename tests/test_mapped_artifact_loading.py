"""Project cubes use whole-array RAM; scripting mappings remain explicit."""

import gc
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from nfit.analysis import artifacts
from nfit.mapped_archive import array_storage_nbytes, is_mapped_array
from nfit.rebin_cache import RebinCache, RebinCacheBudget
from tests.project_gui_test_support import _tiny_mdhisto_data


@pytest.fixture
def mapped_policy(monkeypatch):
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 1)


def test_ordinary_and_none_reads_stay_resident_and_explicit_reads_map(tmp_path, mapped_policy):
    original = _tiny_mdhisto_data(3.0)
    source = tmp_path / "data.npz"
    artifacts.write_dataset_artifact(original, source)
    resident = artifacts.read_dataset_artifact(source)
    ordinary = artifacts.read_dataset_artifact(source, memory_map=None)
    mapped = artifacts.read_dataset_artifact(source, memory_map=True)
    assert not is_mapped_array(resident.signal)
    assert not is_mapped_array(ordinary.signal)
    assert is_mapped_array(mapped.signal)
    for name in ("signal", "errors", "mask", "num_events"):
        np.testing.assert_equal(getattr(mapped, name), getattr(original, name))
        assert not getattr(mapped, name).flags.writeable


def test_none_never_automatically_maps_under_memory_pressure(tmp_path, monkeypatch):
    from nfit import resource_budget

    source = tmp_path / "data.npz"
    artifacts.write_dataset_artifact(_tiny_mdhisto_data(3.0), source)
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 1)
    monkeypatch.setattr(artifacts, "read_mapped_array_archive", lambda *args, **kwargs: pytest.fail("automatic mapping"))
    capacity = artifacts.dataset_artifact_capacity(source)
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: capacity.peak_bytes * 2)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: capacity.peak_bytes + 100)
    assert not is_mapped_array(artifacts.read_dataset_artifact(source, memory_map=None).signal)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: capacity.peak_bytes + 99)
    with pytest.raises(resource_budget.ResourceLimitError):
        artifacts.read_dataset_artifact(source, memory_map=None)


def test_point_list_artifacts_keep_the_existing_reader(tmp_path, mapped_policy):
    from nfit.dataset import PointListData

    source = tmp_path / "points.npz"
    original = PointListData({"x": np.arange(4.0), "signal": np.arange(4.0)})
    artifacts.write_dataset_artifact(original, source)
    result = artifacts.read_dataset_artifact(source, memory_map=True)
    np.testing.assert_equal(result.column("signal"), original.column("signal"))
    assert not is_mapped_array(result.column("signal"))


def test_file_and_project_mapping_use_owner_storage(tmp_path, mapped_policy, monkeypatch):
    from nfit import mapped_archive

    owner = tmp_path / "project-storage"
    owner.mkdir()
    source = owner / "data.npz"
    artifacts.write_dataset_artifact(_tiny_mdhisto_data(3.0), source)
    project = owner / "project.nfit"
    member = "assets/binnings/test/data.npz"
    with zipfile.ZipFile(project, "w") as archive:
        archive.writestr(member, source.read_bytes())
    mkstemp = mapped_archive.tempfile.mkstemp
    destinations = []

    def allocate(**kwargs):
        destinations.append(kwargs["dir"])
        return mkstemp(**kwargs)

    monkeypatch.setattr(mapped_archive.tempfile, "tempdir", str(tmp_path / "missing-system-tmp"))
    monkeypatch.setattr(mapped_archive.tempfile, "mkstemp", allocate)
    data = artifacts.read_dataset_artifact(source, memory_map=True)
    cached = artifacts.read_project_dataset_artifact(project, member, memory_map=True)
    assert is_mapped_array(data.signal) and is_mapped_array(cached.signal)
    assert destinations and all(Path(directory) == owner for directory in destinations)


@pytest.mark.parametrize("outer_compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_lazy_project_cache_stays_in_ram_and_preserves_escaped_view_lifetime(
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
    assert not is_mapped_array(decoded.signal)
    assert cache.get("key")[1] is decoded
    storage = array_storage_nbytes(decoded)
    assert budget.total_bytes() == storage.heap
    assert storage.mapped == 0
    view = decoded.signal.reshape(-1)[1:]
    cache.clear()
    del decoded
    gc.collect()
    np.testing.assert_equal(view, original.signal.reshape(-1)[1:])
    assert project.read_bytes() == before


def test_explicit_mapping_workspace_failure_never_falls_back_to_ram(monkeypatch, mapped_policy):
    from nfit.mapped_archive import MappedWorkspaceError

    original = _tiny_mdhisto_data(2.0)
    stream = io.BytesIO(artifacts.dataset_artifact_bytes(original))

    def unavailable(source, **kwargs):
        source.seek(17)
        raise MappedWorkspaceError("no temporary disk space")

    monkeypatch.setattr(artifacts, "read_mapped_array_archive", unavailable)
    monkeypatch.setattr(artifacts, "_OwnedArchiveArrays", lambda *args: pytest.fail("resident fallback"))
    with pytest.raises(MappedWorkspaceError, match="no temporary disk space"):
        artifacts.read_dataset_artifact(stream, memory_map=True)


def test_corrupt_mapped_payload_is_not_retried_as_resident(monkeypatch, mapped_policy):
    stream = io.BytesIO(artifacts.dataset_artifact_bytes(_tiny_mdhisto_data(2.0)))

    def corrupt(source, **kwargs):
        raise ValueError("invalid NPY payload")

    monkeypatch.setattr(artifacts, "read_mapped_array_archive", corrupt)
    with pytest.raises(ValueError, match="invalid NPY"):
        artifacts.read_dataset_artifact(stream, memory_map=True)


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
