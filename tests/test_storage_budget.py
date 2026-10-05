"""Scientific staging reservations retain filesystem headroom and old saves."""

from types import SimpleNamespace

import pytest

from nfit import storage_budget

GIB = 1024**3


@pytest.fixture(autouse=True)
def controlled_space(monkeypatch):
    assert storage_budget._RESERVED == {}
    monkeypatch.setattr(storage_budget.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * GIB))
    yield
    assert storage_budget._RESERVED == {}


def test_reservation_retains_ten_percent_headroom(tmp_path):
    with storage_budget.reserve_disk_space(tmp_path, 90 * GIB):
        assert sum(storage_budget._RESERVED.values()) == 90 * GIB
    with pytest.raises(OSError, match="headroom"):
        with storage_budget.reserve_disk_space(tmp_path, 90 * GIB + 1):
            pytest.fail("must reserve filesystem headroom")


def test_same_filesystem_directories_share_pending_reservations(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    with storage_budget.reserve_disk_space(first, 80 * GIB):
        with pytest.raises(OSError, match="temporary disk space"):
            with storage_budget.reserve_disk_space(second, 10 * GIB + 1):
                pytest.fail("same filesystem budget spent twice")
        with storage_budget.reserve_disk_space(second, 10 * GIB):
            assert sum(storage_budget._RESERVED.values()) == 90 * GIB
        assert sum(storage_budget._RESERVED.values()) == 80 * GIB


def test_failed_write_releases_its_reservation(tmp_path):
    with pytest.raises(RuntimeError, match="cancelled"):
        with storage_budget.reserve_disk_space(tmp_path, 80 * GIB):
            raise RuntimeError("cancelled")
    with storage_budget.reserve_disk_space(tmp_path, 90 * GIB):
        assert sum(storage_budget._RESERVED.values()) == 90 * GIB


def test_low_space_rejects_atomic_project_save_before_touching_original(tmp_path, monkeypatch):
    from nfit.project_archive import write_project_manifest

    path = tmp_path / "project.nfit"
    write_project_manifest(path, {"name": "original"})
    original = path.read_bytes()
    monkeypatch.setattr(storage_budget.shutil, "disk_usage", lambda root: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="temporary disk space"):
        write_project_manifest(path, {"name": "replacement"}, preserve_existing=True)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
