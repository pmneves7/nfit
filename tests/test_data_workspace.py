"""Scientific storage follows each operation, without startup mount checks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from nfit import data_workspace
from nfit.data_workspace import (
    DataWorkspaceError,
    project_data_workspace,
    temporary_data_directory,
)
from nfit.pipeline import DataGroup, DatasetEntry


def test_staging_is_created_beside_owner_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr(data_workspace.tempfile, "tempdir", str(tmp_path / "unavailable"))
    before = set(tmp_path.iterdir())
    with temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-data-") as name:
        staging = Path(name)
        assert staging.is_relative_to(tmp_path / ".nfit-work")
        assert staging.parent.name.startswith("session-")
        (staging / "data.bin").write_bytes(b"payload")
    assert not staging.exists()
    assert set(tmp_path.iterdir()) == before


def test_unavailable_owner_fails_without_creating_ancestors(tmp_path):
    missing = tmp_path / "unmounted/IPTS-1/project.nfit"
    with pytest.raises(DataWorkspaceError, match="unavailable"):
        temporary_data_directory(missing, prefix="nfit-data-")
    assert not (tmp_path / "unmounted").exists()


def test_broken_owner_link_fails_at_operation(tmp_path):
    alias = tmp_path / "experiment"
    alias.symlink_to(tmp_path / "unmounted", target_is_directory=True)
    with pytest.raises(DataWorkspaceError, match="unavailable"):
        temporary_data_directory(alias / "project.nfit", prefix="nfit-data-")
    assert not (tmp_path / "unmounted").exists()


def test_write_failure_does_not_fall_back(tmp_path, monkeypatch):
    attempted = []

    def denied(**kwargs):
        attempted.append(kwargs["dir"])
        raise PermissionError("read-only experiment")

    monkeypatch.setattr(data_workspace.tempfile, "TemporaryDirectory", denied)
    with pytest.raises(DataWorkspaceError, match="writable project"):
        temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-data-")
    assert len(attempted) == 1
    assert attempted[0].is_relative_to(tmp_path / ".nfit-work")
    assert not (tmp_path / ".nfit-work").exists()


def test_independent_project_ownership_and_failed_save_restoration(tmp_path):
    first = DatasetEntry(name="first", data=None, metadata={"_project_path": "old.nfit"})
    second = DatasetEntry(name="second", data=None)
    group = DataGroup(name="runs", datasets=[first, second])
    project = SimpleNamespace(data_groups=[group])
    other = DatasetEntry(name="other", data=None, metadata={"_project_path": "other.nfit"})
    other_project = SimpleNamespace(data_groups=[DataGroup(name="other", datasets=[other])])
    with pytest.raises(RuntimeError):
        with project_data_workspace(project, tmp_path / "new.nfit"):
            assert first.metadata["_project_path"] == str(tmp_path / "new.nfit")
            with project_data_workspace(other_project, tmp_path / "other-new.nfit"):
                assert second.metadata["_project_path"] == str(tmp_path / "new.nfit")
            raise RuntimeError("save failed")
    assert first.metadata["_project_path"] == "old.nfit"
    assert "_project_path" not in second.metadata
    assert not hasattr(group, "_data_workspace_owner_path")
    assert other.metadata["_project_path"] == str(tmp_path / "other-new.nfit")


def test_workspace_service_is_independent_and_has_no_startup_side_effects():
    tree = ast.parse(Path(data_workspace.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.startswith(("PySide", "PyQt", "nfit.project_gui")) for name in imports)
    # Disk allocation belongs to function calls, never module initialization.
    assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) for node in tree.body)


def test_temporary_inventory_tracks_writer_sizes_without_directory_scans(tmp_path, monkeypatch):
    with temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-owned-") as name:
        target = Path(name) / "cached.bin"
        target.write_bytes(b"cached payload")
        data_workspace.record_temporary_file(target)
        monkeypatch.setattr(Path, "rglob", lambda *_args: pytest.fail("inventory scanned scientific storage"))
        assert dict(data_workspace.temporary_storage_usage())[name] == len(b"cached payload")
        target.write_bytes(b"new")
        data_workspace.record_temporary_file(target, 3)
        assert dict(data_workspace.temporary_storage_usage())[name] == 3
    assert name not in dict(data_workspace.temporary_storage_usage())


def test_temporary_directory_changes_only_future_allocations(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    path = tmp_path / "project.nfit"
    dataset = DatasetEntry("Source", None, metadata={"_project_path": str(path)})
    project = SimpleNamespace(settings={}, data_groups=[DataGroup("Runs", datasets=[dataset])], _project_path=path)
    try:
        data_workspace.set_project_temporary_directory(project, str(first))
        with temporary_data_directory(path, prefix="nfit-old-") as old:
            data_workspace.set_project_temporary_directory(project, str(second))
            with temporary_data_directory(path, prefix="nfit-new-") as new:
                assert Path(old).is_relative_to(first / ".nfit-work")
                assert Path(old).is_dir()
                assert Path(new).is_relative_to(second / ".nfit-work")
    finally:
        data_workspace.set_project_temporary_directory(project, None)


def test_unavailable_saved_temporary_choice_is_deferred_until_allocation(tmp_path):
    path = tmp_path / "project.nfit"
    project = SimpleNamespace(settings={}, data_groups=[], _project_path=path)
    try:
        data_workspace.set_project_temporary_directory(project, str(tmp_path / "missing"), validate=False)
        with pytest.raises(DataWorkspaceError, match="unavailable"):
            temporary_data_directory(path, prefix="nfit-data-")
        assert not (tmp_path / "missing").exists()
    finally:
        data_workspace.set_project_temporary_directory(project, None)


def test_empty_explorer_opens_without_temporary_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.project_gui import NfitProject, NfitProjectExplorer

    def unavailable(*args, **kwargs):
        pytest.fail("empty application startup attempted temporary disk allocation")

    monkeypatch.setattr(data_workspace.tempfile, "TemporaryDirectory", unavailable)
    monkeypatch.setattr(data_workspace.tempfile, "tempdir", str(tmp_path / "missing-mount"))
    explorer = NfitProjectExplorer(NfitProject())
    try:
        explorer.window.show()
        explorer.app.processEvents()
        assert explorer.window.isVisible()
        assert explorer.project_path is None
        assert not (tmp_path / "missing-mount").exists()
    finally:
        explorer._allow_window_close = True
        explorer.window.close()
