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


def _cache_location(dataset, signature):
    from nfit.raw_dgs_cache import cache_event_chunks

    list(cache_event_chunks(dataset, signature, {}, {}, iter(())))
    return dataset._raw_dgs_reduction_cache.content


def test_same_source_unsaved_projects_allocate_independently_and_reset(tmp_path):
    from nfit.project_io import NfitProject

    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    source = tmp_path / "shared.nxs"
    first = DatasetEntry("first", None, metadata={"source_file": str(source)})
    second = DatasetEntry("second", None, metadata={"source_file": str(source)})
    left = NfitProject([DataGroup("left", datasets=[first])])
    right = NfitProject([DataGroup("right", datasets=[second])])
    data_workspace.set_project_temporary_directory(left, str(first_dir))
    data_workspace.set_project_temporary_directory(right, str(second_dir))
    assert str(source) not in data_workspace._WORKSPACE_OVERRIDES
    old_cache = None
    try:
        old = _cache_location(first, "first")
        old_cache = first._raw_dgs_reduction_cache
        assert old.is_relative_to(first_dir)
        assert _cache_location(second, "second").is_relative_to(second_dir)
        data_workspace.set_project_temporary_directory(left, None)
        assert _cache_location(first, "reset").is_relative_to(tmp_path / ".nfit-work")
        assert old.is_file()
        assert _cache_location(second, "unchanged").is_relative_to(second_dir)
        assert "_project_path" not in first.metadata
        assert "_project_path" not in second.metadata
    finally:
        for dataset in (first, second):
            dataset._raw_dgs_reduction_cache.staging.cleanup()
        if old_cache is not None:
            old_cache.staging.cleanup()


def test_same_saved_path_explicit_default_ignores_other_project_choice(tmp_path):
    from nfit.project_io import NfitProject

    selected = tmp_path / "selected"
    selected.mkdir()
    archive = tmp_path / "shared.nfit"
    first = DatasetEntry("first", None, metadata={"_project_path": str(archive)})
    second = DatasetEntry("second", None, metadata={"_project_path": str(archive)})
    left = NfitProject([DataGroup("left", datasets=[first])])
    right = NfitProject([DataGroup("right", datasets=[second])])
    left._project_path = right._project_path = archive
    data_workspace.set_project_temporary_directory(left, None)
    data_workspace.set_project_temporary_directory(right, str(selected))
    try:
        assert _cache_location(first, "default").is_relative_to(tmp_path / ".nfit-work")
        assert _cache_location(second, "selected").is_relative_to(selected)
        with temporary_data_directory(archive, prefix="legacy-") as name:
            assert Path(name).is_relative_to(selected)
    finally:
        first._raw_dgs_reduction_cache.staging.cleanup()
        second._raw_dgs_reduction_cache.staging.cleanup()
        data_workspace.set_project_temporary_directory(right, None)


def test_empty_project_choice_is_inherited_by_new_groups_imports_and_copies(tmp_path):
    import copy

    from nfit.project_gui import create_data_group
    from nfit.project_imports import import_dataset_paths
    from nfit.project_io import NfitProject

    selected = tmp_path / "selected"
    selected.mkdir()
    project = NfitProject()
    data_workspace.set_project_temporary_directory(project, str(selected))
    group = create_data_group(project)
    entry = import_dataset_paths(group, [tmp_path / "source.npz"],
                                 dataset_file_loader=lambda _path: (None, {}))[0]
    alias = entry.copy()
    copied_project = copy.deepcopy(project)
    copied = copied_project.data_groups[0].datasets[0]
    try:
        for dataset in (entry, alias, copied):
            assert _cache_location(dataset, dataset.id).is_relative_to(selected)
        data_workspace.set_project_temporary_directory(project, None)
        assert _cache_location(alias, "alias-default").is_relative_to(tmp_path / ".nfit-work")
        assert _cache_location(copied, "independent").is_relative_to(selected)
        assert "_project_path" not in entry.metadata
    finally:
        for dataset in (entry, alias, copied):
            dataset._raw_dgs_reduction_cache.staging.cleanup()


def test_failed_save_restores_runtime_choices_and_path_compatibility(tmp_path, monkeypatch):
    from nfit import project_gui
    from nfit.project_io import NfitProject

    selected = tmp_path / "selected"
    selected.mkdir()
    old = tmp_path / "old.nfit"
    new = tmp_path / "new.nfit"
    entry = DatasetEntry("run", None, metadata={"source_file": str(tmp_path / "run.nxs")})
    group = DataGroup("runs", datasets=[entry])
    project = NfitProject([group])
    project._project_path = old
    data_workspace.bind_data_workspace(group, old)
    data_workspace.set_project_temporary_directory(project, str(selected))
    preference = entry._data_workspace_preference

    def fail(*_args, **_kwargs):
        assert entry.metadata["_project_path"] == str(new)
        assert _cache_location(entry, "during-save").is_relative_to(selected)
        raise RuntimeError("save failed")

    monkeypatch.setattr(project_gui, "write_project_manifest", fail)
    try:
        with pytest.raises(RuntimeError, match="save failed"):
            project_gui.save_project(project, new)
        assert entry.metadata["_project_path"] == str(old)
        assert group._data_workspace_owner_path == str(old)
        assert project._project_path == old
        assert entry._data_workspace_preference is preference
        assert str(new) not in data_workspace._WORKSPACE_OVERRIDES
        with temporary_data_directory(old, prefix="legacy-") as name:
            assert Path(name).is_relative_to(selected)
        assert _cache_location(entry, "after-save").is_relative_to(selected)
    finally:
        entry._raw_dgs_reduction_cache.staging.cleanup()
        data_workspace.set_project_temporary_directory(project, None)


def test_selected_storage_missing_does_not_spill_for_dataset(tmp_path):
    from nfit.project_io import NfitProject

    entry = DatasetEntry("run", None, metadata={"source_file": str(tmp_path / "run.nxs")})
    project = NfitProject([DataGroup("runs", datasets=[entry])])
    missing = tmp_path / "missing"
    data_workspace.set_project_temporary_directory(project, str(missing), validate=False)
    with pytest.raises(DataWorkspaceError, match="unavailable"):
        _cache_location(entry, "missing")
    assert not missing.exists()
    assert not (tmp_path / ".nfit-work").exists()


def test_composite_scope_keeps_runtime_storage_preference(tmp_path):
    from nfit.pipeline import DatasetGroup
    from nfit.project_composites import _composite_scope
    from nfit.project_io import NfitProject

    selected = tmp_path / "selected"
    selected.mkdir()
    node = DatasetGroup("nested")
    group = DataGroup("root", subgroups=[node])
    project = NfitProject([group])
    data_workspace.set_project_temporary_directory(project, str(selected))
    scope = _composite_scope(group, node)
    with temporary_data_directory(tmp_path / "project.nfit", prefix="scope-", workspace=scope) as name:
        assert Path(name).is_relative_to(selected)


def test_save_reads_edited_storage_setting_and_restores_choice_on_failure(tmp_path):
    from nfit.project_io import NfitProject

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    archive = tmp_path / "saved.nfit"
    entry = DatasetEntry("run", None, metadata={"source_file": str(tmp_path / "run.nxs")})
    project = NfitProject([DataGroup("runs", datasets=[entry])])
    project._project_path = archive
    data_workspace.set_project_temporary_directory(project, str(first))
    project.settings["temporary_storage_directory"] = str(second)
    try:
        with pytest.raises(RuntimeError, match="failed"):
            with project_data_workspace(project, tmp_path / "new.nfit"):
                assert _cache_location(entry, "during").is_relative_to(second)
                raise RuntimeError("failed")
        assert _cache_location(entry, "after").is_relative_to(first)
        with temporary_data_directory(archive, prefix="legacy-") as name:
            assert Path(name).is_relative_to(first)
        with project_data_workspace(project, tmp_path / "new.nfit"):
            assert _cache_location(entry, "retry").is_relative_to(second)
        assert _cache_location(entry, "complete").is_relative_to(second)
    finally:
        entry._raw_dgs_reduction_cache.staging.cleanup()
        data_workspace.set_project_temporary_directory(project, None)


def test_explicit_dataset_context_never_borrows_legacy_path_choice(tmp_path):
    from nfit.project_io import NfitProject

    selected = tmp_path / "selected"
    selected.mkdir()
    source = tmp_path / "same-path.nxs"
    legacy_project = NfitProject()
    legacy_project._project_path = source
    data_workspace.set_project_temporary_directory(legacy_project, str(selected))
    independent = DatasetEntry("independent", None, metadata={"source_file": str(source)})
    try:
        assert _cache_location(independent, "independent").is_relative_to(tmp_path / ".nfit-work")
        with temporary_data_directory(source, prefix="legacy-") as name:
            assert Path(name).is_relative_to(selected)
    finally:
        independent._raw_dgs_reduction_cache.staging.cleanup()
        data_workspace.set_project_temporary_directory(legacy_project, None)


def test_copied_project_save_as_preserves_original_path_compatibility(tmp_path):
    import copy

    from nfit.project_io import NfitProject

    selected = tmp_path / "selected"
    selected.mkdir()
    old, new = tmp_path / "old.nfit", tmp_path / "new.nfit"
    original = NfitProject()
    original._project_path = old
    data_workspace.set_project_temporary_directory(original, str(selected))
    independent = copy.deepcopy(original)
    try:
        with project_data_workspace(independent, new):
            pass
        for path in (old, new):
            with temporary_data_directory(path, prefix="legacy-") as name:
                assert Path(name).is_relative_to(selected)
        with pytest.raises(RuntimeError):
            with project_data_workspace(original, new):
                original._project_path = new
                raise RuntimeError("failed late save")
        assert original._project_path == old
    finally:
        data_workspace.set_project_temporary_directory(original, None)
        independent._project_path = new
        data_workspace.set_project_temporary_directory(independent, None)
