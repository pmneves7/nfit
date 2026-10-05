"""Private staging, reference retirement, and crash cleanup remain conservative."""

import ast
import gc
import json
import os
import socket
from pathlib import Path

import psutil
import pytest

from nfit import workspace_sessions
from nfit.data_workspace import (
    DataWorkspaceError,
    cleanup_temporary_workspaces,
    inspect_temporary_workspaces,
    temporary_data_directory,
    temporary_workspace_path,
)


def _record(owner, name, *, hostname=None, pid=999_999_999, creation=1.0):
    path = temporary_workspace_path(owner) / name
    path.mkdir(parents=True)
    (path / "session.json").write_text(json.dumps({
        "version": 1, "owner_path": str(owner.absolute()),
        "uid": workspace_sessions._effective_uid(),
        "user_identity": workspace_sessions._user_identity(),
        "hostname": socket.gethostname() if hostname is None else hostname,
        "pid": pid, "process_creation_time": creation,
    }))
    (path / "payload.bin").write_bytes(b"scientific staging")
    return path


def test_lookup_and_inspection_create_no_storage(tmp_path):
    owner = tmp_path / "project.nfit"
    assert temporary_workspace_path(owner).is_relative_to(tmp_path / ".nfit-work")
    assert inspect_temporary_workspaces(owner) == ()
    assert cleanup_temporary_workspaces(owner) == ()
    assert not list(tmp_path.iterdir())


def test_owners_have_separate_sessions_and_cleanup_waits_for_last_reference(tmp_path):
    first = tmp_path / "first.nfit"
    second = tmp_path / "second.nfit"
    a = temporary_data_directory(first, prefix="nfit-data-")
    b = temporary_data_directory(first, prefix="nfit-events-")
    c = temporary_data_directory(second, prefix="nfit-data-")
    assert Path(a.name).parent == Path(b.name).parent
    assert Path(a.name).parent != Path(c.name).parent
    assert all(info.state == "live" for info in inspect_temporary_workspaces(first))
    assert cleanup_temporary_workspaces(first) == ()
    a.cleanup()
    assert Path(b.name).exists()
    b.cleanup()
    assert not temporary_workspace_path(first).exists()
    assert Path(c.name).exists()
    del c
    gc.collect()
    assert not (tmp_path / ".nfit-work").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX UID and directory permission semantics")
def test_shared_container_is_traversable_with_separate_private_uid_owners(tmp_path, monkeypatch):
    owner_path = tmp_path / "shared-project.nfit"
    with monkeypatch.context() as user_a:
        user_a.setattr(workspace_sessions.os, "geteuid", lambda: 1001)
        first = temporary_data_directory(owner_path, prefix="nfit-data-")
        first_root = temporary_workspace_path(owner_path)
    with monkeypatch.context() as user_b:
        user_b.setattr(workspace_sessions.os, "geteuid", lambda: 1002)
        second = temporary_data_directory(owner_path, prefix="nfit-data-")
        second_root = temporary_workspace_path(owner_path)
    assert first_root != second_root
    top_mode = (tmp_path / ".nfit-work").stat().st_mode
    assert top_mode & 0o1777 == 0o1777
    assert first_root.stat().st_mode & 0o777 == 0o700
    assert second_root.stat().st_mode & 0o777 == 0o700
    assert Path(first.name).parent.stat().st_mode & 0o777 == 0o700
    first.cleanup()
    assert Path(second.name).exists()
    second.cleanup()
    assert not (tmp_path / ".nfit-work").exists()


def test_existing_top_container_permissions_are_not_modified(tmp_path):
    top = tmp_path / ".nfit-work"
    top.mkdir(mode=0o750)
    original = top.stat().st_mode & 0o7777
    directory = temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-data-")
    assert top.stat().st_mode & 0o7777 == original
    directory.cleanup()


@pytest.mark.skipif(os.name != "posix", reason="POSIX umask and directory permission semantics")
def test_private_owner_and_metadata_modes_survive_restrictive_umask(tmp_path):
    previous = os.umask(0o777)
    try:
        directory = temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-data-")
    finally:
        os.umask(previous)
    session = Path(directory.name).parent
    assert session.stat().st_mode & 0o777 == 0o700
    assert session.parent.stat().st_mode & 0o777 == 0o700
    assert (session / "session.json").stat().st_mode & 0o777 == 0o600
    directory.cleanup()


def test_windows_mode_accepts_existing_folders_and_separates_home_users(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_sessions, "_POSIX_PERMISSIONS", False)
    monkeypatch.delattr(workspace_sessions.os, "geteuid", raising=False)
    monkeypatch.delattr(workspace_sessions.os, "getuid", raising=False)
    owner = tmp_path / "shared.nfit"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-a"))
    first_root = temporary_workspace_path(owner)
    first_root.mkdir(parents=True, mode=0o777)
    # Normal Windows folders can report group/other bits. These are not an
    # indication of their ACL privacy and must not prevent subsequent sessions.
    first_root.chmod(0o777)
    first = temporary_data_directory(owner, prefix="nfit-data-")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-b"))
    second_root = temporary_workspace_path(owner)
    second = temporary_data_directory(owner, prefix="nfit-data-")
    assert first_root != second_root
    assert Path(first.name).is_relative_to(first_root)
    assert Path(second.name).is_relative_to(second_root)
    assert json.loads((Path(second.name).parent / "session.json").read_text())["user_identity"].startswith("home-")
    first.cleanup()
    second.cleanup()
    assert not (tmp_path / ".nfit-work").exists()


def test_dead_local_session_cleaned_but_live_foreign_unknown_and_legacy_preserved(tmp_path, monkeypatch):
    owner = tmp_path / "project.nfit"
    dead = _record(owner, "session-dead")
    live = _record(owner, "session-live", pid=os.getpid(),
                   creation=psutil.Process().create_time())
    foreign = _record(owner, "session-foreign", hostname="another-analysis-node")
    unknown = _record(owner, "session-unknown")
    (unknown / "session.json").write_text("not a manifest")
    legacy = tmp_path / "nfit-reduced-events-before-markers"
    legacy.mkdir()
    states = {record.path: record.state for record in inspect_temporary_workspaces(owner, include_legacy=True)}
    assert states == {dead: "orphaned", live: "live", foreign: "foreign",
                      unknown: "unknown", legacy: "legacy"}
    assert cleanup_temporary_workspaces(owner) == (dead,)
    assert all(path.exists() for path in (live, foreign, unknown, legacy))


def test_pid_reuse_is_distinguished_from_the_original_process(tmp_path, monkeypatch):
    owner = tmp_path / "project.nfit"
    stale = _record(owner, "session-reused-pid", pid=os.getpid(),
                    creation=psutil.Process().create_time() - 100)
    info, = inspect_temporary_workspaces(owner)
    assert info.removable
    assert "reused" in info.reason
    assert cleanup_temporary_workspaces(owner) == (stale,)
    assert not (tmp_path / ".nfit-work").exists()


def test_unknown_process_access_and_owner_mismatch_are_protected(tmp_path, monkeypatch):
    owner = tmp_path / "project.nfit"
    inaccessible = _record(owner, "session-access-denied")
    wrong_owner = _record(owner, "session-other-owner")
    metadata = json.loads((wrong_owner / "session.json").read_text())
    metadata["owner_path"] = str(tmp_path / "different.nfit")
    (wrong_owner / "session.json").write_text(json.dumps(metadata))

    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    assert all(record.state == "unknown" for record in inspect_temporary_workspaces(owner))
    assert cleanup_temporary_workspaces(owner) == ()
    assert inaccessible.exists() and wrong_owner.exists()


def test_cleanup_can_select_one_confirmed_dead_session(tmp_path):
    owner = tmp_path / "project.nfit"
    first = _record(owner, "session-first")
    second = _record(owner, "session-second")
    assert cleanup_temporary_workspaces(owner, session_paths=[first]) == (first,)
    assert second.exists()


def test_inherited_owner_cannot_remove_original_process_staging(tmp_path, monkeypatch):
    owner = temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-events-")
    process = os.getpid()
    with monkeypatch.context() as child:
        child.setattr(workspace_sessions.os, "getpid", lambda: process + 1)
        owner.cleanup()
        assert Path(owner.name).exists()
    owner.cleanup()
    assert not (tmp_path / ".nfit-work").exists()


def test_cleanup_rechecks_process_identity_instead_of_trusting_old_inspection(tmp_path):
    owner = tmp_path / "project.nfit"
    path = _record(owner, "session-recheck")
    assert inspect_temporary_workspaces(owner)[0].removable
    metadata = json.loads((path / "session.json").read_text())
    metadata.update(pid=os.getpid(), process_creation_time=psutil.Process().create_time())
    (path / "session.json").write_text(json.dumps(metadata))
    assert cleanup_temporary_workspaces(owner, session_paths=[path]) == ()
    assert path.exists()


def test_cleanup_does_not_follow_symbolic_links_or_accept_unowned_paths(tmp_path):
    owner = tmp_path / "project.nfit"
    elsewhere = tmp_path / "unrelated-data"
    elsewhere.mkdir()
    root = temporary_workspace_path(owner)
    root.mkdir(parents=True)
    (root / "session-link").symlink_to(elsewhere, target_is_directory=True)
    assert cleanup_temporary_workspaces(owner, session_paths=[elsewhere, root / "session-link"]) == ()
    assert elsewhere.exists()
    assert (root / "session-link").is_symlink()


def test_managed_root_cannot_redirect_data_to_another_location(tmp_path):
    elsewhere = tmp_path / "external"
    elsewhere.mkdir()
    (tmp_path / ".nfit-work").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(DataWorkspaceError):
        temporary_data_directory(tmp_path / "project.nfit", prefix="nfit-data-")
    assert not list(elsewhere.iterdir())


def test_session_owner_markers_do_not_make_the_service_gui_dependent():
    tree = ast.parse(Path(workspace_sessions.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("PySide", "PyQt", "project_gui")) for name in imports)
    assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) for node in tree.body)
