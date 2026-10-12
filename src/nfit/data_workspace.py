"""Lazy, operation-owned storage for scientific temporary files.

File-backed operations stage beside their owner rather than selecting a mount
at application startup. No cluster names, instrument paths or global active
project are needed. In-memory callers without an owner retain Python's normal
temporary-directory policy.
"""

from __future__ import annotations

import os
import tempfile
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .workspace_sessions import (
    SessionTemporaryDirectory,
    WorkspaceSessionInfo,
    cleanup_owner_workspaces,
    inspect_owner_workspaces,
    owner_temporary_directory,
    owner_workspace_path,
)

# Saved-path callers retain the established last-selected directory behavior.
# Contextual scientific operations always use their owning runtime preference.
_WORKSPACE_OVERRIDES: dict[str, str] = {}


@dataclass(eq=False)
class _WorkspacePreference:
    directory: str | None = None


def _register_preference(preference, owner):
    if owner is not None:
        identity = str(Path(owner).absolute())
        if preference.directory:
            _WORKSPACE_OVERRIDES[identity] = preference.directory
        else:
            _WORKSPACE_OVERRIDES.pop(identity, None)


def _path_directory(owner):
    return _WORKSPACE_OVERRIDES.get(str(owner))


def _preference(workspace):
    preference = getattr(workspace, "_data_workspace_preference", None)
    if preference is None and hasattr(workspace, "settings"):
        directory = workspace.settings.get("temporary_storage_directory")
        preference = _WorkspacePreference(str(Path(directory).expanduser().absolute()) if directory else None)
        workspace._data_workspace_preference = preference
    return preference


def _attach_preference(group, preference):
    if preference is not None:
        for owner in (group, *group.iter_subgroups(), *group.iter_datasets()):
            owner._data_workspace_preference = preference


_TEMPORARY_DIRECTORIES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_TEMPORARY_FILES: dict[str, dict[str, int]] = {}
_STORAGE_LOCK = RLock()


def record_temporary_file(path, byte_count=None):
    """Record completed staging on its writer thread, avoiding GUI disk scans."""
    target = Path(path).absolute()
    with _STORAGE_LOCK:
        root = next((name for name in tuple(_TEMPORARY_DIRECTORIES)
                     if target.is_relative_to(Path(name))), None)
        if root is not None:
            size = target.stat().st_size if byte_count is None else int(byte_count)
            _TEMPORARY_FILES.setdefault(root, {})[str(target)] = size


def set_project_temporary_directory(project, directory: str | None, *, validate=True) -> None:
    """Choose storage for future allocations without moving existing files."""
    if directory:
        root = Path(directory).expanduser().absolute()
        if validate and (not root.is_dir() or not os.access(root, os.W_OK)):
            raise DataWorkspaceError(f"Choose an existing writable directory: {root}")
        directory = str(root)
    if hasattr(project, "settings"):
        project.settings["temporary_storage_directory"] = directory or ""
    preference = _preference(project)
    if preference is None:
        preference = project._data_workspace_preference = _WorkspacePreference()
    preference.directory = directory or None
    _register_preference(preference, getattr(project, "_project_path", None))
    for group in project.data_groups:
        _attach_preference(group, preference)


def temporary_storage_usage() -> tuple[tuple[str, int], ...]:
    """List nfit-owned live staging directories, without scanning other data."""
    with _STORAGE_LOCK:
        live = {name for name, owner in tuple(_TEMPORARY_DIRECTORIES.items())
                if owner._finalizer.alive}
        for name in tuple(_TEMPORARY_FILES):
            if name not in live:
                _TEMPORARY_FILES.pop(name, None)
        return tuple((name, sum(_TEMPORARY_FILES.get(name, {}).values())) for name in live)


class DataWorkspaceError(OSError):
    """An operation's selected scientific storage is unavailable."""


def bind_data_workspace(group, path, *, workspace=None):
    """Attach runtime project ownership to a collection and its source entries."""
    _attach_preference(group, _preference(workspace if workspace is not None else group))
    for node in (group, *group.iter_subgroups()):
        node._data_workspace_owner_path = str(path)
    for dataset in group.iter_datasets():
        dataset.metadata["_project_path"] = str(path)


def inherit_temporary_storage(group, parent):
    """Share future allocation preferences without changing source ownership."""
    _attach_preference(group, _preference(parent))


def inherit_data_workspace(group, parent):
    """Keep newly imported runs in the existing project's scientific storage."""
    inherit_temporary_storage(group, parent)
    owner = getattr(parent, "_data_workspace_owner_path", None) or getattr(parent, "_project_path", None)
    if owner is not None:
        bind_data_workspace(group, owner)


@contextmanager
def project_data_workspace(project, path):
    """Use the selected save destination while preparing project caches.

    Dataset ownership is explicit, so concurrent projects and reduction workers
    do not depend on a process-global current project or temporary directory.
    Failed saves restore the previous runtime ownership metadata.
    """

    missing = object()
    datasets = [dataset for group in project.data_groups for dataset in group.iter_datasets()]
    nodes = [node for group in project.data_groups for node in (group, *group.iter_subgroups())]
    owners = [project, *nodes, *datasets]
    previous_preferences = [getattr(owner, "_data_workspace_preference", missing) for owner in owners]
    previous = [dataset.metadata.get("_project_path", missing) for dataset in datasets]
    previous_nodes = [getattr(node, "_data_workspace_owner_path", missing) for node in nodes]
    previous_project_path = getattr(project, "_project_path", missing)
    preference = _preference(project)
    previous_directory = None if preference is None else preference.directory
    if preference is not None and hasattr(project, "settings"):
        selected = project.settings.get("temporary_storage_directory")
        preference.directory = str(Path(selected).expanduser().absolute()) if selected else None
    for group in project.data_groups:
        bind_data_workspace(group, path, workspace=project)
    try:
        yield
    except BaseException:
        if preference is not None:
            preference.directory = previous_directory
        if previous_project_path is missing:
            project.__dict__.pop("_project_path", None)
        else:
            project._project_path = previous_project_path
        for dataset, owner in zip(datasets, previous, strict=True):
            if owner is missing:
                dataset.metadata.pop("_project_path", None)
            else:
                dataset.metadata["_project_path"] = owner
        for node, owner in zip(nodes, previous_nodes, strict=True):
            if owner is missing:
                del node._data_workspace_owner_path
            else:
                node._data_workspace_owner_path = owner
        for owner, previous_preference in zip(owners, previous_preferences, strict=True):
            if previous_preference is missing:
                owner.__dict__.pop("_data_workspace_preference", None)
            else:
                owner._data_workspace_preference = previous_preference
        raise
    else:
        if preference is not None:
            _register_preference(preference, path)


def temporary_data_directory(
    owner_path: str | os.PathLike[str] | None,
    *,
    prefix: str,
    workspace=None,
) -> tempfile.TemporaryDirectory | SessionTemporaryDirectory:
    """Allocate staging in an owner's hidden, private session workspace.

    The parent must already exist. Unavailable storage fails at the operation;
    it never silently spills file-backed scientific data into system scratch.
    The returned object owns cleanup, including retained reduced-event caches.
    ``workspace`` selects runtime project/group/dataset preferences, including
    an explicit default choice; path-only calls retain saved-path compatibility.
    """

    owner = None if owner_path is None else Path(owner_path).absolute()
    preference = _preference(workspace) if workspace is not None else None
    override = (None if preference is None else preference.directory)
    if workspace is None and owner is not None:
        override = _path_directory(owner)
    parent = Path(override) if override else (None if owner is None else owner.parent)
    if owner is None and parent is not None:
        # A selected directory also governs source-free scientific operations.
        owner = parent / ".nfit-unsaved-project"
    if parent is not None and not parent.is_dir():
        raise DataWorkspaceError(
            f"Scientific workspace directory is unavailable: {parent}. "
            "Choose an accessible project or output directory."
        )
    try:
        directory = (tempfile.TemporaryDirectory(prefix=prefix) if owner is None
                     else owner_temporary_directory(owner, parent, prefix=prefix))
        with _STORAGE_LOCK:
            _TEMPORARY_DIRECTORIES[directory.name] = directory
        return directory
    except OSError as exc:
        raise DataWorkspaceError(
            f"Cannot create scientific workspace in {parent or 'system temporary storage'}. "
            "Choose a writable project or output directory before retrying."
        ) from exc


def _workspace_owner_parent(owner_path, directory=None) -> tuple[Path, Path]:
    if owner_path is None:
        raise ValueError("An owner path is required for scientific workspace inspection")
    owner = Path(owner_path).absolute()
    parent = (Path(directory).expanduser().absolute() if directory is not None
              else Path(_path_directory(owner) or owner.parent))
    return owner, parent


def temporary_workspace_path(owner_path, *, directory=None) -> Path:
    """Return the hidden workspace location without creating it."""

    return owner_workspace_path(*_workspace_owner_parent(owner_path, directory))


def inspect_temporary_workspaces(owner_path, *, directory=None, include_legacy=False) -> tuple[WorkspaceSessionInfo, ...]:
    """Inspect ownership without decoding data, creating folders, or deleting.

    Old flat ``nfit-*`` directories have no reliable process identity. Optional
    legacy records flag them for manual review; they are never eligible for
    automatic cleanup. Sessions from other hosts remain protected too.
    """

    owner, parent = _workspace_owner_parent(owner_path, directory)
    records = inspect_owner_workspaces(owner, parent)
    if include_legacy and parent.is_dir():
        records += tuple(WorkspaceSessionInfo(path, None, None, None, "legacy",
                         "No session ownership marker; manual review required")
                         for path in sorted(parent.iterdir())
                         if path.name.startswith("nfit-") and path.is_dir())
    return records


def cleanup_temporary_workspaces(owner_path, *, directory=None, session_paths=None) -> tuple[Path, ...]:
    """Remove confirmed dead local sessions; live/foreign/legacy files remain.

    ``session_paths`` optionally narrows cleanup to inspected session paths.
    Ownership is rechecked immediately before removal. Scientific data in
    current sessions is retained until its last cache or reader releases it.
    """

    return cleanup_owner_workspaces(*_workspace_owner_parent(owner_path, directory), paths=session_paths)
