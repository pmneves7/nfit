"""Lazy, operation-owned storage for scientific temporary files.

File-backed operations stage beside their owner rather than selecting a mount
at application startup. No cluster names, instrument paths or global active
project are needed. In-memory callers without an owner retain Python's normal
temporary-directory policy.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


class DataWorkspaceError(OSError):
    """An operation's selected scientific storage is unavailable."""


def bind_data_workspace(group, path):
    """Attach runtime project ownership to a collection and its source entries."""
    for node in (group, *group.iter_subgroups()):
        node._data_workspace_owner_path = str(path)
    for dataset in group.iter_datasets():
        dataset.metadata["_project_path"] = str(path)


def inherit_data_workspace(group, parent):
    """Keep newly imported runs in the existing project's scientific storage."""
    owner = getattr(parent, "_data_workspace_owner_path", None)
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
    previous = [dataset.metadata.get("_project_path", missing) for dataset in datasets]
    nodes = [node for group in project.data_groups for node in (group, *group.iter_subgroups())]
    previous_nodes = [getattr(node, "_data_workspace_owner_path", missing) for node in nodes]
    for group in project.data_groups:
        bind_data_workspace(group, path)
    try:
        yield
    except BaseException:
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
        raise


def temporary_data_directory(
    owner_path: str | os.PathLike[str] | None,
    *,
    prefix: str,
) -> tempfile.TemporaryDirectory:
    """Allocate staging beside a project, output, or unsaved source file.

    The parent must already exist. Unavailable storage fails at the operation;
    it never silently spills file-backed scientific data into system scratch.
    The returned object owns cleanup, including retained reduced-event caches.
    """

    parent = None if owner_path is None else Path(owner_path).absolute().parent
    if parent is not None and not parent.is_dir():
        raise DataWorkspaceError(
            f"Scientific workspace directory is unavailable: {parent}. "
            "Choose an accessible project or output directory."
        )
    try:
        return tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
    except OSError as exc:
        raise DataWorkspaceError(
            f"Cannot create scientific workspace in {parent or 'system temporary storage'}. "
            "Choose a writable project or output directory before retrying."
        ) from exc
