"""System clipboard presentation for independent nfit project windows."""

from __future__ import annotations

import atexit
import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from .data_workspace import temporary_data_directory
from .operation_control import operation_progress
from .project_clipboard import ProjectClipboardPayload
from .project_transfer import (
    TRANSFER_MIME,
    TRANSFER_SCHEMA,
    TRANSFER_VERSION,
    adopt_project_selection,
    paste_project_selection,
    prepare_project_selection,
    project_selection_script,
    read_project_selection,
    save_project_selection,
    selection_manifest,
    selection_needs_archive,
    write_project_selection,
)

_MAX_DESCRIPTOR_BYTES = 64 * 1024**2


def _install_clipboard_cleanup():
    app = QtWidgets.QApplication.instance()
    if getattr(app, "_nfit_clipboard_cleanup", False):
        return

    def clear_owned_data():
        # Qt owns QMimeData after setMimeData. Release it while Qt is alive;
        # PySide's offscreen platform otherwise crashes during interpreter exit.
        if isValid(app):
            clipboard = app.clipboard()
            mime = clipboard.mimeData()
            stamp = getattr(app, "_nfit_clipboard_stamp", None)
            if mime is not None and mime.hasFormat(TRANSFER_MIME) and (
                app.platformName() == "offscreen" or stamp is not None and bytes(mime.data(TRANSFER_MIME)) == stamp
            ):
                clipboard.clear()

    app.aboutToQuit.connect(clear_owned_data)
    atexit.register(clear_owned_data)
    app._nfit_clipboard_cleanup = True


def clipboard_descriptor():
    """Read a bounded declarative clipboard; ordinary text is not executable."""
    mime = QtWidgets.QApplication.clipboard().mimeData()
    if mime is None or not mime.hasFormat(TRANSFER_MIME):
        return None
    data = bytes(mime.data(TRANSFER_MIME))
    if len(data) > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("The project clipboard descriptor is too large")
    payload = json.loads(data)
    if not isinstance(payload, dict) or payload.get("schema") != TRANSFER_SCHEMA or payload.get("version") != TRANSFER_VERSION:
        raise ValueError("Unsupported project clipboard format")
    return payload


def clipboard_preview():
    """Inspect type compatibility without loading any numerical arrays."""
    try:
        descriptor = clipboard_descriptor()
        return None if descriptor is None else ProjectClipboardPayload(descriptor["kind"], ())
    except (KeyError, TypeError, ValueError):
        return None


def publish_project_clipboard(explorer, role, objects, *, collect_binnings, has_binnings):
    group = explorer._objects_for_item(explorer._current_item())[0]
    selection = prepare_project_selection(explorer.project, role, objects, source_group=group)
    needs_file = selection_needs_archive(selection) or has_binnings(selection.source_project)

    def publish(descriptor, owner=None):
        _install_clipboard_cleanup()
        mime = QtCore.QMimeData()
        encoded = json.dumps(descriptor).encode("utf-8")
        QtWidgets.QApplication.instance()._nfit_clipboard_stamp = encoded
        mime.setData(TRANSFER_MIME, encoded)
        mime.setText(f"nfit copied {selection.payload.kind}: {len(selection.payload.items)} objects")
        QtWidgets.QApplication.clipboard().setMimeData(mime)
        previous = getattr(explorer, "_project_clipboard_owner", None)
        explorer._project_clipboard_owner = owner
        explorer._project_clipboard_stamp = encoded
        if previous is not None:
            previous.cleanup()

    descriptor = {"schema": TRANSFER_SCHEMA, "version": TRANSFER_VERSION, "kind": selection.payload.kind}
    if not needs_file:
        path = explorer.project_path or Path.cwd() / ".nfit-clipboard"
        publish({**descriptor, "manifest": selection_manifest(selection, path)})
        return
    owner_path = explorer.project_path
    if owner_path is None:
        for root in selection.project.data_groups:
            for item in root.iter_datasets():
                value = item.metadata.get("_project_path") or item.metadata.get("source_file")
                if value and Path(value).is_absolute() and Path(value).parent.is_dir():
                    owner_path = Path(value)
                    break
    if owner_path is None:
        from .file_dialogs import get_existing_directory

        directory = get_existing_directory(explorer.window, "Storage for copied scientific data", "")
        if not directory:
            return
        owner_path = Path(directory) / ".nfit-copied-selection"

    def task(_report):
        owner = temporary_data_directory(owner_path, prefix="nfit-copied-selection-")
        try:
            with operation_progress(_report):
                path = write_project_selection(selection, Path(owner.name) / "selection.nfit",
                                               collect_binnings=collect_binnings)
            stat = path.stat()
            return ({**descriptor, "archive": str(path), "size": stat.st_size,
                     "modified_ns": stat.st_mtime_ns}, owner)
        except BaseException:
            owner.cleanup()
            raise

    if explorer._interactive:
        explorer._start_background_task(title="Copying project selection…", failure_title="Copy project selection",
                                        task=task, on_success=lambda result: publish(*result),
                                        success_message="Project selection copied.")
    else:
        publish(*task(None))


def paste_project_clipboard(explorer, *, target_role, data_group, dataset_node, dataset,
                            install_binnings, on_success):
    descriptor = clipboard_descriptor()
    if descriptor is None:
        return False

    def task(_report):
        if "archive" in descriptor:
            path = Path(descriptor["archive"])
            if not path.is_absolute():
                raise ValueError("A copied project archive must have an absolute local path")
            stat = path.stat()
            if stat.st_size != descriptor["size"] or stat.st_mtime_ns != descriptor["modified_ns"]:
                raise OSError("The copied project archive changed; copy the selection again")
            owner_path = explorer.project_path or path
            with operation_progress(_report):
                return adopt_project_selection(path, owner_path)
        return read_project_selection(manifest=descriptor["manifest"])

    def installed(selection):
        try:
            result = paste_project_selection(selection, explorer.project, target_role=target_role,
                                             data_group=data_group, dataset_node=dataset_node, dataset=dataset,
                                             install_binnings=install_binnings)
        except (KeyError, OSError, TypeError, ValueError) as error:
            if selection.owner is not None and selection.owner not in getattr(explorer.project, "_project_transfer_owners", []):
                selection.owner.cleanup()
            QtWidgets.QMessageBox.information(explorer.window, "Cannot paste here", str(error))
            return
        on_success(result, selection.payload)

    if explorer._interactive and "archive" in descriptor:
        explorer._start_background_task(title="Pasting project selection…", failure_title="Paste project selection",
                                        task=task, on_success=installed, success_message="Project selection pasted.")
    else:
        installed(task(None))
    return True


def release_project_clipboard(explorer):
    """Retire this window's disposable packet without touching another copy."""
    stamp = getattr(explorer, "_project_clipboard_stamp", None)
    clipboard = QtWidgets.QApplication.clipboard()
    mime = clipboard.mimeData()
    if stamp is not None and mime is not None and bytes(mime.data(TRANSFER_MIME)) == stamp:
        clipboard.clear()
    owner = getattr(explorer, "_project_clipboard_owner", None)
    if owner is not None:
        owner.cleanup()
        explorer._project_clipboard_owner = None


def save_clipboard_script(explorer):
    """Save a durable selection and its editable, GUI-independent paste script."""
    from .file_dialogs import get_save_file_name

    try:
        descriptor = clipboard_descriptor()
        if descriptor is None:
            QtWidgets.QMessageBox.information(explorer.window, "Save copy/paste script", "Copy a project-tree selection first.")
            return
        filename, _filter = get_save_file_name(explorer.window, "Save copy/paste script", "copy_project_items.py", "Python scripts (*.py)")
        if not filename:
            return
        script_path = Path(filename).with_suffix(".py").absolute()
        packet_path = script_path.with_suffix(".selection.nfit")

        def task(report):
            with operation_progress(report):
                selection = read_project_selection(descriptor["archive"]) if "archive" in descriptor else read_project_selection(manifest=descriptor["manifest"])
                save_project_selection(selection, packet_path)
                script_path.write_text(project_selection_script(packet_path, kind=selection.payload.kind), encoding="utf-8")

        if explorer._interactive:
            explorer._start_background_task(title="Saving copy/paste script…", failure_title="Save copy/paste script",
                                            task=task, on_success=lambda _result: None,
                                            success_message="Selection and script saved.")
        else:
            task(None)
    except (KeyError, OSError, TypeError, ValueError) as error:
        QtWidgets.QMessageBox.warning(explorer.window, "Save copy/paste script", str(error))
