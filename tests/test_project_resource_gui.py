from __future__ import annotations

import gc
import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry
from nfit.project_io import NfitProject
from nfit.qt_resource_jobs import ResourceJobCancelled, run_resource_job
from nfit.rebin_cache import RebinCache
from nfit.resource_budget import ResourceLimitError, reserve_memory, snapshot_memory


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application
    guard = getattr(application, "_nfit_operation_input_filter", None)
    assert guard is None or guard.depth == 0


@pytest.fixture
def explorer(app, monkeypatch):
    from nfit import (
        electronic_structure,
        project_caches,
        project_composites,
        project_data,
        project_gui,
    )

    viewer_cache, composite_cache = RebinCache(), RebinCache()
    monkeypatch.setattr(project_data, "_VIEWER_VIEW_CACHE", viewer_cache)
    monkeypatch.setattr(project_gui, "_VIEWER_VIEW_CACHE", viewer_cache)
    monkeypatch.setattr(project_composites, "_COMPOSITE_DATA_CACHE", composite_cache)
    monkeypatch.setattr(project_data, "_COMPOSITE_DATA_CACHE", composite_cache)
    monkeypatch.setattr(project_gui, "_COMPOSITE_DATA_CACHE", composite_cache)
    monkeypatch.setattr(project_data, "_PREPARED_POINT_LIST_CACHE", OrderedDict())
    monkeypatch.setattr(project_caches, "MODEL_OVERLAY_CACHE", OrderedDict())
    monkeypatch.setattr(electronic_structure, "_FOURIER_COEFFICIENT_CACHE", OrderedDict())
    monkeypatch.setattr(electronic_structure, "_HAMILTONIAN_COMPONENT_CACHE", OrderedDict())
    dataset = DatasetEntry("Sample", None)
    group = DataGroup("Workspace", datasets=[dataset])
    value = project_gui.NfitProjectExplorer(NfitProject([group]))
    monkeypatch.setattr(value, "_confirm_save_before_closing_project", lambda: True)
    state = SimpleNamespace(
        explorer=value, dataset=dataset, group=group, viewer_cache=viewer_cache,
        composite_cache=composite_cache,
    )
    yield state
    value._interactive = False
    value._allow_window_close = True
    from nfit.project_resource_gui import retire_project_resources

    value._close_all_slice_viewers()
    retire_project_resources(value)
    value.window.close()
    viewer_cache.clear()
    composite_cache.clear()
    app.processEvents()


def cube():
    return MDHistoData(
        axes=(
            MDHistoAxis("H", np.arange(5.0), "rlu", "momentum"),
            MDHistoAxis("E", np.arange(5.0), "meV", "energy"),
        ),
        signal=np.ones((4, 4)), errors=np.ones((4, 4)),
        mask=np.zeros((4, 4), dtype=bool), num_events=np.ones((4, 4)),
        metadata={"source_dataset_name": "Sample", "binning_name": "Default"},
    )


def test_resource_job_runs_on_worker_and_processes_gui_timer(app):
    parent = QtWidgets.QWidget()
    gate = threading.Event()
    gui_thread = threading.get_ident()
    gui_ticks = []
    progress_threads = []

    def timer_tick():
        gui_ticks.append(threading.get_ident())
        gate.set()

    def task(report):
        worker_thread = threading.get_ident()
        if not gate.wait(timeout=2):
            raise RuntimeError("GUI event loop did not run")
        report({"message": "Worker completed"})
        return worker_thread

    QtCore.QTimer.singleShot(0, timer_tick)
    actual = run_resource_job(parent, "Test loading", task, on_progress=lambda event: progress_threads.append(threading.get_ident()))
    assert actual != gui_thread
    assert gui_ticks == [gui_thread]
    assert progress_threads == [gui_thread]
    assert not any(isinstance(child, QtWidgets.QProgressDialog) and child.isVisible() for child in parent.children())
    parent.close()


@pytest.mark.parametrize("emit_progress", [False, True])
def test_resource_job_survives_transient_parent_deletion(app, monkeypatch, emit_progress):
    from shiboken6 import isValid

    parent = QtWidgets.QWidget()
    deleted = threading.Event()
    emitted = threading.Event()
    continue_worker = threading.Event()
    watched_threads = []
    watchdog_fired = []
    original_thread_class = QtCore.QThread

    class ObservedThread(original_thread_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            watched_threads.append(self)

    monkeypatch.setattr(QtCore, "QThread", ObservedThread)
    parent.destroyed.connect(lambda *_args: deleted.set())

    def task(report):
        if not deleted.wait(timeout=2):
            raise RuntimeError("Transient parent was not destroyed")
        if emit_progress:
            report({"message": "Progress after parent deletion"})
            emitted.set()
            if not continue_worker.wait(timeout=2):
                raise RuntimeError("GUI did not process the deleted-parent progress")
            report({"message": "Final progress"})
        return "completed"

    # A vanished completion receiver used to strand the nested GUI loop. The
    # watchdog also retires that worker's idle event loop, so this regression
    # fails with a bounded error rather than hanging the entire test suite.
    watchdog = QtCore.QTimer()
    watchdog.setSingleShot(True)

    def time_out():
        watchdog_fired.append(True)
        continue_worker.set()
        for thread in watched_threads:
            if isValid(thread):
                thread.quit()

    watchdog.timeout.connect(time_out)
    watchdog.start(3000)
    progress_timer = QtCore.QTimer()
    progress_timer.setInterval(5)

    def acknowledge_progress():
        if emitted.is_set():
            continue_worker.set()

    progress_timer.timeout.connect(acknowledge_progress)
    progress_timer.start()
    QtCore.QTimer.singleShot(0, parent.deleteLater)
    try:
        try:
            result = run_resource_job(parent, "Transient parent", task)
        except ResourceJobCancelled:
            result = "cancelled"
    finally:
        watchdog.stop()
        progress_timer.stop()
    assert result in {"completed", "cancelled"}
    assert deleted.is_set() and not isValid(parent)
    assert not watchdog_fired
    assert all(not thread.isRunning() for thread in watched_threads if isValid(thread))
    assert app._nfit_operation_input_filter.depth == 0


@pytest.mark.parametrize("memory_error", [False, True])
def test_resource_job_preserves_error_type_and_cleans_reservations(app, monkeypatch, memory_error):
    from nfit import resource_budget

    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: 100_000)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 100_000)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    parent = QtWidgets.QWidget()
    error = (
        ResourceLimitError(requested_bytes=200_000, used_bytes=0, limit_bytes=100_000, operation="Load")
        if memory_error else ValueError("Load failed")
    )

    def task(_report):
        with reserve_memory(1000):
            raise error

    with pytest.raises(type(error)) as caught:
        run_resource_job(parent, "Test failure", task)
    assert caught.value is error
    assert snapshot_memory().reserved_bytes == 0
    assert app._nfit_operation_input_filter.depth == 0
    assert not any(isinstance(child, QtWidgets.QProgressDialog) and child.isVisible() for child in parent.children())
    parent.close()


def test_cancellation_before_real_archive_publication_keeps_original(app, tmp_path, monkeypatch):
    from nfit import project_archive

    path = tmp_path / "original.nfit"
    project_archive.write_project_manifest(path, {"version": "original"})
    original = path.read_bytes()
    original_report = project_archive.report_operation
    acknowledged = threading.Event()
    parent = QtWidgets.QWidget()

    def pause_at_publication(message, **kwargs):
        if message.startswith("Publishing"):
            original_report("Ready for publication")
            if not acknowledged.wait(timeout=2):
                raise RuntimeError("GUI did not acknowledge publication checkpoint")
        original_report(message, **kwargs)

    monkeypatch.setattr(project_archive, "report_operation", pause_at_publication)

    def progress(event):
        if event["message"] == "Ready for publication":
            dialog = next(child for child in parent.children() if isinstance(child, QtWidgets.QProgressDialog))
            dialog.canceled.emit()
            acknowledged.set()

    with pytest.raises(ResourceJobCancelled):
        run_resource_job(
            parent, "Save project", lambda _report: project_archive.write_project_manifest(path, {"version": "replacement"}),
            on_progress=progress,
        )
    assert acknowledged.is_set()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".original.nfit-*.tmp"))
    parent.close()


def test_resources_toolbar_opens_modeless_window_and_wires_settings(explorer, monkeypatch, tmp_path):
    from nfit import performance

    value = explorer.explorer
    saved = []
    monkeypatch.setattr(performance, "save_resource_limits", lambda **limits: saved.append(limits))
    assert value.resource_manager_button.toolTip()
    value.resource_manager_button.click()
    manager = value._resource_manager_window
    assert manager.isVisible() and not manager.isModal()
    assert value.open_resource_manager() is manager
    assert manager.temporary_directory.text() == ""
    assert manager.temporary_directory.placeholderText() == "Alongside project or source file"
    manager.directory_button.click()
    assert not value.project.settings.get("temporary_storage_directory")
    manager.cpu_limit.setValue(4)
    manager.ram_limit_gb.setValue(4.295)
    manager.apply_button.click()
    assert saved == [{"cpu_limit": 4, "ram_limit_mb": 4096}]
    manager.temporary_directory.setText(str(tmp_path))
    manager.directory_button.click()
    assert value.project.settings["temporary_storage_directory"] == str(tmp_path)
    assert value.has_unsaved_changes


def test_memory_limit_dialog_offers_manager_and_cancel_without_overcommit(explorer, monkeypatch):
    from nfit import project_resource_gui

    value = explorer.explorer
    inspected = []
    opened = []

    def choose_manager(message):
        buttons = message.buttons()
        assert len(buttons) == 2
        assert not any("continue" in button.text().lower() for button in buttons)
        manage = next(button for button in buttons if "Resource Manager" in button.text())
        inspected.append(message.defaultButton().text())
        message._test_chosen = manage
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_manager)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", lambda message: message._test_chosen)
    monkeypatch.setattr(project_resource_gui, "open_resource_manager", lambda target: opened.append(target))
    error = ResourceLimitError(requested_bytes=200, used_bytes=80, limit_bytes=100, operation="Opening histogram")
    assert project_resource_gui.handle_resource_limit(value, error) is False
    assert inspected and "Cancel" in inspected[0]
    assert opened == [value]


def test_project_job_marks_manager_busy_and_restores_state_after_typed_error(explorer, monkeypatch):
    from nfit.project_resource_gui import run_project_resource_job

    value = explorer.explorer
    manager = value.open_resource_manager()
    value._interactive = True
    error = ResourceLimitError(requested_bytes=200, used_bytes=80, limit_bytes=100, operation="Opening histogram")

    def task(_report):
        assert value._resource_operation_active
        assert manager._busy
        raise error

    with pytest.raises(ResourceLimitError) as caught:
        run_project_resource_job(value, "Failed load", task)
    assert caught.value is error
    assert not value._resource_operation_active
    assert not manager._busy
    value._interactive = False


def test_manager_load_action_runs_loader_on_worker_with_gui_responsive(explorer, tmp_path, monkeypatch):
    from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
    from nfit.project_resource_gui import project_resources

    value = explorer.explorer
    path = tmp_path / "source.npz"
    write_dataset_artifact(cube(), path)
    explorer.dataset.metadata["source_file"] = str(path)
    manager = value.open_resource_manager()
    resources = project_resources(value)
    gui_thread = threading.get_ident()
    worker_threads = []
    gui_tick = threading.Event()

    def load(dataset):
        worker_threads.append(threading.get_ident())
        assert manager._busy
        if not gui_tick.wait(timeout=2):
            raise RuntimeError("GUI did not respond while loading")
        dataset.replace_data(read_dataset_artifact(path), source_backed=True)

    monkeypatch.setattr(resources, "_load_source", load)
    value._interactive = True
    manager.select_keys((f"source:{explorer.dataset.id}",))
    assert manager.load_button.isEnabled()
    QtCore.QTimer.singleShot(0, gui_tick.set)
    manager.load_button.click()
    assert worker_threads and worker_threads[0] != gui_thread
    assert explorer.dataset.data is not None
    assert not manager._busy
    value._interactive = False


def test_reloading_excluded_recipe_notifies_dirty_on_gui_thread(explorer, monkeypatch):
    from nfit.project_resource_gui import project_resources
    from nfit.project_resources import CACHE_EXCLUSIONS_KEY

    value = explorer.explorer
    gui_thread = threading.get_ident()
    loader_threads = []
    dirty_threads = []
    original_mark_dirty = value._mark_dirty

    def mark_dirty():
        dirty_threads.append(threading.get_ident())
        original_mark_dirty()

    monkeypatch.setattr(value, "_mark_dirty", mark_dirty)
    resources = project_resources(value)
    member = "assets/binnings/recreated/data.npz"
    value.project.settings[CACHE_EXCLUSIONS_KEY] = [member]

    def recreate():
        loader_threads.append(threading.get_ident())
        explorer.viewer_cache[explorer.dataset.id] = ("signature", cube())

    resources.recipes = lambda: (
        (explorer.viewer_cache, explorer.dataset.id, "Sample · Default", member, recreate),
    )
    manager = value.open_resource_manager()
    manager.select_keys((f"recipe:{member}",))
    assert manager.load_button.isEnabled()
    value._interactive = True
    manager.load_button.click()
    value._interactive = False
    assert len(loader_threads) == 1 and loader_threads[0] != gui_thread
    assert dirty_threads == [gui_thread]
    assert not value.project.settings[CACHE_EXCLUSIONS_KEY]
    assert value.has_unsaved_changes


def test_project_job_maps_user_cancellation_to_existing_public_exception(explorer, monkeypatch):
    from nfit import project_gui, qt_resource_jobs
    from nfit.project_resource_gui import run_project_resource_job

    value = explorer.explorer
    value._interactive = True

    def cancel(*_args, **_kwargs):
        raise ResourceJobCancelled("Cancelled")

    monkeypatch.setattr(qt_resource_jobs, "run_resource_job", cancel)
    with pytest.raises(project_gui.RebinCancellationRequested):
        run_project_resource_job(value, "Cancelled load", lambda report: None)
    assert not value._resource_operation_active
    value._interactive = False


@pytest.mark.parametrize("action", ["open", "save", "save_as"])
def test_archive_io_failure_reports_to_gui_and_preserves_dirty_state(explorer, monkeypatch, tmp_path, action):
    from nfit import project_gui, project_resource_gui

    value = explorer.explorer
    value._interactive = True
    value.has_unsaved_changes = True
    old_path = tmp_path / "old.nfit"
    value.project_path = old_path
    notices = []
    error = OSError("Insufficient free space to save this project")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(project_resource_gui, "run_project_resource_job", fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: notices.append(args[1:]))
    monkeypatch.setattr(value, "_project_changed_on_disk", lambda: False)
    monkeypatch.setattr(project_gui, "get_save_file_name", lambda *args: (str(tmp_path / "new.nfit"), ""))
    if action == "open":
        result = value._load_project_path(tmp_path / "new.nfit", remember=False)
    else:
        result = getattr(value, action)()
    assert result is False
    assert notices == [("Opening project failed" if action == "open" else "Saving project failed", str(error))]
    assert value.has_unsaved_changes
    assert value.project_path == old_path
    value._interactive = False


def test_archive_io_failure_preserves_typed_scripting_error(explorer):
    from nfit.project_resource_gui import handle_resource_io_error

    error = OSError("Disk is full")
    with pytest.raises(OSError) as caught:
        handle_resource_io_error(explorer.explorer, error, operation="Saving project")
    assert caught.value is error


def test_unload_closes_only_sharing_viewers_and_retains_plot_settings(explorer):
    from nfit.project_resource_gui import project_resources

    value = explorer.explorer
    data = cube()
    other_data = cube().with_updates(signal=np.full((4, 4), 7.0))
    explorer.viewer_cache[explorer.dataset.id] = ("signature", data)
    first = value._create_slice_viewer(explorer.group, [data.with_updates(metadata=data.metadata)], ["Sample"])
    second = value._create_slice_viewer(explorer.group, [other_data], ["Other"])
    first.apply_plot_settings({"autoscale": False, "manual_vmin": 0.25, "manual_vmax": 3.75})
    first.show()
    second.show()
    settings = first.current_plot_settings()
    resources = project_resources(value)
    row = next(row for row in resources.snapshot().rows if row.kind == "Histogram")
    resources.unload((row.key,))
    assert not first.window.isVisible()
    assert first.loaded_resource_items() == ()
    assert second.window.isVisible()
    key = (settings.get("dataset_name"), settings.get("binning_name"))
    assert value._resource_retired_plot_settings[key] == settings
    reopened = value._create_slice_viewer(explorer.group, [cube()], ["Sample"])
    restored = reopened.current_plot_settings()
    assert restored["autoscale"] is False
    assert restored["manual_vmin"] == 0.25 and restored["manual_vmax"] == 3.75
    assert key not in value._resource_retired_plot_settings


def test_project_replacement_releases_old_cache_arrays_and_closes_manager(explorer):
    value = explorer.explorer
    data = cube()
    reference = weakref.ref(data)
    explorer.viewer_cache[explorer.dataset.id] = ("signature", data)
    explorer.composite_cache[id(explorer.group)] = ("signature", data)
    manager = value.open_resource_manager()
    old_project = value.project
    del data
    assert value.new_project()
    gc.collect()
    assert value.project is not old_project
    assert reference() is None
    assert not explorer.viewer_cache.resource_records()
    assert not explorer.composite_cache.resource_records()
    assert value._resource_manager_window is None
    assert not manager.isVisible()
