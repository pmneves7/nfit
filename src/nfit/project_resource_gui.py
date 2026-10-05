"""Connect the explorer's resources to focused lifecycle services and Qt views."""

from .mapped_archive import array_storage_nbytes
from .project_resources import CACHE_EXCLUSIONS_KEY, ProjectResources, project_binning_member
from .qt_operation_guard import close_operation_window


def _viewer_payloads(explorer):
    for viewers in tuple(explorer._slice_viewers.values()):
        for viewer in tuple(viewers):
            if hasattr(viewer, "loaded_resource_items"):
                for _index, descriptor, payload in viewer.loaded_resource_items():
                    name = descriptor.name if descriptor is not None else "Data viewer"
                    yield f"{name} (viewer {id(viewer):x})", payload


def _release_viewers(explorer, selected):
    def shares(payload):
        return any(array_storage_nbytes((payload, value)).total <
                   array_storage_nbytes(payload).total + array_storage_nbytes(value).total
                   for value in selected)
    for viewers in tuple(explorer._slice_viewers.values()):
        for viewer in tuple(viewers):
            if not hasattr(viewer, "loaded_resource_items"):
                continue
            if any(shares(payload) for _, _, payload in viewer.loaded_resource_items()):
                if hasattr(viewer, "current_plot_settings"):
                    settings = viewer.current_plot_settings()
                    retained = getattr(explorer, "_resource_retired_plot_settings", {})
                    retained[(settings.get("dataset_name"), settings.get("binning_name"))] = settings
                    explorer._resource_retired_plot_settings = retained
                close_operation_window(viewer.window)
                viewer.release_loaded_data()


def _recipes(explorer):
    from . import project_composites as comp
    from . import project_data
    from .project_gui import _project_binning_targets, effective_dataset_masks

    for kind, label, group, target, binning_id, config in _project_binning_targets(explorer.project):
        member = project_binning_member(explorer.project, kind, group, target, binning_id)
        if kind == "dataset":
            fit = config is project_data._fit_dataset_rebin_config(target)
            key = target.id if fit else f"{target.id}:{binning_id}"
            def load(t=target, g=group, c=config, bid=binning_id, fit=fit):
                return project_data.dataset_for_slice_viewer(t, extra_masks=effective_dataset_masks(g, t),
                    force_rebin=True, rebin_config=c, cache_id=None if fit else bid)
            yield project_data._VIEWER_VIEW_CACHE, key, label, member, load
        else:
            fit = config is comp._fit_data_group_composite_config(target)
            key = comp._composite_cache_key(target, None if fit else binning_id)
            def load(t=target, c=config, bid=binning_id, fit=fit):
                return comp.composite_dataset_entry(t, force_rebin=True,
                    config_override=None if fit else c, binning_id=None if fit else bid)
            yield comp._COMPOSITE_DATA_CACHE, key, label, member, load


def project_resources(explorer):
    resources = getattr(explorer, "_resource_service", None)
    if resources is None or resources.project is not explorer.project:
        resources = ProjectResources(explorer.project,
            viewer_payloads=lambda: tuple(_viewer_payloads(explorer)),
            release_viewers=lambda values: _release_viewers(explorer, values),
            busy=lambda: getattr(explorer, "_fit_worker_thread", None) is not None
                         or getattr(explorer, "_resource_operation_active", False),
            changed=explorer._mark_dirty, recipes=lambda: tuple(_recipes(explorer)))
        explorer._resource_service = resources
    return resources


def open_resource_manager(explorer):
    from .data_workspace import set_project_temporary_directory
    from .performance import save_resource_limits
    from .qt_resource_manager import ResourceManagerWindow

    manager = getattr(explorer, "_resource_manager_window", None)
    if manager is not None and getattr(manager, "_nfit_project", None) is not explorer.project:
        manager.close()
        manager = None
    if manager is None:
        def action(method, keys):
            resources = project_resources(explorer)
            worker_load = method == "load" and explorer._interactive
            exclusions = tuple(explorer.project.settings.get(CACHE_EXCLUSIONS_KEY, ()))
            manager.set_busy(True)
            try:
                if worker_load:
                    # Selection validation occurs before the worker locks resource changes.
                    plan = resources.prepare_load(keys)
                    run_project_resource_job(explorer, "Loading selected resources…",
                        lambda report: resources.execute_load(plan, notify=False))
                else:
                    getattr(resources, method)(keys)
                explorer._refresh_cache_badges()
            finally:
                if worker_load and exclusions != tuple(explorer.project.settings.get(CACHE_EXCLUSIONS_KEY, ())):
                    resources.changed()
                manager.set_busy(False)

        def apply_limits(cpu, ram):
            save_resource_limits(cpu_limit=cpu, ram_limit_mb=ram)

        def set_directory(path):
            set_project_temporary_directory(explorer.project, path)
            explorer._mark_dirty()

        manager = ResourceManagerWindow(parent=explorer.window,
            snapshot=lambda: project_resources(explorer).snapshot(),
            unload=lambda keys: action("unload", keys),
            load=lambda keys: action("load", keys),
            delete=lambda keys: action("delete", keys),
            apply_limits=apply_limits, set_temporary_directory=set_directory)
        manager._nfit_project = explorer.project
        explorer._resource_manager_window = manager
    manager.refresh()
    manager.show()
    manager.raise_()
    manager.activateWindow()
    return manager


def handle_resource_limit(explorer, error):
    """Stop the operation and offer user-managed memory; never overcommit."""
    from PySide6 import QtWidgets

    message = QtWidgets.QMessageBox(explorer.window)
    message.setWindowTitle("RAM budget exceeded")
    message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    message.setText(str(error))
    message.setInformativeText("The operation has stopped. Increase the RAM budget or remove objects "
        "from active memory, then retry.")
    manage = message.addButton("Open Resource Manager", QtWidgets.QMessageBox.ButtonRole.ActionRole)
    manage.setToolTip("Inspect loaded data and shared memory; unload selected resources or change budgets.")
    cancel = message.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    message.setDefaultButton(cancel)
    message.exec()
    if message.clickedButton() is manage:
        open_resource_manager(explorer)
    return False


def handle_resource_io_error(explorer, error, *, operation):
    """Report failed archive I/O while preserving typed scripting failures."""
    if not explorer._interactive:
        raise error
    from PySide6 import QtWidgets

    QtWidgets.QMessageBox.warning(explorer.window, f"{operation} failed", str(error))
    return False


def retire_project_resources(explorer):
    """Release unreachable resources before replacing or closing a project."""
    from .project_caches import release_project_memory
    explorer._pending_overlay_groups.clear()
    timer = getattr(explorer, "_overlay_refresh_timer", None)
    if timer is not None:
        timer.stop()
    release_project_memory(explorer.project)
    manager = getattr(explorer, "_resource_manager_window", None)
    if manager is not None:
        close_operation_window(manager)
        manager.deleteLater()
    explorer._resource_manager_window = None
    explorer._resource_service = None
    explorer._resource_retired_plot_settings = {}
    analysis = getattr(explorer, "_analysis_window", None)
    if analysis is not None:
        close_operation_window(analysis.window)
        analysis.window.deleteLater()
        explorer._analysis_window = None


def run_project_resource_job(explorer, title, task, *, on_progress=None):
    """Serialize large GUI I/O, preserving typed errors and public APIs."""
    if not explorer._interactive:
        return task(on_progress)
    from PySide6 import QtWidgets
    from shiboken6 import isValid

    from .qt_resource_jobs import run_resource_job
    explorer._resource_operation_active = True
    manager = getattr(explorer, "_resource_manager_window", None)
    if manager is not None:
        manager.set_busy(True)
    try:
        try:
            candidates = (QtWidgets.QApplication.activeModalWidget(), QtWidgets.QApplication.activeWindow(),
                          manager, explorer.window)
            parent = next((widget for widget in candidates
                if widget is not None and isValid(widget) and widget.isVisible()), explorer.window)
            return run_resource_job(parent, title, task, on_progress=on_progress)
        except Exception as exc:
            from .qt_resource_jobs import ResourceJobCancelled
            if isinstance(exc, ResourceJobCancelled):
                from .project_gui import RebinCancellationRequested
                raise RebinCancellationRequested(str(exc)) from exc
            raise
    finally:
        explorer._resource_operation_active = False
        if manager is not None:
            manager.set_busy(False)
