"""Numbered-source UI uses the public expansion and does not import during preview."""

import pytest


@pytest.fixture
def source_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.source_selection_gui import build_source_selection_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtCore.QThreadPool, "start", lambda _pool, job: job.run())
    imported, files, changed = [], [], []
    config = dict(enabled=True, path=str(tmp_path), prefix="SEQ_", suffix=".dat", numors="1,1,2:3", padding=0)
    panel = build_source_selection_panel(
        config, on_setting_changed=lambda key, value: changed.append((key, value)),
        on_enabled=lambda _enabled: None, on_import=lambda selection, **kwargs: imported.append((selection, kwargs)),
        on_files=lambda: files.append(True), on_clear=lambda: None,
    )
    yield panel, imported, files, changed, tmp_path, app
    panel.deleteLater()
    app.processEvents()


def test_selector_fields_tooltips_and_explicit_file_alternative(source_panel):
    from PySide6 import QtWidgets

    panel, imported, files, _, _, _ = source_panel
    for key in ("path", "prefix", "suffix", "numors", "padding", "preview", "inspect_metadata", "preview_table", "preserve_groups", "expression_help"):
        widget = panel.findChild(QtWidgets.QWidget, f"dataset_importing_{key}")
        assert widget is not None and widget.toolTip()
    panel.findChild(QtWidgets.QPushButton, "dataset_importing_add_files").click()
    assert files == [True] and imported == []


def test_preview_preserves_appearances_missing_and_repeated_identities(source_panel):
    from PySide6 import QtWidgets

    panel, imported, _, _, directory, _ = source_panel
    for run in (1, 2):
        (directory / f"SEQ_{run}.dat").touch()
    panel.findChild(QtWidgets.QPushButton, "dataset_importing_preview").click()
    table = panel.findChild(QtWidgets.QTableWidget, "dataset_importing_preview_table")
    summary = panel.findChild(QtWidgets.QLabel, "dataset_importing_preview_summary").text()
    assert table.rowCount() == 4
    assert "4 appearances" in summary and "1 missing" in summary and "1 repeated" in summary
    assert table.item(0, 4).text() == "shared source"
    assert table.item(3, 3).text() == "missing"
    assert not panel.findChild(QtWidgets.QPushButton, "dataset_importing_import_range").isEnabled()
    assert imported == []


def test_padding_grouping_and_import_use_the_previewed_public_selection(source_panel):
    from PySide6 import QtWidgets

    from nfit.source_selection import resolve_source_selection

    panel, imported, _, changed, directory, _ = source_panel
    panel.findChild(QtWidgets.QSpinBox, "dataset_importing_padding").setValue(3)
    panel.findChild(QtWidgets.QLineEdit, "dataset_importing_numors").setText("1>2")
    for run in (1, 2):
        (directory / f"SEQ_{run:03d}.dat").touch()
    preview = panel.findChild(QtWidgets.QPushButton, "dataset_importing_preview")
    preview.click()
    button = panel.findChild(QtWidgets.QPushButton, "dataset_importing_import_range")
    assert button.isEnabled()
    button.click()
    assert len(imported) == 1
    plan = resolve_source_selection(imported[0][0])
    assert imported[0][1] == {"preserve_groups": False}
    from nfit.source_selection_imports import source_selection_script

    copy_script = panel.findChild(QtWidgets.QPushButton, "dataset_importing_copy_source_script")
    assert copy_script.toolTip()
    copy_script.click()
    assert QtWidgets.QApplication.clipboard().text() == source_selection_script(imported[0][0], preserve_groups=False)
    assert len(plan.slots) == 1 and plan.slots[0].run_numbers == (1, 2)
    assert [appearance.path.rsplit("/", 1)[-1] for appearance in plan.appearances] == ["SEQ_001.dat", "SEQ_002.dat"]
    assert ("padding", "3") in changed
    panel.findChild(QtWidgets.QCheckBox, "dataset_importing_preserve_groups").setChecked(True)
    button.click()
    assert imported[1][1] == {"preserve_groups": True}
    assert ("preserve_groups", True) in changed
    panel.findChild(QtWidgets.QLineEdit, "dataset_importing_numors").setText("2")
    assert not button.isEnabled()


def test_preview_reports_expression_errors_without_import_or_modal(source_panel):
    from PySide6 import QtWidgets

    panel, imported, _, _, _, _ = source_panel
    panel.findChild(QtWidgets.QLineEdit, "dataset_importing_numors").setText("not a run expression")
    panel.findChild(QtWidgets.QPushButton, "dataset_importing_preview").click()
    text = panel.findChild(QtWidgets.QLabel, "dataset_importing_preview_summary").text()
    assert text and "Resolving" not in text
    assert not panel.findChild(QtWidgets.QPushButton, "dataset_importing_import_range").isEnabled()
    assert imported == []


def test_metadata_preview_reads_headers_without_event_arrays(source_panel, monkeypatch):
    import h5py
    from PySide6 import QtWidgets

    from tests.test_raw_dgs import _write_raw_dgs

    panel, imported, _, _, directory, _ = source_panel
    source = directory / "SEQ_1.nxs"
    _write_raw_dgs(source)
    with h5py.File(source, "r+") as handle:
        handle["entry/instrument"].create_dataset("name", data=b"SEQUOIA")
    panel.findChild(QtWidgets.QLineEdit, "dataset_importing_suffix").setText(".nxs")
    read = h5py.Dataset.__getitem__

    def header_only(dataset, selection):
        if dataset.name.rsplit("/", 1)[-1] in {"event_id", "event_time_offset", "event_index", "event_time_zero"}:
            raise AssertionError("source preview must not load event arrays")
        return read(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", header_only)
    panel.findChild(QtWidgets.QLineEdit, "dataset_importing_numors").setText("1")
    panel.findChild(QtWidgets.QCheckBox, "dataset_importing_inspect_metadata").setChecked(True)
    panel.findChild(QtWidgets.QPushButton, "dataset_importing_preview").click()
    table = panel.findChild(QtWidgets.QTableWidget, "dataset_importing_preview_table")
    assert "header_error" not in table.item(0, 5).text()
    assert "instrument" in table.item(0, 5).text()
    assert imported == []


def test_async_preview_keeps_gui_callbacks_on_main_thread_and_discards_stale_results(tmp_path, monkeypatch):
    import threading
    import time

    from PySide6 import QtWidgets

    from nfit import source_selection, source_selection_gui

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for run in (1, 2):
        (tmp_path / f"SEQ_{run}.dat").touch()
    started, release = threading.Event(), threading.Event()
    resolve = source_selection.resolve_source_selection
    worker_threads, render_threads = [], []
    main_thread = threading.get_ident()

    def resolve_slowly(selection, **kwargs):
        worker_threads.append(threading.get_ident())
        started.set()
        assert release.wait(5)
        return resolve(selection, **kwargs)

    render = source_selection_gui._present_preview

    def render_on_gui(*args):
        render_threads.append(threading.get_ident())
        render(*args)

    monkeypatch.setattr(source_selection, "resolve_source_selection", resolve_slowly)
    monkeypatch.setattr(source_selection_gui, "_present_preview", render_on_gui)
    panel = source_selection_gui.build_source_selection_panel(
        dict(enabled=True, path=str(tmp_path), prefix="SEQ_", suffix=".dat", numors="1"),
        on_setting_changed=lambda *_args: None, on_enabled=lambda *_args: None,
        on_import=lambda *_args: None, on_files=lambda: None, on_clear=lambda: None,
    )
    preview = panel.findChild(QtWidgets.QPushButton, "dataset_importing_preview")
    try:
        preview.click()
        assert started.wait(5)
        panel.findChild(QtWidgets.QLineEdit, "dataset_importing_numors").setText("2")
        release.set()
        deadline = time.monotonic() + 5
        while not preview.isEnabled() and time.monotonic() < deadline:
            app.processEvents()
        assert preview.isEnabled()
        assert panel.findChild(QtWidgets.QTableWidget, "dataset_importing_preview_table").rowCount() == 0
        preview.click()
        deadline = time.monotonic() + 5
        while not preview.isEnabled() and time.monotonic() < deadline:
            app.processEvents()
        assert preview.isEnabled()
        assert worker_threads and all(value != main_thread for value in worker_threads)
        assert render_threads == [main_thread]
    finally:
        release.set()
        panel.deleteLater()
        app.processEvents()


def test_explorer_imports_raw_range_as_one_lazy_group_and_keeps_source_recipe(tmp_path, monkeypatch):
    import json

    from PySide6 import QtWidgets

    from nfit import DataGroup, NfitProject, NfitProjectExplorer
    from nfit.source_selection_gui import source_selection_from_config
    from nfit.source_selection_imports import import_source_selection, source_selection_script
    from tests.test_raw_dgs import _write_raw_dgs

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for run in (1, 2):
        _write_raw_dgs(tmp_path / f"SEQ_{run:03d}.nxs")
    root = DataGroup("sample")
    explorer = NfitProjectExplorer(NfitProject([root]))
    try:
        config = explorer._dataset_importing_config(root)
        config.update(enabled=True, path=str(tmp_path), prefix="SEQ_", suffix=".nxs", numors="1:2", padding=3)
        explorer._refresh_tree(select_group=root)
        explorer._import_dataset_importing_range(root)
        assert len(root.subgroups) == 1
        group = root.subgroups[0]
        assert len(group.datasets) == 2 and not group.subgroups
        assert all(dataset.data is None for dataset in group.datasets)
        summary = explorer.details_widget.findChild(QtWidgets.QLabel, "saved_source_selection_text")
        assert summary is not None and summary.toolTip()
        assert "1:2" in summary.text() and "one dataset group" in summary.text()
        copy_expression = explorer.details_widget.findChild(QtWidgets.QPushButton, "saved_source_selection_copy_expression")
        assert copy_expression.toolTip()
        copy_expression.click()
        assert QtWidgets.QApplication.clipboard().text() == "1:2"
        assert group.metadata["source_selection"]["expression"] == "1:2"
        assert group.metadata["source_selection"]["padding"] == 3
        selection = source_selection_from_config(config)
        public_group = import_source_selection(DataGroup("public"), selection)
        assert [dataset.metadata["source_file"] for dataset in public_group.datasets] == [dataset.metadata["source_file"] for dataset in group.datasets]
        script = source_selection_script(selection)
        namespace = {"__name__": "source_import_replay"}
        exec(script, namespace)
        _parent, replay_group = namespace["run"]()
        assert json.loads(json.dumps(replay_group.metadata["raw_dgs"])) == json.loads(json.dumps(group.metadata["raw_dgs"]))
        assert all(dataset.data is None for dataset in replay_group.datasets)
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_destroyed_panel_discards_running_preview_without_touching_widgets(tmp_path, monkeypatch, capsys):
    import threading
    import time

    from PySide6 import QtCore, QtWidgets
    from shiboken6 import isValid

    from nfit import source_selection, source_selection_gui

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    (tmp_path / "SEQ_1.dat").touch()
    started, release = threading.Event(), threading.Event()
    resolve = source_selection.resolve_source_selection
    rendered = []

    def pending(selection, **kwargs):
        started.set()
        assert release.wait(5)
        return resolve(selection, **kwargs)

    monkeypatch.setattr(source_selection, "resolve_source_selection", pending)
    monkeypatch.setattr(source_selection_gui, "_present_preview", lambda *_args: rendered.append(True))
    panel = source_selection_gui.build_source_selection_panel(
        dict(enabled=True, path=str(tmp_path), prefix="SEQ_", suffix=".dat", numors="1"),
        on_setting_changed=lambda *_args: None, on_enabled=lambda *_args: None,
        on_import=lambda *_args: None, on_files=lambda: None, on_clear=lambda: None,
    )
    try:
        panel.findChild(QtWidgets.QPushButton, "dataset_importing_preview").click()
        assert started.wait(5)
        panel.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert not isValid(panel)
        release.set()
        deadline = time.monotonic() + 5
        while app._nfit_source_preview_jobs and time.monotonic() < deadline:
            app.processEvents()
        app.processEvents()
        assert not app._nfit_source_preview_jobs
        assert rendered == []
        assert "RuntimeError" not in capsys.readouterr().err
    finally:
        release.set()
        if isValid(panel):
            panel.deleteLater()
        app.processEvents()
