from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import nfit.project_gui as project_gui
import nfit.project_window_builder as project_window_builder
from nfit.project_gui import NfitProject, NfitProjectExplorer


def _method_node(cls: type, name: str) -> ast.FunctionDef:
    source = inspect.getsource(cls)
    tree = ast.parse(source)
    class_node = tree.body[0]
    assert isinstance(class_node, ast.ClassDef)
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_project_explorer_build_is_a_thin_builder_delegation(monkeypatch):
    method = _method_node(NfitProjectExplorer, "_build")
    assert method.end_lineno - method.lineno < 12
    assert not any(
        isinstance(node, ast.Attribute) and node.attr.startswith("Q")
        for node in ast.walk(method)
    )

    calls = []
    window_factory = object()
    tree_factory = object()
    combo_factory = object()

    def record_builder(explorer, **factories):
        calls.append((explorer, factories))

    monkeypatch.setattr(project_window_builder, "build_project_window", record_builder)
    monkeypatch.setattr(project_gui, "_make_project_window_class", window_factory)
    monkeypatch.setattr(project_gui, "_make_project_tree_class", tree_factory)
    monkeypatch.setattr(project_gui, "_make_refreshing_combo_class", combo_factory)

    explorer = object.__new__(NfitProjectExplorer)
    explorer._build()

    assert calls == [
        (
            explorer,
            {
                "project_window_class_factory": window_factory,
                "project_tree_class_factory": tree_factory,
                "refreshing_combo_class_factory": combo_factory,
            },
        )
    ]


def test_project_window_builder_keeps_ordered_focused_construction_helpers():
    path = Path(project_window_builder.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    coordinator = functions["build_project_window"]
    helper_names = {
        "_build_window_shell",
        "_build_navigation_panel",
        "_build_work_area",
        "_build_fit_controls",
        "_build_fit_settings_panel",
        "_assemble_project_window",
    }
    ordered_calls = [
        node.func.id
        for node in ast.walk(coordinator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in helper_names
    ]
    assert ordered_calls == [
        "_build_window_shell",
        "_build_navigation_panel",
        "_build_work_area",
        "_build_fit_controls",
        "_build_fit_settings_panel",
        "_assemble_project_window",
    ]
    assert all(
        functions[name].end_lineno - functions[name].lineno < 300
        for name in helper_names
    )
    assert not any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("project_gui")
        for node in ast.walk(tree)
    )


def test_project_window_builder_constructs_and_connects_project_actions(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    calls = []

    def recorder(name):
        def callback(_explorer, *_args):
            calls.append(name)

        return callback

    action_callbacks = {
        "New": "new_project",
            "Open": "open_project",
            "Reload from Disk": "reload_project_from_disk",
            "Cache binnings": "_set_cache_binnings_enabled",
            "Save": "save",
        "Save As": "save_as",
        "Preferences…": "show_preferences",
        "Close": "close_project",
        "Quit": "quit_application",
    }
    for action_text, method_name in action_callbacks.items():
        monkeypatch.setattr(
            NfitProjectExplorer,
            method_name,
            recorder(action_text),
        )
    monkeypatch.setattr(
        NfitProjectExplorer,
        "_refresh_recent_projects_menu",
        recorder("Recent projects"),
    )
    monkeypatch.setattr(NfitProjectExplorer, "show_help", recorder("Help"))

    explorer = NfitProjectExplorer(NfitProject())
    try:
        splitter = explorer.window.centralWidget()
        assert isinstance(splitter, QtWidgets.QSplitter)
        assert splitter.orientation() == QtCore.Qt.Orientation.Horizontal
        assert splitter.count() == 2
        assert explorer.tree.parent() is not None
        assert explorer.details_scroll.parent() is not None

        toolbar = explorer.window.findChild(QtWidgets.QToolBar)
        assert toolbar is not None
        file_button = toolbar.findChild(QtWidgets.QToolButton, "file_menu_button")
        help_button = toolbar.findChild(QtWidgets.QToolButton, "help_button")
        assert file_button.menu() is explorer.file_menu
        assert [
            toolbar.widgetForAction(action).text()
            for action in toolbar.actions()
        ] == ["File", "Help"]

        actions = {
            action.text(): action
            for action in explorer.file_menu.actions()
            if not action.isSeparator() and action.menu() is None
        }
        assert list(actions) == list(action_callbacks)
        for action in actions.values():
            action.setEnabled(True)
            action.trigger()
        explorer.recent_projects_menu.aboutToShow.emit()
        help_button.click()

        assert calls == [*action_callbacks, "Recent projects", "Help"]
        assert actions["New"].shortcut().matches(
            QtGui.QKeySequence.StandardKey.New
        ) != QtGui.QKeySequence.SequenceMatch.NoMatch
        assert actions["Save"].shortcut().matches(
            QtGui.QKeySequence.StandardKey.Save
        ) != QtGui.QKeySequence.SequenceMatch.NoMatch
        assert explorer._external_change_timer.parent() is explorer.window
        assert explorer._external_change_timer.interval() == 1500
    finally:
        explorer.window.close()
        explorer.window.deleteLater()
        explorer.app.processEvents()
