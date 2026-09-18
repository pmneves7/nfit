"""State-preservation coverage for rebuilt model editor controls."""

import pytest

from nfit import project_gui
from nfit.pipeline import DataGroup
from nfit.project_gui import NfitProject, NfitProjectExplorer, create_model_component


def test_model_editor_rebuild_preserves_tab_focus_cursor_and_scroll(monkeypatch):
    """Rebuilding a large model editor keeps the user's active fitting field."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Magnetism")
    model = create_model_component(group, "response", type="heisenberg_rpa")
    model.config["orbits"] = [{"label": f"J{index}", "bonds": []} for index in range(1, 25)]
    project_gui.reconcile_model_orbit_parameters(model)
    model.limits["J24"] = [-12.0, 12.0]
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_model=model, refresh_viewers=False)
    explorer.window.resize(800, 500)
    explorer.window.show()
    QtWidgets.QApplication.processEvents()

    tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget, "heisenberg_builder_tabs"
    )
    field = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_parameter_min_J24"
    )
    assert tabs is not None
    assert field is not None
    tabs.setCurrentIndex(1)
    field.setFocus()
    field.setCursorPosition(1)
    scrollbar = explorer.model_parameter_scroll.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    prior_scroll = scrollbar.value()
    assert prior_scroll > 0

    explorer._rebuild_model_parameter_editor_preserving_scroll(model)
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()

    rebuilt_tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget, "heisenberg_builder_tabs"
    )
    rebuilt_field = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_parameter_min_J24"
    )
    assert rebuilt_tabs is not None
    assert rebuilt_tabs.tabText(rebuilt_tabs.currentIndex()) == "Response and fit"
    assert rebuilt_field is not None
    assert explorer.window.focusWidget() is rebuilt_field
    assert rebuilt_field.cursorPosition() == 1
    assert scrollbar.value() == prior_scroll
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_widget_state_does_not_focus_an_unrelated_replacement(monkeypatch):
    """An unnamed removed control must not donate its focus to a new sibling."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_widget_state import preserve_widget_state

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    root = QtWidgets.QWidget(window)
    layout = QtWidgets.QVBoxLayout(root)
    removed = QtWidgets.QLineEdit()
    layout.addWidget(removed)
    window.show()
    removed.setFocus()
    app.processEvents()
    assert window.focusWidget() is removed

    with preserve_widget_state(root):
        layout.removeWidget(removed)
        removed.setParent(None)
        removed.deleteLater()
        replacement = QtWidgets.QLineEdit()
        layout.addWidget(replacement)
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()

    assert window.focusWidget() is not replacement
    window.close()
