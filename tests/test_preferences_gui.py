import pytest


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def test_preferences_colormap_folder_and_controls(qt_app, monkeypatch, tmp_path):
    from PySide6 import QtGui, QtWidgets

    from nfit.preferences_gui import PreferencesDialog

    directory = tmp_path / "palettes"
    monkeypatch.setenv("NFIT_COLORMAP_DIR", str(directory))
    opened = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    dialog = PreferencesDialog()
    try:
        assert dialog.tabs.tabText(0) == "Colormaps"
        assert dialog.folder_path.text() == str(directory)
        assert dialog.folder_path.isReadOnly()
        assert not directory.exists()
        dialog.open_folder_button.click()
        assert directory.is_dir()
        assert opened[0].toLocalFile() == str(directory)
        for control in dialog.findChildren(QtWidgets.QPushButton):
            assert control.toolTip()
        assert dialog.folder_path.toolTip()
        labels = "\n".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
        assert "Restart nfit" in labels
        assert "NFIT_COLORMAP_DIR" in labels
        assert "parula(256)" in labels
    finally:
        dialog.close()


def test_preferences_folder_open_failure(qt_app, monkeypatch, tmp_path):
    from PySide6 import QtGui, QtWidgets

    from nfit.colormaps import open_user_colormap_folder

    monkeypatch.setenv("NFIT_COLORMAP_DIR", str(tmp_path))
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda url: False)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    open_user_colormap_folder()
    assert len(warnings) == 1
    assert "Could not open" in warnings[0][2]


def test_file_menu_opens_preferences(qt_app, monkeypatch):
    from nfit.preferences_gui import PreferencesDialog
    from nfit.project_gui import NfitProjectExplorer

    shown = []
    monkeypatch.setattr(PreferencesDialog, "exec", lambda self: shown.append(self.parent()))
    explorer = NfitProjectExplorer()
    try:
        action = explorer.preferences_action
        assert action in explorer.file_menu.actions()
        assert action.toolTip()
        action.trigger()
        assert shown == [explorer.window]
    finally:
        explorer.window.close()
