import pytest


@pytest.mark.parametrize("module_name", ["project_gui", "qt_slice_controls"])
def test_app_startup_installs_icon_for_existing_and_new_windows(monkeypatch, module_name):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtGui = pytest.importorskip("PySide6.QtGui")
    from importlib import import_module

    module = import_module(f"nfit.{module_name}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setWindowIcon(QtGui.QIcon())
    assert module._qt_app() is app
    assert not app.windowIcon().pixmap(32, 32).isNull()
    window = QtWidgets.QMainWindow()
    try:
        assert not window.windowIcon().pixmap(32, 32).isNull()
    finally:
        window.close()
