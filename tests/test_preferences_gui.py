import pytest


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def test_application_settings_avoid_atomic_locking_on_linux(qt_app, monkeypatch, tmp_path):
    from nfit import application_preferences

    monkeypatch.setattr(application_preferences.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    settings = application_preferences.application_settings()
    settings.setValue("files/last_directory", "/data/run")
    settings.sync()
    restored = application_preferences.application_settings()

    assert not settings.isAtomicSyncRequired()
    assert restored.value("files/last_directory") == "/data/run"
    assert restored.fileName() == str(tmp_path / "nfit" / "settings.json")
    assert not list((tmp_path / "nfit").glob("*.lock*"))


def test_preferences_colormap_folder_and_controls(qt_app, monkeypatch, tmp_path):
    from PySide6 import QtCore, QtGui, QtWidgets

    from nfit.application_preferences import (
        default_continuous_colormap,
        default_waterfall_colormap,
    )
    from nfit.preferences_gui import PreferencesDialog

    directory = tmp_path / "palettes"
    monkeypatch.setenv("NFIT_COLORMAP_DIR", str(directory))
    opened = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    settings = QtCore.QSettings(
        str(tmp_path / "preferences.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    dialog = PreferencesDialog(settings=settings)
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
        assert dialog.continuous_colormap.toolTip()
        assert dialog.waterfall_colormap.toolTip()
        dialog.continuous_colormap.setCurrentText("plasma")
        dialog.waterfall_colormap.setCurrentText("magma")
        settings.sync()
        assert default_continuous_colormap(settings) == "plasma"
        assert default_waterfall_colormap(settings) == "magma"
        labels = "\n".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
        assert "Restart nfit" in labels
        assert "NFIT_COLORMAP_DIR" in labels
        assert "parula(256)" in labels
        assert dialog.version_label.text().startswith("nfit version ")
        assert dialog.version_label.toolTip()
    finally:
        dialog.close()


def test_viewer_preload_preference_defaults_off_and_persists(qt_app, tmp_path):
    from PySide6 import QtCore

    from nfit.application_preferences import (
        preload_viewer_data,
        set_preload_viewer_data,
    )

    settings = QtCore.QSettings(
        str(tmp_path / "preferences.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    assert preload_viewer_data(settings) is False

    set_preload_viewer_data(True, settings)
    settings.sync()
    restored = QtCore.QSettings(
        str(tmp_path / "preferences.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    assert preload_viewer_data(restored) is True

    set_preload_viewer_data(False, restored)
    assert preload_viewer_data(restored) is False


def test_preferences_viewer_preload_control(qt_app, tmp_path):
    from PySide6 import QtCore

    from nfit.application_preferences import preload_viewer_data
    from nfit.preferences_gui import PreferencesDialog

    settings = QtCore.QSettings(
        str(tmp_path / "preferences.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    dialog = PreferencesDialog(settings=settings)
    try:
        checkbox = dialog.preload_viewer_data_checkbox
        assert dialog.tabs.indexOf(checkbox.parentWidget()) == 1
        assert dialog.tabs.tabText(1) == "Data viewer"
        assert not checkbox.isChecked()
        assert "on demand" in checkbox.toolTip()
        assert "faster switching" in checkbox.toolTip()
        assert "newly opened viewers" in checkbox.toolTip()

        checkbox.setChecked(True)
        assert preload_viewer_data(settings) is True
    finally:
        dialog.close()


def test_viewer_uses_local_colormap_defaults_for_new_plots(qt_app, monkeypatch):
    import nfit.qt_slice_viewer as viewer_module
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    from tests.plotting_test_data import tiny_mdhisto_data

    monkeypatch.setattr(viewer_module, "default_continuous_colormap", lambda: "plasma")
    monkeypatch.setattr(viewer_module, "default_waterfall_colormap", lambda: "magma")

    viewer = QtMDHistoSliceViewer(tiny_mdhisto_data())
    try:
        assert viewer.model.cmap == "plasma"
        assert viewer.waterfall_cmap == "magma"
    finally:
        viewer.window.close()


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


@pytest.mark.parametrize(
    ("platform", "environment", "expected"),
    [
        ("darwin", {}, ("Library", "Application Support", "nfit", "colormaps")),
        ("win32", {"APPDATA": "appdata"}, ("appdata", "nfit", "colormaps")),
        ("linux", {"XDG_CONFIG_HOME": "config"}, ("config", "nfit", "colormaps")),
    ],
)
def test_default_colormap_folder_uses_native_nfit_application_data(
    monkeypatch, tmp_path, platform, environment, expected
):
    import nfit.colormaps as colormaps

    monkeypatch.delenv("NFIT_COLORMAP_DIR", raising=False)
    monkeypatch.setattr(colormaps.sys, "platform", platform)
    monkeypatch.setattr(colormaps.Path, "home", classmethod(lambda cls: tmp_path))
    for name, relative_path in environment.items():
        monkeypatch.setenv(name, str(tmp_path / relative_path))
    assert colormaps.user_colormap_directory() == tmp_path.joinpath(*expected)


def test_legacy_colormap_folder_is_moved_to_application_data(monkeypatch, tmp_path):
    import nfit.colormaps as colormaps

    monkeypatch.delenv("NFIT_COLORMAP_DIR", raising=False)
    monkeypatch.setattr(colormaps.Path, "home", classmethod(lambda cls: tmp_path))
    legacy = tmp_path / "nfit_colormaps"
    legacy.mkdir()
    (legacy / "custom.csv").write_text("0 0 0\n1 1 1\n")
    colormaps._migrate_legacy_user_colormap_directory()
    destination = colormaps.user_colormap_directory()
    assert not legacy.exists()
    assert (destination / "custom.csv").is_file()


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


def test_performance_preferences_save_and_apply(qt_app, monkeypatch, tmp_path):
    from nfit.performance import load_performance_settings, load_resource_limits
    from nfit.performance_gui import PerformancePage

    monkeypatch.setenv("NFIT_PERFORMANCE_FILE", str(tmp_path / "performance.json"))
    page = PerformancePage()
    page.cpu_limit.setValue(2)
    page.ram_limit_mb.setValue(4096)
    assert load_performance_settings()["workers"] == 0
    page._save()
    assert load_performance_settings() == {
        "max_batch_mb": 0,
        "workers": 2,
        "transient_memory_percent": 0,
    }
    assert load_resource_limits() == {"cpu_limit": 2, "ram_limit_mb": 4096}
    assert page.ram_limit_mb.toolTip()
    page.close()


def test_performance_calibration_preserves_the_ram_limit(qt_app, monkeypatch, tmp_path):
    from nfit.performance import (
        load_performance_settings,
        load_resource_limits,
        save_resource_limits,
    )
    from nfit.performance_gui import BenchmarkDialog, PerformancePage

    monkeypatch.setenv("NFIT_PERFORMANCE_FILE", str(tmp_path / "performance.json"))
    save_resource_limits(cpu_limit=6, ram_limit_mb=4096)
    page = PerformancePage()

    def execute(dialog):
        dialog.apply({"max_batch_mb": 32, "workers": 2})

    monkeypatch.setattr(BenchmarkDialog, "exec", execute)
    page._calibrate()

    assert load_resource_limits() == {"cpu_limit": 2, "ram_limit_mb": 4096}
    assert load_performance_settings()["max_batch_mb"] == 0
    page.close()


def test_rebin_benchmark_uses_central_resource_limits(qt_app, monkeypatch):
    from PySide6 import QtWidgets

    from nfit import DataGroup, DatasetEntry, NfitProject
    from nfit.performance_gui import BenchmarkDialog
    from nfit.project_gui import NfitProjectExplorer, dataset_rebin_config
    from tests.project_gui_test_support import _tiny_mdhisto_data

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0))
    group = DataGroup("group", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))
    before = dict(dataset_rebin_config(first))
    opened = []

    def execute(dialog):
        opened.append(dialog.target)
        assert dialog.apply_button.isHidden()

    monkeypatch.setattr(BenchmarkDialog, "exec", execute)
    holder = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(holder)
    explorer._add_rebin_performance_controls(row, dataset=first, group=group)
    button = holder.findChild(QtWidgets.QPushButton, "dataset_rebin_benchmark")
    assert button.toolTip()
    assert holder.findChild(QtWidgets.QSpinBox, "dataset_rebin_workers") is None
    button.click()
    assert opened == [{"dataset_id": first.id}]
    assert dataset_rebin_config(first) == before
    holder.close()
    explorer.window.close()
