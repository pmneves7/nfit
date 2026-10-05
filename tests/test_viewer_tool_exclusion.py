"""Navigation and box editing must never compete for the same drag."""

import numpy as np
import pytest

from nfit.mdhisto import MDHistoAxis, MDHistoData


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    axes = tuple(MDHistoAxis(name, np.arange(5.), "", "unknown") for name in ("H", "E"))
    signal = np.arange(16.).reshape(4, 4) + 1
    data = MDHistoData(axes, signal, np.ones_like(signal),
                       np.zeros_like(signal, dtype=bool), np.ones_like(signal))
    result = QtMDHistoSliceViewer(data, x_dim=0, y_dim=1)
    yield result
    result.window.close()


@pytest.mark.parametrize("tool", ["pan", "zoom"])
def test_navigation_deselects_box_and_box_releases_navigation(viewer, tool):
    viewer.roi_button.click()
    assert viewer.rectangle_selector.active
    extents = viewer._roi_extents
    viewer.toolbar._actions[tool].trigger()
    assert not viewer.roi_button.isChecked()
    assert not viewer.rectangle_selector.active
    assert viewer.show_box_check.isChecked()
    assert viewer.toolbar._actions[tool].isChecked()
    assert viewer.canvas.widgetlock.isowner(viewer.toolbar)
    assert viewer._roi_extents == extents

    viewer.roi_button.click()
    assert viewer.rectangle_selector.active
    assert not viewer.toolbar.mode
    assert not viewer.toolbar._actions["pan"].isChecked()
    assert not viewer.toolbar._actions["zoom"].isChecked()
    assert not viewer.canvas.widgetlock.locked()
    # Turning off a navigation tool does not reactivate box editing.
    viewer.toolbar._actions[tool].trigger()
    viewer.toolbar._actions[tool].trigger()
    assert not viewer.roi_button.isChecked()
    assert not viewer.rectangle_selector.active


@pytest.mark.parametrize("tool", ["pan", "zoom"])
def test_restored_box_settings_release_navigation(viewer, tool):
    viewer.roi_button.click()
    settings = viewer.current_plot_settings()
    viewer.toolbar._actions[tool].trigger()
    viewer.apply_plot_settings(settings)
    assert viewer.roi_button.isChecked()
    assert viewer.rectangle_selector.active
    assert not viewer.toolbar.mode
    assert not viewer.canvas.widgetlock.locked()
