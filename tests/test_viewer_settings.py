import json

import numpy as np
import pytest

from nfit.viewer_settings import compatible_viewer_settings, copy_viewer_settings


def test_settings_remap_hidden_axes_without_changing_data_identity():
    settings = {"dataset_name": "source", "binning_name": "source bins",
                "waterfall_dataset_names": ["source"], "x_dim": "H", "y_dim": "E",
                "selections": {2: (1., 2.)}, "integrate_checks": {2: True},
                "brillouin_zone_lattice_parameters": [1] * 6, "font_size": 18.}
    payload = copy_viewer_settings(settings, ["H", "E", "K"])
    # The system clipboard crosses a JSON serialization boundary.
    result = compatible_viewer_settings(json.loads(json.dumps(payload)), ["K", "H", "E"])
    assert result["selections"] == {0: [1., 2.]}
    assert result["integrate_checks"] == {0: True}
    assert result["font_size"] == 18.
    assert "dataset_name" not in result
    assert "binning_name" not in result
    assert "waterfall_dataset_names" not in result
    assert "brillouin_zone_lattice_parameters" not in result
    payload["settings"]["selections"][2] = (9., 10.)
    assert settings["selections"][2] == (1., 2.)


def test_missing_axes_drop_coordinate_dependent_settings():
    payload = copy_viewer_settings({"x_dim": "H", "y_dim": "E", "xlim": [0, 1],
        "ylim": [2, 3], "x_step": .1, "roi_extents": [0, 1, 2, 3],
        "tile_dim": "K", "view_mode": "tiled_slices", "cmap": "magma",
        "selections": {2: [1, 2]}}, ["H", "E", "K"])
    result = compatible_viewer_settings(payload, ["E", "temperature"])
    assert result["cmap"] == "magma"
    for key in ("x_dim", "y_dim", "xlim", "ylim", "x_step", "roi_extents", "tile_dim", "view_mode"):
        assert key not in result
    assert result["selections"] == {}


def test_unrecognized_payload_is_rejected():
    with pytest.raises(ValueError):
        compatible_viewer_settings({"format": "other"}, ["x"])


@pytest.fixture
def viewers(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.mdhisto import MDHistoAxis, MDHistoData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    results = []
    for names in (("x", "y", "scan"), ("scan", "x", "y")):
        axes = tuple(MDHistoAxis(name, np.arange(4.), "", "unknown") for name in names)
        signal = np.arange(27.).reshape(3, 3, 3) + 1
        data = MDHistoData(axes, signal, np.ones_like(signal),
                           np.zeros_like(signal, dtype=bool), np.ones_like(signal))
        results.append(QtMDHistoSliceViewer(data, x_dim=names.index("x"), y_dim=names.index("y")))
    yield results
    for viewer in results:
        viewer.window.close()


def test_clipboard_buttons_transfer_real_controls_and_keep_destination_data(viewers):
    from PySide6 import QtWidgets
    source, target = viewers
    data = target.data
    source.apply_plot_settings({"font_size": 19., "axis_linewidth": 2.,
        "cmap": "magma", "color_scale": "asinh", "smoothing_x": .7,
        "selections": {2: (1., 2.)}, "integrate_checks": {2: True},
        "xlim": (.2, 2.4), "ylim": (.3, 2.5), "show_major_gridlines": True})
    source.hold_view_settings_check.setChecked(True)
    source.copy_settings_button.click()
    assert target.paste_settings_button.isEnabled()
    assert source.copy_settings_button.parent() is source.save_plot_button.parent()
    assert target.paste_settings_button.parent() is target.save_plot_button.parent()
    assert source.copy_settings_button.toolTip()
    assert target.paste_settings_button.toolTip()
    target.paste_settings_button.click()
    assert target.data is data
    assert target.font_size == 19.
    assert target.axis_linewidth == 2.
    assert target.model.cmap == "magma"
    assert target.model.color_scale == "asinh"
    assert target.smoothing_x == .7
    assert target.model.selections[0] == (1., 2.)
    assert target.hold_view_settings_check.isChecked()
    assert target.show_major_gridlines
    np.testing.assert_allclose(target.ax_image.get_xlim(), (.2, 2.4))
    np.testing.assert_allclose(target.ax_image.get_ylim(), (.3, 2.5))
    QtWidgets.QApplication.clipboard().clear()
    assert not target.paste_settings_button.isEnabled()


def test_portable_settings_service_is_gui_independent():
    import ast
    import inspect

    from nfit import viewer_settings

    tree = ast.parse(inspect.getsource(viewer_settings))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(module in {"__future__", "collections.abc", "copy"} for module in imports)
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))


def test_point_list_settings_clipboard_uses_held_view_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = PointListData(columns={"x": np.arange(5.), "first": np.arange(5.)},
                         coordinate_names=["x"],
                         channels=[{"label": "first", "value": "first", "error": None}])
    source, target = (QtMDHistoSliceViewer(data) for _ in range(2))
    try:
        source._set_font_size(21.)
        source.ax_image.set_xlim(.5, 3.5)
        source.ax_image.set_ylim(.1, 5.)
        source.copy_settings_button.click()
        target.paste_settings_button.click()
        assert target.font_size == 21.
        np.testing.assert_allclose(target.ax_image.get_xlim(), (.5, 3.5))
        np.testing.assert_allclose(target.ax_image.get_ylim(), (.1, 5.))
        assert target.data is data
    finally:
        QtWidgets.QApplication.clipboard().clear()
        source.window.close()
        target.window.close()


def test_single_axis_plot_retains_limits():
    payload = copy_viewer_settings({"x_dim": "E", "y_dim": "E",
                                   "xlim": [1, 2], "ylim": [3, 4]}, ["E"])
    result = compatible_viewer_settings(payload, ["E"])
    assert result["xlim"] == [1, 2]
    assert result["ylim"] == [3, 4]
