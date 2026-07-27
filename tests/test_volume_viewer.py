import sys
import types

import numpy as np
import pytest

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.qt_volume_viewer import (
    build_rectilinear_volume_grid,
    crop_volume_arrays,
    default_hidden_axis_index,
    default_volume_axes,
    extract_volume_arrays,
    map_volume_rgba,
    rotate_camera,
    rotation_frame_angles,
    sample_transfer_curve,
    supports_volume_view,
    volume_channel_names,
)


def _axis(name, size):
    return MDHistoAxis(
        name=name,
        values=np.linspace(0.0, float(size), size + 1),
        units="r.l.u.",
        kind="momentum",
    )


def _volume_data(shape=(2, 3, 4, 5)):
    signal = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return MDHistoData(
        axes=tuple(_axis(name, size) for name, size in zip(("E", "H", "K", "L"), shape, strict=True)),
        signal=signal,
        errors=np.sqrt(signal + 1.0),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape, dtype=float),
        metadata={"fit": signal * 2.0},
    )


def test_volume_support_and_default_axes_require_three_grid_dimensions():
    data = _volume_data()
    assert supports_volume_view(data)
    assert default_volume_axes(data) == (1, 2, 3)
    assert "fit" in volume_channel_names(data)

    sparse = _volume_data((2, 3, 1, 1))
    assert default_volume_axes(sparse) == (0, 1, 2)

    two_dimensional = _volume_data((2, 3, 1, 1))
    two_dimensional.signal = two_dimensional.signal[:, :, 0, 0]
    two_dimensional.errors = two_dimensional.errors[:, :, 0, 0]
    two_dimensional.mask = two_dimensional.mask[:, :, 0, 0]
    two_dimensional.num_events = two_dimensional.num_events[:, :, 0, 0]
    two_dimensional.axes = two_dimensional.axes[:2]
    assert not supports_volume_view(two_dimensional)


def test_default_hidden_axis_uses_nearest_measured_bin():
    data = _volume_data((5, 3, 4, 5))
    data.mask[1:4] = True

    assert default_hidden_axis_index(data, 0) == 0
    assert default_hidden_axis_index(data, 0, apply_masks=False) == 2


def test_extract_volume_selects_or_integrates_hidden_dimensions_and_orders_xyz():
    data = _volume_data()
    selected = extract_volume_arrays(
        data,
        axes=(3, 1, 2),
        color_channel="signal",
        opacity_channel="errors",
        hidden_selections={0: 1},
    )
    assert selected.color.shape == (5, 3, 4)
    np.testing.assert_allclose(selected.color, np.transpose(data.signal[1], (2, 0, 1)))

    integrated = extract_volume_arrays(
        data,
        axes=(3, 1, 2),
        color_channel="signal",
        opacity_channel="fit",
        hidden_selections={0: (0, 1)},
    )
    expected = np.transpose(np.sum(data.signal, axis=0), (2, 0, 1))
    np.testing.assert_allclose(integrated.color, expected)
    np.testing.assert_allclose(integrated.opacity, expected * 2.0)


def test_extract_volume_applies_masks_before_hidden_axis_integration():
    data = _volume_data()
    data.mask[0, 0, 0, 0] = True
    arrays = extract_volume_arrays(
        data,
        axes=(1, 2, 3),
        color_channel="signal",
        opacity_channel="signal",
        hidden_selections={0: (0, 1)},
        apply_masks=True,
    )
    assert arrays.color[0, 0, 0] == pytest.approx(data.signal[1, 0, 0, 0])


def test_independent_opacity_channel_changes_alpha_not_color():
    color = np.asarray([0.0, 0.5, 1.0])
    opacity_a = np.asarray([0.0, 0.5, 1.0])
    opacity_b = opacity_a[::-1]
    kwargs = dict(
        cmap="viridis",
        color_range=(0.0, 1.0),
        opacity_range=(0.0, 1.0),
        color_curve=[(0.0, 0.0), (1.0, 1.0)],
        opacity_curve=[(0.0, 0.0), (1.0, 1.0)],
    )
    rgba_a = map_volume_rgba(color, opacity_a, **kwargs)
    rgba_b = map_volume_rgba(color, opacity_b, **kwargs)
    np.testing.assert_array_equal(rgba_a[:, :3], rgba_b[:, :3])
    np.testing.assert_array_equal(rgba_a[:, 3], [0, 128, 255])
    np.testing.assert_array_equal(rgba_b[:, 3], [255, 128, 0])


def test_transfer_curve_interpolates_piecewise_mapping():
    sampled = sample_transfer_curve([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], np.asarray([0.25, 0.5, 0.75]))
    np.testing.assert_allclose(sampled, [0.5, 1.0, 0.5])


def test_rectilinear_grid_preserves_xyz_cell_order_and_rgba():
    data = _volume_data((2, 3, 4, 5))
    arrays = extract_volume_arrays(
        data,
        axes=(3, 1, 2),
        color_channel="signal",
        opacity_channel="errors",
        hidden_selections={0: 0},
    )
    rgba = np.zeros((*arrays.color.shape, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    grid = build_rectilinear_volume_grid(arrays, rgba)
    assert grid.n_cells == 5 * 3 * 4
    assert grid.cell_data["rgba"].shape == (grid.n_cells, 4)
    np.testing.assert_allclose(grid.cell_data["color_channel"], arrays.color.ravel(order="F"))


def test_volume_axis_limits_crop_complete_cells_and_keep_nearest_nonempty_bin():
    arrays = extract_volume_arrays(
        _volume_data(),
        axes=(3, 1, 2),
        color_channel="signal",
        opacity_channel="errors",
        hidden_selections={0: 0},
    )
    cropped = crop_volume_arrays(arrays, ((1.0, 3.0), (0.0, 3.0), (99.0, 100.0)))
    assert cropped.color.shape == (2, 3, 1)
    np.testing.assert_allclose(cropped.x_edges, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cropped.z_edges, [3.0, 4.0])


def test_rotation_movie_angles_complete_exactly_one_orbit():
    angles = rotation_frame_angles(30, 12.0)
    assert angles.size == 360
    assert np.sum(angles) == pytest.approx(360.0)


def test_camera_rotation_supports_displayed_axes_and_screen_vertical():
    class Camera:
        position = (1.0, 0.0, 0.0)
        focal_point = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        azimuth = 0.0

        def Azimuth(self, angle):
            self.azimuth += float(angle)

    camera = Camera()
    rotate_camera(camera, 90.0, "Z axis")
    np.testing.assert_allclose(camera.position, (0.0, 1.0, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(camera.up, (-1.0, 0.0, 0.0), atol=1.0e-12)

    rotate_camera(camera, 15.0, "Vertical axis")
    assert camera.azimuth == pytest.approx(15.0)


def test_data_viewer_exposes_volumetric_mode_for_nd_histograms(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_volume_data())
    mode = viewer.window.findChild(QtWidgets.QComboBox, "data_viewer_mode_combo")
    assert mode is not None
    assert mode.itemText(1) == "Waterfall"
    assert mode.itemText(2) == "Volumetric"
    assert mode.model().item(1).isEnabled()
    assert mode.model().item(2).isEnabled()
    assert "three dimensions" in mode.toolTip()
    viewer.window.close()


def test_volume_panel_exposes_independent_channels_curves_and_camera_exports(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    from nfit.qt_volume_viewer import QtVolumeViewerPanel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Camera:
        up = (0.0, 1.0, 0.0)
        position = (1.0, 0.0, 0.0)
        focal_point = (0.0, 0.0, 0.0)
        total_azimuth = 0.0

        def Azimuth(self, angle):
            self.total_azimuth += float(angle)

    class Writer:
        closed = False

        def close(self):
            self.closed = True

    class FakeInteractor:
        def __init__(self, parent):
            self.interactor = QtWidgets.QWidget(parent)
            self.camera = Camera()
            self.camera_position = None
            self.mwriter = None
            self.frames = 0
            self.screenshot_path = None
            self.background = None
            self.axes_color = None
            self.added_volumes = []
            self.bounds_kwargs = None
            self.close_count = 0

        def set_background(self, color):
            self.background = color

        def add_axes(self, **kwargs):
            self.axes_color = kwargs.get("color")

        def clear(self):
            pass

        def add_volume(self, mesh, **_kwargs):
            self.added_volumes.append(mesh)

        def add_mesh(self, *_args, **_kwargs):
            pass

        def show_bounds(self, **_kwargs):
            self.bounds_kwargs = _kwargs

        def reset_camera(self):
            pass

        def render(self):
            pass

        def screenshot(self, path):
            self.screenshot_path = path

        def open_movie(self, _path, **_kwargs):
            self.mwriter = Writer()

        def write_frame(self):
            self.frames += 1

        def close(self):
            self.close_count += 1

    monkeypatch.setitem(sys.modules, "pyvistaqt", types.SimpleNamespace(QtInteractor=FakeInteractor))
    panel = QtVolumeViewerPanel([_volume_data()], dataset_names=["scan"])
    panel.resize(1000, 700)
    panel.show()
    app.processEvents()

    controls = panel.findChild(QtWidgets.QScrollArea, "volume_controls_scroll")
    controls_body = panel.findChild(QtWidgets.QWidget, "volume_controls_body")
    assert controls.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert controls_body.width() <= controls.viewport().width()
    assert controls.verticalScrollBar().maximum() > 0
    for curve_name in ("volume_color_curve", "volume_opacity_curve"):
        curve = panel.findChild(QtWidgets.QWidget, curve_name)
        assert curve.width() < controls.viewport().width()

    color = panel.findChild(QtWidgets.QComboBox, "volume_color_channel_combo")
    opacity = panel.findChild(QtWidgets.QComboBox, "volume_opacity_channel_combo")
    linked = panel.findChild(QtWidgets.QCheckBox, "volume_link_channels_check")
    linked.setChecked(False)
    opacity.setCurrentText("errors")
    assert color.currentText() == "signal"
    assert opacity.currentText() == "errors"
    assert opacity.isEnabled()
    assert panel.plotter.background == "white"
    assert panel.plotter.axes_color == "black"

    volume_count = len(panel.plotter.added_volumes)
    panel.data.mask[:] = True
    panel._channel_changed()
    assert panel.render_status.isVisible()
    assert "No finite voxels" in panel.render_status.text()
    assert len(panel.plotter.added_volumes) == volume_count
    panel.data.mask[:] = False
    panel._channel_changed()
    assert not panel.render_status.isVisible()
    assert len(panel.plotter.added_volumes) == volume_count + 1

    hidden_low = panel.findChild(QtWidgets.QDoubleSpinBox, "volume_hidden_0_low_spin")
    hidden_high = panel.findChild(QtWidgets.QDoubleSpinBox, "volume_hidden_0_high_spin")
    hidden_integrate = panel.findChild(QtWidgets.QCheckBox, "volume_hidden_0_integrate_check")
    hidden_slider = panel.findChild(QtWidgets.QWidget, "volume_hidden_0_range_slider")
    assert hidden_slider is not None
    assert hidden_slider.toolTip()
    hidden_low.setValue(1.5)
    assert hidden_integrate.isChecked()
    hidden_high.setValue(0.5)
    assert panel._selections()[0] == (0, 1)
    hidden_integrate.setChecked(False)
    hidden_slider.set_indices(1, 0, 1)
    hidden_slider.changed.emit("value")
    assert panel._selections()[0] == 1

    x_min = panel.findChild(QtWidgets.QDoubleSpinBox, "volume_x_min_spin")
    x_max = panel.findChild(QtWidgets.QDoubleSpinBox, "volume_x_max_spin")
    x_min.setValue(0.75)
    x_max.setValue(1.75)
    x_min.editingFinished.emit()
    assert panel._arrays().color.shape[0] == 1

    scale_mode = panel.findChild(QtWidgets.QComboBox, "volume_axis_scale_mode_combo")
    x_scale = panel.findChild(QtWidgets.QDoubleSpinBox, "volume_x_scale_spin")
    scale_mode.setCurrentText("Custom")
    x_scale.setValue(2.5)
    original_x_span = panel.current_grid.bounds.x_max - panel.current_grid.bounds.x_min
    rendered_x_span = panel.current_render_grid.bounds.x_max - panel.current_render_grid.bounds.x_min
    assert rendered_x_span == pytest.approx(2.5 * original_x_span)
    scale_mode.setCurrentText("Equal data units")
    equal_x_span = panel.current_render_grid.bounds.x_max - panel.current_render_grid.bounds.x_min
    assert equal_x_span == pytest.approx(original_x_span)
    assert "xlabel" not in panel.plotter.bounds_kwargs
    assert panel.plotter.bounds_kwargs["xtitle"] == "H"

    reset_limits = panel.findChild(QtWidgets.QPushButton, "volume_reset_axis_limits_button")
    reset_limits.click()
    y_smoothing = panel.findChild(QtWidgets.QDoubleSpinBox, "volume_y_smoothing_spin")
    y_smoothing.setValue(1.0)
    canonical = np.asarray(panel.current_grid.cell_data["color_channel"])
    rendered = np.asarray(panel.current_render_grid.cell_data["color_channel"])
    assert not np.allclose(canonical, rendered)

    names = (
        "volume_color_curve",
        "volume_opacity_curve",
        "volume_rotate_check",
        "volume_rotation_axis_combo",
        "volume_export_still_button",
        "volume_export_movie_button",
        "volume_export_button",
    )
    for name in names:
        widget = panel.findChild(QtWidgets.QWidget, name)
        assert widget is not None
        assert widget.toolTip()

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda _parent, title, *_args: (
            ("/private/tmp/view.png", "PNG image (*.png)")
            if "current" in title
            else ("/private/tmp/rotation.mp4", "MP4 movie (*.mp4)")
        ),
    )
    panel.export_still()
    rotation_axis = panel.findChild(QtWidgets.QComboBox, "volume_rotation_axis_combo")
    assert rotation_axis.currentText() == "Z axis"
    rotation_axis.setCurrentText("Vertical axis")
    panel.export_rotation_movie()
    assert panel.plotter.screenshot_path == "/private/tmp/view.png"
    assert panel.plotter.frames == 360
    assert panel.plotter.camera.total_azimuth == pytest.approx(360.0)
    assert panel.plotter.camera.up == (0.0, 1.0, 0.0)
    assert panel.plotter.mwriter is None
    panel.shutdown()
    panel.shutdown()
    assert panel.plotter.close_count == 1
    panel.close()
    app.processEvents()
