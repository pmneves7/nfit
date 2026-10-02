import numpy as np
import pytest

from nfit.mdhisto import MDHistoAxis, MDHistoData


@pytest.mark.parametrize("angle", [0., 25.])
def test_held_box_settings_survive_channel_binning_and_reordered_dataset(monkeypatch, angle):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    datasets = []
    for source, binning, names in (
        ("first", "Default", ("H", "E")),
        ("first", "Coarse", ("H", "E")),
        ("second", "Default", ("E", "H")),
    ):
        axes = tuple(MDHistoAxis(name, np.arange(5.), "", "unknown") for name in names)
        signal = np.arange(16.).reshape(4, 4) + 1
        datasets.append(MDHistoData(axes, signal, np.ones_like(signal),
            np.zeros_like(signal, dtype=bool), np.ones_like(signal),
            metadata={"source_dataset_name": source, "binning_name": binning}))
    viewer = QtMDHistoSliceViewer(datasets, x_dim=0, y_dim=1)
    try:
        assert viewer.hold_view_settings
        assert viewer.hold_view_settings_check.isChecked()
        viewer.show_box_check.setChecked(True)
        viewer.hist_axes_check.setChecked(True)
        viewer.roi_button.setChecked(True)
        viewer.popout_cuts_check.setChecked(angle != 0.)
        viewer.roi_angle_spin.setValue(angle)
        viewer._set_roi_extents((.7, 1.9, .8, 2.6), update_cuts=True, draw=False)
        viewer.xcut_percent_slider.setValue(30)
        viewer.ycut_percent_slider.setValue(25)
        expected = viewer._roi_extents
        for control, text in ((viewer.channel_combo, "errors"),
                              (viewer.binning_combo, "Coarse"),
                              (viewer.dataset_combo, "second"),
                              (viewer.channel_combo, "signal"),
                              (viewer.dataset_combo, "first")):
            control.setCurrentText(text)
            np.testing.assert_allclose(viewer._roi_extents, expected)
            assert viewer._roi_angle == angle
            assert viewer.show_box_check.isChecked()
            assert viewer.hist_axes_check.isChecked()
            assert viewer.roi_button.isChecked()
            assert viewer.popout_cuts_check.isChecked() == (angle != 0.)
            assert viewer.xcut_percent == 30
            assert viewer.ycut_percent == 25
            assert viewer.roi_x_width_spin.value() == pytest.approx(expected[1] - expected[0])
            assert viewer.roi_y_width_spin.value() == pytest.approx(expected[3] - expected[2])
    finally:
        viewer.window.close()
