from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from nfit import qt_comparison_colors
from nfit.plotting_core import MDHistoSliceViewer
from nfit.qt_comparison_colors import DiffColorController
from nfit.qt_slice_controls import _qt_app
from nfit.slice_comparison import default_difference_colors


@pytest.fixture
def color_controls():
    from PySide6 import QtWidgets

    app = _qt_app()
    widget = QtWidgets.QWidget()
    calls = []
    viewer = SimpleNamespace(
        model=SimpleNamespace(
            COLOR_SCALES=MDHistoSliceViewer.COLOR_SCALES,
            AUTO_LIMITS=MDHistoSliceViewer.AUTO_LIMITS,
            cmap="viridis",
            color_scale="log",
        ),
        _background_panels_active=lambda: True,
        update_plot=lambda **kwargs: calls.append(kwargs),
    )
    controller = DiffColorController(viewer)
    controller.build(QtWidgets.QVBoxLayout(widget))
    yield controller, viewer, calls
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_difference_colors_defaults_and_widget_tooltips(color_controls):
    controller, viewer, calls = color_controls

    assert controller.snapshot() == default_difference_colors()
    assert viewer.diff_color_group.title() == "Diff color"
    assert viewer.diff_cmap_combo.currentText() == "bwr"
    assert viewer.diff_scale_combo.currentText() == "linear"
    assert viewer.diff_symmetric_about_zero_check.isChecked()
    assert viewer.diff_autoscale_check.isChecked()
    assert calls == []
    for name, control in controller._controls.items():
        if name.endswith(("combo", "button", "check", "spin")):
            assert control.toolTip(), name


def test_difference_manual_limits_and_autoscale_are_independent(color_controls):
    controller, viewer, calls = color_controls

    controller.sync_limits(-4, 4)
    assert calls == []
    assert controller.snapshot()["manual_vmin"] is None
    viewer.diff_vmax_spin.setValue(3)
    settings = controller.snapshot()
    assert settings["manual_vmin"] == -3
    assert settings["manual_vmax"] == 3
    assert settings["autoscale"] is False
    assert viewer.diff_vmin_spin.value() == -3
    assert not viewer.diff_autoscale_check.isChecked()
    assert viewer.model.cmap == "viridis"
    assert viewer.model.color_scale == "log"
    assert calls == [{"reuse_slice": True}]

    viewer.diff_symmetric_about_zero_check.setChecked(False)
    viewer.diff_vmin_spin.setValue(-2)
    assert controller.snapshot()["manual_vmin"] == -2
    assert controller.snapshot()["manual_vmax"] == 3
    viewer.diff_autoscale_check.setChecked(True)
    assert controller.snapshot()["autoscale"]


def test_difference_color_controls_restore_and_apply(color_controls):
    controller, viewer, calls = color_controls

    viewer.diff_cmap_combo.setCurrentText("coolwarm")
    viewer.diff_cmap_reverse_button.click()
    viewer.diff_scale_combo.setCurrentText("power")
    viewer.diff_gamma_spin.setValue(1.5)
    viewer.diff_alpha_spin.setValue(2)
    viewer.diff_limits_combo.setCurrentText("Nth percentile")
    viewer.diff_limit_n_spin.setValue(5)
    saved = controller.snapshot()
    assert saved["cmap"] == "coolwarm"
    assert saved["cmap_reversed"]
    assert saved["color_scale"] == "power"
    assert saved["power_gamma"] == 1.5
    assert saved["color_alpha"] == 2
    assert saved["percentile_n"] == 5
    assert saved["autoscale"]
    assert not viewer.diff_gamma_spin.isHidden()
    assert not viewer.diff_limit_n_spin.isHidden()
    calls.clear()

    controller.restore({})
    assert controller.snapshot() == default_difference_colors()
    assert viewer.diff_gamma_spin.isHidden()
    assert viewer.diff_limit_n_spin.isHidden()
    controller.restore(saved)
    model = SimpleNamespace()
    controller.apply_to_model(model)
    assert vars(model) == saved
    assert controller.snapshot() == saved
    saved["cmap"] = "gray"
    assert controller.snapshot()["cmap"] == "coolwarm"
    assert calls == []


def test_difference_color_visibility_tracks_comparison_mode(color_controls):
    controller, viewer, _calls = color_controls

    assert not viewer.diff_color_group.isHidden()
    viewer._background_panels_active = lambda: False
    controller.sync_visibility()
    assert viewer.diff_color_group.isHidden()


def test_difference_colors_service_has_no_gui_coordinator_dependency():
    tree = ast.parse(Path(qt_comparison_colors.__file__).read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imports.intersection({"qt_slice_viewer", "project_gui", "qt_slice_builder"})
