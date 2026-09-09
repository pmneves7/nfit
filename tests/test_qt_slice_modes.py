from __future__ import annotations

import ast
import inspect
from pathlib import Path

import nfit.qt_slice_modes as slice_modes
import nfit.qt_slice_viewer as qt_slice_viewer
from nfit.qt_slice_modes import (
    FitComparisonController,
    PlotLayoutController,
    StandardSliceController,
    TiledSliceController,
    WaterfallController,
)
from nfit.qt_slice_viewer import QtMDHistoSliceViewer


def test_slice_mode_controllers_forward_shared_viewer_state():
    class ViewerState:
        marker_size = 5.0

        def redraw(self):
            return "drawn"

    viewer = ViewerState()
    controller = StandardSliceController(viewer)

    assert controller.marker_size == 5.0
    assert controller.redraw() == "drawn"
    controller.marker_size = 7.0
    assert viewer.marker_size == 7.0


def test_viewer_mode_entry_points_are_thin_compatibility_wrappers():
    controllers = {
        "_ensure_standard_plot_layout": PlotLayoutController,
        "_draw_tiled_view": TiledSliceController,
        "_draw_waterfall_view": WaterfallController,
        "_draw_fit_panels_view": FitComparisonController,
        "_draw_1d_view": StandardSliceController,
    }

    for method_name, controller_type in controllers.items():
        wrapper_source = inspect.getsource(
            getattr(QtMDHistoSliceViewer, method_name)
        )
        implementation_source = inspect.getsource(
            getattr(controller_type, method_name)
        )

        assert "_mode_controllers" in wrapper_source
        assert len(wrapper_source.splitlines()) <= 16
        assert len(implementation_source.splitlines()) > len(
            wrapper_source.splitlines()
        )


def test_mode_controllers_resolve_legacy_viewer_helpers_at_call_time(
    monkeypatch,
):
    sentinel = object()
    controller = TiledSliceController(
        object(),
        helpers=qt_slice_viewer.__dict__,
    )

    monkeypatch.setattr(
        qt_slice_viewer,
        "prepare_mdhisto_tiled_slices",
        sentinel,
    )

    assert controller._helper("prepare_mdhisto_tiled_slices") is sentinel


def test_extracted_mode_calls_preserve_public_viewer_dispatch() -> None:
    path = Path(slice_modes.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    helper_names = {
        "default_tiled_slice_step",
        "default_waterfall_offset",
        "default_waterfall_step",
        "draw_waterfall_traces",
        "inverse_variance_weighted_profile",
        "prepare_mdhisto_tiled_slices",
        "prepare_mdhisto_waterfall",
        "waterfall_colors",
    }

    for class_node in (
        node for node in tree.body if isinstance(node, ast.ClassDef)
    ):
        method_names = {
            node.name
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }
        for node in ast.walk(class_node):
            if not isinstance(node, ast.Call):
                continue
            assert not (
                isinstance(node.func, ast.Name)
                and node.func.id in helper_names
            )
            assert not (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in method_names
                and node.func.attr != "_helper"
            )

    initializer = inspect.getsource(QtMDHistoSliceViewer.__init__)
    assert "SliceModeControllers(self, helpers=globals())" in initializer
