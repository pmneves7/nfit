"""GPU-accelerated Qt/PyVista rendering for three-dimensional Fermi surfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .electronic_structure import FermiSurfaceResult
from .file_dialogs import get_save_file_name
from .qt_electronic_viewer import (
    populate_electronic_calculation_settings,
    refresh_electronic_calculation_settings,
)
from .qt_pyvista import configure_pyvista_interactor, show_then_render
from .qt_viewer_shell import create_viewer_shell

_SHEET_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)


@dataclass(frozen=True)
class FermiSurfaceViewOptions:
    """Scriptable presentation settings for the 3D Fermi-surface viewer."""

    band_opacity: float = 0.65
    text_size: int = 14
    grid_line_width: float = 1.0
    show_legend: bool = True
    shading: str = "smooth"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.band_opacity) <= 1.0:
            raise ValueError("band_opacity must be between zero and one")
        if int(self.text_size) < 1:
            raise ValueError("text_size must be positive")
        if float(self.grid_line_width) <= 0.0:
            raise ValueError("grid_line_width must be positive")
        shading = str(self.shading).strip().lower()
        if shading not in {"smooth", "flat"}:
            raise ValueError("shading must be smooth or flat")
        object.__setattr__(self, "band_opacity", float(self.band_opacity))
        object.__setattr__(self, "text_size", int(self.text_size))
        object.__setattr__(
            self,
            "grid_line_width",
            float(self.grid_line_width),
        )
        object.__setattr__(self, "shading", shading)


def _mesh_summary(result: FermiSurfaceResult) -> str:
    triangle_count = sum(
        int(sheet.connectivity.shape[0]) for sheet in result.sheets
    )
    return (
        f"{len(result.sheets)} band sheet(s)\n"
        f"{triangle_count:,} displayed triangles"
    )


def _sheet_mesh(vertices: Any, connectivity: Any) -> Any:
    """Return a PyVista triangle mesh without changing the scientific result."""

    import pyvista as pv

    triangles = np.asarray(connectivity, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("a 3D Fermi sheet must contain triangular connectivity")
    faces = np.column_stack(
        (np.full(triangles.shape[0], 3, dtype=np.int64), triangles)
    ).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=float), faces=faces)


def _render_fermi_surface(
    plotter: Any,
    result: FermiSurfaceResult,
    *,
    options: FermiSurfaceViewOptions | None = None,
    reset_camera: bool = True,
) -> None:
    """Render a three-dimensional result with VTK's interactive GPU pipeline."""

    if result.dimension != 3:
        raise ValueError("the PyVista Fermi-surface viewer requires a 3D result")
    settings = FermiSurfaceViewOptions() if options is None else options
    plotter.clear()
    plotter.set_background("white")
    if hasattr(plotter, "enable_lightkit"):
        plotter.enable_lightkit()
    for sheet in result.sheets:
        color = _SHEET_COLORS[sheet.band_index % len(_SHEET_COLORS)]
        plotter.add_mesh(
            _sheet_mesh(sheet.vertices_reduced, sheet.connectivity),
            color=color,
            opacity=settings.band_opacity,
            show_edges=False,
            smooth_shading=settings.shading == "smooth",
            label=f"band {sheet.band_index}",
        )
    bounds_actor = plotter.show_bounds(
        bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        xtitle="k₁ (r.l.u.)",
        ytitle="k₂ (r.l.u.)",
        ztitle="k₃ (r.l.u.)",
        color="black",
        grid="back",
        location="outer",
        all_edges=True,
        use_3d_text=False,
        font_size=settings.text_size,
    )
    for axis in ("X", "Y", "Z"):
        for property_name in (
            f"Get{axis}AxesGridlinesProperty",
            f"Get{axis}AxesLinesProperty",
        ):
            getter = getattr(bounds_actor, property_name, None)
            if getter is not None:
                getter().SetLineWidth(settings.grid_line_width)
    if result.sheets and settings.show_legend:
        legend = plotter.add_legend(
            bcolor="white",
            border=True,
            background_opacity=0.85,
        )
        text_property = getattr(legend, "GetEntryTextProperty", None)
        if text_property is not None:
            text_property().SetFontSize(settings.text_size)
    if reset_camera:
        plotter.view_isometric()
        plotter.reset_camera()
    elif hasattr(plotter, "render"):
        plotter.render()


def _populate_settings_panel(
    settings: Any,
    *,
    result: FermiSurfaceResult,
    initial: FermiSurfaceViewOptions,
    on_change: Callable[[FermiSurfaceViewOptions], None],
    copy_figure: Any,
    save_figure: Any,
) -> None:
    from PySide6 import QtWidgets

    layout = settings.layout()
    placeholder = settings.findChild(
        QtWidgets.QLabel,
        "fermi_surface_settings_placeholder",
    )
    if placeholder is not None:
        placeholder.setText(
            "The 3D surface uses GPU-accelerated PyVista rendering."
        )
    state = {"options": initial}

    def update(**changes: Any) -> None:
        state["options"] = replace(state["options"], **changes)
        on_change(state["options"])

    display = QtWidgets.QGroupBox("Surface and axes", settings)
    display.setObjectName("fermi_surface_display_group")
    display_layout = QtWidgets.QFormLayout(display)

    opacity = QtWidgets.QDoubleSpinBox(display)
    opacity.setObjectName("fermi_surface_band_opacity")
    opacity.setRange(0.0, 1.0)
    opacity.setDecimals(2)
    opacity.setSingleStep(0.05)
    opacity.setValue(initial.band_opacity)
    opacity.setToolTip(
        "Set the opacity shared by all displayed band sheets."
    )
    opacity.valueChanged.connect(
        lambda value: update(band_opacity=float(value))
    )
    display_layout.addRow("Band opacity", opacity)

    text_size = QtWidgets.QSpinBox(display)
    text_size.setObjectName("fermi_surface_text_size")
    text_size.setRange(6, 72)
    text_size.setValue(initial.text_size)
    text_size.setToolTip(
        "Set reciprocal-axis title, tick-label, and legend text size."
    )
    text_size.valueChanged.connect(
        lambda value: update(text_size=int(value))
    )
    display_layout.addRow("Text size", text_size)

    grid_width = QtWidgets.QDoubleSpinBox(display)
    grid_width.setObjectName("fermi_surface_grid_line_width")
    grid_width.setRange(0.25, 10.0)
    grid_width.setDecimals(2)
    grid_width.setSingleStep(0.25)
    grid_width.setValue(initial.grid_line_width)
    grid_width.setToolTip(
        "Set the width of reciprocal-cell axes, bounds, and grid lines."
    )
    grid_width.valueChanged.connect(
        lambda value: update(grid_line_width=float(value))
    )
    display_layout.addRow("Grid-line width", grid_width)

    shading = QtWidgets.QComboBox(display)
    shading.setObjectName("fermi_surface_shading")
    shading.setToolTip(
        "Smooth shading interpolates vertex normals across triangles; flat "
        "shading preserves visibly faceted triangles."
    )
    shading.addItem("Smooth", "smooth")
    shading.addItem("Flat / faceted", "flat")
    shading.setCurrentIndex(max(shading.findData(initial.shading), 0))
    shading.currentIndexChanged.connect(
        lambda _index, combo=shading: update(
            shading=str(combo.currentData())
        )
    )
    display_layout.addRow("Shading", shading)

    legend = QtWidgets.QCheckBox("Show legend", display)
    legend.setObjectName("fermi_surface_show_legend")
    legend.setChecked(initial.show_legend)
    legend.setToolTip("Show or hide the band-sheet legend.")
    legend.toggled.connect(
        lambda checked: update(show_legend=bool(checked))
    )
    display_layout.addRow("", legend)
    layout.insertWidget(max(layout.count() - 1, 0), display)

    details = QtWidgets.QLabel(
        _mesh_summary(result),
        settings,
    )
    details.setObjectName("fermi_surface_mesh_summary")
    details.setToolTip(
        "Mesh size extracted from the configured reciprocal-space sampling grid."
    )
    layout.insertWidget(max(layout.count() - 1, 0), details)

    output = QtWidgets.QGroupBox("Output", settings)
    output.setObjectName("fermi_surface_output_group")
    output_layout = QtWidgets.QHBoxLayout(output)
    copy_button = QtWidgets.QPushButton("Copy figure", output)
    copy_button.setObjectName("fermi_surface_copy_figure")
    copy_button.setToolTip("Copy the current Fermi-surface viewport to the clipboard.")
    copy_button.clicked.connect(copy_figure)
    save_button = QtWidgets.QPushButton("Save figure", output)
    save_button.setObjectName("fermi_surface_save_figure")
    save_button.setToolTip("Save the current Fermi-surface viewport as an image.")
    save_button.clicked.connect(save_figure)
    output_layout.addWidget(copy_button)
    output_layout.addWidget(save_button)
    layout.insertWidget(max(layout.count() - 1, 0), output)


def show_fermi_surface_result(
    result: FermiSurfaceResult,
    *,
    parent: Any | None = None,
    settings_config: Mapping[str, Any] | None = None,
    on_apply_settings: Callable[[dict[str, str]], None] | None = None,
    view_options: FermiSurfaceViewOptions | None = None,
) -> Any:
    """Open a GPU-accelerated Qt window for a three-dimensional Fermi surface."""

    from PySide6 import QtGui, QtWidgets
    from pyvistaqt import MainWindow, QtInteractor

    if result.dimension != 3:
        raise ValueError("the PyVista Fermi-surface viewer requires a 3D result")
    initial_options = (
        FermiSurfaceViewOptions() if view_options is None else view_options
    )
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    window = MainWindow(parent=parent, title="Fermi surface")
    window.setObjectName("fermi_surface_viewer")
    central, viewport_layout, settings = create_viewer_shell(
        QtWidgets,
        viewer_key="fermi_surface",
    )
    plotter = QtInteractor(central, auto_update=False)
    configure_pyvista_interactor(plotter)
    plotter.setObjectName("fermi_surface_plotter")
    viewport_layout.addWidget(plotter.interactor)
    window.setCentralWidget(central)
    window.resize(1100, 760)
    window.signal_close.connect(plotter.close)

    def copy_figure() -> None:
        image = np.ascontiguousarray(plotter.screenshot(return_img=True))
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("PyVista returned an unsupported screenshot format")
        height, width, channels = image.shape
        image_format = (
            QtGui.QImage.Format.Format_RGB888
            if channels == 3
            else QtGui.QImage.Format.Format_RGBA8888
        )
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            image_format,
        ).copy()
        QtWidgets.QApplication.clipboard().setImage(qimage)

    def save_figure() -> None:
        path, _selected = get_save_file_name(
            window,
            "Save Fermi-surface figure",
            "nfit_fermi_surface.png",
            "Images (*.png *.jpg *.jpeg)",
        )
        if path:
            plotter.screenshot(str(path))

    window._nfit_plotter = plotter
    window._nfit_application = application
    window._nfit_settings_panel = settings
    window._nfit_result = result
    window._nfit_view_options = initial_options
    window._nfit_copy_figure = copy_figure
    window._nfit_save_figure = save_figure
    close_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Close,
        window,
    )
    close_shortcut.activated.connect(window.close)
    window._nfit_close_shortcut = close_shortcut
    def update_view_options(options: FermiSurfaceViewOptions) -> None:
        window._nfit_view_options = options
        _render_fermi_surface(
            plotter,
            window._nfit_result,
            options=options,
            reset_camera=False,
        )

    _populate_settings_panel(
        settings,
        result=result,
        initial=initial_options,
        on_change=update_view_options,
        copy_figure=copy_figure,
        save_figure=save_figure,
    )
    if settings_config is not None and on_apply_settings is not None:
        populate_electronic_calculation_settings(
            settings,
            viewer_key="fermi_surface",
            values=settings_config,
            on_apply=on_apply_settings,
        )

        def update_calculation_settings(values: Mapping[str, Any]) -> None:
            refresh_electronic_calculation_settings(
                settings,
                viewer_key="fermi_surface",
                values=values,
            )

        window._nfit_update_calculation_settings = update_calculation_settings

    def replace_result(updated: FermiSurfaceResult) -> None:
        if updated.dimension != 3:
            raise ValueError(
                "the PyVista Fermi-surface viewer requires a 3D result"
            )
        window._nfit_result = updated
        summary = settings.findChild(
            QtWidgets.QLabel,
            "fermi_surface_mesh_summary",
        )
        if summary is not None:
            summary.setText(_mesh_summary(updated))
        _render_fermi_surface(
            plotter,
            updated,
            options=window._nfit_view_options,
            reset_camera=False,
        )

    window._nfit_replace_result = replace_result
    show_then_render(
        window,
        lambda: _render_fermi_surface(
            plotter,
            result,
            options=window._nfit_view_options,
        ),
    )
    return window
