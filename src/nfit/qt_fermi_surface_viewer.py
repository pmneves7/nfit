"""GPU-accelerated Qt/PyVista rendering for three-dimensional Fermi surfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .electronic_structure import (
    FermiSurfaceResult,
    electronic_energy_from_meV,
    normalize_electronic_energy_unit,
)
from .qt_electronic_viewer import populate_electronic_calculation_settings
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
    reset_camera: bool = True,
) -> None:
    """Render a three-dimensional result with VTK's interactive GPU pipeline."""

    if result.dimension != 3:
        raise ValueError("the PyVista Fermi-surface viewer requires a 3D result")
    plotter.clear()
    plotter.set_background("white")
    if hasattr(plotter, "enable_lightkit"):
        plotter.enable_lightkit()
    for sheet in result.sheets:
        color = _SHEET_COLORS[sheet.band_index % len(_SHEET_COLORS)]
        plotter.add_mesh(
            _sheet_mesh(sheet.vertices_reduced, sheet.connectivity),
            color=color,
            opacity=0.65,
            show_edges=False,
            smooth_shading=False,
            label=f"band {sheet.band_index}",
        )
    plotter.show_bounds(
        bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        xtitle="k₁ (r.l.u.)",
        ytitle="k₂ (r.l.u.)",
        ztitle="k₃ (r.l.u.)",
        color="black",
        grid="back",
        location="outer",
        all_edges=True,
        use_3d_text=False,
    )
    if result.sheets:
        plotter.add_legend(
            bcolor="white",
            border=True,
            background_opacity=0.85,
        )
    unit = normalize_electronic_energy_unit(
        result.provenance.get("display_energy_unit", "eV")
    )
    target = float(electronic_energy_from_meV(result.target_energy_meV, unit))
    plotter.add_text(
        f"Constant-energy surface at E = {target:g} {unit}",
        position="upper_edge",
        color="black",
        font_size=12,
    )
    if reset_camera:
        plotter.view_isometric()
        plotter.reset_camera()
    elif hasattr(plotter, "render"):
        plotter.render()


def _populate_settings_panel(
    settings: Any,
    *,
    result: FermiSurfaceResult,
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
            "The 3D surface uses GPU-accelerated PyVista rendering. "
            "Additional display controls will be added here."
        )
    triangle_count = sum(
        int(sheet.connectivity.shape[0]) for sheet in result.sheets
    )
    details = QtWidgets.QLabel(
        f"{len(result.sheets)} band sheet(s)\n"
        f"{triangle_count:,} displayed triangles",
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
) -> Any:
    """Open a GPU-accelerated Qt window for a three-dimensional Fermi surface."""

    from PySide6 import QtGui, QtWidgets
    from pyvistaqt import MainWindow, QtInteractor

    if result.dimension != 3:
        raise ValueError("the PyVista Fermi-surface viewer requires a 3D result")
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
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
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
    window._nfit_copy_figure = copy_figure
    window._nfit_save_figure = save_figure
    close_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Close,
        window,
    )
    close_shortcut.activated.connect(window.close)
    window._nfit_close_shortcut = close_shortcut
    _populate_settings_panel(
        settings,
        result=result,
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
    show_then_render(
        window,
        lambda: _render_fermi_surface(plotter, result),
    )
    return window
