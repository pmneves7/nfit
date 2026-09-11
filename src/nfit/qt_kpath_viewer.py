"""Dedicated Qt viewer for experimental HKL path maps."""

from __future__ import annotations

from pprint import pformat
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from .application_preferences import default_continuous_colormap
from .brillouin_zone import build_brillouin_zone_scene
from .colormaps import IMAGE_COLORMAP_GROUPS, populate_qt_colormap_combo
from .crystal import lattice_vectors, primitive_lattice_vectors
from .kpath import KPathNode, plot_mdhisto_kpath, standard_kpath
from .mdhisto import MDHistoData


class QtKPathViewer:
    """Configure, render, and inspect a non-fitting momentum-path cut."""

    def __init__(
        self,
        data: MDHistoData,
        *,
        lattice_parameters: dict[str, float],
        spacegroup: str,
        parent: Any | None = None,
    ) -> None:
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.data = data
        self.lattice_parameters = dict(lattice_parameters)
        self.spacegroup = str(spacegroup)
        self._zone_windows: list[Any] = []
        self.window = QtWidgets.QMainWindow(parent)
        self.window.setWindowTitle("nfit K-path Viewer")
        self.window.resize(1450, 900)
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.window)
        self.toolbar.setToolTip("Pan, zoom, reset, configure, and save the K-path figure.")
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        layout.addWidget(plot_panel, 1)
        controls = QtWidgets.QScrollArea()
        controls.setWidgetResizable(True)
        controls.setFixedWidth(470)
        panel = QtWidgets.QWidget()
        controls.setWidget(panel)
        self.controls_layout = QtWidgets.QVBoxLayout(panel)
        self.controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(controls)
        self._build_path_controls()
        self._build_sampling_controls()
        self._build_presentation_controls()
        self._build_actions()
        try:
            self.set_nodes(standard_kpath(self.lattice_parameters, self.spacegroup))
            self.redraw()
        except (KeyError, TypeError, ValueError) as exc:
            self.status.setText(f"Choose or enter a valid path: {exc}")

    def _build_path_controls(self) -> None:
        group = QtWidgets.QGroupBox("Momentum path (absolute conventional HKL)")
        layout = QtWidgets.QVBoxLayout(group)
        self.path_table = QtWidgets.QTableWidget(0, 5)
        self.path_table.setHorizontalHeaderLabels(["Label", "H", "K", "L", "Break"])
        self.path_table.setToolTip(
            "Path nodes in absolute conventional-cell HKL coordinates. Values are not folded "
            "into the first zone, so higher-zone paths can be entered directly."
        )
        self.path_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.path_table)
        row = QtWidgets.QHBoxLayout()
        automatic = QtWidgets.QPushButton("Use standard path")
        automatic.setToolTip(
            "Replace the table with the Setyawan–Curtarolo path for the current Bravais lattice."
        )
        automatic.clicked.connect(self._set_standard_path)
        add = QtWidgets.QPushButton("Add node")
        add.setToolTip("Append an editable path node.")
        add.clicked.connect(lambda: self._append_node(KPathNode("K", (0.0, 0.0, 0.0))))
        remove = QtWidgets.QPushButton("Remove selected")
        remove.setToolTip("Remove the selected path-node rows.")
        remove.clicked.connect(self._remove_selected_nodes)
        row.addWidget(automatic)
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)
        self.controls_layout.addWidget(group)

    def _build_sampling_controls(self) -> None:
        group = QtWidgets.QGroupBox("Path sampling")
        form = QtWidgets.QFormLayout(group)
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.001, 2.0)
        self.step_spin.setDecimals(4)
        self.step_spin.setValue(0.03)
        self.step_spin.setSuffix(" Å⁻¹")
        self.step_spin.setToolTip("Approximate spacing of displayed samples along the physical path.")
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(0.001, 5.0)
        self.width_spin.setDecimals(4)
        self.width_spin.setValue(0.08)
        self.width_spin.setSuffix(" Å⁻¹")
        self.width_spin.setToolTip(
            "Physical radius of the transverse integration tube around the path."
        )
        self.minimum_spin = QtWidgets.QSpinBox()
        self.minimum_spin.setRange(1, 100000)
        self.minimum_spin.setValue(1)
        self.minimum_spin.setToolTip("Minimum contributing momentum voxels required per path-energy pixel.")
        form.addRow("Path step", self.step_spin)
        form.addRow("Tube radius", self.width_spin)
        form.addRow("Minimum voxels", self.minimum_spin)
        self.controls_layout.addWidget(group)

    def _build_presentation_controls(self) -> None:
        group = QtWidgets.QGroupBox("Presentation")
        form = QtWidgets.QFormLayout(group)
        self.label_mode = QtWidgets.QComboBox()
        self.label_mode.addItem("High-symmetry names", "names")
        self.label_mode.addItem("HKL coordinates", "coordinates")
        self.label_mode.addItem("Names and coordinates", "both")
        self.label_mode.setToolTip("Choose how path nodes are labelled along the momentum axis.")
        self.guides_check = QtWidgets.QCheckBox("Dashed vertical guides")
        self.guides_check.setChecked(True)
        self.guides_check.setToolTip("Draw a vertical guide at every configured path node.")
        self.guide_color = "#ffffff"
        self.guide_color_button = QtWidgets.QPushButton("#FFFFFF")
        self.guide_color_button.setToolTip("Choose the high-symmetry guide-line color.")
        self.guide_color_button.clicked.connect(self._choose_guide_color)
        self.guide_width = QtWidgets.QDoubleSpinBox()
        self.guide_width.setRange(0.1, 10.0)
        self.guide_width.setValue(1.2)
        self.guide_width.setToolTip("Guide-line thickness in points.")
        self.guide_alpha = QtWidgets.QDoubleSpinBox()
        self.guide_alpha.setRange(0.0, 1.0)
        self.guide_alpha.setSingleStep(0.05)
        self.guide_alpha.setValue(0.9)
        self.guide_alpha.setToolTip("Guide-line opacity.")
        self.cmap_combo = QtWidgets.QComboBox()
        populate_qt_colormap_combo(self.cmap_combo, IMAGE_COLORMAP_GROUPS)
        selected_cmap = self.cmap_combo.findText(default_continuous_colormap())
        if selected_cmap >= 0:
            self.cmap_combo.setCurrentIndex(selected_cmap)
        self.cmap_combo.setToolTip("Choose the intensity colormap.")
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["linear", "log", "symlog", "asinh", "power"])
        self.scale_combo.setToolTip("Choose the intensity color normalization.")
        self.color_alpha = QtWidgets.QDoubleSpinBox()
        self.color_alpha.setRange(-20.0, 20.0)
        self.color_alpha.setValue(0.0)
        self.color_alpha.setToolTip("Exponentially warp color placement without moving colorbar ticks.")
        form.addRow("Node labels", self.label_mode)
        form.addRow(self.guides_check)
        form.addRow("Guide color", self.guide_color_button)
        form.addRow("Guide thickness", self.guide_width)
        form.addRow("Guide opacity", self.guide_alpha)
        form.addRow("Colormap", self.cmap_combo)
        form.addRow("Color scale", self.scale_combo)
        form.addRow("Color alpha", self.color_alpha)
        self.controls_layout.addWidget(group)

    def _build_actions(self) -> None:
        update = QtWidgets.QPushButton("Update plot")
        update.setToolTip("Recalculate this visualization without changing the source dataset.")
        update.clicked.connect(self.redraw)
        view_3d = QtWidgets.QPushButton("View path in 3D")
        view_3d.setToolTip("Show the absolute-HKL path with the first Brillouin zone and reciprocal basis.")
        view_3d.clicked.connect(self.view_path_in_3d)
        copy = QtWidgets.QPushButton("Copy script")
        copy.setToolTip("Copy an editable backend-only script for this K-path plot.")
        copy.clicked.connect(self.copy_script)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(update)
        row.addWidget(view_3d)
        row.addWidget(copy)
        self.controls_layout.addLayout(row)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setToolTip("K-path calculation status or validation error.")
        self.controls_layout.addWidget(self.status)

    def _append_node(self, node: KPathNode) -> None:
        row = self.path_table.rowCount()
        self.path_table.insertRow(row)
        for column, value in enumerate((node.label, *node.hkl)):
            self.path_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        check = QtWidgets.QTableWidgetItem()
        check.setFlags(check.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        check.setCheckState(
            QtCore.Qt.CheckState.Checked if node.break_before else QtCore.Qt.CheckState.Unchecked
        )
        self.path_table.setItem(row, 4, check)

    def set_nodes(self, nodes: tuple[KPathNode, ...] | list[KPathNode]) -> None:
        self.path_table.setRowCount(0)
        for node in nodes:
            self._append_node(node)

    def nodes(self) -> tuple[KPathNode, ...]:
        result = []
        for row in range(self.path_table.rowCount()):
            items = [self.path_table.item(row, column) for column in range(5)]
            if any(item is None for item in items):
                raise ValueError(f"path row {row + 1} is incomplete")
            result.append(
                KPathNode(
                    items[0].text().strip(),
                    tuple(float(items[column].text()) for column in range(1, 4)),
                    bool(row and items[4].checkState() == QtCore.Qt.CheckState.Checked),
                )
            )
        if len(result) < 2:
            raise ValueError("enter at least two path nodes")
        return tuple(result)

    def _set_standard_path(self) -> None:
        try:
            self.set_nodes(standard_kpath(self.lattice_parameters, self.spacegroup))
            self.redraw()
        except (KeyError, TypeError, ValueError) as exc:
            self.status.setText(str(exc))

    def _remove_selected_nodes(self) -> None:
        rows = sorted({index.row() for index in self.path_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.path_table.removeRow(row)

    def _choose_guide_color(self) -> None:
        selected = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.guide_color), self.window)
        if selected.isValid():
            self.guide_color = selected.name()
            self.guide_color_button.setText(self.guide_color.upper())

    def redraw(self) -> bool:
        try:
            self.figure.clear()
            axis = self.figure.add_subplot(111)
            figure = plot_mdhisto_kpath(
                self.data,
                self.nodes(),
                step_inv_angstrom=self.step_spin.value(),
                transverse_width_inv_angstrom=self.width_spin.value(),
                minimum_voxels=self.minimum_spin.value(),
                lattice_parameters=self.lattice_parameters,
                label_mode=str(self.label_mode.currentData()),
                show_guides=self.guides_check.isChecked(),
                guide_color=self.guide_color,
                guide_linewidth=self.guide_width.value(),
                guide_alpha=self.guide_alpha.value(),
                cmap=self.cmap_combo.currentText(),
                color_scale=self.scale_combo.currentText(),
                color_alpha=self.color_alpha.value(),
                ax=axis,
            )
            self._prepared = figure._nfit_kpath_data
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self.status.setText(f"Could not draw K path: {exc}")
            self.canvas.draw_idle()
            return False
        self.status.setText(
            f"{len(self._prepared.metadata['kpath']['points_hkl'])} path samples; "
            f"source data unchanged."
        )
        self.canvas.draw_idle()
        return True

    def view_path_in_3d(self) -> bool:
        from .qt_brillouin_zone_viewer import show_brillouin_zone_scene

        try:
            direct = lattice_vectors(self.lattice_parameters)
            primitive = primitive_lattice_vectors(self.lattice_parameters, self.spacegroup)
            reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
            scene = build_brillouin_zone_scene(
                direct,
                [
                    {"label": node.label, "k": list(node.hkl), "break_before": node.break_before}
                    for node in self.nodes()
                ],
                primitive_lattice=primitive,
                coordinate_reciprocal_lattice=reciprocal,
            )
            window = show_brillouin_zone_scene(scene, parent=self.window)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self.status.setText(f"Could not open 3D path: {exc}")
            return False
        self._zone_windows.append(window)
        return True

    def script(self) -> str:
        nodes = [
            {"label": node.label, "hkl": list(node.hkl), "break_before": node.break_before}
            for node in self.nodes()
        ]
        source = self.data.metadata.get("source_file")
        data_line = (
            f"data = load_mantid_mdhisto_nxs({source!r}, copy_metadata=False)"
            if source else "data = ...  # Replace with your MDHistoData object"
        )
        return "\n".join(
            [
                "import matplotlib.pyplot as plt",
                "from nfit import load_mantid_mdhisto_nxs, plot_mdhisto_kpath",
                "",
                data_line,
                f"nodes = {pformat(nodes, sort_dicts=False)}",
                "fig = plot_mdhisto_kpath(",
                "    data, nodes,",
                f"    step_inv_angstrom={self.step_spin.value()!r},",
                f"    transverse_width_inv_angstrom={self.width_spin.value()!r},",
                f"    minimum_voxels={self.minimum_spin.value()!r},",
                f"    lattice_parameters={self.lattice_parameters!r},",
                f"    label_mode={str(self.label_mode.currentData())!r},",
                f"    show_guides={self.guides_check.isChecked()!r},",
                f"    guide_color={self.guide_color!r},",
                f"    guide_linewidth={self.guide_width.value()!r},",
                f"    guide_alpha={self.guide_alpha.value()!r},",
                f"    cmap={self.cmap_combo.currentText()!r},",
                f"    color_scale={self.scale_combo.currentText()!r},",
                f"    color_alpha={self.color_alpha.value()!r},",
                ")",
                "plt.show()",
                "",
            ]
        )

    def copy_script(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.script())
        self.status.setText("Copied editable K-path script.")

    def show(self) -> QtKPathViewer:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self
