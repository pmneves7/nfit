"""Qt windows for Matplotlib electronic-structure figures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from .model_plots import ElectronicPlotStyle, apply_electronic_plot_style
from .qt_viewer_shell import create_viewer_shell

_VIEWER_TITLES = {
    "band_structure": "Band structure",
    "density_of_states": "Density of states",
    "fermi_surface": "Fermi surface",
}

_MARKERS = (
    ("None", ""),
    ("Circle", "o"),
    ("Square", "s"),
    ("Triangle", "^"),
    ("Diamond", "D"),
    ("Plus", "+"),
    ("Cross", "x"),
)
_LINE_STYLES = (
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("Dash-dot", "-."),
)


def _scrollable_settings_content(settings: Any, viewer_key: str) -> Any:
    """Install a compact scroll area inside a fixed-width settings panel."""

    from PySide6 import QtCore, QtWidgets

    layout = settings.layout()
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
    scroll = QtWidgets.QScrollArea(settings)
    scroll.setObjectName(f"{viewer_key}_settings_scroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    content = QtWidgets.QWidget(scroll)
    content.setObjectName(f"{viewer_key}_settings_content")
    content_layout = QtWidgets.QVBoxLayout(content)
    placeholder = QtWidgets.QLabel(
        "Adjust the figure presentation below.",
        content,
    )
    placeholder.setObjectName(f"{viewer_key}_settings_placeholder")
    placeholder.setWordWrap(True)
    content_layout.addWidget(placeholder)
    content_layout.addStretch(1)
    scroll.setWidget(content)
    layout.addWidget(scroll)
    return content


def populate_electronic_plot_settings(
    settings: Any,
    *,
    viewer_key: str,
    initial: ElectronicPlotStyle,
    on_change: Callable[[ElectronicPlotStyle], None],
) -> dict[str, Any]:
    """Populate live, scriptable styling controls for an electronic figure."""

    from PySide6 import QtGui, QtWidgets

    layout = settings.layout()
    state = {"style": initial}
    controls: dict[str, Any] = {}

    def update(**changes: Any) -> None:
        state["style"] = replace(state["style"], **changes)
        on_change(state["style"])

    def color_button(
        field: str,
        object_name: str,
        tooltip: str,
    ) -> Any:
        def button_style(value: str) -> str:
            color = QtGui.QColor(value)
            foreground = "white" if color.lightness() < 128 else "black"
            return (
                f"QPushButton {{ background-color: {value}; "
                f"color: {foreground}; }}"
            )

        value = str(getattr(initial, field))
        button = QtWidgets.QPushButton(value.upper(), settings)
        button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setStyleSheet(button_style(value))

        def choose() -> None:
            current = str(getattr(state["style"], field))
            selected = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(current),
                settings,
                "Choose color",
            )
            if not selected.isValid():
                return
            value = selected.name().upper()
            button.setText(value)
            button.setStyleSheet(button_style(value))
            update(**{field: value})

        button.clicked.connect(choose)
        controls[field] = button
        return button

    def width_spin(
        field: str,
        object_name: str,
        tooltip: str,
        *,
        maximum: float = 10.0,
        minimum: float = 0.0,
    ) -> Any:
        spin = QtWidgets.QDoubleSpinBox(settings)
        spin.setObjectName(object_name)
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setValue(float(getattr(initial, field)))
        spin.setSuffix(" pt")
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(
            lambda value, field=field: update(**{field: float(value)})
        )
        controls[field] = spin
        return spin

    curve = QtWidgets.QGroupBox("Curve and markers", settings)
    curve.setObjectName(f"{viewer_key}_curve_style_group")
    curve_layout = QtWidgets.QFormLayout(curve)
    curve_layout.addRow(
        "Line color",
        color_button(
            "line_color",
            f"{viewer_key}_line_color",
            "Choose the color of electronic band or density-of-states curves.",
        ),
    )
    curve_layout.addRow(
        "Line width",
        width_spin(
            "line_width",
            f"{viewer_key}_line_width",
            "Set the thickness of electronic curves.",
        ),
    )
    marker = QtWidgets.QComboBox(curve)
    marker.setObjectName(f"{viewer_key}_marker")
    marker.setToolTip("Choose a marker shape for electronic curves, or none.")
    for label, value in _MARKERS:
        marker.addItem(label, value)
    marker.setCurrentIndex(max(marker.findData(initial.marker), 0))
    marker.currentIndexChanged.connect(
        lambda _index: update(marker=str(marker.currentData()))
    )
    controls["marker"] = marker
    marker_size = width_spin(
        "marker_size",
        f"{viewer_key}_marker_size",
        "Set the marker size.",
        maximum=50.0,
    )
    marker_fill = color_button(
        "marker_face_color",
        f"{viewer_key}_marker_fill_color",
        "Choose the marker fill color. The default is no fill.",
    )
    clear_fill = QtWidgets.QPushButton("None", curve)
    clear_fill.setObjectName(f"{viewer_key}_marker_fill_none")
    clear_fill.setToolTip("Remove marker fill so the marker interior is transparent.")

    def remove_fill() -> None:
        marker_fill.setText("NONE")
        marker_fill.setStyleSheet("")
        update(marker_face_color="none")

    clear_fill.clicked.connect(remove_fill)
    fill_row = QtWidgets.QWidget(curve)
    fill_layout = QtWidgets.QHBoxLayout(fill_row)
    fill_layout.setContentsMargins(0, 0, 0, 0)
    fill_layout.addWidget(marker_fill, 1)
    fill_layout.addWidget(clear_fill)
    curve_layout.addRow("Marker", marker)
    curve_layout.addRow("Marker size", marker_size)
    curve_layout.addRow("Marker fill", fill_row)
    layout.insertWidget(max(layout.count() - 1, 0), curve)

    figure = QtWidgets.QGroupBox("Text, frame, and legend", settings)
    figure.setObjectName(f"{viewer_key}_figure_style_group")
    figure_layout = QtWidgets.QFormLayout(figure)
    font_size = width_spin(
        "font_size",
        f"{viewer_key}_font_size",
        "Set axes-label, tick-label, and legend font sizes.",
        maximum=48.0,
        minimum=4.0,
    )
    border_width = width_spin(
        "border_width",
        f"{viewer_key}_border_width",
        "Set axes-border, tick, and legend-outline thickness.",
    )
    show_legend = QtWidgets.QCheckBox("Show legend", figure)
    show_legend.setObjectName(f"{viewer_key}_show_legend")
    show_legend.setChecked(initial.show_legend)
    show_legend.setToolTip("Show or hide the plot legend.")
    show_legend.toggled.connect(
        lambda checked: update(show_legend=bool(checked))
    )
    controls["show_legend"] = show_legend
    figure_layout.addRow("Font size", font_size)
    figure_layout.addRow("Border width", border_width)
    figure_layout.addRow("", show_legend)
    layout.insertWidget(max(layout.count() - 1, 0), figure)

    def line_group(
        title: str,
        prefix: str,
        color_field: str,
        width_field: str,
        style_field: str,
        tooltip_name: str,
    ) -> Any:
        group = QtWidgets.QGroupBox(title, settings)
        group.setObjectName(f"{viewer_key}_{prefix}_group")
        group_layout = QtWidgets.QFormLayout(group)
        color = color_button(
            color_field,
            f"{viewer_key}_{prefix}_color",
            f"Choose the {tooltip_name} color.",
        )
        width = width_spin(
            width_field,
            f"{viewer_key}_{prefix}_width",
            f"Set the {tooltip_name} thickness.",
        )
        line_style = QtWidgets.QComboBox(group)
        line_style.setObjectName(f"{viewer_key}_{prefix}_style")
        line_style.setToolTip(f"Choose the {tooltip_name} line style.")
        for label, value in _LINE_STYLES:
            line_style.addItem(label, value)
        line_style.setCurrentIndex(
            max(line_style.findData(getattr(initial, style_field)), 0)
        )
        line_style.currentIndexChanged.connect(
            lambda _index, field=style_field, control=line_style: update(
                **{field: str(control.currentData())}
            )
        )
        controls[style_field] = line_style
        group_layout.addRow("Color", color)
        group_layout.addRow("Width", width)
        group_layout.addRow("Style", line_style)
        return group

    layout.insertWidget(
        max(layout.count() - 1, 0),
        line_group(
            "Fermi level",
            "fermi_line",
            "fermi_line_color",
            "fermi_line_width",
            "fermi_line_style",
            "Fermi-level reference line",
        ),
    )
    if viewer_key == "band_structure":
        layout.insertWidget(
            max(layout.count() - 1, 0),
            line_group(
                "High-symmetry guides",
                "symmetry_line",
                "symmetry_line_color",
                "symmetry_line_width",
                "symmetry_line_style",
                "vertical high-symmetry guide",
            ),
        )
    return controls


def populate_electronic_calculation_settings(
    settings: Any,
    *,
    viewer_key: str,
    values: Mapping[str, Any],
    on_apply: Callable[[dict[str, str]], None],
) -> None:
    """Populate the standard side panel with plot-owned calculation settings."""

    from PySide6 import QtWidgets

    layout = settings.layout()
    placeholder = settings.findChild(
        QtWidgets.QLabel,
        f"{viewer_key}_settings_placeholder",
    )
    if placeholder is not None:
        placeholder.setText(
            "These settings belong to this calculation. Apply changes to "
            "store them in the model and recalculate the viewer."
        )

    editors: dict[str, Any] = {}

    def add_line(
        form: Any,
        key: str,
        label: str,
        tooltip: str,
    ) -> None:
        editor = QtWidgets.QLineEdit(str(values.get(key, "")))
        editor.setObjectName(f"{viewer_key}_setting_{key}")
        editor.setToolTip(tooltip)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip)
        form.addRow(label_widget, editor)
        editors[key] = editor

    def add_choice(
        form: Any,
        key: str,
        label: str,
        choices: tuple[tuple[str, str], ...],
        tooltip: str,
    ) -> None:
        editor = QtWidgets.QComboBox()
        editor.setObjectName(f"{viewer_key}_setting_{key}")
        editor.setToolTip(tooltip)
        for text, value in choices:
            editor.addItem(text, value)
        editor.setCurrentIndex(
            max(editor.findData(str(values.get(key, choices[0][1]))), 0)
        )
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip)
        form.addRow(label_widget, editor)
        editors[key] = editor

    calculation = QtWidgets.QGroupBox("Calculation", settings)
    calculation.setObjectName(f"{viewer_key}_calculation_group")
    form = QtWidgets.QFormLayout(calculation)

    if viewer_key == "band_structure":
        add_choice(
            form,
            "band_path_convention",
            "Path convention",
            (
                ("Manual", "manual"),
                ("Hinuma / Seek-path", "hinuma"),
                ("Setyawan–Curtarolo / ASE", "setyawan_curtarolo"),
            ),
            "Select the convention used to generate the labelled path. "
            "Applying a standard convention regenerates its coordinates in "
            "nfit's primitive reciprocal basis.",
        )
        add_line(
            form,
            "band_points_per_inv_angstrom",
            "Intervals / Å⁻¹",
            "Sampling density along the physical reciprocal-space path. "
            "This remains a manual convergence setting.",
        )
        path_editor = QtWidgets.QPlainTextEdit(
            str(values.get("band_path", ""))
        )
        path_editor.setObjectName("band_structure_setting_band_path")
        path_editor.setToolTip(
            "Ordered JSON list of labelled reduced-coordinate nodes. It is "
            "used directly when the path convention is Manual."
        )
        path_editor.setMaximumHeight(105)
        path_label = QtWidgets.QLabel("Manual path")
        path_label.setToolTip(path_editor.toolTip())
        form.addRow(path_label, path_editor)
        editors["band_path"] = path_editor
    elif viewer_key == "density_of_states":
        add_choice(
            form,
            "dos_method",
            "Integration",
            (
                ("Gaussian broadening", "gaussian"),
                ("Linear tetrahedron", "tetrahedron"),
            ),
            "Gaussian integration applies the displayed broadening. Linear "
            "tetrahedron integration requires a complete three-dimensional mesh.",
        )
        add_line(
            form,
            "dos_mesh",
            "k mesh",
            "Uniform Brillouin-zone mesh. Mesh density remains a manual "
            "convergence choice.",
        )
        add_choice(
            form,
            "dos_symmetry",
            "Symmetry",
            (
                ("Automatic certified", "auto"),
                ("Full mesh", "full"),
                ("Require reduced", "reduced"),
            ),
            "Automatic mode uses symmetry only when equivalence is certified "
            "and otherwise falls back to the full mesh.",
        )
        unit = str(values.get("electronic_energy_unit", "eV"))
        add_line(
            form,
            "dos_energy_min_meV",
            f"Minimum ({unit})",
            "Lower absolute electronic energy included in the DOS grid.",
        )
        add_line(
            form,
            "dos_energy_max_meV",
            f"Maximum ({unit})",
            "Upper absolute electronic energy included in the DOS grid.",
        )
        add_line(
            form,
            "dos_energy_points",
            "Energy points",
            "Number of uniformly spaced samples on the DOS energy axis.",
        )
        add_line(
            form,
            "dos_broadening_meV",
            f"Gaussian σ ({unit})",
            "Gaussian standard deviation. It is ignored by tetrahedron integration.",
        )
    elif viewer_key == "fermi_surface":
        add_line(
            form,
            "fermi_mesh",
            "k mesh",
            "Uniform extraction grid. Mesh density remains a manual "
            "convergence choice.",
        )
        unit = str(values.get("electronic_energy_unit", "eV"))
        add_line(
            form,
            "fermi_energy_meV",
            f"Target energy ({unit})",
            "Absolute constant-energy target. Set it equal to the model "
            "chemical potential for a Fermi surface.",
        )
    else:
        return

    layout.insertWidget(max(layout.count() - 1, 0), calculation)
    apply_button = QtWidgets.QPushButton("Apply and recalculate", settings)
    apply_button.setObjectName(f"{viewer_key}_apply_settings")
    apply_button.setToolTip(
        "Validate these settings, save them with the tight-binding component, "
        "and replace this viewer with a recalculated result."
    )

    def apply() -> None:
        payload: dict[str, str] = {}
        for key, editor in editors.items():
            if isinstance(editor, QtWidgets.QComboBox):
                payload[key] = str(editor.currentData())
            elif isinstance(editor, QtWidgets.QPlainTextEdit):
                payload[key] = editor.toPlainText()
            else:
                payload[key] = editor.text()
        on_apply(payload)

    apply_button.clicked.connect(apply)
    layout.insertWidget(max(layout.count() - 1, 0), apply_button)


def show_electronic_figure(
    figure: Any,
    *,
    viewer_key: str,
    parent: Any | None = None,
    settings_config: Mapping[str, Any] | None = None,
    on_apply_settings: Callable[[dict[str, str]], None] | None = None,
    style: ElectronicPlotStyle | None = None,
) -> Any:
    """Show an electronic-structure figure beside its settings panel."""

    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )
    from PySide6 import QtGui, QtWidgets

    try:
        title = _VIEWER_TITLES[viewer_key]
    except KeyError as exc:
        choices = ", ".join(sorted(_VIEWER_TITLES))
        raise ValueError(
            f"unknown electronic viewer {viewer_key!r}; expected one of {choices}"
        ) from exc

    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow(parent)
    window.setObjectName(f"{viewer_key}_viewer")
    window.setWindowTitle(title)
    central, viewport_layout, settings = create_viewer_shell(
        QtWidgets,
        viewer_key=viewer_key,
    )
    settings_content = _scrollable_settings_content(settings, viewer_key)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setObjectName(f"{viewer_key}_canvas")
    canvas.setToolTip(f"Interactive {title.lower()} visualization.")
    toolbar = NavigationToolbar2QT(canvas, central)
    toolbar.setObjectName(f"{viewer_key}_toolbar")
    viewport_layout.addWidget(toolbar)
    viewport_layout.addWidget(canvas, 1)
    window.setCentralWidget(central)
    window.resize(1100, 760)
    window._nfit_application = application
    window._nfit_owns_application = owns_application
    window._nfit_figure = figure
    window._nfit_canvas = canvas
    window._nfit_settings_panel = settings
    if viewer_key in {"band_structure", "density_of_states"}:
        current_style = ElectronicPlotStyle() if style is None else style

        def update_style(updated: ElectronicPlotStyle) -> None:
            window._nfit_plot_style = updated
            apply_electronic_plot_style(figure, updated)
            canvas.draw_idle()

        window._nfit_plot_style = current_style
        window._nfit_plot_style_controls = populate_electronic_plot_settings(
            settings_content,
            viewer_key=viewer_key,
            initial=current_style,
            on_change=update_style,
        )
        apply_electronic_plot_style(figure, current_style)
    if settings_config is not None and on_apply_settings is not None:
        populate_electronic_calculation_settings(
            settings_content,
            viewer_key=viewer_key,
            values=settings_config,
            on_apply=on_apply_settings,
        )
    close_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Close,
        window,
    )
    close_shortcut.activated.connect(window.close)
    window._nfit_close_shortcut = close_shortcut
    window.show()
    return window
