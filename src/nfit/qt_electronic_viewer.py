"""Qt windows for Matplotlib electronic-structure figures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .qt_viewer_shell import create_viewer_shell

_VIEWER_TITLES = {
    "band_structure": "Band structure",
    "density_of_states": "Density of states",
    "fermi_surface": "Fermi surface",
}


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
    if settings_config is not None and on_apply_settings is not None:
        populate_electronic_calculation_settings(
            settings,
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
