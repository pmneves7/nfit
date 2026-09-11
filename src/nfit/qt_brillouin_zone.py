"""Focused Qt input for optional Brillouin-zone overlays."""

from __future__ import annotations

from typing import Any

from PySide6 import QtWidgets


def prompt_brillouin_zone_context(
    parent: Any,
    context: dict[str, Any],
    *,
    require_lattice: bool,
) -> dict[str, Any] | None:
    """Ask only for crystal information needed by a BZ overlay."""

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Brillouin-zone crystal information")
    layout = QtWidgets.QFormLayout(dialog)
    explanation = QtWidgets.QLabel(
        "Zone boundaries use the reciprocal metric and lattice centering. "
        "This information affects the overlay only."
    )
    explanation.setWordWrap(True)
    layout.addRow(explanation)
    spacegroup = QtWidgets.QLineEdit(str(context.get("spacegroup", "")))
    spacegroup.setPlaceholderText("P, I, F, C, or a full space-group symbol")
    spacegroup.setToolTip(
        "Enter a Hermann–Mauguin space group or just its lattice-centering letter."
    )
    layout.addRow("Space group / centering", spacegroup)
    spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
    if require_lattice:
        lattice = context.get("lattice_parameters")
        lattice = lattice if isinstance(lattice, dict) else {}
        for name, default, suffix in (
            ("a", 1.0, " Å"), ("b", 1.0, " Å"), ("c", 1.0, " Å"),
            ("alpha", 90.0, "°"), ("beta", 90.0, "°"), ("gamma", 90.0, "°"),
        ):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.001 if name in {"a", "b", "c"} else 0.01, 100000.0 if name in {"a", "b", "c"} else 179.99)
            spin.setDecimals(6)
            spin.setValue(float(lattice.get(name, default)))
            spin.setSuffix(suffix)
            spin.setToolTip(
                "Direct-lattice length." if name in {"a", "b", "c"}
                else "Direct-lattice angle in degrees."
            )
            layout.addRow(name, spin)
            spins[name] = spin
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    symbol = spacegroup.text().strip()
    if not symbol:
        QtWidgets.QMessageBox.warning(
            parent, "Missing crystal information", "Enter a space group or centering letter."
        )
        return None
    result = dict(context)
    result["spacegroup"] = symbol
    if spins:
        result["lattice_parameters"] = {
            name: float(spin.value()) for name, spin in spins.items()
        }
    return result
