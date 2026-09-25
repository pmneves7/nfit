"""Readable channel menus with stable scientific channel identifiers."""

from PySide6 import QtCore, QtWidgets

_GROUPS = (
    ("Signal channels", ("signal", "unsubtracted", "background", "errors",
                "imported_signal", "scattering_cross_section")),
    ("Fit diagnostics", ("fit", "residual")),
    ("Coverage and normalization", ("num_events", "coverage_fraction",
                                    "normalization_denominator")),
    ("Masks", ("combined_mask", "file_mask", "nfit_mask", "coverage_mask")),
)
_LABELS = {
    "signal": "Signal",
    "unsubtracted": "Unsubtracted",
    "background": "Background",
    "errors": "Standard uncertainty",
    "imported_signal": "Imported signal",
    "scattering_cross_section": "Scattering cross section",
    "fit": "Fit",
    "residual": "Residual (sigma)",
    "num_events": "Multiplicity / event count",
    "coverage_fraction": "Coverage fraction",
    "normalization_denominator": "Normalization denominator",
    "combined_mask": "Combined mask",
    "file_mask": "File mask",
    "nfit_mask": "User mask",
    "coverage_mask": "Coverage mask",
}


class ChannelComboBox(QtWidgets.QComboBox):
    """Keep identifier-based selection compatible with earlier viewer widgets."""

    def setCurrentText(self, text):
        index = self.findData(text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            super().setCurrentText(text)


def populate_channel_combo(combo, channels, selected, *, point_list=False):
    """Group histogram diagnostics without reading any numerical payloads."""
    blocker = QtCore.QSignalBlocker(combo)
    try:
        combo.clear()
        if point_list:
            groups = [(None, list(channels))]
        else:
            remaining = dict.fromkeys(channels)
            groups = []
            for title, names in _GROUPS:
                members = [name for name in names if name in remaining]
                for name in members:
                    remaining.pop(name)
                if members:
                    groups.append((title, members))
            if remaining:
                groups.insert(1, ("Additional channels", list(remaining)))
        for title, names in groups:
            if title is not None:
                combo.addItem(title)
                item = combo.model().item(combo.count() - 1)
                item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            for name in names:
                label = name if point_list else _LABELS.get(name, name)
                combo.addItem(label, name)
                combo.setItemData(combo.count() - 1, name, QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(combo.findData(selected))
    finally:
        del blocker
