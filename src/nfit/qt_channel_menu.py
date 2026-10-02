"""Readable channel menus with stable scientific channel identifiers."""

from PySide6 import QtCore, QtWidgets

from .histogram_statistics import EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR
from .measurement_diagnostics import event_contribution_description

_GROUPS = (
    (
        "Signal channels",
        (
            "signal",
            "unsubtracted",
            "background",
            "errors",
            "imported_signal",
            "scattering_cross_section",
        ),
    ),
    ("Event statistics", (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR)),
    ("Confidence intervals", ("confidence_lower", "confidence_upper")),
    ("Fit diagnostics", ("fit", "residual")),
    (
        "Coverage and normalization",
        ("num_events", "coverage_fraction", "normalization_denominator"),
    ),
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
    "confidence_lower": "Poisson rate lower bound (68.27%)",
    "confidence_upper": "Poisson rate upper bound (68.27%)",
}


_EVENT_LABELS = {
    "errors": "Observed event standard error",
    "num_events": "Event contributions",
    "normalization_denominator": "Exposure",
    EVENT_SIGNAL_NUMERATOR: "Event signal numerator",
    EVENT_VARIANCE_NUMERATOR: "Event numerator variance",
}
_EVENT_TOOLTIPS = {
    "errors": (
        "Square root of the accumulated event variance divided by known exposure. "
        "A measured zero has zero observed event variance; this is not a confidence "
        "interval for its unknown rate."
    ),
    "num_events": (
        "Number of event contributions in the bin. Symmetry copies can share an "
        "original event and do not create independent measurements."
    ),
    "normalization_denominator": (
        "Known normalization exposure. Covered zero-count measurements contribute "
        "exposure. Uncertainty in the exposure or calibration is not included."
    ),
    EVENT_SIGNAL_NUMERATOR: "Sum of corrected event weights before division by exposure.",
    EVENT_VARIANCE_NUMERATOR: (
        "Accumulated event numerator variance before division by exposure squared. "
        "The saved symmetry policy and represented source dependencies determine "
        "which shared terms are included; inspect Statistics and provenance."
    ),
}


class ChannelComboBox(QtWidgets.QComboBox):
    """Keep identifier-based selection compatible with earlier viewer widgets."""

    def setCurrentText(self, text):
        index = self.findData(text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            super().setCurrentText(text)


def populate_channel_combo(
    combo,
    channels,
    selected,
    *,
    point_list=False,
    labels=None,
    tooltips=None,
    event_statistics=False,
    num_events_semantics=None,
    counting_dependencies=False,
):
    """Group histogram diagnostics without reading any numerical payloads."""
    channel_labels = dict(_LABELS)
    channel_tooltips = {
        name: "Exact equal-tailed 68.27% Garwood bound, computed after pooling final-bin counts and known exposure. Requires an audited independent constant-weight Poisson model; display smoothing is not applied."
        for name in ("confidence_lower", "confidence_upper")
    }
    if event_statistics and not point_list:
        channel_labels.update(_EVENT_LABELS)
        channel_tooltips.update(_EVENT_TOOLTIPS)
        label, tooltip = event_contribution_description(num_events_semantics)
        channel_labels["num_events"] = label
        channel_tooltips["num_events"] = tooltip
        if counting_dependencies:
            channel_labels["errors"] = "Propagated count standard uncertainty"
            channel_tooltips["errors"] = "Standard uncertainty propagated from represented numerator and exposure primitives, including their shared covariance. This is not a low-count confidence interval."
            channel_tooltips["normalization_denominator"] = "Normalization exposure; represented exposure and shared calibration dependencies enter the propagated ratio uncertainty. Inspect Statistics and provenance for assumptions."
    channel_labels.update(labels or {})
    channel_tooltips.update(tooltips or {})
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
                label = name if point_list else channel_labels.get(name, name)
                combo.addItem(label, name)
                tooltip = name if point_list else channel_tooltips.get(name, name)
                combo.setItemData(combo.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(combo.findData(selected))
    finally:
        del blocker
