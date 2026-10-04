"""Shared Qt editors for explicit Mantid-compatible and native DGS choices."""

from __future__ import annotations

from .dgs_reduction_policy import (
    validated_event_precision_policy,
    validated_monitor_variance_policy,
    validated_symmetry_variance_policy,
)


def add_dgs_policy_controls(layout, config, *, start_row, prefix, raw, on_changed, script_factory):
    """Add policy editors and a script export using GUI-independent services."""
    from PySide6 import QtWidgets

    from .reduction_recipes import reduction_settings_schema

    settings = {field.key: field for field in reduction_settings_schema(
        "raw-direct-geometry-nexus" if raw else "mantid-mdevent"
    )}
    fields = [
        ("Event precision", "event_precision_policy", validated_event_precision_policy),
        ("Symmetry uncertainty", "symmetry_variance_policy", validated_symmetry_variance_policy),
    ]
    if raw:
        fields.insert(0, ("Monitor peak fitting", "monitor_variance_policy", validated_monitor_variance_policy))
    row = start_row
    for label, key, validate in fields:
        field = settings[key]
        tooltip = field.tooltip
        caption = QtWidgets.QLabel(label)
        selector = QtWidgets.QComboBox()
        selector.setObjectName(f"{prefix}_{key}")
        caption.setToolTip(tooltip)
        selector.setToolTip(tooltip)
        for value, title in field.choices:
            selector.addItem(title, value)
        selector.setCurrentIndex(selector.findData(validate(config.get(key, field.default))))
        selector.currentIndexChanged.connect(
            lambda _index, key=key, selector=selector: on_changed(key, selector.currentData())
        )
        layout.addWidget(caption, row, 0)
        layout.addWidget(selector, row, 1, 1, 3)
        row += 1
    export = QtWidgets.QPushButton("Copy policy script")
    export.setObjectName(f"{prefix}_copy_policy_script")
    export.setToolTip(
        "Copy editable Python using nfit's public API to reproduce these scientific "
        "policies on an already imported group. Source import and binning settings "
        "are not included in this snippet."
    )
    export.clicked.connect(lambda _checked=False: QtWidgets.QApplication.clipboard().setText(script_factory()))
    layout.addWidget(export, row, 1, 1, 3)
