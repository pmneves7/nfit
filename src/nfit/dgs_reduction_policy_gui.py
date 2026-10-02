"""Shared Qt editors for explicit Mantid-compatible and native DGS choices."""

from __future__ import annotations

from .dgs_reduction_policy import (
    DEFAULT_EVENT_PRECISION_POLICY,
    DEFAULT_MONITOR_VARIANCE_POLICY,
    DEFAULT_SYMMETRY_VARIANCE_POLICY,
    EVENT_PRECISION_POLICIES,
    MONITOR_VARIANCE_POLICIES,
    SYMMETRY_VARIANCE_POLICIES,
    validated_event_precision_policy,
    validated_monitor_variance_policy,
    validated_symmetry_variance_policy,
)


def add_dgs_policy_controls(layout, config, *, start_row, prefix, raw, on_changed, script_factory):
    """Add policy editors and a script export using GUI-independent services."""
    from PySide6 import QtWidgets

    fields = [
        (
            "Event precision", "event_precision_policy", EVENT_PRECISION_POLICIES,
            DEFAULT_EVENT_PRECISION_POLICY, validated_event_precision_policy,
            "Mantid compatibility follows its event-coordinate and histogram rounding "
            "and ray-derived He-3 tube geometry. "
            "High precision keeps double precision and nominal tube geometry; a saved MDE file cannot "
            "recover precision already lost during reduction. Changing raw event "
            "precision can require regeneration of reduced-event caches.",
        ),
        (
            "Symmetry uncertainty", "symmetry_variance_policy", SYMMETRY_VARIANCE_POLICIES,
            DEFAULT_SYMMETRY_VARIANCE_POLICY, validated_symmetry_variance_policy,
            "Independent copies matches Mantid's diagonal event-variance convention. "
            "Within-bin covariance includes shared-source terms when symmetry copies "
            "land in the same final bin. It does not propagate covariance between "
            "different bins through later slices or cuts; bin original events directly "
            "onto the final grid for that calculation.",
        ),
    ]
    if raw:
        fields.insert(0, (
            "Monitor peak fitting", "monitor_variance_policy", MONITOR_VARIANCE_POLICIES,
            DEFAULT_MONITOR_VARIANCE_POLICY, validated_monitor_variance_policy,
            "Mantid compatibility reproduces GetEi peak-tail arithmetic and stopping "
            "rules without calling Mantid. Stable variance uses a nonnegative "
            "equivalent derivative variance; peak selection and resolved Ei/T0 can "
            "change slightly. Changing this setting regenerates reduced events.",
        ))
    row = start_row
    for label, key, choices, default, validate, tooltip in fields:
        caption = QtWidgets.QLabel(label)
        selector = QtWidgets.QComboBox()
        selector.setObjectName(f"{prefix}_{key}")
        caption.setToolTip(tooltip)
        selector.setToolTip(tooltip)
        for value, title in choices:
            selector.addItem(title, value)
        selector.setCurrentIndex(selector.findData(validate(config.get(key, default))))
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
