"""Shared Qt presentation for the direct-geometry normalization convention."""

from __future__ import annotations

from .dgs_normalization import (
    DEFAULT_TRAJECTORY_ENERGY_POLICY,
    TRAJECTORY_ENERGY_POLICIES,
    validated_trajectory_energy_policy,
)


def trajectory_energy_selector(config, *, object_name, on_changed):
    """Build an editor whose callback saves a public reduction setting."""
    from PySide6 import QtWidgets

    selector = QtWidgets.QComboBox()
    selector.setObjectName(object_name)
    selector.setToolTip(
        "Incident energy used for detector-trajectory normalization. First run Ei "
        "matches Mantid MDNorm's first-experiment convention; Each run Ei uses "
        "each run's resolved scalar incident energy. The shared Ei override takes "
        "precedence. Event coordinates and each run's accepted energy range are "
        "unchanged. Pulse-resolved incident energy is not available in this importer."
    )
    for value, label in TRAJECTORY_ENERGY_POLICIES:
        selector.addItem(label, value)
    policy = validated_trajectory_energy_policy(
        config.get("trajectory_energy_policy", DEFAULT_TRAJECTORY_ENERGY_POLICY)
    )
    selector.setCurrentIndex(selector.findData(policy))
    selector.currentIndexChanged.connect(lambda _index: on_changed(selector.currentData()))
    return selector
