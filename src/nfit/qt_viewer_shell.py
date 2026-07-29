"""Shared Qt layout for scientific viewers with a right settings panel."""

from __future__ import annotations

from typing import Any


def create_viewer_shell(
    QtWidgets: Any,
    *,
    viewer_key: str,
) -> tuple[Any, Any, Any]:
    """Return a central widget, visualization layout, and settings panel."""

    central = QtWidgets.QWidget()
    central.setObjectName(f"{viewer_key}_viewer_central")
    root = QtWidgets.QHBoxLayout(central)

    viewport = QtWidgets.QWidget(central)
    viewport.setObjectName(f"{viewer_key}_viewer_viewport")
    viewport_layout = QtWidgets.QVBoxLayout(viewport)
    viewport_layout.setContentsMargins(0, 0, 0, 0)
    root.addWidget(viewport, 1)

    settings = QtWidgets.QGroupBox("Settings", central)
    settings.setObjectName(f"{viewer_key}_settings_panel")
    settings.setFixedWidth(280)
    settings.setToolTip(
        "Controls that change this viewer's presentation will be collected here."
    )
    settings_layout = QtWidgets.QVBoxLayout(settings)
    placeholder = QtWidgets.QLabel("Viewer settings will appear here.", settings)
    placeholder.setObjectName(f"{viewer_key}_settings_placeholder")
    placeholder.setWordWrap(True)
    settings_layout.addWidget(placeholder)
    settings_layout.addStretch(1)
    root.addWidget(settings)
    return central, viewport_layout, settings
