"""Schema-driven, metadata-only editors for saved acquisition recipes."""

from __future__ import annotations

import ast
import json
from typing import Any


def _resolved_numeric(resolved, key):
    value = resolved.get(key)
    return value.get("value") if isinstance(value, dict) else value


def _field_editor(field, value, *, object_name, on_changed, browse=None, resolved_value=None):
    """Build one editor; automatic values use an explicit checkbox, never a sentinel."""
    from PySide6 import QtCore, QtWidgets

    if field.kind == "bool":
        editor = QtWidgets.QCheckBox()
        editor.setChecked(bool(value))
        editor.toggled.connect(lambda checked: on_changed(bool(checked)))
    elif field.kind == "choice":
        editor = QtWidgets.QComboBox()
        for choice, title in field.choices:
            editor.addItem(title, choice)
            editor.setItemData(editor.count() - 1, field.tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
        editor.setCurrentIndex(editor.findData(value))
        editor.currentIndexChanged.connect(lambda _index: on_changed(editor.currentData()))
    elif field.kind == "float":
        editor = QtWidgets.QDoubleSpinBox()
        editor.setDecimals(8)
        editor.setRange(
            -1e12 if field.minimum is None else field.minimum,
            1e12 if field.maximum is None else field.maximum,
        )
        fallback = 1.0 if field.key == "incident_energy_override" else 0.0
        editor.setValue(float(value if value is not None else resolved_value if resolved_value is not None else field.default or fallback))
        editor.valueChanged.connect(
            lambda number: on_changed(float(number)) if editor.isEnabled() else None
        )
    else:
        editor = QtWidgets.QLineEdit(
            json.dumps(value) if field.kind == "matrix" else str(value or "")
        )

        def changed():
            text = editor.text().strip()
            parsed: Any = text or None
            if field.kind == "matrix":
                try:
                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    parsed = text
            on_changed(parsed)

        editor.editingFinished.connect(changed)
    editor.setObjectName(object_name)
    editor.setToolTip(field.tooltip)
    if not field.automatic and not (field.kind == "path" and browse is not None):
        return editor

    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(editor)
    if field.automatic and field.kind != "path":
        automatic = QtWidgets.QCheckBox("Automatic")
        automatic.setObjectName(f"{object_name}_automatic")
        automatic.setToolTip(
            f"Resolve {field.label.lower()} from this run. Turn off to use the numeric override."
            + (" Negative time-zero values are ordinary overrides." if field.key == "t0_override" else "")
        )
        automatic.setChecked(value is None)
        editor.setEnabled(value is not None)

        def set_automatic(checked):
            editor.setEnabled(not checked)
            on_changed(None if checked else float(editor.value()))

        automatic.toggled.connect(set_automatic)
        layout.addWidget(automatic)
    if field.kind == "path" and browse is not None:
        button = QtWidgets.QPushButton("Browse…")
        button.setObjectName(f"{object_name}_browse")
        button.setToolTip(f"Choose {field.label.lower()} for this reduction recipe.")
        button.clicked.connect(lambda _checked=False: browse(field, editor, on_changed))
        layout.addWidget(button)
    return container


def build_reduction_recipe_panel(
    group, *, on_shared_changed, on_run_changed, parent=None, scopes=None, export=True
):
    """Present shared settings, explicit run overrides, and resolved provenance.

    Callbacks use the public reduction API through the explorer coordinator.
    Constructing this panel never reads event arrays or source files.
    """
    from PySide6 import QtCore, QtWidgets

    from .dgs_reduction_settings import dgs_reduction_policy_script
    from .file_dialogs import get_open_file_name
    from .reduction_recipes import (
        effective_reduction_config,
        get_reduction_overrides,
        reduction_recipe_script,
        reduction_settings_schema,
        resolved_reduction_values,
    )

    config = effective_reduction_config(group)
    prefix = "mdevent" if "mdevent" in group.metadata else "raw_dgs"
    corelli = config.get("format") == "corelli-correlation-nexus"
    title = "CORELLI finite-energy reconstruction" if corelli else (
        "MDEvent shared setup" if prefix == "mdevent" else "Raw TOF shared setup"
    )
    histogram_only = scopes is not None and set(scopes) == {"histogram"}
    if histogram_only:
        title = "Histogram conventions"
    box = QtWidgets.QGroupBox(title, parent)
    box.setObjectName(f"{prefix}_histogram_recipe" if histogram_only else f"{prefix}_reduction_recipe")
    box.setToolTip(
        "Saved acquisition and reduction settings. Shared defaults apply to runs without "
        "an explicit override. Edits invalidate only affected reductions and dependent histograms."
    )
    layout = QtWidgets.QVBoxLayout(box)
    fields = tuple(field for field in reduction_settings_schema(group)
                   if scopes is None or field.scope in scopes)

    def browse(field, edit, apply):
        path, _filter = get_open_file_name(
            parent, f"Choose {field.label.lower()}", edit.text(),
            "Calibration files (*.nxs *.nxs.h5 *.nx5 *.h5 *.hdf5 *.xml);;All files (*)",
        )
        if path:
            edit.setText(path)
            apply(path)

    def shared_changed(key, value):
        if on_shared_changed(key, value) is not False:
            refresh_resolved()

    mixed = QtWidgets.QLabel()
    mixed.setObjectName(f"{prefix}_mixed_histogram_values" if histogram_only else f"{prefix}_mixed_reduction_values")
    mixed.setWordWrap(True)
    mixed.setToolTip("Mixed means enabled runs have different effective settings. Shared editors always show the inherited default, not an average of overrides.")
    layout.addWidget(mixed)

    shared = QtWidgets.QWidget()
    shared_layout = QtWidgets.QFormLayout(shared)
    scope_titles = {
        "reduction": "Acquisition and reduction",
        "normalization": "Normalization",
        "coordinates": "Coordinate transform",
        "histogram": "Histogram uncertainty",
        "storage": "Reduced event storage",
    }
    ordered_fields = sorted(fields, key=lambda field: list(scope_titles).index(field.scope))
    previous_scope = None
    first = next(iter(group.iter_datasets()), None)
    first_resolved = {} if first is None else resolved_reduction_values(group, first)
    for field in ordered_fields:
        if field.scope != previous_scope:
            caption = QtWidgets.QLabel(scope_titles[field.scope])
            caption.setStyleSheet("font-weight: bold")
            shared_layout.addRow(caption)
            previous_scope = field.scope
        label = QtWidgets.QLabel(f"{field.label} ({field.units})" if field.units else field.label)
        label.setToolTip(field.tooltip)
        editor = _field_editor(
            field, config.get(field.key, field.default), object_name=f"{prefix}_{field.key}",
            on_changed=lambda value, key=field.key: shared_changed(key, value),
            browse=browse,
            resolved_value=_resolved_numeric(first_resolved, field.key),
        )
        shared_layout.addRow(label, editor)
    layout.addWidget(shared)

    def refresh_mixed():
        participating = [item for item in group.iter_datasets() if item.enabled]
        configurations = [effective_reduction_config(group, item) for item in participating]
        mixed_fields = [field.label for field in fields if len({
            json.dumps(settings.get(field.key, field.default), sort_keys=True, default=str)
            for settings in configurations
        }) > 1]
        overrides_count = sum(bool(get_reduction_overrides(group, item)) for item in participating)
        mixed.setText(
            f"Shared defaults; {overrides_count} of {len(participating)} enabled runs have explicit overrides. "
            + ("Mixed effective values: " + ", ".join(mixed_fields) if mixed_fields else "Effective settings are uniform across enabled runs.")
        )

    refresh_mixed()
    if not any(field.per_run for field in fields):
        refresh_resolved = refresh_mixed
        return box

    run_box = QtWidgets.QGroupBox("Individual run overrides and resolved values")
    run_box.setToolTip(
        "Select a run to inspect its acquisition provenance and edit overrides. "
        "Inherit uses the shared setting; Automatic resolves a value from the selected run."
    )
    run_layout = QtWidgets.QVBoxLayout(run_box)
    selector = QtWidgets.QComboBox()
    selector.setObjectName(f"{prefix}_reduction_run_selector")
    selector.setToolTip("Choose a run without loading its reduced events or histogram caches.")
    datasets = list(group.iter_datasets())
    for dataset in datasets:
        selector.addItem(dataset.name, dataset.id)
    run_layout.addWidget(selector)
    resolved = QtWidgets.QLabel()
    resolved.setObjectName(f"{prefix}_resolved_reduction_values")
    resolved.setWordWrap(True)
    resolved.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    resolved.setToolTip(
        "Resolved automatic Ei (meV), T0 (microseconds), acquisition geometry, and calibration "
        "provenance retained from source metadata or the most recent reduction. "
        "Resolved sample rotations are time-weighted over accepted acquisition intervals."
    )
    run_layout.addWidget(resolved)
    overrides = QtWidgets.QWidget()
    overrides.setObjectName(f"{prefix}_run_override_fields")
    override_layout = QtWidgets.QFormLayout(overrides)
    run_layout.addWidget(overrides)

    def resolved_summary(dataset):
        summary = {key: dataset.metadata[key] for key in (
            "source_file", "run_number", "instrument", "instrument_name", "instrument_id",
            "geometry_identity", "geometry_signature", "l1", "omega", "phi", "chi",
            "corelli_goniometer_angles", "calibration_source", "calibration_warning",
        ) if key in dataset.metadata}
        summary["resolved_settings"] = resolved_reduction_values(group, dataset)
        record = dataset.metadata.get("resolved_reduction", {})
        automatic = record.get("automatic_values", {})
        angles = {key: automatic[key] for key in ("omega_degrees", "phi_degrees", "chi_degrees")
                  if key in automatic}
        if angles:
            summary["resolved_sample_rotation"] = {
                **angles,
                "averaging": record.get("provenance", {}).get("goniometer_averaging"),
                "stale": bool(record.get("stale")),
            }
        return json.dumps(summary, indent=2, default=str)

    def refresh_resolved():
        refresh_mixed()
        dataset = next((item for item in datasets if item.id == selector.currentData()), None)
        if dataset is not None:
            resolved.setText(resolved_summary(dataset))

    def run_changed(dataset, key, value, inherit):
        if on_run_changed(dataset, key, value, inherit) is False:
            return False
        refresh_resolved()
        return True

    def rebuild_run(_index=0):
        while override_layout.rowCount():
            override_layout.removeRow(0)
        dataset = next((item for item in datasets if item.id == selector.currentData()), None)
        if dataset is None:
            resolved.setText("No runs in this group.")
            return
        resolved.setText(resolved_summary(dataset))
        settings = effective_reduction_config(group, dataset)
        explicit = get_reduction_overrides(group, dataset)
        for field in fields:
            if not field.per_run:
                continue
            key = field.key
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            editor = _field_editor(
                field, settings.get(key, field.default), object_name=f"{prefix}_run_{key}",
                on_changed=lambda value, key=key, dataset=dataset: run_changed(
                    dataset, key, value, False
                ), browse=browse,
                resolved_value=_resolved_numeric(resolved_reduction_values(group, dataset), key),
            )
            editor.setEnabled(key in explicit)
            row_layout.addWidget(editor)
            inherit = QtWidgets.QCheckBox("Inherit")
            inherit.setObjectName(f"{prefix}_run_{key}_inherit")
            inherit.setToolTip(f"Use the shared {field.label.lower()} instead of a per-run override.")
            inherit.setChecked(key not in explicit)

            def change_inherit(checked, key=key, dataset=dataset):
                settings = effective_reduction_config(group, dataset)
                if run_changed(dataset, key, settings.get(key), bool(checked)):
                    rebuild_run()

            inherit.toggled.connect(change_inherit)
            row_layout.addWidget(inherit)
            label = QtWidgets.QLabel(f"{field.label} ({field.units})" if field.units else field.label)
            label.setToolTip(field.tooltip)
            override_layout.addRow(label, row)

    selector.currentIndexChanged.connect(rebuild_run)
    rebuild_run()
    layout.addWidget(run_box)
    refresh_resolved()
    if corelli:
        caveat = QtWidgets.QLabel(
            "Energy channels share measured events and have correlated statistical errors. "
            "Stored errors contain the diagonal variance."
        )
        caveat.setWordWrap(True)
        caveat.setToolTip("The same phase-tagged events reconstruct each CORELLI energy channel.")
        layout.addWidget(caveat)
    if not export:
        return box
    buttons = QtWidgets.QHBoxLayout()
    export = QtWidgets.QPushButton("Copy reduction recipe script")
    export.setObjectName(f"{prefix}_copy_reduction_recipe_script")
    export.setToolTip(
        "Copy editable Python using nfit's public API for the source files, shared settings, "
        "per-run overrides, and coordinate transform. Binning and display settings are separate."
    )
    export.clicked.connect(lambda _checked=False: QtWidgets.QApplication.clipboard().setText(
        reduction_recipe_script(group)
    ))
    buttons.addWidget(export)
    if not corelli:
        policies = QtWidgets.QPushButton("Copy policy script")
        policies.setObjectName(f"{prefix}_copy_policy_script")
        policies.setToolTip("Copy the numerical DGS policies for an already imported group.")
        policies.clicked.connect(lambda _checked=False: QtWidgets.QApplication.clipboard().setText(
            dgs_reduction_policy_script(group)
        ))
        buttons.addWidget(policies)
    buttons.addStretch(1)
    layout.addLayout(buttons)
    return box
