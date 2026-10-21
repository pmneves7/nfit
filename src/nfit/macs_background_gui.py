"""Visible MACS background sources and their shared and individual editors."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def background_source_path(item):
    """Read a source row's path without creating a dataset or opening its file."""
    from PySide6 import QtCore

    return item.data(0, QtCore.Qt.ItemDataRole.UserRole)


def add_macs_background_tree(parent_item, node, *, remember, icon, style, expanded):
    """Append editable source references, deliberately outside the dataset inventory."""
    from PySide6 import QtCore, QtWidgets

    from .macs_background_sources import list_macs_background_sources
    from .macs_reduction import macs_group_config

    if macs_group_config(node) is None:
        return
    folder = QtWidgets.QTreeWidgetItem(["Backgrounds"])
    folder.setToolTip(0, "Background files and subtraction settings for this MACS collection. Select a file to adjust its scale and reduction settings.")
    icon(folder, "background_folder")
    remember(folder, "macs_backgrounds", node=node)
    parent_item.addChild(folder)
    for source in list_macs_background_sources(node):
        path = source["source_file"]
        item = QtWidgets.QTreeWidgetItem([Path(path).name])
        item.setToolTip(0, f"{path}\nBackground source; select to edit its scale and reduction overrides. Applied only to energy-matched sample runs.")
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, path)
        icon(item, "dataset")
        style(item, source["enabled"])
        remember(item, "macs_background_file", node=node)
        folder.addChild(item)
    folder.setExpanded(expanded)


def build_macs_background_panel(node, path=None, *, on_changed, parent=None):
    """Build schema-based background controls using public recipe callbacks."""
    from PySide6 import QtCore, QtWidgets

    from .file_dialogs import get_open_file_name, get_open_file_names
    from .macs_background_sources import (
        effective_macs_background_config,
        list_macs_background_sources,
    )
    from .reduction_recipe_gui import _field_editor
    from .reduction_recipes import (
        effective_reduction_config,
        reduction_recipe_script,
        reduction_settings_schema,
    )

    box = QtWidgets.QGroupBox("MACS background subtraction" if path is None else "MACS background source", parent)
    box.setObjectName("macs_background_settings" if path is None else "macs_background_source_settings")
    box.setToolTip("Background sources are saved with this collection. Shared settings control subtraction; individual source settings override inherited sample reduction settings.")
    layout = QtWidgets.QVBoxLayout(box)
    summary = QtWidgets.QLabel(
        "Select background scans, choose how they are applied, and set their common scale. "
        "Select a file in the Backgrounds folder to adjust that source individually."
        if path is None else str(path)
    )
    summary.setWordWrap(True)
    summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(summary)
    form = QtWidgets.QFormLayout()
    layout.addLayout(form)

    def browse(field, edit, apply):
        if field.kind == "files":
            try:
                current = ast.literal_eval(edit.text())
            except (ValueError, SyntaxError):
                current = []
            paths, _ = get_open_file_names(parent, f"Choose {field.label.lower()}", current[0] if isinstance(current, list) and current else "", "MACS scans (*.nxs *.nxs.h5 *.nx5 *.h5 *.hdf5 *.ng0 *.bt9);;All files (*)")
            if paths:
                edit.setText(json.dumps(paths))
                apply(paths)
        else:
            selected, _ = get_open_file_name(parent, f"Choose {field.label.lower()}", edit.text(), "All files (*)")
            if selected:
                edit.setText(selected)
                apply(selected)

    if path is None:
        sources = list_macs_background_sources(node)
        enabled = QtWidgets.QCheckBox("Use background sources")
        enabled.setObjectName("macs_background_sources_enabled")
        enabled.setToolTip("Enable or bypass every background file in this collection while retaining individual scale factors and reduction overrides.")
        states = [source["enabled"] for source in sources]
        enabled.setTristate(bool(any(states) and not all(states)))
        enabled.setCheckState(
            QtCore.Qt.CheckState.Checked if states and all(states)
            else QtCore.Qt.CheckState.PartiallyChecked if any(states)
            else QtCore.Qt.CheckState.Unchecked
        )
        enabled.setEnabled(bool(sources))
        enabled.checkStateChanged.connect(lambda state: on_changed("enabled", state != QtCore.Qt.CheckState.Unchecked))
        form.addRow(enabled)
        config = effective_reduction_config(node)
        for field in reduction_settings_schema(node):
            if not field.key.startswith("background_") or field.key == "background_source_overrides":
                continue
            editor = _field_editor(field, config.get(field.key, field.default), object_name=f"macs_{field.key}", on_changed=lambda value, key=field.key: on_changed(key, value), browse=browse)
            label = QtWidgets.QLabel(f"{field.label} ({field.units})" if field.units else field.label)
            label.setToolTip(field.tooltip)
            form.addRow(label, editor)
    else:
        source = next(item for item in list_macs_background_sources(node) if item["source_file"] == path)
        enabled = QtWidgets.QCheckBox("Use this background source")
        enabled.setObjectName("macs_background_source_enabled")
        enabled.setToolTip("Temporarily bypass this file while retaining its scale and reduction overrides.")
        enabled.setChecked(source["enabled"])
        enabled.toggled.connect(lambda checked: on_changed("enabled", bool(checked)))
        form.addRow(enabled)
        scale = QtWidgets.QDoubleSpinBox()
        scale.setObjectName("macs_background_source_scale_factor")
        scale.setToolTip("Dimensionless multiplier for this file, multiplied by the collection's common background scale. The statistical uncertainty is scaled by its absolute value.")
        scale.setRange(-1e12, 1e12)
        scale.setDecimals(8)
        scale.setKeyboardTracking(False)
        scale.setValue(source["scale_factor"])
        scale.valueChanged.connect(lambda value: on_changed("scale_factor", float(value)))
        label = QtWidgets.QLabel("Individual scale factor")
        label.setToolTip(scale.toolTip())
        form.addRow(label, scale)
        explanation = QtWidgets.QLabel("Inherited settings follow each matched sample run. Explicit overrides below apply only to this background file. Monitor targets and reference wavevectors are converted to the matched sample's reporting units; use the individual scale factor to rescale the correction.")
        explanation.setWordWrap(True)
        form.addRow(explanation)
        config = effective_macs_background_config(node, path)
        explicit = source["reduction_overrides"]

        def reduction_editor(field, settings):
            return _field_editor(
                field, settings.get(field.key, field.default),
                object_name=f"macs_background_run_{field.key}",
                on_changed=lambda value, key=field.key: on_changed(key, value),
                browse=browse, ub_frame="ipns",
            )

        for field in reduction_settings_schema(node):
            if not field.per_run:
                continue
            key = field.key
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            editor = reduction_editor(field, config)
            editor.setEnabled(key in explicit)
            row_layout.addWidget(editor)
            editor_holder = {"widget": editor}
            inherit = QtWidgets.QCheckBox("Inherit")
            inherit.setObjectName(f"macs_background_run_{key}_inherit")
            inherit.setToolTip(f"Use each matched sample run's {field.label.lower()} instead of an override for this background file.")
            inherit.setChecked(key not in explicit)

            def change_inherit(checked, key=key, field=field, row_layout=row_layout, editor_holder=editor_holder):
                settings = effective_macs_background_config(node, path)
                if on_changed(key, settings.get(key), inherit=bool(checked)) is not False:
                    # Inheritance can change both the value and Automatic mode.
                    # Replace only the editor; the checkbox emitting this signal
                    # remains alive, and the detached editor is destroyed later.
                    current = reduction_editor(field, effective_macs_background_config(node, path))
                    current.setEnabled(not checked)
                    previous = editor_holder["widget"]
                    row_layout.replaceWidget(previous, current)
                    for control in (previous, *previous.findChildren(QtCore.QObject)):
                        control.blockSignals(True)
                    previous.setParent(None)
                    previous.deleteLater()
                    editor_holder["widget"] = current

            inherit.toggled.connect(change_inherit)
            row_layout.addWidget(inherit)
            label = QtWidgets.QLabel(f"{field.label} ({field.units})" if field.units else field.label)
            label.setToolTip(field.tooltip)
            form.addRow(label, row)
    export = QtWidgets.QPushButton("Copy reduction recipe script")
    export.setObjectName("macs_background_copy_recipe")
    export.setToolTip("Copy editable Python for this collection's shared reduction settings, background files, individual source settings, and sample run overrides.")
    export.clicked.connect(lambda _checked=False: QtWidgets.QApplication.clipboard().setText(reduction_recipe_script(node)))
    layout.addWidget(export)
    return box
