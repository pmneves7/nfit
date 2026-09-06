"""Metadata-dimension editor, kept separate from spatial rebin controls."""

from __future__ import annotations

import numpy as np

from .metadata_dimensions import (
    MetadataDimension,
    metadata_channels,
    metadata_dimension_coordinates,
    metadata_dimension_indices,
)


def metadata_dimensions_panel(explorer, group):
    from PySide6 import QtCore, QtWidgets

    from .project_gui import (
        _composite_candidates,
        _composite_root,
        _dataset_composite_kind,
        _ensure_dataset_data_loaded,
        set_metadata_dimensions,
    )

    box = QtWidgets.QGroupBox("Metadata dimensions")
    box.setToolTip(
        "Add discrete sample-condition axes to this collection's composite, independently of momentum and energy binning."
    )
    layout = QtWidgets.QVBoxLayout(box)
    explanation = QtWidgets.QLabel(
        "Keep each metadata coordinate as an independent slice. Enable the composite below to plot the added dimensions."
    )
    explanation.setWordWrap(True)
    layout.addWidget(explanation)
    recipes = list(group.metadata.get("metadata_dimensions", []))
    items = QtWidgets.QListWidget()
    items.setObjectName("metadata_dimensions_list")
    items.setToolTip(
        "Select a dimension to edit or remove. All listed dimensions are applied together."
    )
    items.setMaximumHeight(100)
    for spec in recipes:
        items.addItem(f"{spec['name']} ({spec['units']}) — {spec['source']}")
    layout.addWidget(items)
    row = QtWidgets.QHBoxLayout()
    layout.addLayout(row)

    def save(specs):
        try:
            set_metadata_dimensions(group, specs)
        except (ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(box, "Metadata dimensions", str(exc))
            return
        explorer._after_group_composite_changed(group)

    def edit(existing=None):
        candidates = _composite_candidates(group)
        if not candidates:
            return
        try:
            _ensure_dataset_data_loaded(candidates[0])
            channels = metadata_channels(candidates[0])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(box, "Metadata channels", str(exc))
            return
        dialog = QtWidgets.QDialog(box)
        dialog.setWindowTitle("Metadata dimension")
        dialog.resize(780, 520)
        form = QtWidgets.QFormLayout(dialog)
        current = {} if existing is None else recipes[existing]
        source = QtWidgets.QComboBox()
        source.setEditable(True)
        source.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        source.setMinimumContentsLength(40)
        source.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        source.completer().setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        source.setObjectName("metadata_dimension_source")
        source.setToolTip(
            "Choose a numeric channel from the first enabled source, or type a path. Every enabled source must supply it. Both / and > separate path components."
        )
        source.addItems(list(channels))
        preferred = next(
            (
                path
                for path in (
                    "entry/data/temp/average_value",
                    "parameters/temperature",
                    "temperature",
                )
                if path in channels
            ),
            "",
        )
        source.setCurrentText(current.get("source", preferred))
        name = QtWidgets.QLineEdit(current.get("name", "Temperature"))
        name.setObjectName("metadata_dimension_name")
        name.setToolTip(
            "Axis label shown in plots and the data viewer; it must differ from existing axis names."
        )
        units = QtWidgets.QLineEdit(current.get("units", channels.get(source.currentText(), "")))
        units.setObjectName("metadata_dimension_units")
        units.setToolTip(
            "Physical units of this coordinate, for example K or degree. This labels values; it does not convert units."
        )
        source.currentTextChanged.connect(lambda text: units.setText(channels.get(text, "")))
        sampling = QtWidgets.QComboBox()
        sampling.setObjectName("metadata_dimension_sampling")
        sampling.setToolTip(
            "Mean or median assigns one condition to the whole dataset. Per point requires scalar or explicitly aligned point/scan values; asynchronous time logs are rejected."
        )
        for label, key in (
            ("One value per dataset: mean", "dataset_mean"),
            ("One value per dataset: median", "dataset_median"),
            ("One value per measured point", "per_point"),
        ):
            sampling.addItem(label, key)
        sampling.setCurrentIndex(sampling.findData(current.get("sampling", "dataset_mean")))
        centers = QtWidgets.QLineEdit(
            ", ".join(f"{value:g}" for value in (current.get("centers") or []))
        )
        centers.setObjectName("metadata_dimension_centers")
        centers.setPlaceholderText("5, 10, 20, 30, 40, 50 (blank: exact unique values)")
        centers.setToolTip(
            "Optional strictly increasing coordinates. Each value goes wholly to its nearest coordinate within tolerance. Empty combinations stay masked; there is no interpolation."
        )
        tolerance = QtWidgets.QDoubleSpinBox()
        tolerance.setObjectName("metadata_dimension_tolerance")
        tolerance.setDecimals(8)
        tolerance.setRange(0, 1e10)
        tolerance.setValue(current.get("tolerance", 0.5))
        tolerance.setToolTip(
            "Maximum allowed distance from an explicit coordinate, in the axis units. Outliers and equal-distance ties are errors. Unused with exact unique values."
        )
        for label, widget in (
            ("Channel", source),
            ("Axis name", name),
            ("Units", units),
            ("Read values", sampling),
            ("Discrete coordinates", centers),
            ("Assignment tolerance", tolerance),
        ):
            form.addRow(label, widget)
        preview = QtWidgets.QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setObjectName("metadata_dimension_preview")
        preview.setToolTip(
            "Resolved metadata ranges and assigned coordinates for every enabled source. Validation errors prevent applying the recipe."
        )
        form.addRow(preview)

        def specification():
            text = centers.text().strip()
            return MetadataDimension(
                name.text().strip(),
                source.currentText().strip(),
                units.text().strip(),
                str(sampling.currentData()),
                [float(value) for value in text.replace(",", " ").split()] if text else None,
                tolerance.value(),
            )

        def validate():
            spec = specification()
            resolved = []
            for dataset in candidates:
                _ensure_dataset_data_loaded(dataset)
                resolved.append(metadata_dimension_coordinates(dataset, spec))
            targets = (
                np.asarray(spec.centers)
                if spec.centers is not None
                else np.unique(np.concatenate(resolved))
            )
            lines = []
            for dataset, values in zip(candidates, resolved, strict=True):
                assigned = metadata_dimension_indices(
                    values, targets, spec.tolerance if spec.centers is not None else 0
                )
                labels = ", ".join(f"{value:g}" for value in np.unique(targets[assigned]))
                lines.append(f"{dataset.name}: {np.min(values):g} … {np.max(values):g} → {labels}")
            preview.setPlainText("\n".join(lines))
            return spec

        def show_preview():
            try:
                validate()
            except Exception as exc:
                preview.setPlainText(str(exc))

        check = QtWidgets.QPushButton("Preview assignments")
        check.setObjectName("metadata_dimension_check")
        check.setToolTip(
            "Read the selected metadata channel for all sources and check the assignment before applying."
        )
        check.clicked.connect(show_preview)
        form.addRow(check)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setToolTip(
            "Validate and save this dimension on the collection."
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
            "Close without changing the collection's dimensions."
        )

        def accept():
            try:
                spec = validate()
                updated = list(recipes)
                if existing is None:
                    updated.append(spec.to_dict())
                else:
                    updated[existing] = spec.to_dict()
                set_metadata_dimensions(group, updated)
            except Exception as exc:
                preview.setPlainText(str(exc))
                return
            dialog.accept()
            explorer._after_group_composite_changed(group)

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.exec()

    def copy_script():
        from .workflow import composite_workflow_script

        try:
            text = composite_workflow_script(
                explorer.project,
                _composite_root(group).name,
                node_id=group.node.id
                if hasattr(group, "node") and hasattr(group.node, "id")
                else None,
            )
            QtWidgets.QApplication.clipboard().setText(text)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(box, "Composite workflow script", str(exc))

    for text, key, tooltip, callback in (
        (
            "Add dimension…",
            "add",
            "Choose a metadata channel and assign discrete coordinates.",
            lambda: edit(),
        ),
        (
            "Edit…",
            "edit",
            "Edit the selected metadata dimension.",
            lambda: edit(items.currentRow()) if items.currentRow() >= 0 else None,
        ),
        (
            "Remove",
            "remove",
            "Remove the selected coordinate from this composite; source datasets are preserved.",
            lambda: (
                save([spec for i, spec in enumerate(recipes) if i != items.currentRow()])
                if items.currentRow() >= 0
                else None
            ),
        ),
        (
            "Copy composite script",
            "script",
            "Copy an editable Python recipe using the saved project, current metadata dimensions, and current rebin grid.",
            copy_script,
        ),
    ):
        button = QtWidgets.QPushButton(text)
        button.setObjectName(f"metadata_dimensions_{key}")
        button.setToolTip(tooltip)
        if key == "add":
            candidates = _composite_candidates(group)
            supported = bool(candidates) and all(
                _dataset_composite_kind(dataset) in {"point_data_4d", "mdhisto"}
                for dataset in candidates
            )
            button.setEnabled(supported)
            if not supported:
                button.setToolTip(
                    "Metadata dimensions require neutron point datasets or histograms. Raw event logs need an explicit alignment adapter."
                )
        button.clicked.connect(callback)
        row.addWidget(button)
    return box
