"""Metadata-channel editor and per-dimension composite rebin controls."""

from __future__ import annotations

import numpy as np

from .metadata_dimensions import (
    MetadataBinning,
    MetadataDimension,
    metadata_channels,
    metadata_dimension_coordinates,
    metadata_dimension_grid,
)
from .qt_controls import (
    COMPACT_SHORT_TEXT_FIELD_WIDTH,
    constrain_input_width,
)


def metadata_rebin_rows(explorer, group, layout, start_row):
    """Append metadata axes after the physical axes in the composite rebinner."""
    from PySide6 import QtWidgets

    from .project_data import (
        _composite_candidates,
        _ensure_dataset_data_loaded,
        set_metadata_dimensions,
    )

    recipes = list(group.metadata.get("metadata_dimensions", []))

    def add_row(index, recipe):
        spec = MetadataDimension(**recipe)
        binning = spec.binning
        event_pulse_time = spec.sampling == "event_pulse_time"
        existing_edges = None if binning is None else binning.edges()
        default_lower = (
            None
            if binning is None
            else binning.lower
            if binning.lower is not None
            else float(existing_edges[0])
        )
        default_upper = (
            None
            if binning is None
            else binning.upper
            if binning.upper is not None
            else float(existing_edges[-1])
        )
        row = start_row + index
        label = QtWidgets.QLabel(f"{spec.name} ({spec.units})" if spec.units else spec.name)
        label.setObjectName(f"metadata_rebin_label_{index}")
        label.setToolTip(
            f"Metadata channel: {spec.source}. Its values enter the same N-dimensional rebin as the physical coordinates."
        )
        layout.addWidget(label, row, 0)
        lower = QtWidgets.QLineEdit("" if default_lower is None else str(default_lower))
        upper = QtWidgets.QLineEdit("" if default_upper is None else str(default_upper))
        mode = QtWidgets.QComboBox()
        choices = (
            (("Step", "step"), ("Number of bins", "bins"), ("Edges", "edges"))
            if event_pulse_time
            else (
                ("Discrete", "discrete"),
                ("Step", "step"),
                ("Number of bins", "bins"),
                ("Edges", "edges"),
                ("Tolerance", "tolerance"),
            )
        )
        for title, value in choices:
            mode.addItem(title, value)
        selected_mode = (
            "discrete"
            if binning is None
            else (
                "edges"
                if binning.bin_edges is not None
                else "step"
                if binning.step is not None
                else "bins"
                if binning.num_bins is not None
                else "tolerance"
            )
        )
        mode.setCurrentIndex(mode.findData(selected_mode))
        if mode.currentIndex() < 0:
            mode.setCurrentIndex(0)
        resolution = QtWidgets.QLineEdit(
            ""
            if binning is None
            else str(
                binning.step
                or binning.num_bins
                or binning.tolerance
                or (float(np.median(np.diff(existing_edges))) if existing_edges.size > 1 else "")
            )
        )
        edges = QtWidgets.QLineEdit(
            ", ".join(f"{v:g}" for v in (binning.bin_edges or [])) if binning else ""
        )
        for key, widget, tooltip in (
            (
                "lower",
                lower,
                "First metadata bin center. With Step, a start of 0 K, stop of 300 K, and step of 10 K produces centers 0, 10, …, 300 K.",
            ),
            (
                "upper",
                upper,
                "Last metadata bin center. For exact interval edges instead, choose Edges and enter the full edge list.",
            ),
            (
                "mode",
                mode,
                "Step uses the displayed start, stop, and spacing. Number of bins divides the displayed range into that many bins. Raw event-pulse metadata require one of these explicit grids.",
            ),
            (
                "resolution",
                resolution,
                "For Step, enter the spacing in axis units (for example 10 K). For Number of bins, enter a positive integer count.",
            ),
            (
                "edges",
                edges,
                "Strictly increasing metadata bin edges, separated by commas. Interior edges enter the bin to their right; the final edge is inclusive. Values outside are excluded.",
            ),
        ):
            widget.setObjectName(f"metadata_rebin_{key}_{index}")
            widget.setToolTip(tooltip)
        lower.setPlaceholderText("first coordinate")
        upper.setPlaceholderText("last coordinate")
        resolution.setPlaceholderText("step / count")
        edges.setPlaceholderText("bin edges")
        layout.addWidget(lower, row, 1)
        layout.addWidget(upper, row, 2)
        layout.addWidget(resolution, row, 3)
        layout.addWidget(edges, row, 4)
        layout.addWidget(mode, row, 5)
        assignment = QtWidgets.QComboBox()
        assignment.setObjectName(f"metadata_rebin_assignment_{index}")
        assignment.addItem("Fractional", True)
        assignment.addItem("Discrete", False)
        assignment.setCurrentIndex(
            max(
                assignment.findData(
                    False if binning is None else bool(binning.fractional)
                ),
                0,
            )
        )
        assignment.setToolTip(
            "Fractional distributes a metadata coordinate between neighboring bins. "
            "Discrete assigns it wholly to one bin. Discrete and Tolerance grids require "
            "discrete assignment."
        )
        layout.addWidget(assignment, row, 6)

        def enable_fields():
            selected = mode.currentData()
            uniform = selected in {"step", "bins"}
            lower.setEnabled(uniform)
            upper.setEnabled(uniform)
            resolution.setEnabled(uniform or selected == "tolerance")
            edges.setEnabled(mode.currentData() == "edges")
            assignment.setEnabled(selected not in {"discrete", "tolerance"})

        def apply():
            try:
                selected = mode.currentData()
                if selected == "discrete":
                    updated_binning = None
                elif selected == "tolerance":
                    updated_binning = MetadataBinning(
                        tolerance=float(resolution.text()),
                        fractional=False,
                    )
                elif selected == "edges":
                    updated_binning = MetadataBinning(
                        bin_edges=[
                            float(v) for v in edges.text().strip("[] ").replace(",", " ").split()
                        ],
                        fractional=bool(assignment.currentData()),
                    )
                else:
                    updated_binning = MetadataBinning(
                        lower=float(lower.text()),
                        upper=float(upper.text()),
                        **{
                            ("step" if selected == "step" else "num_bins"): float(resolution.text())
                        },
                        fractional=bool(assignment.currentData()),
                    )
                updated = [dict(item) for item in recipes]
                updated[index]["binning"] = updated_binning
                set_metadata_dimensions(group, updated)
            except (ValueError, TypeError) as exc:
                QtWidgets.QMessageBox.warning(mode, "Metadata rebinning", str(exc))
                return
            explorer._after_group_composite_changed(group)

        def change_mode():
            enable_fields()
            try:
                if event_pulse_time:
                    if not lower.text():
                        lower.setText(str(default_lower if default_lower is not None else 0.0))
                    if not upper.text():
                        upper.setText(str(default_upper if default_upper is not None else 1.0))
                    if mode.currentData() == "step" and not resolution.text():
                        resolution.setText(
                            str(
                                float(np.median(np.diff(existing_edges)))
                                if existing_edges is not None and existing_edges.size > 1
                                else 1.0
                            )
                        )
                    elif mode.currentData() == "bins" and not resolution.text():
                        resolution.setText(
                            str(max((existing_edges.size - 1) if existing_edges is not None else 1, 1))
                        )
                    apply()
                    return
                if mode.currentData() != "discrete":
                    coordinates = []
                    for entry in _composite_candidates(group, include_backgrounds=True):
                        _ensure_dataset_data_loaded(entry)
                        coordinates.append(metadata_dimension_coordinates(entry, spec))
                    values = (
                        np.asarray(spec.centers)
                        if spec.centers is not None
                        else np.unique(np.concatenate(coordinates))
                    )
                    if not lower.text():
                        lower.setText(str(values[0]))
                    if not upper.text():
                        upper.setText(str(values[-1]))
                    if mode.currentData() == "step":
                        resolution.setText(
                            str(float(np.min(np.diff(values))) if len(values) > 1 else 1.0)
                        )
                    elif mode.currentData() == "bins":
                        if lower.text() == upper.text():
                            lower.setText(str(values[0] - 0.5))
                            upper.setText(str(values[0] + 0.5))
                        resolution.setText(str(len(values)))
                    elif mode.currentData() == "tolerance":
                        resolution.setText(
                            str(float(np.min(np.diff(values))) / 3 if len(values) > 1 else 0.1)
                        )
                    elif not edges.text():
                        from dataclasses import replace

                        from .metadata_dimensions import discrete_metadata_axis

                        boundaries = discrete_metadata_axis(
                            replace(spec, binning=None), values
                        ).values
                        edges.setText(", ".join(str(v) for v in boundaries))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(mode, "Metadata rebinning", str(exc))
                return
            apply()

        enable_fields()
        mode.currentIndexChanged.connect(change_mode)
        assignment.currentIndexChanged.connect(apply)
        for widget in (lower, upper, resolution, edges):
            widget.editingFinished.connect(apply)

    for index, recipe in enumerate(recipes):
        add_row(index, recipe)
    return len(recipes)


def metadata_dimensions_panel(explorer, group, *, embedded: bool = False):
    from PySide6 import QtCore, QtWidgets

    from .project_data import (
        _composite_candidates,
        _composite_root,
        _dataset_composite_kind,
        _ensure_dataset_data_loaded,
        set_metadata_dimensions,
    )

    box = QtWidgets.QWidget() if embedded else QtWidgets.QGroupBox("Metadata dimensions")
    box.setToolTip(
        "Add aligned sample-condition coordinates to this collection's central composite rebin."
    )
    layout = QtWidgets.QVBoxLayout(box)
    explanation = QtWidgets.QLabel(
        "Choose aligned metadata coordinates for the composite. Configure their grids and assignment modes under Rebin settings."
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
        candidates = _composite_candidates(group, include_backgrounds=True)
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
        constrain_input_width(name, COMPACT_SHORT_TEXT_FIELD_WIDTH)
        name.setObjectName("metadata_dimension_name")
        name.setToolTip(
            "Axis label shown in plots and the data viewer; it must differ from existing axis names."
        )
        units = QtWidgets.QLineEdit(current.get("units", channels.get(source.currentText(), "")))
        constrain_input_width(units, COMPACT_SHORT_TEXT_FIELD_WIDTH)
        units.setObjectName("metadata_dimension_units")
        units.setToolTip(
            "Physical units of this coordinate, for example K or degree. This labels values; it does not convert units."
        )
        source.currentTextChanged.connect(lambda text: units.setText(channels.get(text, "")))
        sampling = QtWidgets.QComboBox()
        sampling.setObjectName("metadata_dimension_sampling")
        sampling.setToolTip(
            "Mean or median assigns one condition to the whole dataset. Per point requires scalar or explicitly aligned point/scan values. Event pulse time linearly aligns a timestamped NeXus log to each raw CORELLI event pulse."
        )
        for label, key in (
            ("One value per dataset: mean", "dataset_mean"),
            ("One value per dataset: median", "dataset_median"),
            ("One value per measured point", "per_point"),
            ("One value per raw event pulse time", "event_pulse_time"),
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
                binning=current.get("binning"),
            )

        def validate():
            spec = specification()
            if spec.sampling == "event_pulse_time":
                preview.setPlainText(
                    "The timestamped NeXus log is interpolated at every raw event pulse "
                    "during rebinning. Configure its explicit grid under Rebin settings."
                )
                return spec
            resolved = []
            for dataset in candidates:
                _ensure_dataset_data_loaded(dataset)
                resolved.append(metadata_dimension_coordinates(dataset, spec))
            targets, assignments = metadata_dimension_grid(spec, resolved)
            lines = []
            for dataset, values, assigned in zip(candidates, resolved, assignments, strict=True):
                labels = ", ".join(f"{value:g}" for value in np.unique(targets[assigned[assigned >= 0]]))
                if np.any(assigned < 0):
                    labels += " (out-of-range values excluded)"
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
            candidates = _composite_candidates(group, include_backgrounds=True)
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
