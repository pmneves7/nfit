"""Dataset-condition preview controls backed by the public selection API."""

from __future__ import annotations

import copy

import numpy as np
from PySide6 import QtWidgets

from .dataset_criteria import criterion_matches, dataset_criterion_preview


def dataset_criterion_panel(group, mask, *, changed, start_task, parent=None):
    """Build one focused editor; the coordinator supplies guarded job execution."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    widget = QtWidgets.QWidget(parent)
    layout = QtWidgets.QVBoxLayout(widget)
    controls = QtWidgets.QFormLayout()
    layout.addLayout(controls)

    def named(control, name, tip):
        control.setObjectName(f"dataset_criterion_{name}")
        control.setToolTip(tip)
        return control

    channel = named(
        QtWidgets.QComboBox(),
        "channel",
        "Choose the unsubtracted elastic-window intensity or one numeric metadata channel per dataset.",
    )
    channel.addItem("Elastic line intensity", "elastic")
    channel.addItem("Metadata", "metadata")
    channel.setCurrentIndex(max(0, channel.findData(mask.parameters.get("channel", "elastic"))))
    controls.addRow("Channel", channel)
    energy = QtWidgets.QWidget()
    energy_row = QtWidgets.QHBoxLayout(energy)
    energy_row.setContentsMargins(0, 0, 0, 0)
    energies = []
    for side, default in (("min", -0.5), ("max", 0.5)):
        field = named(
            QtWidgets.QLineEdit(str(mask.parameters.get(f"energy_{side}", default))),
            f"energy_{side}",
            "Energy transfer in meV; include the lower endpoint and exclude the upper endpoint.",
        )
        energy_row.addWidget(field)
        energies.append(field)
    controls.addRow("Energy window (meV)", energy)
    source = named(
        QtWidgets.QComboBox(),
        "source",
        "Select or type a numeric metadata path. Timestamped NeXus logs support duration-weighted time averages.",
    )
    source.setEditable(True)
    source.addItem(str(mask.parameters.get("source", "metadata/incident_energy")))
    controls.addRow("Metadata channel", source)
    statistic = named(
        QtWidgets.QComboBox(),
        "statistic",
        "Minimum/maximum use recorded values. Time average integrates a stepwise log over the run's start/end times; scalar channels use their scalar value.",
    )
    for label, key in (("Minimum", "min"), ("Maximum", "max"), ("Time average", "time_average")):
        statistic.addItem(label, key)
    statistic.setCurrentIndex(
        max(0, statistic.findData(mask.parameters.get("statistic", "time_average")))
    )
    controls.addRow("Per-dataset statistic", statistic)

    plot = Figure(figsize=(7, 3.5), constrained_layout=True)
    canvas = FigureCanvasQTAgg(plot)
    canvas.setMinimumHeight(260)
    canvas.setToolTip(
        "One value per dataset. Red points match this condition; blue points do not. Other masks and Enabled flags also affect selection."
    )
    canvas.setObjectName("dataset_criterion_plot")
    ax = plot.add_subplot(111)
    toolbar = NavigationToolbar2QT(canvas, widget)
    toolbar.setToolTip(
        "Matplotlib navigation: zoom, pan, configure axes and save the preview figure."
    )
    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    condition = QtWidgets.QHBoxLayout()
    condition.addWidget(QtWidgets.QLabel("Exclude when"))
    bounds, comparisons = [], []
    for side in ("left", "right"):
        field = named(
            QtWidgets.QLineEdit(
                ""
                if mask.parameters.get(f"{side}_value") is None
                else str(mask.parameters[f"{side}_value"])
            ),
            f"{side}_value",
            "A blank limit is inactive. Changing a limit moves its dashed line without recalculating data.",
        )
        combo = named(
            QtWidgets.QComboBox(),
            f"{side}_operator",
            "Comparison in the displayed expression: left limit op value op right limit. Both active comparisons must hold to exclude a dataset.",
        )
        combo.addItems(["<", "<=", ">", ">="])
        combo.setCurrentText(
            mask.parameters.get(f"{side}_operator", "<" if side == "left" else ">")
        )
        if side == "left":
            condition.addWidget(field)
            condition.addWidget(combo)
            condition.addWidget(QtWidgets.QLabel("value"))
        else:
            condition.addWidget(combo)
            condition.addWidget(field)
        bounds.append(field)
        comparisons.append(combo)
    layout.addLayout(condition)
    options = QtWidgets.QHBoxLayout()
    invert = named(
        QtWidgets.QCheckBox("Invert"),
        "invert",
        "Exclude datasets outside the active condition instead. Blank limits leave the condition inactive.",
    )
    invert.setChecked(mask.invert)
    additive = named(
        QtWidgets.QCheckBox("Additive"),
        "additive",
        "Restore matching datasets excluded by earlier dataset-condition masks. Does not override Enabled flags or coordinate masks.",
    )
    additive.setChecked(mask.additive)
    yscale = named(
        QtWidgets.QComboBox(),
        "yscale",
        "Change the preview y axis to linear, logarithmic or symmetric logarithmic scale.",
    )
    yscale.addItems(["linear", "log", "symlog"])
    options.addWidget(invert)
    options.addWidget(additive)
    options.addWidget(QtWidgets.QLabel("Y scale"))
    options.addWidget(yscale)
    options.addStretch()
    layout.addLayout(options)
    status = named(
        QtWidgets.QLabel(),
        "status",
        "Preview calculations read source events/metadata in the background. Changed sources/settings require Calculate values again.",
    )
    status.setWordWrap(True)
    layout.addWidget(status)
    calculate = named(
        QtWidgets.QPushButton("Calculate values"),
        "calculate",
        "Read/reuse one scalar per run. Existing reduced-event caches are streamed lazily; no 4D cube or background subtraction is constructed.",
    )
    layout.addWidget(calculate)
    state = {
        "rows": mask.metadata.get("dataset_criterion_preview", []),
        "dirty": False,
    }
    lines = [ax.axhline(0, color="black", linestyle="--", visible=False) for _ in range(2)]
    points = ax.scatter([], [], s=14)

    def recipe():
        result = dict(
            mask.parameters,
            channel=channel.currentData(),
            source=source.currentText(),
            statistic=statistic.currentData(),
        )
        for side, field, combo in zip(("left", "right"), bounds, comparisons, strict=True):
            result[f"{side}_value"] = None if not field.text().strip() else float(field.text())
            result[f"{side}_operator"] = combo.currentText()
        for side, field in zip(("min", "max"), energies, strict=True):
            result[f"energy_{side}"] = float(field.text())
        if (
            not np.isfinite([result["energy_min"], result["energy_max"]]).all()
            or result["energy_min"] >= result["energy_max"]
        ):
            raise ValueError("Energy limits must be finite and increasing")
        criterion_matches(0.0, result)
        return result

    def draw(*_args, reset=False):
        try:
            params = recipe()
        except ValueError:
            status.setText("Enter finite numbers; blank exclusion limits are inactive.")
            return
        rows = state["rows"]
        matches = [criterion_matches(row["value"], params) for row in rows]
        if invert.isChecked() and any(
            params[f"{side}_value"] is not None for side in ("left", "right")
        ):
            matches = [not match for match in matches]
        xy = np.array([[r["number"], r["value"]] for r in rows]).reshape(-1, 2)
        points.set_offsets(xy)
        points.set_color(["tab:red" if match else "tab:blue" for match in matches])
        for side, line in zip(("left", "right"), lines, strict=True):
            value = params[f"{side}_value"]
            line.set_visible(value is not None)
            if value is not None:
                line.set_ydata([value, value])
        ax.set_xlabel("Dataset number (run number when available)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        unit = rows[0].get("units", "") if rows else ""
        ax.set_ylabel(f"Value ({unit})" if unit else "Value")
        ax.set_yscale(yscale.currentText())
        if reset and len(xy):
            from matplotlib.transforms import Bbox

            ax.dataLim = Bbox.null()
            ax.update_datalim(xy)
            ax.autoscale_view()
        status.setText(
            "Calculate values to preview the datasets."
            if not rows
            else f"{sum(matches)} of {len(rows)} datasets match. "
            + (
                "Diagnostic settings changed; calculate again before using the mask."
                if state["dirty"]
                else "Blue: no match. Red: match."
                + (" Matching runs restore earlier condition exclusions." if mask.additive else "")
                + (" This mask is disabled." if not mask.enabled else "")
            )
        )
        canvas.draw_idle()

    def edit(*_args, diagnostic=False):
        try:
            params = recipe()
        except ValueError:
            draw()
            return
        if diagnostic and params != mask.parameters:
            state["dirty"] = True
            state["rows"] = []
            mask.metadata.pop("dataset_criterion_preview", None)
        if (
            params != mask.parameters
            or invert.isChecked() != mask.invert
            or additive.isChecked() != mask.additive
        ):
            mask.parameters = params
            mask.invert, mask.additive = invert.isChecked(), additive.isChecked()
            if not state["rows"] and any(
                params[f"{side}_value"] is not None for side in ("left", "right")
            ):
                state["dirty"] = True
            changed()
        metadata = channel.currentData() == "metadata"
        for field, visible in ((source, metadata), (statistic, metadata), (energy, not metadata)):
            field.setVisible(visible)
            controls.labelForField(field).setVisible(visible)
        draw()

    def metadata_choices():
        if channel.currentData() == "metadata" and source.count() == 1:
            from .metadata_dimensions import metadata_channels

            entries = list(group.iter_datasets())
            if entries:
                try:
                    paths = metadata_channels(entries[0])
                    source.addItems(
                        [
                            path
                            for path in paths
                            if "/DASlogs/" in path
                            or path.startswith(("metadata/", "parameters/"))
                            or path == "temperature"
                        ]
                    )
                except (OSError, ValueError):
                    pass  # An editable path remains available for unavailable sources.
        edit(diagnostic=True)

    def compute():
        try:
            params = recipe()
        except ValueError:
            status.setText("Enter finite numbers and an increasing energy window.")
            return
        edit(diagnostic=True)
        working = copy.deepcopy(mask)
        working.parameters = params

        def success(rows):
            mask.parameters = working.parameters
            mask.metadata.update(working.metadata)
            state.update(rows=rows, dirty=False)
            changed()
            draw(reset=True)

        start_task(
            task=lambda progress: dataset_criterion_preview(
                group, working, progress_callback=progress
            ),
            on_success=success,
        )

    for field in bounds:
        field.textChanged.connect(draw)
        field.editingFinished.connect(edit)
    for combo in comparisons:
        combo.currentTextChanged.connect(edit)
    for field in energies:
        field.editingFinished.connect(lambda: edit(diagnostic=True))
    channel.currentIndexChanged.connect(metadata_choices)
    source.activated.connect(lambda: edit(diagnostic=True))
    source.lineEdit().editingFinished.connect(lambda: edit(diagnostic=True))
    statistic.currentIndexChanged.connect(lambda: edit(diagnostic=True))
    invert.toggled.connect(edit)
    additive.toggled.connect(edit)
    yscale.currentTextChanged.connect(draw)
    calculate.clicked.connect(compute)
    # Keep tests and scripting adapters independent of matplotlib internals.
    widget.criterion_axes = ax
    widget.criterion_lines = lines
    edit()
    draw(reset=True)
    return widget
