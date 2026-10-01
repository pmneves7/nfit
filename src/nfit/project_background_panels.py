"""Background link and shared source-symmetry presentation."""

from __future__ import annotations

from typing import Any

from PySide6 import QtWidgets

from .pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from .project_binning_policy import background_rebin_explanation
from .project_composites import (
    _composite_scope,
    data_group_composite_config,
    data_group_composite_enabled,
)
from .symmetry import symmetry_spec_from_config


def build_background_details(
    self: Any,
    group: DataGroup | None,
    owner: DatasetEntry | DataGroup | DatasetGroup,
    background: BackgroundSpec,
) -> None:
    self._clear_details_panel()
    box = QtWidgets.QGroupBox("Background subtraction")
    box.setToolTip(
        "A |Q|-energy source is interpolated onto the target grid. A single-crystal "
        "histogram must have identical axes and bins. Unrebinned neutron point references "
        "on a group background follow the group's grid automatically. The scaled signal is subtracted "
        "and the scaled variance is added."
    )
    layout = QtWidgets.QGridLayout(box)
    enabled = QtWidgets.QCheckBox("Enabled")
    enabled.setChecked(background.enabled)
    enabled.setToolTip("Enable or temporarily bypass this background subtraction.")
    enabled.toggled.connect(
        lambda checked: self._update_background(group, owner, background, enabled=bool(checked))
    )
    layout.addWidget(enabled, 0, 0, 1, 2)
    layout.addWidget(QtWidgets.QLabel("Source"), 1, 0)
    source_combo = QtWidgets.QComboBox()
    source_combo.setObjectName("background_source_dataset")
    source_combo.setToolTip(
        "Powder |Q|-energy data, an identically binned histogram, or a dataset group including its enabled runs. "
        "For group backgrounds, unrebinned neutron point data automatically use the sample grid."
    )
    candidates = (
        []
        if group is None
        else [
            candidate
            for candidate in group.iter_datasets()
            if candidate is not owner
            and candidate.data_type in {"powder_inelastic", "single_crystal_inelastic"}
        ]
    )
    for candidate in candidates:
        source_combo.addItem(candidate.name, candidate.id)
    if group is not None:
        for candidate in group.iter_subgroups():
            if (
                candidate is owner
                or any(node is owner for node in candidate.iter_subgroups())
                or any(node is owner for node in candidate.iter_datasets())
            ):
                continue
            if (
                candidate.id == background.source_group_id
                or data_group_composite_enabled(_composite_scope(group, candidate))
                or any(
                    node.data_type in {"powder_inelastic", "single_crystal_inelastic"}
                    for node in candidate.iter_datasets()
                )
            ):
                source_combo.addItem(f"{candidate.name} [dataset group]", f"group:{candidate.id}")
    selected_source = (
        f"group:{background.source_group_id}"
        if background.source_group_id
        else background.source_dataset_id
    )
    source_index = source_combo.findData(selected_source)
    if source_index < 0:
        source_combo.addItem(f"Unresolved source: {selected_source or '(none)'}", selected_source)
        source_index = source_combo.count() - 1
    source_combo.setCurrentIndex(source_index)

    def change_source(_index: int) -> None:
        self._set_background_source(group, owner, background, source_combo.currentData())
        self._set_background_details(group, owner, background)

    source_combo.currentIndexChanged.connect(change_source)
    layout.addWidget(source_combo, 1, 1)
    layout.addWidget(QtWidgets.QLabel("Scale"), 2, 0)
    scale = QtWidgets.QDoubleSpinBox()
    scale.setObjectName("background_scale")
    scale.setRange(-1.0e12, 1.0e12)
    scale.setDecimals(8)
    scale.setValue(float(background.scale))
    scale.setToolTip(
        "Multiplier applied to this background before subtraction. Its absolute value also scales the uncertainty."
    )
    scale.valueChanged.connect(
        lambda value: self._update_background(group, owner, background, scale=float(value))
    )
    layout.addWidget(scale, 2, 1)
    layout.addWidget(QtWidgets.QLabel("Projection"), 3, 0)
    projection = QtWidgets.QComboBox()
    projection.setObjectName("background_projection")
    projection.addItem("Voxel center", "center")
    supports_trajectory_projection = isinstance(owner, DatasetGroup) and "mdevent" in owner.metadata
    if supports_trajectory_projection or background.projection == "sample_trajectories":
        projection.addItem("Powder through sample trajectories", "sample_trajectories")
    if supports_trajectory_projection or background.projection == "measured_events":
        projection.addItem("Measured background at sample angles", "measured_events")
    projection.setToolTip(
        "Voxel center is the fast interpolation of B(|Q|, E) at each target-bin center. "
        "Powder through sample trajectories forward-projects a spherical powder background through "
        "every MDEvent sample angle and the same detector-trajectory normalization; "
        "Measured background at sample angles instead replays the measured lab-frame events, "
        "preserving out-of-plane dependence. It requires a referenced MDEvent group with "
        "matching detector geometry and incident energy; its powder binning and interpolation "
        "are not used. These replay modes require an MDEvent sample group."
    )
    projection.setCurrentIndex(max(projection.findData(background.projection), 0))
    projection.currentIndexChanged.connect(
        lambda _index: self._update_background(
            group,
            owner,
            background,
            projection=str(projection.currentData()),
        )
    )
    layout.addWidget(projection, 3, 1)
    layout.addWidget(QtWidgets.QLabel("Interpolation"), 4, 0)
    interpolation = QtWidgets.QComboBox()
    interpolation.setObjectName("background_interpolation")
    interpolation.addItem("Linear", "linear")
    interpolation.addItem("Nearest", "nearest")
    interpolation.setToolTip(
        "Interpolation used for powder data. Measured-event replay and identically binned single-crystal subtraction do not interpolate."
    )
    interpolation.setCurrentIndex(max(interpolation.findData(background.interpolation), 0))
    interpolation.setEnabled(background.projection != "measured_events")
    projection.currentIndexChanged.connect(
        lambda _index: interpolation.setEnabled(projection.currentData() != "measured_events")
    )
    interpolation.currentIndexChanged.connect(
        lambda _index: self._update_background(
            group,
            owner,
            background,
            interpolation=str(interpolation.currentData()),
        )
    )
    layout.addWidget(interpolation, 4, 1)
    binning_explanation = QtWidgets.QLabel()
    binning_explanation.setObjectName("background_binning_explanation")
    binning_explanation.setWordWrap(True)
    binning_explanation.setToolTip(
        "Identifies which binning recipe this background subtraction actually uses. "
        "Selecting a different projection can change whether the source's viewing grid is used."
    )

    def update_binning_explanation(*_args: Any) -> None:
        binning_explanation.setText(background_rebin_explanation(background, owner=owner))

    update_binning_explanation()
    projection.currentIndexChanged.connect(update_binning_explanation)
    source_combo.currentIndexChanged.connect(update_binning_explanation)
    enabled.toggled.connect(update_binning_explanation)
    layout.addWidget(binning_explanation, 5, 0, 1, 2)
    self.details_layout.addWidget(box)
    source_group = background.source_group
    if source_group is None and group is not None:
        source_group = next(
            (node for node in group.iter_subgroups() if node.id == background.source_group_id), None
        )
    if source_group is not None and group is not None:
        self.details_layout.addWidget(_source_symmetry_box(self, group, source_group))
    self.details_layout.addStretch(1)


def _source_symmetry_box(
    explorer: Any, root: DataGroup, source: DatasetGroup
) -> QtWidgets.QGroupBox:
    scope = _composite_scope(root, source)
    config = data_group_composite_config(scope)
    spec = symmetry_spec_from_config(config.get("symmetry"))
    box = QtWidgets.QGroupBox("Source symmetry (shared)")
    box.setToolTip(
        "Symmetry of the linked dataset group, shared by all its background subtractions."
    )
    layout = QtWidgets.QFormLayout(box)
    enabled = QtWidgets.QCheckBox("Apply symmetry")
    enabled.setObjectName("background_source_symmetry_enabled")
    enabled.setChecked(spec.enabled)
    enabled.setToolTip(
        "Apply the source group's reciprocal-HKL symmetry before background binning; energy is unchanged."
    )
    enabled.toggled.connect(
        lambda checked: explorer._set_group_composite_symmetry_enabled(scope, checked)
    )
    layout.addRow(enabled)
    mode = QtWidgets.QComboBox()
    mode.setObjectName("background_source_symmetry_mode")
    for label, value in (
        ("Space group", "space_group"),
        ("Point group", "point_group"),
        ("Operations", "operations"),
        ("Generators", "generators"),
    ):
        mode.addItem(label, value)
    displayed = (
        spec.mode
        if spec.mode != "none"
        else config.get("symmetry", {}).get("last_mode", "space_group")
    )
    mode.setCurrentIndex(max(mode.findData(displayed), 0))
    mode.setToolTip("Notation used for the shared source symmetry expression.")
    mode.currentIndexChanged.connect(
        lambda _index: explorer._set_group_composite_symmetry_mode(scope, str(mode.currentData()))
    )
    layout.addRow("Notation", mode)
    expression = QtWidgets.QLineEdit(spec.expression)
    expression.setObjectName("background_source_symmetry_expression")
    expression.setToolTip(
        "Reciprocal-HKL operations or generators, for example rotate(order=3, axis=[1,1,1])."
    )
    expression.editingFinished.connect(
        lambda: explorer._set_group_composite_symmetry_expression(scope, expression.text())
    )
    layout.addRow("Expression", expression)
    note = QtWidgets.QLabel(
        "These settings belong to the source dataset group and apply to every subtraction using it. For single-crystal subtraction, each sample binning owns its background histogram, built using the source UB and symmetry. Changing symmetry requires rebuilding affected histograms; changing the background scale reuses them."
    )
    note.setWordWrap(True)
    layout.addRow(note)
    return box
