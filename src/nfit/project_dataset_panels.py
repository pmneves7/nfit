"""Qt builders for dataset detail, rebin, point-list, and spectral panels."""

from __future__ import annotations

from typing import Any

from .project_rebin_panels import rebin_symmetry_expression_width
from .qt_controls import (
    COMPACT_SCALAR_FIELD_WIDTH,
    COMPACT_SHORT_TEXT_FIELD_WIDTH,
    constrain_input_width,
)

# The compatibility dispatcher supplies these established project_gui globals
# at invocation time. Keeping the contract explicit avoids a reverse import.
DATA_TYPE_DEFINITIONS: Any = None
DEFAULT_DATA_TYPE: Any = None
DataGroup: Any = None
DatasetEntry: Any = None
HEAT_CAPACITY_CHANNEL_LABEL: Any = None
HEAT_CAPACITY_OVER_T_CHANNEL_LABEL: Any = None
MDHistoData: Any = None
PPMS_HEAT_CAPACITY_UNITS: Any = None
PointData4D: Any = None
PointListData: Any = None
QUANTITY_TYPES: Any = None
_CompositeScope: Any = None
_add_metadata_tree_item: Any = None
_bold_label: Any = None
_compact_point_list_form: Any = None
_composite_root: Any = None
_dataset_can_rebin: Any = None
_dataset_metadata_mapping: Any = None
_dataset_rebin_status_text: Any = None
_format_number: Any = None
_momentum_rebin_matrix: Any = None
_momentum_rebin_vector_text: Any = None
_parameter_to_text: Any = None
_parse_parameter_text: Any = None
_peek_cached_dataset_view: Any = None
_rebin_axis_bound_is_auto: Any = None
_rebin_axis_mode: Any = None
_rebin_axis_fractional: Any = None
_rebin_max_batch_mb: Any = None
_rebin_mean_weighting: Any = None
_rebin_minimum_coverage: Any = None
_rebin_minimum_samples: Any = None
_sanitize_rebin_axis_config: Any = None
_saved_binning_compressed_size: Any = None
available_data_types: Any = None
available_ions: Any = None
data_group_composite_config: Any = None
dataset_rebin_config: Any = None
dataset_rebin_binnings: Any = None
np: Any = None
point_list_config: Any = None
signal_semantics: Any = None
symmetry_spec_from_config: Any = None
effective_dataset_masks: Any = None


def _dataset_point_list_group_box(self, dataset: DatasetEntry, group: DataGroup | None) -> Any:
    from PySide6 import QtWidgets

    config = point_list_config(dataset)
    data = dataset.data
    columns = list(data.column_names)
    definition = DATA_TYPE_DEFINITIONS.get(dataset.data_type, {})

    group_box = QtWidgets.QGroupBox("Variables and Channels")
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)

    # Independent coordinates (comma-separated column names).
    coord_row = QtWidgets.QHBoxLayout()
    coord_row.addWidget(QtWidgets.QLabel("Coordinates"))
    coord_edit = QtWidgets.QLineEdit(", ".join(config.get("coordinate_names", [])))
    coord_edit.setObjectName("point_list_coordinates")
    coord_edit.setToolTip("Comma-separated column names to use as independent coordinates.")
    coord_edit.editingFinished.connect(
        lambda editor=coord_edit: self._set_point_list_coordinates(dataset, group, editor.text())
    )
    coord_row.addWidget(coord_edit, 1)
    layout.addLayout(coord_row)

    # Channels: value + error column per channel.
    channels_box = QtWidgets.QGroupBox("Channels")
    channels_layout = QtWidgets.QGridLayout(channels_box)
    channels_layout.setContentsMargins(8, 6, 8, 6)
    channels_layout.addWidget(_bold_label(QtWidgets, "Value"), 0, 0)
    channels_layout.addWidget(_bold_label(QtWidgets, "Error"), 0, 1)
    channels_layout.addWidget(_bold_label(QtWidgets, "Quantity"), 0, 2)
    channels_layout.addWidget(_bold_label(QtWidgets, "Units"), 0, 3)
    for row, channel in enumerate(config.get("channels", []), start=1):
        value_combo = QtWidgets.QComboBox()
        value_combo.addItems(columns)
        value_combo.setCurrentText(str(channel.get("value", "")))
        value_combo.setToolTip("Column used as the signal values for this plotted or fitted channel.")
        value_combo.currentTextChanged.connect(
            lambda text, index=row - 1: self._set_point_list_channel(dataset, group, index, "value", text)
        )
        error_combo = QtWidgets.QComboBox()
        error_combo.addItems(["(none)", *columns])
        error_combo.setCurrentText(str(channel.get("error") or "(none)"))
        error_combo.setToolTip("Optional column containing one-sigma uncertainties for this channel.")
        error_combo.currentTextChanged.connect(
            lambda text, index=row - 1: self._set_point_list_channel(dataset, group, index, "error", text)
        )
        channels_layout.addWidget(value_combo, row, 0)
        channels_layout.addWidget(error_combo, row, 1)
        quantity_combo = QtWidgets.QComboBox()
        from .quantities import QUANTITY_TYPES

        quantity_combo.addItems(QUANTITY_TYPES)
        quantity_combo.setCurrentText(str(channel.get("quantity_type", "unknown")))
        quantity_combo.setToolTip(
            "Physical quantity represented by this channel. Models use it "
            "to validate that their output is comparable to the data."
        )
        quantity_combo.currentTextChanged.connect(
            lambda text, index=row - 1: self._set_point_list_channel(
                dataset, group, index, "quantity_type", text
            )
        )
        channels_layout.addWidget(quantity_combo, row, 2)
        unit_edit = QtWidgets.QLineEdit(str(channel.get("unit", "")))
        constrain_input_width(unit_edit, COMPACT_SHORT_TEXT_FIELD_WIDTH)
        unit_edit.setToolTip(
            "Physical unit of the value and uncertainty columns. Unit text "
            "is normalized and checked when a model predicts this channel."
        )
        unit_edit.editingFinished.connect(
            lambda editor=unit_edit, index=row - 1: self._set_point_list_channel(
                dataset, group, index, "unit", editor.text()
            )
        )
        channels_layout.addWidget(unit_edit, row, 3)
        remove_button = QtWidgets.QPushButton("Remove")
        remove_button.setToolTip("Remove this channel definition from the dataset configuration.")
        remove_button.clicked.connect(
            lambda _checked=False, index=row - 1: self._remove_point_list_channel(dataset, group, index)
        )
        channels_layout.addWidget(remove_button, row, 4)
    add_button = QtWidgets.QPushButton("Add channel")
    add_button.setToolTip("Add another signal channel using columns from this point-list dataset.")
    add_button.clicked.connect(lambda: self._add_point_list_channel(dataset, group))
    channels_layout.addWidget(add_button, len(config.get("channels", [])) + 1, 0)
    layout.addWidget(channels_box)

    # Magnetization already has dedicated physical-unit and normalization
    # controls, so the generic scale/unit transform would be redundant.
    if definition.get("scale") and not definition.get("susceptibility"):
        layout.addWidget(self._point_list_scale_box(dataset, group, config, columns))
    if definition.get("susceptibility"):
        layout.addWidget(self._point_list_susceptibility_box(dataset, group, config))
        layout.addWidget(self._magnetization_absolute_box(dataset, group))
    if definition.get("heat_capacity"):
        layout.addWidget(self._heat_capacity_box(dataset, group, config))
    if definition.get("wavelength"):
        layout.addWidget(self._point_list_wavelength_box(dataset, group, config))
    return group_box


def _heat_capacity_box(self, dataset, group, config) -> Any:
    from PySide6 import QtWidgets

    box = QtWidgets.QGroupBox("Heat-capacity normalization")
    grid = QtWidgets.QGridLayout(box)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.setColumnStretch(1, 1)
    enabled = QtWidgets.QCheckBox("Use sample mass and molar mass")
    enabled.setObjectName("heat_capacity_absolute_enabled")
    enabled.setChecked(bool(dataset.parameters.get("absolute_units", False)))
    enabled.setToolTip(
        "Convert the PPMS sample heat capacity from microjoules per kelvin "
        "to mJ/(mol K). Required by the Debye and low-temperature models."
    )
    enabled.toggled.connect(
        lambda checked: self._set_heat_capacity_setting(dataset, group, "absolute_units", bool(checked))
    )
    grid.addWidget(enabled, 0, 0, 1, 2)
    for row, (key, label, object_name, tooltip) in enumerate((
        ("sample_mass_mg", "Sample mass (mg)", "heat_capacity_sample_mass_mg",
         "Measured sample mass in milligrams."),
        ("molar_mass_g_mol", "Molar mass (g/mol)", "heat_capacity_molar_mass_g_mol",
         "Formula-unit molar mass in grams per mole."),
    ), start=1):
        widget_label = QtWidgets.QLabel(label)
        widget_label.setToolTip(tooltip)
        editor = QtWidgets.QLineEdit(_parameter_to_text(dataset.parameters.get(key, "")))
        constrain_input_width(editor, COMPACT_SCALAR_FIELD_WIDTH)
        editor.setObjectName(object_name)
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda ed=editor, setting=key: self._set_heat_capacity_setting(
                dataset, group, setting, _parse_parameter_text(ed.text())
            )
        )
        grid.addWidget(widget_label, row, 0)
        grid.addWidget(editor, row, 1)
    grid.addWidget(QtWidgets.QLabel("Plot and fit channel"), 3, 0)
    channel_combo = QtWidgets.QComboBox()
    channel_combo.setObjectName("heat_capacity_fit_channel")
    channel_combo.addItems([HEAT_CAPACITY_CHANNEL_LABEL, HEAT_CAPACITY_OVER_T_CHANNEL_LABEL])
    channel_combo.setCurrentText(
        str(config.get("heat_capacity", {}).get("fit_channel", HEAT_CAPACITY_CHANNEL_LABEL))
    )
    channel_combo.setToolTip(
        "Choose molar heat capacity C or C/T for the viewer and fit. Both physical models "
        "evaluate consistently in either representation."
    )
    channel_combo.currentTextChanged.connect(
        lambda text: self._set_heat_capacity_fit_channel(dataset, group, text)
    )
    grid.addWidget(channel_combo, 3, 1)
    grid.addWidget(QtWidgets.QLabel("PPMS source units"), 4, 0)
    unit_combo = QtWidgets.QComboBox()
    unit_combo.setObjectName("heat_capacity_source_unit")
    from .heat_capacity import PPMS_HEAT_CAPACITY_UNITS

    for unit in PPMS_HEAT_CAPACITY_UNITS:
        unit_combo.addItem(unit.replace("uJ", "µJ"), unit)
    current_unit = str(config.get("heat_capacity", {}).get("source_unit", "uJ/K"))
    unit_combo.setCurrentIndex(max(unit_combo.findData(current_unit), 0))
    unit_combo.setToolTip(
        "Unit selected when the PPMS heat-capacity file was exported. The file header "
        "seeds this value when possible; change it here if the header is ambiguous."
    )
    unit_combo.currentIndexChanged.connect(
        lambda _index, combo=unit_combo: self._set_heat_capacity_source_unit(
            dataset, group, str(combo.currentData())
        )
    )
    grid.addWidget(unit_combo, 4, 1)
    atoms_label = QtWidgets.QLabel("Atoms / formula unit")
    atoms_tooltip = (
        "Required only for PPMS gram-atom units and used in Debye beta conversions; "
        "enter the number of atoms represented by one formula unit."
    )
    atoms_label.setToolTip(atoms_tooltip)
    atoms_edit = QtWidgets.QLineEdit(
        _parameter_to_text(dataset.parameters.get("atoms_per_formula_unit", ""))
    )
    constrain_input_width(atoms_edit, COMPACT_SCALAR_FIELD_WIDTH)
    atoms_edit.setObjectName("heat_capacity_atoms_per_formula_unit")
    atoms_edit.setToolTip(atoms_tooltip)
    atoms_edit.editingFinished.connect(
        lambda ed=atoms_edit: self._set_heat_capacity_setting(
            dataset, group, "atoms_per_formula_unit", _parse_parameter_text(ed.text())
        )
    )
    grid.addWidget(atoms_label, 5, 0)
    grid.addWidget(atoms_edit, 5, 1)
    hint = QtWidgets.QLabel("Choose Temperature squared as the viewer x coordinate for C/T vs T².")
    hint.setWordWrap(True)
    hint.setToolTip(
        "T² is a derived coordinate, so the existing coordinate selector can switch axes "
        "without adding another permanent toolbar control."
    )
    grid.addWidget(hint, 6, 0, 1, 2)
    _compact_point_list_form(box, QtWidgets)
    return box


def _magnetization_absolute_box(self, dataset, group) -> Any:
    """Absolute-unit normalization controls for a magnetization dataset.

    When enabled the model predicts the moment in absolute emu using the
    sample's mass and molar mass to pin the emu/mol conversion, rather than
    fitting a free scale factor. Written to ``dataset.parameters`` so the
    magnetization evaluator can read them.
    """

    from PySide6 import QtWidgets

    box = QtWidgets.QGroupBox("Moment normalization")
    grid = QtWidgets.QGridLayout(box)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.setColumnStretch(1, 1)
    enable = QtWidgets.QCheckBox("Use sample mass and molar mass")
    enable.setObjectName("magnetization_absolute_enabled")
    enable.setToolTip(
        "Use the sample mass and formula-unit molar mass to put the measured "
        "moment and model prediction on an absolute scale. This is required "
        "for emu/mol and mu_B/f.u. output. Leave off to fit an arbitrary scale."
    )
    enable.setChecked(bool(dataset.parameters.get("absolute_units", False)))
    enable.toggled.connect(
        lambda checked: self._set_magnetization_absolute(
            dataset, group, "absolute_units", bool(checked)
        )
    )
    grid.addWidget(enable, 0, 0, 1, 2)
    grid.addWidget(QtWidgets.QLabel("Moment display / fit units"), 1, 0)
    output_combo = QtWidgets.QComboBox()
    output_combo.setObjectName("magnetization_output_unit")
    output_combo.addItem("Sample moment (emu)", "emu")
    output_combo.addItem("Sample moment (A m^2)", "A m^2")
    output_combo.addItem("Molar moment (emu/mol)", "emu/mol")
    output_combo.addItem("Formula-unit moment (μ_B/f.u.)", "mu_B/f.u.")
    output_index = output_combo.findData(
        str(dataset.parameters.get("magnetization_output_unit", "emu"))
    )
    output_combo.setCurrentIndex(max(output_index, 0))
    output_combo.setToolTip(
        "Unit used for the magnetic-moment plot and fit. Formula-unit moment "
        "is normalized using the sample mass and the formula-unit molar mass."
    )
    output_combo.currentIndexChanged.connect(
        lambda _index, combo=output_combo: self._set_magnetization_absolute(
            dataset, group, "magnetization_output_unit", combo.currentData()
        )
    )
    grid.addWidget(output_combo, 1, 1)
    mass_label = QtWidgets.QLabel("Sample mass (mg)")
    mass_tooltip = "Sample mass in milligrams, used to convert emu/mol to the measured emu moment."
    mass_label.setToolTip(mass_tooltip)
    grid.addWidget(mass_label, 2, 0)
    mass_edit = QtWidgets.QLineEdit(
        _parameter_to_text(dataset.parameters.get("sample_mass_mg", ""))
    )
    constrain_input_width(mass_edit, COMPACT_SCALAR_FIELD_WIDTH)
    mass_edit.setObjectName("magnetization_sample_mass_mg")
    mass_edit.setToolTip(mass_tooltip)
    mass_edit.editingFinished.connect(
        lambda ed=mass_edit: self._set_magnetization_absolute(
            dataset, group, "sample_mass_mg", _parse_parameter_text(ed.text())
        )
    )
    grid.addWidget(mass_edit, 2, 1)
    molar_label = QtWidgets.QLabel("Molar mass (g/mol)")
    molar_tooltip = "Formula-unit molar mass in grams per mole; sets the amount of substance for the emu/mol conversion."
    molar_label.setToolTip(molar_tooltip)
    grid.addWidget(molar_label, 3, 0)
    molar_edit = QtWidgets.QLineEdit(
        _parameter_to_text(dataset.parameters.get("molar_mass_g_mol", ""))
    )
    constrain_input_width(molar_edit, COMPACT_SCALAR_FIELD_WIDTH)
    molar_edit.setObjectName("magnetization_molar_mass_g_mol")
    molar_edit.setToolTip(molar_tooltip)
    molar_edit.editingFinished.connect(
        lambda ed=molar_edit: self._set_magnetization_absolute(
            dataset, group, "molar_mass_g_mol", _parse_parameter_text(ed.text())
        )
    )
    grid.addWidget(molar_edit, 3, 1)
    _compact_point_list_form(box, QtWidgets)
    return box


def _point_list_scale_box(self, dataset, group, config, columns) -> Any:
    from PySide6 import QtWidgets

    scale = config["scale"]
    box = QtWidgets.QGroupBox("Scale && units")
    grid = QtWidgets.QGridLayout(box)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.addWidget(QtWidgets.QLabel("Channel"), 0, 0)
    channel_combo = QtWidgets.QComboBox()
    channel_combo.addItems([str(c["label"]) for c in config.get("channels", [])])
    channel_combo.setCurrentText(str(scale.get("channel", "")))
    channel_combo.setToolTip("Signal channel to scale and assign units for viewing and fitting.")
    channel_combo.currentTextChanged.connect(
        lambda text: self._set_point_list_scale(dataset, group, "channel", text)
    )
    grid.addWidget(channel_combo, 0, 1)
    grid.addWidget(QtWidgets.QLabel("Factor"), 1, 0)
    factor_spin = QtWidgets.QDoubleSpinBox()
    factor_spin.setDecimals(8)
    factor_spin.setRange(-1.0e12, 1.0e12)
    factor_spin.setValue(float(scale.get("factor", 1.0)))
    factor_spin.setObjectName("point_list_scale_factor")
    factor_spin.setToolTip("Multiplicative factor applied to the selected signal channel.")
    factor_spin.valueChanged.connect(
        lambda value: self._set_point_list_scale(dataset, group, "factor", value)
    )
    grid.addWidget(factor_spin, 1, 1)
    grid.addWidget(QtWidgets.QLabel("Units"), 2, 0)
    units_edit = QtWidgets.QLineEdit(str(scale.get("units", "")))
    constrain_input_width(units_edit, COMPACT_SHORT_TEXT_FIELD_WIDTH)
    units_edit.setToolTip("Display and export units for the scaled signal channel.")
    units_edit.editingFinished.connect(
        lambda editor=units_edit: self._set_point_list_scale(dataset, group, "units", editor.text())
    )
    grid.addWidget(units_edit, 2, 1)
    return box


def _point_list_susceptibility_box(self, dataset, group, config) -> Any:
    from PySide6 import QtWidgets

    susc = config["susceptibility"]
    box = QtWidgets.QGroupBox("Susceptibility (moment / field)")
    grid = QtWidgets.QGridLayout(box)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.setColumnStretch(1, 1)
    enable = QtWidgets.QCheckBox("Plot and fit susceptibility")
    enable.setObjectName("point_list_susceptibility_enabled")
    enable.setChecked(bool(susc.get("enabled", False)))
    enable.setToolTip(
        "Use susceptibility rather than magnetic moment for viewing and fitting. "
        "The selected moment is divided by the selected applied-field column."
    )
    enable.toggled.connect(lambda checked: self._set_point_list_susceptibility(dataset, group, "enabled", checked))
    grid.addWidget(enable, 0, 0, 1, 2)
    grid.addWidget(QtWidgets.QLabel("Moment"), 1, 0)
    moment_combo = QtWidgets.QComboBox()
    moment_combo.addItems([str(c["label"]) for c in config.get("channels", [])])
    moment_combo.setCurrentText(str(susc.get("moment", "")))
    moment_combo.setToolTip("Signal channel containing the measured moment.")
    moment_combo.currentTextChanged.connect(
        lambda text: self._set_point_list_susceptibility(dataset, group, "moment", text)
    )
    grid.addWidget(moment_combo, 1, 1)
    grid.addWidget(QtWidgets.QLabel("Field"), 2, 0)
    field_combo = QtWidgets.QComboBox()
    field_combo.addItems(list(dataset.data.column_names))
    field_combo.setCurrentText(str(susc.get("field", "")))
    field_combo.setToolTip("Column containing the applied field used to compute susceptibility.")
    field_combo.currentTextChanged.connect(
        lambda text: self._set_point_list_susceptibility(dataset, group, "field", text)
    )
    grid.addWidget(field_combo, 2, 1)
    grid.addWidget(QtWidgets.QLabel("Moment input unit"), 3, 0)
    moment_unit_combo = QtWidgets.QComboBox()
    moment_unit_combo.setObjectName("point_list_moment_input_unit")
    moment_unit_combo.addItem("emu", "emu")
    moment_unit_combo.addItem("A m^2", "A m^2")
    current_moment_unit = str(susc.get("moment_unit") or "emu")
    moment_unit_index = moment_unit_combo.findData(current_moment_unit)
    moment_unit_combo.setCurrentIndex(max(moment_unit_index, 0))
    moment_unit_combo.setToolTip(
        "Unit in which the selected raw moment column is recorded. MPMS moment data are normally emu."
    )
    moment_unit_combo.currentIndexChanged.connect(
        lambda _index, combo=moment_unit_combo: self._set_point_list_susceptibility(
            dataset, group, "moment_unit", combo.currentData()
        )
    )
    grid.addWidget(moment_unit_combo, 3, 1)
    grid.addWidget(QtWidgets.QLabel("Field input unit"), 4, 0)
    field_unit_combo = QtWidgets.QComboBox()
    field_unit_combo.setObjectName("point_list_field_input_unit")
    field_unit_combo.addItem("Oe", "Oe")
    field_unit_combo.addItem("T", "T")
    field_unit_combo.addItem("A/m", "A/m")
    current_field_unit = str(susc.get("field_unit") or "Oe")
    field_unit_index = field_unit_combo.findData(current_field_unit)
    field_unit_combo.setCurrentIndex(max(field_unit_index, 0))
    field_unit_combo.setToolTip(
        "Unit in which the selected raw applied-field column is recorded. MPMS field data are normally Oe."
    )
    field_unit_combo.currentIndexChanged.connect(
        lambda _index, combo=field_unit_combo: self._set_point_list_susceptibility(
            dataset, group, "field_unit", combo.currentData()
        )
    )
    grid.addWidget(field_unit_combo, 4, 1)
    grid.addWidget(QtWidgets.QLabel("Absolute susceptibility units"), 5, 0)
    unit_combo = QtWidgets.QComboBox()
    unit_combo.setObjectName("point_list_susceptibility_output_unit")
    unit_combo.addItem("CGS molar (emu/(mol Oe))", "cm^3/mol")
    unit_combo.addItem("SI molar (m^3/mol)", "m^3/mol")
    unit_index = unit_combo.findData(str(susc.get("output_unit", "cm^3/mol")))
    unit_combo.setCurrentIndex(max(unit_index, 0))
    unit_combo.setToolTip(
        "Output convention used when absolute units are enabled. CGS molar "
        "susceptibility is numerically cm^3/mol; conversion to rationalized "
        "SI m^3/mol includes the 4 pi factor."
    )
    unit_combo.currentIndexChanged.connect(
        lambda _index, combo=unit_combo: self._set_point_list_susceptibility(
            dataset, group, "output_unit", combo.currentData()
        )
    )
    grid.addWidget(unit_combo, 5, 1)
    _compact_point_list_form(box, QtWidgets)
    return box


def _point_list_wavelength_box(self, dataset, group, config) -> Any:
    from PySide6 import QtWidgets

    wavelength = config["wavelength"]
    box = QtWidgets.QGroupBox("Neutron wavelength -> q")
    grid = QtWidgets.QGridLayout(box)
    grid.setContentsMargins(8, 6, 8, 6)
    grid.addWidget(QtWidgets.QLabel("2theta column"), 0, 0)
    two_theta_combo = QtWidgets.QComboBox()
    two_theta_combo.addItems(list(dataset.data.column_names))
    two_theta_combo.setCurrentText(str(wavelength.get("two_theta", "")))
    two_theta_combo.setToolTip("Column containing scattering angle 2theta, used with wavelength to compute |Q|.")
    two_theta_combo.currentTextChanged.connect(
        lambda text: self._set_point_list_wavelength(dataset, group, "two_theta", text)
    )
    grid.addWidget(two_theta_combo, 0, 1)
    grid.addWidget(QtWidgets.QLabel("Wavelength (A)"), 1, 0)
    wavelength_spin = QtWidgets.QDoubleSpinBox()
    wavelength_spin.setObjectName("point_list_wavelength")
    wavelength_spin.setDecimals(5)
    wavelength_spin.setRange(0.0, 100.0)
    wavelength_spin.setValue(float(wavelength.get("value", 0.0)))
    wavelength_spin.setToolTip("Neutron wavelength in Angstroms used with 2theta to compute |Q|.")
    wavelength_spin.valueChanged.connect(
        lambda value: self._set_point_list_wavelength(dataset, group, "value", value)
    )
    grid.addWidget(wavelength_spin, 1, 1)
    return box


def _dataset_type_group_box(
    self,
    dataset: DatasetEntry,
    group: DataGroup | None,
    lines: list[str],
) -> Any:
    from PySide6 import QtCore, QtWidgets

    group_box = QtWidgets.QGroupBox("Dataset")
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)
    label = QtWidgets.QLabel("\n".join(lines) if lines else "-")
    label.setWordWrap(True)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(label)

    type_row = QtWidgets.QHBoxLayout()
    type_row.addWidget(QtWidgets.QLabel("Data type"))
    type_combo = QtWidgets.QComboBox()
    type_combo.setObjectName("dataset_data_type")
    type_combo.setToolTip(
        "Interpret this dataset as a specific experimental data type. "
        "This controls metadata handling, viewer readouts, and fitting compatibility."
    )
    for name, type_label in available_data_types():
        type_combo.addItem(type_label, name)
    current = dataset.data_type or DEFAULT_DATA_TYPE
    index = type_combo.findData(current)
    type_combo.setCurrentIndex(max(index, 0))
    type_combo.currentIndexChanged.connect(
        lambda _index, combo=type_combo: self._set_selected_data_type(dataset, group, combo.currentData())
    )
    type_row.addWidget(type_combo, 1)
    layout.addLayout(type_row)
    return group_box


def _dataset_axes_group_box(
    self,
    dataset: DatasetEntry,
    group: DataGroup | None,
    lines: list[str],
) -> Any:
    from PySide6 import QtCore, QtWidgets

    from .project_rebin_panels import add_rebin_assignment_items, add_rebin_mode_items

    group_box = QtWidgets.QGroupBox("Axes")
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)
    label = QtWidgets.QLabel("\n".join(lines) if lines else "-")
    label.setWordWrap(True)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(label)

    if not _dataset_can_rebin(dataset):
        return group_box

    rebin_box = QtWidgets.QGroupBox("Rebin")
    rebin_layout = QtWidgets.QVBoxLayout(rebin_box)
    rebin_layout.setContentsMargins(10, 8, 10, 8)

    binnings = dataset_rebin_binnings(dataset)
    selected_binning = self._selected_dataset_binning(dataset)
    config = selected_binning["config"]
    binning_row = QtWidgets.QHBoxLayout()
    binning_row.addWidget(QtWidgets.QLabel("Binning"))
    binning_combo = QtWidgets.QComboBox()
    binning_combo.setObjectName("dataset_rebin_binning")
    for item in binnings:
        binning_combo.addItem(
            f"{item['name']}{' (fit)' if item['fit'] else ''}", item["id"]
        )
    binning_combo.setCurrentIndex(max(binning_combo.findData(selected_binning["id"]), 0))
    binning_combo.setToolTip("Choose the named binning configuration edited below.")
    binning_combo.currentIndexChanged.connect(
        lambda _index, combo=binning_combo: self._select_dataset_binning(
            dataset, str(combo.currentData())
        )
    )
    add_binning = QtWidgets.QPushButton("Add…")
    add_binning.setObjectName("dataset_rebin_add_binning")
    add_binning.setToolTip("Create a visualization binning from the fit binning.")
    add_binning.clicked.connect(lambda: self._add_dataset_binning(dataset, duplicate=False))
    duplicate_binning = QtWidgets.QPushButton("Duplicate")
    duplicate_binning.setObjectName("dataset_rebin_duplicate_binning")
    duplicate_binning.setToolTip("Duplicate the selected binning.")
    duplicate_binning.clicked.connect(lambda: self._add_dataset_binning(dataset, duplicate=True))
    rename_binning = QtWidgets.QPushButton("Rename…")
    rename_binning.setObjectName("dataset_rebin_rename_binning")
    rename_binning.setToolTip("Rename the selected binning.")
    rename_binning.clicked.connect(lambda: self._rename_dataset_binning(dataset))
    remove_binning = QtWidgets.QPushButton("Remove")
    remove_binning.setObjectName("dataset_rebin_remove_binning")
    remove_binning.setEnabled(not selected_binning["fit"])
    remove_binning.setToolTip("Remove the selected visualization binning and its cache.")
    remove_binning.clicked.connect(lambda: self._remove_dataset_binning(dataset))
    fit_binning = QtWidgets.QCheckBox("Use for fitting")
    fit_binning.setObjectName("dataset_rebin_fit_binning")
    fit_binning.setChecked(bool(selected_binning["fit"]))
    fit_binning.setEnabled(not selected_binning["fit"])
    fit_binning.setToolTip("Make this the sole binning used during fit iterations.")
    fit_binning.toggled.connect(
        lambda checked: checked and self._make_dataset_fit_binning(dataset)
    )
    binning_row.addWidget(binning_combo, 1)
    binning_row.addWidget(add_binning)
    binning_row.addWidget(duplicate_binning)
    binning_row.addWidget(rename_binning)
    binning_row.addWidget(remove_binning)
    binning_row.addWidget(fit_binning)
    rebin_layout.addLayout(binning_row)
    enable_check = QtWidgets.QCheckBox("Use rebinned data")
    enable_check.setObjectName("dataset_rebin_enabled")
    enable_check.setChecked(bool(config.get("enabled", False)))
    enable_check.setToolTip(
        "Use this named rebin for viewing and fitting."
        if selected_binning["fit"]
        else "Make this visualization-only rebin available in the data viewer."
    )
    enable_check.toggled.connect(lambda checked: self._set_dataset_rebin_enabled(dataset, group, checked))
    settings_row = QtWidgets.QHBoxLayout()
    settings_row.addWidget(enable_check)
    settings_row.addStretch(1)
    copy_settings_button = QtWidgets.QPushButton("Copy settings")
    copy_settings_button.setObjectName("dataset_rebin_copy_settings")
    copy_settings_button.setToolTip(
        "Copy every setting in this rebin panel to the system clipboard. The copied JSON can be "
        "pasted into a compatible dataset or dataset-group rebin panel."
    )
    copy_settings_button.clicked.connect(
        lambda _checked=False: self._copy_rebin_settings(config)
    )
    paste_settings_button = QtWidgets.QPushButton("Paste settings")
    paste_settings_button.setObjectName("dataset_rebin_paste_settings")
    paste_settings_button.setToolTip(
        "Replace every setting in this rebin panel with compatible nfit rebin settings from the "
        "system clipboard."
    )
    paste_settings_button.clicked.connect(
        lambda _checked=False: self._paste_dataset_rebin_settings(dataset, group)
    )
    settings_row.addWidget(copy_settings_button)
    settings_row.addWidget(paste_settings_button)
    rebin_layout.addLayout(settings_row)

    controls = QtWidgets.QTabWidget()
    controls.setObjectName("dataset_rebin_controls")
    controls.setToolTip(
        "Switch between rebin settings and an expandable summary of the output grid."
    )
    controls.setVisible(bool(config.get("enabled", False)))
    settings_tab = QtWidgets.QWidget()
    settings_tab.setObjectName("dataset_rebin_settings_tab")
    controls_layout = QtWidgets.QGridLayout(settings_tab)
    controls_layout.setContentsMargins(0, 0, 0, 0)
    show_momentum_matrix = len(config.get("axes", [])) == 4 and (
        isinstance(dataset.data, (MDHistoData, PointData4D))
        or dataset.data_type.startswith("single_crystal")
    )
    header_row = 0
    if show_momentum_matrix:
        matrix_label = QtWidgets.QLabel("Momentum coordinates")
        matrix_edit = QtWidgets.QLineEdit(
            _parameter_to_text(_momentum_rebin_matrix(config["axes"]))
        )
        matrix_edit.setObjectName("dataset_rebin_momentum_matrix")
        matrix_tooltip = (
            "Complete 3x3 output momentum-coordinate matrix. Each row contains [H, K, L] "
            "coefficients, and all three rows must be linearly independent. DeltaE remains a "
            "separate fixed coordinate and cannot be mixed with momentum."
        )
        matrix_label.setToolTip(matrix_tooltip)
        matrix_edit.setToolTip(matrix_tooltip)
        matrix_edit.editingFinished.connect(
            lambda editor=matrix_edit: self._set_dataset_rebin_momentum_matrix(
                dataset, group, editor.text()
            )
        )
        controls_layout.addWidget(matrix_label, 0, 0)
        controls_layout.addWidget(matrix_edit, 0, 1, 1, 6)
        header_row = 1
    show_vectors = False
    headers = ["Axis"]
    if show_vectors:
        headers.append("Momentum row [H,K,L]")
    headers.extend(
        ["Min center", "Max center", "Value", "Edges", "Grid", "Mode"]
    )
    last_column = len(headers) - 1
    controls_layout.setColumnStretch(last_column, 1)
    for column, text in enumerate(headers):
        header = QtWidgets.QLabel(text)
        header.setStyleSheet("font-weight: 600")
        controls_layout.addWidget(header, header_row, column)
    for row, axis_config in enumerate(config.get("axes", []), start=header_row + 1):
        axis_index = row - header_row - 1
        axis_config = _sanitize_rebin_axis_config(axis_config)
        axis_mode = _rebin_axis_mode(config, axis_config)
        resolution_key = (
            "num_bins"
            if axis_mode == "bins"
            else "tolerance"
            if axis_mode == "tolerance"
            else "step_size"
        )
        axis_label = QtWidgets.QLabel(
            str(axis_config.get("name", f"Axis {axis_index + 1}"))
        )
        axis_label.setObjectName(f"dataset_rebin_axis_variable_{axis_index}")
        axis_label.setToolTip(
            "Scalar variable for this output coordinate. The plotted axis name is generated from this variable "
            "and the Coord axis vector."
        )
        lower_edit = QtWidgets.QLineEdit(
            ""
            if _rebin_axis_bound_is_auto(axis_config, "lower")
            else _format_number(axis_config["lower"])
        )
        upper_edit = QtWidgets.QLineEdit(
            ""
            if _rebin_axis_bound_is_auto(axis_config, "upper")
            else _format_number(axis_config["upper"])
        )
        lower_edit.setObjectName(f"dataset_rebin_axis_lower_{axis_index}")
        upper_edit.setObjectName(f"dataset_rebin_axis_upper_{axis_index}")
        resolution_text = (
            ""
            if axis_mode in {"discrete", "edges"}
            else
            str(int(axis_config["num_bins"]))
            if resolution_key == "num_bins"
            else _format_number(axis_config[resolution_key])
        )
        resolution_edit = QtWidgets.QLineEdit(resolution_text)
        resolution_edit.setObjectName(f"dataset_rebin_resolution_value_{axis_index}")
        edges_edit = QtWidgets.QLineEdit(
            _parameter_to_text(axis_config.get("bin_edges", ""))
        )
        edges_edit.setObjectName(f"dataset_rebin_axis_edges_{axis_index}")
        edges_edit.setPlaceholderText("uniform")
        lower_edit.setPlaceholderText("auto")
        upper_edit.setPlaceholderText("auto")
        lower_edit.setEnabled(axis_mode in {"step", "bins"})
        upper_edit.setEnabled(axis_mode in {"step", "bins"})
        resolution_edit.setEnabled(axis_mode not in {"discrete", "edges"})
        edges_edit.setEnabled(axis_mode == "edges")
        lower_edit.setToolTip(
            "First bin center (lower edge for one-bin integration). Leave blank to cover the projected data and align uniform bins so zero is a bin center."
        )
        upper_edit.setToolTip(
            "Last bin center (upper edge for one-bin integration). Leave blank to cover the projected data and align uniform bins so zero is a bin center."
        )
        resolution_edit.setToolTip(
            "Approximate bin step size. Editing it recalculates the bin count."
            if resolution_key == "step_size"
            else "Number of bins. Editing it recalculates the displayed step size."
            if resolution_key == "num_bins"
            else "Maximum distance from a point to its automatically determined bin center."
        )
        edges_edit.setToolTip(
            "Optional strictly increasing edge list for only this axis, for example "
            "[-2, -1, 0, 0.5, 2]. Leave blank to use the uniform Resolution setting."
        )
        for editor in (lower_edit, upper_edit, resolution_edit, edges_edit):
            editor.setMinimumWidth(72)
        lower_edit.editingFinished.connect(
            lambda index=axis_index, editor=lower_edit: self._set_dataset_rebin_axis_value(dataset, group, index, "lower", editor.text())
        )
        upper_edit.editingFinished.connect(
            lambda index=axis_index, editor=upper_edit: self._set_dataset_rebin_axis_value(dataset, group, index, "upper", editor.text())
        )
        resolution_edit.editingFinished.connect(
            lambda index=axis_index, key=resolution_key, editor=resolution_edit: self._set_dataset_rebin_axis_value(
                dataset, group, index, key, editor.text()
            )
        )
        edges_edit.editingFinished.connect(
            lambda index=axis_index, editor=edges_edit: self._set_dataset_rebin_axis_edges(
                dataset, group, index, editor.text()
            )
        )
        controls_layout.addWidget(axis_label, row, 0)
        column = 1
        if show_vectors:
            variable = str(
                axis_config.get("variable", ("H", "K", "L", "E")[row - 1])
            ).upper()
            if variable == "E":
                energy_label = QtWidgets.QLabel("fixed E")
                energy_label.setToolTip(
                    "Energy remains a separate DeltaE coordinate and cannot be mixed with H, K, or L."
                )
                controls_layout.addWidget(energy_label, row, column)
            else:
                vector_edit = QtWidgets.QLineEdit(
                    _momentum_rebin_vector_text(axis_config, row - 1)
                )
                vector_edit.setObjectName(f"dataset_rebin_axis_vector_{row - 1}")
                vector_edit.setMinimumWidth(110)
                vector_edit.setToolTip(
                    "One row of the 3x3 output momentum-coordinate matrix, expressed as [H, K, L] coefficients. "
                    "Its three rows must be linearly independent. Changing a row regenerates the plotted axis name "
                    "and expands the rebin bounds to contain the source data while preserving approximate bin "
                    "resolution. Energy remains a separate fixed coordinate."
                )
                vector_edit.editingFinished.connect(
                    lambda row=row - 1, editor=vector_edit: self._set_dataset_rebin_axis_vector(dataset, group, row, editor.text())
                )
                controls_layout.addWidget(vector_edit, row, column)
            column += 1
        controls_layout.addWidget(lower_edit, row, column)
        controls_layout.addWidget(upper_edit, row, column + 1)
        controls_layout.addWidget(resolution_edit, row, column + 2)
        controls_layout.addWidget(edges_edit, row, column + 3)
        mode_combo = QtWidgets.QComboBox()
        mode_combo.setObjectName(f"dataset_rebin_axis_mode_{axis_index}")
        add_rebin_mode_items(mode_combo)
        mode_combo.setCurrentIndex(max(mode_combo.findData(axis_mode), 0))
        mode_combo.setToolTip(
            "Choose how this axis's bin centers or edges are constructed. Point assignment is "
            "controlled separately; Discrete and Tolerance grids require discrete assignment."
        )
        mode_combo.currentIndexChanged.connect(
            lambda _index, index=axis_index, combo=mode_combo: self._set_dataset_rebin_axis_mode(
                dataset, group, index, str(combo.currentData() or "step")
            )
        )
        controls_layout.addWidget(mode_combo, row, column + 4)
        assignment_combo = QtWidgets.QComboBox()
        assignment_combo.setObjectName(f"dataset_rebin_axis_assignment_{axis_index}")
        add_rebin_assignment_items(assignment_combo)
        assignment_combo.setCurrentIndex(
            max(assignment_combo.findData(_rebin_axis_fractional(config, axis_config)), 0)
        )
        assignment_combo.setEnabled(axis_mode not in {"discrete", "tolerance"})
        assignment_combo.setToolTip(
            "Fractional distributes a point between neighboring bins on this axis. Discrete "
            "assigns it wholly to one bin. Tolerance always uses discrete assignment."
        )
        assignment_combo.currentIndexChanged.connect(
            lambda _index, index=axis_index, combo=assignment_combo: self._set_dataset_rebin_axis_fractional(
                dataset, group, index, bool(combo.currentData())
            )
        )
        controls_layout.addWidget(assignment_combo, row, column + 5)

    option_row = QtWidgets.QHBoxLayout()
    auto_check = QtWidgets.QCheckBox("Automatic rebinning")
    auto_check.setObjectName("dataset_rebin_auto")
    auto_check.setChecked(bool(config.get("auto_rebin", True)))
    auto_check.setToolTip(
        "Automatically recompute the rebinned data when rebin settings change. It turns off when an edit exceeds "
        "5,000,000 estimated point contributions or 2,000,000 output bins, and can then be manually re-enabled. "
        "With automatic rebinning off, edits are marked pending until Rebin now is pressed or an operation such as fitting, opening the data viewer, "
        "or saving a rebinned dataset requires an up-to-date rebin."
    )
    auto_check.toggled.connect(lambda checked: self._set_dataset_rebin_auto(dataset, group, checked))
    mean_label = QtWidgets.QLabel("Mean")
    mean_combo = QtWidgets.QComboBox()
    mean_combo.setObjectName("dataset_rebin_mean_weighting")
    mean_combo.setToolTip(
        "Choose how multiple source points in a rebinned bin are averaged. "
        "A physical normalization denominator, when present, always contributes to the data weight. Inverse variance additionally uses 1/sigma^2; uniform does not."
    )
    mean_combo.addItem("Inverse variance", "inverse_variance")
    mean_combo.addItem("Uniform", "uniform")
    mean_index = mean_combo.findData(_rebin_mean_weighting(config))
    mean_combo.setCurrentIndex(max(mean_index, 0))
    mean_combo.currentIndexChanged.connect(
        lambda _index, combo=mean_combo: self._set_dataset_rebin_mean_weighting(
            dataset,
            group,
            str(combo.currentData() or "uniform"),
        )
    )
    symmetry = symmetry_spec_from_config(config.get("symmetry"))
    symmetry_check = QtWidgets.QCheckBox("Apply symmetry")
    symmetry_check.setObjectName("dataset_rebin_symmetry_enabled")
    symmetry_check.setChecked(symmetry.enabled)
    symmetry_check.setToolTip(
        "Transform HKL coordinates by a crystallographic point group before binning. "
        "Energy is unchanged and space-group translations are ignored."
    )
    symmetry_check.toggled.connect(
        lambda checked: self._set_dataset_rebin_symmetry_enabled(dataset, group, checked)
    )
    symmetry_mode = QtWidgets.QComboBox()
    symmetry_mode.setObjectName("dataset_rebin_symmetry_mode")
    symmetry_mode.addItem("Space group", "space_group")
    symmetry_mode.addItem("Point group", "point_group")
    symmetry_mode.addItem("Operations", "operations")
    symmetry_mode.addItem("Generators", "generators")
    displayed_symmetry_mode = (
        symmetry.mode
        if symmetry.mode != "none"
        else str(config.get("symmetry", {}).get("last_mode", "space_group"))
    )
    symmetry_mode.setCurrentIndex(
        max(symmetry_mode.findData(displayed_symmetry_mode), 0)
    )
    symmetry_mode.setToolTip(
        "Choose a space group, point group, semicolon-separated Jones-faithful operations, or geometric generators."
    )
    symmetry_mode.currentIndexChanged.connect(
        lambda _index, combo=symmetry_mode: self._set_dataset_rebin_symmetry_mode(dataset, group, str(combo.currentData()))
    )
    symmetry_expression = QtWidgets.QLineEdit(symmetry.expression)
    symmetry_expression.setObjectName("dataset_rebin_symmetry_expression")
    constrain_input_width(
        symmetry_expression,
        rebin_symmetry_expression_width(displayed_symmetry_mode),
    )
    symmetry_expression.setPlaceholderText("P -1")
    symmetry_expression.setToolTip(
        "Examples: P -1; -1; x,y,z;-x,-y,-z; rotate(order=3, axis=[1,1,1]); mirror(plane=(0,0,1))."
    )
    symmetry_expression.editingFinished.connect(
        lambda editor=symmetry_expression: self._set_dataset_rebin_symmetry_expression(dataset, group, editor.text())
    )
    symmetry_preview = QtWidgets.QLabel(self._dataset_rebin_symmetry_preview(dataset, group))
    symmetry_preview.setObjectName("dataset_rebin_symmetry_preview")
    symmetry_preview.setToolTip("Resolved reciprocal-HKL operation count and any syntax error.")
    create_button = QtWidgets.QPushButton("Create dataset from rebin")
    create_button.setObjectName("dataset_rebin_create")
    create_button.setEnabled(_dataset_can_rebin(dataset))
    create_button.setToolTip("Materialize the current rebinned data as a new independent dataset.")
    create_button.clicked.connect(self.materialize_rebin_for_selection)
    rebin_now_button = QtWidgets.QPushButton("Rebin now")
    rebin_now_button.setObjectName("dataset_rebin_now")
    rebin_now_button.setEnabled(_dataset_can_rebin(dataset))
    rebin_now_button.setToolTip(
        "Compute the current rebin immediately and update the cached viewer/fit data. Use this when automatic rebinning is off."
    )
    rebin_now_button.clicked.connect(
        lambda _checked=False: self.rebin_dataset_now(dataset, group)
    )
    rebin_all_button = QtWidgets.QPushButton("Rebin all now")
    rebin_all_button.setObjectName("dataset_rebin_all_now")
    rebin_all_button.setEnabled(_dataset_can_rebin(dataset))
    rebin_all_button.setVisible(len(binnings) > 1)
    rebin_all_button.setToolTip(
        "Compute every enabled named binning for this dataset and update open data viewers."
    )
    rebin_all_button.clicked.connect(
        lambda _checked=False: self.rebin_all_dataset_binnings_now(dataset, group)
    )
    save_rebin_button = QtWidgets.QPushButton("Save rebin to disk")
    save_rebin_button.setObjectName("dataset_rebin_save")
    save_rebin_button.setEnabled(_dataset_can_rebin(dataset))
    save_rebin_button.setToolTip("Export the current rebinned dataset directly to a NumPy archive.")
    save_rebin_button.clicked.connect(self.save_rebin_for_selection)
    coverage_label = QtWidgets.QLabel("Minimum coverage")
    coverage_edit = QtWidgets.QLineEdit(_format_number(_rebin_minimum_coverage(config)))
    coverage_edit.setObjectName("dataset_rebin_minimum_coverage")
    coverage_edit.setMaximumWidth(70)
    coverage_tooltip = (
        "Mask rebinned output bins whose measured geometric support is below this fraction "
        "of the requested bin volume. Enter a value from 0 to 1."
    )
    coverage_label.setToolTip(coverage_tooltip)
    coverage_edit.setToolTip(coverage_tooltip)
    coverage_edit.editingFinished.connect(
        lambda editor=coverage_edit: self._set_dataset_rebin_minimum_coverage(
            dataset, group, editor
        )
    )
    samples_label = QtWidgets.QLabel("Minimum samples")
    samples_edit = QtWidgets.QLineEdit(
        _format_number(_rebin_minimum_samples(config))
    )
    samples_edit.setObjectName("dataset_rebin_minimum_samples")
    samples_edit.setMaximumWidth(70)
    samples_tooltip = (
        "Mask bins receiving less than this effective number of source samples. "
        "Fractional binning sums fractional sample contributions."
    )
    samples_label.setToolTip(samples_tooltip)
    samples_edit.setToolTip(samples_tooltip)
    samples_edit.editingFinished.connect(
        lambda editor=samples_edit: self._set_dataset_rebin_minimum_samples(
            dataset, group, editor
        )
    )
    option_row.addWidget(auto_check)
    option_row.addWidget(mean_label)
    option_row.addWidget(mean_combo)
    option_row.addStretch(1)
    footer_row = header_row + len(config.get("axes", [])) + 1
    controls_layout.addLayout(option_row, footer_row, 0, 1, last_column + 1)
    quality_row = QtWidgets.QHBoxLayout()
    quality_row.addWidget(coverage_label)
    quality_row.addWidget(coverage_edit)
    quality_row.addSpacing(12)
    quality_row.addWidget(samples_label)
    quality_row.addWidget(samples_edit)
    quality_row.addSpacing(12)
    self._add_rebin_performance_controls(quality_row, dataset=dataset, group=group)
    quality_row.addStretch(1)
    controls_layout.addLayout(quality_row, footer_row + 1, 0, 1, last_column + 1)
    symmetry_row = QtWidgets.QHBoxLayout()
    symmetry_row.addWidget(symmetry_check)
    symmetry_row.addWidget(symmetry_mode)
    symmetry_row.addWidget(symmetry_expression, 1)
    symmetry_row.addWidget(symmetry_preview)
    controls_layout.addLayout(symmetry_row, footer_row + 2, 0, 1, last_column + 1)
    from .project_rebin_panels import rebin_memory_estimate_label

    cached_rebin_data = _peek_cached_dataset_view(
        dataset,
        extra_masks=(effective_dataset_masks(group, dataset) if group is not None else None),
        rebin_config=config,
        cache_id=None if selected_binning["fit"] else selected_binning["id"],
        resident_only=True,
    )
    compressed_disk_bytes = (
        _saved_binning_compressed_size(
            self.project,
            kind="dataset",
            group=group,
            target=dataset,
            binning_id=selected_binning["id"],
            config=config,
        )
        if group is not None
        else None
    )
    memory_label = rebin_memory_estimate_label(
        config,
        data=cached_rebin_data,
        object_prefix="dataset_rebin",
        compressed_disk_bytes=compressed_disk_bytes,
    )
    controls_layout.addWidget(memory_label, footer_row + 3, 0, 1, last_column + 1)
    status_label = QtWidgets.QLabel(_dataset_rebin_status_text(dataset, config))
    status_label.setObjectName("dataset_rebin_status")
    status_label.setWordWrap(True)
    status_label.setToolTip(
        "Shows whether the cached rebinned data is current. Pending manual rebinning will be forced automatically for fit, view, and export operations."
    )
    controls_layout.addWidget(status_label, footer_row + 4, 0, 1, last_column + 1)
    action_row = QtWidgets.QHBoxLayout()
    action_row.addWidget(rebin_now_button)
    action_row.addWidget(rebin_all_button)
    action_row.addWidget(create_button)
    action_row.addWidget(save_rebin_button)
    action_row.addStretch(1)
    controls_layout.addLayout(action_row, footer_row + 5, 0, 1, last_column + 1)
    controls.addTab(settings_tab, "Rebin settings")
    from .project_rebin_panels import rebin_bin_information_widget

    controls.addTab(
        rebin_bin_information_widget(
            config,
            data=cached_rebin_data,
            object_prefix="dataset_rebin",
            compressed_disk_bytes=compressed_disk_bytes,
        ),
        "Bin information",
    )
    for button in controls.findChildren(QtWidgets.QToolButton):
        button.setToolTip("Scroll rebin tabs when the tab bar is too narrow.")
    rebin_layout.addWidget(controls)
    layout.addWidget(rebin_box)
    return group_box


def _add_rebin_performance_controls(self, row, *, dataset=None, group=None, composite=False):
    from PySide6 import QtWidgets

    from .performance_gui import BenchmarkDialog

    prefix = "group_composite" if composite else "dataset_rebin"
    benchmark = QtWidgets.QPushButton("Benchmark this rebin…")
    benchmark.setObjectName(prefix + "_benchmark")
    benchmark.setToolTip("Benchmark the full current rebin using the CPU and RAM limits in Preferences.")
    if composite:
        target = dict(group_name=_composite_root(group).name,
                      node_id=group.node.id if isinstance(group, _CompositeScope) else None)
    else:
        target = dict(dataset_id=dataset.id)
    benchmark.clicked.connect(lambda: BenchmarkDialog(
        self.window, project=self.project, target=target,
    ).exec())
    row.addWidget(benchmark)


def _details_group_box(self, title: str, lines: list[str]) -> Any:
    from PySide6 import QtCore, QtWidgets

    group_box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)
    text = "\n".join(lines) if lines else "-"
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(label)
    return group_box


def _dataset_metadata_group_box(self, dataset: DatasetEntry) -> Any:
    metadata = dict(_dataset_metadata_mapping(dataset))
    if dataset.parameters:
        metadata["Parameters"] = dict(dataset.parameters)
    return self._metadata_tree_group_box(
        "Metadata",
        metadata,
        object_name="dataset_metadata_tree",
        empty_text="No additional metadata.",
    )


def _analysis_table_group_box(self, data: PointListData) -> Any:
    from PySide6 import QtCore, QtGui, QtWidgets

    box = QtWidgets.QGroupBox("Results table")
    layout = QtWidgets.QVBoxLayout(box)
    table = QtWidgets.QTableWidget(data.size, len(data.column_names))
    table.setObjectName("analysis_output_table")
    table.setToolTip("Persisted analysis values. Rejected Bragg reflections are shown in red.")
    table.setHorizontalHeaderLabels(data.column_names)
    accepted = data.column("Accepted") if "Accepted" in data.columns else np.ones(data.size)
    for row in range(data.size):
        for column, name in enumerate(data.column_names):
            value = float(data.column(name)[row])
            item = QtWidgets.QTableWidgetItem(_format_number(value) if np.isfinite(value) else "-")
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            if not bool(accepted[row]):
                item.setForeground(QtGui.QColor("#d94b45"))
            table.setItem(row, column, item)
    table.resizeColumnsToContents()
    table.setMinimumHeight(180)
    table.setMaximumHeight(420)
    layout.addWidget(table)
    return box


def _dataset_signal_semantics_group_box(
    self, dataset: DatasetEntry, group: DataGroup | None
) -> Any:
    """Build the explicit signal-value convention control for histogram data."""

    from PySide6 import QtWidgets

    box = QtWidgets.QGroupBox("Signal convention")
    layout = QtWidgets.QFormLayout(box)
    layout.setContentsMargins(10, 8, 10, 8)
    combo = QtWidgets.QComboBox()
    combo.setObjectName("dataset_signal_semantics")
    combo.addItem("Density-valued signal", "density")
    combo.addItem("Bin-integral signal", "bin_integral")
    combo.addItem("Unspecified", "unknown")
    data = dataset.data
    semantics = (
        signal_semantics(data)
        if isinstance(data, MDHistoData)
        else "density"
    )
    combo.setCurrentIndex(max(combo.findData(semantics), 0))
    combo.setEnabled(isinstance(data, MDHistoData))
    combo.setToolTip(
        "Choose how each histogram signal value is interpreted. Density is the default for normalized, "
        "variance-weighted rebins and is weighted by reciprocal-space bin volume during integration. "
        "Use Bin-integral only when each stored value is already the total for its bin."
    )
    combo.currentIndexChanged.connect(
        lambda _index, selector=combo: self._set_dataset_signal_semantics(
            dataset, group, str(selector.currentData())
        )
    )
    layout.addRow("Signal values", combo)
    return box


def _dataset_spectral_channels_group_box(
    self,
    dataset: DatasetEntry,
    group: DataGroup | None,
    *,
    config_override=None,
    setting_changed=None,
    status_override=None,
) -> Any:
    """Build INS representation, unit, and correction controls."""

    from PySide6 import QtWidgets

    from .form_factors import available_ions
    from .project_coordinates import _parse_parameter_text
    from .project_rebinning import _parameter_to_text

    config = config_override if config_override is not None else self._dataset_spectral_channel_config(dataset)
    set_setting = setting_changed or self._set_dataset_spectral_channel_setting
    box = QtWidgets.QGroupBox("INS representations")
    form = QtWidgets.QFormLayout(box)
    form.setContentsMargins(10, 8, 10, 8)
    form.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    )

    enabled = QtWidgets.QCheckBox("Create paired cross-section and χ″ channels")
    enabled.setObjectName("ins_channels_enabled")
    enabled.setChecked(bool(config["enabled"]))
    enabled.setToolTip(
        "Keep the imported signal and create named scattering-cross-section "
        "and dynamical-susceptibility channels. A temperature is required "
        "to convert between them."
    )
    enabled.toggled.connect(
        lambda checked: set_setting(
            dataset, group, "enabled", bool(checked)
        )
    )
    form.addRow(enabled)

    source = QtWidgets.QComboBox()
    source.setObjectName("ins_source_representation")
    source.addItem("Scattering cross section / intensity", "cross_section")
    source.addItem("Dynamical susceptibility χ″", "chi_double_prime")
    source.setCurrentIndex(max(source.findData(config["source_representation"]), 0))
    source.setToolTip(
        "Physical meaning of the imported signal. Counts and arbitrary "
        "intensity are treated as cross-section-shaped data with an unknown scale."
    )
    source.currentIndexChanged.connect(
        lambda _index, combo=source: set_setting(
            dataset, group, "source_representation", str(combo.currentData())
        )
    )
    form.addRow("Imported quantity", source)

    units = QtWidgets.QComboBox()
    units.setObjectName("ins_source_unit")
    if config["normalization_basis"] == "per_magnetic_ion":
        label = str(config.get("normalization_label", "") or "").strip()
        basis_suffix = (
            f" / {label}" if label else " / magnetic ion",
            f"/{label}" if label else "/magnetic ion",
        )
    else:
        basis_suffix = {
            "per_formula_unit": (" / f.u.", "/f.u."),
            "per_unit_cell": (" / unit cell", "/unit cell"),
            "unknown": ("", ""),
        }[config["normalization_basis"]]
    if config["source_representation"] == "cross_section":
        units.addItem("Arbitrary units", "arbitrary")
        units.addItem(
            f"mbarn / sr / meV{basis_suffix[0]}",
            f"mbarn/sr/meV{basis_suffix[1]}",
        )
        units.addItem(
            f"barn / sr / meV{basis_suffix[0]}",
            f"barn/sr/meV{basis_suffix[1]}",
        )
        units.addItem(
            f"Normalized intensity (1 / meV{basis_suffix[0]})",
            f"1/meV{basis_suffix[1]}",
        )
    else:
        units.addItem("Arbitrary units", "arbitrary")
        units.addItem(
            f"μ_B² / meV{basis_suffix[0]}",
            f"mu_B^2/meV{basis_suffix[1]}",
        )
        units.addItem(
            f"spin² / meV{basis_suffix[0]}",
            f"spin^2/meV{basis_suffix[1]}",
        )
    units.setCurrentIndex(max(units.findData(config["source_unit"]), 0))
    units.setToolTip(
        "Units carried by the imported signal. Choose arbitrary units when "
        "the beam flux and illuminated formula-unit count are not calibrated."
    )
    units.currentIndexChanged.connect(
        lambda _index, combo=units: set_setting(
            dataset, group, "source_unit", str(combo.currentData())
        )
    )
    form.addRow("Imported units", units)

    fit_representation = QtWidgets.QComboBox()
    fit_representation.setObjectName("ins_fit_representation")
    fit_representation.addItem("Scattering cross section", "cross_section")
    fit_representation.addItem("Dynamical susceptibility χ″", "chi_double_prime")
    fit_representation.setCurrentIndex(
        max(fit_representation.findData(config["fit_representation"]), 0)
    )
    fit_representation.setToolTip(
        "Representation used as the dataset signal for fitting and the default "
        "viewer channel. Both named channels remain selectable in the viewer."
    )
    fit_representation.currentIndexChanged.connect(
        lambda _index, combo=fit_representation: set_setting(
            dataset, group, "fit_representation", str(combo.currentData())
        )
    )
    form.addRow("Plot and fit", fit_representation)

    basis = QtWidgets.QComboBox()
    basis.setObjectName("ins_normalization_basis")
    for label, value in (
        ("Per formula unit", "per_formula_unit"),
        ("Per magnetic ion", "per_magnetic_ion"),
        ("Per unit cell", "per_unit_cell"),
        ("Unknown", "unknown"),
    ):
        basis.addItem(label, value)
    basis.setCurrentIndex(max(basis.findData(config["normalization_basis"]), 0))
    basis.setToolTip(
        "Amount-of-sample basis for absolute channels. This label does not "
        "perform a hidden rescaling; the upstream calibration must use the same basis."
    )
    basis.currentIndexChanged.connect(
        lambda _index, combo=basis: set_setting(
            dataset, group, "normalization_basis", str(combo.currentData())
        )
    )
    form.addRow("Normalization", basis)

    advanced = QtWidgets.QGroupBox("Corrections and conventions")
    advanced_columns = QtWidgets.QHBoxLayout(advanced)
    advanced_form = QtWidgets.QFormLayout()
    advanced_columns.addLayout(advanced_form, 1)
    advanced_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.addRow(advanced)
    main_form = form
    form = advanced_form

    normalization_label = QtWidgets.QLineEdit(
        str(config.get("normalization_label", "") or "")
    )
    constrain_input_width(normalization_label, COMPACT_SHORT_TEXT_FIELD_WIDTH)
    normalization_label.setObjectName("ins_normalization_label")
    normalization_label.setPlaceholderText("for example V")
    normalization_label.setEnabled(
        config["normalization_basis"] == "per_magnetic_ion"
    )
    normalization_label.setToolTip(
        "Optional atom or ion label for per-ion data, for example V. "
        "It changes the displayed denominator from 'magnetic ion' to '/V' "
        "without rescaling the imported values."
    )
    normalization_label.editingFinished.connect(
        lambda editor=normalization_label: set_setting(
            dataset, group, "normalization_label", editor.text().strip()
        )
    )
    form.addRow("Atom / ion label", normalization_label)

    calibration = QtWidgets.QLineEdit(
        _parameter_to_text(config.get("signal_per_mbarn", 0.0))
    )
    constrain_input_width(calibration, COMPACT_SCALAR_FIELD_WIDTH)
    calibration.setObjectName("ins_signal_per_mbarn")
    calibration.setToolTip(
        "For an arbitrary imported cross section, the number of imported "
        "signal units corresponding to 1 mbarn/(sr meV) on the selected basis. "
        "Leave 0 to retain arbitrary units."
    )
    calibration.editingFinished.connect(
        lambda editor=calibration: set_setting(
            dataset,
            group,
            "signal_per_mbarn",
            float(_parse_parameter_text(editor.text()) or 0.0),
        )
    )
    form.addRow("Signal units / mbarn", calibration)

    ion = QtWidgets.QComboBox()
    ion.setObjectName("ins_form_factor_ion")
    ion.addItem("No form-factor correction", "")
    for ion_name in available_ions():
        ion.addItem(ion_name, ion_name)
    ion.setCurrentIndex(max(ion.findData(config.get("form_factor_ion", "")), 0))
    ion.setToolTip(
        "Magnetic ion used for |f(Q)|². Selecting none leaves the form factor "
        "in both representations; choose an ion to remove it from χ″."
    )
    ion.currentIndexChanged.connect(
        lambda _index, combo=ion: set_setting(
            dataset, group, "form_factor_ion", str(combo.currentData())
        )
    )
    form.addRow("Magnetic form factor", ion)

    polarization = QtWidgets.QComboBox()
    polarization.setObjectName("ins_polarization_mode")
    for label, value in (
        ("Already corrected", "already_corrected"),
        ("Isotropic χ″ per component (P = 2)", "isotropic_single_component"),
        ("Isotropic trace χ″ (P = 2/3)", "isotropic_trace"),
        ("Custom scalar", "custom_scalar"),
    ):
        polarization.addItem(label, value)
    polarization.setCurrentIndex(
        max(polarization.findData(config["polarization_mode"]), 0)
    )
    polarization.setToolTip(
        "Contraction with δαβ − Q̂αQ̂β. For χ″ defined as the isotropic "
        "three-component trace, use 2/3; for one Cartesian component, use 2."
    )
    polarization.currentIndexChanged.connect(
        lambda _index, combo=polarization: set_setting(
            dataset, group, "polarization_mode", str(combo.currentData())
        )
    )
    form.addRow("Polarization convention", polarization)

    polarization_scalar = QtWidgets.QLineEdit(
        _parameter_to_text(config["polarization_scalar"])
    )
    constrain_input_width(polarization_scalar, COMPACT_SCALAR_FIELD_WIDTH)
    polarization_scalar.setObjectName("ins_polarization_scalar")
    polarization_scalar.setEnabled(config["polarization_mode"] == "custom_scalar")
    polarization_scalar.setToolTip(
        "Positive custom polarization factor P(Q), used only with Custom scalar."
    )
    polarization_scalar.editingFinished.connect(
        lambda editor=polarization_scalar: set_setting(
            dataset,
            group,
            "polarization_scalar",
            float(_parse_parameter_text(editor.text())),
        )
    )
    form.addRow("Custom P", polarization_scalar)

    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    advanced_columns.addLayout(form, 1)

    moment = QtWidgets.QComboBox()
    moment.setObjectName("ins_moment_unit")
    moment.addItem("Magnetic-moment χ″ (μ_B²)", "mu_B_squared")
    moment.addItem("Spin-operator χ″ (spin²)", "spin_squared")
    moment.setCurrentIndex(max(moment.findData(config["moment_unit"]), 0))
    moment.setToolTip(
        "μ_B² susceptibility already contains the magnetic moment and receives "
        "no extra g². Spin-operator susceptibility is multiplied by g²."
    )
    moment.currentIndexChanged.connect(
        lambda _index, combo=moment: set_setting(
            dataset, group, "moment_unit", str(combo.currentData())
        )
    )
    form.addRow("χ″ convention", moment)

    g_factor = QtWidgets.QLineEdit(_parameter_to_text(config["g_factor"]))
    constrain_input_width(g_factor, COMPACT_SCALAR_FIELD_WIDTH)
    g_factor.setObjectName("ins_g_factor")
    g_factor.setEnabled(config["moment_unit"] == "spin_squared")
    g_factor.setToolTip(
        "Landé g factor. It is applied exactly once, and only when χ″ is "
        "declared in spin² rather than magnetic-moment μ_B²."
    )
    g_factor.editingFinished.connect(
        lambda editor=g_factor: set_setting(
            dataset, group, "g_factor", float(_parse_parameter_text(editor.text()))
        )
    )
    form.addRow("Landé g", g_factor)

    kinematic = QtWidgets.QComboBox()
    kinematic.setObjectName("ins_kf_ki_state")
    kinematic.addItem("Removed upstream (S(Q,E)-like)", "removed")
    kinematic.addItem("Included in imported cross section", "included")
    kinematic.setCurrentIndex(max(kinematic.findData(config["kf_ki_state"]), 0))
    kinematic.setToolTip(
        "Whether the imported signal still contains the k_f/k_i phase-space "
        "factor. Included data require fixed Ei or Ef so nfit can remove it from χ″."
    )
    kinematic.currentIndexChanged.connect(
        lambda _index, combo=kinematic: set_setting(
            dataset, group, "kf_ki_state", str(combo.currentData())
        )
    )
    form.addRow("k_f/k_i state", kinematic)

    for label, key, object_name, tooltip in (
        (
            "Fixed Ei (meV)",
            "incident_energy_meV",
            "ins_incident_energy",
            "Fixed incident energy for direct-geometry data. Leave blank when k_f/k_i was removed upstream.",
        ),
        (
            "Fixed Ef (meV)",
            "final_energy_meV",
            "ins_final_energy",
            "Fixed final energy for indirect-geometry data. Leave blank when k_f/k_i was removed upstream.",
        ),
    ):
        editor = QtWidgets.QLineEdit(_parameter_to_text(config.get(key, "")))
        constrain_input_width(editor, COMPACT_SCALAR_FIELD_WIDTH)
        editor.setObjectName(object_name)
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda ed=editor, setting=key: set_setting(
                dataset,
                group,
                setting,
                _parse_parameter_text(ed.text()),
            )
        )
        form.addRow(label, editor)
    temperature = dataset.parameters.get(
        "temperature", getattr(dataset.data, "metadata", {}).get("temperature")
    )
    if temperature in (None, ""):
        status_text = "Set T in Conditions to create the paired channel."
    elif (
        config["kf_ki_state"] == "included"
        and config.get("incident_energy_meV") in (None, "")
        and config.get("final_energy_meV") in (None, "")
    ):
        status_text = "Provide fixed Ei or Ef to remove k_f/k_i."
    else:
        status_text = "Cross-section and χ″ channels are available."
    status = QtWidgets.QLabel(status_override or status_text)
    status.setObjectName("ins_channel_status")
    status.setWordWrap(True)
    status.setToolTip(
        "Readiness of the paired INS conversion. The imported signal remains "
        "available even when a required conversion input is missing."
    )
    main_form.addRow("Status", status)
    return box


def _metadata_tree_group_box(
    self,
    title: str,
    mapping: dict[str, Any],
    *,
    object_name: str,
    empty_text: str,
) -> Any:
    from PySide6 import QtCore, QtWidgets

    group_box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)
    if not mapping:
        label = QtWidgets.QLabel(empty_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        return group_box

    tree = QtWidgets.QTreeWidget()
    tree.setObjectName(object_name)
    tree.setToolTip("Expandable metadata table. Expand rows to inspect nested fields and hover values for full text.")
    tree.setColumnCount(2)
    tree.setHeaderLabels(["Field", "Value"])
    tree.setRootIsDecorated(True)
    tree.setAlternatingRowColors(True)
    tree.setUniformRowHeights(True)
    tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
    tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    tree.setMinimumHeight(120)
    tree.setMaximumHeight(260)
    tree.header().setStretchLastSection(True)
    tree.header().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

    for key in sorted(mapping):
        item = _add_metadata_tree_item(tree, str(key), mapping[key])
        if key == "Parameters":
            item.setExpanded(True)
    for index in range(min(4, tree.topLevelItemCount())):
        tree.topLevelItem(index).setExpanded(True)
    tree.resizeColumnToContents(0)
    layout.addWidget(tree)
    return group_box
