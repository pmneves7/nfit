"""Generic model-parameter editor orchestration for the project GUI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .fit_config import parameter_is_derived_by_closure, sharing_mode
from .form_factors import available_ions
from .model_registry import (
    model_config_tooltip,
    model_definition,
    model_parameter_tooltip,
)
from .pipeline import ModelComponentSpec

CUSTOM_FORM_FACTOR_CHOICE = "__custom__"


def model_parameter_names(model: ModelComponentSpec) -> list[str]:
    """Return all parameter names of a model, static plus config-derived.

    Models such as ``heisenberg_rpa`` emit one exchange parameter per bond
    orbit in their configuration; this mirrors
    :func:`nfit.fit_config.component_parameter_names` for the GUI.
    """

    from .fit_config import component_parameter_names

    return list(component_parameter_names(model))


def model_crystal_config(model: ModelComponentSpec) -> dict[str, Any]:
    """Return (creating if needed) the nested crystal config of a model.

    Shape: ``{"lattice": {"a", "b", "c", "alpha", "beta", "gamma"},
    "spacegroup": str, "sites": [{"label", "element", "position", "ion"}],
    "provenance": {...}}``. Provenance is optional. The dict lives inside
    ``model.config`` and is plain JSON data, so it serializes with the project.
    """

    crystal = model.config.get("crystal")
    if not isinstance(crystal, dict):
        crystal = {}
        model.config["crystal"] = crystal
    lattice = crystal.setdefault(
        "lattice",
        {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )
    for name, fallback in (
        ("a", 5.0),
        ("b", 5.0),
        ("c", 5.0),
        ("alpha", 90.0),
        ("beta", 90.0),
        ("gamma", 90.0),
    ):
        lattice.setdefault(name, fallback)
    crystal.setdefault("spacegroup", "P 1")
    crystal.setdefault("sites", [])
    return crystal


def _parameter_is_at_bound(
    value: Any,
    bound: Any,
    other_bound: Any = None,
) -> bool:
    """Return whether a fitted value is effectively pinned to a finite bound.

    For two-sided bounds, proximity is measured against the allowed interval.
    This catches optimizer solutions that stop very close to a boundary without
    requiring bit-for-bit equality. One-sided bounds retain a small
    scale-aware numerical tolerance.
    """

    try:
        value_float = float(value)
        bound_float = float(bound)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value_float) or not np.isfinite(bound_float):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(bound_float))
    try:
        other_float = float(other_bound)
    except (TypeError, ValueError):
        other_float = np.nan
    if np.isfinite(other_float):
        span = abs(other_float - bound_float)
        if span > 0.0:
            tolerance = max(tolerance, 1.0e-4 * span)
    return bool(abs(value_float - bound_float) <= tolerance)


def _model_parameter_limit_side(model: ModelComponentSpec, parameter_name: str) -> str | None:
    """Return the reached side for a fitted model parameter, if any."""

    if not bool(model.fit_parameters.get(parameter_name, False)):
        return None
    raw = model.limits.get(parameter_name)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    value = model.parameters.get(parameter_name)
    for side, bound, other_bound in (
        ("lower", raw[0], raw[1]),
        ("upper", raw[1], raw[0]),
    ):
        if bound not in (None, "") and _parameter_is_at_bound(value, bound, other_bound):
            return side
    return None


def _model_limit_texts(model: ModelComponentSpec, parameter_name: str) -> tuple[str, str]:
    """Return display texts for a parameter's (min, max) bounds."""

    raw = model.limits.get(parameter_name)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return "", ""
    lower, upper = raw
    return (
        "" if lower in (None, "") else _parameter_to_text(lower),
        "" if upper in (None, "") else _parameter_to_text(upper),
    )


def _sharing_groups_text(model: ModelComponentSpec, parameter_name: str) -> str:
    """Return grouped-sharing assignments in the editor's compact syntax."""

    entry = model.sharing.get(parameter_name)
    groups = entry.get("groups") if isinstance(entry, dict) else {}
    if not isinstance(groups, dict):
        return ""
    return ", ".join(f"{dataset}={tie_group}" for dataset, tie_group in groups.items())


def _sampling_float_matches(saved: Any, current: Any) -> bool:
    """Return whether a serialized certificate scalar matches current state."""

    try:
        saved_value = float(saved)
        current_value = float(current)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(saved_value) and np.isfinite(current_value) and saved_value == current_value
    )


def _parameter_to_text(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _rebuild_model_parameter_editor(self, model: ModelComponentSpec) -> None:
    from PySide6 import QtWidgets

    self._clear_model_parameter_editor()
    fit_group = QtWidgets.QGroupBox("Fit Parameters")
    fit_group.setObjectName("model_fit_parameters_group")
    fit_layout = QtWidgets.QGridLayout(fit_group)
    fit_layout.setVerticalSpacing(4)
    fit_layout.setColumnStretch(1, 1)
    header_label = QtWidgets.QLabel("Plot label")
    header_min = QtWidgets.QLabel("Min")
    header_max = QtWidgets.QLabel("Max")
    plot_label_tooltip = (
        "Optional short label used in fit diagnostic plots. "
        "Accepts plain text or Matplotlib mathtext such as $\\Gamma$."
    )
    header_label.setToolTip(plot_label_tooltip)
    fit_layout.addWidget(header_label, 0, 2)
    fit_layout.addWidget(header_min, 0, 3)
    fit_layout.addWidget(header_max, 0, 4)
    parameter_names = model_parameter_names(model)
    if model.type == "tight_binding":
        note = QtWidgets.QLabel(
            "Onsite and hopping coefficients use the common parameter, "
            "bounds, fit-selection, and sharing machinery. Edit them in "
            "their orbital-aware tables below."
        )
        note.setObjectName("tight_binding_fit_parameter_note")
        note.setWordWrap(True)
        note.setToolTip(
            "The tight-binding component is calculation-only until a "
            "compatible electronic-response model supplies a dataset observable."
        )
        fit_layout.addWidget(note, 1, 0, 1, 8)
        parameter_names = []
    for index, parameter_name in enumerate(parameter_names):
        row = index + 1
        label = QtWidgets.QLabel(parameter_name)
        editor = QtWidgets.QLineEdit(_parameter_to_text(model.parameters.get(parameter_name, "")))
        editor.setObjectName(f"model_parameter_value_{parameter_name}")
        parameter_labels = model.metadata.get("parameter_labels")
        plot_label_editor = QtWidgets.QLineEdit(
            str(parameter_labels.get(parameter_name, ""))
            if isinstance(parameter_labels, dict)
            else ""
        )
        plot_label_editor.setObjectName(f"model_parameter_plot_label_{parameter_name}")
        plot_label_editor.setPlaceholderText(f"p{index + 1}")
        lower_text, upper_text = _model_limit_texts(model, parameter_name)
        min_editor = QtWidgets.QLineEdit(lower_text)
        min_editor.setObjectName(f"model_parameter_min_{parameter_name}")
        min_editor.setPlaceholderText("-inf")
        min_editor.setMaximumWidth(70)
        max_editor = QtWidgets.QLineEdit(upper_text)
        max_editor.setObjectName(f"model_parameter_max_{parameter_name}")
        max_editor.setPlaceholderText("inf")
        max_editor.setMaximumWidth(70)
        fit_check = QtWidgets.QCheckBox("Fit")
        fit_check.setObjectName(f"model_parameter_fit_{parameter_name}")
        closure_derived = parameter_is_derived_by_closure(model, parameter_name)
        fit_check.setChecked(
            bool(model.fit_parameters.get(parameter_name, False)) and not closure_derived
        )
        fit_check.setEnabled(not closure_derived)
        sharing_combo = QtWidgets.QComboBox()
        sharing_combo.setObjectName(f"model_parameter_sharing_{parameter_name}")
        sharing_combo.addItem("Global", "global")
        sharing_combo.addItem("Per dataset", "per_dataset")
        sharing_combo.addItem("Grouped", "grouped")
        mode = sharing_mode(model, parameter_name)
        sharing_combo.setCurrentIndex(max(sharing_combo.findData(mode), 0))
        sharing_groups = QtWidgets.QLineEdit(_sharing_groups_text(model, parameter_name))
        sharing_groups.setObjectName(f"model_parameter_sharing_groups_{parameter_name}")
        sharing_groups.setPlaceholderText("scan1=A, scan2=A")
        sharing_groups.setVisible(mode == "grouped")
        tooltip = model_parameter_tooltip(model.type, parameter_name)
        limit_side = _model_parameter_limit_side(model, parameter_name)
        if limit_side is not None:
            warning = f"Warning: this fitted parameter is at its {limit_side} bound."
            editor.setStyleSheet("color: #c0392b; font-weight: 600;")
            label.setStyleSheet("color: #c0392b; font-weight: 600;")
            tooltip = f"{tooltip}\n{warning}"
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        plot_label_editor.setToolTip(plot_label_tooltip)
        limits_tooltip = (
            "Optional bound applied when the optimizer varies this parameter. "
            "Leave blank for an unbounded side."
        )
        min_editor.setToolTip(limits_tooltip)
        max_editor.setToolTip(limits_tooltip)
        fit_check.setToolTip(
            (
                (
                    "Rotational invariance derives U_prime = U - 2 J_H and "
                    "J_pair = J_H; this stored value is not independently fitted."
                )
                if model.type == "hubbard_hund_rpa"
                else (
                    "Takahashi TAC solves chi0_eff from the conserved amplitude; "
                    "the stored chi0 is only a root-search seed and cannot be fitted."
                )
            )
            if closure_derived
            else (
                f"Optimize parameter {parameter_name!r} during fitting. "
                "Unchecked parameters stay fixed at their current value."
            )
        )
        sharing_tooltip = (
            f"Choose how parameter {parameter_name!r} is shared: one global "
            "value, one value per dataset, or values tied by named groups."
        )
        sharing_combo.setToolTip(sharing_tooltip)
        sharing_groups.setToolTip(
            "For grouped sharing, enter comma-separated dataset=group pairs. "
            "Unlisted datasets remain independent."
        )
        editor.editingFinished.connect(
            lambda parameter_name=parameter_name, editor=editor: self._set_model_parameter(
                parameter_name, editor.text()
            )
        )
        plot_label_editor.editingFinished.connect(
            lambda parameter_name=parameter_name, editor=plot_label_editor: (
                self._set_model_parameter_plot_label(parameter_name, editor.text())
            )
        )
        min_editor.editingFinished.connect(
            lambda parameter_name=parameter_name, editor=min_editor: self._set_model_limit(
                parameter_name, 0, editor.text()
            )
        )
        max_editor.editingFinished.connect(
            lambda parameter_name=parameter_name, editor=max_editor: self._set_model_limit(
                parameter_name, 1, editor.text()
            )
        )
        fit_check.toggled.connect(
            lambda checked, parameter_name=parameter_name: self._set_model_fit_parameter(
                parameter_name, checked
            )
        )
        sharing_combo.currentIndexChanged.connect(
            lambda _index, parameter_name=parameter_name, combo=sharing_combo: (
                self._set_model_sharing_mode(parameter_name, str(combo.currentData()))
            )
        )
        sharing_groups.editingFinished.connect(
            lambda parameter_name=parameter_name, editor=sharing_groups: (
                self._set_model_sharing_groups(parameter_name, editor.text())
            )
        )
        fit_layout.addWidget(label, row, 0)
        fit_layout.addWidget(editor, row, 1)
        fit_layout.addWidget(plot_label_editor, row, 2)
        fit_layout.addWidget(min_editor, row, 3)
        fit_layout.addWidget(max_editor, row, 4)
        fit_layout.addWidget(fit_check, row, 5)
        fit_layout.addWidget(sharing_combo, row, 6)
        fit_layout.addWidget(sharing_groups, row, 7)
    if model.type not in {"tight_binding", "lindhard"}:
        self.model_parameter_layout.addWidget(fit_group, 0, 0, 1, 4)

    scope_group = QtWidgets.QGroupBox("Dataset Scope")
    scope_group.setObjectName("model_dataset_scope_group")
    scope_layout = QtWidgets.QGridLayout(scope_group)
    scope_layout.setColumnStretch(1, 1)
    applies_label = QtWidgets.QLabel("Applies to")
    applies_editor = QtWidgets.QLineEdit(
        "" if model.applies_to is None else ", ".join(model.applies_to)
    )
    applies_editor.setObjectName("model_applies_to_editor")
    applies_tooltip = (
        "Comma-separated dataset names this model component fits. "
        "Leave blank to apply the model to every compatible dataset."
    )
    applies_label.setToolTip(applies_tooltip)
    applies_editor.setToolTip(applies_tooltip)
    applies_editor.setPlaceholderText("all compatible datasets")
    applies_editor.editingFinished.connect(
        lambda editor=applies_editor: self._set_model_applies_to(editor.text())
    )
    scope_layout.addWidget(applies_label, 0, 0)
    scope_layout.addWidget(applies_editor, 0, 1)
    if model.type not in {"tight_binding", "lindhard"}:
        self.model_parameter_layout.addWidget(scope_group, 1, 0, 1, 4)

    config_group = QtWidgets.QGroupBox(
        "Advanced model and execution"
        if model.type in {"tight_binding", "lindhard", "heisenberg_rpa"}
        else "Configuration Settings"
    )
    config_group.setObjectName("model_config_group")
    config_layout = QtWidgets.QGridLayout(config_group)
    config_layout.setColumnStretch(1, 1)
    definition = model_definition(model.type)
    config_definitions = {field.name: field for field in definition.config_fields}
    resource_settings = {
        "response_workers",
        "response_max_batch_mb",
        "response_transition_max_batch_mb",
        "response_cache_mb",
        "electronic_workers",
        "electronic_max_batch_mb",
    }
    if not config_definitions:
        config_layout.addWidget(QtWidgets.QLabel("No configuration settings."), 0, 0, 1, 2)
    row = 0
    if model.type == "tight_binding":
        primitive = QtWidgets.QCheckBox("Attempt certified primitive-cell reduction")
        primitive.setObjectName("tight_binding_use_primitive_cell")
        primitive.setChecked(bool(model.config.get("use_primitive_cell", True)))
        primitive.setEnabled(not bool(str(model.config.get("source_path", "")).strip()))
        primitive.setToolTip(
            "Try to fold a GUI-built conventional-cell Hamiltonian onto "
            "the primitive translation lattice. nfit uses the reduced "
            "model only when the basis and Hamiltonian pass the exact "
            "compatibility checks; otherwise it records a conventional-cell "
            "fallback and its reason. Disable only for diagnostic comparison."
        )
        primitive.toggled.connect(
            lambda checked: self._set_tight_binding_use_primitive_cell(bool(checked))
        )
        config_layout.addWidget(primitive, row, 0, 1, 2)
        row += 1
    for setting_name in config_definitions:
        if setting_name == "form_factor_coefficients" or setting_name in resource_settings:
            continue
        if model.type == "lindhard" and setting_name in {
            "electronic_component",
            "response_mesh",
            "response_sampling_mode",
            "response_sampling_accuracy",
            "response_sampling_custom_rtol",
            "response_sampling_max_refinements",
            "response_sampling_max_mesh_points",
            "response_sampling_points_per_dataset",
            "response_sampling_certificate",
            "response_mesh_shift",
            "response_symmetry",
            "response_q_evaluation",
            "response_q_interpolation_rtol",
            "chemical_potential_mode",
            "filling_per_cell",
            "formula_units_mode",
            "formula_units_per_cell",
            "magnetic_normalization_mode",
            "magnetic_normalization_species",
            "magnetic_centers_per_model_cell",
            "bulk_g_factor",
            "plot_q_reduced",
            "plot_energy_min_meV",
            "plot_energy_max_meV",
            "plot_energy_points",
            "plot_temperature_K",
            "convergence_mesh_scales",
            "convergence_broadening_scales",
            "convergence_energy_points",
            "convergence_relative_floor",
        }:
            continue
        if model.type == "tight_binding" and setting_name in {
            "source_path",
            "model_digest",
            "model_data",
            "model_stale",
            "use_primitive_cell",
            "hopping_parameterization",
            "crystal",
            "orbital_manifolds",
            "spin_treatment",
            "soc_terms",
            "onsite_terms",
            "hopping_cutoff_angstrom",
            "spatial_orbits",
            "hopping_candidates",
            "hopping_terms",
            "electronic_energy_unit",
            "chemical_potential_meV",
            "band_path_convention",
            "band_path_metadata",
            "band_path",
            "band_points_per_inv_angstrom",
            "dos_method",
            "dos_mesh",
            "dos_sampling_mode",
            "dos_sampling_accuracy",
            "dos_sampling_custom_rtol",
            "dos_sampling_certificate",
            "dos_symmetry",
            "dos_energy_min_meV",
            "dos_energy_max_meV",
            "dos_energy_points",
            "dos_broadening_meV",
            "fermi_mesh_mode",
            "fermi_spacing_inv_angstrom",
            "fermi_mesh",
            "fermi_energy_meV",
        }:
            continue
        label = QtWidgets.QLabel(setting_name)
        tooltip = model_config_tooltip(model.type, setting_name)
        label.setToolTip(tooltip)
        if model.type == "tight_binding":
            label.setText(
                {
                    "periodic_axes": "Periodic axes",
                    "projection_groups": "Custom projection groups",
                }.get(setting_name, setting_name)
            )
        if model.type == "tight_binding" and setting_name == "electronic_energy_unit":
            from .electronic_structure import (
                ELECTRONIC_ENERGY_UNITS,
                normalize_electronic_energy_unit,
            )

            label.setText("Electronic energy unit")
            tooltip = (
                f"{tooltip}\n"
                "This controls tight-binding energy entry and electronic "
                "plot labels. Canonical model data and neutron-response "
                "calculations remain in meV; changing this choice does not "
                "change any physical value."
            )
            label.setToolTip(tooltip)
            combo = QtWidgets.QComboBox()
            combo.setObjectName("tight_binding_energy_unit")
            combo.setToolTip(tooltip)
            for unit in ELECTRONIC_ENERGY_UNITS:
                combo.addItem(unit, unit)
            current = normalize_electronic_energy_unit(model.config.get(setting_name, "eV"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_tight_binding_energy_unit(
                    str(combo.currentData())
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if model.type == "tight_binding" and setting_name == "electronic_backend":
            label.setText("Electronic backend")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("tight_binding_electronic_backend")
            combo.setToolTip(tooltip)
            for backend in ("auto", "numpy", "threaded", "cupy"):
                combo.addItem(backend, backend)
            current = str(model.config.get(setting_name, "auto"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_config_setting(
                    "electronic_backend",
                    str(combo.currentData()),
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if model.type == "tight_binding" and setting_name == "dos_method":
            label.setText("DOS integration")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("tight_binding_dos_method")
            combo.setToolTip(tooltip)
            combo.addItem("Gaussian broadening", "gaussian")
            combo.addItem("Linear tetrahedron (ASE)", "tetrahedron")
            current = str(model.config.get(setting_name, "gaussian"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_config_setting(
                    "dos_method",
                    str(combo.currentData()),
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if model.type == "tight_binding" and setting_name == "dos_symmetry":
            label.setText("DOS symmetry")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("tight_binding_dos_symmetry")
            combo.setToolTip(tooltip)
            for policy in ("full", "auto", "reduced"):
                combo.addItem(policy, policy)
            current = str(model.config.get(setting_name, "full"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_config_setting(
                    "dos_symmetry",
                    str(combo.currentData()),
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if model.type == "lindhard" and setting_name == "response_symmetry":
            label.setText("Response symmetry")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("lindhard_response_symmetry")
            combo.setToolTip(tooltip)
            for policy in ("auto", "full", "reduced"):
                combo.addItem(policy, policy)
            current = str(model.config.get(setting_name, "auto"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_config_setting(
                    "response_symmetry",
                    str(combo.currentData()),
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if model.type == "lindhard" and setting_name == "response_backend":
            label.setText("Response backend")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("lindhard_response_backend")
            combo.setToolTip(tooltip)
            for backend in ("auto", "numpy", "threaded", "cupy"):
                combo.addItem(backend, backend)
            current = str(model.config.get(setting_name, "auto"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_config_setting(
                    "response_backend",
                    str(combo.currentData()),
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if model.type == "lindhard" and setting_name == "response_q_evaluation":
            label.setText("Q evaluation")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("lindhard_response_q_evaluation")
            combo.setToolTip(tooltip)
            for policy in (
                "auto",
                "direct",
                "commensurate",
                "interpolated",
            ):
                combo.addItem(policy, policy)
            current = str(model.config.get(setting_name, "auto"))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_config_setting(
                    "response_q_evaluation",
                    str(combo.currentData()),
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if setting_name in definition.component_reference_fields:
            reference_types = definition.metadata.get("reference_types", {})
            allowed_types = (
                tuple(reference_types.get(setting_name, ()))
                if isinstance(reference_types, Mapping)
                else ()
            )
            if not allowed_types and setting_name == "electronic_component":
                allowed_types = ("tight_binding",)
            if setting_name == "electronic_component":
                label.setText("Electronic structure")
                prompt = "Select tight-binding model…"
            elif setting_name == "response_component":
                label.setText("Bare response")
                prompt = "Select Lindhard response…"
            else:
                prompt = "Select component…"
            combo = QtWidgets.QComboBox()
            combo.setObjectName(f"{model.type}_{setting_name}")
            combo.setToolTip(tooltip)
            combo.addItem(prompt, "")
            owner = self._group_for_model(model)
            if owner is not None:
                for candidate in owner.models.values():
                    if (
                        candidate.type in allowed_types
                        and candidate.enabled
                        and candidate is not model
                    ):
                        combo.addItem(candidate.name, candidate.name)
            current = str(model.config.get(setting_name, ""))
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo, setting_name=setting_name: (
                    self._set_model_config_setting(
                        setting_name,
                        str(combo.currentData()),
                    )
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            continue
        if config_definitions[setting_name].choices == "form_factor_ions":
            label.setText("form_factor")
            combo = QtWidgets.QComboBox()
            combo.setObjectName(f"model_config_choice_{setting_name}")
            combo.addItem("(none)", "")
            for ion in available_ions():
                combo.addItem(ion, ion)
            combo.addItem("Custom...", CUSTOM_FORM_FACTOR_CHOICE)
            current = str(model.config.get(setting_name, "") or "")
            if str(model.config.get("form_factor_coefficients", "") or "").strip():
                current = CUSTOM_FORM_FACTOR_CHOICE
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.setToolTip(tooltip)
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo: self._set_model_form_factor_choice(
                    str(combo.currentData() or "")
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(combo, row, 1)
            row += 1
            if current == CUSTOM_FORM_FACTOR_CHOICE:
                coeff_tooltip = model_config_tooltip(model.type, "form_factor_coefficients")
                coeff_label = QtWidgets.QLabel("custom coefficients")
                coeff_label.setToolTip(coeff_tooltip)
                coeff_editor = QtWidgets.QLineEdit(
                    _parameter_to_text(model.config.get("form_factor_coefficients", ""))
                )
                coeff_editor.setObjectName("model_config_form_factor_coefficients")
                coeff_editor.setToolTip(coeff_tooltip)
                coeff_editor.editingFinished.connect(
                    lambda editor=coeff_editor: self._set_model_config_setting(
                        "form_factor_coefficients", editor.text()
                    )
                )
                config_layout.addWidget(coeff_label, row, 0)
                config_layout.addWidget(coeff_editor, row, 1)
                row += 1
            continue
        tight_binding_energy_fields = {
            "chemical_potential_meV",
            "dos_energy_min_meV",
            "dos_energy_max_meV",
            "dos_broadening_meV",
            "fermi_energy_meV",
        }
        if model.type == "tight_binding" and setting_name in tight_binding_energy_fields:
            from .electronic_structure import (
                electronic_energy_from_meV,
                normalize_electronic_energy_unit,
            )

            unit = normalize_electronic_energy_unit(
                model.config.get("electronic_energy_unit", "eV")
            )
            displayed = electronic_energy_from_meV(model.config.get(setting_name, 0.0), unit)
            label.setText(f"{setting_name.removesuffix('_meV')} ({unit})")
            tooltip = (
                f"{tooltip}\n"
                f"Enter this electronic energy in {unit}. nfit converts it "
                "to canonical meV immediately. Neutron energy transfer and "
                "spin-response linewidths are unaffected."
            )
            label.setToolTip(tooltip)
            editor = QtWidgets.QLineEdit(_parameter_to_text(displayed))
            editor.setObjectName(f"model_config_{setting_name}")
            editor.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda setting_name=setting_name, editor=editor: (
                    self._set_tight_binding_energy_config(setting_name, editor.text())
                )
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(editor, row, 1)
            row += 1
            continue
        editor = QtWidgets.QLineEdit(_parameter_to_text(model.config.get(setting_name, "")))
        editor.setObjectName(f"model_config_{setting_name}")
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda setting_name=setting_name, editor=editor: self._set_model_config_setting(
                setting_name, editor.text()
            )
        )
        config_layout.addWidget(label, row, 0)
        config_layout.addWidget(editor, row, 1)
        row += 1
    if model.type == "tight_binding":
        self._build_model_crystal_editor(model, structure_only=True)
        self._build_tight_binding_orbital_editor(model)
        self._build_tight_binding_spin_editor(model)
        self._build_tight_binding_onsite_editor(model)
        self._build_tight_binding_hopping_editor(model)
        self._build_tight_binding_state_editor(model)
        self._build_tight_binding_dos_sampling_editor(model)
        self._build_tight_binding_editor(model)
        self.model_parameter_layout.addWidget(config_group, 10, 0, 1, 4)
        self._organize_tight_binding_editor(model)
    elif model.type == "lindhard":
        self._build_lindhard_editor(model)
        self._build_model_plot_actions(model)
        self._organize_lindhard_editor(
            model,
            fit_group=fit_group,
            scope_group=scope_group,
            advanced_group=config_group,
        )
    elif model.type == "heisenberg_rpa":
        self.model_parameter_layout.addWidget(config_group, 2, 0, 1, 4)
        self._build_model_crystal_editor(model)
        self._build_heisenberg_advanced_editor(model)
        self._build_model_plot_actions(model)
        self._organize_heisenberg_editor(
            model,
            fit_group=fit_group,
            scope_group=scope_group,
            advanced_group=config_group,
        )
    elif definition.structured_config:
        self.model_parameter_layout.addWidget(config_group, 2, 0, 1, 4)
        self._build_model_crystal_editor(model)
        self._build_model_plot_actions(model)
    else:
        self.model_parameter_layout.addWidget(config_group, 2, 0, 1, 4)
        self._build_model_plot_actions(model)


def _rebuild_model_parameter_editor_preserving_scroll(
    self,
    model: ModelComponentSpec,
) -> None:
    """Rebuild model controls without losing scroll position or focus."""

    from .qt_widget_state import preserve_widget_state

    with preserve_widget_state(
        self.model_parameter_scroll,
        is_current=lambda: self._selected_model_and_group()[1] is model,
    ):
        self._rebuild_model_parameter_editor(model)
