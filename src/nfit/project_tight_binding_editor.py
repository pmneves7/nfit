"""Qt builders for tight-binding model controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .fit_config import sharing_mode
from .model_registry import model_definition
from .pipeline import ModelComponentSpec
from .project_model_editor import (
    _parameter_to_text,
    _sampling_float_matches,
    _sharing_groups_text,
    model_crystal_config,
)


def _build_tight_binding_state_editor(
    self,
    model: ModelComponentSpec,
) -> None:
    """Build the compact energy-reference controls shared by plots."""

    from PySide6 import QtWidgets

    from .electronic_structure import (
        ELECTRONIC_ENERGY_UNITS,
        electronic_energy_from_meV,
        normalize_electronic_energy_unit,
    )

    group = QtWidgets.QGroupBox("Electronic state")
    group.setObjectName("tight_binding_state_group")
    layout = QtWidgets.QGridLayout(group)

    unit_tooltip = (
        "Electronic-structure entries and plot labels use this display "
        "unit. nfit stores canonical energies in meV and converts without "
        "changing the physical Hamiltonian or neutron-energy convention."
    )
    unit_label = QtWidgets.QLabel("Energy unit")
    unit_label.setToolTip(unit_tooltip)
    unit = QtWidgets.QComboBox()
    unit.setObjectName("tight_binding_energy_unit")
    unit.setToolTip(unit_tooltip)
    for choice in ELECTRONIC_ENERGY_UNITS:
        unit.addItem(choice, choice)
    current_unit = normalize_electronic_energy_unit(
        model.config.get("electronic_energy_unit", "eV")
    )
    unit.setCurrentIndex(max(unit.findData(current_unit), 0))
    unit.currentIndexChanged.connect(
        lambda _index, combo=unit: self._set_tight_binding_energy_unit(str(combo.currentData()))
    )
    layout.addWidget(unit_label, 0, 0)
    layout.addWidget(unit, 0, 1)

    chemical_tooltip = (
        "Energy reference used for electronic plots and, when selected by "
        "a linked Lindhard component, for occupations. Enter it in the "
        "displayed electronic unit; nfit stores canonical meV."
    )
    chemical_label = QtWidgets.QLabel(f"Chemical potential ({current_unit})")
    chemical_label.setToolTip(chemical_tooltip)
    chemical = QtWidgets.QLineEdit(
        _parameter_to_text(
            electronic_energy_from_meV(
                model.config.get("chemical_potential_meV", 0.0),
                current_unit,
            )
        )
    )
    chemical.setObjectName("model_config_chemical_potential_meV")
    chemical.setToolTip(chemical_tooltip)
    chemical.editingFinished.connect(
        lambda editor=chemical: self._set_tight_binding_energy_config(
            "chemical_potential_meV",
            editor.text(),
        )
    )
    layout.addWidget(chemical_label, 1, 0)
    layout.addWidget(chemical, 1, 1)

    path_tooltip = (
        "Shared labelled path used by the Brillouin-zone and band viewers. "
        "Generate a standard Hinuma/HPKOT path with Seek-path or a "
        "Setyawan–Curtarolo path with ASE, or select Manual and edit its "
        "nodes in the band viewer."
    )
    path_label = QtWidgets.QLabel("Band-path convention")
    path_label.setToolTip(path_tooltip)
    path = QtWidgets.QComboBox()
    path.setObjectName("tight_binding_path_convention")
    path.setToolTip(path_tooltip)
    path.addItem("Manual path", "manual")
    path.addItem("Hinuma / Seek-path", "hinuma")
    path.addItem(
        "Setyawan–Curtarolo / ASE",
        "setyawan_curtarolo",
    )
    path.setCurrentIndex(
        max(
            path.findData(
                str(
                    model.config.get(
                        "band_path_convention",
                        "hinuma",
                    )
                )
            ),
            0,
        )
    )
    path.currentIndexChanged.connect(
        lambda _index, combo=path: self._generate_tight_binding_standard_path(
            str(combo.currentData())
        )
    )
    layout.addWidget(path_label, 2, 0)
    layout.addWidget(path, 2, 1, 1, 2)

    projection_count = len(model.config.get("projection_groups") or {})
    projection_summary = QtWidgets.QLabel(
        f"{projection_count} automatic/custom projection group(s)"
    )
    projection_summary.setObjectName("tight_binding_projection_summary")
    projection_summary.setToolTip(
        "Projection groups are generated from the active orbital "
        "manifolds. Custom basis-index groups remain available under Advanced."
    )
    layout.addWidget(projection_summary, 3, 0, 1, 3)
    self.model_parameter_layout.addWidget(group, 11, 0, 1, 4)


def _tight_binding_dos_sampling_status(
    self,
    model: ModelComponentSpec,
) -> str:
    config = model.config
    certificate = config.get("dos_sampling_certificate", {})
    if str(config.get("dos_sampling_mode", "automatic")) == "manual":
        return "Manual mesh: no automatic accuracy claim."
    if not isinstance(certificate, dict) or not certificate:
        return "Not certified. Check/refine before production use."
    if certificate.get("status") != "certified":
        return "Not certified within the configured refinement budget."
    if list(certificate.get("chosen_mesh") or ()) != list(config.get("dos_mesh", ())):
        return "Certificate is stale because the mesh changed."
    if bool(config.get("model_stale", False)):
        return "Certificate is stale because the electronic model must be rebuilt."
    if config.get("model_digest") and certificate.get("provenance", {}).get(
        "model_digest"
    ) != config.get("model_digest"):
        return "Certificate is stale because the electronic model changed."
    if certificate.get("provenance", {}).get("method") != str(config.get("dos_method", "gaussian")):
        return "Certificate is stale because the integration method changed."
    if not _sampling_float_matches(
        certificate.get("provenance", {}).get("broadening_meV"),
        config.get("dos_broadening_meV", 5.0),
    ):
        return "Certificate is stale because the broadening changed."
    if certificate.get("provenance", {}).get("symmetry") != str(config.get("dos_symmetry", "auto")):
        return "Certificate is stale because the symmetry policy changed."
    domain = certificate.get("domain", {})
    if int(domain.get("energy_points", -1)) != int(config.get("dos_energy_points", 600)):
        return "Certificate is stale because the DOS energy grid changed."
    if not bool(config.get("dos_auto_energy_range", False)):
        if not _sampling_float_matches(
            domain.get("energy_min_meV"),
            config.get("dos_energy_min_meV", -500.0),
        ) or not _sampling_float_matches(
            domain.get("energy_max_meV"),
            config.get("dos_energy_max_meV", 500.0),
        ):
            return "Certificate is stale because the DOS energy limits changed."
    return (
        "Certified at normalized tolerance "
        f"{certificate.get('policy', {}).get('relative_tolerance', '?')}."
    )


def _build_tight_binding_dos_sampling_editor(
    self,
    model: ModelComponentSpec,
) -> None:
    """Build model-level DOS mesh selection and certification controls."""

    from PySide6 import QtWidgets

    from .electronic_structure import (
        electronic_energy_from_meV,
        normalize_electronic_energy_unit,
    )

    group = QtWidgets.QGroupBox("Density-of-states sampling")
    group.setObjectName("tight_binding_dos_sampling_group")
    layout = QtWidgets.QFormLayout(group)
    mode = str(model.config.get("dos_sampling_mode", "automatic"))

    mode_tooltip = (
        "Automatic mode compares successive density-of-states calculations "
        "and stores a certified production mesh. Manual mode uses the "
        "entered mesh directly."
    )
    mode_combo = QtWidgets.QComboBox()
    mode_combo.setObjectName("tight_binding_dos_sampling_mode")
    mode_combo.setToolTip(mode_tooltip)
    mode_combo.addItem("Automatic certification", "automatic")
    mode_combo.addItem("Manual mesh", "manual")
    mode_combo.setCurrentIndex(max(mode_combo.findData(mode), 0))
    mode_combo.currentIndexChanged.connect(
        lambda _index, combo=mode_combo: self._set_model_config_setting(
            "dos_sampling_mode", str(combo.currentData())
        )
    )
    mode_label = QtWidgets.QLabel("Mesh selection")
    mode_label.setToolTip(mode_tooltip)
    layout.addRow(mode_label, mode_combo)

    accuracy = str(model.config.get("dos_sampling_accuracy", "standard"))
    accuracy_tooltip = (
        "Preview, Standard, and High require at most 5%, 1%, and 0.2% "
        "normalized change, respectively, for two successive refinements."
    )
    accuracy_combo = QtWidgets.QComboBox()
    accuracy_combo.setObjectName("tight_binding_dos_sampling_accuracy")
    accuracy_combo.setToolTip(accuracy_tooltip)
    for label, value in (
        ("Preview (5%)", "preview"),
        ("Standard (1%)", "standard"),
        ("High (0.2%)", "high"),
        ("Custom", "custom"),
    ):
        accuracy_combo.addItem(label, value)
    accuracy_combo.setCurrentIndex(max(accuracy_combo.findData(accuracy), 0))
    accuracy_combo.setEnabled(mode == "automatic")
    accuracy_combo.currentIndexChanged.connect(
        lambda _index, combo=accuracy_combo: self._set_model_config_setting(
            "dos_sampling_accuracy", str(combo.currentData())
        )
    )
    accuracy_label = QtWidgets.QLabel("Accuracy")
    accuracy_label.setToolTip(accuracy_tooltip)
    layout.addRow(accuracy_label, accuracy_combo)

    unit = normalize_electronic_energy_unit(model.config.get("electronic_energy_unit", "eV"))
    method = str(model.config.get("dos_method", "gaussian"))
    method_tooltip = (
        "Choose the integration used for every candidate mesh. Gaussian "
        "broadening uses the displayed standard deviation. Linear "
        "tetrahedron integration is unbroadened and requires a complete "
        "three-dimensional mesh topology."
    )
    method_combo = QtWidgets.QComboBox()
    method_combo.setObjectName("tight_binding_dos_method")
    method_combo.setToolTip(method_tooltip)
    method_combo.addItem("Gaussian broadening", "gaussian")
    method_combo.addItem("Linear tetrahedron", "tetrahedron")
    method_combo.setCurrentIndex(max(method_combo.findData(method), 0))
    method_label = QtWidgets.QLabel("Integration")
    method_label.setToolTip(method_tooltip)
    layout.addRow(method_label, method_combo)

    broadening_tooltip = (
        f"Gaussian standard deviation in {unit}. It is converted "
        "immediately to canonical meV and held fixed throughout automatic "
        "mesh certification. Tetrahedron integration ignores this value."
    )
    broadening = QtWidgets.QLineEdit(
        _parameter_to_text(
            electronic_energy_from_meV(
                model.config.get("dos_broadening_meV", 5.0),
                unit,
            )
        )
    )
    broadening.setObjectName("model_config_dos_broadening_meV")
    broadening.setToolTip(broadening_tooltip)
    broadening.setEnabled(method == "gaussian")
    broadening.editingFinished.connect(
        lambda editor=broadening: self._set_tight_binding_energy_config(
            "dos_broadening_meV", editor.text()
        )
    )
    broadening_label = QtWidgets.QLabel(f"Gaussian σ ({unit})")
    broadening_label.setToolTip(broadening_tooltip)
    layout.addRow(broadening_label, broadening)

    def set_method(_index: int) -> None:
        selected = str(method_combo.currentData())
        self._set_model_config_setting("dos_method", selected)
        broadening.setEnabled(selected == "gaussian")

    method_combo.currentIndexChanged.connect(set_method)

    symmetry_tooltip = (
        "Choose the Brillouin-zone symmetry policy for DOS evaluation. "
        "Auto uses a certified reduced eigensolve when it is exactly "
        "reconstructible and otherwise falls back to the full mesh. Full "
        "mesh disables reduction. Require reduced mesh raises instead of "
        "falling back. Tetrahedron integration always reconstructs the "
        "complete ordered grid before integration."
    )
    symmetry_combo = QtWidgets.QComboBox()
    symmetry_combo.setObjectName("tight_binding_dos_symmetry")
    symmetry_combo.setToolTip(symmetry_tooltip)
    symmetry_combo.addItem("Auto", "auto")
    symmetry_combo.addItem("Full mesh", "full")
    symmetry_combo.addItem("Require reduced mesh", "reduced")
    symmetry_combo.setCurrentIndex(
        max(
            symmetry_combo.findData(str(model.config.get("dos_symmetry", "auto"))),
            0,
        )
    )
    symmetry_combo.currentIndexChanged.connect(
        lambda _index, combo=symmetry_combo: self._set_model_config_setting(
            "dos_symmetry", str(combo.currentData())
        )
    )
    symmetry_label = QtWidgets.QLabel("Symmetry")
    symmetry_label.setToolTip(symmetry_tooltip)
    layout.addRow(symmetry_label, symmetry_combo)

    if mode == "automatic" and accuracy == "custom":
        custom_tooltip = (
            "Positive normalized DOS error required for two successive mesh refinements."
        )
        custom = QtWidgets.QLineEdit(
            _parameter_to_text(model.config.get("dos_sampling_custom_rtol", 0.01))
        )
        custom.setObjectName("model_config_dos_sampling_custom_rtol")
        custom.setToolTip(custom_tooltip)
        custom.editingFinished.connect(
            lambda editor=custom: self._set_model_config_setting(
                "dos_sampling_custom_rtol", editor.text()
            )
        )
        custom_label = QtWidgets.QLabel("Custom tolerance")
        custom_label.setToolTip(custom_tooltip)
        layout.addRow(custom_label, custom)

    automatic_range = bool(model.config.get("dos_auto_energy_range", False))
    range_tooltip = (
        "Derive fixed DOS limits from the sampled band extrema before "
        "testing mesh convergence. Disable this to certify a user-defined "
        f"energy interval in {unit}."
    )
    range_checkbox = QtWidgets.QCheckBox("Use automatic energy limits")
    range_checkbox.setObjectName("tight_binding_dos_auto_energy_range")
    range_checkbox.setChecked(automatic_range)
    range_checkbox.setToolTip(range_tooltip)
    range_label = QtWidgets.QLabel("Energy range")
    range_label.setToolTip(range_tooltip)
    layout.addRow(range_label, range_checkbox)

    limit_editors = []
    for field, label_text, default in (
        ("dos_energy_min_meV", "Lower energy", -500.0),
        ("dos_energy_max_meV", "Upper energy", 500.0),
    ):
        tooltip = (
            f"Limit of the fixed DOS convergence domain in {unit}. "
            "The value is converted immediately to canonical meV."
        )
        editor = QtWidgets.QLineEdit(
            _parameter_to_text(
                electronic_energy_from_meV(
                    model.config.get(field, default),
                    unit,
                )
            )
        )
        editor.setObjectName(f"model_config_{field}")
        editor.setEnabled(not automatic_range)
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda field=field, editor=editor: self._set_tight_binding_energy_config(
                field, editor.text()
            )
        )
        label = QtWidgets.QLabel(f"{label_text} ({unit})")
        label.setToolTip(tooltip)
        layout.addRow(label, editor)
        limit_editors.append(editor)

    def set_automatic_range(
        checked: bool,
        editors: tuple[Any, ...] = tuple(limit_editors),
    ) -> None:
        self._set_model_config_setting(
            "dos_auto_energy_range",
            "true" if checked else "false",
        )
        for editor in editors:
            editor.setEnabled(not checked)

    range_checkbox.toggled.connect(set_automatic_range)

    points_tooltip = (
        "Number of uniformly spaced energies used to compare successive "
        "DOS curves. More points resolve sharper structure but increase "
        "tetrahedron integration and comparison cost."
    )
    points = QtWidgets.QLineEdit(_parameter_to_text(model.config.get("dos_energy_points", 600)))
    points.setObjectName("model_config_dos_energy_points")
    points.setToolTip(points_tooltip)
    points.editingFinished.connect(
        lambda editor=points: self._set_model_config_setting("dos_energy_points", editor.text())
    )
    points_label = QtWidgets.QLabel("Energy points")
    points_label.setToolTip(points_tooltip)
    layout.addRow(points_label, points)

    mesh_tooltip = (
        "Concrete Brillouin-zone mesh used by DOS calculations. Automatic "
        "certification replaces it only after the requested accuracy passes."
    )
    mesh = QtWidgets.QLineEdit(_parameter_to_text(model.config.get("dos_mesh", [40, 40, 40])))
    mesh.setObjectName("model_config_dos_mesh")
    mesh.setToolTip(mesh_tooltip)
    mesh.setEnabled(mode == "manual")
    mesh.editingFinished.connect(
        lambda editor=mesh: self._set_model_config_setting("dos_mesh", editor.text())
    )
    mesh_label = QtWidgets.QLabel("Production mesh")
    mesh_label.setToolTip(mesh_tooltip)
    layout.addRow(mesh_label, mesh)

    status = QtWidgets.QLabel(self._tight_binding_dos_sampling_status(model))
    status.setObjectName("tight_binding_dos_sampling_status")
    status.setWordWrap(True)
    status.setToolTip(
        "The certificate records every attempted mesh and its observable "
        "errors. Relevant model or DOS-setting changes make it stale."
    )
    layout.addRow(status)

    certify = QtWidgets.QPushButton("Check/refine convergence")
    certify.setObjectName("tight_binding_dos_certify_sampling")
    certify.setEnabled(mode == "automatic")
    certify.setToolTip(
        "Search physically spaced meshes for the configured DOS energy "
        "range, integration method, broadening, and projections."
    )
    certify.clicked.connect(
        lambda _checked=False, model=model: self._certify_tight_binding_dos_sampling(model)
    )
    layout.addRow(certify)
    self.model_parameter_layout.addWidget(group, 12, 0, 1, 4)


def _organize_tight_binding_editor(
    self,
    model: ModelComponentSpec,
) -> None:
    """Collect the long tight-binding workflow into focused tabs."""

    from PySide6 import QtWidgets

    from .electronic_spin import resolved_spin_treatment

    def take(name: str) -> Any | None:
        widget = self.model_parameter_widget.findChild(
            QtWidgets.QWidget,
            name,
        )
        if widget is not None:
            self.model_parameter_layout.removeWidget(widget)
        return widget

    groups = {
        name: take(name)
        for name in (
            "tight_binding_source_group",
            "model_crystal_group",
            "model_crystal_sites_group",
            "tight_binding_orbitals_group",
            "tight_binding_onsite_group",
            "tight_binding_hopping_group",
            "tight_binding_spin_group",
            "tight_binding_state_group",
            "tight_binding_dos_sampling_group",
            "tight_binding_actions_group",
            "model_config_group",
        )
    }

    manifolds = model.config.get("orbital_manifolds") or []
    orbital_count = sum(
        len(item.get("orbitals") or ()) for item in manifolds if isinstance(item, dict)
    )
    try:
        spin = resolved_spin_treatment(
            str(model.config.get("spin_treatment", "auto")),
            model.config.get("soc_terms") or (),
        )
    except ValueError:
        spin = str(model.config.get("spin_treatment", "auto"))
    reduction = "primitive reduction off"
    if bool(model.config.get("use_primitive_cell", True)):
        reduction = "primitive reduction automatic"
        model_data = model.config.get("model_data")
        if isinstance(model_data, dict):
            provenance = model_data.get("provenance")
            if isinstance(provenance, dict):
                detail = provenance.get("primitive_reduction")
                if isinstance(detail, dict):
                    if detail.get("status") == "conventional_fallback":
                        reduction = "conventional-cell fallback"
                    elif "cell_multiplicity" in detail:
                        reduction = (
                            f"primitive cell ({int(detail['cell_multiplicity'])}× reduction)"
                        )
    stale = " · rebuild pending" if model.config.get("model_stale") else ""
    summary = QtWidgets.QGroupBox("Model summary")
    summary.setObjectName("tight_binding_model_summary_group")
    summary_layout = QtWidgets.QVBoxLayout(summary)
    summary_label = QtWidgets.QLabel(
        f"{orbital_count} spatial orbital(s) · {spin} spin · "
        f"{len(model.config.get('onsite_terms') or ())} onsite and "
        f"{len(model.config.get('hopping_terms') or ())} hopping "
        f"coefficient(s) · {reduction}{stale}"
    )
    summary_label.setObjectName("tight_binding_model_summary")
    summary_label.setWordWrap(True)
    summary_label.setToolTip(
        "Resolved compact status of the orbital basis, spin treatment, "
        "active Hamiltonian terms, and cell-reduction policy."
    )
    summary_layout.addWidget(summary_label)

    tabs = QtWidgets.QTabWidget()
    tabs.setObjectName("tight_binding_builder_tabs")
    tabs.setToolTip(
        "Build the structure and basis, define Hamiltonian terms, inspect "
        "electronic results, and access uncommon overrides."
    )

    def add_page(label: str, names: tuple[str, ...]) -> Any:
        page = QtWidgets.QWidget()
        page.setObjectName(f"tight_binding_{label.lower().replace(' ', '_')}_tab")
        page_layout = QtWidgets.QVBoxLayout(page)
        for name in names:
            widget = groups.get(name)
            if widget is not None:
                page_layout.addWidget(widget)
        page_layout.addStretch(1)
        tabs.addTab(page, label)
        return page

    add_page(
        "Structure and basis",
        (
            "tight_binding_source_group",
            "model_crystal_group",
            "model_crystal_sites_group",
            "tight_binding_orbitals_group",
        ),
    )

    hamiltonian_page = QtWidgets.QWidget()
    hamiltonian_page.setObjectName("tight_binding_hamiltonian_tab")
    hamiltonian_layout = QtWidgets.QVBoxLayout(hamiltonian_page)
    term_tabs = QtWidgets.QTabWidget()
    term_tabs.setObjectName("tight_binding_hamiltonian_tabs")
    for label, name in (
        ("Onsite", "tight_binding_onsite_group"),
        ("Hoppings", "tight_binding_hopping_group"),
        ("Spin and SOC", "tight_binding_spin_group"),
    ):
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        widget = groups.get(name)
        if widget is not None:
            page_layout.addWidget(widget)
        page_layout.addStretch(1)
        term_tabs.addTab(page, label)
    hamiltonian_layout.addWidget(term_tabs)
    tabs.addTab(hamiltonian_page, "Hamiltonian")

    add_page(
        "Calculate and inspect",
        (
            "tight_binding_state_group",
            "tight_binding_dos_sampling_group",
            "tight_binding_actions_group",
        ),
    )
    add_page("Advanced", ("model_config_group",))

    self.model_parameter_layout.addWidget(summary, 0, 0, 1, 4)
    self.model_parameter_layout.addWidget(tabs, 1, 0, 1, 4)


def _build_tight_binding_editor(self, model: ModelComponentSpec) -> None:
    """Build source and scriptable plot actions for an electronic model."""

    from PySide6 import QtWidgets

    source_group = QtWidgets.QGroupBox("Model source")
    source_group.setObjectName("tight_binding_source_group")
    source_layout = QtWidgets.QGridLayout(source_group)
    source = str(model.config.get("source_path", "")).strip()
    source_label = QtWidgets.QLabel(Path(source).name if source else "Crystal and orbital builder")
    source_label.setObjectName("tight_binding_source_summary")
    source_label.setToolTip(
        source
        or (
            "This model is constructed from the editable crystal, orbital "
            "manifolds, and Hamiltonian terms below."
        )
    )
    source_layout.addWidget(source_label, 0, 0, 1, 3)

    import_button = QtWidgets.QPushButton("Import Wannier90...")
    import_button.setObjectName("tight_binding_import_wannier90")
    import_button.setToolTip(
        "Import seedname_tb.dat, or seedname_hr.dat with its associated "
        ".win, centres, and optional wsvec files. Wannier90 Hamiltonians "
        "are read in eV, converted once to canonical meV, and retain the "
        "conversion in model provenance."
    )
    import_button.clicked.connect(
        lambda _checked=False, model=model: self._import_wannier90_model(model)
    )
    source_layout.addWidget(import_button, 1, 0, 1, 2)
    structure_script = QtWidgets.QPushButton("Copy builder script")
    structure_script.setObjectName("tight_binding_structure_script")
    structure_script.setToolTip(
        "Copy editable Python that rebuilds the crystal, orbital manifolds, "
        "onsite, hopping, spin, and SOC terms, periodic axes, and canonical "
        "model."
    )
    structure_script.clicked.connect(
        lambda _checked=False, model=model: self._copy_tight_binding_structure_script(model)
    )
    source_layout.addWidget(structure_script, 1, 2)
    self.model_parameter_layout.addWidget(source_group, 9, 0, 1, 4)

    group = QtWidgets.QGroupBox("Electronic structure")
    group.setObjectName("tight_binding_actions_group")
    layout = QtWidgets.QGridLayout(group)

    zone_button = QtWidgets.QPushButton("View Brillouin zone in 3D")
    zone_button.setObjectName("tight_binding_brillouin_zone")
    zone_button.setToolTip(
        "Show the first Brillouin-zone polyhedron, the configured labelled "
        "band path, and primitive reciprocal basis vectors b₁, b₂, and b₃. "
        "The viewer includes scriptable styling, visibility, projection, "
        "copy, and save controls."
    )
    zone_button.clicked.connect(
        lambda _checked=False, model=model: self._open_brillouin_zone(model)
    )
    zone_script = QtWidgets.QPushButton("Copy Brillouin-zone script")
    zone_script.setObjectName("tight_binding_brillouin_zone_script")
    zone_script.setToolTip(
        "Copy editable Python that reconstructs the same labelled 3D "
        "Brillouin-zone scene without project widgets."
    )
    zone_script.clicked.connect(
        lambda _checked=False, model=model: self._copy_brillouin_zone_script(model)
    )
    layout.addWidget(zone_button, 0, 0, 1, 2)
    layout.addWidget(zone_script, 0, 2)

    matrix_button = QtWidgets.QPushButton("Inspect matrices")
    matrix_button.setObjectName("tight_binding_matrix_inspector")
    matrix_button.setToolTip(
        "Inspect H(k), named Hamiltonian terms, spin operators, exact "
        "matrix elements, and orbital-subspace block norms."
    )
    matrix_button.clicked.connect(
        lambda _checked=False, model=model: self._open_electronic_matrix_inspector(model)
    )
    matrix_script = QtWidgets.QPushButton("Copy matrix-viewer script")
    matrix_script.setObjectName("tight_binding_matrix_script")
    matrix_script.setToolTip(
        "Copy editable Python that reconstructs the electronic matrix "
        "catalog and opens the standalone inspector."
    )
    matrix_script.clicked.connect(
        lambda _checked=False, model=model: self._copy_electronic_matrix_script(model)
    )
    definition = model_definition(model.type)
    for row, plot in enumerate(definition.plots, start=1):
        calculate = QtWidgets.QPushButton(plot.label)
        calculate.setObjectName(f"model_plot_{plot.key}")
        calculate.setToolTip(plot.description)
        calculate.clicked.connect(
            lambda _checked=False, model=model, key=plot.key: self._open_model_plot(model, key)
        )
        copy_script = QtWidgets.QPushButton("Copy script")
        copy_script.setObjectName(f"model_plot_script_{plot.key}")
        copy_script.setToolTip(
            f"Copy editable Python that reproduces the {plot.label.lower()} calculation."
        )
        copy_script.clicked.connect(
            lambda _checked=False, model=model, key=plot.key: self._copy_model_plot_script(
                model, key
            )
        )
        layout.addWidget(calculate, row, 0, 1, 2)
        layout.addWidget(copy_script, row, 2)
    matrix_row = 1 + len(definition.plots)
    layout.addWidget(matrix_button, matrix_row, 0, 1, 2)
    layout.addWidget(matrix_script, matrix_row, 2)
    self.model_parameter_layout.addWidget(group, 10, 0, 1, 4)


def _build_tight_binding_orbital_editor(self, model: ModelComponentSpec) -> None:
    """Build editable site-attached orbital manifolds."""

    from PySide6 import QtWidgets

    from .electronic_builder import (
        ORBITAL_PRESETS,
        OrbitalManifold,
        site_point_group_symbol,
        site_symmetry_harmonic_submanifolds,
    )

    group = QtWidgets.QGroupBox("Orbitals")
    group.setObjectName("tight_binding_orbitals_group")
    layout = QtWidgets.QGridLayout(group)
    tooltip = (
        "Attach an ordered orbital manifold to a crystallographic site. "
        "Analytic spherical harmonics transform in the displayed local "
        "frame. Custom bases remain usable but automatic symmetry is "
        "disabled unless representation matrices are supplied in a later stage."
    )
    site_combo = QtWidgets.QComboBox()
    site_combo.setObjectName("tight_binding_orbital_site")
    site_combo.setToolTip(tooltip)
    for site in model_crystal_config(model).get("sites", ()):
        label = str(site.get("label", ""))
        if label:
            site_combo.addItem(label, label)
    preset_combo = QtWidgets.QComboBox()
    preset_combo.setObjectName("tight_binding_orbital_preset")
    preset_combo.setToolTip(
        "Choose an effective scalar, a complete real spherical-harmonic "
        "s, p, d, or f shell, or a custom numerical basis."
    )
    for preset in ORBITAL_PRESETS:
        preset_combo.addItem(preset, preset)
    symmetry_label = QtWidgets.QLabel()
    symmetry_label.setObjectName("tight_binding_orbital_site_symmetry")
    symmetry_label.setToolTip(
        "The crystallographic point group formed by the space-group "
        "operations that leave the selected site fixed."
    )
    use_site_symmetry = QtWidgets.QCheckBox("Select a site-symmetry subspace")
    use_site_symmetry.setObjectName("tight_binding_orbital_use_site_symmetry")
    use_site_symmetry.setToolTip(
        "Split the complete harmonic shell using the selected site's "
        "actual point-group representation, then add one symmetry-closed "
        "subspace. This is available for s, p, d, and f shells."
    )
    submanifold_combo = QtWidgets.QComboBox()
    submanifold_combo.setObjectName("tight_binding_orbital_submanifold")
    submanifold_combo.setToolTip(
        "Choose a symmetry-closed subspace calculated from the site point "
        "group in the crystal Cartesian frame. The dimension and dominant "
        "complete-shell orbitals are shown for identification."
    )
    symmetry_cache: dict[tuple[str, str], tuple[Any, ...]] = {}

    def refresh_site_symmetry(*_args: Any) -> None:
        site_label = str(site_combo.currentData() or "")
        shell = str(preset_combo.currentData() or "")
        available = bool(site_label and shell in {"s", "p", "d", "f"})
        use_site_symmetry.setEnabled(available)
        if not available:
            use_site_symmetry.setChecked(False)
        submanifold_combo.clear()
        if not site_label:
            symmetry_label.setText("Site symmetry: select a site")
        else:
            try:
                point_group = site_point_group_symbol(
                    model_crystal_config(model),
                    site_label,
                )
                symmetry_label.setText(f"Site symmetry: {point_group}")
                if available:
                    cache_key = (site_label, shell)
                    if cache_key not in symmetry_cache:
                        symmetry_cache[cache_key] = site_symmetry_harmonic_submanifolds(
                            model_crystal_config(model),
                            site_label,
                            shell,
                        )
                    for option in symmetry_cache[cache_key]:
                        submanifold_combo.addItem(
                            option.label,
                            option.identifier,
                        )
            except (RuntimeError, ValueError) as exc:
                symmetry_label.setText(f"Site symmetry unavailable: {exc}")
                use_site_symmetry.setChecked(False)
                use_site_symmetry.setEnabled(False)
        submanifold_combo.setEnabled(
            use_site_symmetry.isChecked() and submanifold_combo.count() > 0
        )

    site_combo.currentIndexChanged.connect(refresh_site_symmetry)
    preset_combo.currentIndexChanged.connect(refresh_site_symmetry)
    use_site_symmetry.toggled.connect(refresh_site_symmetry)
    add_button = QtWidgets.QPushButton("Add manifold")
    add_button.setObjectName("tight_binding_orbital_add")
    add_button.setToolTip(
        "Add the complete selected shell, or the selected point-group "
        "subspace when site-symmetry selection is enabled. The crystal "
        "Cartesian frame is the default local frame."
    )
    add_button.setEnabled(site_combo.count() > 0)
    add_button.clicked.connect(
        lambda _checked=False, sites=site_combo, presets=preset_combo, use_symmetry=use_site_symmetry, submanifolds=submanifold_combo: (
            self._add_tight_binding_manifold(
                str(sites.currentData() or ""),
                str(presets.currentData() or "effective"),
                (str(submanifolds.currentData() or "") if use_symmetry.isChecked() else None),
            )
        )
    )
    refresh_site_symmetry()
    layout.addWidget(QtWidgets.QLabel("Site"), 0, 0)
    layout.addWidget(site_combo, 0, 1)
    layout.addWidget(QtWidgets.QLabel("Shell or basis"), 0, 2)
    layout.addWidget(preset_combo, 0, 3)
    layout.addWidget(add_button, 0, 4)
    layout.addWidget(symmetry_label, 1, 0, 1, 2)
    layout.addWidget(use_site_symmetry, 1, 2, 1, 2)
    layout.addWidget(submanifold_combo, 1, 4, 1, 4)

    headers = (
        "Site",
        "Manifold",
        "Basis",
        "Orbitals",
        "Magnetic form factors",
        "Local frame",
        "Degeneracy groups",
        "Correlated shell",
        "",
    )
    for column, text in enumerate(headers):
        layout.addWidget(QtWidgets.QLabel(text), 2, column)
    manifolds = [
        OrbitalManifold.from_dict(item) for item in model.config.get("orbital_manifolds", ())
    ]
    for index, manifold in enumerate(manifolds):
        row = index + 3
        site = QtWidgets.QLabel(manifold.site_label)
        site.setToolTip(tooltip)
        layout.addWidget(site, row, 0)
        label = QtWidgets.QLineEdit(manifold.label)
        label.setObjectName(f"tight_binding_manifold_label_{index}")
        label.setToolTip(
            "Stable, unique manifold label used in basis-state names, "
            "projection groups, scripts, and later interaction definitions."
        )
        label.editingFinished.connect(
            lambda index=index, editor=label: self._set_tight_binding_manifold_field(
                index, "label", editor.text()
            )
        )
        layout.addWidget(label, row, 1)
        basis = QtWidgets.QLabel(f"{manifold.preset} ({manifold.basis_kind})")
        basis.setToolTip(
            "The shell or basis choice is a convenience, while basis_kind "
            "determines how symmetry acts. A site-symmetry entry is a "
            "calculated subspace of the named complete shell."
        )
        layout.addWidget(basis, row, 2)
        orbitals = QtWidgets.QLineEdit(_parameter_to_text(list(manifold.orbitals)))
        orbitals.setObjectName(f"tight_binding_manifold_orbitals_{index}")
        orbitals.setReadOnly(manifold.basis_kind != "custom")
        orbitals.setToolTip(
            "Ordered basis labels. Built-in harmonic labels are fixed by "
            "their transformation convention; custom labels are editable JSON."
        )
        orbitals.editingFinished.connect(
            lambda index=index, editor=orbitals: self._set_tight_binding_manifold_field(
                index, "orbitals", editor.text()
            )
        )
        layout.addWidget(orbitals, row, 3)
        from .form_factors import compact_form_factor_profile

        form_factors = QtWidgets.QLineEdit(
            _parameter_to_text(
                {
                    orbital: compact_form_factor_profile(profile)
                    for orbital, profile in manifold.magnetic_form_factors.items()
                }
            )
        )
        form_factors.setObjectName(f"tight_binding_manifold_form_factors_{index}")
        form_factors.setToolTip(
            "JSON mapping from orbital label to its magnetic radial profile. "
            'Use an ion string such as {"d_xy": "V3"}, or a full '
            "profile mapping with form_factor_mode and custom/mixture fields. "
            "An omitted orbital has unit probe amplitude. These amplitudes "
            "enter before orbital interference is contracted. Electronic "
            "RPA with orbital-specific profiles currently requires an "
            "implicit-spin tight-binding model."
        )
        form_factors.editingFinished.connect(
            lambda index=index, editor=form_factors: self._set_tight_binding_manifold_field(
                index, "magnetic_form_factors", editor.text()
            )
        )
        layout.addWidget(form_factors, row, 4)
        frame = QtWidgets.QLineEdit(_parameter_to_text(np.asarray(manifold.local_frame).tolist()))
        frame.setObjectName(f"tight_binding_manifold_frame_{index}")
        frame.setToolTip(
            "Right-handed orthonormal local axes as a 3x3 JSON matrix whose "
            "columns are local x, y, z in crystal Cartesian coordinates. "
            "Identity is the crystal frame."
        )
        frame.editingFinished.connect(
            lambda index=index, editor=frame: self._set_tight_binding_manifold_field(
                index, "local_frame", editor.text()
            )
        )
        layout.addWidget(frame, row, 5)
        degeneracy = QtWidgets.QLineEdit(
            _parameter_to_text([list(group) for group in manifold.degeneracy_groups])
        )
        degeneracy.setObjectName(f"tight_binding_manifold_degeneracy_{index}")
        degeneracy.setToolTip(
            "Optional JSON groups constrained to share one diagonal onsite "
            'energy, for example [["d_yz", "d_zx"]]. Groups must remain '
            "compatible with the selected symmetry representation."
        )
        degeneracy.editingFinished.connect(
            lambda index=index, editor=degeneracy: self._set_tight_binding_manifold_field(
                index, "degeneracy_groups", editor.text()
            )
        )
        layout.addWidget(degeneracy, row, 6)
        shell = QtWidgets.QLineEdit(manifold.correlated_shell)
        shell.setObjectName(f"tight_binding_manifold_shell_{index}")
        shell.setToolTip(
            "Optional correlated-shell identifier that later Hubbard-Hund "
            "and other interaction dressings will use."
        )
        shell.editingFinished.connect(
            lambda index=index, editor=shell: self._set_tight_binding_manifold_field(
                index, "correlated_shell", editor.text()
            )
        )
        layout.addWidget(shell, row, 7)
        remove = QtWidgets.QPushButton("Remove")
        remove.setObjectName(f"tight_binding_manifold_remove_{index}")
        remove.setToolTip("Remove this manifold and regenerate the remaining onsite basis.")
        remove.clicked.connect(
            lambda _checked=False, label=manifold.label: self._remove_tight_binding_manifold(label)
        )
        layout.addWidget(remove, row, 8)
    status_text = (
        f"{sum(item.dimension for item in manifolds)} orbital(s) in {len(manifolds)} manifold(s)."
        if manifolds
        else "Add an orbital manifold to begin constructing the electronic basis."
    )
    status = QtWidgets.QLabel(status_text)
    status.setObjectName("tight_binding_orbital_status")
    status.setToolTip(tooltip)
    status.setWordWrap(True)
    layout.addWidget(status, len(manifolds) + 3, 0, 1, len(headers))
    self.model_parameter_layout.addWidget(group, 5, 0, 1, 4)


def _build_tight_binding_spin_editor(
    self,
    model: ModelComponentSpec,
) -> None:
    """Build compact opt-in spin and onsite SOC controls."""

    from PySide6 import QtWidgets

    from .electronic_builder import OrbitalManifold
    from .electronic_spin import SPIN_TREATMENTS, SpinOrbitTerm
    from .electronic_structure import electronic_energy_from_meV

    group = QtWidgets.QGroupBox("Spin and SOC")
    group.setObjectName("tight_binding_spin_group")
    group.setToolTip(
        "Spin remains an implicit twofold degeneracy unless collinear or "
        "spinor structure is requested. Active SOC promotes Auto to the "
        "full spinor basis."
    )
    layout = QtWidgets.QGridLayout(group)
    treatment_label = QtWidgets.QLabel("Spin treatment")
    treatment = QtWidgets.QComboBox()
    treatment.setObjectName("tight_binding_spin_treatment")
    treatment.setToolTip(
        "Auto keeps an N-orbital SU(2)-symmetric model implicit and only "
        "creates 2N basis states when SOC requires spin mixing. Collinear "
        "and spinor explicitly force the corresponding representation."
    )
    treatment_labels = {
        "auto": "Auto (efficient)",
        "implicit": "Implicit spin degeneracy",
        "collinear": "Explicit collinear",
        "spinor": "Full spinor",
    }
    for option in SPIN_TREATMENTS:
        treatment.addItem(treatment_labels[option], option)
    selected = str(model.config.get("spin_treatment", "auto"))
    treatment.setCurrentIndex(max(0, treatment.findData(selected)))
    treatment.currentIndexChanged.connect(
        lambda _index, combo=treatment: self._set_tight_binding_spin_treatment(
            str(combo.currentData())
        )
    )
    resolved = ""
    model_data = model.config.get("model_data")
    if isinstance(model_data, dict):
        provenance = model_data.get("provenance", {})
        if isinstance(provenance, dict):
            resolved = str(provenance.get("resolved_spin_treatment", ""))
    if bool(model.config.get("model_stale", False)):
        resolved_text = "pending lazy rebuild"
    else:
        resolved_text = resolved or "not built"
    resolved_label = QtWidgets.QLabel(f"Resolved: {resolved_text}")
    resolved_label.setObjectName("tight_binding_spin_resolved")
    resolved_label.setToolTip(
        "The actual representation used by the canonical model. Implicit "
        "uses N basis states; collinear and spinor use 2N."
    )
    layout.addWidget(treatment_label, 0, 0)
    layout.addWidget(treatment, 0, 1, 1, 2)
    layout.addWidget(resolved_label, 0, 3, 1, 3)

    headers = (
        "Manifold",
        "Spatial orbitals",
        "SOC",
        "Prescription",
        "λ",
        "Lower",
        "Upper",
        "Fit",
        "Sharing",
        "Groups",
    )
    for column, text in enumerate(headers):
        label = QtWidgets.QLabel(text)
        label.setToolTip(group.toolTip())
        layout.addWidget(label, 1, column)
    manifolds = [
        OrbitalManifold.from_dict(item) for item in model.config.get("orbital_manifolds", ())
    ]
    terms = {
        term.manifold_label: term
        for term in (SpinOrbitTerm.from_dict(item) for item in model.config.get("soc_terms", ()))
    }
    unit = str(model.config.get("electronic_energy_unit", "eV"))
    for index, manifold in enumerate(manifolds):
        row = index + 2
        term = terms.get(manifold.label)
        supported = (
            manifold.basis_kind in {"real_harmonic", "complex_harmonic"}
            and manifold.l is not None
            and manifold.l > 0
        ) or (term is not None and term.orbital_operators is not None)
        manifold_name = QtWidgets.QLabel(manifold.label)
        manifold_name.setToolTip(
            "SOC is attached once to this spatial manifold and repeated "
            "on every symmetry-equivalent site."
        )
        layout.addWidget(manifold_name, row, 0)
        orbitals = QtWidgets.QLabel(f"{manifold.dimension}: {', '.join(manifold.orbitals)}")
        orbitals.setToolTip(
            "Spatial orbitals before optional spinor expansion. The 3D "
            "geometry viewer does not duplicate these tokens for spin."
        )
        layout.addWidget(orbitals, row, 1)
        enabled = QtWidgets.QCheckBox()
        enabled.setObjectName(f"tight_binding_soc_enabled_{index}")
        enabled.setChecked(term is not None)
        enabled.setEnabled(supported)
        enabled.setToolTip(
            "Enable λ L·S for this manifold. Complete harmonic shells use "
            "atomic L; symmetry-selected subspaces use projected P L P. "
            "Custom bases require explicit operators through the API."
        )
        enabled.toggled.connect(
            lambda checked, label=manifold.label: self._toggle_tight_binding_soc(
                label,
                bool(checked),
            )
        )
        layout.addWidget(enabled, row, 2)
        prescription = QtWidgets.QComboBox()
        prescription.setObjectName(f"tight_binding_soc_prescription_{index}")
        prescription.addItem("Auto", "auto")
        prescription.addItem("Atomic shell", "atomic")
        prescription.addItem("Projected subspace", "projected")
        prescription.addItem("Effective operators", "effective")
        prescription.setCurrentIndex(
            max(
                0,
                prescription.findData("auto" if term is None else term.prescription),
            )
        )
        prescription.setEnabled(term is not None)
        prescription.setToolTip(
            "Atomic requires a complete l shell. Projected uses P(L·S)P "
            "and may omit virtual coupling to excluded orbitals. Effective "
            "requires explicitly supplied Lx, Ly, and Lz matrices."
        )
        prescription.currentIndexChanged.connect(
            lambda _index, label=manifold.label, combo=prescription: (
                self._set_tight_binding_soc_field(
                    label,
                    "prescription",
                    str(combo.currentData()),
                )
            )
        )
        layout.addWidget(prescription, row, 3)
        value = QtWidgets.QLineEdit(
            ""
            if term is None
            else _parameter_to_text(electronic_energy_from_meV(term.value_meV, unit))
        )
        value.setObjectName(f"tight_binding_soc_value_{index}")
        value.setEnabled(term is not None)
        value.setToolTip(
            f"Spin-orbit coupling λ in {unit}; converted immediately to canonical meV."
        )
        value.editingFinished.connect(
            lambda label=manifold.label, editor=value: self._set_tight_binding_soc_field(
                label,
                "value",
                editor.text(),
            )
        )
        layout.addWidget(value, row, 4)
        for column, (field, bound) in enumerate(
            zip(
                ("lower", "upper"),
                (None, None) if term is None else term.bounds_meV,
                strict=True,
            ),
            start=5,
        ):
            editor = QtWidgets.QLineEdit(
                "" if bound is None else _parameter_to_text(electronic_energy_from_meV(bound, unit))
            )
            editor.setObjectName(f"tight_binding_soc_{field}_{index}")
            editor.setEnabled(term is not None)
            editor.setToolTip(f"Optional fit bound for λ in {unit}; empty is unbounded.")
            editor.editingFinished.connect(
                lambda label=manifold.label, field=field, editor=editor: (
                    self._set_tight_binding_soc_field(
                        label,
                        field,
                        editor.text(),
                    )
                )
            )
            layout.addWidget(editor, row, column)
        fit = QtWidgets.QCheckBox()
        fit.setObjectName(f"tight_binding_soc_fit_{index}")
        fit.setChecked(False if term is None else term.fit)
        fit.setEnabled(term is not None)
        fit.setToolTip("Allow a compatible future electronic-response fit to vary λ.")
        fit.toggled.connect(
            lambda checked, label=manifold.label: self._set_tight_binding_soc_fit(
                label,
                bool(checked),
            )
        )
        layout.addWidget(fit, row, 7)
        if term is None:
            sharing = QtWidgets.QLabel("—")
            groups = QtWidgets.QLabel("")
        else:
            sharing, groups = self._tight_binding_sharing_controls(
                model,
                term.identifier,
                object_prefix=f"tight_binding_soc_{index}",
            )
        layout.addWidget(sharing, row, 8)
        layout.addWidget(groups, row, 9)
    status = QtWidgets.QLabel(
        "Implicit spin avoids redundant diagonalization. SOC, transverse "
        "spin mixing, and noncollinear extensions use the full spinor basis."
    )
    status.setObjectName("tight_binding_spin_status")
    status.setWordWrap(True)
    status.setToolTip(group.toolTip())
    layout.addWidget(status, len(manifolds) + 2, 0, 1, len(headers))
    self.model_parameter_layout.addWidget(group, 6, 0, 1, 4)


def _tight_binding_sharing_controls(
    self,
    model: ModelComponentSpec,
    identifier: str,
    *,
    object_prefix: str,
) -> tuple[Any, Any]:
    """Build common sharing controls for one orbital-aware parameter row."""

    from PySide6 import QtWidgets

    sharing = QtWidgets.QComboBox()
    sharing.setObjectName(f"{object_prefix}_sharing")
    sharing.addItem("Global", "global")
    sharing.addItem("Per dataset", "per_dataset")
    sharing.addItem("Grouped", "grouped")
    mode = sharing_mode(model, identifier)
    sharing.setCurrentIndex(max(sharing.findData(mode), 0))
    sharing.setToolTip(
        "Choose whether a compatible response fit uses one coefficient for "
        "all datasets, one per dataset, or values tied by named groups."
    )
    groups = QtWidgets.QLineEdit(_sharing_groups_text(model, identifier))
    groups.setObjectName(f"{object_prefix}_sharing_groups")
    groups.setPlaceholderText("scan1=A, scan2=A")
    groups.setVisible(mode == "grouped")
    groups.setToolTip(
        "For grouped sharing, enter comma-separated dataset=group pairs. "
        "Unlisted datasets remain independent."
    )
    sharing.currentIndexChanged.connect(
        lambda _index, identifier=identifier, combo=sharing: self._set_model_sharing_mode(
            identifier,
            str(combo.currentData()),
        )
    )
    groups.editingFinished.connect(
        lambda identifier=identifier, editor=groups: self._set_model_sharing_groups(
            identifier,
            editor.text(),
        )
    )
    return sharing, groups


def _build_tight_binding_onsite_editor(self, model: ModelComponentSpec) -> None:
    """Build generated onsite-energy and onsite-hybridization controls."""

    from PySide6 import QtWidgets

    from .electronic_builder import OnsiteInvariant
    from .electronic_structure import electronic_energy_from_meV

    group = QtWidgets.QGroupBox("Onsite terms")
    group.setObjectName("tight_binding_onsite_group")
    layout = QtWidgets.QGridLayout(group)
    unit = str(model.config.get("electronic_energy_unit", "eV"))
    terms = [OnsiteInvariant.from_dict(item) for item in model.config.get("onsite_terms", ())]
    headers = (
        "Term",
        "Site",
        "Kind",
        f"Value ({unit})",
        f"Lower ({unit})",
        f"Upper ({unit})",
        "Fit",
        "Sharing",
        "Groups",
        "Matrix basis",
    )
    for column, text in enumerate(headers):
        header = QtWidgets.QLabel(text)
        layout.addWidget(header, 0, column)
    for index, term in enumerate(terms):
        row = index + 1
        layout.addWidget(QtWidgets.QLabel(term.label), row, 0)
        layout.addWidget(QtWidgets.QLabel(term.site_label), row, 1)
        layout.addWidget(QtWidgets.QLabel(term.kind), row, 2)
        value = QtWidgets.QLineEdit(
            _parameter_to_text(electronic_energy_from_meV(term.value_meV, unit))
        )
        value.setObjectName(f"tight_binding_onsite_value_{index}")
        value.setToolTip(
            f"Static one-electron onsite coefficient in {unit}. It is "
            "converted immediately to canonical meV. This is distinct from "
            "a future dynamical many-body self-energy Sigma(k,E)."
        )
        value.editingFinished.connect(
            lambda identifier=term.identifier, editor=value: self._set_tight_binding_onsite_field(
                identifier, "value", editor.text()
            )
        )
        layout.addWidget(value, row, 3)
        for column, (field, bound) in enumerate(
            zip(("lower", "upper"), term.bounds_meV, strict=True), start=4
        ):
            editor = QtWidgets.QLineEdit(
                "" if bound is None else _parameter_to_text(electronic_energy_from_meV(bound, unit))
            )
            editor.setObjectName(f"tight_binding_onsite_{field}_{index}")
            editor.setToolTip(
                f"Optional fit bound in {unit}. Empty means unbounded. "
                "The value is stored in the component's common bounds state."
            )
            editor.editingFinished.connect(
                lambda identifier=term.identifier, field=field, editor=editor: (
                    self._set_tight_binding_onsite_field(identifier, field, editor.text())
                )
            )
            layout.addWidget(editor, row, column)
        fit = QtWidgets.QCheckBox()
        fit.setObjectName(f"tight_binding_onsite_fit_{index}")
        fit.setChecked(term.fit)
        fit.setToolTip(
            "Vary this coefficient when a compatible electronic-response "
            "component supplies a dataset observable. Tight binding alone "
            "remains calculation-only."
        )
        fit.toggled.connect(
            lambda checked, identifier=term.identifier: self._set_tight_binding_onsite_fit(
                identifier, checked
            )
        )
        layout.addWidget(fit, row, 6)
        sharing, groups = self._tight_binding_sharing_controls(
            model,
            term.identifier,
            object_prefix=f"tight_binding_onsite_{index}",
        )
        layout.addWidget(sharing, row, 7)
        layout.addWidget(groups, row, 8)
        matrix = QtWidgets.QLabel(
            f"{len(term.basis_labels)}x{len(term.basis_labels)}; {term.source}"
        )
        matrix.setToolTip(
            "Unit-Frobenius Hermitian matrix multiplying this coefficient. "
            f"Ordered local basis: {', '.join(term.basis_labels)}\n"
            f"Matrix:\n{np.array2string(term.matrix, precision=4)}"
        )
        layout.addWidget(matrix, row, 9)
    status_text = (
        f"{len(terms)} onsite invariant(s)."
        if terms
        else "No onsite invariants. Add orbitals, then regenerate."
    )
    status = QtWidgets.QLabel(status_text)
    status.setObjectName("tight_binding_onsite_status")
    status.setToolTip(
        "Onsite energies and symmetry-allowed onsite hybridizations form "
        "the static R=0 Hamiltonian."
    )
    status.setWordWrap(True)
    layout.addWidget(status, len(terms) + 1, 0, 1, len(headers))
    self.model_parameter_layout.addWidget(group, 7, 0, 1, 4)


def _build_tight_binding_hopping_editor(self, model: ModelComponentSpec) -> None:
    """Build generated hopping suggestions and active-term controls."""

    from PySide6 import QtCore, QtWidgets

    from .electronic_builder import (
        HoppingInvariant,
        hopping_endpoint_orbitals,
    )
    from .electronic_structure import electronic_energy_from_meV

    group = QtWidgets.QGroupBox("Hoppings")
    group.setObjectName("tight_binding_hopping_group")
    layout = QtWidgets.QGridLayout(group)
    parameterization_label = QtWidgets.QLabel("Parameterization")
    parameterization = QtWidgets.QComboBox()
    parameterization.setObjectName("tight_binding_hopping_parameterization")
    parameterization.addItem(
        "Slater–Koster integrals",
        "slater_koster",
    )
    parameterization.addItem(
        "General symmetry matrices",
        "general",
    )
    parameterization.setCurrentIndex(
        max(
            0,
            parameterization.findData(
                str(
                    model.config.get(
                        "hopping_parameterization",
                        "slater_koster",
                    )
                )
            ),
        )
    )
    parameterization_tooltip = (
        "Slater–Koster generates compact V_ll′σ, V_ll′π, V_ll′δ, and "
        "V_ll′φ coefficients for real-harmonic s, p, d, and f manifolds, "
        "including selected crystal-field subspaces and their local "
        "frames. General symmetry matrices span every real spinless "
        "hopping allowed by the bond stabilizer and support arbitrary "
        "declared bases. Changing convention regenerates suggestions; "
        "terms without the same stable identifier become inactive."
    )
    parameterization_label.setToolTip(parameterization_tooltip)
    parameterization.setToolTip(parameterization_tooltip)
    parameterization.currentIndexChanged.connect(
        lambda _index, combo=parameterization: self._set_tight_binding_hopping_parameterization(
            str(combo.currentData())
        )
    )
    layout.addWidget(parameterization_label, 0, 0)
    layout.addWidget(parameterization, 0, 1, 1, 3)
    cutoff_label = QtWidgets.QLabel("Cutoff (Å)")
    cutoff = QtWidgets.QLineEdit(
        _parameter_to_text(float(model.config.get("hopping_cutoff_angstrom", 0.0)) or 5.0)
    )
    cutoff.setObjectName("tight_binding_hopping_cutoff")
    cutoff_tooltip = (
        "Maximum real-space distance used to generate symmetry-distinct "
        "bonds between every orbital-bearing site."
    )
    cutoff_label.setToolTip(cutoff_tooltip)
    cutoff.setToolTip(cutoff_tooltip)
    generate = QtWidgets.QPushButton("Generate suggestions")
    generate.setObjectName("tight_binding_hopping_generate")
    generate.setToolTip(
        "Generate spatial bond orbits and hopping candidates in the "
        "selected parameterization. Candidates do not enter the "
        "Hamiltonian until selected below."
    )
    generate.setEnabled(bool(model.config.get("orbital_manifolds")))
    generate.clicked.connect(
        lambda _checked=False, editor=cutoff: self._regenerate_tight_binding_hoppings(editor.text())
    )
    layout.addWidget(cutoff_label, 1, 0)
    layout.addWidget(cutoff, 1, 1)
    layout.addWidget(generate, 1, 2, 1, 2)

    multiplicities = {
        str(item.get("label", "")): len(item.get("bonds", ()))
        for item in model.config.get("spatial_orbits", ())
    }
    sites = list(model.config.get("expanded_crystal_sites", ()))

    def endpoint_text(
        term: HoppingInvariant,
    ) -> tuple[str, str]:
        from_orbitals, to_orbitals = hopping_endpoint_orbitals(term)
        bond = term.representative_bond
        from_site = (
            str(sites[bond.site_j].get("label", f"site {bond.site_j}"))
            if bond.site_j < len(sites)
            else f"site {bond.site_j}"
        )
        to_site = (
            str(sites[bond.site_i].get("label", f"site {bond.site_i}"))
            if bond.site_i < len(sites)
            else f"site {bond.site_i}"
        )
        offset = tuple(int(value) for value in bond.offset)
        return (
            f"{from_site} + {offset}: {', '.join(from_orbitals)}",
            f"{to_site}: {', '.join(to_orbitals)}",
        )

    active = [HoppingInvariant.from_dict(item) for item in model.config.get("hopping_terms", ())]
    active_ids = {item.identifier for item in active}
    candidates = [
        HoppingInvariant.from_dict(item)
        for item in model.config.get("hopping_candidates", ())
        if str(item.get("identifier", "")) not in active_ids
    ]
    suggestion_headers = (
        "Term",
        "Orbit",
        "Distance (Å)",
        "Multiplicity",
        "From orbital(s)",
        "To orbital(s)",
        "Matrix basis",
    )
    suggestion_table = QtWidgets.QTableWidget(
        len(candidates),
        len(suggestion_headers),
    )
    suggestion_table.setObjectName("tight_binding_hopping_suggestions")
    suggestion_table.setToolTip(
        "Symmetry-allowed candidates. T_ij(R) maps the listed source "
        "orbital(s) in cell R to the listed destination orbital(s) in the "
        "home cell. Select rows and add only the terms required by the model."
    )
    suggestion_table.setHorizontalHeaderLabels(suggestion_headers)
    suggestion_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    suggestion_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    suggestion_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    for row, term in enumerate(candidates):
        from_text, to_text = endpoint_text(term)
        values = (
            term.label,
            term.orbit_label,
            f"{term.distance_angstrom:.5g}",
            str(multiplicities.get(term.orbit_label, 0)),
            from_text,
            to_text,
            f"{len(term.basis_i)}x{len(term.basis_j)}",
        )
        for column, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                term.identifier,
            )
            if column == 6:
                item.setToolTip(
                    "Unit-Frobenius representative matrix:\n"
                    f"{np.array2string(term.matrix, precision=4)}"
                )
            suggestion_table.setItem(row, column, item)
    suggestion_table.resizeColumnsToContents()
    suggestion_table.resizeRowsToContents()
    visible_rows = min(max(len(candidates), 3), 9)
    row_height = max(
        suggestion_table.verticalHeader().defaultSectionSize(),
        24,
    )
    table_height = (
        suggestion_table.horizontalHeader().height()
        + visible_rows * row_height
        + 2 * suggestion_table.frameWidth()
        + 8
    )
    suggestion_table.setMinimumHeight(table_height)
    suggestion_table.setMaximumHeight(table_height)
    layout.addWidget(suggestion_table, 2, 0, 1, 10)
    add_selected = QtWidgets.QPushButton("Add selected hopping terms")
    add_selected.setObjectName("tight_binding_hopping_add_selected")
    add_selected.setToolTip(
        "Add the selected generated matrix terms to the active tight-binding Hamiltonian."
    )
    add_selected.setEnabled(bool(candidates))
    add_selected.clicked.connect(
        lambda _checked=False, table=suggestion_table: self._add_selected_tight_binding_hoppings(
            table
        )
    )
    layout.addWidget(add_selected, 3, 0, 1, 3)

    unit = str(model.config.get("electronic_energy_unit", "eV"))
    active_headers = (
        "Active term",
        "From orbital(s)",
        "To orbital(s)",
        "Distance (Å)",
        f"Value ({unit})",
        f"Lower ({unit})",
        f"Upper ({unit})",
        "Fit",
        "Sharing",
        "Groups",
        "Matrix basis",
        "",
    )
    for column, text in enumerate(active_headers):
        header = QtWidgets.QLabel(text)
        layout.addWidget(header, 4, column)
    for index, term in enumerate(active):
        row = index + 5
        from_text, to_text = endpoint_text(term)
        for column, text in enumerate((term.label, from_text, to_text)):
            label = QtWidgets.QLabel(text)
            label.setToolTip(text)
            label.setWordWrap(True)
            label.setMaximumWidth(280)
            layout.addWidget(label, row, column)
        layout.addWidget(
            QtWidgets.QLabel(f"{term.distance_angstrom:.5g}"),
            row,
            3,
        )
        value = QtWidgets.QLineEdit(
            _parameter_to_text(electronic_energy_from_meV(term.value_meV, unit))
        )
        value.setObjectName(f"tight_binding_hopping_value_{index}")
        value.setToolTip(
            f"Coefficient of this unit-Frobenius representative hopping "
            f"matrix in {unit}; converted immediately to canonical meV."
        )
        value.editingFinished.connect(
            lambda identifier=term.identifier, editor=value: self._set_tight_binding_hopping_field(
                identifier,
                "value",
                editor.text(),
            )
        )
        layout.addWidget(value, row, 4)
        for column, (field, bound) in enumerate(
            zip(("lower", "upper"), term.bounds_meV, strict=True),
            start=5,
        ):
            editor = QtWidgets.QLineEdit(
                "" if bound is None else _parameter_to_text(electronic_energy_from_meV(bound, unit))
            )
            editor.setObjectName(f"tight_binding_hopping_{field}_{index}")
            editor.setToolTip(
                f"Optional fit bound in {unit}; empty is unbounded. The "
                "canonical bound is shared with the fitting machinery."
            )
            editor.editingFinished.connect(
                lambda identifier=term.identifier, field=field, editor=editor: (
                    self._set_tight_binding_hopping_field(
                        identifier,
                        field,
                        editor.text(),
                    )
                )
            )
            layout.addWidget(editor, row, column)
        fit = QtWidgets.QCheckBox()
        fit.setObjectName(f"tight_binding_hopping_fit_{index}")
        fit.setChecked(term.fit)
        fit.setToolTip(
            "Vary this coefficient when a compatible electronic-response "
            "component supplies a dataset observable. Tight binding alone "
            "remains calculation-only."
        )
        fit.toggled.connect(
            lambda checked, identifier=term.identifier: self._set_tight_binding_hopping_fit(
                identifier,
                checked,
            )
        )
        layout.addWidget(fit, row, 7)
        sharing, groups = self._tight_binding_sharing_controls(
            model,
            term.identifier,
            object_prefix=f"tight_binding_hopping_{index}",
        )
        layout.addWidget(sharing, row, 8)
        layout.addWidget(groups, row, 9)
        matrix = QtWidgets.QLabel(f"{len(term.basis_i)}x{len(term.basis_j)}")
        matrix.setToolTip(
            "Representative unit-Frobenius hopping matrix. Every "
            "symmetry-equivalent pathway is generated by orbital "
            "covariance, including Hermitian reversal.\n"
            f"Rows: {', '.join(term.basis_i)}\n"
            f"Columns: {', '.join(term.basis_j)}\n"
            f"Matrix:\n{np.array2string(term.matrix, precision=4)}"
        )
        layout.addWidget(matrix, row, 10)
        remove = QtWidgets.QPushButton("Remove")
        remove.setObjectName(f"tight_binding_hopping_remove_{index}")
        remove.setToolTip(
            "Remove this term from the Hamiltonian while retaining it in "
            "the generated suggestion list."
        )
        remove.clicked.connect(
            lambda _checked=False, identifier=term.identifier: self._remove_tight_binding_hopping(
                identifier
            )
        )
        layout.addWidget(remove, row, 11)
    status = QtWidgets.QLabel(
        f"{len(candidates)} available suggestion(s); "
        f"{len(active)} active hopping coefficient(s) on "
        f"{len(multiplicities)} spatial orbit(s)."
    )
    status.setObjectName("tight_binding_hopping_status")
    status.setWordWrap(True)
    layout.addWidget(
        status,
        len(active) + 5,
        0,
        1,
        len(active_headers),
    )

    self.model_parameter_layout.addWidget(group, 8, 0, 1, 4)
