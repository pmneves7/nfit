"""Qt builders for Lindhard response-model controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from .pipeline import ModelComponentSpec
from .project_model_editor import _parameter_to_text, _sampling_float_matches


def _lindhard_source_component(
    self,
    model: ModelComponentSpec,
) -> ModelComponentSpec | None:
    """Resolve an explicit or uniquely implied electronic source."""

    owner = self._group_for_model(model)
    if owner is None:
        return None
    requested = str(model.config.get("electronic_component", "")).strip()
    if requested:
        candidate = owner.models.get(requested)
        return (
            candidate
            if candidate is not None and candidate.enabled and candidate.type == "tight_binding"
            else None
        )
    candidates = [
        candidate
        for candidate in owner.models.values()
        if candidate.enabled and candidate.type == "tight_binding" and candidate is not model
    ]
    return candidates[0] if len(candidates) == 1 else None


def _build_lindhard_editor(self, model: ModelComponentSpec) -> None:
    """Build the ordinary Lindhard workflow without exposing every override."""

    from PySide6 import QtWidgets

    from .crystal import infer_crystal_formula_units

    source = self._lindhard_source_component(model)
    mesh = model.config.get("response_mesh", [16, 16, 16])
    broadening = model.parameters.get("broadening", 5.0)
    q_policy = str(model.config.get("response_q_evaluation", "auto"))
    q_rtol = float(model.config.get("response_q_interpolation_rtol", 0.0))
    q_atol = float(model.config.get("response_q_interpolation_atol", 0.0))
    accuracy = (
        "exact"
        if q_policy == "auto" and q_rtol == 0.0 and q_atol == 0.0
        else ("validated" if q_policy == "auto" and (q_rtol > 0.0 or q_atol > 0.0) else "advanced")
    )

    summary = QtWidgets.QGroupBox("Model summary")
    summary.setObjectName("lindhard_model_summary_group")
    summary_layout = QtWidgets.QVBoxLayout(summary)
    source_text = source.name if source is not None else "electronic source unresolved"
    backend = str(model.config.get("response_backend", "auto"))
    workers = int(model.config.get("response_workers", 0))
    worker_text = "automatic CPU allocation" if workers == 0 else f"{workers} CPU worker(s)"
    summary_label = QtWidgets.QLabel(
        f"{source_text} · {tuple(mesh)} k mesh · η={broadening:g} meV · "
        f"{'exact Q' if accuracy == 'exact' else 'validated Q interpolation' if accuracy == 'validated' else q_policy + ' Q'} · "
        f"{backend} backend, {worker_text}"
    )
    summary_label.setObjectName("lindhard_model_summary")
    summary_label.setWordWrap(True)
    summary_label.setToolTip(
        "Resolved electronic source, integration mesh, lifetime broadening, "
        "wavevector policy, and execution allocation."
    )
    summary_layout.addWidget(summary_label)
    self.model_parameter_layout.addWidget(summary, 10, 0, 1, 4)

    response = QtWidgets.QGroupBox("Electronic response")
    response.setObjectName("lindhard_response_group")
    response_layout = QtWidgets.QFormLayout(response)

    source_tooltip = (
        "Tight-binding Hamiltonian used for band energies, occupations, and "
        "spin matrix elements. With exactly one enabled tight-binding model, "
        "Automatic is deterministic and scriptable."
    )
    source_combo = QtWidgets.QComboBox()
    source_combo.setObjectName("lindhard_electronic_component")
    source_combo.setToolTip(source_tooltip)
    owner = self._group_for_model(model)
    candidates = (
        [
            candidate
            for candidate in owner.models.values()
            if candidate.enabled and candidate.type == "tight_binding" and candidate is not model
        ]
        if owner is not None
        else []
    )
    automatic_label = (
        f"Automatic ({candidates[0].name})"
        if len(candidates) == 1
        else "Automatic (requires one tight-binding model)"
    )
    source_combo.addItem(automatic_label, "")
    for candidate in candidates:
        source_combo.addItem(candidate.name, candidate.name)
    source_combo.setCurrentIndex(
        max(
            source_combo.findData(str(model.config.get("electronic_component", ""))),
            0,
        )
    )
    source_combo.currentIndexChanged.connect(
        lambda _index, combo=source_combo: self._set_lindhard_config_values(
            electronic_component=str(combo.currentData())
        )
    )
    source_label = QtWidgets.QLabel("Electronic structure")
    source_label.setToolTip(source_tooltip)
    response_layout.addRow(source_label, source_combo)

    occupation_tooltip = (
        "Use the chemical potential stored in the electronic model, or solve "
        "for the chemical potential that gives a specified electron count."
    )
    occupation = QtWidgets.QComboBox()
    occupation.setObjectName("lindhard_chemical_potential_mode")
    occupation.setToolTip(occupation_tooltip)
    occupation.addItem("Use source chemical potential", "source")
    occupation.addItem("Solve from filling", "filling")
    occupation.setCurrentIndex(
        max(
            occupation.findData(str(model.config.get("chemical_potential_mode", "source"))),
            0,
        )
    )
    occupation.currentIndexChanged.connect(
        lambda _index, combo=occupation: self._set_lindhard_config_values(
            chemical_potential_mode=str(combo.currentData())
        )
    )
    occupation_label = QtWidgets.QLabel("Occupations")
    occupation_label.setToolTip(occupation_tooltip)
    response_layout.addRow(occupation_label, occupation)
    if occupation.currentData() == "filling":
        filling_tooltip = (
            "Total electron count in the electronic model cell. nfit solves "
            "the finite-temperature chemical potential for this filling."
        )
        filling = QtWidgets.QLineEdit(_parameter_to_text(model.config.get("filling_per_cell", 1.0)))
        filling.setObjectName("model_config_filling_per_cell")
        filling.setToolTip(filling_tooltip)
        filling.editingFinished.connect(
            lambda editor=filling: self._set_model_config_setting("filling_per_cell", editor.text())
        )
        filling_label = QtWidgets.QLabel("Filling (electrons/cell)")
        filling_label.setToolTip(filling_tooltip)
        response_layout.addRow(filling_label, filling)
    self.model_parameter_layout.addWidget(response, 11, 0, 1, 4)

    sampling = QtWidgets.QGroupBox("Brillouin-zone sampling")
    sampling.setObjectName("lindhard_sampling_group")
    sampling_layout = QtWidgets.QFormLayout(sampling)
    sampling_mode = str(model.config.get("response_sampling_mode", "automatic"))
    mode_tooltip = (
        "Automatic mode searches observable-specific meshes before fitting "
        "and stores one concrete production mesh. Manual mode uses the "
        "entered mesh directly. Neither mode changes a mesh inside an optimizer."
    )
    mode_combo = QtWidgets.QComboBox()
    mode_combo.setObjectName("lindhard_sampling_mode")
    mode_combo.setToolTip(mode_tooltip)
    mode_combo.addItem("Automatic certification", "automatic")
    mode_combo.addItem("Manual mesh", "manual")
    mode_combo.setCurrentIndex(max(mode_combo.findData(sampling_mode), 0))
    mode_combo.currentIndexChanged.connect(
        lambda _index, combo=mode_combo: self._set_lindhard_config_values(
            response_sampling_mode=str(combo.currentData())
        )
    )
    mode_label = QtWidgets.QLabel("Mesh selection")
    mode_label.setToolTip(mode_tooltip)
    sampling_layout.addRow(mode_label, mode_combo)

    profile = str(model.config.get("response_sampling_accuracy", "standard"))
    profile_tooltip = (
        "Preview, Standard, and High require at most 5%, 1%, and 0.2% "
        "normalized change, respectively, for two successive refinements. "
        "The independent Advanced budget may stop without certification."
    )
    profile_combo = QtWidgets.QComboBox()
    profile_combo.setObjectName("lindhard_sampling_accuracy")
    profile_combo.setToolTip(profile_tooltip)
    for label, value in (
        ("Preview (5%)", "preview"),
        ("Standard (1%)", "standard"),
        ("High (0.2%)", "high"),
        ("Custom", "custom"),
    ):
        profile_combo.addItem(label, value)
    profile_combo.setCurrentIndex(max(profile_combo.findData(profile), 0))
    profile_combo.setEnabled(sampling_mode == "automatic")
    profile_combo.currentIndexChanged.connect(
        lambda _index, combo=profile_combo: self._set_lindhard_config_values(
            response_sampling_accuracy=str(combo.currentData())
        )
    )
    profile_label = QtWidgets.QLabel("Accuracy")
    profile_label.setToolTip(profile_tooltip)
    sampling_layout.addRow(profile_label, profile_combo)
    if sampling_mode == "automatic" and profile == "custom":
        custom_tooltip = (
            "Positive normalized error required for two successive response-mesh refinements."
        )
        custom = QtWidgets.QLineEdit(
            _parameter_to_text(
                model.config.get(
                    "response_sampling_custom_rtol",
                    0.01,
                )
            )
        )
        custom.setObjectName("model_config_response_sampling_custom_rtol")
        custom.setToolTip(custom_tooltip)
        custom.editingFinished.connect(
            lambda editor=custom: self._set_model_config_setting(
                "response_sampling_custom_rtol", editor.text()
            )
        )
        custom_label = QtWidgets.QLabel("Custom tolerance")
        custom_label.setToolTip(custom_tooltip)
        sampling_layout.addRow(custom_label, custom)

    symmetry_tooltip = (
        "Auto uses a certified little-group reduction when it is exact and "
        "otherwise falls back to the full mesh. Full disables reduction. "
        "Require reduced raises instead of falling back."
    )
    symmetry = QtWidgets.QComboBox()
    symmetry.setObjectName("lindhard_response_symmetry")
    symmetry.setToolTip(symmetry_tooltip)
    symmetry.addItem("Auto", "auto")
    symmetry.addItem("Full mesh", "full")
    symmetry.addItem("Require reduced mesh", "reduced")
    symmetry.setCurrentIndex(
        max(
            symmetry.findData(str(model.config.get("response_symmetry", "auto"))),
            0,
        )
    )
    symmetry.currentIndexChanged.connect(
        lambda _index, combo=symmetry: self._set_lindhard_config_values(
            response_symmetry=str(combo.currentData())
        )
    )
    symmetry_label = QtWidgets.QLabel("Symmetry")
    symmetry_label.setToolTip(symmetry_tooltip)
    sampling_layout.addRow(symmetry_label, symmetry)

    shift_tooltip = (
        "Fractional offsets in mesh-step units along the electronic model's "
        "periodic axes. The default 0.5 half shift reduces special-point "
        "and Fermi-surface shell artifacts in metallic systems. Use zero "
        "for an explicitly Γ-centered mesh."
    )
    shift = QtWidgets.QLineEdit(
        _parameter_to_text(model.config.get("response_mesh_shift", [0.5, 0.5, 0.5]))
    )
    shift.setObjectName("model_config_response_mesh_shift")
    shift.setToolTip(shift_tooltip)
    shift.editingFinished.connect(
        lambda editor=shift: self._set_model_config_setting("response_mesh_shift", editor.text())
    )
    shift_label = QtWidgets.QLabel("Mesh shift")
    shift_label.setToolTip(shift_tooltip)
    sampling_layout.addRow(shift_label, shift)

    domain = QtWidgets.QGroupBox("Model inspection domain")
    domain.setObjectName("lindhard_certification_domain_group")
    domain_layout = QtWidgets.QFormLayout(domain)

    def add_domain_editor(
        name: str,
        label_text: str,
        default: Any,
        tooltip: str,
    ) -> Any:
        editor = QtWidgets.QLineEdit(_parameter_to_text(model.config.get(name, default)))
        editor.setObjectName(f"model_config_{name}")
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda name=name, editor=editor: self._set_model_config_setting(name, editor.text())
        )
        label = QtWidgets.QLabel(label_text)
        label.setToolTip(tooltip)
        domain_layout.addRow(label, editor)
        return editor

    add_domain_editor(
        "plot_q_reduced",
        "Representative Q (r.l.u.)",
        [0.5, 0.5, 0.5],
        "Transferred wavevector used by model-owned response and convergence "
        "plots, in the linked electronic model's reciprocal basis. Project "
        "mesh certification instead derives its domain from fitted datasets.",
    )
    add_domain_editor(
        "plot_energy_min_meV",
        "Minimum energy (meV)",
        -100.0,
        "Lower energy transfer used by model-owned inspection plots.",
    )
    add_domain_editor(
        "plot_energy_max_meV",
        "Maximum energy (meV)",
        100.0,
        "Upper energy transfer used by model-owned inspection plots.",
    )
    add_domain_editor(
        "convergence_energy_points",
        "Representative energies",
        9,
        "Number of uniformly spaced energies used by the model-owned "
        "mesh/broadening inspection plot.",
    )
    add_domain_editor(
        "plot_temperature_K",
        "Temperature (K)",
        10.0,
        "Temperature used by model-owned response and convergence plots. "
        "Project certification uses each fitted dataset's temperature.",
    )
    sampling_layout.addRow(domain)

    budget = QtWidgets.QGroupBox("Search budget")
    budget.setObjectName("lindhard_sampling_budget_group")
    budget_layout = QtWidgets.QFormLayout(budget)
    for name, label_text, default, tooltip in (
        (
            "response_sampling_max_refinements",
            "Candidate meshes",
            7,
            "Maximum number of meshes evaluated before reporting that the "
            "accuracy target was not certified.",
        ),
        (
            "response_sampling_max_mesh_points",
            "Maximum mesh points",
            500000,
            "Largest allowed full-mesh point count. This is a safety budget, "
            "even when symmetry reduces the actual eigensolves.",
        ),
        (
            "response_sampling_points_per_dataset",
            "Points per dataset",
            32,
            "Maximum deterministic representative fit points evaluated "
            "from each applicable dataset at every candidate mesh.",
        ),
    ):
        editor = QtWidgets.QLineEdit(_parameter_to_text(model.config.get(name, default)))
        editor.setObjectName(f"model_config_{name}")
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda name=name, editor=editor: self._set_model_config_setting(name, editor.text())
        )
        label = QtWidgets.QLabel(label_text)
        label.setToolTip(tooltip)
        budget_layout.addRow(label, editor)
    sampling_layout.addRow(budget)

    mesh_tooltip = (
        "Concrete uniform integration mesh used by plots and fits. Automatic "
        "certification updates this value before fitting; it remains fixed "
        "inside the optimizer."
    )
    mesh_editor = QtWidgets.QLineEdit(_parameter_to_text(mesh))
    mesh_editor.setObjectName("model_config_response_mesh")
    mesh_editor.setToolTip(mesh_tooltip)
    mesh_editor.setEnabled(sampling_mode == "manual")
    mesh_editor.editingFinished.connect(
        lambda editor=mesh_editor: self._set_model_config_setting("response_mesh", editor.text())
    )
    mesh_label = QtWidgets.QLabel("Integration mesh")
    mesh_label.setToolTip(mesh_tooltip)
    sampling_layout.addRow(mesh_label, mesh_editor)

    accuracy_tooltip = (
        "Exact evaluates arbitrary experimental Q directly. Validated "
        "interpolation may accelerate repeated fits, but nfit accepts it only "
        "after deterministic direct evaluations satisfy the requested error."
    )
    accuracy_combo = QtWidgets.QComboBox()
    accuracy_combo.setObjectName("lindhard_q_accuracy")
    accuracy_combo.setToolTip(accuracy_tooltip)
    accuracy_combo.addItem("Exact Q evaluation", "exact")
    accuracy_combo.addItem("Validated interpolation", "validated")
    if accuracy == "advanced":
        accuracy_combo.addItem(
            f"Advanced override ({q_policy})",
            "advanced",
        )
    accuracy_combo.setCurrentIndex(max(accuracy_combo.findData(accuracy), 0))
    accuracy_combo.currentIndexChanged.connect(
        lambda _index, combo=accuracy_combo: self._set_lindhard_q_accuracy(str(combo.currentData()))
    )
    accuracy_label = QtWidgets.QLabel("Experimental Q")
    accuracy_label.setToolTip(accuracy_tooltip)
    sampling_layout.addRow(accuracy_label, accuracy_combo)
    if accuracy == "validated":
        tolerance_tooltip = (
            "Maximum accepted relative complex-susceptibility error at "
            "deterministic off-mesh validation points."
        )
        tolerance = QtWidgets.QLineEdit(_parameter_to_text(q_rtol))
        tolerance.setObjectName("model_config_response_q_interpolation_rtol")
        tolerance.setToolTip(tolerance_tooltip)
        tolerance.editingFinished.connect(
            lambda editor=tolerance: self._set_model_config_setting(
                "response_q_interpolation_rtol", editor.text()
            )
        )
        tolerance_label = QtWidgets.QLabel("Relative tolerance")
        tolerance_label.setToolTip(tolerance_tooltip)
        sampling_layout.addRow(tolerance_label, tolerance)
    certificate = model.config.get("response_sampling_certificate", {})
    pipeline_domain = bool(
        isinstance(certificate, dict)
        and certificate.get("domain", {}).get("kind") == "fit_datasets"
    )
    pipeline_state_stale = False
    if pipeline_domain and owner is not None:
        try:
            from .electronic_pipeline_sampling import (
                electronic_pipeline_state_digest,
            )

            pipeline_state_stale = certificate.get("provenance", {}).get(
                "pipeline_state_digest"
            ) != electronic_pipeline_state_digest(model, owner.models)
        except (TypeError, ValueError):
            pipeline_state_stale = True
    if sampling_mode == "manual":
        status_text = "Manual mesh: no automatic accuracy claim."
    elif not isinstance(certificate, dict) or not certificate:
        status_text = "Not certified. Check/refine before fitting."
    elif certificate.get("status") != "certified":
        status_text = "Not certified within the configured refinement budget."
    elif list(certificate.get("chosen_mesh") or ()) != list(mesh):
        status_text = "Certificate is stale because the mesh changed."
    elif source is not None and bool(source.config.get("model_stale", False)):
        status_text = "Certificate is stale because the electronic model must be rebuilt."
    elif (
        source is not None
        and source.config.get("model_digest")
        and certificate.get("provenance", {}).get("model_digest")
        != source.config.get("model_digest")
    ):
        status_text = "Certificate is stale because the electronic model changed."
    elif pipeline_state_stale:
        status_text = "Certificate is stale because a TB, Lindhard, or RPA component changed."
    elif not _sampling_float_matches(
        certificate.get("provenance", {}).get("broadening_meV"),
        model.parameters.get("broadening", 5.0),
    ):
        status_text = "Certificate is stale because the response broadening changed."
    elif certificate.get("provenance", {}).get("symmetry") != str(
        model.config.get("response_symmetry", "auto")
    ):
        status_text = "Certificate is stale because the symmetry policy changed."
    elif not pipeline_domain and not _sampling_float_matches(
        certificate.get("domain", {}).get("temperature_K"),
        model.config.get("plot_temperature_K", 10.0),
    ):
        status_text = "Certificate is stale because the certified temperature changed."
    elif not pipeline_domain and (
        not _sampling_float_matches(
            certificate.get("domain", {}).get("energy_min_meV"),
            model.config.get("plot_energy_min_meV", -100.0),
        )
        or not _sampling_float_matches(
            certificate.get("domain", {}).get("energy_max_meV"),
            model.config.get("plot_energy_max_meV", 100.0),
        )
    ):
        status_text = "Certificate is stale because the certified energy range changed."
    elif not pipeline_domain and int(certificate.get("domain", {}).get("points", -1)) != int(
        model.config.get("convergence_energy_points", 9)
    ):
        status_text = "Certificate is stale because the representative energy count changed."
    elif not pipeline_domain and (
        not np.allclose(
            np.asarray(
                certificate.get("domain", {}).get(
                    "q_min_reduced",
                    [np.nan, np.nan, np.nan],
                ),
                dtype=float,
            ),
            np.asarray(
                model.config.get("plot_q_reduced", [0.5, 0.5, 0.5]),
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.allclose(
            np.asarray(
                certificate.get("domain", {}).get(
                    "q_max_reduced",
                    [np.nan, np.nan, np.nan],
                ),
                dtype=float,
            ),
            np.asarray(
                model.config.get("plot_q_reduced", [0.5, 0.5, 0.5]),
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        status_text = "Certificate is stale because the representative Q changed."
    elif not np.allclose(
        np.asarray(
            certificate.get("provenance", {}).get(
                "mesh_shift",
                [np.nan] * len(model.config.get("response_mesh_shift", [])),
            ),
            dtype=float,
        ),
        np.asarray(
            model.config.get("response_mesh_shift", [0.5, 0.5, 0.5]),
            dtype=float,
        ),
        rtol=0.0,
        atol=1.0e-12,
    ):
        status_text = "Certificate is stale because the mesh shift changed."
    elif (
        str(model.config.get("chemical_potential_mode", "source")) == "source"
        and source is not None
        and not _sampling_float_matches(
            certificate.get("provenance", {}).get("chemical_potential_meV"),
            source.config.get("chemical_potential_meV", 0.0),
        )
    ):
        status_text = "Certificate is stale because the source chemical potential changed."
    elif str(
        model.config.get("chemical_potential_mode", "source")
    ) == "filling" and not _sampling_float_matches(
        certificate.get("provenance", {}).get("filling_per_cell"),
        model.config.get("filling_per_cell", 1.0),
    ):
        status_text = "Certificate is stale because the electronic filling changed."
    else:
        tolerance = certificate.get("policy", {}).get(
            "relative_tolerance",
            "?",
        )
        if pipeline_domain:
            status_text = (
                "Complete pipeline certified on "
                f"{certificate.get('domain', {}).get('sampled_points', '?')} "
                "representative point(s) across "
                f"{certificate.get('domain', {}).get('dataset_count', '?')} "
                f"fit dataset(s) at tolerance {tolerance}."
            )
        else:
            status_text = (
                f"Certified on {certificate.get('domain', {}).get('points', '?')} "
                f"representative point(s) at tolerance {tolerance}."
            )
    convergence = QtWidgets.QLabel(status_text)
    convergence.setObjectName("lindhard_convergence_status")
    convergence.setWordWrap(True)
    convergence.setToolTip(
        "The certificate records all attempted meshes and observable errors. "
        "Changing relevant physics marks it stale; final parameters should "
        "be recertified after fitting."
    )
    sampling_layout.addRow(convergence)
    q = np.asarray(
        model.config.get("plot_q_reduced", [0.5, 0.5, 0.5]),
        dtype=float,
    )
    convergence_energies = np.linspace(
        float(model.config.get("plot_energy_min_meV", -100.0)),
        float(model.config.get("plot_energy_max_meV", 100.0)),
        int(model.config.get("convergence_energy_points", 9)),
    )
    gamma_static = (
        q.shape == (3,)
        and np.allclose(q - np.rint(q), 0.0, rtol=0.0, atol=1.0e-12)
        and np.any(
            np.isclose(
                convergence_energies,
                0.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )
    if gamma_static:
        warning = QtWidgets.QLabel(
            "The model inspection domain includes Q = 0 and E = 0. This "
            "static Pauli limit samples a very narrow Fermi-surface shell "
            "and may require much denser meshes than a finite-Q neutron "
            "response. This warning applies to the model-owned convergence "
            "plot; project certification uses the fitted datasets."
        )
        warning.setObjectName("lindhard_gamma_static_warning")
        warning.setWordWrap(True)
        warning.setToolTip(
            "At finite temperature the Q=0, E=0 intraband limit contains "
            "-df/dE. Its width is set by kBT rather than the response "
            "broadening, so coarse uniform meshes can converge irregularly."
        )
        sampling_layout.addRow("Convergence warning", warning)
    certify = QtWidgets.QPushButton("Check/refine convergence")
    certify.setObjectName("lindhard_certify_sampling")
    certify.setEnabled(sampling_mode == "automatic")
    certify.setToolTip(
        "Search deterministic physically spaced meshes against the complete "
        "TB-Lindhard-RPA observable on representative points from every "
        "applicable fit dataset. Store a concrete mesh only after every "
        "dataset passes two successive refinements."
    )
    certify.clicked.connect(
        lambda _checked=False, model=model: self._certify_lindhard_sampling(model)
    )
    sampling_layout.addRow(certify)
    self.model_parameter_layout.addWidget(sampling, 12, 0, 1, 4)

    experiment = QtWidgets.QGroupBox("Experimental coupling")
    experiment.setObjectName("lindhard_experiment_group")
    experiment_layout = QtWidgets.QFormLayout(experiment)
    formula_tooltip = (
        "Automatic mode derives the reduced chemical formula and the number "
        "of formula units in the actual electronic model cell from a complete "
        "crystal. Manual mode is an explicit normalization override."
    )
    formula_mode = QtWidgets.QComboBox()
    formula_mode.setObjectName("lindhard_formula_units_mode")
    formula_mode.setToolTip(formula_tooltip)
    formula_mode.addItem("Automatic from crystal", "auto")
    formula_mode.addItem("Manual override", "manual")
    formula_mode.setCurrentIndex(
        max(
            formula_mode.findData(str(model.config.get("formula_units_mode", "manual"))),
            0,
        )
    )
    formula_mode.currentIndexChanged.connect(
        lambda _index, combo=formula_mode: self._set_lindhard_config_values(
            formula_units_mode=str(combo.currentData())
        )
    )
    formula_label = QtWidgets.QLabel("Formula units")
    formula_label.setToolTip(formula_tooltip)
    experiment_layout.addRow(formula_label, formula_mode)
    if formula_mode.currentData() == "manual":
        formula_value = QtWidgets.QLineEdit(
            _parameter_to_text(model.config.get("formula_units_per_cell", 1.0))
        )
        formula_value.setObjectName("model_config_formula_units_per_cell")
        formula_value.setToolTip(
            "Explicit formula units represented by the electronic model "
            "cell. This converts model-cell spectral and bulk responses "
            "when the dataset is normalized per formula unit."
        )
        formula_value.editingFinished.connect(
            lambda editor=formula_value: self._set_lindhard_coupling_text(
                "formula_units_per_cell", editor.text()
            )
        )
        experiment_layout.addRow("Formula units / model cell", formula_value)
    else:
        formula_status = "Automatic normalization requires a complete linked crystal."
        if source is not None:
            try:
                crystal = source.config["crystal"]
                model_data = source.config.get("model_data")
                lattice = (
                    model_data.get("direct_lattice")
                    if isinstance(model_data, dict)
                    and not bool(source.config.get("model_stale", False))
                    else None
                )
                if lattice is None and not bool(source.config.get("use_primitive_cell", True)):
                    from .crystal import lattice_vectors

                    lattice = lattice_vectors(crystal["lattice"])
                if lattice is None:
                    raise ValueError("the actual electronic model cell has not been resolved yet")
                result = infer_crystal_formula_units(
                    crystal,
                    model_lattice=lattice,
                )
                formula_status = (
                    f"{result.formula}: {result.formula_units_per_model_cell} "
                    "formula unit(s) per electronic model cell"
                )
            except (KeyError, TypeError, ValueError):
                pass
        status = QtWidgets.QLabel(formula_status)
        status.setObjectName("lindhard_formula_units_status")
        status.setWordWrap(True)
        status.setToolTip(formula_tooltip)
        experiment_layout.addRow(status)

    magnetic_tooltip = (
        "Normalization per magnetic ion counts represented tight-binding "
        "sites independently of the form-factor profile. Automatic mode "
        "requires one selected species; manual mode supplies the count."
    )
    magnetic_mode = QtWidgets.QComboBox()
    magnetic_mode.setObjectName("lindhard_magnetic_normalization_mode")
    magnetic_mode.setToolTip(magnetic_tooltip)
    magnetic_mode.addItem("Automatic from represented sites", "auto")
    magnetic_mode.addItem("Manual count", "manual")
    magnetic_mode.setCurrentIndex(
        max(
            magnetic_mode.findData(str(model.config.get("magnetic_normalization_mode", "auto"))),
            0,
        )
    )
    magnetic_mode.currentIndexChanged.connect(
        lambda _index, combo=magnetic_mode: self._set_lindhard_config_values(
            magnetic_normalization_mode=str(combo.currentData())
        )
    )
    magnetic_label = QtWidgets.QLabel("Magnetic-center count")
    magnetic_label.setToolTip(magnetic_tooltip)
    experiment_layout.addRow(magnetic_label, magnetic_mode)
    if magnetic_mode.currentData() == "manual":
        magnetic_count = QtWidgets.QLineEdit(
            _parameter_to_text(model.config.get("magnetic_centers_per_model_cell", 1.0))
        )
        magnetic_count.setObjectName("lindhard_magnetic_centers_per_model_cell")
        magnetic_count.setToolTip("Explicit reference magnetic centers per electronic model cell.")
        magnetic_count.editingFinished.connect(
            lambda editor=magnetic_count: self._set_lindhard_coupling_text(
                "magnetic_centers_per_model_cell", editor.text()
            )
        )
        experiment_layout.addRow("Centers / model cell", magnetic_count)
    else:
        magnetic_species = QtWidgets.QLineEdit(
            str(model.config.get("magnetic_normalization_species", "") or "")
        )
        magnetic_species.setObjectName("lindhard_magnetic_normalization_species")
        magnetic_species.setPlaceholderText("automatic when unique")
        magnetic_species.setToolTip(
            "Optional represented basis species or element, such as V4+ or V. "
            "This selects a count only; it does not select a form factor."
        )
        magnetic_species.editingFinished.connect(
            lambda editor=magnetic_species: self._set_lindhard_config_values(
                magnetic_normalization_species=editor.text().strip()
            )
        )
        experiment_layout.addRow("Reference species", magnetic_species)

    normalization_text = "The intrinsic Lindhard response is per electronic model cell."
    if source is not None:
        try:
            from .electronic_normalization import (
                resolve_electronic_response_normalization,
            )
            from .electronic_structure import electronic_model_from_component

            electronic_model = electronic_model_from_component(source)
            formula_normalization = resolve_electronic_response_normalization(
                electronic_model,
                model.config,
                crystal=source.config.get("crystal"),
                target_basis="per_formula_unit",
            )
            magnetic_normalization = resolve_electronic_response_normalization(
                electronic_model,
                model.config,
                crystal=source.config.get("crystal"),
                target_basis="per_magnetic_ion",
            )
            normalization_text = (
                "Electronic model cell: "
                f"{formula_normalization.entities_per_model_cell:g} f.u.; "
                f"{magnetic_normalization.entities_per_model_cell:g} "
                f"{magnetic_normalization.magnetic_reference_species or 'magnetic'} "
                "center(s). The dataset basis chooses the divisor."
            )
        except (KeyError, TypeError, ValueError):
            pass
    normalization_status = QtWidgets.QLabel(normalization_text)
    normalization_status.setObjectName("lindhard_normalization_status")
    normalization_status.setWordWrap(True)
    normalization_status.setToolTip(
        "Orbital form factors affect Q-dependent neutron amplitude. Formula-"
        "unit and magnetic-center counts affect only response normalization."
    )
    experiment_layout.addRow(normalization_status)

    g_tooltip = (
        "Landé g factor used only when the q=0 spin response is converted "
        "to bulk susceptibility or field-induced moment."
    )
    g_factor = QtWidgets.QLineEdit(_parameter_to_text(model.config.get("bulk_g_factor", 2.0)))
    g_factor.setObjectName("model_config_bulk_g_factor")
    g_factor.setToolTip(g_tooltip)
    g_factor.editingFinished.connect(
        lambda editor=g_factor: self._set_lindhard_coupling_text("bulk_g_factor", editor.text())
    )
    g_label = QtWidgets.QLabel("Bulk g factor")
    g_label.setToolTip(g_tooltip)
    experiment_layout.addRow(g_label, g_factor)

    powder = QtWidgets.QLabel(
        "Powder |Q| datasets are converted automatically by orientational "
        "averaging in the linked reciprocal lattice."
    )
    powder.setObjectName("lindhard_powder_conversion_status")
    powder.setWordWrap(True)
    powder.setToolTip(
        "The number of deterministic sphere directions is available under "
        "Advanced model and execution."
    )
    experiment_layout.addRow(powder)
    self.model_parameter_layout.addWidget(experiment, 13, 0, 1, 4)


def _organize_lindhard_editor(
    self,
    model: ModelComponentSpec,
    *,
    fit_group: Any,
    scope_group: Any,
    advanced_group: Any,
) -> None:
    """Collect the response workflow into focused tabs."""

    from PySide6 import QtWidgets

    def take(widget: Any | None) -> Any | None:
        if widget is not None:
            self.model_parameter_layout.removeWidget(widget)
        return widget

    groups = {
        "summary": take(
            self.model_parameter_widget.findChild(
                QtWidgets.QWidget,
                "lindhard_model_summary_group",
            )
        ),
        "response": take(
            self.model_parameter_widget.findChild(
                QtWidgets.QWidget,
                "lindhard_response_group",
            )
        ),
        "sampling": take(
            self.model_parameter_widget.findChild(
                QtWidgets.QWidget,
                "lindhard_sampling_group",
            )
        ),
        "experiment": take(
            self.model_parameter_widget.findChild(
                QtWidgets.QWidget,
                "lindhard_experiment_group",
            )
        ),
        "plots": take(
            self.model_parameter_widget.findChild(
                QtWidgets.QWidget,
                "model_plot_actions_group",
            )
        ),
        "fit": take(fit_group),
        "scope": take(scope_group),
        "advanced": take(advanced_group),
    }

    tabs = QtWidgets.QTabWidget()
    tabs.setObjectName("lindhard_builder_tabs")
    tabs.setToolTip(
        "Define the response, choose sampling accuracy, connect it to "
        "experimental normalization, inspect calculations, and access "
        "uncommon execution overrides."
    )

    def add_page(label: str, names: tuple[str, ...]) -> None:
        page = QtWidgets.QWidget()
        page.setObjectName(f"lindhard_{label.lower().replace(' ', '_')}_tab")
        layout = QtWidgets.QVBoxLayout(page)
        for name in names:
            widget = groups.get(name)
            if widget is not None:
                layout.addWidget(widget)
        layout.addStretch(1)
        tabs.addTab(page, label)

    add_page("Response", ("response", "fit", "scope"))
    add_page("Sampling", ("sampling",))
    add_page("Experimental coupling", ("experiment",))
    add_page("Calculate and inspect", ("plots",))
    add_page("Advanced", ("advanced",))
    if groups["summary"] is not None:
        self.model_parameter_layout.addWidget(groups["summary"], 0, 0, 1, 4)
    self.model_parameter_layout.addWidget(tabs, 1, 0, 1, 4)
