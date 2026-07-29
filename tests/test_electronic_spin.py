import numpy as np
import pytest

from nfit import (
    DataGroup,
    OrbitalManifold,
    add_tight_binding_hopping_term,
    add_tight_binding_orbital_manifold,
    create_model_component,
    electronic_matrix_catalog,
    electronic_matrix_script,
    electronic_model_from_component,
    manifold_orbital_operators,
    orbital_manifold_preset,
    regenerate_tight_binding_hopping_terms,
    set_tight_binding_hopping_term,
    set_tight_binding_soc_term,
    set_tight_binding_spin_treatment,
    spin_half_operators,
    spin_orbit_term,
    spinor_manifold_representation,
    spinor_rotation_representation,
    spinor_time_reversal_residual,
    tight_binding_structure_script,
)


def _crystal():
    return {
        "lattice": {
            "a": 4.0,
            "b": 4.0,
            "c": 4.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": "P 1",
        "sites": [
            {
                "label": "M1",
                "element": "Fe",
                "position": [0.0, 0.0, 0.0],
                "ion": "",
            }
        ],
    }


def _component(shell="p"):
    group = DataGroup("Electronic")
    component = create_model_component(group, "bands", type="tight_binding")
    component.config["crystal"] = _crystal()
    add_tight_binding_orbital_manifold(
        component,
        orbital_manifold_preset("M1", shell),
    )
    return component


def test_automatic_spin_stays_implicit_until_soc_requires_spinors():
    component = _component("p")
    implicit = electronic_model_from_component(component)

    assert implicit.n_basis == 3
    assert implicit.spin_operators is None
    assert implicit.provenance["resolved_spin_treatment"] == "implicit"
    assert implicit.provenance["implicit_spin_degeneracy"] == 2

    set_tight_binding_soc_term(
        component,
        "M1_p",
        value=0.020,
        lower=0.0,
        upper=0.1,
        energy_unit="eV",
        fit=True,
    )
    spinor = electronic_model_from_component(component)

    assert spinor.n_basis == 6
    assert [state.spin for state in spinor.basis[:4]] == [
        "up",
        "down",
        "up",
        "down",
    ]
    assert spinor.spin_operators.shape == (3, 6, 6)
    assert spinor.provenance["resolved_spin_treatment"] == "spinor"
    assert component.parameters["M1_p:soc:lambda"] == pytest.approx(20.0)
    assert component.limits["M1_p:soc:lambda"] == [0.0, 100.0]
    assert component.fit_parameters["M1_p:soc:lambda"] is True
    np.testing.assert_allclose(
        np.linalg.eigvalsh(spinor.hamiltonian([0.0, 0.0, 0.0])),
        [-20.0, -20.0, 10.0, 10.0, 10.0, 10.0],
        atol=1e-9,
    )
    assert spinor_time_reversal_residual(spinor) < 1e-10


def test_explicit_collinear_lift_preserves_parameters_without_soc():
    component = _component("p")
    orbital_model = electronic_model_from_component(component)
    set_tight_binding_spin_treatment(component, "collinear")
    collinear = electronic_model_from_component(component)

    assert collinear.n_basis == 2 * orbital_model.n_basis
    assert collinear.provenance["resolved_spin_treatment"] == "collinear"
    expected = np.kron(
        orbital_model.hamiltonian([0.17, 0.0, 0.0]),
        np.eye(2),
    )
    np.testing.assert_allclose(
        collinear.hamiltonian([0.17, 0.0, 0.0]),
        expected,
    )
    for name in orbital_model.parameter_values:
        assert name in collinear.parameter_values
        np.testing.assert_allclose(
            collinear.parameter_blocks[name],
            np.asarray(
                [
                    np.kron(block, np.eye(2))
                    for block in orbital_model.parameter_blocks[name]
                ]
            ),
        )


def test_projected_soc_is_explicit_and_atomic_requires_complete_shell():
    transform = np.eye(3, dtype=np.complex128)[:, :2]
    manifold = OrbitalManifold(
        site_label="M1",
        label="M1_p_pair",
        basis_kind="real_harmonic",
        orbitals=("p_x", "p_y"),
        l=1,
        harmonic_transform=transform,
        preset="custom_subspace",
    )
    projected = manifold_orbital_operators(
        manifold,
        prescription="projected",
    )
    assert projected.shape == (3, 2, 2)
    with pytest.raises(ValueError, match="complete"):
        manifold_orbital_operators(manifold, prescription="atomic")


def test_effective_soc_operators_round_trip_as_complex_arrays():
    operators = spin_half_operators(1)
    term = spin_orbit_term(
        "M1_effective_pair",
        prescription="effective",
        orbital_operators=operators,
    )

    restored = type(term).from_dict(term.to_dict())

    assert restored.prescription == "effective"
    np.testing.assert_allclose(restored.orbital_operators, operators)


def test_spinor_double_group_representation_is_unitary():
    manifold = orbital_manifold_preset("M1", "p")
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    representation = spinor_manifold_representation(manifold, rotation)
    np.testing.assert_allclose(
        representation.conj().T @ representation,
        np.eye(6),
        atol=1e-9,
    )
    spin = spin_half_operators(1)
    spinor = spinor_rotation_representation(rotation)
    for axis in range(3):
        transformed = spinor @ spin[axis] @ spinor.conj().T
        expected = sum(rotation[other, axis] * spin[other] for other in range(3))
        np.testing.assert_allclose(transformed, expected, atol=1e-9)


def test_soc_builder_script_round_trip_preserves_digest():
    component = _component("p")
    set_tight_binding_soc_term(
        component,
        "M1_p",
        value=0.012,
        energy_unit="eV",
    )
    electronic_model_from_component(component)
    script = tight_binding_structure_script(
        component.config["crystal"],
        component.config["periodic_axes"] or (0, 1, 2),
        orbital_manifolds=component.config["orbital_manifolds"],
        onsite_terms=component.config["onsite_terms"],
        spin_treatment=component.config["spin_treatment"],
        soc_terms=component.config["soc_terms"],
        expected_model_digest=component.config["model_digest"],
    )
    namespace = {}
    exec(compile(script, "<spin-builder-script>", "exec"), namespace)
    assert (
        namespace["electronic_model"].content_digest
        == component.config["model_digest"]
    )
    assert namespace["model"].config["soc_terms"] == component.config["soc_terms"]


def test_matrix_catalog_includes_hopping_elements_and_subspace_blocks():
    component = _component("p")
    generation = regenerate_tight_binding_hopping_terms(component, 4.01)
    term = generation.terms[0]
    add_tight_binding_hopping_term(component, term.identifier)
    set_tight_binding_hopping_term(
        component,
        term.identifier,
        value=-0.1,
        energy_unit="eV",
    )
    catalog = electronic_matrix_catalog(component)

    assert catalog[0].key == "hamiltonian:k"
    representative = next(
        item for item in catalog if item.key == f"builder:{term.identifier}"
    )
    assert representative.basis_matrix.shape == term.matrix.shape
    np.testing.assert_allclose(
        representative.contribution_matrix,
        -100.0 * term.matrix,
    )
    blocks = representative.subspace_blocks()
    assert blocks
    assert all(block.frobenius_norm > 0.0 for block in blocks)
    script = electronic_matrix_script(component)
    compile(script, "<matrix-viewer-script>", "exec")
    assert "show_electronic_matrix_catalog" in script
    assert term.identifier in script


def test_matrix_viewer_exposes_exact_values_and_decomposition(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    component = _component("p")
    catalog = electronic_matrix_catalog(component)

    from nfit.qt_electronic_matrix_viewer import (
        show_electronic_matrix_catalog,
    )

    window = show_electronic_matrix_catalog(catalog)
    selection = window.findChild(
        QtWidgets.QComboBox,
        "electronic_matrix_selection",
    )
    elements = window.findChild(
        QtWidgets.QTableWidget,
        "electronic_matrix_elements",
    )
    decomposition = window.findChild(
        QtWidgets.QTableWidget,
        "electronic_matrix_decomposition",
    )
    assert selection.count() == len(catalog)
    assert elements.rowCount() == catalog[0].basis_matrix.shape[0]
    assert decomposition.toolTip()
    copy_button = window.findChild(
        QtWidgets.QPushButton,
        "electronic_matrix_copy",
    )
    copy_button.click()
    assert "\t" in QtWidgets.QApplication.clipboard().text()
    assert window._nfit_close_shortcut is not None
    window._nfit_close_shortcut.activated.emit()
    assert not window.isVisible()
    assert application is not None
