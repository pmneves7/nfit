from __future__ import annotations

import numpy as np
import pytest

from nfit import (
    BasisState,
    ElectronicModel,
    ModelComponentSpec,
    configure_lindhard_experimental_coupling,
    resolve_electronic_response_normalization,
)


def _liv2o4_model_and_crystal():
    lattice = np.eye(3)
    basis = tuple(
        BasisState(label=f"V{index}:d", site=f"V{index}", species="V4+", orbital="d")
        for index in range(4)
    )
    model = ElectronicModel(
        direct_lattice=lattice,
        basis=basis,
        translations=np.zeros((1, 3), dtype=int),
        hamiltonian_blocks=np.zeros((1, 4, 4), dtype=complex),
        interpolation_weights=np.ones(1),
        orbital_centers=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ]
        ),
    )
    elements = ["Li"] * 2 + ["V"] * 4 + ["O"] * 8
    sites = [
        {
            "label": f"{element}{index}",
            "element": element,
            "position": [index / len(elements), 0.0, 0.0],
        }
        for index, element in enumerate(elements)
    ]
    crystal = {
        "lattice": {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": "P 1",
        "sites": sites,
    }
    return model, crystal


def test_liv2o4_formula_unit_and_magnetic_center_normalizations_are_independent():
    model, crystal = _liv2o4_model_and_crystal()
    config = {
        "formula_units_mode": "auto",
        "magnetic_normalization_mode": "auto",
        "ion": "V3",
    }
    per_formula = resolve_electronic_response_normalization(
        model,
        config,
        crystal=crystal,
        target_basis="per_formula_unit",
    )
    per_v = resolve_electronic_response_normalization(
        model,
        config,
        crystal=crystal,
        target_basis="per_magnetic_ion",
    )

    assert per_formula.entities_per_model_cell == pytest.approx(2.0)
    assert per_v.entities_per_model_cell == pytest.approx(4.0)
    assert per_v.magnetic_reference_species == "V4+"
    np.testing.assert_allclose(per_formula.apply([8.0]), [4.0])
    np.testing.assert_allclose(per_v.apply([8.0]), [2.0])
    # The V3 probe profile above deliberately differs from the V4+ basis
    # metadata; it must not alter either normalization count.


def test_manual_magnetic_center_count_supports_selected_subspace_normalization():
    model, _crystal = _liv2o4_model_and_crystal()
    resolved = resolve_electronic_response_normalization(
        model,
        {
            "magnetic_normalization_mode": "manual",
            "magnetic_normalization_species": "effective V",
            "magnetic_centers_per_model_cell": 2.5,
        },
        target_basis="per_magnetic_ion",
    )
    assert resolved.target_label == "effective V"
    assert resolved.scale_from_model_cell == pytest.approx(0.4)


def test_lindhard_coupling_configuration_is_public_atomic_and_scriptable():
    component = ModelComponentSpec(
        name="response",
        type="lindhard",
        config={
            "formula_units_mode": "manual",
            "formula_units_per_cell": 1.0,
            "magnetic_normalization_mode": "auto",
            "bulk_g_factor": 2.0,
        },
    )
    configure_lindhard_experimental_coupling(
        component,
        formula_units_per_cell=2.0,
        bulk_g_factor=2.1,
    )
    assert component.config["formula_units_per_cell"] == pytest.approx(2.0)
    assert component.config["bulk_g_factor"] == pytest.approx(2.1)

    before = dict(component.config)
    with pytest.raises(KeyError, match="unknown Lindhard coupling field"):
        configure_lindhard_experimental_coupling(
            component,
            form_factor_mode="single_ion",
        )
    assert component.config == before
