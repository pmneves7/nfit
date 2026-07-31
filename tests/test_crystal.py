import numpy as np
import pytest

pytest.importorskip("gemmi")

from nfit.crystal import (
    Bond,
    crystal_from_cif,
    expand_crystal_sites,
    expand_magnetic_sites,
    generate_bond_orbits,
    generate_spatial_bond_orbits,
    infer_crystal_formula_units,
    lattice_vectors,
    orbits_from_config,
    orbits_to_config,
    primitive_lattice_vectors,
    sites_to_config,
)

PYROCHLORE = {
    "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "spacegroup": "F d -3 m:2",
    "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "Yb3"}],  # 16c
}

SPINEL_V = {
    "lattice": {
        "a": 8.24062,
        "b": 8.24062,
        "c": 8.24062,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    },
    "spacegroup": "227",
    "sites": [{"label": "V1", "position": [0.5, 0.5, 0.5], "ion": "V2"}],  # 16d
}

FCC = {
    "lattice": {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "spacegroup": "F m -3 m",
    "sites": [{"label": "Ni1", "position": [0.0, 0.0, 0.0], "ion": "Ni2"}],
}


def test_expand_pyrochlore_16c_gives_16_sites():
    sites = expand_magnetic_sites(PYROCHLORE, ["M1"])
    assert len(sites) == 16
    assert all(site.ion == "Yb3" for site in sites)
    positions = np.asarray([site.position for site in sites])
    assert np.all((positions >= 0.0) & (positions < 1.0))


def test_generic_site_and_spatial_orbit_apis_preserve_heisenberg_geometry():
    crystal = {
        **FCC,
        "sites": [
            {
                **FCC["sites"][0],
                "element": "Ni",
            }
        ],
    }
    generic_sites = expand_crystal_sites(crystal, ["Ni1"])
    magnetic_sites = expand_magnetic_sites(crystal, ["Ni1"])
    assert generic_sites == magnetic_sites
    assert all(site.element == "Ni" for site in generic_sites)

    sites_b, spatial = generate_spatial_bond_orbits(
        crystal, ["Ni1"], cutoff_angstrom=3.0
    )
    sites_j, exchange = generate_bond_orbits(
        crystal, ["Ni1"], cutoff_angstrom=3.0
    )
    assert sites_b == sites_j
    assert [orbit.label for orbit in spatial] == ["B1"]
    assert [orbit.label for orbit in exchange] == ["J1"]
    assert spatial[0].bonds == exchange[0].bonds
    assert spatial[0].operations == exchange[0].operations


def test_public_lattice_vectors_use_column_vector_convention():
    basis = lattice_vectors(FCC["lattice"])
    np.testing.assert_allclose(basis, np.diag([4.0, 4.0, 4.0]), atol=1.0e-14)


def test_formula_units_follow_complete_composition_and_model_cell():
    crystal = {
        "lattice": {
            "a": 8.24,
            "b": 8.24,
            "c": 8.24,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": "F d -3 m:2",
        "sites": [
            {
                "label": "Li1",
                "element": "Li+",
                "position": [0.125, 0.125, 0.125],
            },
            {
                "label": "V1",
                "element": "V4+",
                "position": [0.5, 0.5, 0.5],
            },
            {
                "label": "O1",
                "element": "O2-",
                "position": [0.261, 0.261, 0.261],
            },
        ],
    }
    conventional = infer_crystal_formula_units(crystal)
    assert conventional.formula == "LiO4V2"
    assert conventional.conventional_formula_units == 8
    assert conventional.formula_units_per_model_cell == 8

    primitive = infer_crystal_formula_units(
        crystal,
        model_lattice=primitive_lattice_vectors(
            crystal["lattice"],
            crystal["spacegroup"],
        ),
    )
    assert primitive.cell_multiplicity == 4
    assert primitive.formula_units_per_model_cell == 2


def test_formula_unit_inference_rejects_partial_occupancy():
    crystal = {
        **FCC,
        "sites": [
            {
                **FCC["sites"][0],
                "element": "Ni",
                "occupancy": 0.5,
            }
        ],
    }
    with pytest.raises(ValueError, match="partially occupied"):
        infer_crystal_formula_units(crystal)


def test_expand_unknown_label_raises():
    with pytest.raises(ValueError, match="M9"):
        expand_magnetic_sites(PYROCHLORE, ["M9"])


def test_fcc_first_neighbor_orbit():
    sites, orbits = generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=3.0)
    assert len(sites) == 4
    assert len(orbits) == 1
    (j1,) = orbits
    assert j1.label == "J1"
    assert j1.distance_angstrom == pytest.approx(4.0 / np.sqrt(2.0), abs=1e-4)
    # 12 nearest neighbors per site, 4 sites, each bond stored once
    assert j1.multiplicity == 4 * 12 // 2


def test_pyrochlore_third_neighbor_shell_splits_into_j3a_j3b():
    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn * 2.0 + 0.01)
    labels = [orbit.label for orbit in orbits]
    assert labels == ["J1", "J2", "J3a", "J3b"]

    j1 = orbits[0]
    assert j1.distance_angstrom == pytest.approx(nn, abs=1e-4)
    assert j1.multiplicity == 16 * 6 // 2

    j3a, j3b = orbits[2], orbits[3]
    assert j3a.distance_angstrom == pytest.approx(j3b.distance_angstrom, abs=1e-4)
    assert j3a.distance_angstrom == pytest.approx(2.0 * nn, abs=1e-4)
    # the twelve third neighbors split six/six between the two orbits
    assert j3a.multiplicity == j3b.multiplicity == 16 * 6 // 2


def test_numeric_fd3m_uses_reference_setting_for_spinel_v_pyrochlore_site():
    sites, orbits = generate_bond_orbits(SPINEL_V, ["V1"], cutoff_angstrom=9.0)

    assert len(sites) == 16
    labels = [orbit.label for orbit in orbits]
    assert labels[:4] == ["J1", "J2", "J3a", "J3b"]

    j3a, j3b = orbits[2], orbits[3]
    assert j3a.distance_angstrom == pytest.approx(j3b.distance_angstrom, abs=1e-4)
    assert j3a.distance_angstrom == pytest.approx(
        SPINEL_V["lattice"]["a"] / np.sqrt(2.0), abs=1e-4
    )
    assert j3a.multiplicity == j3b.multiplicity == 16 * 6 // 2


def test_explicit_fd3m_origin_choice_is_preserved():
    crystal = {**SPINEL_V, "spacegroup": "F d -3 m:1"}
    sites, orbits = generate_bond_orbits(crystal, ["V1"], cutoff_angstrom=9.0)

    assert len(sites) == 8
    assert [orbit.label for orbit in orbits] == ["J1", "J2", "J3", "J4", "J5"]


def test_bond_canonicalization_treats_directions_as_one_bond():
    bond = Bond(2, 1, (1, 0, -1))
    assert bond.canonical() == Bond(1, 2, (-1, 0, 1))
    assert bond.canonical() == bond.reversed().canonical()


def test_orbit_config_round_trip():
    _, orbits = generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=3.0)
    payload = orbits_to_config(orbits)
    restored = orbits_from_config(payload)
    assert restored == orbits


def test_orbits_feed_rpa_geometry():
    from nfit.spin_fluctuations import build_rpa_geometry, rpa_exchange_matrix

    sites, orbits = generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=3.0)
    H = np.array([0.0, 0.3, 1.0])
    zeros = np.zeros_like(H)
    geometry = build_rpa_geometry(
        H, zeros, zeros, sites_to_config(sites), orbits_to_config(orbits)
    )
    exchange = rpa_exchange_matrix(geometry, {"J1": 1.0})
    np.testing.assert_allclose(
        exchange, np.conj(np.transpose(exchange, (0, 2, 1))), atol=1e-12
    )
    # at Q = 0 every neighbor phase is 1: the uniform mode sees all 12
    # nearest neighbors, and inter-sublattice bonds leave the trace at zero
    lam = np.linalg.eigvalsh(exchange[0])
    assert lam.max() == pytest.approx(12.0, abs=1e-8)
    assert lam.sum() == pytest.approx(0.0, abs=1e-8)


def test_crystal_from_cif_round_trip(tmp_path):
    cif = tmp_path / "test.cif"
    cif.write_text(
        """
data_test
_cell_length_a 4.10
_cell_length_b 4.10
_cell_length_c 4.10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0.00000 0.00000 0.00000
O1 O 0.50000 0.50000 0.50000
"""
    )
    crystal = crystal_from_cif(str(cif))
    assert crystal["lattice"]["a"] == pytest.approx(4.10)
    assert crystal["spacegroup"].replace(" ", "").lower() == "fm-3m"
    labels = [site["label"] for site in crystal["sites"]]
    assert labels == ["Ni1", "O1"]
    assert crystal["sites"][1]["position"] == [0.5, 0.5, 0.5]
    assert crystal["provenance"]["path"] == str(cif.resolve())
    assert len(crystal["provenance"]["sha256"]) == 64

    crystal["sites"][0]["ion"] = "Ni2"
    sites, orbits = generate_bond_orbits(crystal, ["Ni1"], cutoff_angstrom=3.0)
    assert len(sites) == 4
    assert orbits[0].label == "J1"


def test_crystal_from_cif_normalizes_legacy_spacegroup_suffix(tmp_path):
    cif = tmp_path / "spinel.cif"
    cif.write_text(
        """
data_spinel
_cell_length_a 8.24
_cell_length_b 8.24
_cell_length_c 8.24
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F d -3 m Z'
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
V1 V 0.5 0.5 0.5
"""
    )
    assert crystal_from_cif(str(cif))["spacegroup"] == "F d -3 m:2"


def test_crystal_from_cif_preserves_explicit_spacegroup_origin_choice(tmp_path):
    cif = tmp_path / "spinel_origin_one.cif"
    cif.write_text(
        """
data_spinel
_cell_length_a 8.24
_cell_length_b 8.24
_cell_length_c 8.24
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F d -3 m:1'
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
V1 V 0.5 0.5 0.5
"""
    )
    assert crystal_from_cif(str(cif))["spacegroup"] == "F d -3 m:1"


def test_generate_bond_orbits_validates_cutoff():
    with pytest.raises(ValueError, match="cutoff"):
        generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=0.0)


def test_pyrochlore_reduces_to_four_sublattices_with_identical_chipp():
    """Primitive-cell folding of the F-centered pyrochlore network is exact.

    The conventional Fd-3m cell carries 16 magnetic sites; the F-centering
    translations fold them to 4. chi'' must be identical to machine precision
    at generic Q with all four exchange constants nonzero.
    """

    from nfit.spin_fluctuations import (
        build_rpa_geometry,
        heisenberg_rpa_chipp,
        reduce_site_network,
    )

    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn * 2.0 + 0.01)
    positions = sites_to_config(sites)
    orbit_payload = orbits_to_config(orbits)

    reduced_positions, reduced_orbits = reduce_site_network(positions, orbit_payload)
    assert reduced_positions.shape == (4, 3)
    for full_orbit, reduced_orbit in zip(
        orbit_payload,
        reduced_orbits,
        strict=True,
    ):
        assert len(reduced_orbit["bonds"]) * 4 == len(full_orbit["bonds"])

    rng = np.random.default_rng(11)
    hkl = rng.uniform(-2.5, 2.5, size=(40, 3))
    E = rng.uniform(0.3, 6.0, size=40)
    j_values = {"J1": 0.12, "J2": -0.05, "J3a": 0.02, "J3b": -0.01}

    full_geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbit_payload)
    reduced_geometry = build_rpa_geometry(
        hkl[:, 0], hkl[:, 1], hkl[:, 2], reduced_positions, reduced_orbits
    )
    full = heisenberg_rpa_chipp(full_geometry, E, chi0=0.9, gamma0=3.0, j_values=j_values)
    reduced = heisenberg_rpa_chipp(reduced_geometry, E, chi0=0.9, gamma0=3.0, j_values=j_values)
    np.testing.assert_allclose(reduced, full, rtol=1e-12)


P1_TRICLINIC = {
    "lattice": {"a": 5.0, "b": 6.0, "c": 7.0, "alpha": 80.0, "beta": 95.0, "gamma": 105.0},
    "spacegroup": "P 1",
    "sites": [
        {"label": "A", "position": [0.0, 0.0, 0.0], "ion": ""},
        {"label": "B", "position": [0.3, 0.4, 0.2], "ion": ""},
    ],
}


def _bond_vector_cartesian(bond, positions, lattice_matrix):
    delta = (
        positions[bond.site_j]
        + np.asarray(bond.offset, dtype=float)
        - positions[bond.site_i]
    )
    return lattice_matrix @ delta


def test_bond_orbits_record_symmetry_operations():
    from nfit.crystal import _lattice_vectors, cartesian_rotation

    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=2 * nn + 0.01)
    lattice_matrix = _lattice_vectors(PYROCHLORE["lattice"])
    positions = np.asarray([site.position for site in sites])

    for orbit in orbits:
        assert orbit.operations is not None
        assert len(orbit.operations) == len(orbit.bonds)
        # The representative carries the identity, unreversed.
        np.testing.assert_allclose(np.asarray(orbit.operations[0].rotation), np.eye(3))
        assert orbit.operations[0].reverses is False
        representative_vector = _bond_vector_cartesian(
            orbit.bonds[0], positions, lattice_matrix
        )
        for bond, symmetry in zip(orbit.bonds, orbit.operations, strict=True):
            rotation = cartesian_rotation(
                np.asarray(symmetry.rotation), PYROCHLORE["lattice"]
            )
            # Cartesian rotations of crystallographic ops are orthogonal.
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
            # The recorded op maps the representative bond vector onto this
            # bond's vector (with a sign flip when the op reversed the bond).
            target = _bond_vector_cartesian(bond, positions, lattice_matrix)
            if symmetry.reverses:
                target = -target
            np.testing.assert_allclose(rotation @ representative_vector, target, atol=1e-8)

    # Sites record their generating rotation too.
    assert all(site.rotation is not None for site in sites)
    np.testing.assert_allclose(np.asarray(sites[0].rotation), np.eye(3))


def test_pyrochlore_nn_exchange_basis_matches_published_form():
    """Pyrochlore NN allows 4 couplings incl. Heisenberg (Ross et al. PRB 84, 064430):
    after removing the isotropic part, 2 symmetric-traceless + 1 DM."""
    from nfit.crystal import symmetry_allowed_exchange_basis

    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn + 0.01)
    basis = symmetry_allowed_exchange_basis(PYROCHLORE, sites, orbits[0])
    kinds = [element["kind"] for element in basis]
    assert kinds == ["symmetric", "symmetric", "dm"]
    for element in basis:
        matrix = np.asarray(element["matrix"])
        np.testing.assert_allclose(np.linalg.norm(matrix), 1.0, rtol=1e-9)
        if element["kind"] == "symmetric":
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
            assert abs(np.trace(matrix)) < 1e-9
        else:
            np.testing.assert_allclose(matrix, -matrix.T, atol=1e-9)


def test_fcc_nn_bond_inversion_center_forbids_dm():
    """FCC NN bond midpoints are inversion centers: Moriya's rules give D = 0."""
    from nfit.crystal import symmetry_allowed_exchange_basis

    sites, orbits = generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=3.0)
    basis = symmetry_allowed_exchange_basis(FCC, sites, orbits[0])
    kinds = [element["kind"] for element in basis]
    assert kinds == ["symmetric", "symmetric"]


def test_p1_bond_allows_the_full_anisotropic_space():
    from nfit.crystal import symmetry_allowed_exchange_basis

    sites, orbits = generate_bond_orbits(P1_TRICLINIC, ["A", "B"], cutoff_angstrom=4.0)
    basis = symmetry_allowed_exchange_basis(P1_TRICLINIC, sites, orbits[0])
    kinds = [element["kind"] for element in basis]
    assert kinds.count("symmetric") == 5 and kinds.count("dm") == 3


def test_pyrochlore_sia_is_uniaxial_along_local_111():
    from nfit.crystal import symmetry_allowed_sia_basis

    basis = symmetry_allowed_sia_basis(PYROCHLORE, "M1")
    assert len(basis) == 1
    matrix = np.asarray(basis[0]["matrix"])
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    reference = 3.0 * np.outer(axis, axis) - np.eye(3)
    reference /= np.linalg.norm(reference)
    assert abs(np.sum(matrix * reference)) == pytest.approx(1.0, abs=1e-9)


def test_cubic_site_symmetry_forbids_sia():
    from nfit.crystal import symmetry_allowed_sia_basis

    assert symmetry_allowed_sia_basis(FCC, "Ni1") == []


def test_p1_site_sia_is_five_dimensional():
    from nfit.crystal import symmetry_allowed_sia_basis

    basis = symmetry_allowed_sia_basis(P1_TRICLINIC, "B")
    assert len(basis) == 5
    for element in basis:
        matrix = np.asarray(element["matrix"])
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
        assert abs(np.trace(matrix)) < 1e-9


def test_orbit_config_round_trips_symmetry_operations():
    sites, orbits = generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=3.0)
    payload = orbits_to_config(orbits)
    assert all("rotation" in bond and "reversed" in bond for bond in payload[0]["bonds"])
    rebuilt = orbits_from_config(payload)
    assert rebuilt[0].operations is not None
    for original, restored in zip(
        orbits[0].operations,
        rebuilt[0].operations,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(original.rotation), np.asarray(restored.rotation)
        )
        assert original.reverses == restored.reverses
    # Legacy payloads without rotations still load, with operations=None.
    legacy = [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]}]
    assert orbits_from_config(legacy)[0].operations is None


def test_screw_axis_subscripts_resolve_with_or_without_underscores():
    """``P 2_1/c`` is the ITA typography of ``P 21/c`` and must resolve.

    Gemmi accepts the underscore form for some symbols but not for short ones,
    so nfit normalizes the screw-axis subscript before lookup.
    """
    from nfit.crystal import _symmetry_operations

    assert len(_symmetry_operations("P 2_1/c")) == len(
        _symmetry_operations("P 21/c")
    )
    assert len(_symmetry_operations("P 2_1 2_1 2_1")) == 4
    with pytest.raises(ValueError, match="unknown space group"):
        _symmetry_operations("P not-a-group")


def test_bond_shells_survive_rounded_rhombohedral_site_coordinates():
    """Symmetry-equivalent bonds must land in one distance shell.

    Symmetry-expanded fractional coordinates are rounded, so a rhombohedral
    1/3, 2/3 site makes equivalent bond lengths differ by ~1e-5 Angstrom.
    Binning shells by a rounded value split such an orbit and made orbit
    detection raise; shells are clustered with a tolerance instead.
    """
    crystal = {
        "lattice": {
            "a": 4.0,
            "b": 4.0,
            "c": 10.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 120.0,
        },
        "spacegroup": "R -3 m",
        "sites": [
            {"label": "M1", "element": "Fe", "position": [0.0, 0.0, 0.0], "ion": ""}
        ],
    }
    sites, orbits = generate_spatial_bond_orbits(crystal, ["M1"], 4.6)
    assert orbits
    # Every bond of an orbit sits at the same length, and each orbit is closed
    # under the space group (a split shell would have raised above).
    lattice = lattice_vectors(crystal["lattice"])
    positions = np.asarray([site.position for site in sites])
    for orbit in orbits:
        lengths = [
            float(
                np.linalg.norm(
                    lattice
                    @ (
                        positions[bond.site_j]
                        + np.asarray(bond.offset, dtype=float)
                        - positions[bond.site_i]
                    )
                )
            )
            for bond in orbit.bonds
        ]
        assert max(lengths) - min(lengths) < 1.0e-3
