import numpy as np
import pytest

pytest.importorskip("gemmi")

from metallix.crystal import (
    Bond,
    crystal_from_cif,
    expand_magnetic_sites,
    generate_bond_orbits,
    orbits_from_config,
    orbits_to_config,
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
    from metallix.spin_fluctuations import build_rpa_geometry, rpa_exchange_matrix

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

    crystal["sites"][0]["ion"] = "Ni2"
    sites, orbits = generate_bond_orbits(crystal, ["Ni1"], cutoff_angstrom=3.0)
    assert len(sites) == 4
    assert orbits[0].label == "J1"


def test_generate_bond_orbits_validates_cutoff():
    with pytest.raises(ValueError, match="cutoff"):
        generate_bond_orbits(FCC, ["Ni1"], cutoff_angstrom=0.0)
