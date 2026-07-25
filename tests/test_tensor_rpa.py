import numpy as np
import pytest

from nfit.crystal import (
    generate_bond_orbits,
    orbits_to_config,
    site_rotations_to_config,
    sites_to_config,
    symmetry_allowed_exchange_basis,
    symmetry_allowed_sia_basis,
)
from nfit.fitting import reciprocal_basis_from_lattice_parameters
from nfit.spin_fluctuations import (
    build_rpa_geometry,
    heisenberg_rpa_chipp,
    reduce_site_network_with_tensors,
)
from nfit.tensor_rpa import (
    build_tensor_structure,
    cartesian_qhat_per_point,
    tensor_rpa_unpolarized_chipp,
    tensor_susceptibility,
)


PYROCHLORE = {
    "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "spacegroup": "F d -3 m:2",
    "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "Yb3"}],
}


def _two_site_geometry(rng, n=40):
    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11], [0.6, 0.2, 0.8]]
    orbits = [
        {"label": "J1", "bonds": [
            {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
            {"site_i": 1, "site_j": 2, "offset": [0, -1, 0]}]},
        {"label": "J2", "bonds": [{"site_i": 0, "site_j": 2, "offset": [0, 0, 1]}]},
    ]
    hkl = rng.uniform(-2.0, 2.0, size=(n, 3))
    geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbits)
    return positions, orbits, geometry


def test_tensor_heisenberg_only_matches_scalar_intensity():
    """Tier-A tensor path with no anisotropy equals 2 * scalar chi''."""
    rng = np.random.default_rng(0)
    positions, orbits, geometry = _two_site_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    kwargs = dict(chi0=0.4, gamma0=2.5)
    j_values = {"J1": 0.12, "J2": -0.07}

    scalar = heisenberg_rpa_chipp(geometry, E, j_values=j_values, **kwargs)
    structure = build_tensor_structure(geometry, positions, orbits)
    q_hat = cartesian_qhat_per_point(geometry, np.eye(3))
    tensor = tensor_rpa_unpolarized_chipp(
        structure, geometry, E, q_hat, param_values=j_values, **kwargs
    )
    np.testing.assert_allclose(tensor, 2.0 * scalar, rtol=1e-11, atol=1e-14)


def test_tensor_heisenberg_susceptibility_is_isotropic():
    rng = np.random.default_rng(2)
    positions, orbits, geometry = _two_site_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    structure = build_tensor_structure(geometry, positions, orbits)
    chi = tensor_susceptibility(
        structure, geometry, E, chi0=0.4, gamma0=2.5, param_values={"J1": 0.1, "J2": -0.05}
    )
    # chi = scalar * I3: diagonal equal, off-diagonal zero.
    np.testing.assert_allclose(chi[:, 0, 1], 0.0, atol=1e-13)
    np.testing.assert_allclose(chi[:, 0, 2], 0.0, atol=1e-13)
    np.testing.assert_allclose(chi[:, 0, 0], chi[:, 1, 1], atol=1e-13)
    np.testing.assert_allclose(chi[:, 0, 0], chi[:, 2, 2], atol=1e-13)


def test_anisotropy_vanishing_reduces_to_heisenberg():
    rng = np.random.default_rng(1)
    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn + 0.01)
    positions = sites_to_config(sites)
    orbit_config = orbits_to_config(orbits)
    basis = symmetry_allowed_exchange_basis(PYROCHLORE, sites, orbits[0])

    hkl = rng.uniform(-2.0, 2.0, size=(60, 3))
    geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbit_config)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    basis_matrix = reciprocal_basis_from_lattice_parameters(a, a, a, 90, 90, 90)
    q_hat = cartesian_qhat_per_point(geometry, basis_matrix)
    kwargs = dict(chi0=0.02, gamma0=3.0)

    anisotropy = {"J1": {"enabled": True, "basis": basis}}
    structure = build_tensor_structure(
        geometry, positions, orbits=orbit_config, lattice=PYROCHLORE["lattice"], anisotropy=anisotropy
    )
    heisenberg = build_tensor_structure(geometry, positions, orbits=orbit_config)

    values = {"J1": 0.05, "J2": 0.02}
    with_zero_aniso = dict(values)
    with_zero_aniso.update({name: 0.0 for name in structure.parameter_names if name not in values})
    full = tensor_rpa_unpolarized_chipp(structure, geometry, E, q_hat, param_values=with_zero_aniso, **kwargs)
    plain = tensor_rpa_unpolarized_chipp(heisenberg, geometry, E, q_hat, param_values=values, **kwargs)
    np.testing.assert_allclose(full, plain, rtol=1e-12, atol=1e-14)

    # An anisotropic coefficient breaks the isotropy and changes the intensity.
    with_aniso = dict(with_zero_aniso)
    with_aniso["J1_S1"] = 0.01
    perturbed = tensor_rpa_unpolarized_chipp(structure, geometry, E, q_hat, param_values=with_aniso, **kwargs)
    assert np.max(np.abs(perturbed - full)) > 1e-8


def test_tensor_instability_raises():
    rng = np.random.default_rng(3)
    positions, orbits, geometry = _two_site_geometry(rng, n=5)
    E = np.full(geometry.point_index.size, 1.0)
    structure = build_tensor_structure(geometry, positions, orbits)
    with pytest.raises(ValueError, match="instability"):
        tensor_rpa_unpolarized_chipp(
            structure, geometry, E, cartesian_qhat_per_point(geometry, np.eye(3)),
            chi0=5.0, gamma0=1.0, param_values={"J1": 1.0, "J2": 1.0},
        )


def test_pyrochlore_tensor_reduction_matches_full_cell():
    """16 -> 4 primitive fold is exact with anisotropic exchange and SIA on.

    Pure lattice translations do not rotate spins, so the bond-resolved
    anisotropic exchange tensors and the on-site single-ion anisotropy are each
    invariant under the F-centering fold. chi'' from the reduced 4-site network
    must match the full 16-site cell to machine precision.
    """

    rng = np.random.default_rng(7)
    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn + 0.01)
    positions = sites_to_config(sites)  # 16 sites
    orbit_config = orbits_to_config(orbits)
    site_rotations = site_rotations_to_config(sites)
    aniso_basis = symmetry_allowed_exchange_basis(PYROCHLORE, sites, orbits[0])
    sia_basis = symmetry_allowed_sia_basis(PYROCHLORE, "M1")
    assert len(positions) == 16 and sia_basis  # trigonal site: 1 allowed tensor

    anisotropy = {"J1": {"enabled": True, "basis": aniso_basis}}
    sia = {"M1": {"enabled": True, "sites": list(range(len(positions))), "basis": sia_basis}}

    r_positions, r_orbits, r_rotations, r_sia = reduce_site_network_with_tensors(
        positions, orbit_config, site_rotations=site_rotations, sia=sia
    )
    assert np.asarray(r_positions).shape == (4, 3)
    assert len(r_rotations) == 4
    assert len(r_sia["M1"]["sites"]) == 4

    hkl = rng.uniform(-2.5, 2.5, size=(48, 3))
    E = rng.uniform(0.3, 6.0, size=48)
    basis_matrix = reciprocal_basis_from_lattice_parameters(a, a, a, 90, 90, 90)
    kwargs = dict(chi0=0.02, gamma0=3.0)
    params = {
        "J1": 0.05,
        "J1_S1": 0.01,
        "J1_S2": -0.008,
        "J1_D1": 0.006,
        "K1_M1": 0.02,
    }

    def evaluate(pos, orbs, rots, sia_spec):
        geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], pos, orbs)
        structure = build_tensor_structure(
            geometry,
            pos,
            orbits=orbs,
            lattice=PYROCHLORE["lattice"],
            anisotropy=anisotropy,
            sia=sia_spec,
            site_rotations=rots,
        )
        q_hat = cartesian_qhat_per_point(geometry, basis_matrix)
        return tensor_rpa_unpolarized_chipp(
            structure, geometry, E, q_hat, param_values=params, **kwargs
        )

    full = evaluate(positions, orbit_config, site_rotations, sia)
    folded = evaluate(r_positions, r_orbits, r_rotations, r_sia)
    np.testing.assert_allclose(folded, full, rtol=1e-11, atol=1e-14)


def test_tensor_reduction_declines_when_dipole_active():
    """The dipole Ewald sum needs the primitive lattice, so folding is skipped.

    Keeping the caller's conventional lattice on a reduced cell would give the
    wrong long-range sum, so the tensor reduction returns the full network
    unchanged whenever the dipole term is enabled.
    """

    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn + 0.01)
    positions = sites_to_config(sites)
    orbit_config = orbits_to_config(orbits)
    site_rotations = site_rotations_to_config(sites)

    r_positions, r_orbits, r_rotations, _ = reduce_site_network_with_tensors(
        positions,
        orbit_config,
        site_rotations=site_rotations,
        dipole_enabled=True,
    )
    assert np.asarray(r_positions).shape == (16, 3)
    assert len(r_orbits) == len(orbit_config)
    assert len(r_rotations) == 16


def _rotation_about_axis(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def test_intensity_is_invariant_under_a_global_frame_rotation():
    """Rotating every spin tensor and Q-hat by the same R leaves I unchanged.

    chi rotates as R chi R^T and the unpolarized weight as R W R^T, so
    sum W chi'' is a rotational invariant. This locks the polarization
    contraction and the tensor bookkeeping together.
    """
    import copy

    rng = np.random.default_rng(7)
    positions, orbits, geometry = _two_site_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    chi0, gamma0 = 0.3, 2.5

    # Give the structure genuine anisotropy by hand-setting per-bond tensors.
    structure = build_tensor_structure(geometry, positions, orbits)
    sym = np.array([[0.2, 0.1, 0.0], [0.1, -0.2, 0.05], [0.0, 0.05, 0.0]])
    for term in structure.terms["J1"]:
        term.tensor = term.tensor + sym.astype(complex)

    q_hat = cartesian_qhat_per_point(geometry, np.eye(3))
    values = {"J1": 0.1, "J2": -0.04}
    base = tensor_rpa_unpolarized_chipp(
        structure, geometry, E, q_hat, param_values=values, chi0=chi0, gamma0=gamma0
    )

    rotation = _rotation_about_axis([0.3, -0.5, 0.8], 0.7)
    rotated = copy.deepcopy(structure)
    for terms in rotated.terms.values():
        for term in terms:
            term.tensor = rotation @ term.tensor @ rotation.T
    rotated_qhat = q_hat @ rotation.T
    rotated_intensity = tensor_rpa_unpolarized_chipp(
        rotated, geometry, E, rotated_qhat, param_values=values, chi0=chi0, gamma0=gamma0
    )
    np.testing.assert_allclose(rotated_intensity, base, rtol=1e-10, atol=1e-13)


def test_zeeman_reduces_to_tier_a_at_zero_field():
    from nfit.tensor_rpa import (
        tensor_rpa_zeeman_unpolarized_chipp,
        zeeman_cartesian_propagator,
    )

    rng = np.random.default_rng(4)
    positions, orbits, geometry = _two_site_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    structure = build_tensor_structure(geometry, positions, orbits)
    q_hat = cartesian_qhat_per_point(geometry, np.eye(3))
    values = {"J1": 0.12, "J2": -0.07}
    chi0, gamma0 = 0.4, 2.5

    tier_a = tensor_rpa_unpolarized_chipp(
        structure, geometry, E, q_hat, param_values=values, chi0=chi0, gamma0=gamma0
    )
    propagator = zeeman_cartesian_propagator(
        E, np.array([0.0, 0.0, 1.0]), chi0=chi0, gamma0=gamma0, omega_larmor=0.0
    )
    tier_b = tensor_rpa_zeeman_unpolarized_chipp(
        structure, geometry, E, q_hat, propagator, param_values=values
    )
    np.testing.assert_allclose(tier_b, tier_a, rtol=1e-11, atol=1e-14)


def test_zeeman_propagator_is_gyrotropic_and_larmor_resonant():
    from nfit.tensor_rpa import (
        MU_B_MEV_PER_T,
        tensor_zeeman_susceptibility,
        zeeman_cartesian_propagator,
    )

    b_hat = np.array([0.0, 0.0, 1.0])
    E = np.linspace(-3.0, 6.0, 9)
    zero = zeeman_cartesian_propagator(E, b_hat, chi0=0.5, gamma0=1.0, omega_larmor=0.0)
    # Zero field: scalar chi0(omega) I3.
    np.testing.assert_allclose(zero[:, 0, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(zero[:, 0, 0], zero[:, 2, 2], atol=1e-14)

    field = zeeman_cartesian_propagator(E, b_hat, chi0=0.5, gamma0=1.0, omega_larmor=1.5)
    # Transverse block is gyrotropic (antisymmetric), longitudinal unchanged.
    np.testing.assert_allclose(field[:, 0, 1], -field[:, 1, 0], atol=1e-14)
    assert np.max(np.abs(field[:, 0, 1])) > 1e-6

    # A decoupled site's transverse chi'' resonates near omega_L (relaxational
    # peak is at omega_L + Gamma).
    line = np.linspace(0.05, 6.0, 400)
    geometry = build_rpa_geometry(
        np.zeros(400), np.zeros(400), np.zeros(400), [[0.0, 0.0, 0.0]],
        [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
    )
    structure = build_tensor_structure(
        geometry, [[0.0, 0.0, 0.0]],
        [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
    )
    gamma = 0.3
    propagator = zeeman_cartesian_propagator(line, b_hat, chi0=0.5, gamma0=gamma, omega_larmor=2.5)
    chi = tensor_zeeman_susceptibility(structure, geometry, line, propagator, param_values={"J1": 0.0})
    peak = line[np.argmax(chi[:, 0, 0].imag)]
    assert peak == pytest.approx(2.5 + gamma, abs=0.05)
