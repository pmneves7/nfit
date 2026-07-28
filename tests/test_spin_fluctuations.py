import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nfit.spin_fluctuations import (
    build_rpa_geometry,
    heisenberg_rpa_chipp,
    local_relaxational_chipp,
    mmp_chipp,
    reduce_site_network,
    rpa_exchange_matrix,
)


def _chain_geometry(H, offsets=((1, 0, 0),)):
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 0, "offset": list(offset)} for offset in offsets
            ],
        }
    ]
    zeros = np.zeros_like(np.asarray(H, dtype=float))
    return build_rpa_geometry(H, zeros, zeros, [[0.0, 0.0, 0.0]], orbits)


def _two_site_geometry(rng, n_points=7):
    hkl = rng.uniform(-2.0, 2.0, size=(n_points, 3))
    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]]
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 0, "site_j": 1, "offset": [-1, 0, 0]},
            ],
        },
        {
            "label": "J2",
            "bonds": [{"site_i": 0, "site_j": 0, "offset": [0, 0, 1]}],
        },
    ]
    geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbits)
    return geometry


def test_local_relaxational_is_odd_and_peaks_at_gamma():
    E = np.linspace(-10.0, 10.0, 201)
    chipp = local_relaxational_chipp(E, chi_loc=2.0, gamma=3.0)
    np.testing.assert_allclose(chipp, -chipp[::-1], atol=1e-14)
    peak = E[np.argmax(chipp)]
    assert peak == pytest.approx(3.0, abs=0.11)
    np.testing.assert_allclose(
        local_relaxational_chipp(3.0, chi_loc=2.0, gamma=3.0), 2.0 * 0.5 * 3.0 / 3.0
    )


def test_mmp_chipp_is_odd_in_energy_and_peaks_at_q0():
    E = np.linspace(-5.0, 5.0, 101)
    chipp = mmp_chipp(0.25, E, chi_pk=1.0, xi=2.0, omega_sf=1.5)
    np.testing.assert_allclose(chipp, -chipp[::-1], atol=1e-14)

    q_sq = np.array([0.0, 0.5, 2.0])
    at_q0 = mmp_chipp(q_sq, 1.0, chi_pk=1.0, xi=2.0, omega_sf=1.0)
    assert np.all(np.diff(at_q0) < 0.0)
    # at Q0 the lineshape is relaxational with Gamma = omega_sf
    E = np.linspace(-8.0, 8.0, 161)
    np.testing.assert_allclose(
        mmp_chipp(0.0, E, chi_pk=2.0, xi=3.0, omega_sf=1.7),
        local_relaxational_chipp(E, chi_loc=2.0, gamma=1.7),
        rtol=1e-12,
    )


def test_mmp_requires_positive_omega_sf():
    with pytest.raises(ValueError):
        mmp_chipp(0.0, 1.0, chi_pk=1.0, xi=1.0, omega_sf=0.0)


def test_rpa_weights_sum_rule_two_site_random_crystal():
    rng = np.random.default_rng(7)
    geometry = _two_site_geometry(rng)
    exchange = rpa_exchange_matrix(geometry, {"J1": 1.3, "J2": -0.4})
    # Hermiticity of J(Q)
    np.testing.assert_allclose(exchange, np.conj(np.transpose(exchange, (0, 2, 1))), atol=1e-12)
    _, modes = np.linalg.eigh(exchange)
    amplitudes = modes.sum(axis=1)
    weights = np.abs(amplitudes) ** 2 / geometry.n_sites
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)


def test_rpa_is_invariant_under_cell_description():
    """The same physical chain must give identical chi'' in any cell choice.

    A uniform nearest-neighbor chain is described three ways: a 1-site cell,
    a 2-site doubled cell (half the r.l.u. unit), and the doubled cell with a
    shifted origin. This is the decisive phase-convention lock: pairing the
    extended-zone J(Q) with anything other than the uniform neutron weight
    breaks it (the sum rule alone holds for any unit-modulus weight vector).
    """

    J, chi0, gamma0 = 0.3, 0.8, 2.0
    Qa = np.linspace(0.05, 0.95, 7)
    E = np.full(Qa.shape, 1.3)
    zeros = np.zeros_like(Qa)

    single = build_rpa_geometry(
        Qa,
        zeros,
        zeros,
        [[0.0, 0.0, 0.0]],
        [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
    )
    doubled_orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 1, "site_j": 0, "offset": [1, 0, 0]},
            ],
        }
    ]
    # The doubled cell is twice as long, so the same absolute momentum is 2*Qa
    # in its reciprocal lattice units.
    doubled = build_rpa_geometry(
        2.0 * Qa, zeros, zeros, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], doubled_orbits
    )
    shifted = build_rpa_geometry(
        2.0 * Qa, zeros, zeros, [[0.13, 0.0, 0.0], [0.63, 0.0, 0.0]], doubled_orbits
    )

    reference = heisenberg_rpa_chipp(single, E, chi0=chi0, gamma0=gamma0, j_values={"J1": J})
    for geometry in (doubled, shifted):
        chipp = heisenberg_rpa_chipp(geometry, E, chi0=chi0, gamma0=gamma0, j_values={"J1": J})
        np.testing.assert_allclose(chipp, reference, rtol=1e-12)


def test_rpa_with_zero_exchange_reduces_to_local_relaxational():
    rng = np.random.default_rng(3)
    geometry = _two_site_geometry(rng)
    E = np.linspace(-4.0, 6.0, geometry.point_index.size)
    coupled = heisenberg_rpa_chipp(
        geometry, E, chi0=1.5, gamma0=2.5, j_values={"J1": 0.0, "J2": 0.0}
    )
    local = local_relaxational_chipp(E, chi_loc=1.5, gamma=2.5)
    np.testing.assert_allclose(coupled, local, rtol=1e-12)


def test_rpa_chain_matches_closed_form_dispersion():
    H = np.linspace(0.0, 1.0, 21)
    E = np.full(H.shape, 1.3)
    chi0, gamma0, J = 0.8, 2.0, 0.3
    geometry = _chain_geometry(H)
    chipp = heisenberg_rpa_chipp(geometry, E, chi0=chi0, gamma0=gamma0, j_values={"J1": J})

    denom = 1.0 - 2.0 * J * chi0 * np.cos(2.0 * np.pi * H)
    chi_q = chi0 / denom
    gamma_q = gamma0 * denom
    expected = chi_q * gamma_q * E / (E**2 + gamma_q**2)
    np.testing.assert_allclose(chipp, expected, rtol=1e-12)


def test_rpa_is_odd_in_energy():
    H = np.array([0.1, 0.1, 0.4, 0.4])
    E = np.array([1.0, -1.0, 2.5, -2.5])
    geometry = _chain_geometry(H)
    chipp = heisenberg_rpa_chipp(geometry, E, chi0=0.5, gamma0=1.0, j_values={"J1": 0.2})
    np.testing.assert_allclose(chipp[0], -chipp[1], rtol=1e-12)
    np.testing.assert_allclose(chipp[2], -chipp[3], rtol=1e-12)


def test_rpa_instability_raises():
    geometry = _chain_geometry(np.array([0.0]))
    with pytest.raises(ValueError, match="instability"):
        heisenberg_rpa_chipp(
            geometry, np.array([1.0]), chi0=1.0, gamma0=1.0, j_values={"J1": 0.5}
        )


def test_rpa_relaxation_softens_at_ordering_vector():
    # antiferromagnetic chain: J < 0 favors q = 0.5; the mode weight sum
    # rule means chi'' at small E is largest where Gamma(q) is smallest
    H = np.array([0.0, 0.25, 0.5])
    E = np.full(H.shape, 0.05)
    geometry = _chain_geometry(H)
    chipp = heisenberg_rpa_chipp(
        geometry, E, chi0=0.9, gamma0=2.0, j_values={"J1": -0.5}
    )
    assert chipp[2] > chipp[1] > chipp[0]


def test_geometry_deduplicates_repeated_q_points():
    H = np.array([0.2, 0.2, 0.2, 0.7])
    geometry = _chain_geometry(H)
    assert geometry.n_q == 2
    assert geometry.point_index.size == 4


def test_geometry_validates_inputs():
    with pytest.raises(ValueError, match="site index"):
        build_rpa_geometry(
            [0.0],
            [0.0],
            [0.0],
            [[0.0, 0.0, 0.0]],
            [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 5, "offset": [0, 0, 0]}]}],
        )
    with pytest.raises(ValueError, match="no bonds"):
        build_rpa_geometry([0.0], [0.0], [0.0], [[0.0, 0.0, 0.0]], [{"label": "J1", "bonds": []}])
    geometry = _chain_geometry(np.array([0.1]))
    with pytest.raises(KeyError, match="J9"):
        rpa_exchange_matrix(geometry, {"J9": 1.0})
    with pytest.raises(ValueError, match="gamma0"):
        heisenberg_rpa_chipp(geometry, [1.0], chi0=1.0, gamma0=0.0, j_values={"J1": 0.0})
    with pytest.raises(ValueError, match="E has"):
        heisenberg_rpa_chipp(geometry, [1.0, 2.0], chi0=1.0, gamma0=1.0, j_values={"J1": 0.0})


_DOUBLED_CHAIN_ORBITS = [
    {
        "label": "J1",
        "bonds": [
            {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
            {"site_i": 1, "site_j": 0, "offset": [1, 0, 0]},
        ],
    }
]


def test_reduce_site_network_folds_doubled_chain():
    positions, orbits = reduce_site_network(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], _DOUBLED_CHAIN_ORBITS
    )
    assert positions.shape == (1, 3)
    (orbit,) = orbits
    assert orbit["label"] == "J1"
    (bond,) = orbit["bonds"]
    assert bond["site_i"] == bond["site_j"] == 0
    np.testing.assert_allclose(np.abs(bond["offset"]), [0.5, 0.0, 0.0])

    # chi'' is unchanged by the fold
    Q = np.linspace(0.05, 1.95, 9)
    E = np.full(Q.shape, 1.3)
    zeros = np.zeros_like(Q)
    full = build_rpa_geometry(
        Q, zeros, zeros, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], _DOUBLED_CHAIN_ORBITS
    )
    reduced = build_rpa_geometry(Q, zeros, zeros, positions, orbits)
    np.testing.assert_allclose(
        heisenberg_rpa_chipp(reduced, E, chi0=0.8, gamma0=2.0, j_values={"J1": 0.3}),
        heisenberg_rpa_chipp(full, E, chi0=0.8, gamma0=2.0, j_values={"J1": 0.3}),
        rtol=1e-12,
    )


def test_reduce_site_network_folds_body_centered_pair():
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 0, "site_j": 1, "offset": [-1, 0, 0]},
                {"site_i": 0, "site_j": 1, "offset": [0, -1, 0]},
                {"site_i": 0, "site_j": 1, "offset": [0, 0, -1]},
                {"site_i": 0, "site_j": 1, "offset": [-1, -1, 0]},
                {"site_i": 0, "site_j": 1, "offset": [-1, 0, -1]},
                {"site_i": 0, "site_j": 1, "offset": [0, -1, -1]},
                {"site_i": 0, "site_j": 1, "offset": [-1, -1, -1]},
            ],
        }
    ]
    positions, reduced_orbits = reduce_site_network(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], orbits
    )
    assert positions.shape == (1, 3)
    assert len(reduced_orbits[0]["bonds"]) == 4  # eight bonds / group order two


def test_reduce_site_network_declines_broken_symmetry():
    # Same doubled chain, but the two bonds carry different labels: the
    # half-cell translation no longer preserves the labeled network.
    orbits = [
        {"label": "Ja", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
        {"label": "Jb", "bonds": [{"site_i": 1, "site_j": 0, "offset": [1, 0, 0]}]},
    ]
    site_positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    positions, reduced = reduce_site_network(site_positions, orbits)
    assert positions.shape == (2, 3)
    assert [orbit["label"] for orbit in reduced] == ["Ja", "Jb"]
    assert reduced[0]["bonds"] == orbits[0]["bonds"]


def test_reduce_site_network_leaves_primitive_networks_alone():
    positions, reduced = reduce_site_network(
        [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]],
        [
            {
                "label": "J1",
                "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}],
            }
        ],
    )
    assert positions.shape == (2, 3)
    assert len(reduced[0]["bonds"]) == 1


def _gradient_geometry(rng, n_points=25):
    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11], [0.6, 0.2, 0.8]]
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 1, "site_j": 2, "offset": [0, -1, 0]},
            ],
        },
        {"label": "J2", "bonds": [{"site_i": 0, "site_j": 2, "offset": [0, 0, 1]}]},
    ]
    hkl = rng.uniform(-2.0, 2.0, size=(n_points, 3))
    return build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbits)


def test_rpa_gradients_value_matches_plain_evaluation():
    from nfit.spin_fluctuations import heisenberg_rpa_chipp_and_gradients

    rng = np.random.default_rng(5)
    geometry = _gradient_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    kwargs = dict(chi0=0.4, gamma0=2.5, j_values={"J1": 0.12, "J2": -0.07})
    value, _ = heisenberg_rpa_chipp_and_gradients(geometry, E, **kwargs)
    np.testing.assert_allclose(value, heisenberg_rpa_chipp(geometry, E, **kwargs), rtol=1e-11)


def test_rpa_gradients_match_finite_differences():
    from nfit.spin_fluctuations import heisenberg_rpa_chipp_and_gradients

    rng = np.random.default_rng(5)
    geometry = _gradient_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    chi0, gamma0 = 0.4, 2.5
    j_values = {"J1": 0.12, "J2": -0.07}
    _, grads = heisenberg_rpa_chipp_and_gradients(
        geometry, E, chi0=chi0, gamma0=gamma0, j_values=j_values
    )

    def evaluate(chi0_, gamma0_, j_):
        return heisenberg_rpa_chipp(geometry, E, chi0=chi0_, gamma0=gamma0_, j_values=j_)

    h = 1e-6
    fd = {
        "chi0": (evaluate(chi0 + h, gamma0, j_values) - evaluate(chi0 - h, gamma0, j_values)) / (2 * h),
        "gamma0": (evaluate(chi0, gamma0 + h, j_values) - evaluate(chi0, gamma0 - h, j_values)) / (2 * h),
    }
    for label in j_values:
        plus = {**j_values, label: j_values[label] + h}
        minus = {**j_values, label: j_values[label] - h}
        fd[label] = (evaluate(chi0, gamma0, plus) - evaluate(chi0, gamma0, minus)) / (2 * h)

    for name, reference in fd.items():
        np.testing.assert_allclose(grads[name], reference, rtol=2e-6, atol=1e-9)


def test_rpa_gradients_exact_at_band_degeneracy():
    # A uniform three-site ring at Q=0 has a doubly degenerate J(Q); the
    # gradient must still match finite differences (no eigenvector-derivative
    # blow-up, because the resolvent form never differentiates eigenvectors).
    from nfit.spin_fluctuations import heisenberg_rpa_chipp_and_gradients

    positions = [[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0], [2 / 3, 0.0, 0.0]]
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 1, "site_j": 2, "offset": [0, 0, 0]},
                {"site_i": 2, "site_j": 0, "offset": [1, 0, 0]},
            ],
        }
    ]
    H = np.array([0.0, 0.0])
    zeros = np.zeros_like(H)
    E = np.array([1.3, 2.1])
    geometry = build_rpa_geometry(H, zeros, zeros, positions, orbits)
    chi0, gamma0 = 0.5, 2.0
    _, grads = heisenberg_rpa_chipp_and_gradients(
        geometry, E, chi0=chi0, gamma0=gamma0, j_values={"J1": 0.2}
    )
    h = 1e-6
    fd = (
        heisenberg_rpa_chipp(geometry, E, chi0=chi0, gamma0=gamma0, j_values={"J1": 0.2 + h})
        - heisenberg_rpa_chipp(geometry, E, chi0=chi0, gamma0=gamma0, j_values={"J1": 0.2 - h})
    ) / (2 * h)
    np.testing.assert_allclose(grads["J1"], fd, rtol=1e-6, atol=1e-9)


def test_numba_openmp_initialization_avoids_deprecated_nested_info(tmp_path):
    pytest.importorskip("numba")
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nfit import spin_fluctuations as sf; "
            "print(sf.available_rpa_backends()); "
            "print(sf._NUMBA_KERNELS._OPENMP_PARALLEL_INITIALIZED)",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "numba" in result.stdout
    assert "True" in result.stdout
    assert "omp_set_nested routine deprecated" not in result.stderr


def test_numba_backend_matches_numpy_value_and_gradients():
    pytest.importorskip("numba")
    from nfit import spin_fluctuations as sf

    assert "numba" in sf.available_rpa_backends()

    rng = np.random.default_rng(9)
    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11], [0.6, 0.2, 0.8]]
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 1, "site_j": 2, "offset": [0, -1, 0]},
            ],
        },
        {"label": "J2", "bonds": [{"site_i": 0, "site_j": 2, "offset": [0, 0, 1]}]},
    ]
    hkl = rng.uniform(-2.0, 2.0, size=(2000, 3))
    geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbits)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    kwargs = dict(chi0=0.4, gamma0=2.5, j_values={"J1": 0.12, "J2": -0.07})

    def clear_cache():
        # The eigendecomposition is cached on the geometry keyed by j_values;
        # clear it between backends so the numba path really recomputes the
        # eigendecomposition (with the numba solver) instead of reusing the
        # numpy backend's cached modes.
        geometry.__dict__.pop("_rpa_modes_cache", None)

    try:
        sf.set_rpa_backend("numpy")
        clear_cache()
        val_np = heisenberg_rpa_chipp(geometry, E, **kwargs)
        clear_cache()
        cp_np, gr_np = sf.heisenberg_rpa_chipp_and_gradients(geometry, E, **kwargs)
        sf.set_rpa_backend("numba")
        # Guard against a vacuous pass: the forced numba backend must actually
        # be selected, and it must actually use the numba eigensolver (not the
        # cached numpy modes), otherwise this would compare numpy against numpy.
        assert sf._select_rpa_backend(E.size) == "numba"
        assert sf._use_numba_eigh(geometry.n_q, geometry.n_sites)
        clear_cache()
        val_nb = heisenberg_rpa_chipp(geometry, E, **kwargs)
        clear_cache()
        cp_nb, gr_nb = sf.heisenberg_rpa_chipp_and_gradients(geometry, E, **kwargs)
    finally:
        sf.set_rpa_backend("auto")

    # Different backend AND a different eigensolver (numba Jacobi vs LAPACK), so
    # results agree to floating-point precision, not bit-for-bit.
    np.testing.assert_allclose(val_nb, val_np, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(cp_nb, cp_np, rtol=1e-10, atol=1e-13)
    for name in gr_np:
        np.testing.assert_allclose(gr_nb[name], gr_np[name], rtol=1e-9, atol=1e-12)


def test_backend_selection_is_size_gated():
    from nfit import spin_fluctuations as sf

    try:
        sf.set_rpa_backend("auto")
        # Tiny problems always resolve to numpy regardless of installed backends.
        assert sf._select_rpa_backend(100) == "numpy"
        sf.set_rpa_backend("numpy")
        assert sf._select_rpa_backend(10_000_000) == "numpy"
    finally:
        sf.set_rpa_backend("auto")

    with pytest.raises(ValueError):
        sf.set_rpa_backend("nonsense")


def test_forcing_unavailable_backend_falls_back_to_numpy():
    from nfit import spin_fluctuations as sf

    backends = sf.available_rpa_backends()
    try:
        if "cupy" not in backends:
            sf.set_rpa_backend("cupy")
            assert sf._select_rpa_backend(10_000_000) == "numpy"
        if "numba" not in backends:
            sf.set_rpa_backend("numba")
            assert sf._select_rpa_backend(10_000_000) == "numpy"
    finally:
        sf.set_rpa_backend("auto")
    assert "numpy" in backends


def test_thread_budget_override_and_affinity_default():
    from nfit import spin_fluctuations as sf

    try:
        sf.set_num_threads(3)
        assert sf.num_threads() == 3
        sf.set_num_threads(None)  # back to auto
        assert sf.num_threads() == sf._detect_cpu_budget()
        # env override is honored when no explicit override is set
        import os

        os.environ["NFIT_NUM_THREADS"] = "5"
        try:
            assert sf.num_threads() == 5
        finally:
            del os.environ["NFIT_NUM_THREADS"]
    finally:
        sf.set_num_threads(None)


def test_batched_eigh_matches_single_thread_when_forced_parallel():
    from nfit import spin_fluctuations as sf

    rng = np.random.default_rng(4)
    # N=16 crosses the work threshold so the threaded, BLAS-pinned path runs.
    a = rng.standard_normal((6000, 16, 16)) + 1j * rng.standard_normal((6000, 16, 16))
    h = a + np.conj(np.transpose(a, (0, 2, 1)))
    lam_ref, _ = np.linalg.eigh(h)
    lam, vec = sf._batched_eigh(h)
    np.testing.assert_allclose(lam, lam_ref, rtol=1e-12, atol=1e-12)
    # reconstruct H from the returned decomposition as an eigenvector check
    recon = np.einsum("qan,qn,qbn->qab", vec, lam, np.conj(vec))
    np.testing.assert_allclose(recon, h, rtol=1e-10, atol=1e-10)


def test_numba_eigensolver_matches_lapack():
    pytest.importorskip("numba")
    from nfit import _rpa_numba

    rng = np.random.default_rng(21)
    for n in (2, 3, 4, 8, 12, 16):
        m = rng.standard_normal((300, n, n)) + 1j * rng.standard_normal((300, n, n))
        hermitian = m + np.conj(np.transpose(m, (0, 2, 1)))
        lam, vec = _rpa_numba.batched_hermitian_eigh(np.ascontiguousarray(hermitian))
        # A valid decomposition: eigenvalues match LAPACK (sorted), V unitary,
        # and V diag(lam) V^dagger reconstructs the matrix. Order/phase are free.
        np.testing.assert_allclose(
            np.sort(lam, axis=1), np.linalg.eigvalsh(hermitian), rtol=1e-9, atol=1e-9
        )
        recon = np.einsum("bij,bj,bkj->bik", vec, lam, np.conj(vec))
        np.testing.assert_allclose(recon, hermitian, rtol=1e-9, atol=1e-9)
        identity = np.einsum("bij,bkj->bik", vec, np.conj(vec))
        np.testing.assert_allclose(identity, np.broadcast_to(np.eye(n), (300, n, n)), atol=1e-9)

    # Degenerate / diagonal / zero matrices must not break the sweep.
    special = np.stack(
        [
            np.diag([1.0, 1.0, 1.0, 3.0]).astype(complex),
            np.zeros((4, 4), dtype=complex),
            np.diag([2.0, 2.0, 5.0, 5.0]).astype(complex),
        ]
    )
    lam, vec = _rpa_numba.batched_hermitian_eigh(np.ascontiguousarray(special))
    recon = np.einsum("bij,bj,bkj->bik", vec, lam, np.conj(vec))
    np.testing.assert_allclose(recon, special, atol=1e-12)


def test_numba_eigh_leaves_observable_numerically_unchanged():
    """The numba eigensolver path must reproduce the LAPACK-path observable."""
    pytest.importorskip("numba")
    from nfit import spin_fluctuations as sf

    rng = np.random.default_rng(3)
    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11], [0.6, 0.2, 0.8]]
    orbits = [
        {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
        {"label": "J2", "bonds": [{"site_i": 1, "site_j": 2, "offset": [0, 0, 1]}]},
    ]
    hkl = rng.uniform(-2.0, 2.0, size=(2500, 3))
    geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbits)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    kwargs = dict(chi0=0.5, gamma0=2.0, j_values={"J1": 0.1, "J2": -0.06})

    lam_lapack, modes_lapack = sf._batched_eigh(sf.rpa_exchange_matrix(geometry, kwargs["j_values"]))
    lam_numba, modes_numba = sf._NUMBA_KERNELS.batched_hermitian_eigh(
        np.ascontiguousarray(sf.rpa_exchange_matrix(geometry, kwargs["j_values"]))
    )
    # eigenvalue sets agree
    np.testing.assert_allclose(
        np.sort(lam_numba, axis=1), np.sort(lam_lapack, axis=1), rtol=1e-9, atol=1e-9
    )

    def chipp_with(lam, modes):
        # Force the exact modes into the geometry cache, then evaluate via the
        # numpy contraction so only the eigensolver differs.
        key = tuple(sorted((str(k), float(v)) for k, v in kwargs["j_values"].items()))
        geometry.__dict__["_rpa_modes_cache"] = (key, lam, modes)
        try:
            sf.set_rpa_backend("numpy")
            return sf.heisenberg_rpa_chipp_and_gradients(geometry, E, **kwargs)
        finally:
            sf.set_rpa_backend("auto")
            geometry.__dict__.pop("_rpa_modes_cache", None)

    cp_l, gr_l = chipp_with(lam_lapack, modes_lapack)
    cp_n, gr_n = chipp_with(lam_numba, modes_numba)
    np.testing.assert_allclose(cp_n, cp_l, rtol=1e-10, atol=1e-13)
    for name in gr_l:
        np.testing.assert_allclose(gr_n[name], gr_l[name], rtol=1e-9, atol=1e-12)


def test_eigendecomposition_is_shared_between_value_and_gradients():
    """Part 1: value + gradients at the same params trigger only one eigh."""
    from nfit import spin_fluctuations as sf

    rng = np.random.default_rng(5)
    geometry = _gradient_geometry(rng)
    E = rng.uniform(0.3, 5.0, size=geometry.point_index.size)
    kwargs = dict(chi0=0.4, gamma0=2.5, j_values={"J1": 0.12, "J2": -0.07})

    calls = {"n": 0}
    original = sf._batched_eigh

    def counting(matrices):
        calls["n"] += 1
        return original(matrices)

    try:
        sf.set_rpa_backend("numpy")  # ensure the LAPACK path (counted) is used
        geometry.__dict__.pop("_rpa_modes_cache", None)
        sf._batched_eigh = counting
        val = heisenberg_rpa_chipp(geometry, E, **kwargs)
        cp, gr = sf.heisenberg_rpa_chipp_and_gradients(geometry, E, **kwargs)
    finally:
        sf._batched_eigh = original
        sf.set_rpa_backend("auto")

    # One eigendecomposition served both the value and the gradients.
    assert calls["n"] == 1
    # The value (mode-sum form) and the gradient function's chipp (resolvent
    # form) are mathematically equal; they agree to floating-point precision.
    np.testing.assert_allclose(val, cp, rtol=1e-12, atol=1e-15)

    # Changing a J invalidates the cache and recomputes.
    calls["n"] = 0
    try:
        sf.set_rpa_backend("numpy")
        sf._batched_eigh = counting
        heisenberg_rpa_chipp(geometry, E, chi0=0.4, gamma0=2.5, j_values={"J1": 0.2, "J2": -0.07})
    finally:
        sf._batched_eigh = original
        sf.set_rpa_backend("auto")
    assert calls["n"] == 1


def test_eigh_backend_gating_by_size_and_matrix_dimension():
    from nfit import spin_fluctuations as sf

    if sf._NUMBA_KERNELS is None:
        assert not sf._use_numba_eigh(10_000_000, 4)
        return
    try:
        sf.set_rpa_backend("auto")
        assert not sf._use_numba_eigh(10, 4)          # tiny batch -> LAPACK
        assert sf._use_numba_eigh(500_000, 4)          # large batch -> numba
        assert not sf._use_numba_eigh(500_000, 64)     # large matrices -> LAPACK
        sf.set_rpa_backend("numpy")
        assert not sf._use_numba_eigh(500_000, 4)      # forced numpy -> LAPACK
        sf.set_rpa_backend("numba")
        assert sf._use_numba_eigh(10, 4)               # forced numba honors small batch
    finally:
        sf.set_rpa_backend("auto")
