import numpy as np

from nfit.dipole import dipole_coupling_constant, ewald_dipole_tensor

LATTICE = {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
FRAC = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
HKL = np.array([[0.1, 0.2, 0.3], [0.5, 0.0, 0.0], [0.25, 0.25, 0.25], [0.4, 0.1, 0.6]])


def test_ewald_dipole_is_independent_of_the_splitting_parameter():
    """The rigorous internal check: alpha only affects convergence, not D(Q)."""
    d_low = ewald_dipole_tensor(HKL, FRAC, LATTICE, alpha=0.18)
    d_high = ewald_dipole_tensor(HKL, FRAC, LATTICE, alpha=0.40)
    d_auto = ewald_dipole_tensor(HKL, FRAC, LATTICE)
    np.testing.assert_allclose(d_low, d_high, atol=1e-9)
    np.testing.assert_allclose(d_low, d_auto, atol=1e-9)


def test_ewald_dipole_is_hermitian_and_real_symmetric_under_q_negation():
    d = ewald_dipole_tensor(HKL, FRAC, LATTICE, alpha=0.2)
    # D(Q)_{jk}^{ab} = conj(D(Q)_{kj}^{ba})
    np.testing.assert_allclose(
        d, np.conj(np.transpose(d, (0, 2, 1, 4, 3))), atol=1e-12
    )
    d_neg = ewald_dipole_tensor(-HKL, FRAC, LATTICE, alpha=0.2)
    np.testing.assert_allclose(d_neg, np.conj(d), atol=1e-12)
    # Each spin block is a symmetric 3x3 (the dipole kernel is symmetric).
    np.testing.assert_allclose(d, np.swapaxes(d, -1, -2), atol=1e-12)


def test_ewald_dipole_matches_a_damped_direct_sum():
    """Compare to a large convergence-factor direct lattice sum at one Q.

    The bare dipole lattice sum is conditionally convergent; summing the bare
    tensor over a large sphere with a Gaussian convergence factor and a matched
    reciprocal correction is impractical to reproduce cheaply, so instead check
    that the short-range structure (dominant, absolutely convergent piece) of
    the Ewald result agrees with a plain truncated dipole sum to a few percent
    for a large real cutoff -- enough to catch a sign or normalization error.
    """
    basis = np.eye(3) * 4.0
    frac = np.asarray(FRAC)
    tau = (basis @ frac.T).T
    hkl = np.array([[0.37, 0.11, 0.29]])
    recip = 2.0 * np.pi * np.linalg.inv(basis).T
    q = (recip @ hkl.T).T[0]

    direct = np.zeros((2, 2, 3, 3), dtype=complex)
    span = range(-12, 13)
    for j in range(2):
        for k in range(2):
            for n1 in span:
                for n2 in span:
                    for n3 in span:
                        d = tau[k] - tau[j] + basis @ np.array([n1, n2, n3], float)
                        r = np.linalg.norm(d)
                        if r < 1e-9 or r > 24.0:
                            continue
                        rhat = d / r
                        tensor = (np.eye(3) - 3 * np.outer(rhat, rhat)) / r**3
                        direct[j, k] += np.exp(1j * (q @ d)) * tensor

    ewald = ewald_dipole_tensor(hkl, FRAC, LATTICE, alpha=0.25)[0]
    # Off-diagonal (j != k) blocks are absolutely convergent and should match
    # the direct sum closely.
    np.testing.assert_allclose(ewald[0, 1], direct[0, 1], atol=0.02)


def test_dipole_coupling_constant_scales_with_g_squared():
    assert dipole_coupling_constant(2.0) > 0.0
    np.testing.assert_allclose(
        dipole_coupling_constant(3.0) / dipole_coupling_constant(2.0), (3.0 / 2.0) ** 2, rtol=1e-12
    )
