"""Ewald-summed dipole-dipole interaction tensor for the RPA models.

Computes the lattice Fourier transform of the point-dipole coupling

``T_{alpha beta}(r) = (delta_{alpha beta} - 3 rhat_alpha rhat_beta) / r^3``

as a Hermitian ``(n_q, N, N, 3, 3)`` array ``D(Q)_{jk}^{alpha beta}`` over the
unique-Q grid. The conditionally convergent lattice sum is Ewald-split into a
short-range real-space part (erfc-screened dipole kernel), a smooth reciprocal
part (Gaussian-screened), and a self correction; the tinfoil (metallic)
boundary condition is used, so the ``Q + G = 0`` reciprocal term is dropped.

The split parameter ``alpha`` and the real/reciprocal cutoffs affect only
convergence, not the result -- a property the test suite checks directly, which
is the rigorous internal validation of the implementation.

References: S. W. de Leeuw, J. W. Perram, E. R. Smith, Proc. R. Soc. A 373, 27
(1980); M. Enjalran & M. J. P. Gingras, Phys. Rev. B 70, 174426 (2004).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .crystal import _lattice_vectors

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

# Bohr magneton in meV/T; mu0/(4 pi) (g mu_B)^2 sets the physical dipolar
# strength (meV*Angstrom^3). For g = 2 this is ~0.215 meV*Angstrom^3.
_DIPOLE_CONSTANT_G2 = 0.05368  # meV*Angstrom^3 per unit g^2


def dipole_coupling_constant(g_factor: float = 2.0) -> float:
    """Physical dipolar strength ``mu0/(4 pi) (g mu_B)^2`` in meV*Angstrom^3.

    This is the natural default value of the fitted ``D_dip`` coefficient that
    multiplies the (dimensionless-per-volume) Ewald tensor; pin ``D_dip`` to it
    (vary off) for the physical dipolar interaction, or fit it as an effective
    moment.

    The value is positive: the dipolar Hamiltonian is
    ``H = +(1/2) D_dip sum_{i != j} S_i T(r_ij) S_j`` with ``T`` as returned by
    :func:`ewald_dipole_tensor`, and :func:`nfit.tensor_rpa.build_tensor_structure`
    supplies the sign flip into nfit's ``H = -(1/2) sum S_i J S_j`` interaction
    matrix. A positive ``D_dip`` therefore gives the physical dipole coupling,
    which favours head-to-tail moment alignment along the shortest lattice
    direction.
    """

    return _DIPOLE_CONSTANT_G2 * float(g_factor) ** 2


def _erfc(x: FloatArray) -> FloatArray:
    from scipy.special import erfc

    return erfc(x)


def _axis_shells(basis: FloatArray, radius: float, minimum: int) -> list[int]:
    """Per-axis image counts covering ``radius`` in every lattice direction.

    A single index range is wrong for an anisotropic cell: six shells along a
    2 Angstrom axis reach 12 Angstrom while six along a 20 Angstrom axis reach
    120, so the sum is either under-converged or wastefully large depending on
    the axis. Each axis instead gets enough images to cover ``radius``, using
    the interplanar spacing (the component of the lattice vector perpendicular
    to the other two) rather than its length.
    """

    shells = []
    for axis in range(3):
        others = [basis[:, other] for other in range(3) if other != axis]
        normal = np.cross(others[0], others[1])
        norm = float(np.linalg.norm(normal))
        spacing = (
            abs(float(basis[:, axis] @ normal)) / norm
            if norm > 0.0
            else float(np.linalg.norm(basis[:, axis]))
        )
        spacing = max(spacing, 1.0e-12)
        shells.append(max(int(minimum), int(np.ceil(radius / spacing))))
    return shells


def _lattice_images(basis: FloatArray, shells: Sequence[int]) -> FloatArray:
    ranges = [np.arange(-count, count + 1, dtype=float) for count in shells]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ basis.T


# Cap on the (n_q, n_G, 3, 3) reciprocal working set, in complex128 elements.
_RECIPROCAL_BLOCK_ELEMENTS = 4.0e6


def ewald_dipole_tensor(
    hkl: FloatArray,
    site_positions_frac: Sequence[Sequence[float]],
    lattice: Mapping[str, Any],
    *,
    alpha: float | None = None,
    real_shells: int | None = None,
    recip_shells: int | None = None,
) -> ComplexArray:
    """Return the Ewald dipole tensor ``D(Q)`` shaped ``(n_q, N, N, 3, 3)``.

    ``hkl`` are the unique momentum points in RLU; ``site_positions_frac`` the
    fractional magnetic-site coordinates; ``lattice`` the cell parameters. The
    result is in inverse-cubic-angstrom units, so an energy coupling is
    ``D_dip * D(Q)`` with ``D_dip`` in meV*Angstrom^3. The returned kernel is
    the physical ``T = (delta - 3 rhat rhat) / r^3``; see
    :func:`dipole_coupling_constant` for the sign convention used when it
    enters an RPA interaction matrix.

    ``real_shells`` and ``recip_shells`` override the automatic per-axis image
    counts with a single count on every axis. Leaving them at ``None`` chooses
    each axis independently so that a strongly anisotropic cell converges;
    ``alpha`` and the cutoffs affect convergence only, not the converged
    result.
    """

    basis = _lattice_vectors(lattice)  # columns = a, b, c (Angstrom)
    frac = np.asarray(site_positions_frac, dtype=float)
    tau = (basis @ frac.T).T  # (N, 3) Cartesian site positions
    n_sites = tau.shape[0]
    volume = float(abs(np.linalg.det(basis)))
    recip = 2.0 * np.pi * np.linalg.inv(basis).T  # columns = reciprocal vectors
    q_cart = (recip @ np.asarray(hkl, dtype=float).T).T  # (n_q, 3)
    n_q = q_cart.shape[0]

    if alpha is None:
        alpha = np.sqrt(np.pi) / volume ** (1.0 / 3.0)
    # erfc(alpha r) and exp(-k^2 / 4 alpha^2) both fall below ~1e-16 by the
    # time their argument reaches 6, which sets the two cutoff radii.
    if real_shells is None:
        real_shell_counts = _axis_shells(basis, 6.0 / alpha, 6)
    else:
        real_shell_counts = [int(real_shells)] * 3
    if recip_shells is None:
        recip_shell_counts = _axis_shells(recip, 12.0 * alpha, 6)
    else:
        recip_shell_counts = [int(recip_shells)] * 3

    tau_diff = tau[None, :, :] - tau[:, None, :]  # (N, N, 3): tau_k - tau_j

    dipole = np.zeros((n_q, n_sites, n_sites, 3, 3), dtype=complex)
    two_alpha_sqrtpi = 2.0 * alpha / np.sqrt(np.pi)
    eye3 = np.eye(3)

    # --- real space: erfc-screened dipole kernel over lattice images ---
    r_lattice = _lattice_images(basis, real_shell_counts)
    for j in range(n_sites):
        for k in range(n_sites):
            d = tau_diff[j, k][None, :] + r_lattice  # (n_R, 3)
            r = np.linalg.norm(d, axis=1)
            keep = r > 1e-9
            d = d[keep]
            r = r[keep]
            rhat = d / r[:, None]
            exp_term = np.exp(-(alpha ** 2) * r ** 2)
            b_coeff = _erfc(alpha * r) / r ** 3 + two_alpha_sqrtpi * exp_term / r ** 2
            c_coeff = 3.0 * _erfc(alpha * r) / r ** 3 + two_alpha_sqrtpi * exp_term * (
                3.0 / r ** 2 + 2.0 * alpha ** 2
            )
            tensor = b_coeff[:, None, None] * eye3[None] - c_coeff[:, None, None] * (
                rhat[:, :, None] * rhat[:, None, :]
            )  # (n_R, 3, 3)
            phase = np.exp(1j * (q_cart @ d.T))  # (n_q, n_R)
            dipole[:, j, k] += np.einsum("qr,rab->qab", phase, tensor)

    # --- reciprocal space: Gaussian-screened smooth part (tinfoil: drop k=0) ---
    g_lattice = _lattice_images(recip, recip_shell_counts)
    n_g = g_lattice.shape[0]
    site_phases = np.asarray(
        [
            [np.exp(1j * (g_lattice @ tau_diff[j, k])) for k in range(n_sites)]
            for j in range(n_sites)
        ]
    )  # (N, N, n_G)
    # The (n_q, n_G, 3, 3) working set is the memory peak of this routine, so
    # walk the momentum axis in blocks instead of materializing it whole.
    block = max(1, min(n_q, int(_RECIPROCAL_BLOCK_ELEMENTS / max(9 * n_g, 1))))
    for start in range(0, n_q, block):
        stop = min(start + block, n_q)
        kk = g_lattice[None, :, :] - q_cart[start:stop, None, :]  # (m, n_G, 3)
        k2 = np.einsum("qga,qga->qg", kk, kk)  # (m, n_G)
        finite = k2 > 1e-12
        coeff = np.zeros_like(k2)
        coeff[finite] = (
            (4.0 * np.pi / volume)
            * np.exp(-k2[finite] / (4.0 * alpha ** 2))
            / k2[finite]
        )
        recip_block = coeff[:, :, None, None] * (
            kk[:, :, :, None] * kk[:, :, None, :]
        )  # (m, n_G, 3, 3)
        for j in range(n_sites):
            for k in range(n_sites):
                dipole[start:stop, j, k] += np.einsum(
                    "qgab,g->qab", recip_block, site_phases[j, k]
                )

    # --- self correction: the reciprocal sum includes the d = 0 smooth term ---
    self_coeff = (4.0 * alpha ** 3) / (3.0 * np.sqrt(np.pi))
    for j in range(n_sites):
        dipole[:, j, j] -= self_coeff * eye3

    return dipole
