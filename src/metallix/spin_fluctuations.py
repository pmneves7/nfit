"""Dynamic susceptibility models for itinerant, nearly magnetically ordered metals.

This module implements the imaginary part of the dynamic spin susceptibility
chi''(Q, E) for three increasingly structured models. Conversion of chi'' to
measured neutron intensity (Bose factor, magnetic form factor, polarization
factor, overall scale) is handled separately by
:func:`metallix.cross_section.intensity_from_chipp`.

Models
------
1. **Local relaxational** (:func:`local_relaxational_chipp`): a fully local
   spin relaxing at rate ``Gamma``,

   ``chi''(E) = chi_loc * Gamma * E / (E^2 + Gamma^2)``.

   This is the single-site quasi-elastic Lorentzian used for local-moment
   systems and as the Q-independent limit of the coupled models below.

2. **Millis-Monien-Pines (MMP) relaxational** (:func:`mmp_chipp`): the
   phenomenological nearly-antiferromagnetic-liquid form

   ``chi(q, w) = chi_pk / (1 + xi^2 |q - Q0|^2 - i w / omega_sf)``

   so that

   ``chi''(q, w) = chi_pk (w/omega_sf) / [(1 + xi^2 |q - Q0|^2)^2 + (w/omega_sf)^2]``

   with correlation length ``xi`` in Angstrom and ``|q - Q0|`` in inverse
   Angstrom [Millis, Monien & Pines, PRB 42, 167 (1990); Monthoux & Pines,
   PRB 47, 6069 (1993)]. The same relaxational response describes the normal
   state of nearly antiferromagnetic metals such as the optimally doped iron
   pnictides [Inosov et al., Nat. Phys. 6, 178 (2010)].

3. **Heisenberg RPA** (:func:`heisenberg_rpa_chipp`): a single-site
   relaxational susceptibility ``chi0(w) = chi0 / (1 - i w / Gamma0)`` coupled
   through real-space Heisenberg exchange in the random phase approximation,

   ``chi(Q, w) = [1 - chi0(w) J(Q)]^{-1} chi0(w)``,

   where ``J(Q)`` is the lattice Fourier transform of the exchange couplings.
   For ``N`` magnetic sites in the unit cell ``J(Q)`` is an ``N x N`` Hermitian
   matrix; diagonalizing it (eigenvalues ``lambda_nu(Q)``, eigenvectors
   ``U(Q)``) decouples the RPA into modes with

   ``chi_Qnu = chi0 / (1 - lambda_nu chi0)``,
   ``Gamma_nu = Gamma0 (1 - lambda_nu chi0)``,

   so each mode is again relaxational and the relaxation rate softens as the
   Stoner-like criterion ``max_Q lambda_nu(Q) chi0 -> 1`` is approached --- the
   standard phenomenology of nearly ordered itinerant magnets [Moriya, *Spin
   Fluctuations in Itinerant Electron Magnetism* (Springer, 1985); Bernhoeft &
   Lonzarich, J. Phys.: Condens. Matter 7, 7325 (1995)]. The measured response
   sums the modes with neutron structure-factor weights

   ``chi''(Q, E) = sum_nu w_nu(Q) chi_Qnu Gamma_nu E / (E^2 + Gamma_nu^2)``,
   ``w_nu(Q) = |sum_a U_{a nu}(Q) e^{2 pi i Q . r_a}|^2 / N``.

Phase convention
----------------
All phases are dimensionless with Q in reciprocal lattice units (H, K, L) and
site positions ``r_a`` fractional:

``J(Q)_{ab} = sum_{bonds (a, b, n)} J_bond exp(2 pi i (H,K,L) . (r_b + n - r_a))``

(the "extended zone" convention, matching Sunny.jl). Each physical bond is
stored once; its Hermitian conjugate is added automatically. The neutron
weight uses the matching site phase ``exp(2 pi i Q . r_a)``, which guarantees
the exact sum rule ``sum_nu w_nu(Q) = 1`` because ``U`` is unitary; the
sum rule is enforced in the test suite as a phase-convention lock.

Units
-----
Energies (``E``, ``Gamma``, ``omega_sf``, ``J``) in meV; ``chi0``/``chi_loc``/
``chi_pk`` in 1/meV (so that ``J * chi0`` is dimensionless) up to the overall
intensity normalization; ``xi`` in Angstrom; Q in r.l.u.

References
----------
- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*,
  Springer Series in Solid-State Sciences 56 (Springer, 1985).
- A. J. Millis, H. Monien, and D. Pines, Phys. Rev. B 42, 167 (1990).
- P. Monthoux and D. Pines, Phys. Rev. B 47, 6069 (1993).
- D. S. Inosov et al., Nature Physics 6, 178 (2010).
- N. Bernhoeft and G. G. Lonzarich, J. Phys.: Condens. Matter 7, 7325 (1995).
- D. Dahlbom et al., Sunny.jl, https://github.com/SunnySuite/Sunny.jl
  (symmetry-distinct bond convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import relaxational_chipp


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def local_relaxational_chipp(
    E: ArrayLike,
    *,
    chi_loc: float,
    gamma: float,
) -> FloatArray:
    """Local-spin relaxational ``chi''(E) = chi_loc Gamma E / (E^2 + Gamma^2)``.

    ``gamma`` (meV) must be positive; ``chi_loc`` carries the static local
    susceptibility (1/meV up to intensity normalization).
    """

    return relaxational_chipp(float(chi_loc), float(gamma), E)


def mmp_chipp(
    q_minus_q0_sq_inv_angstrom2: ArrayLike,
    E: ArrayLike,
    *,
    chi_pk: float,
    xi: float,
    omega_sf: float,
) -> FloatArray:
    """Millis-Monien-Pines relaxational ``chi''`` near an ordering vector.

    ``chi'' = chi_pk (E/omega_sf) / [(1 + xi^2 |q - Q0|^2)^2 + (E/omega_sf)^2]``

    with ``|q - Q0|^2`` supplied in inverse square Angstrom, ``xi`` in
    Angstrom, and ``omega_sf`` in meV [Millis, Monien & Pines, PRB 42, 167
    (1990)].
    """

    if omega_sf <= 0:
        raise ValueError("omega_sf must be positive")
    q_sq = np.asarray(q_minus_q0_sq_inv_angstrom2, dtype=float)
    x = np.asarray(E, dtype=float) / float(omega_sf)
    a = 1.0 + float(xi) ** 2 * q_sq
    return float(chi_pk) * x / (a * a + x * x)


@dataclass
class RpaGeometry:
    """Q-dependent, exchange-independent RPA precomputation for one dataset.

    Built once per (H, K, L) grid by :func:`build_rpa_geometry`; only the small
    per-Q eigenproblems depend on the exchange constants, so fitting re-uses
    this object across optimizer iterations. Points sharing the same Q (e.g.
    every energy bin of a constant-Q cut) are deduplicated: arrays are stored
    at ``n_q`` unique Q points and ``point_index`` maps each of the original
    points back to its unique Q row.
    """

    site_phases: ComplexArray
    """``(n_q, n_sites)`` complex ``exp(2 pi i Q . r_a)``."""

    bond_phases: dict[str, ComplexArray]
    """Orbit label -> ``(n_q, n_sites, n_sites)`` Hermitian phase sums."""

    point_index: NDArray[np.intp]
    """``(n_points,)`` map from original points to unique Q rows."""

    n_sites: int

    @property
    def n_q(self) -> int:
        return int(self.site_phases.shape[0])


def _site_positions(sites: ArrayLike) -> FloatArray:
    positions = np.atleast_2d(np.asarray(sites, dtype=float))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("site positions must be an (n_sites, 3) array of fractional coordinates")
    return positions


def build_rpa_geometry(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    site_positions: ArrayLike,
    orbits: Sequence[Mapping[str, Any]],
) -> RpaGeometry:
    """Precompute the Q-dependent phase factors entering ``J(Q)``.

    Parameters
    ----------
    H, K, L
        Momentum transfer in reciprocal lattice units, one value per point.
    site_positions
        ``(n_sites, 3)`` fractional coordinates of the magnetic sites in the
        unit cell.
    orbits
        Bond orbits as plain mappings ``{"label": str, "bonds": [{"site_i":
        int, "site_j": int, "offset": [n1, n2, n3]}, ...]}``. Each physical
        bond appears exactly once; the Hermitian conjugate term is added here.
        All bonds of one orbit share a single exchange constant.
    """

    positions = _site_positions(site_positions)
    n_sites = positions.shape[0]

    hkl = np.column_stack(
        [
            np.asarray(H, dtype=float).ravel(),
            np.asarray(K, dtype=float).ravel(),
            np.asarray(L, dtype=float).ravel(),
        ]
    )
    unique_hkl, point_index = np.unique(hkl, axis=0, return_inverse=True)
    point_index = np.asarray(point_index, dtype=np.intp).ravel()

    site_phases = np.exp(2j * np.pi * (unique_hkl @ positions.T))

    bond_phases: dict[str, ComplexArray] = {}
    for orbit in orbits:
        label = str(orbit["label"])
        if label in bond_phases:
            raise ValueError(f"duplicate bond orbit label {label!r}")
        phases = np.zeros((unique_hkl.shape[0], n_sites, n_sites), dtype=complex)
        bonds = orbit["bonds"]
        if not bonds:
            raise ValueError(f"bond orbit {label!r} contains no bonds")
        for bond in bonds:
            i = int(bond["site_i"])
            j = int(bond["site_j"])
            if not (0 <= i < n_sites and 0 <= j < n_sites):
                raise ValueError(
                    f"bond orbit {label!r} references site index outside "
                    f"0..{n_sites - 1}"
                )
            offset = np.asarray(bond["offset"], dtype=float)
            if offset.shape != (3,):
                raise ValueError(f"bond orbit {label!r} has a non-3-vector cell offset")
            delta = positions[j] + offset - positions[i]
            phase = np.exp(2j * np.pi * (unique_hkl @ delta))
            phases[:, i, j] += phase
            phases[:, j, i] += np.conj(phase)
        bond_phases[label] = phases

    return RpaGeometry(
        site_phases=site_phases,
        bond_phases=bond_phases,
        point_index=point_index,
        n_sites=n_sites,
    )


def rpa_exchange_matrix(
    geometry: RpaGeometry, j_values: Mapping[str, float]
) -> ComplexArray:
    """Assemble ``J(Q)`` (meV) at the unique Q points of ``geometry``."""

    matrix = np.zeros(
        (geometry.n_q, geometry.n_sites, geometry.n_sites), dtype=complex
    )
    for label, value in j_values.items():
        try:
            phases = geometry.bond_phases[label]
        except KeyError:
            raise KeyError(
                f"unknown bond orbit {label!r}; geometry defines "
                f"{sorted(geometry.bond_phases)}"
            ) from None
        matrix += float(value) * phases
    return matrix


def heisenberg_rpa_chipp(
    geometry: RpaGeometry,
    E: ArrayLike,
    *,
    chi0: float,
    gamma0: float,
    j_values: Mapping[str, float],
) -> FloatArray:
    """RPA ``chi''(Q, E)`` for relaxational local spins coupled by ``J(Q)``.

    See the module docstring for the full model and conventions. Raises
    ``ValueError`` when the RPA denominator ``1 - lambda_nu(Q) chi0`` is not
    positive at some Q, i.e. when the parameters are at or beyond the magnetic
    instability ``max_Q lambda_nu(Q) chi0 = 1``.
    """

    if gamma0 <= 0:
        raise ValueError("gamma0 must be positive")
    if chi0 <= 0:
        raise ValueError("chi0 must be positive")
    energy = np.asarray(E, dtype=float).ravel()
    if energy.shape != geometry.point_index.shape:
        raise ValueError(
            f"E has {energy.size} points but the geometry was built for "
            f"{geometry.point_index.size}"
        )

    exchange = rpa_exchange_matrix(geometry, j_values)
    lam, modes = np.linalg.eigh(exchange)

    denominator = 1.0 - lam * float(chi0)
    if np.any(denominator <= 0.0):
        raise ValueError(
            "RPA instability: 1 - lambda(Q) * chi0 <= 0 "
            f"(max lambda * chi0 = {float(np.max(lam * chi0)):.6g}); "
            "the parameters describe a magnetically ordered state"
        )

    chi_q = float(chi0) / denominator
    gamma_q = float(gamma0) * denominator
    amplitudes = np.einsum("qa,qan->qn", geometry.site_phases, modes)
    weights = np.abs(amplitudes) ** 2 / geometry.n_sites

    idx = geometry.point_index
    chi_pts = (weights * chi_q)[idx]
    gamma_pts = gamma_q[idx]
    e = energy[:, np.newaxis]
    chipp = chi_pts * gamma_pts * e / (e * e + gamma_pts * gamma_pts)
    return np.asarray(chipp.sum(axis=1), dtype=float)
