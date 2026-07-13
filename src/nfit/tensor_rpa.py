"""Tensor (anisotropic) RPA evaluator.

Extends the scalar Heisenberg RPA of :mod:`nfit.spin_fluctuations` to the full
``3N x 3N`` interaction matrix ``J(Q)_{(a alpha),(b beta)}`` needed for
anisotropic exchange, Dzyaloshinskii-Moriya, single-ion anisotropy, and
dipole-dipole coupling (Cartesian spin indices alpha, beta on ``N`` magnetic
sublattices). The scalar model is the ``T = 1_3`` special case; this evaluator
reduces to it exactly (used as the continuity lock).

Design
------
The interaction is a sum of parameter-times-structure terms
``J(Q) = sum_p theta_p P_p(Q)``. Each ``P_p`` is stored **bond-resolved** as a
list of block contributions ``(site_i, site_j, tensor_3x3, phase(n_q))`` so the
dense ``(n_q, 3N, 3N)`` matrix is materialized only once per evaluation (as the
assembled ``J(Q)``), not per parameter. On-site (single-ion) terms are diagonal
blocks with unit phase.

The response is computed **polarization-resolved**: the full ``3x3`` dynamic
susceptibility tensor ``chi''_{alpha beta}(Q, E)`` is formed and contracted
against per-channel weight matrices. The only channel wired up now is the
unpolarized cross section ``W = (1/3)(delta_{alpha beta} - Qhat_alpha
Qhat_beta)``, whose isotropic limit equals the scalar model's ``2/3 chi''``, but
the tensor ``chi''`` is available internally so polarized channels can be added
without touching the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .crystal import cartesian_rotation
from .spin_fluctuations import RpaGeometry, _batched_eigh, _use_numba_eigh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass
class _BlockTerm:
    """One ``(site_i, site_j)`` block contribution of a structure matrix."""

    site_i: int
    site_j: int
    tensor: ComplexArray  # (3, 3)
    phase: ComplexArray  # (n_q,)


@dataclass
class TensorStructure:
    """Bond-resolved structure matrices ``P_p(Q)`` for the tensor RPA.

    ``parameter_names`` are the fittable coefficients (Heisenberg orbit labels,
    anisotropy ``label_S1``/``label_D1`` etc., single-ion ``K...`` labels), and
    ``terms[name]`` lists the block contributions assembled with that
    coefficient. Hermiticity is completed at assembly time (the transposed
    conjugate block is added automatically), so each physical block appears once.
    """

    n_sites: int
    n_q: int
    parameter_names: tuple[str, ...]
    terms: dict[str, list[_BlockTerm]] = field(default_factory=dict)
    dense_terms: dict[str, ComplexArray] = field(default_factory=dict)
    """Parameter name -> pre-assembled Hermitian ``(n_q, 3N, 3N)`` structure
    matrix (the dipole Ewald tensor). Added directly at assembly (no Hermitian
    completion, unlike the bond-resolved ``terms``)."""


def _identity3() -> ComplexArray:
    return np.eye(3, dtype=complex)


def build_tensor_structure(
    geometry: RpaGeometry,
    site_positions: Sequence[Sequence[float]],
    orbits: Sequence[Mapping[str, Any]],
    *,
    lattice: Mapping[str, Any] | None = None,
    anisotropy: Mapping[str, Any] | None = None,
    sia: Mapping[str, Any] | None = None,
    site_rotations: Sequence[Any] | None = None,
    dipole: Mapping[str, Any] | None = None,
) -> TensorStructure:
    """Assemble the bond-resolved structure matrices for a tensor RPA component.

    ``geometry`` supplies the per-orbit bond phases (already deduplicated over
    Q). ``anisotropy`` maps an orbit label to ``{"enabled", "basis": [{"name",
    "kind", "matrix"}]}``; each enabled basis element becomes a parameter named
    ``"<orbit>_<name>"`` whose per-bond tensor is the basis matrix rotated onto
    that bond (transposed when the bond-carrying operation reverses it).
    ``sia`` maps a magnetic-site label to ``{"enabled", "sites": [i...],
    "basis": [{"name", "matrix"}]}``; each element becomes a parameter
    ``"<name>_<label>"`` acting on the listed sites' diagonal blocks (rotated by
    each site's recorded generator). Rotations require ``lattice``.
    """

    positions = np.asarray(site_positions, dtype=float)
    n_sites = positions.shape[0]
    ones = np.ones(geometry.n_q, dtype=complex)

    parameter_names: list[str] = []
    terms: dict[str, list[_BlockTerm]] = {}

    # Heisenberg (isotropic) part: one parameter per orbit, tensor = identity.
    for orbit in orbits:
        label = str(orbit["label"])
        phases = geometry.bond_phases[label]  # (n_q, N, N), Hermitian already
        # Recover per-bond phase from the orbit definition to keep the tensor
        # blocks bond-resolved (identity tensor -> the sum reproduces `phases`).
        block_terms: list[_BlockTerm] = []
        for bond in orbit["bonds"]:
            i = int(bond["site_i"])
            j = int(bond["site_j"])
            delta = positions[j] + np.asarray(bond["offset"], dtype=float) - positions[i]
            phase = np.exp(2j * np.pi * (geometry.unique_hkl @ delta))
            block_terms.append(_BlockTerm(i, j, _identity3(), phase))
        parameter_names.append(label)
        terms[label] = block_terms

    # Anisotropic exchange: per orbit, per allowed basis element.
    if anisotropy:
        if lattice is None:
            raise ValueError("anisotropic exchange requires lattice parameters")
        orbit_by_label = {str(o["label"]): o for o in orbits}
        for orbit_label, spec in anisotropy.items():
            if not spec or not spec.get("enabled"):
                continue
            orbit = orbit_by_label[str(orbit_label)]
            for element in spec.get("basis", []):
                base = np.asarray(element["matrix"], dtype=float)
                name = f"{orbit_label}_{element['name']}"
                block_terms = []
                for bond in orbit["bonds"]:
                    rotation = bond.get("rotation")
                    reverses = bool(bond.get("reversed", False))
                    if rotation is None:
                        rotated = base
                    else:
                        r_cart = cartesian_rotation(np.asarray(rotation, dtype=float), lattice)
                        source = base.T if reverses else base
                        rotated = r_cart @ source @ r_cart.T
                    i = int(bond["site_i"])
                    j = int(bond["site_j"])
                    delta = positions[j] + np.asarray(bond["offset"], dtype=float) - positions[i]
                    phase = np.exp(2j * np.pi * (geometry.unique_hkl @ delta))
                    block_terms.append(
                        _BlockTerm(i, j, rotated.astype(complex), phase)
                    )
                parameter_names.append(name)
                terms[name] = block_terms

    # Single-ion anisotropy: diagonal, Q-independent blocks per site class.
    if sia:
        if lattice is None:
            raise ValueError("single-ion anisotropy requires lattice parameters")
        for class_label, spec in sia.items():
            if not spec or not spec.get("enabled"):
                continue
            site_indices = [int(index) for index in spec.get("sites", [])]
            for element in spec.get("basis", []):
                base = np.asarray(element["matrix"], dtype=float)
                name = f"{element['name']}_{class_label}"
                block_terms = []
                for site_index in site_indices:
                    rotation = None
                    if site_rotations is not None and site_index < len(site_rotations):
                        rotation = site_rotations[site_index]
                    if rotation is None:
                        rotated = base
                    else:
                        r_cart = cartesian_rotation(np.asarray(rotation, dtype=float), lattice)
                        rotated = r_cart @ base @ r_cart.T
                    # Diagonal block: assembled term adds it once and the
                    # Hermitian completion doubles the diagonal, so halve it to
                    # land the intended on-site tensor after symmetrization.
                    block_terms.append(
                        _BlockTerm(site_index, site_index, 0.5 * rotated.astype(complex), ones)
                    )
                parameter_names.append(name)
                terms[name] = block_terms

    dense_terms: dict[str, ComplexArray] = {}
    # Dipole-dipole: one strength D_dip times the Ewald-summed dipole tensor,
    # cached densely (it is not bond-local).
    if dipole and dipole.get("enabled"):
        if lattice is None:
            raise ValueError("dipole-dipole coupling requires lattice parameters")
        from .dipole import ewald_dipole_tensor

        tensor = ewald_dipole_tensor(geometry.unique_hkl, positions, lattice)
        # (n_q, N, N, 3, 3) -> (n_q, 3N, 3N); block (j, k) at rows 3j.., cols 3k..
        dim = 3 * n_sites
        dense = np.transpose(tensor, (0, 1, 3, 2, 4)).reshape(geometry.n_q, dim, dim)
        parameter_names.append("D_dip")
        dense_terms["D_dip"] = np.ascontiguousarray(dense)

    return TensorStructure(
        n_sites=n_sites,
        n_q=geometry.n_q,
        parameter_names=tuple(parameter_names),
        terms=terms,
        dense_terms=dense_terms,
    )


def assemble_tensor_exchange(
    structure: TensorStructure, param_values: Mapping[str, float]
) -> ComplexArray:
    """Assemble the Hermitian ``(n_q, 3N, 3N)`` interaction matrix ``J(Q)``."""

    dim = 3 * structure.n_sites
    matrix = np.zeros((structure.n_q, dim, dim), dtype=complex)
    for name in structure.parameter_names:
        coeff = float(param_values.get(name, 0.0))
        if coeff == 0.0:
            continue
        for term in structure.terms.get(name, ()):
            block = coeff * term.phase[:, None, None] * term.tensor[None, :, :]
            ai = 3 * term.site_i
            bj = 3 * term.site_j
            matrix[:, ai : ai + 3, bj : bj + 3] += block
            # Hermitian conjugate block (transpose over spin indices, conj phase).
            matrix[:, bj : bj + 3, ai : ai + 3] += np.conj(
                np.swapaxes(block, -1, -2)
            )
        dense = structure.dense_terms.get(name)
        if dense is not None:
            # Already Hermitian over the full 3N x 3N; add it directly.
            matrix += coeff * dense
    return matrix


def _tensor_eigh(matrices: ComplexArray) -> tuple[FloatArray, ComplexArray]:
    n = matrices.shape[-1]
    if _use_numba_eigh(matrices.shape[0], n):
        from . import _rpa_numba

        return _rpa_numba.batched_hermitian_eigh(np.ascontiguousarray(matrices))
    return _batched_eigh(matrices)


def tensor_susceptibility(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
) -> ComplexArray:
    """Return the ``(n_points, 3, 3)`` dynamic susceptibility ``chi_{alpha beta}``.

    ``chi(Q, w) = (1/N) sum_nu w_nu w_nu^dagger chi0(w) / (1 - chi0(w) lam_nu)``
    with uniform Cartesian mode amplitudes ``w_{alpha nu} = sum_a U[(a
    alpha), nu]`` and ``chi0(w) = chi0 / (1 - i w / gamma0)``. Raises on RPA
    instability (``1 - lam chi0 <= 0``). ``lambda_shift`` is the Onsager
    reaction field (rigid shift of every eigenvalue); 0.0 leaves the
    computation untouched.
    """

    exchange = assemble_tensor_exchange(structure, param_values)
    lam, modes = _tensor_eigh(exchange)  # (n_q, 3N), (n_q, 3N, 3N)
    if lambda_shift != 0.0:
        lam = lam - lambda_shift
    if np.any(1.0 - lam * chi0 <= 0.0):
        raise ValueError(
            "RPA instability: 1 - lambda(Q) * chi0 <= 0 "
            f"(max lambda * chi0 = {float(np.max(lam * chi0)):.6g}); the "
            "parameters describe a magnetically ordered state"
        )

    n_sites = structure.n_sites
    # Uniform Cartesian amplitude of each mode: sum over sites within each of
    # the 3 Cartesian rows. modes shape (n_q, 3N, 3N) = (q, (a,alpha), nu).
    amplitudes = modes.reshape(geometry.n_q, n_sites, 3, 3 * n_sites).sum(axis=1)
    # amplitudes: (n_q, 3, 3N) = w_{alpha nu}(Q)

    idx = geometry.point_index
    f = chi0 / (1.0 - 1j * energy / gamma0)  # (n_points,)
    lam_pts = lam[idx]  # (n_points, 3N)
    denom = 1.0 - f[:, None] * lam_pts  # (n_points, 3N)
    weight = f[:, None] / denom  # chi0(w)/(1 - chi0(w) lam), (n_points, 3N)
    amp_pts = amplitudes[idx]  # (n_points, 3, 3N)

    chi = (
        np.einsum("pan,pn,pbn->pab", amp_pts, weight, np.conj(amp_pts))
        / n_sites
    )
    return chi


# Bohr magneton in meV/T (Larmor energy omega_L = g * MU_B_MEV_PER_T * B).
MU_B_MEV_PER_T = 0.05788381


def _perpendicular_frame(b_hat: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return two unit vectors completing a right-handed frame with ``b_hat``.

    The transverse response is invariant to rotations about ``b_hat``, so any
    consistent perpendicular pair works.
    """

    reference = np.array([1.0, 0.0, 0.0]) if abs(b_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = reference - np.dot(reference, b_hat) * b_hat
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(b_hat, x_axis)
    return x_axis, y_axis


def zeeman_cartesian_propagator(
    energy: FloatArray,
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float = 1.0,
    gamma_perp_ratio: float = 1.0,
) -> ComplexArray:
    """Return the single-site local propagator ``X0(omega)`` per point (3x3).

    In the field frame (z = B-hat) the longitudinal response is
    ``chi_par / (1 - i w / Gamma_par)`` and the transverse circular modes are
    ``chi_perp / (1 - i (w -/+ omega_L) / Gamma_perp)``; the Cartesian tensor
    has a gyrotropic (antisymmetric) part proportional to ``omega_L``. At
    ``omega_L -> 0`` with unit ratios it becomes the scalar ``chi0(omega) I3``.
    """

    e = np.asarray(energy, dtype=float)
    x_par = chi0 / (1.0 - 1j * e / gamma0)
    chi_perp = chi_perp_ratio * chi0
    gamma_perp = gamma_perp_ratio * gamma0
    x_plus = chi_perp / (1.0 - 1j * (e - omega_larmor) / gamma_perp)
    x_minus = chi_perp / (1.0 - 1j * (e + omega_larmor) / gamma_perp)
    diag_perp = 0.5 * (x_plus + x_minus)
    off_perp = 0.5j * (x_plus - x_minus)  # gyrotropic, -> 0 as omega_L -> 0

    x_axis, y_axis = _perpendicular_frame(np.asarray(b_hat, dtype=float))
    z_axis = np.asarray(b_hat, dtype=float)
    # X0_frame = diag_perp (xx,yy) + off_perp (xy antisymmetric) + x_par (zz).
    # Build directly in Cartesian: X0 = a (I - zz^T) + s (zz^T) + g [xy antisym],
    # where the antisymmetric transverse part is off_perp * (x y^T - y x^T).
    n = e.shape[0]
    identity = np.eye(3)
    zz = np.outer(z_axis, z_axis)
    antisym = np.outer(x_axis, y_axis) - np.outer(y_axis, x_axis)
    propagator = (
        diag_perp[:, None, None] * (identity - zz)[None]
        + x_par[:, None, None] * zz[None]
        + off_perp[:, None, None] * antisym[None]
    )
    return propagator.astype(complex)


def cartesian_qhat_per_point(
    geometry: RpaGeometry, rlu_to_inv_angstrom: FloatArray
) -> FloatArray:
    """Return the Cartesian unit momentum direction of every fitted point.

    ``rlu_to_inv_angstrom`` is the ``3x3`` matrix with reciprocal-lattice basis
    vectors as columns (``metadata["rlu_to_inv_angstrom_matrix"]``). Points at
    ``Q = 0`` get a zero vector (their polarization factor is then the isotropic
    average, harmless since the response there is finite).
    """

    if geometry.unique_hkl is None:
        raise ValueError("geometry has no unique_hkl; rebuild it for the tensor path")
    q_cart = geometry.unique_hkl @ np.asarray(rlu_to_inv_angstrom, dtype=float).T
    norms = np.linalg.norm(q_cart, axis=1, keepdims=True)
    q_hat_unique = np.divide(q_cart, norms, out=np.zeros_like(q_cart), where=norms > 0)
    return q_hat_unique[geometry.point_index]


def _unpolarized_weight(q_hat: FloatArray) -> FloatArray:
    """``W_{alpha beta} = (1/3)(delta - Qhat Qhat)`` per point, (n_points, 3, 3)."""

    identity = np.eye(3)[None, :, :]
    outer = q_hat[:, :, None] * q_hat[:, None, :]
    return (identity - outer) / 3.0


_ZEEMAN_SOLVE_BLOCK = 200_000


def tensor_zeeman_susceptibility(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    propagator: ComplexArray,
    *,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
) -> ComplexArray:
    """Return ``chi_{alpha beta}`` (n_points, 3, 3) with a tensor local propagator.

    ``propagator`` is the per-point ``X0(omega)`` 3x3 (from
    :func:`zeeman_cartesian_propagator`), which no longer commutes with
    ``J(Q)``'s eigenbasis, so this solves ``[1 - X0 J(Q)] chi = X0`` per point by
    batched LU (chunked to bound memory) rather than diagonalizing once per Q.
    ``chi_{alpha beta} = (1/N) phi_alpha^dagger (1 - X0 J)^{-1} X0 phi_beta`` with
    ``phi_alpha`` the uniform-site Cartesian source. ``lambda_shift`` is the
    Onsager reaction field, ``J(Q) -> J(Q) - lambda_shift``; 0.0 leaves the
    computation untouched.
    """

    exchange = assemble_tensor_exchange(structure, param_values)  # (n_q, 3N, 3N)
    n_sites = structure.n_sites
    dim = 3 * n_sites
    if lambda_shift != 0.0:
        exchange = exchange - lambda_shift * np.eye(dim)[None]
    idx = geometry.point_index
    n_points = energy.shape[0]
    identity = np.eye(dim)

    chi = np.empty((n_points, 3, 3), dtype=complex)
    block = _ZEEMAN_SOLVE_BLOCK
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        sel = slice(start, stop)
        j_block = exchange[idx[sel]]  # (m, 3N, 3N)
        x0 = propagator[sel]  # (m, 3, 3)
        m = stop - start
        # X0_full @ J : apply the site-local 3x3 X0 to the alpha index of each
        # site row of J. Reshape J to (m, N, 3, 3N).
        j_reshaped = j_block.reshape(m, n_sites, 3, dim)
        x0_j = np.einsum("pag,pngB->pnaB", x0, j_reshaped).reshape(m, dim, dim)
        a_matrix = identity[None] - x0_j
        # RHS: X0_full phi_beta has block (a, alpha) = X0[alpha, beta] for all a.
        rhs = np.broadcast_to(x0[:, None, :, :], (m, n_sites, 3, 3)).reshape(m, dim, 3)
        solution = np.linalg.solve(a_matrix, rhs)  # (m, 3N, 3)
        chi[sel] = solution.reshape(m, n_sites, 3, 3).sum(axis=1) / n_sites
    return chi


def _chipp_from_chi(chi: ComplexArray, q_hat: FloatArray) -> FloatArray:
    chi_dd = (chi - np.conj(np.swapaxes(chi, -1, -2))) / 2.0j
    return np.einsum("pab,pab->p", _unpolarized_weight(q_hat), chi_dd).real


def tensor_rpa_zeeman_unpolarized_chipp(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    q_hat: FloatArray,
    propagator: ComplexArray,
    *,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
) -> FloatArray:
    """Unpolarized ``chi''`` per point for the field-on (Tier-B) path."""

    chi = tensor_zeeman_susceptibility(
        structure,
        geometry,
        energy,
        propagator,
        param_values=param_values,
        lambda_shift=lambda_shift,
    )
    return _chipp_from_chi(chi, q_hat)


def tensor_rpa_unpolarized_chipp(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    q_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
) -> FloatArray:
    """Unpolarized ``chi''`` per point: ``sum_{ab} W_{ab}(Qhat) chi''_{ab}``.

    ``q_hat`` is the Cartesian unit momentum-transfer direction per fitted
    point. The isotropic limit equals ``(2/3) chi''_scalar`` (continuity lock).
    """

    chi = tensor_susceptibility(
        structure,
        geometry,
        energy,
        chi0=chi0,
        gamma0=gamma0,
        param_values=param_values,
        lambda_shift=lambda_shift,
    )
    return _chipp_from_chi(chi, q_hat)
