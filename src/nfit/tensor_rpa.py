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

    return TensorStructure(
        n_sites=n_sites,
        n_q=geometry.n_q,
        parameter_names=tuple(parameter_names),
        terms=terms,
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
        for term in structure.terms[name]:
            block = coeff * term.phase[:, None, None] * term.tensor[None, :, :]
            ai = 3 * term.site_i
            bj = 3 * term.site_j
            matrix[:, ai : ai + 3, bj : bj + 3] += block
            # Hermitian conjugate block (transpose over spin indices, conj phase).
            matrix[:, bj : bj + 3, ai : ai + 3] += np.conj(
                np.swapaxes(block, -1, -2)
            )
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
) -> ComplexArray:
    """Return the ``(n_points, 3, 3)`` dynamic susceptibility ``chi_{alpha beta}``.

    ``chi(Q, w) = (1/N) sum_nu w_nu w_nu^dagger chi0(w) / (1 - chi0(w) lam_nu)``
    with uniform Cartesian mode amplitudes ``w_{alpha nu} = sum_a U[(a
    alpha), nu]`` and ``chi0(w) = chi0 / (1 - i w / gamma0)``. Raises on RPA
    instability (``1 - lam chi0 <= 0``).
    """

    exchange = assemble_tensor_exchange(structure, param_values)
    lam, modes = _tensor_eigh(exchange)  # (n_q, 3N), (n_q, 3N, 3N)
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


def tensor_rpa_unpolarized_chipp(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    q_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    param_values: Mapping[str, float],
) -> FloatArray:
    """Unpolarized ``chi''`` per point: ``sum_{ab} W_{ab}(Qhat) chi''_{ab}``.

    ``q_hat`` is the Cartesian unit momentum-transfer direction per fitted
    point. The isotropic limit equals ``(2/3) chi''_scalar`` (continuity lock).
    """

    chi = tensor_susceptibility(
        structure, geometry, energy, chi0=chi0, gamma0=gamma0, param_values=param_values
    )
    chi_dd = (chi - np.conj(np.swapaxes(chi, -1, -2))) / 2.0j  # Hermitian chi''
    weight = _unpolarized_weight(q_hat)
    return np.einsum("pab,pab->p", weight, chi_dd).real
