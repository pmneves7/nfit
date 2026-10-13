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
unpolarized cross section ``W = delta_{alpha beta} - Qhat_alpha Qhat_beta``,
whose isotropic limit equals the scalar model's ``2 chi''``. The tensor
``chi''`` is available internally so polarized channels can be added without
touching the kernel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .crystal import cartesian_rotation
from .spin_fluctuations import (
    RpaGeometry,
    _batched_eigh,
    _gradient_block_size,
    _select_rpa_backend,
    _use_numba_eigh,
    _validate_rpa_scalars,
)

try:  # optional acceleration, mirroring the scalar path
    from . import _rpa_numba as _NUMBA_KERNELS
except Exception:  # pragma: no cover - numba not installed
    _NUMBA_KERNELS = None

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
        # Recover per-bond phase from the orbit definition to keep the tensor
        # blocks bond-resolved; their sum reproduces the scalar bond phases.
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
    #
    # Sign: ``ewald_dipole_tensor`` returns the physical kernel
    # ``T = (delta - 3 rhat rhat) / r^3``, whose Hamiltonian is
    # ``H = +(1/2) D_dip sum_{i != j} S_i T(r_ij) S_j``. nfit's interaction
    # matrix uses the opposite sign convention, ``H = -(1/2) sum S_i J S_j``,
    # so that a positive Heisenberg J is ferromagnetic. The structure matrix
    # multiplying D_dip is therefore ``-T(Q)``, which makes a *positive*
    # ``D_dip`` the physical point-dipole interaction (see
    # :func:`nfit.dipole.dipole_coupling_constant`).
    if dipole and dipole.get("enabled"):
        if lattice is None:
            raise ValueError("dipole-dipole coupling requires lattice parameters")
        from .dipole import ewald_dipole_tensor

        tensor = ewald_dipole_tensor(geometry.unique_hkl, positions, lattice)
        # (n_q, N, N, 3, 3) -> (n_q, 3N, 3N); block (j, k) at rows 3j.., cols 3k..
        dim = 3 * n_sites
        dense = np.transpose(tensor, (0, 1, 3, 2, 4)).reshape(geometry.n_q, dim, dim)
        parameter_names.append("D_dip")
        dense_terms["D_dip"] = np.ascontiguousarray(-dense)

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


def check_tensor_exchange_stability(
    exchange: ComplexArray,
    n_sites: int,
    static_local_propagator: ComplexArray,
    *,
    lambda_shift: float = 0.0,
) -> float:
    """Check stability for preassembled ``(n_q, 3N, 3N)`` exchange matrices."""

    margins = 1.0 - tensor_exchange_stability_eigenvalues(
        exchange,
        n_sites,
        static_local_propagator,
        lambda_shift=lambda_shift,
    )
    margin = float(np.min(margins))
    if margin <= 0.0:
        raise ValueError(
            "RPA instability: largest eigenvalue of X0(0)^(1/2) "
            "[J(Q)-lambda_shift I] X0(0)^(1/2) is >= 1 "
            f"(value = {1.0 - margin:.6g})"
        )
    return margin


def tensor_exchange_stability_eigenvalues(
    exchange: ComplexArray,
    n_sites: int,
    static_local_propagator: ComplexArray,
    *,
    lambda_shift: float = 0.0,
) -> FloatArray:
    """Return dimensionless static feedback eigenvalues for each sampled Q.

    ``C`` is the positive-definite, site-local static susceptibility in
    inverse meV, repeated over sites. Exchange and the reaction-field shift
    are in meV. Stability requires every eigenvalue of
    ``C^1/2 [J-lambda I] C^1/2`` to be below one; using ``chi0 * eig(J)``
    instead is valid only for an isotropic local static response.
    """

    local = np.asarray(static_local_propagator, dtype=complex)
    if local.shape != (3, 3) or not np.all(np.isfinite(local)):
        raise ValueError("static local propagator must be a finite 3x3 matrix")
    if not np.allclose(local, local.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError("static local propagator must be Hermitian")
    values, vectors = np.linalg.eigh(local)
    if np.any(values <= 0.0):
        raise ValueError("static local susceptibility must be positive definite")
    root = (vectors * np.sqrt(values)[None, :]) @ vectors.conj().T
    exchange = np.asarray(exchange, dtype=complex)
    dim = 3 * int(n_sites)
    if exchange.ndim != 3 or exchange.shape[1:] != (dim, dim):
        raise ValueError("exchange must have shape (n_q, 3N, 3N)")
    root_full = np.kron(np.eye(int(n_sites)), root)
    shifted = exchange - float(lambda_shift) * np.eye(dim)[None, :, :]
    stability = root_full[None] @ shifted @ root_full[None]
    return np.asarray(np.linalg.eigvalsh(stability), dtype=float)


# Target complex elements per point block, matching the scalar path's budget:
# block * 3N stays near this so the (block, 3, 3N) temporaries hold a few tens
# of megabytes regardless of dataset size.
_TENSOR_BLOCK_ELEMENTS = 2.0e6


def _tensor_point_block(dim: int, n_points: int) -> int:
    """Point-block size bounding the ``(block, 3, 3N)`` tensor temporaries."""

    budget = int(_TENSOR_BLOCK_ELEMENTS / max(dim, 1))
    return max(1, min(n_points, budget))


def _tensor_modes(
    structure: TensorStructure,
    geometry: RpaGeometry,
    param_values: Mapping[str, float],
    *,
    chi0: float,
    lambda_shift: float = 0.0,
) -> tuple[FloatArray, ComplexArray]:
    """Eigen-factor the tensor ``J(Q)`` and return ``(lam, amplitudes)``.

    ``amplitudes`` is ``(n_q, 3, 3N)``: the uniform Cartesian mode amplitude
    ``w_{alpha nu}(Q) = sum_a U[(a alpha), nu]``, which is all the per-point
    kernels need from the eigenvectors. Raises on RPA instability
    (``1 - lam chi0 <= 0``).
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
    # Sum over sites within each of the 3 Cartesian rows.
    # modes shape (n_q, 3N, 3N) = (q, (a, alpha), nu).
    amplitudes = modes.reshape(geometry.n_q, n_sites, 3, 3 * n_sites).sum(axis=1)
    return lam, amplitudes


def tensor_susceptibility(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
    inverse_mode_energy_sq: float = 0.0,
) -> ComplexArray:
    """Return the ``(n_points, 3, 3)`` dynamic susceptibility ``chi_{alpha beta}``.

    ``chi(Q, w) = (1/N) sum_nu w_nu w_nu^dagger chi0(w) / (1 - chi0(w) lam_nu)``
    with uniform Cartesian mode amplitudes ``w_{alpha nu} = sum_a U[(a
    alpha), nu]`` and
    ``chi0(w) = chi0 / (1 - a_E w^2 - i w / gamma0)``. The coefficient
    ``inverse_mode_energy_sq`` is :math:`a_E`; zero recovers the relaxational
    response exactly. Raises on RPA
    instability (``1 - lam chi0 <= 0``). ``lambda_shift`` is the Onsager
    reaction field (rigid shift of every eigenvalue); 0.0 leaves the
    computation untouched.
    """

    _validate_rpa_scalars(chi0, gamma0)
    inertia = float(inverse_mode_energy_sq)
    if not np.isfinite(inertia) or inertia < 0.0:
        raise ValueError("inverse_mode_energy_sq must be finite and nonnegative")

    lam, amplitudes = _tensor_modes(
        structure, geometry, param_values, chi0=chi0, lambda_shift=lambda_shift
    )
    n_sites = structure.n_sites
    idx = geometry.point_index
    n_points = energy.shape[0]

    chi = np.empty((n_points, 3, 3), dtype=complex)
    # Blocked so the per-point (block, 3, 3N) gathers stay bounded: at a few
    # million points the unblocked form allocated hundreds of megabytes of
    # complex temporaries. Every point is independent, so the values are
    # unchanged.
    block = _tensor_point_block(3 * n_sites, n_points)
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        sel = slice(start, stop)
        idx_c = idx[sel]
        f = chi0 / (
            1.0
            - inertia * energy[sel] ** 2
            - 1j * energy[sel] / gamma0
        )  # (block,)
        denom = 1.0 - f[:, None] * lam[idx_c]  # (block, 3N)
        weight = f[:, None] / denom  # chi0(w)/(1 - chi0(w) lam)
        amp_c = amplitudes[idx_c]  # (block, 3, 3N)
        chi[sel] = (
            np.einsum("pan,pn,pbn->pab", amp_c, weight, np.conj(amp_c)) / n_sites
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
    ``chi_perp * (Gamma_perp +/- i omega_L) /
    (Gamma_perp - i (w -/+ omega_L))``. Here ``w`` and ``omega_L`` are
    transferred and Larmor energies in meV, not angular frequencies.
    This is relaxation and precession of the departure from the instantaneous
    field-induced equilibrium spin, so both circular static responses equal
    ``chi_perp``. The Cartesian tensor has a gyrotropic part at finite energy. At
    ``omega_L -> 0`` with unit ratios it becomes the scalar ``chi0(omega) I3``.
    """

    return _zeeman_cartesian_response(
        energy,
        b_hat,
        chi0=chi0,
        gamma0=gamma0,
        omega_larmor=omega_larmor,
        chi_perp_ratio=chi_perp_ratio,
        gamma_perp_ratio=gamma_perp_ratio,
    )


def zeeman_cartesian_propagator_zero_energy_derivative(
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float = 1.0,
    gamma_perp_ratio: float = 1.0,
) -> ComplexArray:
    """Return the exact ``dX0/dE`` at E=0 as a 3x3 matrix in meV^-2.

    Differentiates the same causal circular responses as
    :func:`zeeman_cartesian_propagator`, including their equilibrium-driving
    numerators. Its anti-Hermitian part determines zero-energy absorption;
    the gyrotropic derivative also has a Hermitian part.
    """

    return _zeeman_cartesian_response(
        np.zeros(1),
        b_hat,
        chi0=chi0,
        gamma0=gamma0,
        omega_larmor=omega_larmor,
        chi_perp_ratio=chi_perp_ratio,
        gamma_perp_ratio=gamma_perp_ratio,
        energy_derivative=True,
    )[0]


def zeeman_cartesian_affine_inverse(
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float = 1.0,
    gamma_perp_ratio: float = 1.0,
) -> tuple[ComplexArray, ComplexArray]:
    """Return analytic ``(D, D^-1)`` for ``X0(E)^-1 = X0(0)^-1 - i E D``.

    Both matrices are dimensionless. In the field frame, the circular
    eigenvalues of D are ``1/[chi_perp (gamma_perp +/- i omega_larmor)]``;
    its longitudinal eigenvalue is ``1/(chi0 gamma0)``. Constructing both
    matrices directly avoids inverting a poorly conditioned Cartesian
    propagator or D when the linewidth is much smaller than the field energy.
    """

    chi_perp, gamma_perp, zz, antisym = _zeeman_cartesian_terms(
        b_hat,
        chi0=chi0,
        gamma0=gamma0,
        omega_larmor=omega_larmor,
        chi_perp_ratio=chi_perp_ratio,
        gamma_perp_ratio=gamma_perp_ratio,
    )
    transverse = np.eye(3) - zz
    circular_inverse = (1.0 / complex(gamma_perp, omega_larmor)) / chi_perp
    dynamic = (
        zz / (chi0 * gamma0)
        + circular_inverse.real * transverse
        + circular_inverse.imag * antisym
    )
    mobility = chi0 * gamma0 * zz + chi_perp * (
        gamma_perp * transverse + omega_larmor * antisym
    )
    return dynamic.astype(complex), mobility.astype(complex)


def zeeman_cartesian_zero_energy_dissipative_factor(
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float = 1.0,
    gamma_perp_ratio: float = 1.0,
) -> ComplexArray:
    """Factor the local absorptive derivative as ``L L^dagger``, in meV^-2.

    The three columns of L are the field-frame directions times the square
    roots of their dissipative slopes. Keeping longitudinal and transverse
    columns separate preserves a small transverse slope when neutron
    polarization removes the much larger longitudinal contribution.
    """

    chi_perp, gamma_perp, _, _ = _zeeman_cartesian_terms(
        b_hat,
        chi0=chi0,
        gamma0=gamma0,
        omega_larmor=omega_larmor,
        chi_perp_ratio=chi_perp_ratio,
        gamma_perp_ratio=gamma_perp_ratio,
    )
    axis = np.asarray(b_hat, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x_axis, y_axis = _perpendicular_frame(axis)
    transverse = np.sqrt(chi_perp) * np.sqrt(gamma_perp) / np.hypot(
        gamma_perp, omega_larmor
    )
    longitudinal = np.sqrt(chi0) / np.sqrt(gamma0)
    return np.column_stack(
        (transverse * x_axis, transverse * y_axis, longitudinal * axis)
    ).astype(complex)


def zeeman_cartesian_dissipative_factor(
    energy: FloatArray,
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float = 1.0,
    gamma_perp_ratio: float = 1.0,
) -> ComplexArray:
    """Factor local absorption as ``sign(E) L L^dagger`` at each energy.

    The circular transverse columns have squared amplitudes
    ``chi_perp gamma_perp |E| / [gamma_perp^2 + (E -/+ omega_larmor)^2]``;
    the longitudinal column has the analogous unshifted relaxational weight.
    Each factor has units meV^-1/2. Projecting these columns before squaring
    avoids subtracting large longitudinal signals from a transverse channel.
    At E=0 the factor is zero; use the dissipative slope factor for that limit.
    """

    chi_perp, gamma_perp, _, _ = _zeeman_cartesian_terms(
        b_hat,
        chi0=chi0,
        gamma0=gamma0,
        omega_larmor=omega_larmor,
        chi_perp_ratio=chi_perp_ratio,
        gamma_perp_ratio=gamma_perp_ratio,
    )
    axis = np.asarray(b_hat, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x_axis, y_axis = _perpendicular_frame(axis)
    e = np.asarray(energy, dtype=float).ravel()
    root_energy = np.sqrt(np.abs(e))
    transverse = np.sqrt(chi_perp) * np.sqrt(gamma_perp) * root_energy
    plus = transverse / np.hypot(gamma_perp, e - omega_larmor)
    minus = transverse / np.hypot(gamma_perp, e + omega_larmor)
    longitudinal = np.sqrt(chi0) * np.sqrt(gamma0) * root_energy / np.hypot(gamma0, e)
    return np.stack(
        (
            plus[:, None] * (x_axis + 1j * y_axis)[None] / np.sqrt(2.0),
            minus[:, None] * (x_axis - 1j * y_axis)[None] / np.sqrt(2.0),
            longitudinal[:, None] * axis[None],
        ),
        axis=-1,
    )


def _zeeman_cartesian_terms(
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float,
    gamma_perp_ratio: float,
) -> tuple[float, float, FloatArray, FloatArray]:
    """Validate field parameters and build their shared Cartesian projectors."""

    field_axis = np.asarray(b_hat, dtype=float).reshape(3)
    norm = float(np.linalg.norm(field_axis))
    if not np.all(np.isfinite(field_axis)) or norm == 0.0:
        raise ValueError("b_hat must be a finite nonzero 3-vector")
    _validate_rpa_scalars(chi0, gamma0)
    if not np.isfinite(omega_larmor):
        raise ValueError("omega_larmor must be finite")
    if not np.isfinite(chi_perp_ratio) or chi_perp_ratio <= 0.0:
        raise ValueError("chi_perp_ratio must be finite and positive")
    if not np.isfinite(gamma_perp_ratio) or gamma_perp_ratio <= 0.0:
        raise ValueError("gamma_perp_ratio must be finite and positive")
    # The frame construction below (and the zz / antisymmetric projectors) is
    # only correct for a unit vector, so normalize rather than trusting callers.
    b_hat = field_axis / norm
    x_axis, y_axis = _perpendicular_frame(b_hat)
    zz = np.outer(b_hat, b_hat)
    antisym = np.outer(x_axis, y_axis) - np.outer(y_axis, x_axis)
    return chi_perp_ratio * chi0, gamma_perp_ratio * gamma0, zz, antisym


def _zeeman_cartesian_response(
    energy: FloatArray,
    b_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    omega_larmor: float,
    chi_perp_ratio: float,
    gamma_perp_ratio: float,
    energy_derivative: bool = False,
) -> ComplexArray:
    """Evaluate the field-frame responses or their exact energy derivatives."""

    chi_perp, gamma_perp, zz, antisym = _zeeman_cartesian_terms(
        b_hat,
        chi0=chi0,
        gamma0=gamma0,
        omega_larmor=omega_larmor,
        chi_perp_ratio=chi_perp_ratio,
        gamma_perp_ratio=gamma_perp_ratio,
    )
    e = np.asarray(energy, dtype=float)
    x_par = chi0 / (1.0 - 1j * e / gamma0)
    # The precession term also acts on the field-induced equilibrium spin.
    # Keeping only shifted denominators loses that driving term and produces
    # an antisymmetric static response and negative low-energy absorption.
    x_plus = chi_perp * (gamma_perp + 1j * omega_larmor) / (
        gamma_perp - 1j * (e - omega_larmor)
    )
    x_minus = chi_perp * (gamma_perp - 1j * omega_larmor) / (
        gamma_perp - 1j * (e + omega_larmor)
    )
    if energy_derivative:
        x_par = 1j * x_par / (gamma0 - 1j * e)
        x_plus = 1j * x_plus / (gamma_perp - 1j * (e - omega_larmor))
        x_minus = 1j * x_minus / (gamma_perp - 1j * (e + omega_larmor))
    diag_perp = 0.5 * (x_plus + x_minus)
    # Electron moments are -g mu_B S, hence H_Z = +E_L S_z. The retarded
    # xy response then has the -i sign for the positive-energy circular pole.
    off_perp = -0.5j * (x_plus - x_minus)

    # X0_frame = diag_perp (xx,yy) + off_perp (xy antisymmetric) + x_par (zz).
    # Build directly in Cartesian: X0 = a (I - zz^T) + s (zz^T) + g [xy antisym],
    # where the antisymmetric transverse part is off_perp * (x y^T - y x^T).
    identity = np.eye(3)
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
    ``Q = 0`` get a zero vector; :func:`_unpolarized_weight` recognizes those
    rows and applies the isotropic directional average.
    """

    if geometry.unique_hkl is None:
        raise ValueError("geometry has no unique_hkl; rebuild it for the tensor path")
    q_cart = geometry.unique_hkl @ np.asarray(rlu_to_inv_angstrom, dtype=float).T
    norms = np.linalg.norm(q_cart, axis=1, keepdims=True)
    q_hat_unique = np.divide(q_cart, norms, out=np.zeros_like(q_cart), where=norms > 0)
    return q_hat_unique[geometry.point_index]


def _unpolarized_weight(q_hat: FloatArray) -> FloatArray:
    """``W_{alpha beta} = delta - Qhat Qhat`` per point, (n_points, 3, 3)."""

    identity = np.eye(3)[None, :, :]
    outer = q_hat[:, :, None] * q_hat[:, None, :]
    weight = identity - outer
    zero_q = np.linalg.norm(q_hat, axis=1) <= np.finfo(float).eps
    weight[zero_q] = (2.0 / 3.0) * identity
    return weight


def _unpolarized_amplitude_weights(
    amplitudes: ComplexArray, q_hat: FloatArray
) -> FloatArray:
    """Project Cartesian amplitudes before forming their nonnegative weights."""

    parallel = np.einsum("pa,pan->pn", q_hat, amplitudes)
    transverse = amplitudes - q_hat[:, :, None] * parallel[:, None, :]
    weights = np.sum(np.abs(transverse)**2, axis=1)
    zero_q = np.linalg.norm(q_hat, axis=1) <= np.finfo(float).eps
    weights[zero_q] = (2.0 / 3.0) * np.sum(np.abs(amplitudes[zero_q])**2, axis=1)
    return weights


_ZEEMAN_SOLVE_BLOCK = 200_000


def _shifted_tensor_exchange(
    structure: TensorStructure,
    param_values: Mapping[str, float],
    lambda_shift: float,
    static_local_propagator: ComplexArray | None,
) -> ComplexArray:
    """Assemble, check static stability when available, and shift exchange."""

    exchange = assemble_tensor_exchange(structure, param_values)
    if static_local_propagator is not None:
        check_tensor_exchange_stability(
            exchange, structure.n_sites, static_local_propagator, lambda_shift=lambda_shift
        )
    if lambda_shift != 0.0:
        exchange = exchange - lambda_shift * np.eye(3 * structure.n_sites)[None]
    return exchange


def _tensor_local_denominator(
    exchange: ComplexArray, local: ComplexArray, n_sites: int
) -> ComplexArray:
    """Build I-X0 J for a bounded batch of site-local tensor propagators."""

    count, dim = exchange.shape[0], 3 * n_sites
    local = np.broadcast_to(local, (count, 3, 3))
    local_exchange = np.einsum(
        "pag,pngB->pnaB", local, exchange.reshape(count, n_sites, 3, dim)
    ).reshape(count, dim, dim)
    return np.eye(dim)[None] - local_exchange


def _tensor_local_factor_amplitudes(
    exchange: ComplexArray, local: ComplexArray, factor: ComplexArray, n_sites: int
) -> ComplexArray:
    """Return phi^dagger (I-X0 J)^-1 L in Cartesian source columns."""

    denominator = _tensor_local_denominator(exchange, local, n_sites)
    source = np.tile(np.eye(3), (n_sites, 1))
    z = np.linalg.solve(
        np.conj(np.swapaxes(denominator, -1, -2)), source[None]
    ).reshape(-1, n_sites, 3, 3)
    factors = np.broadcast_to(factor, (exchange.shape[0], 3, 3))
    amplitudes = np.einsum("pnab,pac->pnbc", np.conj(z), factors)
    return amplitudes.transpose(0, 2, 1, 3).reshape(-1, 3, 3 * n_sites)


def tensor_zeeman_susceptibility(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    propagator: ComplexArray,
    *,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
    static_local_propagator: ComplexArray | None = None,
) -> ComplexArray:
    """Return ``chi_{alpha beta}`` (n_points, 3, 3) with a tensor local propagator.

    ``propagator`` is the per-point ``X0(omega)`` 3x3 (from
    :func:`zeeman_cartesian_propagator`), which no longer commutes with
    ``J(Q)``'s eigenbasis, so this solves ``[1 - X0 J(Q)] chi = X0`` per point by
    batched LU (chunked to bound memory) rather than diagonalizing once per Q.
    ``chi_{alpha beta} = (1/N) phi_alpha^dagger (1 - X0 J)^{-1} X0 phi_beta`` with
    ``phi_alpha`` the uniform-site Cartesian source. ``lambda_shift`` is the
    Onsager reaction field, ``J(Q) -> J(Q) - lambda_shift``; 0.0 leaves the
    computation untouched. If ``static_local_propagator`` is supplied, the
    static RPA stability criterion is checked before solving. Otherwise a
    zero-energy row in ``energy`` is used to infer the local static response;
    when all requested energies are nonzero, this low-level API cannot certify
    static stability and callers that need that guarantee should pass the
    zero-energy propagator explicitly.
    """

    n_sites = structure.n_sites
    dim = 3 * n_sites
    if static_local_propagator is None:
        zero_rows = np.flatnonzero(np.asarray(energy, dtype=float) == 0.0)
        if zero_rows.size:
            static_local_propagator = propagator[int(zero_rows[0])]
    exchange = _shifted_tensor_exchange(
        structure, param_values, lambda_shift, static_local_propagator
    )
    idx = geometry.point_index
    n_points = energy.shape[0]

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
        a_matrix = _tensor_local_denominator(j_block, x0, n_sites)
        # RHS: X0_full phi_beta has block (a, alpha) = X0[alpha, beta] for all a.
        rhs = np.broadcast_to(x0[:, None, :, :], (m, n_sites, 3, 3)).reshape(m, dim, 3)
        solution = np.linalg.solve(a_matrix, rhs)  # (m, 3N, 3)
        chi[sel] = solution.reshape(m, n_sites, 3, 3).sum(axis=1) / n_sites
    return chi


def _chipp_from_chi(chi: ComplexArray, q_hat: FloatArray) -> FloatArray:
    chi_dd = (chi - np.conj(np.swapaxes(chi, -1, -2))) / 2.0j
    return np.einsum("pab,pab->p", _unpolarized_weight(q_hat), chi_dd).real


def unpolarized_static_chi(chi: ComplexArray, q_hat: FloatArray) -> FloatArray:
    """Contract ``Re chi(Q, 0)`` with the unpolarized neutron projector."""

    return np.einsum(
        "pab,pab->p",
        _unpolarized_weight(q_hat),
        np.asarray(chi, dtype=complex).real,
    )


def tensor_rpa_zeeman_unpolarized_chipp(
    structure: TensorStructure,
    geometry: RpaGeometry,
    energy: FloatArray,
    q_hat: FloatArray,
    propagator: ComplexArray,
    *,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
    static_local_propagator: ComplexArray | None = None,
    local_dissipative_factor: ComplexArray | None = None,
) -> FloatArray:
    """Unpolarized ``chi''`` per point for the field-on (Tier-B) path.

    ``local_dissipative_factor`` optionally supplies per-point factors L with
    local absorption ``sign(E) L L^dagger``. Source amplitudes are projected
    before their positive weights are summed, preserving small transverse
    signals. Use :func:`zeeman_cartesian_dissipative_factor` for this model.
    The generic complex susceptibility remains available without a factor.
    """

    if local_dissipative_factor is not None:
        factor = np.asarray(local_dissipative_factor, dtype=complex)
        energy = np.asarray(energy, dtype=float).ravel()
        if factor.shape != (energy.size, 3, 3) or not np.all(np.isfinite(factor)):
            raise ValueError("local dissipative factor must have finite shape (n_points,3,3)")
        propagator = np.asarray(propagator, dtype=complex)
        absorptive = (propagator - np.conj(np.swapaxes(propagator, -1, -2))) / 2j
        signed_factor = np.sign(energy)[:, None, None] * (factor @ np.conj(np.swapaxes(factor, -1, -2)))
        if not np.allclose(signed_factor, absorptive, rtol=1e-10, atol=1e-12):
            raise ValueError("local dissipative factor does not match the absorptive response")
        if static_local_propagator is None:
            zero_rows = np.flatnonzero(energy == 0.0)
            if zero_rows.size:
                static_local_propagator = propagator[int(zero_rows[0])]
        exchange = _shifted_tensor_exchange(
            structure, param_values, lambda_shift, static_local_propagator
        )
        n_sites = structure.n_sites
        result = np.empty(energy.size, dtype=float)
        block = _gradient_block_size(3 * n_sites, energy.size)
        for start in range(0, energy.size, block):
            stop = min(start + block, energy.size)
            amplitudes = _tensor_local_factor_amplitudes(
                exchange[geometry.point_index[start:stop]], propagator[start:stop],
                factor[start:stop], n_sites,
            )
            result[start:stop] = np.sign(energy[start:stop]) * np.sum(
                _unpolarized_amplitude_weights(amplitudes, q_hat[start:stop]), axis=1
            ) / n_sites
        return result

    chi = tensor_zeeman_susceptibility(
        structure,
        geometry,
        energy,
        propagator,
        param_values=param_values,
        lambda_shift=lambda_shift,
        static_local_propagator=static_local_propagator,
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
    inverse_mode_energy_sq: float = 0.0,
) -> FloatArray:
    """Unpolarized ``chi''`` per point: ``sum_{ab} W_{ab}(Qhat) chi''_{ab}``.

    ``q_hat`` is the Cartesian unit momentum-transfer direction per fitted
    point. The isotropic limit equals ``2 chi''_scalar`` (continuity lock).

    Large problems use a fused Numba kernel that builds each point's ``3 x 3``
    susceptibility and contracts it with the projector in one pass, avoiding
    the ``(n_points, 3, 3N)`` and ``(n_points, 3, 3)`` intermediates of the
    NumPy path. Both agree to floating-point precision (locked by the test
    suite); the NumPy path remains the reference and the only path on installs
    without Numba.
    """

    _validate_rpa_scalars(chi0, gamma0)
    if (
        float(inverse_mode_energy_sq) == 0.0
        and _NUMBA_KERNELS is not None
        and _select_rpa_backend(energy.shape[0]) != "numpy"
    ):
        lam, amplitudes = _tensor_modes(
            structure, geometry, param_values, chi0=chi0, lambda_shift=lambda_shift
        )
        return np.asarray(
            _NUMBA_KERNELS.tensor_unpolarized_chipp_kernel(
                lam,
                np.ascontiguousarray(amplitudes),
                geometry.point_index,
                np.ascontiguousarray(energy, dtype=float),
                np.ascontiguousarray(q_hat, dtype=float),
                float(chi0),
                float(gamma0),
                structure.n_sites,
            ),
            dtype=float,
        )

    chi = tensor_susceptibility(
        structure,
        geometry,
        energy,
        chi0=chi0,
        gamma0=gamma0,
        param_values=param_values,
        lambda_shift=lambda_shift,
        inverse_mode_energy_sq=inverse_mode_energy_sq,
    )
    return _chipp_from_chi(chi, q_hat)


def tensor_rpa_zero_energy_slope(
    structure: TensorStructure,
    geometry: RpaGeometry,
    q_hat: FloatArray,
    *,
    chi0: float,
    gamma0: float,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
    inverse_mode_energy_sq: float = 0.0,
) -> FloatArray:
    """Exact unpolarized limit ``chi''(Q,E)/E`` at E=0, in meV^-2.

    The Cartesian mode matrix has weight
    ``chi0 / [gamma0 (1 - chi0 lambda_nu)^2]``. The result already includes
    the neutron polarization projector and equals twice the scalar slope
    in the isotropic limit. Inertia does not enter the linear energy term.
    """

    _validate_rpa_scalars(chi0, gamma0)
    inertia = float(inverse_mode_energy_sq)
    if not np.isfinite(inertia) or inertia < 0.0:
        raise ValueError("inverse_mode_energy_sq must be finite and nonnegative")
    lam, amplitudes = _tensor_modes(
        structure, geometry, param_values, chi0=chi0, lambda_shift=lambda_shift
    )
    weight = chi0 / (gamma0 * (1.0 - chi0 * lam)**2)
    n_points = geometry.point_index.size
    result = np.empty(n_points, dtype=float)
    block = _tensor_point_block(3 * structure.n_sites, n_points)
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        idx = geometry.point_index[start:stop]
        neutron_weight = _unpolarized_amplitude_weights(
            amplitudes[idx], q_hat[start:stop]
        )
        result[start:stop] = np.sum(neutron_weight * weight[idx], axis=1) / structure.n_sites
    return result


def tensor_rpa_zeeman_zero_energy_slope(
    structure: TensorStructure,
    geometry: RpaGeometry,
    q_hat: FloatArray,
    static_local_propagator: ComplexArray,
    local_energy_derivative: ComplexArray,
    *,
    param_values: Mapping[str, float],
    lambda_shift: float = 0.0,
    local_dissipative_factor: ComplexArray | None = None,
) -> FloatArray:
    """Exact unpolarized zero-energy slope for a tensor local response.

    ``C=X0(0)`` must be Hermitian positive definite, in meV^-1; ``D=X0'(0)``
    is a finite 3x3 matrix in meV^-2. For the repeated site-local matrices,
    the RPA derivative is ``A^-1 D A^-dagger``, with
    ``A=I-C [J-lambda_shift I]``. Solving ``A^dagger z=phi`` gives its
    neutron projection as ``z^dagger D z/N`` without forming a dense inverse.
    Static stability is checked at all supplied momenta before solving.
    If supplied, ``local_dissipative_factor=L`` factors the local absorptive
    derivative as ``L L^dagger``. Its source amplitudes are projected before
    squaring, retaining small transverse responses in strongly anisotropic
    limits; use :func:`zeeman_cartesian_zero_energy_dissipative_factor` for
    the Zeeman local model.
    """

    n_sites = structure.n_sites
    dim = 3 * n_sites
    local = np.asarray(static_local_propagator, dtype=complex)
    derivative = np.asarray(local_energy_derivative, dtype=complex)
    if derivative.shape != (3, 3) or not np.all(np.isfinite(derivative)):
        raise ValueError("local energy derivative must be a finite 3x3 matrix")
    factor = None
    if local_dissipative_factor is not None:
        factor = np.asarray(local_dissipative_factor, dtype=complex)
        if factor.shape != (3, 3) or not np.all(np.isfinite(factor)):
            raise ValueError("local dissipative factor must be a finite 3x3 matrix")
        absorptive = (derivative - derivative.conj().T) / 2j
        if not np.allclose(factor @ factor.conj().T, absorptive, rtol=1e-10, atol=1e-12):
            raise ValueError("local dissipative factor does not match the absorptive derivative")
    exchange = _shifted_tensor_exchange(structure, param_values, lambda_shift, local)
    source = np.tile(np.eye(3), (n_sites, 1))
    response_derivative = np.empty((geometry.n_q, 3, 3), dtype=complex)
    factor_amplitudes = (
        np.empty((geometry.n_q, 3, 3 * n_sites), dtype=complex)
        if factor is not None else None
    )
    block = _gradient_block_size(dim, geometry.n_q)
    for start in range(0, geometry.n_q, block):
        stop = min(start + block, geometry.n_q)
        if factor is not None:
            factor_amplitudes[start:stop] = _tensor_local_factor_amplitudes(
                exchange[start:stop], local, factor, n_sites
            )
            continue
        a_matrix = _tensor_local_denominator(exchange[start:stop], local, n_sites)
        z = np.linalg.solve(
            np.conj(np.swapaxes(a_matrix, -1, -2)), source[None]
        ).reshape(-1, n_sites, 3, 3)
        response_derivative[start:stop] = np.einsum(
            "qnab,ac,qncd->qbd", np.conj(z), derivative, z
        ) / n_sites
    if factor_amplitudes is not None:
        n_points = geometry.point_index.size
        result = np.empty(n_points, dtype=float)
        block = _tensor_point_block(dim, n_points)
        for start in range(0, n_points, block):
            stop = min(start + block, n_points)
            amplitudes = factor_amplitudes[geometry.point_index[start:stop]]
            result[start:stop] = np.sum(
                _unpolarized_amplitude_weights(amplitudes, q_hat[start:stop]), axis=1
            ) / n_sites
        return result
    return _chipp_from_chi(response_derivative[geometry.point_index], q_hat)
