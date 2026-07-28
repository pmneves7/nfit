"""Brillouin-zone and energy integrals for spin-fluctuation sum rules.

This module provides the numerical machinery behind the moment sum rule and
its self-consistency closures (see ``docs/theory_notes.md``): the equal-time
local fluctuation amplitude

``<m^2> = (1/pi) * (1/N_Q) sum_Q  Tr integral_0^Lambda dw coth(w/2T) chi''(Q, w)``

evaluated per magnetic site, plus Kramers-Kronig static susceptibilities and
the Brillouin-zone sampling grid. Everything is expressed in the package's
model units: ``chi`` in 1/meV, energies in meV, so ``<m^2>`` is dimensionless
(the squared spin amplitude in the units where ``J * chi0`` is dimensionless).

For the relaxational (Lorentzian) modes of the Tier-A RPA the energy integral
has closed forms, used in two complementary regimes (both validated to
machine precision against split-interval adaptive quadrature):

- ``Lambda / k_B T >= 20`` (thermal tail above the cutoff < e^-20 of the
  thermal part): zero-point ``(chi Gamma / 2 pi) ln(1 + Lambda^2/Gamma^2)``
  plus the *infinite-cutoff* thermal part
  ``(chi Gamma / pi) [ln z - 1/(2z) - psi(z)]``, ``z = Gamma/(2 pi k_B T)``,
  from the identity ``psi(z) = ln z - 1/(2z) - 2 I(z)`` with
  ``I(z) = int_0^inf t dt / ((t^2+z^2)(e^{2 pi t}-1))``.
- otherwise (warm/classical regime): the exact finite-cutoff Matsubara
  representation ``coth(w/2T) = 2 k_B T sum_n w/(w^2 + nu_n^2)`` (all integer
  n, ``nu_n = 2 pi n k_B T``), each term integrating to arctan form; the sum
  is truncated once ``nu_n >= 4 max(Lambda, Gamma)`` and completed with its
  ``1/nu^2 + 1/nu^4 + 1/nu^6`` asymptotics summed exactly via the Hurwitz
  zeta function.

The brute-force quadrature (`trace_moment_quadrature`) is retained both as
the numerical referee for the closed forms and as the building block for the
field-on (Tier-B) case, whose gyrotropic ``chi''(w)`` is not a finite sum of
Lorentzians.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import digamma, zeta

from .cross_section import KB_MEV_PER_K

try:
    from ._sum_rules_numba import tier_a_matsubara_moment_sums
except Exception:  # pragma: no cover - Numba is an optional runtime fallback.
    tier_a_matsubara_moment_sums = None

FloatArray = NDArray[np.float64]

# Above this Lambda / k_B T the thermal weight beyond the cutoff is < e^-20 of
# the thermal part and the infinite-cutoff digamma form is exact in double
# precision; below it the finite-cutoff Matsubara sum stays short (<~100
# terms) because nu_1 = 2 pi k_B T is comparable to the cutoff.
_DIGAMMA_REGIME_CUTOFF_OVER_KT = 20.0

# Bulk-susceptibility unit conversion. The model susceptibility is the spin
# susceptibility chi_spin in 1/meV per magnetic site; the molar magnetic
# susceptibility is chi_molar = N_A (g mu_B)^2 chi_spin. In CGS-emu units
# (mu_B in erg/G, energy in erg) this is
#   chi_molar[emu/mol] = EMU_PER_MOL_PER_MODEL_CHI * g^2 * chi_model[1/meV]
# per magnetic site. A response per mole of formula units therefore multiplies
# by the number of equivalent magnetic sites per formula unit.
_MU_B_CGS_ERG_PER_G = 9.2740100783e-21
_MEV_IN_ERG = 1.602176634e-15
_AVOGADRO = 6.02214076e23
_MU_0_SI = 4.0 * np.pi * 1.0e-7
_MU_B_SI_J_PER_T = 9.2740100783e-24
_MEV_IN_J = 1.602176634e-22
# 1 tesla in oersted (CGS field unit MPMS reports).
OERSTED_PER_TESLA = 1.0e4
EMU_PER_MOL_PER_MODEL_CHI = _AVOGADRO * _MU_B_CGS_ERG_PER_G**2 / _MEV_IN_ERG
# Rationalized SI M/H susceptibility explicitly contains mu_0:
#   chi_molar[m^3/mol] = SI_M3_PER_MOL_PER_MODEL_CHI
#                        * g^2 * chi_spin[1/meV].
# This equals 4 pi 10^-6 times the CGS molar factor above.
SI_M3_PER_MOL_PER_MODEL_CHI = (
    _MU_0_SI * _AVOGADRO * _MU_B_SI_J_PER_T**2 / _MEV_IN_J
)
# One Bohr magneton per formula unit, expressed as an emu molar moment.
EMU_PER_MOL_PER_MU_B = _AVOGADRO * _MU_B_CGS_ERG_PER_G


def coth_weight(energy_mev: ArrayLike, temperature_K: float) -> FloatArray:
    """``coth(w / 2 k_B T)`` with the T=0 limit (=1) handled exactly.

    ``energy_mev`` must be positive; the detailed-balance weight for the
    two-sided energy integral folded onto ``w > 0``.
    """

    omega = np.asarray(energy_mev, dtype=float)
    if temperature_K <= 0.0:
        return np.ones_like(omega)
    x = omega / (2.0 * KB_MEV_PER_K * temperature_K)
    # tanh is well-conditioned for both small and large arguments.
    return 1.0 / np.tanh(x)


def lorentzian_zero_point(
    chi: ArrayLike, gamma: ArrayLike, cutoff_mev: float
) -> FloatArray:
    """Zero-point amplitude of a relaxational mode up to the cutoff.

    ``(1/pi) int_0^Lambda chi Gamma w / (w^2 + Gamma^2) dw
    = (chi Gamma / 2 pi) ln(1 + Lambda^2 / Gamma^2)``.
    """

    chi = np.asarray(chi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    return (chi * gamma / (2.0 * np.pi)) * np.log1p((cutoff_mev / gamma) ** 2)


def _thermal_moment_digamma(
    chi: FloatArray, gamma: FloatArray, temperature_K: float
) -> FloatArray:
    """Thermal amplitude ``(2/pi) int_0^inf n_B(w) chi'' dw`` (no cutoff).

    Closed form via the digamma function: with ``z = Gamma / (2 pi k_B T)``,
    ``int_0^inf n_B(w) w dw / (w^2 + Gamma^2) = (1/2)[ln z - 1/(2z) - psi(z)]``.
    Used when ``Lambda >> k_B T`` so the (dropped) tail above the cutoff is
    below double precision of the thermal part.
    """

    kt = KB_MEV_PER_K * temperature_K
    z = gamma / (2.0 * np.pi * kt)
    bracket = np.log(z) - 0.5 / z - digamma(z)
    return (chi * gamma / np.pi) * bracket


def _full_moment_matsubara(
    chi: FloatArray, gamma: FloatArray, temperature_K: float, cutoff_mev: float
) -> FloatArray:
    """Exact finite-cutoff moment (zero-point + thermal) via the Matsubara sum.

    ``coth(w/2T) = 2 k_B T sum_{n=-inf}^{inf} w / (w^2 + nu_n^2)`` with
    ``nu_n = 2 pi n k_B T`` turns each mode's integral into

    ``(2 kT chi Gamma / pi) [ arctan(L/Gamma)/Gamma
        + 2 sum_{n>=1} (nu_n arctan(L/nu_n) - Gamma arctan(L/Gamma))
                        / (nu_n^2 - Gamma^2) ]``.

    The sum is truncated at ``nu_N >= 4 max(L, Gamma)`` and completed with the
    term asymptotics ``c2/nu^2 + c4/nu^4 + c6/nu^6`` summed exactly through
    the Hurwitz zeta function (validated to <1e-11 relative against adaptive
    quadrature across the warm regime this branch serves).
    """

    kt = KB_MEV_PER_K * temperature_K
    nu1 = 2.0 * np.pi * kt
    cutoff = float(cutoff_mev)
    arctan_g = np.arctan(cutoff / gamma)
    gamma_sq = gamma * gamma

    scale = max(cutoff, float(np.max(gamma)))
    n_terms = max(32, int(np.ceil(4.0 * scale / nu1)))
    nu = nu1 * np.arange(1, n_terms + 1, dtype=float)
    nu_b = nu.reshape(nu.shape + (1,) * chi.ndim)
    numerator = nu_b * np.arctan(cutoff / nu_b) - gamma * arctan_g
    denominator = nu_b * nu_b - gamma_sq
    # nu_n == Gamma is an ordinary limit of the summand, not a pole:
    # lim = arctan(L/Gamma)/(2 Gamma) - L / (2 (Gamma^2 + L^2)).
    collision = np.abs(denominator) < 1e-10 * gamma_sq
    safe = np.where(collision, 1.0, denominator)
    limit = arctan_g / (2.0 * gamma) - cutoff / (2.0 * (gamma_sq + cutoff**2))
    terms = np.where(collision, np.broadcast_to(limit, numerator.shape), numerator / safe)
    total = arctan_g / gamma + 2.0 * terms.sum(axis=0)

    c2 = cutoff - gamma * arctan_g
    c4 = gamma_sq * c2 - cutoff**3 / 3.0
    c6 = gamma_sq * c4 + cutoff**5 / 5.0
    total = total + 2.0 * (
        c2 * (zeta(2, n_terms + 1) / nu1**2)
        + c4 * (zeta(4, n_terms + 1) / nu1**4)
        + c6 * (zeta(6, n_terms + 1) / nu1**6)
    )
    return (2.0 * kt * chi * gamma / np.pi) * total


def lorentzian_moment(
    chi: ArrayLike,
    gamma: ArrayLike,
    temperature_K: float,
    cutoff_mev: float,
    *,
    parts: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Equal-time amplitude of relaxational modes with a hard energy cutoff.

    ``(1/pi) int_0^Lambda coth(w/2T) chi Gamma w/(w^2+Gamma^2) dw`` per mode,
    broadcast over array-valued ``chi``/``gamma``. With ``parts=True`` returns
    ``(zero_point, thermal)`` separately (needed by the Takahashi TAC closure,
    which budgets the two against each other). Exact for any (T, Gamma,
    Lambda); see the module docstring for the two closed-form regimes.
    """

    chi, gamma = np.broadcast_arrays(
        np.asarray(chi, dtype=float), np.asarray(gamma, dtype=float)
    )
    zero_point = lorentzian_zero_point(chi, gamma, cutoff_mev)
    kt = KB_MEV_PER_K * max(temperature_K, 0.0)
    if kt <= 0.0:
        thermal = np.zeros_like(zero_point)
    elif cutoff_mev / kt >= _DIGAMMA_REGIME_CUTOFF_OVER_KT:
        thermal = _thermal_moment_digamma(chi, gamma, temperature_K)
    else:
        thermal = (
            _full_moment_matsubara(chi, gamma, temperature_K, cutoff_mev)
            - zero_point
        )
    if parts:
        return zero_point, thermal
    return zero_point + thermal


def trace_moment_quadrature(
    chipp_trace: ArrayLike,
    omega_grid: ArrayLike,
    temperature_K: float,
) -> FloatArray:
    """Energy integral ``(1/pi) int coth(w/2T) Tr chi''(w) dw`` on a grid.

    Building block for the field-on (Tier-B) closure, where ``chi''(Q, w)`` is
    only available numerically. ``chipp_trace`` has the energy grid on its
    first axis; remaining axes (e.g. Q) are preserved.
    """

    omega = np.asarray(omega_grid, dtype=float)
    values = np.asarray(chipp_trace, dtype=float)
    weight = coth_weight(omega, temperature_K)
    shape = weight.shape + (1,) * (values.ndim - 1)
    return np.trapezoid(values * weight.reshape(shape), omega, axis=0) / np.pi


def tier_b_omega_grid(cutoff_mev: float, n_points: int = 200) -> FloatArray:
    """Energy quadrature grid for Tier-B trace integrals.

    Log-spaced below 1 meV to resolve narrow quasielastic weight, linear
    above; starts slightly off zero (the coth-weighted integrand is finite at
    ``w -> 0`` but ``chi''`` itself vanishes there, so no weight is lost).
    """

    n_log = max(n_points // 4, 8)
    n_lin = max(n_points - n_log, 8)
    low = np.geomspace(1e-4, min(1.0, 0.1 * cutoff_mev), n_log, endpoint=False)
    high = np.linspace(min(1.0, 0.1 * cutoff_mev), float(cutoff_mev), n_lin)
    return np.concatenate([low, high])


def bz_sample_hkl(n: int) -> FloatArray:
    """Uniform ``n^3`` Brillouin-zone sampling grid in fractional H, K, L.

    Midpoint-shifted Monkhorst-Pack points ``(i + 1/2)/n`` on the reciprocal
    torus ``[0, 1)^3``. Uniform sampling of the conventional-cell torus is
    also uniform on the primitive reciprocal torus (the primitive reciprocal
    lattice contains the conventional one, so the primitive torus is a
    quotient group and uniform measure pushes forward), which is why the same
    grid is exact for both the full and the primitive-reduced site network --
    locked by the 16<->4 pyrochlore test. The half-step shift avoids placing
    points exactly on Gamma/zone-boundary high-symmetry points where the soft
    mode diverges at the RPA instability.
    """

    if n < 1:
        raise ValueError("BZ grid size must be >= 1")
    ticks = (np.arange(n, dtype=float) + 0.5) / float(n)
    h, k, l = np.meshgrid(ticks, ticks, ticks, indexing="ij")
    return np.column_stack([h.ravel(), k.ravel(), l.ravel()])


def mode_amplitude_per_site(
    eigenvalues: ArrayLike,
    *,
    chi0: float,
    gamma0: float,
    temperature_K: float,
    cutoff_mev: float,
    n_sites: int,
    lambda_shift: float = 0.0,
    isotropic_components: int = 3,
    parts: bool = False,
) -> float | tuple[float, float]:
    """Per-site fluctuation amplitude ``<m^2>`` from RPA mode eigenvalues.

    ``eigenvalues`` is the ``(n_q, n_modes)`` spectrum of ``J(Q)`` on a
    uniform BZ grid (scalar model: ``n_modes = N`` and each mode carries
    ``isotropic_components=3`` spin components; tensor Tier A:
    ``n_modes = 3 N`` Cartesian modes and ``isotropic_components=1``). The
    Onsager reaction field enters as the rigid shift
    ``lambda -> lambda - lambda_shift``. Each mode is relaxational with
    ``chi_nu = chi0 / (1 - lambda' chi0)`` and
    ``Gamma_nu = gamma0 (1 - lambda' chi0)``; the local (site-traced)
    amplitude uses unit mode weight -- the trace over sublattices of the RPA
    susceptibility is ``sum_nu chi_nu`` regardless of eigenvector structure.

    Raises ``ValueError`` at or beyond the instability
    ``max (lambda - lambda_shift) chi0 >= 1``.
    """

    lam = np.asarray(eigenvalues, dtype=float) - float(lambda_shift)
    denominator = 1.0 - lam * float(chi0)
    if np.any(denominator <= 0.0):
        raise ValueError(
            "RPA instability on the BZ grid: 1 - (lambda - lambda_shift) * chi0 "
            f"<= 0 (max lambda' * chi0 = {float(np.max(lam * chi0)):.6g})"
        )
    n_q = lam.shape[0]
    norm = float(isotropic_components) / (n_q * n_sites)
    kt = KB_MEV_PER_K * max(float(temperature_K), 0.0)
    use_numba_matsubara = (
        tier_a_matsubara_moment_sums is not None
        and kt > 0.0
        and cutoff_mev / kt < _DIGAMMA_REGIME_CUTOFF_OVER_KT
    )
    if use_numba_matsubara:
        gamma_max = float(np.max(float(gamma0) * denominator))
        nu1 = 2.0 * np.pi * kt
        n_terms = max(
            32,
            int(np.ceil(4.0 * max(float(cutoff_mev), gamma_max) / nu1)),
        )
        tail2 = float(zeta(2, n_terms + 1) / nu1**2)
        tail4 = float(zeta(4, n_terms + 1) / nu1**4)
        tail6 = float(zeta(6, n_terms + 1) / nu1**6)
        zero_sum, total_sum = tier_a_matsubara_moment_sums(
            np.ascontiguousarray(lam.ravel()),
            float(chi0),
            float(gamma0),
            float(temperature_K),
            float(cutoff_mev),
            KB_MEV_PER_K,
            tail2,
            tail4,
            tail6,
            n_terms,
        )
        zero_total = float(zero_sum * norm)
        thermal_total = float((total_sum - zero_sum) * norm)
        if parts:
            return zero_total, thermal_total
        return zero_total + thermal_total

    chi_modes = float(chi0) / denominator
    gamma_modes = float(gamma0) * denominator
    zero_point, thermal = lorentzian_moment(
        chi_modes, gamma_modes, temperature_K, cutoff_mev, parts=True
    )
    if parts:
        return float(zero_point.sum() * norm), float(thermal.sum() * norm)
    return float((zero_point + thermal).sum() * norm)


def static_chi_modes(
    eigenvalues: ArrayLike,
    weights: ArrayLike,
    *,
    chi0: float,
    lambda_shift: float = 0.0,
) -> FloatArray:
    """Static susceptibility ``chi(Q, 0) = sum_nu w_nu chi0/(1 - lambda' chi0)``.

    Exact Kramers-Kronig of the relaxational modes (the KK integral of a
    Lorentzian closes to its static amplitude). ``eigenvalues`` and
    ``weights`` are ``(n_q, n_modes)``; returns ``(n_q,)``.
    """

    lam = np.asarray(eigenvalues, dtype=float) - float(lambda_shift)
    denominator = 1.0 - lam * float(chi0)
    if np.any(denominator <= 0.0):
        raise ValueError("RPA instability: 1 - lambda' * chi0 <= 0")
    chi_modes = float(chi0) / denominator
    return np.asarray((np.asarray(weights, dtype=float) * chi_modes).sum(axis=1))


def kk_static_chi(chipp: ArrayLike, omega_grid: ArrayLike) -> FloatArray:
    """Kramers-Kronig static susceptibility ``(2/pi) int chi''(w)/w dw``.

    Numerical form for spectra without a closed form (Tier B). ``chipp`` has
    the energy grid on its first axis; remaining axes are preserved.
    """

    omega = np.asarray(omega_grid, dtype=float)
    values = np.asarray(chipp, dtype=float)
    shape = omega.shape + (1,) * (values.ndim - 1)
    return (2.0 / np.pi) * np.trapezoid(values / omega.reshape(shape), omega, axis=0)
