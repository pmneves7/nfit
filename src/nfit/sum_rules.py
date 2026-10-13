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

Field-on (Tier-B) responses with an affine inverse local propagator have
complex matrix poles. Their trace uses exact logarithmic zero-point integrals
and finite-cutoff Matsubara integrals with a bounded tail. General propagators
use checked adaptive direct solves in :mod:`nfit.closures`. The sampled-data
``trace_moment_quadrature`` remains a numerical referee.
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


class MomentIntegrationError(ValueError):
    """A numerical moment integral could not meet its error tolerance."""


_MOMENT_RTOL = 2.0e-12
_MOMENT_ATOL = 2.0e-14


def _pole_log_ratio(poles: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Log[(z-i)/z], continuous for poles with positive real part."""

    return np.log1p(-1j / poles)


def _pole_power_integrals(
    poles: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], ...]:
    """Integrals of x^p/(z-i x) on [0,1], for p=1,3,5.

    Polynomial division is accurate near poles; a geometric series avoids its
    cancellation when the pole is far beyond the integration interval.
    """

    near = np.abs(poles) <= 4.0
    current = 1j * _pole_log_ratio(poles[near])
    result = []
    for power in range(1, 6):
        current = 1j / power - 1j * poles[near] * current
        if power not in (1, 3, 5):
            continue
        integral = np.empty_like(poles)
        integral[near] = current
        inverse = 1.0 / poles[~near]
        term = inverse.copy()
        series = term / (power + 1)
        for order in range(1, 32):
            term *= 1j * inverse
            series += term / (power + order + 1)
        integral[~near] = series
        result.append(integral)
    return tuple(result)


def _matsubara_pole_kernel(
    poles: NDArray[np.complex128], frequencies: FloatArray
) -> NDArray[np.complex128]:
    """Integral x/[(x^2+v^2)(z-i x)] on [0,1].

    The log1p divided difference handles z=v without an artificial singularity.
    Both energies are normalized by the finite integration cutoff.
    """

    z = poles[None, :]
    v = frequencies[:, None]
    difference = -1j * (z - v) / (v * (z - 1j))
    ratio = np.empty_like(difference)
    small = np.abs(difference) < 1e-4
    x = difference[small]
    ratio[small] = 1.0 - x / 2.0 + x**2 / 3.0 - x**3 / 4.0 + x**4 / 5.0
    ratio[~small] = np.log1p(difference[~small]) / difference[~small]
    return 1j * (np.arctan(1.0 / v) - z / (v * (z - 1j)) * ratio) / (z + v)


def _rational_trace_moment_parts(
    poles: NDArray[np.complex128],
    residues: NDArray[np.complex128],
    temperature_K: float,
    cutoff_mev: float,
    *,
    initial_terms: int = 32,
    absolute_tolerance: float = _MOMENT_ATOL,
) -> tuple[float, float]:
    """Integrate Im sum r/(lambda-i E), including nonreciprocal spectra.

    The zero-point integral is an exact logarithm. For the thermal part, the
    coth Matsubara expansion has analytic finite-cutoff integrals. Its tail
    through nu^-6 is summed with Hurwitz zeta; positivity bounds the remaining
    tail by Lambda^6 int E chi'' dE sum nu^-8. No conjugate-pole pairing or
    momentum inversion is assumed. The static trace must be real.
    """

    poles = np.asarray(poles, dtype=complex).ravel()
    residues = np.asarray(residues, dtype=complex).ravel()
    if (
        poles.shape != residues.shape or np.any(poles.real <= 0)
        or not np.all(np.isfinite(poles)) or not np.all(np.isfinite(residues))
    ):
        raise MomentIntegrationError("moment integration requires poles in the stable half-plane")
    static = np.sum(residues / poles)
    if abs(static.imag) > 1e-9 * max(abs(static.real), 1.0):
        raise MomentIntegrationError("moment integration requires a real static trace")
    cutoff = float(cutoff_mev)
    zero_point = float(np.real(np.sum(residues * _pole_log_ratio(poles / cutoff))) / np.pi)
    if not np.isfinite(zero_point) or zero_point < -absolute_tolerance:
        raise MomentIntegrationError("moment integration produced nonpositive spectral amplitude")
    zero_point = max(zero_point, 0.0)
    if temperature_K <= 0:
        return zero_point, 0.0
    kt = KB_MEV_PER_K * temperature_K
    # Positivity gives a rigorous omitted thermal bound:
    # 2 n_B(40 kT) * zero_point(full cutoff) < 9e-18 * zero_point.
    thermal_cutoff = min(cutoff, 40.0 * kt)
    z = poles / thermal_cutoff
    nu1 = 2.0 * np.pi * kt / thermal_cutoff
    log_ratio = _pole_log_ratio(z)
    zero_point_low = float(np.real(np.sum(residues * log_ratio)) / np.pi)
    classical = -float(np.imag(np.sum((residues / z) * log_ratio)))
    power_integrals = _pole_power_integrals(z)
    m1, m3, m5 = [float(np.imag(np.sum(residues * integral))) for integral in power_integrals]
    prefactor = 4.0 * kt / (np.pi * thermal_cutoff)
    count = max(32, min(int(initial_terms), 256), int(np.ceil(8.0 / nu1)))
    summed = 0.0
    completed = 0
    while count <= 4096:
        # Bound the complex working arrays independently of the number of poles.
        frequency_block = max(1, min(16, 250_000 // max(poles.size, 1)))
        for start in range(completed + 1, count + 1, frequency_block):
            frequencies = nu1 * np.arange(start, min(start + frequency_block, count + 1))
            summed += float(np.imag(np.sum(_matsubara_pole_kernel(z, frequencies) * residues[None])))
        tail = (
            m1 * zeta(2, count + 1) / nu1**2
            - m3 * zeta(4, count + 1) / nu1**4
            + m5 * zeta(6, count + 1) / nu1**6
        )
        total_low = prefactor * (0.5 * classical + summed + tail)
        error_bound = prefactor * abs(m1) * zeta(8, count + 1) / nu1**8
        tolerance = absolute_tolerance + _MOMENT_RTOL * max(abs(total_low), abs(zero_point))
        if error_bound <= tolerance:
            thermal = total_low - zero_point_low
            if not np.isfinite(thermal) or thermal < -tolerance or zero_point < -tolerance:
                raise MomentIntegrationError("moment integration produced nonpositive spectral amplitude")
            return max(zero_point, 0.0), max(thermal, 0.0)
        completed = count
        count *= 2
    raise MomentIntegrationError("moment integration did not converge its Matsubara tail")

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
    *,
    zero_energy_slope: ArrayLike | None = None,
) -> FloatArray:
    """Energy integral ``(1/pi) int coth(w/2T) Tr chi''(w) dw`` on a grid.

    ``chipp_trace`` has energy on its first axis; remaining axes are preserved.
    At zero energy the finite integrand is ``2 k_B T d chi''/dE``. Supply that
    slope when known; otherwise the first positive row estimates it. This
    sampled-data helper cannot resolve spectral features absent from its grid;
    field-on closures use checked rational or adaptive integration instead.
    """

    omega = np.asarray(omega_grid, dtype=float)
    values = np.asarray(chipp_trace, dtype=float)
    if omega.ndim != 1 or omega.size < 2 or np.any(np.diff(omega) <= 0):
        raise ValueError("energy grid must be strictly increasing with at least two rows")
    if omega[0] < 0 or values.shape[0] != omega.size:
        raise ValueError("spectral rows must match a nonnegative energy grid")
    weight = coth_weight(np.where(omega == 0, 1.0, omega), temperature_K)
    shape = weight.shape + (1,) * (values.ndim - 1)
    integrand = values * weight.reshape(shape)
    if omega[0] == 0:
        slope = values[1] / omega[1] if zero_energy_slope is None else zero_energy_slope
        integrand[0] = 2.0 * KB_MEV_PER_K * max(temperature_K, 0.0) * np.asarray(slope)
    return np.trapezoid(integrand, omega, axis=0) / np.pi


def tier_b_omega_grid(cutoff_mev: float, n_points: int = 200) -> FloatArray:
    """Energy quadrature grid for Tier-B trace integrals.

    Includes zero, with logarithmic spacing below the linear segment. The
    lower positive energy scales with the cutoff and point count, so even a
    very small cutoff gives an ordered grid. No fixed grid guarantees that an
    arbitrarily narrow resonance is resolved; this is a plotting/referee grid.
    """

    cutoff = float(cutoff_mev)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("energy cutoff must be finite and positive")
    count = max(int(n_points), 16)
    n_log = max(count // 4, 4)
    n_lin = count - n_log - 1
    join = min(1.0, 0.1 * cutoff)
    low = np.geomspace(join / count**2, join, n_log, endpoint=False)
    high = np.linspace(join, cutoff, n_lin)
    return np.concatenate([[0.0], low, high])


def bz_sample_hkl(n: int) -> FloatArray:
    """Uniform ``n^3`` Brillouin-zone sampling grid in fractional H, K, L.

    Midpoint-shifted Monkhorst-Pack points ``(i + 1/2)/n`` on the reciprocal
    torus ``[0, 1)^3`` of the declared cell. A full conventional-cell site
    trace includes all folded primitive momenta; reducing the site network
    without also transforming and expanding the momentum grid does not
    preserve a local moment integral. The half-step shift avoids Gamma. It
    does not remove every zone-boundary or high-symmetry point.
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
    integrand = values / np.where(omega == 0, 1.0, omega).reshape(shape)
    if omega[0] == 0:
        integrand[0] = values[1] / omega[1]
    return (2.0 / np.pi) * np.trapezoid(integrand, omega, axis=0)
