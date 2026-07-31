from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

KB_MEV_PER_K = 0.08617333262
# Standard magnetic neutron cross-section constant
# (gamma r_0 / 2)^2 = 0.07265 barn / mu_B^2. Equivalently, a spin-operator
# response carries (gamma r_0)^2 (g/2)^2 = 0.07265 g^2 barn. This is the only
# form used inside nfit; every cross-section expression multiplies by it once.
MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ = 0.07265
# Compatibility alias: the square root of the constant above, i.e. gamma r_0 / 2
# in sqrt(barn) / mu_B. Retained because it is part of the public API; prefer
# MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ, which needs no squaring at use.
MAGNETIC_GAMMA0_PER_MU_B = np.sqrt(MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ)
FloatArray = NDArray[np.float64]
MILLIBARN_PER_BARN = 1000.0


def magnetic_moment_factor(
    moment_unit: str = "mu_B_squared",
    g_factor: float | None = None,
) -> float:
    """Return the numerical factor converting a declared response to moment units.

    A susceptibility already expressed for the magnetic moment in
    ``mu_B^2/meV`` carries no additional Landé factor. A spin-operator
    susceptibility in ``spin^2/meV`` is multiplied by ``g^2``. With nfit's
    constant ``(gamma r_0 / 2)^2``, this is algebraically identical to writing
    the cross section with ``(gamma r_0)^2 (g/2)^2``.

    No vacuum-permeability factor belongs here. ``mu_0`` enters when converting
    microscopic moment susceptibility to rationalized SI ``M/H`` units, not
    when converting a response already in ``spin^2/meV`` or ``mu_B^2/meV`` to
    a neutron cross section.
    """

    if moment_unit == "mu_B_squared":
        return 1.0
    if moment_unit != "spin_squared":
        raise ValueError("moment_unit must be 'mu_B_squared' or 'spin_squared'")
    if g_factor is None or not np.isfinite(g_factor) or g_factor <= 0.0:
        raise ValueError("spin_squared susceptibility requires a finite positive g_factor")
    return float(g_factor) ** 2


def kf_over_ki(
    E_meV: ArrayLike,
    *,
    incident_energy_meV: float | None = None,
    final_energy_meV: float | None = None,
) -> FloatArray:
    """Return ``k_f/k_i`` for ``E = E_i - E_f``.

    Direct geometry supplies fixed ``E_i`` and indirect geometry supplies
    fixed ``E_f``.  Energetically inaccessible points are returned as NaN.
    """

    if incident_energy_meV is None and final_energy_meV is None:
        raise ValueError("provide incident_energy_meV or final_energy_meV")
    energy = np.asarray(E_meV, dtype=float)
    if incident_energy_meV is not None:
        incident = float(incident_energy_meV)
        if not np.isfinite(incident) or incident <= 0.0:
            raise ValueError("incident_energy_meV must be finite and positive")
        ratio_sq = (incident - energy) / incident
    else:
        final = float(final_energy_meV)
        if not np.isfinite(final) or final <= 0.0:
            raise ValueError("final_energy_meV must be finite and positive")
        ratio_sq = final / (final + energy)
    ratio = np.full(np.shape(ratio_sq), np.nan, dtype=float)
    np.sqrt(
        ratio_sq,
        out=ratio,
        where=np.isfinite(ratio_sq) & (ratio_sq >= 0.0),
    )
    return ratio


def bose_denominator(
    E_meV: ArrayLike,
    temperature_K: float | ArrayLike,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Return ``1 - exp(-E/k_B T)`` using meV and K.

    This is the fluctuation-dissipation factor denominator connecting chi'' to
    S(Q,E), up to convention-dependent constants. For positive E at T=0 the
    limiting denominator is one.

    ``temperature_K`` may be a scalar or an array broadcastable against
    ``E_meV`` (e.g. one temperature per point).
    """

    E = np.asarray(E_meV, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    if np.any(temperature < 0):
        raise ValueError("temperature_K must be nonnegative")

    E, temperature = np.broadcast_arrays(E, temperature)
    denom = np.empty(E.shape, dtype=float)

    zero = temperature == 0.0
    if np.any(zero):
        denom[zero] = np.where(E[zero] >= 0.0, 1.0, np.nan)

    finite = ~zero
    if np.any(finite):
        x = E[finite] / (KB_MEV_PER_K * temperature[finite])
        d = -np.expm1(-x)
        small = np.abs(x) < eps
        if np.any(small):
            d = np.array(d, dtype=float, copy=True)
            d[small] = x[small]
        denom[finite] = d
    return denom


def intensity_from_chipp(
    chipp: ArrayLike,
    E_meV: ArrayLike,
    temperature_K: float | ArrayLike,
    *,
    scale: float = 1.0,
    form_factor_sq: float | ArrayLike = 1.0,
    polarization: float | ArrayLike = 1.0,
    background: float | ArrayLike = 0.0,
    include_bose: bool = True,
) -> FloatArray:
    """Convert chi''(Q,E) to measured neutron intensity.

    The convention is

    ``I = scale * gamma_0^2 * |f(Q)|^2 * P(Q) * chi'' /
    {pi [1 - exp(-E/kBT)]} + background``, where
    ``gamma_0^2 = (gamma r_0 / 2)^2 = 0.07265 barn / mu_B^2``.

    ``temperature_K`` may be a scalar or an array broadcastable to ``chipp``
    (per-point temperatures). ``form_factor_sq``, ``polarization``, and
    ``background`` may be scalars or arrays broadcastable to ``chipp``.
    The kinematic ``k_f/k_i`` factor is normalized on the dataset side when a
    dataset declares that it was not already included in its reduction.
    """

    signal = np.asarray(chipp, dtype=float)
    if include_bose:
        signal = signal / bose_denominator(E_meV, temperature_K)
    return (
        float(scale)
        * MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ
        / np.pi
        * np.asarray(form_factor_sq, dtype=float)
        * np.asarray(polarization, dtype=float)
        * signal
        + np.asarray(background, dtype=float)
    )


def chipp_from_intensity(
    intensity: ArrayLike,
    E_meV: ArrayLike,
    temperature_K: float | ArrayLike,
    *,
    scale: float = 1.0,
    form_factor_sq: float | ArrayLike = 1.0,
    polarization: float | ArrayLike = 1.0,
    background: float | ArrayLike = 0.0,
    include_bose: bool = True,
) -> FloatArray:
    """Invert :func:`intensity_from_chipp` under the same convention."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    form_factor = np.asarray(form_factor_sq, dtype=float)
    polarization_array = np.asarray(polarization, dtype=float)
    if np.any(~np.isfinite(form_factor)) or np.any(form_factor <= 0.0):
        raise ValueError("form_factor_sq must be finite and positive")
    if np.any(~np.isfinite(polarization_array)) or np.any(polarization_array <= 0.0):
        raise ValueError("polarization must be finite and positive")
    signal = (np.asarray(intensity, dtype=float) - np.asarray(background, dtype=float))
    signal = signal * np.pi / (
        float(scale)
        * MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ
        * form_factor
        * polarization_array
    )
    if include_bose:
        signal = signal * bose_denominator(E_meV, temperature_K)
    return np.asarray(signal, dtype=float)


def cross_section_from_chipp(
    chipp: ArrayLike,
    E_meV: ArrayLike,
    temperature_K: float | ArrayLike,
    *,
    form_factor_sq: float | ArrayLike = 1.0,
    polarization: float | ArrayLike = 1.0,
    kf_ki: float | ArrayLike = 1.0,
    include_bose: bool = True,
    moment_unit: str = "mu_B_squared",
    g_factor: float | None = None,
) -> FloatArray:
    """Return absolute magnetic ``d2sigma/dOmega/dE`` in barn/(sr meV).

    ``chipp`` is in ``mu_B^2/meV`` when ``moment_unit="mu_B_squared"`` and in
    ``spin^2/meV`` when ``moment_unit="spin_squared"``. The returned cross
    section retains the declared per-ion/formula-unit/cell basis. ``kf_ki`` is
    explicit so data normalized to remove the kinematic factor can leave it at
    one.
    """

    return np.asarray(kf_ki, dtype=float) * intensity_from_chipp(
        np.asarray(chipp, dtype=float) * magnetic_moment_factor(moment_unit, g_factor),
        E_meV,
        temperature_K,
        scale=1.0,
        form_factor_sq=form_factor_sq,
        polarization=polarization,
        include_bose=include_bose,
    )


def quasistatic_cross_section_from_chi(
    chi_static: ArrayLike,
    temperature_K: float | ArrayLike,
    *,
    form_factor_sq: float | ArrayLike = 1.0,
    polarization: float | ArrayLike = 1.0,
    moment_unit: str = "mu_B_squared",
    g_factor: float | None = None,
) -> FloatArray:
    """Return energy-integrated magnetic ``dsigma/dOmega`` in barn/sr.

    In the quasistatic limit, where the magnetic linewidth is small compared
    with ``k_B T`` and lies inside the experimental energy acceptance, the
    fluctuation--dissipation theorem gives

    ``S(Q) = integral S(Q,E) dE ~= k_B T chi'(Q,0)``.

    ``chi_static`` is one Cartesian static-susceptibility component in
    ``spin^2/meV`` when ``moment_unit="spin_squared"`` or in
    ``mu_B^2/meV`` when ``moment_unit="mu_B_squared"``. The returned cross
    section retains the declared per-ion/formula-unit/cell normalization.
    """

    temperature = np.asarray(temperature_K, dtype=float)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError(
            "temperature_K must be finite and positive for the quasistatic approximation"
        )
    equal_time_response = (
        KB_MEV_PER_K
        * temperature
        * np.asarray(chi_static, dtype=float)
        * magnetic_moment_factor(moment_unit, g_factor)
    )
    return (
        MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ
        * np.asarray(form_factor_sq, dtype=float)
        * np.asarray(polarization, dtype=float)
        * equal_time_response
    )


def chipp_from_cross_section(
    cross_section: ArrayLike,
    E_meV: ArrayLike,
    temperature_K: float | ArrayLike,
    *,
    form_factor_sq: float | ArrayLike = 1.0,
    polarization: float | ArrayLike = 1.0,
    kf_ki: float | ArrayLike = 1.0,
    include_bose: bool = True,
    moment_unit: str = "mu_B_squared",
    g_factor: float | None = None,
) -> FloatArray:
    """Invert the cross section to the response selected by ``moment_unit``."""

    ratio = np.asarray(kf_ki, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("kf_ki must be finite and positive")
    response = chipp_from_intensity(
        np.asarray(cross_section, dtype=float) / ratio,
        E_meV,
        temperature_K,
        scale=1.0,
        form_factor_sq=form_factor_sq,
        polarization=polarization,
        include_bose=include_bose,
    )
    return response / magnetic_moment_factor(moment_unit, g_factor)
