from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


KB_MEV_PER_K = 0.08617333262
FloatArray = NDArray[np.float64]


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

    ``I = scale * |f(Q)|^2 * P(Q) * chi'' / [1 - exp(-E/kBT)] + background``.

    ``temperature_K`` may be a scalar or an array broadcastable to ``chipp``
    (per-point temperatures). ``form_factor_sq``, ``polarization``, and
    ``background`` may be scalars or arrays broadcastable to ``chipp``.
    Absolute prefactors are intentionally not hidden here; users should
    document the normalization in metadata.
    """

    signal = np.asarray(chipp, dtype=float)
    if include_bose:
        signal = signal / bose_denominator(E_meV, temperature_K)
    return (
        float(scale)
        * np.asarray(form_factor_sq, dtype=float)
        * np.asarray(polarization, dtype=float)
        * signal
        + np.asarray(background, dtype=float)
    )
