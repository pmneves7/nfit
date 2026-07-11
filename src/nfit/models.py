from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dataset import PointData4D


FloatArray = NDArray[np.float64]
PointModelFunction = Callable[[PointData4D, dict[str, float]], FloatArray]


def constant_intensity_model(
    data: PointData4D,
    params: dict[str, float],
    *,
    parameter: str = "constant",
    value: float | None = None,
) -> FloatArray:
    """Measured-intensity model constant in momentum and energy.

    By default the constant value is read from ``params[parameter]``. Pass a
    numeric ``value`` to make a fixed component that does not consume a fitting
    parameter.
    """

    level = float(params[parameter] if value is None else value)
    return np.full(data.size, level, dtype=float)


def make_constant_intensity_model(
    parameter: str = "constant",
    *,
    value: float | None = None,
) -> PointModelFunction:
    """Return a reusable constant measured-intensity model callable."""

    def model(data: PointData4D, params: dict[str, float]) -> FloatArray:
        return constant_intensity_model(data, params, parameter=parameter, value=value)

    return model


def compound_additive_model(*components: PointModelFunction) -> PointModelFunction:
    """Return a model that sums multiple primitive measured-intensity models."""

    if len(components) == 0:
        raise ValueError("compound_additive_model requires at least one component")

    def model(data: PointData4D, params: dict[str, float]) -> FloatArray:
        total = np.zeros(data.size, dtype=float)
        for component in components:
            values = np.asarray(component(data, params), dtype=float)
            if values.shape != total.shape:
                raise ValueError(
                    f"compound component returned shape {values.shape}, expected {total.shape}"
                )
            total += values
        return total

    return model


def quadratic_distance_rlu(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    q0: tuple[float, float, float],
    kappa: float | tuple[float, float, float],
) -> FloatArray:
    """Dimensionless squared distance from ``q0`` in RLU.

    ``kappa`` is an inverse correlation-length-like width in RLU. It may be a
    scalar isotropic width or a three-tuple for diagonal anisotropy.
    """

    H_arr, K_arr, L_arr = np.broadcast_arrays(H, K, L)
    dh = np.asarray(H_arr, dtype=float) - q0[0]
    dk = np.asarray(K_arr, dtype=float) - q0[1]
    dl = np.asarray(L_arr, dtype=float) - q0[2]

    if np.isscalar(kappa):
        width = float(kappa)
        if width <= 0:
            raise ValueError("kappa must be positive")
        return (dh * dh + dk * dk + dl * dl) / (width * width)

    if len(kappa) != 3:
        raise ValueError("anisotropic kappa must contain exactly three widths")
    widths = np.asarray(kappa, dtype=float)
    if np.any(widths <= 0):
        raise ValueError("all kappa widths must be positive")
    return (dh / widths[0]) ** 2 + (dk / widths[1]) ** 2 + (dl / widths[2]) ** 2


def paramagnon_chipp(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    E: ArrayLike,
    *,
    amplitude: float,
    q0: tuple[float, float, float],
    kappa: float | tuple[float, float, float],
    omega_sf: float,
    enforce_odd: bool = True,
) -> FloatArray:
    """Overdamped single-Q paramagnon chi''(Q,E).

    Formula used by the initial package:

    ``a(Q) = 1 + |Q - Q0|^2 / kappa^2``

    ``chi''(Q,E) = A * (E / omega_sf) / (a(Q)^2 + (E / omega_sf)^2)``

    Parameters use RLU for H,K,L and meV for E and ``omega_sf``. If
    ``enforce_odd`` is true, negative energy transfer gives negative chi''.
    """

    if omega_sf <= 0:
        raise ValueError("omega_sf must be positive")
    if amplitude < 0:
        raise ValueError("amplitude must be nonnegative")

    H_arr, K_arr, L_arr, E_arr = np.broadcast_arrays(H, K, L, E)
    E_float = np.asarray(E_arr, dtype=float)
    energy = E_float if enforce_odd else np.abs(E_float)
    q2 = quadratic_distance_rlu(H_arr, K_arr, L_arr, q0=q0, kappa=kappa)
    a_q = 1.0 + q2
    x = energy / float(omega_sf)
    return float(amplitude) * x / (a_q * a_q + x * x)


def multi_q_paramagnon_chipp(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    E: ArrayLike,
    *,
    centers: Sequence[tuple[float, float, float]],
    amplitude: float,
    kappa: float | tuple[float, float, float],
    omega_sf: float,
    weights: Sequence[float] | None = None,
    enforce_odd: bool = True,
) -> FloatArray:
    """Sum of symmetry-related or independent overdamped paramagnon peaks."""

    if len(centers) == 0:
        raise ValueError("centers must contain at least one Q position")
    if weights is None:
        weights = [1.0] * len(centers)
    if len(weights) != len(centers):
        raise ValueError("weights must match centers")

    _, _, _, E_arr = np.broadcast_arrays(H, K, L, E)
    total = np.zeros_like(np.asarray(E_arr, dtype=float), dtype=float)
    for q0, weight in zip(centers, weights):
        total += float(weight) * paramagnon_chipp(
            H,
            K,
            L,
            E,
            amplitude=amplitude,
            q0=q0,
            kappa=kappa,
            omega_sf=omega_sf,
            enforce_odd=enforce_odd,
        )
    return total


def relaxational_chipp(chi_q: ArrayLike, gamma_q: ArrayLike, E: ArrayLike) -> FloatArray:
    """Relaxational chi'' in meV units.

    ``chi''(Q,E) = chi_q * E * Gamma_q / (E^2 + Gamma_q^2)``

    ``Gamma_q`` must be positive and use the same energy units as ``E``.
    """

    chi_arr, gamma_arr, E_arr = np.broadcast_arrays(chi_q, gamma_q, E)
    gamma = np.asarray(gamma_arr, dtype=float)
    if np.any(gamma <= 0):
        raise ValueError("gamma_q must be positive")
    energy = np.asarray(E_arr, dtype=float)
    return np.asarray(chi_arr, dtype=float) * energy * gamma / (energy * energy + gamma * gamma)


def constant_background(E: ArrayLike, c0: float) -> FloatArray:
    """Constant measured-intensity background."""

    return np.full_like(np.asarray(E, dtype=float), fill_value=float(c0), dtype=float)


def linear_background(E: ArrayLike, c0: float, c1: float) -> FloatArray:
    """Measured-intensity background linear in energy transfer E in meV."""

    return float(c0) + float(c1) * np.asarray(E, dtype=float)
