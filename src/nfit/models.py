from __future__ import annotations

from collections.abc import Callable

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
