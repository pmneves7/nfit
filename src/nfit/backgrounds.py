from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .analysis.coordinates import q_modulus_for_spectral
from .mdhisto import MDHistoData, mdhisto_measured_bins


def subtract_powder_background(
    data: MDHistoData,
    background: MDHistoData,
    *,
    scale: float = 1.0,
    interpolation: str = "linear",
) -> MDHistoData:
    """Interpolate a powder ``|Q|, E`` background and subtract it from data.

    Input datasets are assumed statistically independent, so the scaled
    background variance is added to the data variance. Target bins outside the
    background domain are masked rather than extrapolated.
    """

    if interpolation not in {"linear", "nearest"}:
        raise ValueError("background interpolation must be 'linear' or 'nearest'")
    q_dim, energy_dim = _powder_dimensions(background)
    q_centers = np.asarray(background.axes[q_dim].centers, dtype=float)
    energy_centers = np.asarray(background.axes[energy_dim].centers, dtype=float)
    if q_centers.size < 1 or energy_centers.size < 1:
        raise ValueError("powder background axes must not be empty")
    values = np.moveaxis(background.signal, (q_dim, energy_dim), (0, 1))
    errors = np.moveaxis(background.errors, (q_dim, energy_dim), (0, 1))
    measured = np.moveaxis(mdhisto_measured_bins(background), (q_dim, energy_dim), (0, 1))
    if values.ndim != 2:
        raise ValueError("powder background must have exactly |Q| and energy dimensions")
    safe_values = np.where(measured, values, np.nan)
    safe_variance = np.where(measured, np.square(errors), np.nan)
    value_interpolator = RegularGridInterpolator(
        (q_centers, energy_centers),
        safe_values,
        method=interpolation,
        bounds_error=False,
        fill_value=np.nan,
    )
    variance_interpolator = RegularGridInterpolator(
        (q_centers, energy_centers),
        safe_variance,
        method=interpolation,
        bounds_error=False,
        fill_value=np.nan,
    )
    q = np.broadcast_to(q_modulus_for_spectral(data), data.shape)
    target_energy_dim = _energy_dimension(data)
    energy_shape = [1] * data.signal.ndim
    energy_shape[target_energy_dim] = data.shape[target_energy_dim]
    energy = np.broadcast_to(
        data.axes[target_energy_dim].centers.reshape(energy_shape), data.shape
    )
    points = np.column_stack((q.ravel(), energy.ravel()))
    interpolated = value_interpolator(points).reshape(data.shape)
    interpolated_variance = variance_interpolator(points).reshape(data.shape)
    valid_background = np.isfinite(interpolated) & np.isfinite(interpolated_variance)
    factor = float(scale)
    output_signal = np.asarray(data.signal, dtype=float) - factor * interpolated
    output_errors = np.sqrt(
        np.square(np.asarray(data.errors, dtype=float))
        + factor**2 * interpolated_variance
    )
    metadata = dict(data.metadata)
    history = list(metadata.get("background_subtractions", []))
    history.append(
        {
            "scale": factor,
            "interpolation": interpolation,
            "source": background.metadata.get("source_file"),
        }
    )
    metadata["background_subtractions"] = history
    return replace(
        data,
        signal=np.where(valid_background, output_signal, np.nan),
        errors=np.where(valid_background, output_errors, np.nan),
        mask=np.asarray(data.mask, dtype=bool) | ~valid_background,
        metadata=metadata,
    )


def _powder_dimensions(data: MDHistoData) -> tuple[int, int]:
    q_dimensions = [
        index for index, axis in enumerate(data.axes) if axis.role == "q_modulus"
    ]
    energy_dimensions = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(q_dimensions) != 1 or len(energy_dimensions) != 1:
        raise ValueError("background source must contain one |Q| axis and one energy axis")
    return q_dimensions[0], energy_dimensions[0]


def _energy_dimension(data: MDHistoData) -> int:
    dimensions = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(dimensions) != 1:
        raise ValueError("background subtraction requires one energy-transfer axis")
    return dimensions[0]
