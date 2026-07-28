from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dataset import PointData4D


FloatArray = NDArray[np.float64]
ModelFunction = Callable[[PointData4D, dict[str, float]], FloatArray]
ParameterLike = float | int | str


@dataclass(frozen=True)
class EnergyGaussianResolution:
    """Gaussian energy broadening evaluated on an oversampled energy grid.

    The FWHM may be constant or polynomial in energy. Numeric entries are fixed;
    string entries are interpreted as fitting-parameter names and read from the
    parameter dictionary at evaluation time.
    """

    fwhm: ParameterLike | Sequence[ParameterLike]
    oversampling: int = 5
    tail_sigma: float = 4.0
    use_absolute_energy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.oversampling < 1:
            raise ValueError("oversampling must be at least 1")
        if self.tail_sigma < 0.0:
            raise ValueError("tail_sigma must be nonnegative")

    def evaluate_model(
        self,
        data: PointData4D,
        model: ModelFunction,
        params: dict[str, float],
    ) -> FloatArray:
        """Evaluate ``model`` on dense energy grids and convolve to data points."""

        out = np.empty(data.size, dtype=float)
        groups = _group_indices_by_q(data)
        for indices in groups:
            out[indices] = self._evaluate_group(data, indices, model, params)
        return out

    def _evaluate_group(
        self,
        data: PointData4D,
        indices: FloatArray,
        model: ModelFunction,
        params: dict[str, float],
    ) -> FloatArray:
        order = np.argsort(data.E[indices])
        sorted_indices = np.asarray(indices, dtype=int)[order]
        target_e = data.E[sorted_indices]

        target_fwhm = self.fwhm_values(target_e, params)
        max_sigma = float(np.max(_sigma_from_fwhm(target_fwhm)))
        if target_e.size < 2:
            step = max_sigma / self.oversampling
        else:
            step = _dense_energy_step(target_e, self.oversampling)
        dense_e = np.arange(
            float(target_e[0] - self.tail_sigma * max_sigma),
            float(target_e[-1] + self.tail_sigma * max_sigma + 0.5 * step),
            step,
            dtype=float,
        )
        dense_data = _dense_group_data(data, sorted_indices, dense_e)
        dense_model = np.asarray(model(dense_data, params), dtype=float)
        if dense_model.shape != dense_e.shape:
            raise ValueError(
                f"model returned shape {dense_model.shape} on dense energy grid, expected {dense_e.shape}"
            )

        convolved_sorted = _convolve_to_targets(
            dense_e=dense_e,
            dense_values=dense_model,
            target_e=target_e,
            target_fwhm=target_fwhm,
        )
        convolved = np.empty_like(convolved_sorted)
        convolved[order] = convolved_sorted
        return convolved

    def fwhm_values(self, energy: FloatArray, params: dict[str, float]) -> FloatArray:
        """Return FWHM values in meV at each energy."""

        x = np.abs(energy) if self.use_absolute_energy else np.asarray(energy, dtype=float)
        if isinstance(self.fwhm, (str, int, float)):
            value = _resolve_parameter_like(self.fwhm, params)
            values = np.full_like(x, value, dtype=float)
        else:
            values = np.zeros_like(x, dtype=float)
            for power, coefficient in enumerate(self.fwhm):
                values += _resolve_parameter_like(coefficient, params) * (x**power)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("energy-resolution FWHM must be finite and positive")
        return values


def constant_fwhm_energy_resolution(
    fwhm: ParameterLike = "resolution_fwhm",
    *,
    oversampling: int = 5,
    tail_sigma: float = 4.0,
) -> EnergyGaussianResolution:
    """Create Gaussian energy resolution with constant FWHM."""

    return EnergyGaussianResolution(
        fwhm=fwhm,
        oversampling=oversampling,
        tail_sigma=tail_sigma,
        metadata={"kind": "constant_fwhm"},
    )


def polynomial_fwhm_energy_resolution(
    coefficients: Sequence[ParameterLike],
    *,
    oversampling: int = 5,
    tail_sigma: float = 4.0,
    use_absolute_energy: bool = False,
) -> EnergyGaussianResolution:
    """Create Gaussian energy resolution with polynomial FWHM.

    Coefficients are ordered from constant term upward:
    ``FWHM(E) = c0 + c1*E + c2*E**2 + ...``.
    """

    return EnergyGaussianResolution(
        fwhm=tuple(coefficients),
        oversampling=oversampling,
        tail_sigma=tail_sigma,
        use_absolute_energy=use_absolute_energy,
        metadata={"kind": "polynomial_fwhm"},
    )


def _resolve_parameter_like(value: ParameterLike, params: dict[str, float]) -> float:
    if isinstance(value, str):
        if value not in params:
            raise KeyError(f"unknown resolution parameter {value!r}")
        return float(params[value])
    return float(value)


def _group_indices_by_q(data: PointData4D) -> list[FloatArray]:
    q = np.column_stack([data.H, data.K, data.L])
    _, inverse = np.unique(q, axis=0, return_inverse=True)
    return [np.flatnonzero(inverse == group_id) for group_id in range(int(inverse.max()) + 1)]


def _dense_energy_step(energy: FloatArray, oversampling: int) -> float:
    diffs = np.diff(np.unique(energy))
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        raise ValueError("at least two distinct energy points are required for energy resolution")
    return float(np.median(diffs) / oversampling)


def _dense_group_data(data: PointData4D, sorted_indices: FloatArray, dense_e: FloatArray) -> PointData4D:
    first = int(sorted_indices[0])
    temperature: float | FloatArray | None
    if isinstance(data.temperature, np.ndarray):
        source_e = data.E[sorted_indices]
        source_t = data.temperature[sorted_indices]
        temperature = np.interp(dense_e, source_e, source_t)
    else:
        temperature = data.temperature
    return PointData4D(
        H=np.full_like(dense_e, data.H[first], dtype=float),
        K=np.full_like(dense_e, data.K[first], dtype=float),
        L=np.full_like(dense_e, data.L[first], dtype=float),
        E=dense_e,
        intensity=np.zeros_like(dense_e, dtype=float),
        sigma=np.ones_like(dense_e, dtype=float),
        temperature=temperature,
        magnetic_field=None if data.magnetic_field is None else np.array(data.magnetic_field),
        metadata=dict(data.metadata),
    )


def _subset_point_data(data: PointData4D, indices: FloatArray) -> PointData4D:
    if isinstance(data.temperature, np.ndarray):
        temperature: float | FloatArray | None = data.temperature[indices]
    else:
        temperature = data.temperature
    return PointData4D(
        H=data.H[indices],
        K=data.K[indices],
        L=data.L[indices],
        E=data.E[indices],
        intensity=data.intensity[indices],
        sigma=data.sigma[indices],
        mask=np.ones(len(indices), dtype=bool),
        temperature=temperature,
        magnetic_field=None if data.magnetic_field is None else np.array(data.magnetic_field),
        metadata=dict(data.metadata),
    )


def _sigma_from_fwhm(fwhm: FloatArray) -> FloatArray:
    return np.asarray(fwhm, dtype=float) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _convolve_to_targets(
    *,
    dense_e: FloatArray,
    dense_values: FloatArray,
    target_e: FloatArray,
    target_fwhm: FloatArray,
) -> FloatArray:
    sigma = _sigma_from_fwhm(target_fwhm)
    out = np.empty_like(target_e, dtype=float)
    for i, (energy, width) in enumerate(zip(target_e, sigma, strict=True)):
        weights = np.exp(-0.5 * ((dense_e - energy) / width) ** 2)
        denominator = _trapezoid(weights, dense_e)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("energy-resolution kernel normalization failed")
        out[i] = float(_trapezoid(dense_values * weights, dense_e) / denominator)
    return out


def _trapezoid(y: FloatArray, x: FloatArray) -> float:
    """Integrate with NumPy 1.x/2.x compatibility."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
