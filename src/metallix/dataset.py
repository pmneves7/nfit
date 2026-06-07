from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class PointData4D:
    """Flattened 4D neutron data for fitting.

    Parameters
    ----------
    H, K, L
        Reciprocal lattice coordinates in reciprocal lattice units (RLU).
    E
        Energy transfer in meV. Positive E is neutron energy loss.
    intensity
        Measured neutron intensity in arbitrary or absolute units.
    sigma
        One-standard-deviation uncertainty of ``intensity`` in the same units.
    mask
        Optional boolean mask. True means the point is eligible for analysis.
    temperature
        Sample temperature in K, either scalar or one value per point.
    metadata
        Free-form metadata. Use this for units, sample, scan, and normalization
        notes; do not hide physics assumptions in code.
    """

    H: ArrayLike
    K: ArrayLike
    L: ArrayLike
    E: ArrayLike
    intensity: ArrayLike
    sigma: ArrayLike
    mask: ArrayLike | None = None
    temperature: float | ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.H = _as_float_1d("H", self.H)
        self.K = _as_float_1d("K", self.K)
        self.L = _as_float_1d("L", self.L)
        self.E = _as_float_1d("E", self.E)
        self.intensity = _as_float_1d("intensity", self.intensity)
        self.sigma = _as_float_1d("sigma", self.sigma)

        shape = self.H.shape
        for name in ("K", "L", "E", "intensity", "sigma"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match H shape {shape}")

        if self.mask is None:
            self.mask = np.ones(shape, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != shape:
                raise ValueError(f"mask shape {self.mask.shape} does not match H shape {shape}")

        if self.temperature is not None and not np.isscalar(self.temperature):
            temp = _as_float_1d("temperature", self.temperature)
            if temp.shape != shape:
                raise ValueError(
                    f"temperature shape {temp.shape} does not match H shape {shape}"
                )
            self.temperature = temp

    @property
    def size(self) -> int:
        """Number of flattened data points."""

        return int(self.H.size)

    def valid_mask(self, require_positive_sigma: bool = True) -> BoolArray:
        """Return mask of finite, usable points.

        By default, nonpositive uncertainties are excluded. The original mask is
        respected and never modified in place.
        """

        mask = np.asarray(self.mask, dtype=bool).copy()
        mask &= np.isfinite(self.H)
        mask &= np.isfinite(self.K)
        mask &= np.isfinite(self.L)
        mask &= np.isfinite(self.E)
        mask &= np.isfinite(self.intensity)
        mask &= np.isfinite(self.sigma)
        if require_positive_sigma:
            mask &= self.sigma > 0.0
        if isinstance(self.temperature, np.ndarray):
            mask &= np.isfinite(self.temperature)
        return mask

    def valid(self, require_positive_sigma: bool = True) -> PointData4D:
        """Return a new data object containing only valid points."""

        mask = self.valid_mask(require_positive_sigma=require_positive_sigma)
        temp: float | FloatArray | None
        if isinstance(self.temperature, np.ndarray):
            temp = self.temperature[mask]
        else:
            temp = self.temperature
        return PointData4D(
            H=self.H[mask],
            K=self.K[mask],
            L=self.L[mask],
            E=self.E[mask],
            intensity=self.intensity[mask],
            sigma=self.sigma[mask],
            mask=np.ones(int(np.count_nonzero(mask)), dtype=bool),
            temperature=temp,
            metadata=dict(self.metadata),
        )

    def coordinates(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return ``(H, K, L, E)`` arrays."""

        return self.H, self.K, self.L, self.E


def from_arrays(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    E: ArrayLike,
    intensity: ArrayLike,
    sigma: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    temperature: float | ArrayLike | None = None,
    metadata: dict[str, Any] | None = None,
) -> PointData4D:
    """Construct :class:`PointData4D` from array-like inputs."""

    return PointData4D(
        H=H,
        K=K,
        L=L,
        E=E,
        intensity=intensity,
        sigma=sigma,
        mask=mask,
        temperature=temperature,
        metadata={} if metadata is None else dict(metadata),
    )


def _as_float_1d(name: str, value: ArrayLike) -> FloatArray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got shape {arr.shape}")
    return arr

