# Initial Code Skeleton

This is a suggested starting point for the core model and fitting functions. Treat it as scaffolding, not final API design.

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chi4d"
version = "0.0.1"
description = "4D inelastic neutron scattering susceptibility analysis"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scipy",
    "matplotlib",
    "xarray",
    "h5py",
]

[project.optional-dependencies]
fit = ["lmfit"]
uncertainty = ["emcee"]
parallel = ["dask", "zarr"]
dev = ["pytest", "ruff", "black"]
```

## src/chi4d/dataset.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class PointData4D:
    H: np.ndarray
    K: np.ndarray
    L: np.ndarray
    E: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray | None = None
    temperature: float | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = [self.H, self.K, self.L, self.E, self.intensity, self.sigma]
        arrays = [np.asarray(a, dtype=float) for a in arrays]
        self.H, self.K, self.L, self.E, self.intensity, self.sigma = arrays
        shape = self.H.shape
        for name, arr in zip(["K", "L", "E", "intensity", "sigma"], arrays[1:]):
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match H shape {shape}")
        if self.mask is None:
            self.mask = np.ones(shape, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != shape:
                raise ValueError(f"mask shape {self.mask.shape} does not match H shape {shape}")

    def valid_mask(self, sigma_positive: bool = True) -> np.ndarray:
        mask = np.asarray(self.mask, dtype=bool).copy()
        mask &= np.isfinite(self.H) & np.isfinite(self.K) & np.isfinite(self.L)
        mask &= np.isfinite(self.E) & np.isfinite(self.intensity) & np.isfinite(self.sigma)
        if sigma_positive:
            mask &= self.sigma > 0
        return mask

    def valid(self) -> "PointData4D":
        m = self.valid_mask()
        temp = self.temperature
        if isinstance(temp, np.ndarray):
            temp = temp[m]
        return PointData4D(
            H=self.H[m], K=self.K[m], L=self.L[m], E=self.E[m],
            intensity=self.intensity[m], sigma=self.sigma[m],
            mask=np.ones(np.count_nonzero(m), dtype=bool),
            temperature=temp,
            metadata=dict(self.metadata),
        )
```

## src/chi4d/models.py

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np


def _safe_positive_energy(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.asarray(E, dtype=float)


def quadratic_distance_rlu(
    H: np.ndarray,
    K: np.ndarray,
    L: np.ndarray,
    q0: tuple[float, float, float],
    widths: tuple[float, float, float] | float,
) -> np.ndarray:
    """Return dimensionless squared distance from q0 in RLU.

    If widths is a scalar, use isotropic width. If widths is a 3-tuple,
    use diagonal anisotropic widths.
    """
    dh = np.asarray(H) - q0[0]
    dk = np.asarray(K) - q0[1]
    dl = np.asarray(L) - q0[2]
    if np.isscalar(widths):
        w = float(widths)
        return (dh * dh + dk * dk + dl * dl) / (w * w)
    wh, wk, wl = widths
    return (dh / wh) ** 2 + (dk / wk) ** 2 + (dl / wl) ** 2


def paramagnon_chipp(
    H: np.ndarray,
    K: np.ndarray,
    L: np.ndarray,
    E: np.ndarray,
    amplitude: float,
    q0: tuple[float, float, float],
    kappa: float | tuple[float, float, float],
    omega_sf: float,
) -> np.ndarray:
    """Overdamped paramagnon chi'' for positive energy transfer.

    Formula:
        a(Q) = 1 + |Q-Q0|^2/kappa^2
        chi'' = amplitude * (E/omega_sf) / [a(Q)^2 + (E/omega_sf)^2]

    E and omega_sf must use the same energy units, typically meV.
    """
    E = np.asarray(E, dtype=float)
    if omega_sf <= 0:
        raise ValueError("omega_sf must be positive")
    q2 = quadratic_distance_rlu(H, K, L, q0=q0, widths=kappa)
    a = 1.0 + q2
    x = E / omega_sf
    return amplitude * x / (a * a + x * x)


def multi_q_paramagnon_chipp(
    H: np.ndarray,
    K: np.ndarray,
    L: np.ndarray,
    E: np.ndarray,
    centers: Sequence[tuple[float, float, float]],
    amplitude: float,
    kappa: float | tuple[float, float, float],
    omega_sf: float,
) -> np.ndarray:
    total = np.zeros_like(np.asarray(E, dtype=float))
    for q0 in centers:
        total += paramagnon_chipp(H, K, L, E, amplitude, q0, kappa, omega_sf)
    return total


def relaxational_chipp(chi_q: np.ndarray, gamma_q: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Relaxational chi'' = chi_q * E Gamma_q / (E^2 + Gamma_q^2)."""
    E = np.asarray(E, dtype=float)
    gamma_q = np.asarray(gamma_q, dtype=float)
    chi_q = np.asarray(chi_q, dtype=float)
    return chi_q * E * gamma_q / (E * E + gamma_q * gamma_q)
```

## src/chi4d/cross_section.py

```python
from __future__ import annotations

import numpy as np

KB_MEV_PER_K = 0.08617333262


def bose_denominator(E_meV: np.ndarray, temperature_K: float, eps: float = 1e-12) -> np.ndarray:
    """Return 1 - exp(-E/kBT) using expm1 for numerical stability."""
    E = np.asarray(E_meV, dtype=float)
    if temperature_K <= 0:
        # At T=0 and E>0, denominator is 1. For E<0, this limit is singular.
        return np.where(E >= 0, 1.0, np.nan)
    x = E / (KB_MEV_PER_K * temperature_K)
    denom = -np.expm1(-x)
    small = np.abs(denom) < eps
    if np.any(small):
        denom = denom.copy()
        # Linearized small-x limit: 1 - exp(-x) ~ x.
        denom[small] = x[small]
    return denom


def scattering_from_chipp(
    chipp: np.ndarray,
    E_meV: np.ndarray,
    temperature_K: float,
    scale: float = 1.0,
    form_factor_sq: float | np.ndarray = 1.0,
    polarization: float | np.ndarray = 1.0,
    background: float | np.ndarray = 0.0,
    include_bose: bool = True,
) -> np.ndarray:
    signal = np.asarray(chipp, dtype=float)
    if include_bose:
        signal = signal / bose_denominator(E_meV, temperature_K)
    return scale * form_factor_sq * polarization * signal + background
```

## src/chi4d/fitting.py

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
from scipy.optimize import least_squares

from .dataset import PointData4D


@dataclass
class ParameterSpec:
    name: str
    value: float
    min: float | None = None
    max: float | None = None
    vary: bool = True
    unit: str = ""
    description: str = ""


@dataclass
class FitResult:
    params: dict[str, float]
    success: bool
    message: str
    cost: float
    residuals: np.ndarray
    covariance: np.ndarray | None = None
    parameter_names: list[str] | None = None


def pack_parameters(specs: list[ParameterSpec]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[str], dict[str, float]]:
    x0 = []
    lo = []
    hi = []
    names = []
    fixed = {}
    for p in specs:
        if p.vary:
            x0.append(p.value)
            lo.append(-np.inf if p.min is None else p.min)
            hi.append(np.inf if p.max is None else p.max)
            names.append(p.name)
        else:
            fixed[p.name] = p.value
    return np.array(x0, dtype=float), (np.array(lo), np.array(hi)), names, fixed


def unpack_parameters(x: np.ndarray, names: list[str], fixed: dict[str, float]) -> dict[str, float]:
    params = dict(fixed)
    params.update({name: float(val) for name, val in zip(names, x)})
    return params


def fit_least_squares(
    data: PointData4D,
    model: Callable[[PointData4D, dict[str, float]], np.ndarray],
    parameter_specs: list[ParameterSpec],
    **kwargs: Any,
) -> FitResult:
    d = data.valid()
    x0, bounds, names, fixed = pack_parameters(parameter_specs)

    def residual_vector(x: np.ndarray) -> np.ndarray:
        params = unpack_parameters(x, names, fixed)
        y_model = model(d, params)
        return (d.intensity - y_model) / d.sigma

    opt = least_squares(residual_vector, x0, bounds=bounds, **kwargs)
    best = unpack_parameters(opt.x, names, fixed)
    residuals = residual_vector(opt.x)

    covariance = None
    if opt.jac is not None and opt.jac.size > 0 and len(opt.x) > 0:
        try:
            _, s, VT = np.linalg.svd(opt.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(opt.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            cov = (VT.T / s**2) @ VT
            dof = max(1, residuals.size - opt.x.size)
            covariance = cov * (2 * opt.cost / dof)
        except np.linalg.LinAlgError:
            covariance = None

    return FitResult(
        params=best,
        success=opt.success,
        message=opt.message,
        cost=float(opt.cost),
        residuals=residuals,
        covariance=covariance,
        parameter_names=names,
    )
```

