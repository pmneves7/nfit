from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dataset import PointData4D

try:  # pragma: no cover - exercised only when SciPy is importable.
    from scipy.optimize import least_squares as _scipy_least_squares
except Exception:  # SciPy may be absent or have a broken compiled dependency.
    _scipy_least_squares = None


FloatArray = NDArray[np.float64]
ModelFunction = Callable[[PointData4D, dict[str, float]], FloatArray]


@dataclass(frozen=True)
class ParameterSpec:
    """Named scalar fitting parameter.

    Bounds use the same units as the parameter value. Fixed parameters are
    included in the parameter dictionary passed to model functions but excluded
    from the optimizer vector.
    """

    name: str
    value: float
    min: float | None = None
    max: float | None = None
    vary: bool = True
    unit: str = ""
    description: str = ""


@dataclass
class FitResult:
    """Result returned by :func:`fit_least_squares`."""

    params: dict[str, float]
    success: bool
    message: str
    cost: float
    chi2: float
    reduced_chi2: float
    residuals: FloatArray
    model_values: FloatArray
    covariance: FloatArray | None
    stderr: dict[str, float] | None
    variable_names: list[str]
    fixed_params: dict[str, float]


def pack_parameters(
    specs: Sequence[ParameterSpec],
) -> tuple[FloatArray, tuple[FloatArray, FloatArray], list[str], dict[str, float]]:
    """Pack variable parameters for SciPy least squares."""

    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    fixed: dict[str, float] = {}

    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate parameter name {spec.name!r}")
        seen.add(spec.name)
        lo = -np.inf if spec.min is None else float(spec.min)
        hi = np.inf if spec.max is None else float(spec.max)
        if lo > hi:
            raise ValueError(f"lower bound exceeds upper bound for {spec.name!r}")
        if not (lo <= spec.value <= hi):
            raise ValueError(f"initial value for {spec.name!r} is outside bounds")
        if spec.vary:
            names.append(spec.name)
            x0.append(float(spec.value))
            lower.append(lo)
            upper.append(hi)
        else:
            fixed[spec.name] = float(spec.value)

    return (
        np.asarray(x0, dtype=float),
        (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        names,
        fixed,
    )


def unpack_parameters(x: Sequence[float], names: Sequence[str], fixed: dict[str, float]) -> dict[str, float]:
    """Combine optimizer vector values with fixed parameters."""

    params = dict(fixed)
    params.update({name: float(value) for name, value in zip(names, x)})
    return params


def fit_least_squares(
    data: PointData4D,
    model: ModelFunction,
    parameter_specs: Sequence[ParameterSpec],
    *,
    require_positive_sigma: bool = True,
    kwargs: dict | None = None,
) -> FitResult:
    """Fit a measured-intensity model to :class:`PointData4D`.

    The model receives a valid-data subset and a dictionary of parameter values.
    It must return measured intensity, not chi''. Weighted residuals are
    ``(I_obs - I_model) / sigma``.
    """

    valid_data = data.valid(require_positive_sigma=require_positive_sigma)
    if valid_data.size == 0:
        raise ValueError("no valid data points remain after masking")

    x0, bounds, names, fixed = pack_parameters(parameter_specs)
    model_kwargs = {} if kwargs is None else dict(kwargs)

    def residual_fn(x: FloatArray) -> FloatArray:
        params = unpack_parameters(x, names, fixed)
        values = np.asarray(model(valid_data, params, **model_kwargs), dtype=float)
        if values.shape != valid_data.intensity.shape:
            raise ValueError(
                f"model returned shape {values.shape}, expected {valid_data.intensity.shape}"
            )
        return (valid_data.intensity - values) / valid_data.sigma

    if len(names) == 0:
        residuals = residual_fn(np.asarray([], dtype=float))
        params = dict(fixed)
        model_values = valid_data.intensity - residuals * valid_data.sigma
        chi2 = float(np.sum(residuals * residuals))
        return FitResult(
            params=params,
            success=True,
            message="no variable parameters",
            cost=0.5 * chi2,
            chi2=chi2,
            reduced_chi2=np.nan,
            residuals=residuals,
            model_values=model_values,
            covariance=None,
            stderr=None,
            variable_names=[],
            fixed_params=fixed,
        )

    result = _run_least_squares(residual_fn, x0=x0, bounds=bounds)
    params = unpack_parameters(result.x, names, fixed)
    residuals = residual_fn(result.x)
    model_values = valid_data.intensity - residuals * valid_data.sigma
    chi2 = float(np.sum(residuals * residuals))
    dof = valid_data.size - len(names)
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan
    covariance = _covariance_from_jacobian(result.jac, chi2=chi2, dof=dof)
    stderr = None
    if covariance is not None:
        stderr = {
            name: float(np.sqrt(covariance[i, i]))
            for i, name in enumerate(names)
            if covariance[i, i] >= 0
        }

    return FitResult(
        params=params,
        success=bool(result.success),
        message=str(result.message),
        cost=float(result.cost),
        chi2=chi2,
        reduced_chi2=float(reduced_chi2),
        residuals=residuals,
        model_values=model_values,
        covariance=covariance,
        stderr=stderr,
        variable_names=list(names),
        fixed_params=fixed,
    )


def _covariance_from_jacobian(jacobian: FloatArray, *, chi2: float, dof: int) -> FloatArray | None:
    if dof <= 0 or jacobian.size == 0:
        return None
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    eps = np.finfo(float).eps
    threshold = eps * max(jacobian.shape) * singular_values[0]
    keep = singular_values > threshold
    if not np.any(keep):
        return None
    vt = vt[keep]
    singular_values = singular_values[keep]
    cov = (vt.T / (singular_values * singular_values)) @ vt
    return cov * (chi2 / dof)


@dataclass
class _LeastSquaresResult:
    x: FloatArray
    jac: FloatArray
    success: bool
    message: str
    cost: float


def _run_least_squares(
    residual_fn: Callable[[FloatArray], FloatArray],
    *,
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
) -> _LeastSquaresResult:
    if _scipy_least_squares is not None:
        scipy_result = _scipy_least_squares(residual_fn, x0=x0, bounds=bounds)
        return _LeastSquaresResult(
            x=np.asarray(scipy_result.x, dtype=float),
            jac=np.asarray(scipy_result.jac, dtype=float),
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            cost=float(scipy_result.cost),
        )
    return _numpy_least_squares(residual_fn, x0=x0, bounds=bounds)


def _numpy_least_squares(
    residual_fn: Callable[[FloatArray], FloatArray],
    *,
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    max_nfev: int = 400,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
) -> _LeastSquaresResult:
    """Small Levenberg-Marquardt fallback for environments without SciPy.

    This is intentionally modest. It is good enough for the package's synthetic
    validation tests, while SciPy remains the preferred optimizer when present.
    """

    lower, upper = bounds
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    residual = residual_fn(x)
    cost = 0.5 * float(np.dot(residual, residual))
    damping = 1e-3
    n_eval = 1

    success = False
    message = "maximum function evaluations reached in NumPy fallback"
    jac = _finite_difference_jacobian(residual_fn, x, residual, bounds)

    while n_eval < max_nfev:
        jac = _finite_difference_jacobian(residual_fn, x, residual, bounds)
        gradient = jac.T @ residual
        hessian = jac.T @ jac
        step = _solve_damped_step(hessian, gradient, damping)
        if np.linalg.norm(step) <= xtol * (xtol + np.linalg.norm(x)):
            success = True
            message = "xtol reached in NumPy fallback"
            break

        trial = np.clip(x - step, lower, upper)
        if np.allclose(trial, x, rtol=0.0, atol=xtol):
            success = True
            message = "bounded step became smaller than xtol in NumPy fallback"
            break

        trial_residual = residual_fn(trial)
        n_eval += 1
        trial_cost = 0.5 * float(np.dot(trial_residual, trial_residual))
        if trial_cost < cost:
            if abs(cost - trial_cost) <= ftol * max(1.0, cost):
                x = trial
                residual = trial_residual
                cost = trial_cost
                success = True
                message = "ftol reached in NumPy fallback"
                break
            x = trial
            residual = trial_residual
            cost = trial_cost
            damping = max(damping / 5.0, 1e-12)
        else:
            damping = min(damping * 10.0, 1e12)

    jac = _finite_difference_jacobian(residual_fn, x, residual, bounds)
    return _LeastSquaresResult(x=x, jac=jac, success=success, message=message, cost=cost)


def _finite_difference_jacobian(
    residual_fn: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    residual: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
) -> FloatArray:
    lower, upper = bounds
    jac = np.empty((residual.size, x.size), dtype=float)
    for i, value in enumerate(x):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(value))
        plus = x.copy()
        minus = x.copy()
        plus[i] = min(upper[i], value + step)
        minus[i] = max(lower[i], value - step)
        if plus[i] == minus[i]:
            jac[:, i] = 0.0
        elif plus[i] == value:
            jac[:, i] = (residual - residual_fn(minus)) / (value - minus[i])
        elif minus[i] == value:
            jac[:, i] = (residual_fn(plus) - residual) / (plus[i] - value)
        else:
            jac[:, i] = (residual_fn(plus) - residual_fn(minus)) / (plus[i] - minus[i])
    return jac


def _solve_damped_step(hessian: FloatArray, gradient: FloatArray, damping: float) -> FloatArray:
    matrix = hessian + damping * np.eye(hessian.shape[0])
    try:
        return np.linalg.solve(matrix, gradient)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ gradient
