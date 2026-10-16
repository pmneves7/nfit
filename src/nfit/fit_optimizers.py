"""Numerical least-squares backends independent of fit models and GUI state.

Residual and Jacobian callables define the scientific problem. This module owns
optimizer execution, progress/cancellation, and local numerical diagnostics.
"""
from __future__ import annotations

import contextlib
import inspect
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from threading import RLock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _parallel

try:  # pragma: no cover - exercised only when SciPy is importable.
    from scipy.optimize import least_squares as _scipy_least_squares
except Exception:  # SciPy may be absent or have a broken compiled dependency.
    _scipy_least_squares = None

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover - declared dependency is unavailable
    _threadpool_limits = None

FloatArray = NDArray[np.float64]
ProgressCallback = Callable[[dict[str, Any]], None]


class FitCancellationRequested(RuntimeError):
    """Request that an active deterministic optimizer return its best point."""


def _resolve_parallel_workers(value: Any) -> int:
    """Resolve user-facing worker counts.

    ``-1`` means "auto": use a conservative number based on available CPUs.
    Thread pools are useful for expensive NumPy/SciPy-heavy evaluations, but
    oversubscribing can easily make small fits slower.
    """

    requested = int(value or 1)
    if requested == -1:
        available = _parallel.num_threads()
        return max(1, min(available - 1, 8))
    return _parallel.bounded_worker_count(requested)


def _covariance_from_jacobian(
    jacobian: FloatArray | None,
    *,
    chi2: float,
    dof: int,
    covariance_mode: str,
    diagnostics: dict[str, Any] | None = None,
) -> FloatArray | None:
    if jacobian is None or dof <= 0 or jacobian.size == 0:
        return None
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    eps = np.finfo(float).eps
    threshold = eps * max(jacobian.shape) * singular_values[0]
    keep = singular_values > threshold
    rank = int(np.count_nonzero(keep))
    if rank < jacobian.shape[1]:
        if diagnostics is not None:
            diagnostics.update(reason="rank_deficient", rank=rank)
        return None
    vt = vt[keep]
    singular_values = singular_values[keep]
    cov = (vt.T / (singular_values * singular_values)) @ vt
    if covariance_mode == "residual":
        cov *= chi2 / dof
    return cov


@dataclass
class _LeastSquaresResult:
    x: FloatArray
    jac: FloatArray | None
    success: bool
    message: str
    cost: float
    cancelled: bool = False


def _run_least_squares(
    residual_fn: Callable[[FloatArray], FloatArray],
    *,
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    kwargs: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    names: Sequence[str] = (),
    jac: Callable[[FloatArray], FloatArray] | None = None,
    scipy_least_squares: Callable | None = _scipy_least_squares,
) -> _LeastSquaresResult:
    optimizer_kwargs = {} if kwargs is None else dict(kwargs)
    x0 = _nudge_inside_bounds(x0, bounds)
    wrapped_residual_fn = residual_fn
    best_x: FloatArray | None = None
    best_cost = np.inf
    if progress_callback is not None:
        evaluation = 0
        fit_start = time.perf_counter()
        progress_lock = RLock()

        def wrapped_residual_fn(x: FloatArray) -> FloatArray:
            nonlocal best_cost, best_x, evaluation
            residual = residual_fn(x)
            cost = _least_squares_objective_cost(
                residual,
                loss=optimizer_kwargs.get("loss", "linear"),
                f_scale=optimizer_kwargs.get("f_scale", 1.0),
            )
            with progress_lock:
                evaluation += 1
                if np.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best_x = np.asarray(x, dtype=float).copy()
                if evaluation == 1 or evaluation % 10 == 0:
                    params = {
                        name: float(value)
                        for name, value in zip(names, x, strict=True)
                    }
                    elapsed = time.perf_counter() - fit_start
                    progress_callback(
                        {
                            "stage": "least_squares",
                            "iteration": evaluation,
                            "elapsed_seconds": elapsed,
                            "seconds_per_step": elapsed / max(evaluation, 1),
                            "parameters": {
                                name: float(params[name]) for name in names
                            },
                            "cost": cost,
                            "message": (
                                "least-squares residual evaluation "
                                f"{evaluation}"
                            ),
                        }
                    )
            return residual

    finite_difference_request = optimizer_kwargs.pop(
        "finite_difference_workers",
        1,
    )
    finite_difference_workers = min(
        len(names),
        _resolve_parallel_workers(finite_difference_request),
    )
    finite_difference_pool: ThreadPool | None = None
    if scipy_least_squares is not None:
        if jac is not None:
            optimizer_kwargs.setdefault("jac", jac)
        elif finite_difference_workers > 1:
            supports_workers = (
                "workers"
                in inspect.signature(scipy_least_squares).parameters
            )
            if supports_workers:
                finite_difference_pool = ThreadPool(
                    finite_difference_workers
                )
                blas_threads = max(
                    1,
                    _parallel.num_threads()
                    // finite_difference_workers,
                )

                def worker_map(function: Callable, values: Any) -> list[Any]:
                    limiter = (
                        _threadpool_limits(
                            limits=blas_threads,
                            user_api="blas",
                        )
                        if _threadpool_limits is not None
                        else contextlib.nullcontext()
                    )
                    with limiter:
                        return finite_difference_pool.map(function, values)

                optimizer_kwargs.setdefault("workers", worker_map)
            else:
                warnings.warn(
                    "parallel finite differences require SciPy >= 1.16; "
                    "using serial numerical derivatives",
                    RuntimeWarning,
                    stacklevel=2,
                )
        try:
            scipy_result = scipy_least_squares(
                wrapped_residual_fn,
                x0=x0,
                bounds=bounds,
                **optimizer_kwargs,
            )
        except FitCancellationRequested:
            return _cancelled_least_squares_result(best_x, best_cost)
        finally:
            if finite_difference_pool is not None:
                finite_difference_pool.close()
                finite_difference_pool.join()
        return _LeastSquaresResult(
            x=np.asarray(scipy_result.x, dtype=float),
            jac=np.asarray(scipy_result.jac, dtype=float),
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            cost=float(scipy_result.cost),
        )
    supported = {"max_nfev", "xtol", "ftol"}
    unsupported = set(optimizer_kwargs) - supported
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"NumPy least-squares fallback does not support optimizer kwargs: {names}")
    try:
        return _numpy_least_squares(
            wrapped_residual_fn,
            x0=x0,
            bounds=bounds,
            **optimizer_kwargs,
        )
    except FitCancellationRequested:
        return _cancelled_least_squares_result(best_x, best_cost)


def _least_squares_objective_cost(
    residual: FloatArray,
    *,
    loss: str | Callable[[FloatArray], FloatArray],
    f_scale: float,
) -> float:
    """Return SciPy's least-squares objective for one residual evaluation."""

    values = np.asarray(residual, dtype=float)
    if loss == "linear":
        return 0.5 * float(np.dot(values, values))
    scale = float(f_scale)
    z = (values / scale) ** 2
    if callable(loss):
        rho = np.asarray(loss(z), dtype=float)
        rho0 = rho[0]
    elif loss == "soft_l1":
        rho0 = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho0 = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho0 = np.log1p(z)
    elif loss == "arctan":
        rho0 = np.arctan(z)
    else:
        return 0.5 * float(np.dot(values, values))
    return 0.5 * scale**2 * float(np.sum(rho0))


def _cancelled_least_squares_result(
    best_x: FloatArray | None,
    best_cost: float,
) -> _LeastSquaresResult:
    if best_x is None:
        raise FitCancellationRequested(
            "least-squares terminated before any parameter set was evaluated"
        )
    return _LeastSquaresResult(
        x=best_x,
        jac=None,
        success=False,
        message=(
            "least-squares terminated by user; returning the lowest-objective "
            "parameter set evaluated so far"
        ),
        cost=float(best_cost),
        cancelled=True,
    )


def _nudge_inside_bounds(
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
) -> FloatArray:
    """Move initial values strictly inside finite bounds.

    Recent SciPy releases can satisfy ``ftol`` immediately when a start value
    sits exactly on a bound, returning without optimizing. A nudge of the
    finite-difference step scale keeps the optimizer productive without
    meaningfully changing the start point.
    """

    lower, upper = bounds
    if x0.size == 0:
        return x0
    out = np.asarray(x0, dtype=float).copy()
    step = np.maximum(1.0e-8, 1.0e-8 * np.abs(out))
    span = upper - lower
    finite_span = np.isfinite(span)
    step = np.where(finite_span, np.minimum(step, 0.25 * span), step)
    at_lower = np.isfinite(lower) & (out <= lower + step)
    at_upper = np.isfinite(upper) & (out >= upper - step)
    out = np.where(at_lower, lower + step, out)
    out = np.where(at_upper & ~at_lower, upper - step, out)
    return out


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
