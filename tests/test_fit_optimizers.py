"""Numerical backends remain usable without fit models or GUI orchestration."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nfit import fit_optimizers, fitting


def test_optimizer_helpers_have_one_authoritative_definition():
    for name in (
        "FitCancellationRequested", "_LeastSquaresResult",
        "_cancelled_least_squares_result", "_covariance_from_jacobian",
        "_finite_difference_jacobian", "_least_squares_objective_cost",
        "_nudge_inside_bounds", "_numpy_least_squares",
        "_resolve_parallel_workers", "_solve_damped_step",
    ):
        assert getattr(fitting, name) is getattr(fit_optimizers, name)


def test_optimizer_backend_does_not_depend_on_scientific_or_gui_coordinators():
    tree = ast.parse(Path(fit_optimizers.__file__).read_text())
    relative_imports = {
        node.module or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
        for alias in node.names
    }
    assert relative_imports == {"_parallel"}
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(name.startswith(("PySide", "PyQt", "qt_", "project_", "nfit.fitting")) for name in imports)


def test_numpy_and_scipy_backends_preserve_bounded_solution_and_covariance():
    scipy_solver = pytest.importorskip("scipy.optimize").least_squares
    coordinate = np.linspace(-2, 2, 9)
    design = np.column_stack((coordinate, np.ones(coordinate.size)))
    observed = design @ np.array([1.7, -0.4])
    observed += np.array([0.01, -0.01, 0, 0.01, -0.02, 0, 0.01, -0.01, 0.01])
    def residual_fn(x):
        return (observed - design @ x) / 0.1

    results = []
    for backend in (scipy_solver, None):
        result = fit_optimizers._run_least_squares(
            residual_fn,
            x0=np.array([0., 0.]),
            bounds=(np.array([0., -1.]), np.array([2., 1.])),
            names=("slope", "offset"),
            scipy_least_squares=backend,
        )
        assert result.success
        covariance = fit_optimizers._covariance_from_jacobian(
            result.jac, chi2=2 * result.cost, dof=7, covariance_mode="absolute",
        )
        np.testing.assert_allclose(covariance, np.linalg.inv(design.T @ design / 0.1**2), rtol=1e-7)
        results.append(result)
    np.testing.assert_allclose(results[0].x, results[1].x, rtol=1e-7, atol=1e-8)
    assert results[0].cost == pytest.approx(results[1].cost, rel=1e-10)


@pytest.mark.parametrize("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan"])
def test_robust_objective_matches_scipy_at_the_same_residual(loss):
    scipy_solver = pytest.importorskip("scipy.optimize").least_squares
    residual = np.array([-10., -1., 0., 0.5, 4.])
    expected = scipy_solver(lambda x: residual, [0.], loss=loss, f_scale=2.).cost
    assert fit_optimizers._least_squares_objective_cost(residual, loss=loss, f_scale=2.) == pytest.approx(expected)


def test_callable_robust_objective_matches_scipy():
    scipy_solver = pytest.importorskip("scipy.optimize").least_squares

    def loss(z):
        return np.array([np.log1p(z), 1 / (1 + z), -1 / (1 + z)**2])

    residual = np.array([0.5, -4., 12.])
    expected = scipy_solver(lambda x: residual, [0.], loss=loss, f_scale=3.).cost
    assert fit_optimizers._least_squares_objective_cost(residual, loss=loss, f_scale=3.) == pytest.approx(expected)


def test_cancelled_robust_fit_returns_best_transformed_objective():
    def backend(fun, **kwargs):
        for point in (0., 1.):
            fun(np.array([point]))
        raise fitting.FitCancellationRequested("stop")

    events = []
    result = fit_optimizers._run_least_squares(
        lambda x: np.array([0., 10.]) if x[0] == 0 else np.array([4., 4.]),
        x0=np.array([0.]), bounds=(np.array([-1.]), np.array([2.])),
        kwargs={"loss": "cauchy"}, names=("level",),
        progress_callback=events.append, scipy_least_squares=backend,
    )
    assert result.cancelled and not result.success
    assert result.jac is None
    np.testing.assert_array_equal(result.x, [0.])
    assert result.cost == pytest.approx(0.5 * np.log(101))
    assert events[0]["parameters"] == {"level": 0.}


def test_numpy_cancellation_uses_the_shared_exception_and_retains_evaluated_point():
    def cancel(event):
        raise fitting.FitCancellationRequested("stop")

    result = fit_optimizers._run_least_squares(
        lambda x: x - 2., x0=np.array([0.]),
        bounds=(np.array([-1.]), np.array([3.])), names=("level",),
        progress_callback=cancel, scipy_least_squares=None,
    )
    assert result.cancelled
    np.testing.assert_array_equal(result.x, [0.])
    assert result.cost == 2.


def test_cancellation_before_any_successful_evaluation_propagates():
    def residual(x):
        raise fitting.FitCancellationRequested("stop")

    with pytest.raises(fitting.FitCancellationRequested, match="before any parameter set"):
        fit_optimizers._run_least_squares(
            residual, x0=np.array([0.]),
            bounds=(np.array([-1.]), np.array([3.])), names=("level",),
            progress_callback=lambda event: None, scipy_least_squares=None,
        )


def test_legacy_solver_hook_and_analytic_jacobian_are_forwarded(monkeypatch):
    def jacobian(x):
        return np.ones((2, 1))

    options = {"finite_difference_workers": 8, "xtol": 1e-9}

    def backend(fun, *, x0, bounds, jac, **kwargs):
        assert jac is jacobian
        assert kwargs == {"xtol": 1e-9}
        return SimpleNamespace(x=x0, jac=jac(x0), success=True, message="hook", cost=0.)

    monkeypatch.setattr(fitting, "_scipy_least_squares", backend)
    result = fitting._run_least_squares(
        lambda x: np.repeat(x, 2), x0=np.array([0.]),
        bounds=(np.array([-1.]), np.array([1.])), names=("level",),
        fixed={"other": 3.}, jac=jacobian, kwargs=options,
    )
    assert result.message == "hook"
    assert options == {"finite_difference_workers": 8, "xtol": 1e-9}


def test_legacy_solver_hook_can_select_numpy_and_reject_unsupported_options(monkeypatch):
    monkeypatch.setattr(fitting, "_scipy_least_squares", None)
    with pytest.raises(ValueError, match="fallback does not support optimizer kwargs: loss"):
        fitting._run_least_squares(
            lambda x: x, x0=np.array([0.]),
            bounds=(np.array([-1.]), np.array([1.])), kwargs={"loss": "huber"},
        )


def test_parallel_derivative_pool_is_released_after_cancellation(monkeypatch):
    lifecycle = []

    class Pool:
        def __init__(self, count):
            lifecycle.append(("create", count))

        def map(self, function, values):
            return list(map(function, values))

        def close(self):
            lifecycle.append("close")

        def join(self):
            lifecycle.append("join")

    def backend(fun, *, x0, bounds, workers=None):
        assert callable(workers)
        workers(fun, [x0])
        raise fitting.FitCancellationRequested("stop")

    monkeypatch.setattr(fit_optimizers, "ThreadPool", Pool)
    monkeypatch.setattr(fit_optimizers._parallel, "num_threads", lambda: 4)
    monkeypatch.setattr(fit_optimizers, "_threadpool_limits", None)
    result = fit_optimizers._run_least_squares(
        lambda x: x - 1., x0=np.array([0., 0.]),
        bounds=(np.array([-1., -1.]), np.array([2., 2.])),
        names=("a", "b"), kwargs={"finite_difference_workers": 2},
        progress_callback=lambda event: None, scipy_least_squares=backend,
    )
    assert result.cancelled
    assert lifecycle == [("create", 2), "close", "join"]


def test_old_scipy_backend_warns_and_keeps_serial_derivatives():
    def backend(fun, *, x0, bounds):
        return SimpleNamespace(x=x0, jac=np.ones((2, 2)), success=True, message="serial", cost=0.)

    with pytest.warns(RuntimeWarning, match="using serial numerical derivatives"):
        result = fit_optimizers._run_least_squares(
            lambda x: x, x0=np.array([0., 0.]),
            bounds=(np.array([-1., -1.]), np.array([2., 2.])),
            names=("a", "b"), kwargs={"finite_difference_workers": 2},
            scipy_least_squares=backend,
        )
    assert result.success
