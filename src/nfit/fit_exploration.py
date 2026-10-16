"""Tempered objective exploration kept separate from posterior inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fit_sampling import (
    InvalidInitialEnsemble,
    SamplerConfig,
    SamplingCancelled,
    SamplingResult,
    best_sampling_sample,
    sample_parameter_vectors,
)
from .fitting import (
    FitProblem,
    OptimizationConfig,
    ProgressCallback,
    _initial_walker_positions,
    _resolve_parallel_workers,
    evaluate_problem_objective,
    pack_parameters,
    unpack_parameters,
)


@dataclass(frozen=True)
class ExplorationConfig(SamplerConfig):
    """Search settings; temperature is dimensionless and must be at least one.

    ``bounds`` starts throughout finite parameter bounds; ``local`` starts near
    the supplied parameter values. The target is ``exp(-cost / temperature)``
    within the existing bounds. These samples never estimate posterior errors.
    """

    temperature: float = 10.0
    initialization: str = "bounds"


def explore_problem_parameters(
    problem: FitProblem,
    config: ExplorationConfig | None = None,
    *,
    initial_params: dict[str, float] | None = None,
    initial_walkers: np.ndarray | None = None,
    initial_random_state: dict[str, Any] | None = None,
    optimization_config: OptimizationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SamplingResult:
    """Explore remote fit basins with a tempered original optimizer objective.

    For ordinary least squares the target is proportional to
    ``exp(-chi2 / (2*T))``. Robust losses use the optimizer's actual robust cost.
    Bounds remain hard constraints. Temperature does not rescale stored data or
    official posterior uncertainty; run :func:`sample_problem_parameters` after
    local refinement for untempered inference.
    """
    settings = ExplorationConfig() if config is None else config
    if settings.method != "emcee":
        raise ValueError("only method='emcee' is implemented")
    temperature = float(settings.temperature)
    if not np.isfinite(temperature) or temperature < 1:
        raise ValueError("exploration temperature must be finite and at least one")
    if settings.initialization not in {"local", "bounds"}:
        raise ValueError("exploration initialization must be 'local' or 'bounds'")
    center, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    if not names:
        raise ValueError("emcee exploration requires at least one variable parameter")
    lower, upper = bounds
    if initial_params is not None:
        center = np.asarray(
            [initial_params.get(name, center[i]) for i, name in enumerate(names)], float
        )
    if np.any(~np.isfinite(center)) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("initial exploration parameters must be finite and inside bounds")
    n_walkers = (
        max(32, 2 * len(names) + 2) if settings.n_walkers is None else int(settings.n_walkers)
    )
    if n_walkers < 2 * len(names):
        raise ValueError("emcee requires at least 2 * n_parameters walkers")

    def log_probability(vector):
        if np.any(~np.isfinite(vector)) or np.any(vector < lower) or np.any(vector > upper):
            return -np.inf
        try:
            cost = evaluate_problem_objective(
                problem, unpack_parameters(vector, names, fixed), config=optimization_config
            )
        except (ValueError, ArithmeticError, RuntimeError):
            return -np.inf
        return -cost / temperature if np.isfinite(cost) else -np.inf

    rng = np.random.default_rng(settings.random_seed)
    if initial_walkers is not None:
        walkers = np.asarray(initial_walkers, float)
        if walkers.shape != (n_walkers, len(names)):
            raise ValueError(f"initial_walkers must have shape ({n_walkers}, {len(names)})")
        if np.any(~np.isfinite(walkers)) or np.any(walkers < lower) or np.any(walkers > upper):
            raise ValueError("initial_walkers must be finite and inside bounds")
    elif settings.initialization == "local":
        walkers = _initial_walker_positions(center, bounds, n_walkers, rng)
    else:
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError(
                "bounds initialization requires finite bounds for every variable; use local initialization or specify finite bounds"
            )
        if np.any(lower >= upper):
            raise ValueError("exploration requires strictly ordered bounds for variable parameters")
        walkers = rng.uniform(lower, upper, size=(n_walkers, len(names)))
        for index in range(n_walkers):
            for _attempt in range(100):
                if np.isfinite(log_probability(walkers[index])):
                    break
                walkers[index] = rng.uniform(lower, upper)
            else:
                raise ValueError(
                    "could not initialize valid exploration walkers within parameter bounds"
                )
    backend_kwargs = dict(settings.kwargs)
    worker_request = backend_kwargs.pop("workers", backend_kwargs.pop("parallel_workers", 1))
    workers = _resolve_parallel_workers(worker_request)
    options = {} if optimization_config is None else optimization_config.kwargs
    loss = options.get("loss", "linear")
    objective_metadata = {
        "initialization": settings.initialization,
        "objective": "least_squares_cost",
        "objective_loss": loss if isinstance(loss, str) else "callable",
        "objective_f_scale": float(options.get("f_scale", 1.0)),
        "require_positive_sigma": True
        if optimization_config is None
        else optimization_config.require_positive_sigma,
    }
    try:
        result = sample_parameter_vectors(
            log_probability,
            walkers,
            list(names),
            settings,
            workers=workers,
            worker_request=worker_request,
            purpose="exploration",
            temperature=temperature,
            initial_random_state=initial_random_state,
            progress_callback=progress_callback,
        )
    except InvalidInitialEnsemble as exc:
        try:
            evaluate_problem_objective(
                problem, unpack_parameters(walkers[0], names, fixed), config=optimization_config
            )
        except Exception as cause:
            raise ValueError(f"Cannot initialize objective exploration: {cause}") from cause
        raise exc
    except SamplingCancelled as exc:
        exc.result.metadata.update(objective_metadata)
        raise
    result.metadata.update(objective_metadata)
    return result


def best_exploration_candidate(
    problem: FitProblem,
    result: SamplingResult,
    *,
    baseline_params: dict[str, float],
    optimization_config: OptimizationConfig | None = None,
) -> dict[str, Any] | None:
    """Reevaluate the best search point and baseline using the original objective.

    A tempered score or rounded legacy score is never compared to a cold fit's
    score. Improvements smaller than floating-point comparison tolerance do not
    produce a promotion candidate. The caller must additionally validate target
    provenance before applying a stored candidate to project state.
    """
    if result.metadata.get("purpose") != "exploration":
        raise ValueError("candidate selection requires exploration samples")
    names = [spec.name for spec in problem.parameter_specs if spec.vary]
    if result.variable_names != names:
        raise ValueError("exploration parameter names no longer match the fit problem")
    if optimization_config is None:
        loss = result.metadata.get("objective_loss", "linear")
        if loss == "callable":
            raise ValueError("callable exploration loss requires its original optimization_config")
        optimization_config = OptimizationConfig(
            require_positive_sigma=bool(result.metadata.get("require_positive_sigma", True)),
            kwargs={"loss": loss, "f_scale": result.metadata.get("objective_f_scale", 1.0)},
        )
    else:
        loss = optimization_config.kwargs.get("loss", "linear")
        expected_loss = loss if isinstance(loss, str) else "callable"
        if result.metadata.get(
            "objective_loss", expected_loss
        ) != expected_loss or result.metadata.get(
            "objective_f_scale", optimization_config.kwargs.get("f_scale", 1.0)
        ) != optimization_config.kwargs.get("f_scale", 1.0):
            raise ValueError("exploration objective settings no longer match the fit problem")
    best = best_sampling_sample(result, include_burn_in=True)
    if best is None:
        return None
    params, _tempered_score, location = best
    baseline_params = {
        spec.name: float(baseline_params.get(spec.name, spec.value))
        if spec.vary
        else float(spec.value)
        for spec in problem.parameter_specs
    }
    baseline = evaluate_problem_objective(problem, baseline_params, config=optimization_config)
    merged = dict(baseline_params)
    merged.update(params)
    candidate = evaluate_problem_objective(problem, merged, config=optimization_config)
    tolerance = 1.0e-10 * max(1.0, abs(baseline), abs(candidate))
    if candidate >= baseline - tolerance:
        return None
    return {
        "parameters": problem.resolve_parameters(merged),
        "objective": candidate,
        "baseline_objective": baseline,
        "temperature": result.metadata.get("temperature"),
        **location,
    }
