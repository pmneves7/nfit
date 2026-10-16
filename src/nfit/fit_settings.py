"""Translate saved project fitting settings into public optimizer configuration."""

from __future__ import annotations

from typing import Any

from .fitting import OptimizationConfig

OPTIMIZER_KWARG_NAMES = ("max_nfev", "xtol", "ftol", "gtol", "loss", "f_scale")


def optimizer_kwargs(optimizer_config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve saved local and optional global optimizer settings."""
    config = optimizer_config if isinstance(optimizer_config, dict) else {}
    kwargs: dict[str, Any] = {"finite_difference_workers": -1}
    for name in OPTIMIZER_KWARG_NAMES:
        value = config.get(name)
        if value in (None, ""):
            continue
        if name == "max_nfev":
            kwargs[name] = int(value)
        elif name == "loss":
            kwargs[name] = str(value)
        else:
            kwargs[name] = float(value)
    init_config = config.get("initialization")
    if isinstance(init_config, dict) and init_config.get("enabled", False):
        legacy_workers = init_config.get("workers", 1)
        kwargs["initialization"] = {
            key: value for key, value in init_config.items() if key not in {"enabled", "workers"}
        }
        kwargs["initialization"].setdefault("method", "differential_evolution")
        try:
            legacy_parallel = int(legacy_workers or 1) != 1
        except (TypeError, ValueError):
            legacy_parallel = False
        deferred = str(kwargs["initialization"].get("updating", "")).lower() == "deferred"
        if legacy_parallel or deferred:
            kwargs["initialization"]["workers"] = -1
        if legacy_parallel:
            kwargs["initialization"].setdefault("updating", "deferred")
    return kwargs


def covariance_mode(optimizer_config: dict[str, Any] | None) -> str:
    """Read the saved covariance convention, preserving the absolute default."""
    if not isinstance(optimizer_config, dict):
        return "absolute"
    mode = str(optimizer_config.get("covariance_mode", "absolute"))
    if mode not in {"absolute", "residual"}:
        raise ValueError("covariance_mode must be 'absolute' or 'residual'")
    return mode


def optimization_config_from_settings(
    settings: dict[str, Any] | None,
    *,
    local_only: bool = False,
) -> OptimizationConfig:
    """Build the same optimizer configuration for the GUI and exported scripts.

    ``local_only`` omits global initialization when refining a chosen candidate.
    """
    kwargs = optimizer_kwargs(settings)
    if local_only:
        kwargs.pop("initialization", None)
    return OptimizationConfig(kwargs=kwargs, covariance_mode=covariance_mode(settings))
