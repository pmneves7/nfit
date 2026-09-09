"""Lazy resolution for model-owned fit, plot, and report hooks.

The model registry stores callables supplied by several higher-level modules.
Resolving those providers through this neutral module keeps the registry at the
bottom of the dependency graph while preserving the historical lazy-import
behavior of its adapters.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDER_MODULES = {
    "fit_config": ".fit_config",
    "model_plots": ".model_plots",
    "report": ".report",
}


def resolve_model_hook(provider: str, name: str) -> Any:
    """Return a named hook from one of the registered provider modules.

    Lookup deliberately happens on every call. Besides matching the previous
    lazy imports, this keeps runtime extension and test monkeypatch behavior
    intact instead of retaining stale hook objects.
    """

    try:
        module_name = _PROVIDER_MODULES[provider]
    except KeyError as exc:
        raise ValueError(f"unknown model-hook provider {provider!r}") from exc
    module = import_module(module_name, package=__package__)
    return getattr(module, name)
