"""Calculation lifecycle shared by named dataset and composite binnings."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


def binning_is_enabled(config: Mapping[str, Any]) -> bool:
    """Whether a named binning may calculate or persist numerical results.

    Missing values preserve the behavior of projects saved before this control
    existed. This flag is independent of the legacy ``enabled`` rebin setting.
    """

    return bool(config.get("binning_enabled", True))


def set_binning_enabled(config: MutableMapping[str, Any], enabled: bool) -> None:
    """Set calculation availability without invalidating an existing result."""

    config["binning_enabled"] = bool(enabled)


def disabled_binning_status(config: Mapping[str, Any]) -> str | None:
    """Explain a suspended binning without resolving its grid or loading data."""

    if binning_is_enabled(config):
        return None
    if config.get("stale", False):
        return "Binning disabled; stale results are discarded. Enable it to calculate a new result."
    return "Binning disabled; current in-memory results are retained. Calculation and cache saving are off."
