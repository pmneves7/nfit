"""Resolve temperatures shared by plotting, fitting, and analysis."""

from __future__ import annotations

import numpy as np

from .mdhisto import MDHistoData


def _validated_temperature(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    temperature = np.asarray(value, dtype=float)
    if not temperature.size or np.any(~np.isfinite(temperature)) or np.any(temperature < 0):
        raise ValueError("Temperature must be finite and nonnegative (K).")
    return float(temperature) if temperature.ndim == 0 else temperature


def data_temperature(data):
    """Return imported temperature in K, retaining an explicit temperature axis."""
    if data is None:
        return None
    values = _temperature_coordinate(data)
    if values is not None:
        return _validated_temperature(values)
    values = getattr(data, "temperature", None)
    if values is not None:
        return _validated_temperature(values)
    metadata = getattr(data, "metadata", {})
    value = metadata.get("temperature", metadata.get("temperature_K"))
    if value is None:
        value = metadata.get("spectral_observable", {}).get("temperature_K")
    return _validated_temperature(value)


def _temperature_coordinate(data):
    if isinstance(data, MDHistoData):
        from .metadata_dimensions import metadata_temperature_grid

        values = metadata_temperature_grid(data)
        if values is not None:
            return values
    values = getattr(data, "temperature", None)
    return values if isinstance(values, np.ndarray) else None


def dataset_temperature(dataset, data=None):
    """Resolve a dataset temperature, with manual overrides for scalar imports.

    A temperature coordinate or per-point array defines an experimental axis
    and takes precedence over a scalar override. Otherwise the manual dataset
    value precedes imported data and dataset metadata.
    """
    data = dataset.data if data is None else data
    coordinate = _temperature_coordinate(data)
    if coordinate is not None:
        return _validated_temperature(coordinate)
    override = _validated_temperature(dataset.parameters.get("temperature"))
    if override is not None:
        return override
    imported = data_temperature(data)
    if imported is not None:
        return imported
    return _validated_temperature(
        dataset.metadata.get("temperature", dataset.metadata.get("temperature_K"))
    )
