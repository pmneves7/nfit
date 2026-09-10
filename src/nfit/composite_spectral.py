"""Spectral conventions and temperature resolution for composite histograms."""

from __future__ import annotations

import numpy as np

from .mdhisto import MDHistoData
from .metadata_dimensions import metadata_temperature_grid
from .spectral_channels import (
    SPECTRAL_CHANNEL_CONFIG_KEY,
    normalized_spectral_channel_config,
    with_paired_spectral_channels,
)


def common_source_value(datasets, names, *, point_temperature=False):
    """Resolve a physical value only when every source agrees (within 0.1%)."""
    values = []
    for dataset in datasets:
        value = None
        if point_temperature:
            value = getattr(dataset.data, "temperature", None)
        if value is None:
            for source in (dataset.parameters, dataset.parameters.get(SPECTRAL_CHANNEL_CONFIG_KEY, {}), dataset.metadata,
                           getattr(dataset.data, "metadata", {})):
                value = next((source[key] for key in names if source.get(key) is not None), None)
                if value is not None:
                    break
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if not array.size or not np.all(np.isfinite(array)) or np.any(array <= 0):
            return None
        values.extend((float(array.min()), float(array.max())))
    if not values or not np.allclose(values, values[0], rtol=1e-3, atol=1e-6):
        return None
    return float(np.mean(values))


def composite_spectral_config(config, datasets):
    """Return explicit settings with only unambiguous source energy defaults."""
    saved = config.get(SPECTRAL_CHANNEL_CONFIG_KEY)
    inherited = {}
    source_configs = [normalized_spectral_channel_config(d.parameters.get(SPECTRAL_CHANNEL_CONFIG_KEY)) for d in datasets]
    if source_configs:
        for key, value in source_configs[0].items():
            if all(c.get(key) == value for c in source_configs):
                inherited[key] = value
    result = normalized_spectral_channel_config({**inherited, **(saved or {})})
    if saved is None:
        result["enabled"] = False
    for key, names in (
        ("incident_energy_meV", ("incident_energy_meV", "incident_energy", "Ei", "ei")),
        ("final_energy_meV", ("final_energy_meV", "final_energy", "Ef", "ef")),
    ):
        if result.get(key) in (None, ""):
            result[key] = common_source_value(datasets, names)
    return result


def apply_composite_spectral_channels(data, config, datasets):
    """Convert after background subtraction, using a K axis or a common T."""
    if not isinstance(data, MDHistoData) or SPECTRAL_CHANNEL_CONFIG_KEY not in config:
        return data
    convention = composite_spectral_config(config, datasets)
    if not convention["enabled"]:
        return data
    temperatures = metadata_temperature_grid(data)
    if temperatures is None:
        temperatures = convention.get("temperature_K")
        if temperatures in (None, ""):
            temperatures = common_source_value(datasets, ("temperature",), point_temperature=True)
        if temperatures is None:
            raise ValueError(
                "Composite susceptibility conversion requires a common source temperature. "
                "For a temperature series, add a temperature metadata dimension in K; "
                "for one nominal temperature, set the composite temperature explicitly."
            )
        temperatures = float(temperatures)
        if not np.isfinite(temperatures) or temperatures <= 0:
            raise ValueError("Composite temperature must be positive and finite (K).")
    result = with_paired_spectral_channels(data, convention, temperature_K=temperatures)
    if result.metadata.get("spectral_channel_error"):
        raise ValueError(result.metadata["spectral_channel_error"])
    return result
