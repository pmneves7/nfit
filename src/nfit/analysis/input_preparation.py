"""Scientific input and parameter routing for registered analyses."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..form_factors import form_factor_sq
from ..measurement_dependencies import SourceReplayRequired
from ..sample_temperature import data_temperature
from .coordinates import q_modulus_for_spectral
from .corrections import SpectralConvention


def analysis_histogram(item):
    """Supply group lattice context without replacing an imported orientation."""
    data = item.data
    metadata = dict(data.metadata)
    if not isinstance(metadata.get("lattice_parameters"), dict) and item.context.lattice_parameters:
        metadata["lattice_parameters"] = dict(item.context.lattice_parameters)
        data = data.with_updates(metadata=metadata)
    return data


def analysis_temperature(item, parameters):
    source = parameters["temperature_source"]
    if source not in {"dataset", "fixed"}:
        raise ValueError("temperature_source must be 'dataset' or 'fixed'")
    temperature = parameters["temperature_K"] if source == "fixed" else item.context.temperature_K
    if temperature is None and source == "dataset":
        temperature = data_temperature(item.data)
    if temperature is None:
        raise ValueError("Set a dataset or composite temperature, or enter a fixed temperature (K).")
    values = np.asarray(temperature, dtype=float)
    if not values.size or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Temperature must be finite and nonnegative (K).")
    if source == "fixed" and values.ndim == 0:
        coordinate = data_temperature(item.data)
        if not isinstance(coordinate, np.ndarray) and isinstance(item.context.temperature_K, np.ndarray):
            coordinate = item.context.temperature_K
        if isinstance(coordinate, np.ndarray) and not np.allclose(coordinate, values, rtol=1e-10, atol=0.0):
            raise ValueError(
                "Manual temperature differs from the temperature coordinate; "
                "choose dataset/composite temperature to retain the experimental axis."
            )
    return float(values) if values.ndim == 0 else values


def analysis_convention(parameters):
    convention = SpectralConvention.from_dict(dict(parameters["spectral_convention"]))
    g_factor = parameters.get("g_factor")
    if g_factor is not None:
        if not np.isfinite(g_factor) or g_factor <= 0:
            raise ValueError("g_factor must be finite and positive")
        convention = replace(convention, g_factor=float(g_factor))
    return convention


def analysis_form_factor(data, context, parameters):
    ion = str(parameters.get("form_factor_ion") or "").strip()
    coefficients = parameters.get("custom_form_factor")
    if not ion and coefficients is None:
        return parameters.get("form_factor_sq", 1.0)
    return form_factor_sq(q_modulus_for_spectral(data, context), ion=ion or None, coefficients=coefficients)


def analysis_input_channel(data, name):
    """Select a measured channel with its own uncertainties, never another's."""
    if name == "signal":
        return data
    if name not in data.auxiliary_channels:
        raise ValueError(f"Unknown input channel {name!r}")
    channel = data.auxiliary_channels[name]
    if channel.errors is None:
        raise ValueError(f"Input channel {name!r} has no measurement uncertainties")
    if (
        data.source_dependencies is not None
        or data.counting_dependencies is not None
        or data.metadata.get("measurement_contract", {}).get("dependence") == "shared_sources"
    ):
        raise SourceReplayRequired("Auxiliary-channel integration requires its own source dependencies; use a converted primary dataset.")
    selected = data.with_updates(signal=channel.values, errors=channel.errors)
    metadata = {**selected.metadata, "signal_unit": channel.unit, "signal_quantity_type": channel.quantity_type}
    metadata.pop("normalization_denominator", None)
    return selected.with_updates(metadata=metadata)
