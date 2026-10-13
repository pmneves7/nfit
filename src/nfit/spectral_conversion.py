"""Physical spectral factors and lineage-preserving histogram conversions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .cross_section import (
    MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ,
    bose_denominator,
    magnetic_moment_factor,
)


def spectral_conversion_factor(
    energy, temperature_K, *, source_representation, target_representation,
    source_scale=1.0, form_factor_sq=1.0, polarization=1.0, kf_ki=1.0,
    form_factor_state="included", polarization_state="included",
    bose_state="included", moment_unit="mu_B_squared", g_factor=None,
    absolute=True,
):
    """Return the one multiplier from a declared source to a physical target.

    Cross-section factors are in barn/(sr meV); ``source_scale`` converts the
    source ordinate to those units. Susceptibility and structure-factor inputs
    keep their declared spin or magnetic-moment units. Invalid physical cells
    return NaN rather than preventing conversion of accessible cells. A forward
    Bose conversion at exactly zero energy requires a slope, which ordinates
    alone do not supply, so that cell is unavailable.
    """
    source = source_representation
    target = target_representation
    if source not in {"measured_intensity", "cross_section", "s_qw", "chi_double_prime"}:
        raise ValueError(f"unknown source spectral representation {source!r}")
    if target not in {"cross_section", "chi_double_prime"}:
        raise ValueError(f"unknown target spectral representation {target!r}")
    for state in (form_factor_state, polarization_state, bose_state):
        if state not in {"included", "removed"}:
            raise ValueError("spectral correction states must be included or removed")

    source_cross = source in {"cross_section", "measured_intensity"}
    target_cross = target == "cross_section"
    if target_cross:
        def correction_power(state):
            return int(state == "removed")
        thermal_power = -int(bose_state == "removed") if source != "s_qw" else 0
    else:
        def correction_power(state):
            return -int(state == "included")
        thermal_power = int(bose_state == "included") if source != "s_qw" else 1
    kinematic_power = -1 if source_cross and not target_cross else 1 if not source_cross and target_cross else 0
    moment_power = -1 if source_cross and not target_cross else 1 if not source_cross and target_cross else 0
    constant = MAGNETIC_CROSS_SECTION_BARN_PER_MU_B_SQ if absolute else np.pi
    multiplier = float(source_scale) if source_cross else 1.0
    if source_cross and not target_cross:
        multiplier *= np.pi / constant
    elif source == "chi_double_prime" and target_cross:
        multiplier *= constant / np.pi
    elif source == "s_qw":
        multiplier *= constant if target_cross else np.pi
    if moment_power:
        multiplier *= magnetic_moment_factor(moment_unit, g_factor) ** moment_power

    factor = np.asarray(multiplier, dtype=float)
    terms = (
        (form_factor_sq, correction_power(form_factor_state)),
        (polarization, correction_power(polarization_state)),
        (kf_ki, kinematic_power),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for values, power in terms:
            if power:
                values = np.asarray(values, dtype=float)
                valid = np.isfinite(values) & (values > 0 if power < 0 else values >= 0)
                factor = factor * np.where(valid, values ** power, np.nan)
        if thermal_power:
            denominator = bose_denominator(energy, temperature_K)
            factor = factor * denominator ** thermal_power
    return np.where(np.isfinite(factor), factor, np.nan)


def convert_histogram_spectral_values(data, factor, *, offset=0.0, unit, quantity_type, metadata):
    """Apply an affine spectral calibration while preserving valid statistics.

    A deterministic intensity subtraction preserves primary sensitivities,
    but changes a count measurement into a derived response. Its raw counting
    contract and likelihood are therefore retired rather than reused.
    """
    from .background_channels import (
        BACKGROUND_EXCEPTIONS_KEY,
        accumulate_background_channel,
        scale_background_channels,
    )
    from .histogram_statistics import EVENT_STATISTICS_CHANNELS, EVENT_STATISTICS_KEY
    from .measurement_aggregation import MEASUREMENT_STATISTICS_CHANNELS, MEASUREMENT_STATISTICS_KEY
    from .measurement_scaling import scale_measurement_data

    declaration = metadata.get("measurement_contract")
    if isinstance(declaration, dict):
        metadata = {**metadata, "measurement_contract": {**declaration, "quantity": quantity_type, "value_units": unit}}
    source = scale_background_channels(data, factor, unit=unit, quantity_type=quantity_type)
    if BACKGROUND_EXCEPTIONS_KEY in source.metadata:
        # The caller's metadata was copied before the diagnostic originals were
        # scaled. Do not overwrite the transformed exception payload with it.
        metadata = {**metadata, BACKGROUND_EXCEPTIONS_KEY: source.metadata[BACKGROUND_EXCEPTIONS_KEY]}
    converted = scale_measurement_data(source, factor, metadata=metadata)
    offset = np.broadcast_to(np.asarray(offset, dtype=float), data.shape)
    if np.all(offset == 0.0):
        return converted
    metadata = dict(converted.metadata)
    channels = dict(converted.auxiliary_channels)
    # Offsets of continuous means act on their additive weighted numerator.
    if isinstance(metadata.get(MEASUREMENT_STATISTICS_KEY), dict):
        numerator, _, weight, _ = MEASUREMENT_STATISTICS_CHANNELS
        if numerator in channels and weight in channels:
            channel = channels[numerator]
            channels[numerator] = replace(channel, values=channel.values + offset * channels[weight].values)
    counting = EVENT_STATISTICS_KEY in metadata or isinstance(declaration, dict) and declaration.get("kind") == "counting"
    if counting:
        metadata.pop("measurement_contract", None)
        metadata.pop(EVENT_STATISTICS_KEY, None)
        for name in EVENT_STATISTICS_CHANNELS:
            channels.pop(name, None)
        metadata.update(measurement_target_required=True, measurement_derivation={
            "version": 1, "operation": "deterministic_background_subtraction",
            "original_contract": data.metadata.get("measurement_contract"),
            "quantity": quantity_type, "value_units": unit,
            "uncertainty": "retained_source_sensitivities",
        })
    if "poisson_count_model" in metadata:
        metadata.pop("poisson_count_model")
        metadata["poisson_count_model_invalidation"] = "deterministic background subtraction changes the raw count target"
    shifted = converted.with_updates(
        signal=converted.signal + offset,
        mask=converted.mask | ~np.isfinite(offset),
        metadata=metadata, auxiliary_channels=channels,
        source_dependencies=converted.source_dependencies, counting_dependencies=None,
    )
    return accumulate_background_channel(
        converted, shifted, contribution=-offset,
        contribution_errors=np.zeros(data.shape),
        valid=~converted.mask & np.isfinite(offset),
    )
