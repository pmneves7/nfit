"""Metadata-only statistical descriptions and model-specific final-bin intervals."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from .histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_STATISTICS_KEY,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
    has_event_statistics,
    poisson_rate_interval,
)
from .measurement_contracts import MeasurementContract
from .measurement_likelihoods import PoissonCountModel

CONFIDENCE_CHANNELS = ("confidence_lower", "confidence_upper")
DEFAULT_CONFIDENCE = 0.682689492137


def event_contribution_description(semantics):
    """Describe multiplicity without equating copies or cells with events."""
    if semantics == "contributing_histogram_cells":
        return "Contributing histogram cells", "Number of source histogram cells contributing to this bin; not independent neutron events."
    if semantics == "kernel_weighted_event_contributions":
        return "Kernel-weighted event contributions", "Sum of event contributions weighted by the scientific smoothing kernel; not a new integer count or independent measurement."
    return "Event contributions", "Number of event contributions in the bin. Symmetry copies can share an original event and do not create independent measurements."


def _poisson_model(metadata):
    model = PoissonCountModel.from_dict(metadata.get("poisson_count_model"))
    declaration = metadata.get("measurement_contract")
    if declaration is not None:
        contract = MeasurementContract.from_dict(declaration)
        if (contract.kind, contract.dependence, contract.normalizer) != ("counting", "independent", "known"):
            raise ValueError("Rate intervals require independent counts and known exposure")
    if any(metadata.get(key) for key in (
        "background_subtractions", "background_projection", "background_channels_version",
        "histogram_arithmetic", "bose_separation", "measurement_smoothing",
    )):
        raise ValueError("Rate intervals do not apply to subtracted or reconstructed values")
    symmetry = metadata.get("symmetry_operations_hkl")
    if symmetry is not None and (len(symmetry) != 1 or not np.array_equal(symmetry[0], np.eye(3))):
        raise ValueError("Symmetry copies require a source-level interval model")
    return model


def available_confidence_channels(data):
    """Describe interval availability without reading lazy histogram arrays."""
    if not hasattr(data, "auxiliary_channels") or not has_event_statistics(data):
        return ()
    if data.source_dependencies is not None or data.counting_dependencies is not None:
        return ()
    try:
        _poisson_model(data.metadata)
    except (ValueError, KeyError, TypeError):
        return ()
    return CONFIDENCE_CHANNELS


def poisson_interval_channels(view, *, confidence=DEFAULT_CONFIDENCE):
    """Compute Garwood rate bounds after pooling the requested final bins.

    The view must carry an audited ``PoissonCountModel`` and retained additive
    statistics. Integer-looking corrected data are insufficient. Masks and zero
    exposure produce NaN; a covered count zero retains its positive upper bound.
    These bounds must be recomputed after coarsening, never averaged or summed.
    """
    model = _poisson_model(view)
    if view.get("source_dependencies") is not None or view.get("counting_dependencies") is not None:
        raise ValueError("Rate intervals require independent primitive counts")
    marker = view.get(EVENT_STATISTICS_KEY)
    if not isinstance(marker, Mapping) or marker.get("version") != 1:
        raise ValueError("Rate intervals require retained count statistics")
    if marker.get("exposure") != "known" or marker.get("covariance") != "diagonal_only":
        raise ValueError("Rate intervals require independent counts and known exposure")
    names = (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR)
    try:
        numerator, variance, exposure = (np.asarray(view[name], float) for name in names)
    except KeyError as error:
        raise ValueError("Rate intervals require numerator, variance and exposure") from error
    if numerator.shape != variance.shape or numerator.shape != exposure.shape:
        raise ValueError("Rate interval statistics must have matching shapes")
    counts = numerator / model.constant_weight
    good = np.isfinite(exposure) & (exposure > 0)
    if view.get("masks_applied", True):
        good &= ~np.broadcast_to(np.asarray(view.get("mask", False), bool), counts.shape)
    if np.any(~np.isfinite(exposure) | (exposure < 0)):
        raise ValueError("Rate interval exposure must be finite and nonnegative")
    if not (
        np.all(np.isfinite(counts[good]) & (counts[good] >= 0) & (counts[good] == np.floor(counts[good])))
        and np.allclose(variance[good], counts[good] * model.constant_weight**2, rtol=1e-12, atol=0)
        and np.allclose(np.asarray(view["signal"])[good], numerator[good] / exposure[good], rtol=1e-12, atol=0)
        and np.allclose(np.asarray(view["errors"])[good]**2, variance[good] / exposure[good]**2, rtol=1e-12, atol=0)
    ):
        raise ValueError("Statistics do not reproduce the declared constant-weight Poisson model")
    bounds = poisson_rate_interval(
        np.where(good, counts, 0), np.where(good, exposure, 0),
        constant_weight=model.constant_weight, confidence=confidence,
    )
    return dict(zip(CONFIDENCE_CHANNELS, bounds, strict=True))


def measurement_diagnostics(data):
    """Return bounded, JSON-compatible provenance without numerical-array I/O.

    This is a description of available information, not an accuracy certificate.
    Full acquisition lists and array-valued metadata are deliberately omitted.
    """
    metadata = data.metadata
    result = {"standard_uncertainty": "stored standard uncertainty; no confidence interpretation inferred"}
    for key in ("measurement_contract", "event_statistics", "measurement_statistics",
                "measurement_derivation", "poisson_count_model", "counting_uncertainty",
                "num_events_semantics", "dgs_reduction_policies", "resolved_run_reductions",
                "raw_dgs_calibration", "corelli_reconstruction", "measurement_smoothing"):
        if key in metadata:
            result[key] = _bounded_description(metadata[key])
    result["source_dependencies"] = getattr(data, "source_dependencies", None) is not None
    result["counting_dependencies"] = getattr(data, "counting_dependencies", None) is not None
    if hasattr(data, "auxiliary_channels"):
        result["channels"] = [
            {"name": name, "label": channel.label, "unit": channel.unit}
            for name, channel in list(data.auxiliary_channels.items())[:100]
        ]
    result["confidence_intervals"] = (
        {"model": "exact equal-tailed Poisson rate", "confidence": DEFAULT_CONFIDENCE,
         "channels": list(CONFIDENCE_CHANNELS)}
        if available_confidence_channels(data)
        else "No supported audited interval model. Standard uncertainty is shown separately."
    )
    result["smoothing"] = "Plot smoothing is a display approximation. Scientific count smoothing pools numerator/exposure and requires propagated source dependencies."
    return result


def _bounded_description(value, depth=0):
    if depth >= 4:
        return "[nested metadata omitted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, Mapping):
        return {str(key): _bounded_description(item, depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, (tuple, list)):
        return [_bounded_description(item, depth + 1) for item in value[:20]]
    return f"[{type(value).__name__} payload omitted]"


def measurement_diagnostics_text(data):
    """Human-readable report shared by the GUI and scripts."""
    return json.dumps(measurement_diagnostics(data), indent=2, ensure_ascii=False)
