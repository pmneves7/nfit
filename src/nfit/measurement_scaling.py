"""Known calibrations of primary values and their retained measurement payloads."""

from __future__ import annotations

import numpy as np

from .dataset import PointData4D
from .histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_STATISTICS_KEY,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
)
from .mdhisto import MDHistoChannel, MDHistoData
from .measurement_aggregation import (
    MEASUREMENT_STATISTICS_CHANNELS,
    MEASUREMENT_STATISTICS_KEY,
)
from .measurement_dependencies import (
    CountingDependencies,
    SourceReplayRequired,
    project_source_dependencies,
)


def _scaled_dependencies(dependencies, factor):
    if dependencies is None:
        return None
    factors = np.broadcast_to(factor, dependencies.shape).ravel()
    indices = np.flatnonzero(np.isfinite(factors))
    return project_source_dependencies(dependencies, indices, indices, factors[indices], dependencies.shape)


def _scaled_payload(payload, factor, metadata, *, finite_scalar=False):
    result = dict(payload)
    valid = True if finite_scalar else np.isfinite(factor)
    if EVENT_STATISTICS_KEY in metadata:
        if all(name in payload for name in (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR)):
            c = payload[EVENT_SIGNAL_NUMERATOR]*factor
            v = (payload[EVENT_VARIANCE_NUMERATOR]*factor)*factor
            result[EVENT_SIGNAL_NUMERATOR] = c if finite_scalar else np.where(valid, c, np.nan)
            result[EVENT_VARIANCE_NUMERATOR] = v if finite_scalar else np.where(valid, v, np.nan)
            result[NORMALIZATION_DENOMINATOR] = payload[NORMALIZATION_DENOMINATOR] if finite_scalar else np.where(valid, payload[NORMALIZATION_DENOMINATOR], 0)
    marker = metadata.get(MEASUREMENT_STATISTICS_KEY)
    if isinstance(marker, dict) and all(name in payload for name in MEASUREMENT_STATISTICS_CHANNELS):
        numerator, variance, weight, count = (payload[name] for name in MEASUREMENT_STATISTICS_CHANNELS)
        adjustment = np.ones(factor.shape)
        if marker.get("estimator") == "inverse_variance_mean":
            used = valid & np.isfinite(weight) & (weight > 0)
            magnitudes = np.abs(factor[used])
            if np.any(magnitudes == 0) and np.any(magnitudes > 0):
                raise SourceReplayRequired("Mixed zero/nonzero precision calibrations require an exact-constraint target or source replay")
            if len(magnitudes) and np.all(magnitudes == 0):
                metadata[MEASUREMENT_STATISTICS_KEY] = {**marker, "deterministic_zero_scale": True}
            elif len(magnitudes):
                reference = float(np.min(magnitudes))
                adjustment = np.zeros(factor.shape)
                adjustment[used] = (reference/np.abs(factor[used]))**2
                scale = float(marker["precision_scale"])*reference
                if not np.isfinite(scale) or scale <= 0:
                    raise SourceReplayRequired("Transformed precision scale is not representable; rescale physical units or replay sources")
                metadata[MEASUREMENT_STATISTICS_KEY] = {**marker, "precision_scale": scale}
        result.update(dict(zip(MEASUREMENT_STATISTICS_CHANNELS,
            (np.where(valid, (numerator*factor)*adjustment, 0),
             np.where(valid, ((variance*factor)*factor)*adjustment**2, 0),
             np.where(valid, weight*adjustment, 0), np.where(valid, count, 0)), strict=True)))
    model = metadata.get("poisson_count_model")
    if model is not None:
        finite = factor[np.isfinite(factor)]
        if finite.size and np.all(finite == finite[0]) and finite[0] > 0 and np.all(valid):
            from .measurement_likelihoods import PoissonCountModel

            original = PoissonCountModel.from_dict(model)
            metadata["poisson_count_model"] = PoissonCountModel(constant_weight=original.constant_weight*float(finite[0]),
                provenance=f"{original.provenance}; known scalar calibration {float(finite[0]):g}").to_dict()
        else:
            metadata.pop("poisson_count_model", None)
            metadata["poisson_count_model_invalidation"] = "zero, nonfinite or spatially varying calibration is not one positive constant event weight"
    return result


def scale_measurement_data(data, factor, *, metadata=None):
    """Apply a known scalar or pointwise calibration without discarding lineage.

    Counts scale numerator/variance while keeping exposure. Continuous uniform
    sums preserve observation counts. Precision sums update their reference
    scale and weights, so grouping transformed observations uses the transformed
    uncertainties. Primitive sensitivities and uncertain-exposure numerator
    bundles receive the same multiplier. Nonfinite factors mask their cells.
    """
    if not isinstance(data, (MDHistoData, PointData4D)):
        raise TypeError("Measurement scaling supports histograms and fit point data")
    shape = data.shape if isinstance(data, MDHistoData) else data.intensity.shape
    scalar = np.asarray(factor, float)
    finite_scalar = scalar.ndim == 0 and bool(np.isfinite(scalar))
    factor = np.broadcast_to(scalar, shape)
    updated_metadata = dict(data.metadata)
    if metadata is not None:
        updated_metadata.update(metadata)
    if isinstance(data, MDHistoData):
        payload = {name: channel.values for name, channel in data.auxiliary_channels.items()}
    else:
        payload = {} if data.measurement_payload is None else dict(data.measurement_payload)
    scaled = _scaled_payload(payload, factor, updated_metadata, finite_scalar=finite_scalar)
    declaration = updated_metadata.get("measurement_contract")
    eligible = ~data.mask if isinstance(data, MDHistoData) else data.mask
    if isinstance(declaration, dict) and declaration.get("kind") == "counting" and np.any(eligible & np.isfinite(factor) & (factor < 0)):
        # A signed physical response still has useful C/N sensitivities, but is
        # no longer a nonnegative count measurement under the scalar contract.
        updated_metadata.pop("measurement_contract", None)
        updated_metadata.update(measurement_target_required=True,
            measurement_derivation={"version": 1, "operation": "signed_calibration",
                "original_contract": data.metadata.get("measurement_contract"),
                "quantity": updated_metadata.get("signal_quantity_type", declaration["quantity"]),
                "value_units": declaration["value_units"], "uncertainty": "retained_source_sensitivities"})
    # These arrays are either existing immutable inputs or owned new results.
    for values in scaled.values():
        values.setflags(write=False)
    dependencies = _scaled_dependencies(data.source_dependencies, factor)
    if isinstance(data, PointData4D):
        values, errors = data.intensity*factor, data.sigma*np.abs(factor)
        values.setflags(write=False)
        errors.setflags(write=False)
        return data.with_updates(intensity=values, sigma=errors,
            mask=data.mask if finite_scalar else data.mask & np.isfinite(factor), metadata=updated_metadata,
            measurement_payload=None if data.measurement_payload is None else scaled, source_dependencies=dependencies,
            normalization_denominator=data.normalization_denominator if finite_scalar or data.normalization_denominator is None
            else np.where(np.isfinite(factor), data.normalization_denominator, 0))
    bundle = data.counting_dependencies
    if bundle is not None:
        bundle = CountingDependencies(numerator_dependencies=_scaled_dependencies(bundle.numerator_dependencies, factor),
            exposure_dependencies=bundle.exposure_dependencies)
    channels = {name: MDHistoChannel(values, data.auxiliary_channels[name].errors,
        data.auxiliary_channels[name].label, data.auxiliary_channels[name].unit, data.auxiliary_channels[name].quantity_type)
        for name, values in scaled.items()}
    denominator = updated_metadata.get("normalization_denominator")
    if not finite_scalar and isinstance(denominator, np.ndarray) and denominator.shape == shape:
        updated_metadata["normalization_denominator"] = np.where(np.isfinite(factor), denominator, 0)
    values, errors = data.signal*factor, data.errors*np.abs(factor)
    values.setflags(write=False)
    errors.setflags(write=False)
    return data.with_updates(signal=values, errors=errors,
        mask=data.mask if finite_scalar else data.mask | ~np.isfinite(factor), metadata=updated_metadata, auxiliary_channels=channels,
        source_dependencies=dependencies, counting_dependencies=bundle)
