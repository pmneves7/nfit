"""Explicit count smoothing with bounded propagation of primitive dependencies."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product

import numpy as np

from .histogram_statistics import (
    EVENT_STATISTICS_KEY,
    event_statistics_channels,
    normalized_event_statistics,
    selected_event_statistics,
)
from .mdhisto import MDHistoChannel, mdhisto_coverage_fraction
from .measurement_contracts import MeasurementContract
from .measurement_dependencies import (
    DEFAULT_SOURCE_DEPENDENCY_BYTES,
    CountingDependencies,
    SourceReplayRequired,
    project_source_dependencies,
    ratio_source_dependencies,
)


def smooth_count_histogram(data, sigma, *, truncate=4., fill_missing=False,
                           max_bytes=DEFAULT_SOURCE_DEPENDENCY_BYTES):
    """Smooth count numerator and exposure, then divide with shared covariance.

    ``sigma`` gives Gaussian widths in native bins, one per axis. The finite
    kernel uses zero extension at boundaries. Masked/unexposed cells supply no
    observations. The result is a kernel-weighted response with a changed
    resolution, not the original point response. All primitive sensitivities
    remain available for later cuts/coaddition. The input is immutable.

    This optional scientific operation requires a counting contract and explicit
    numerator/exposure dependencies. It does not infer independent primitives
    from diagonal errors and is bounded for selected histograms, not full DGS
    event volumes. Display blur is a separate, approximate plotting operation.
    """
    sigma = np.broadcast_to(np.asarray(sigma, float), (len(data.shape),))
    if np.any(~np.isfinite(sigma) | (sigma < 0)) or not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("Smoothing widths must be nonnegative and truncate finite and positive")
    if not any(sigma):
        return data
    declaration = data.metadata.get("measurement_contract")
    if declaration is None:
        raise ValueError("Scientific count smoothing requires a declared counting contract")
    contract = MeasurementContract.from_dict(declaration)
    if contract.kind != "counting":
        raise ValueError("Scientific count smoothing requires a counting contract")
    bundle = data.counting_dependencies
    if bundle is None:
        raise SourceReplayRequired("Scientific smoothing requires numerator/exposure source dependencies; replay primitives at the requested grid")
    limit = min(int(max_bytes), bundle.numerator_dependencies.max_bytes, bundle.exposure_dependencies.max_bytes)
    radii = [min(size - 1, int(math.ceil(truncate * width))) if width else 0
             for size, width in zip(data.shape, sigma, strict=True)]
    offset_count = math.prod(2 * radius + 1 for radius in radii)
    if data.signal.size * offset_count * 64 > limit:
        raise SourceReplayRequired("Scientific smoothing kernel exceeds its work budget; select a smaller histogram or narrower kernel")
    statistics = selected_event_statistics(data)
    if statistics is None:
        raise ValueError("Scientific smoothing requires valid retained count statistics")
    numerator, _variance, exposure = statistics
    good = ~data.mask & np.isfinite(data.signal) & np.isfinite(data.errors) & (exposure > 0)
    indices = np.arange(data.signal.size).reshape(data.shape)
    inputs, outputs, coefficients = [], [], []
    support = np.zeros(data.shape)
    possible = np.zeros(data.shape)
    norm = math.prod(sum(math.exp(-.5 * (offset / width)**2) for offset in range(-radius, radius + 1))
                     if width else 1. for width, radius in zip(sigma, radii, strict=True))
    for offset in product(*(range(-radius, radius + 1) for radius in radii)):
        weight = math.exp(-.5 * sum((shift / width)**2 for shift, width in zip(offset, sigma, strict=True) if width)) / norm
        source = tuple(slice(max(0, -shift), min(size, size - shift))
                       for shift, size in zip(offset, data.shape, strict=True))
        destination = tuple(slice(max(0, shift), min(size, size + shift))
                            for shift, size in zip(offset, data.shape, strict=True))
        valid = good[source]
        possible[destination] += weight
        support[destination] += weight * valid
        inputs.append(indices[source][valid])
        outputs.append(indices[destination][valid])
        coefficients.append(np.full(np.count_nonzero(valid), weight))
    inputs, outputs, coefficients = map(np.concatenate, (inputs, outputs, coefficients))
    smoothed_c = np.bincount(outputs, weights=coefficients * numerator.ravel()[inputs], minlength=data.signal.size).reshape(data.shape)
    smoothed_n = np.bincount(outputs, weights=coefficients * exposure.ravel()[inputs], minlength=data.signal.size).reshape(data.shape)
    transformed = CountingDependencies(
        numerator_dependencies=project_source_dependencies(bundle.numerator_dependencies, inputs, outputs, coefficients, data.shape, max_bytes=limit),
        exposure_dependencies=project_source_dependencies(bundle.exposure_dependencies, inputs, outputs, coefficients, data.shape, max_bytes=limit),
    )
    smoothed_v = transformed.numerator_dependencies.variance()
    signal, _ = normalized_event_statistics(smoothed_c, smoothed_v, smoothed_n)
    dependencies = ratio_source_dependencies(transformed, smoothed_c, smoothed_n)
    errors = np.sqrt(dependencies.variance())
    mask = ~(smoothed_n > 0)
    if not fill_missing:
        mask |= ~good
    errors = np.where(smoothed_n > 0, errors, np.nan)
    channels = event_statistics_channels(smoothed_c, smoothed_v, smoothed_n)
    for name, channel in tuple(channels.items()):
        original = data.auxiliary_channels[name]
        channels[name] = MDHistoChannel(channel.values, label=original.label,
                                       unit=original.unit, quantity_type=original.quantity_type)
    channels["coverage_fraction"] = MDHistoChannel(
        np.divide(support, possible, out=np.zeros(data.shape), where=possible > 0),
        label="Measured smoothing kernel fraction", unit="fraction",
    )
    # Original diagnostics remain on the unmodified input; a transformed result
    # must not retain stale fitted/background/calibration channel arrays.
    channels["original_coverage_fraction"] = MDHistoChannel(
        mdhisto_coverage_fraction(data), label="Original coverage fraction", unit="fraction",
    )
    metadata = dict(data.metadata)
    metadata.pop("poisson_count_model", None)
    for name in ("fit", "residual", "fit_likelihood", "file_mask", "nfit_mask", "coverage_mask"):
        metadata.pop(name, None)
    metadata["measurement_contract"] = replace(contract, dependence="shared_sources").to_dict()
    metadata[EVENT_STATISTICS_KEY] = {**metadata[EVENT_STATISTICS_KEY], "covariance": "source_dependencies"}
    metadata["measurement_smoothing"] = {
        "version": 1, "sigma_bins": sigma.tolist(), "truncate": float(truncate),
        "fill_missing": bool(fill_missing), "kernel": "Gaussian_zero_extension",
        "target": "kernel_weighted_response", "uncertainty": "propagated_primitive_dependencies",
    }
    metadata["coverage_semantics"] = "fraction_of_smoothing_kernel_with_measured_support"
    metadata["num_events_semantics"] = "kernel_weighted_event_contributions"
    events = np.bincount(outputs, weights=coefficients * data.num_events.ravel()[inputs], minlength=data.signal.size).reshape(data.shape)
    return data.with_updates(signal=signal, errors=errors, mask=mask, num_events=events,
                             metadata=metadata, auxiliary_channels=channels,
                             source_dependencies=dependencies, counting_dependencies=transformed)
