"""Original-count sensitivities for symmetry-expanded MACS sample histograms."""

from __future__ import annotations

import hashlib

import numpy as np

from .event_bin_indices import flat_bin_indices
from .histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
)
from .macs_background import _source, _target_symmetry
from .mdhisto import MDHistoChannel
from .measurement_dependencies import SourceDependencies, SourceReplayRequired
from .rebin import ArrayRebinSource, SymmetryRebinSource


def track_macs_sample_dependencies(
    sample, target, *, datasets=None, inherited_masks=None, inherited_masks_by_id=None,
    scale=1., max_batch_bytes=64 * 1024**2, progress_callback=None,
):
    """Replay original counts onto physical bins, retaining symmetry covariance.

    Repeated images of an observation share its acquisition/channel primitive.
    Its dimensionless numerator coefficient includes run fit weight, run scale,
    and the sample composite scale. The exact final monitor exposure normalizes
    those coefficients. Fractional assignment still requires its own replay.
    """
    rebin = target.metadata.get("rebin", {})
    if (target.metadata.get("measurement_reduction", {}).get("assignment") != "nearest_physical_bin"
            or any(rebin.get("fractional_axes", []))):
        raise SourceReplayRequired("MACS sample sensitivities require nonfractional physical bins")
    channel = target.auxiliary_channels.get(NORMALIZATION_DENOMINATOR)
    if channel is None:
        raise ValueError("MACS sample sensitivities require retained monitor exposure")
    denominator = np.asarray(channel.values, float).ravel()
    basis = np.asarray(rebin.get("vectors", np.eye(4)), float)
    if basis.shape != (4, 4) or np.any(~np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("MACS sample sensitivities require an invertible four-dimensional basis")
    inverse = np.linalg.inv(basis)
    operations = _target_symmetry(target)
    edges = [np.asarray(axis.values, float) for axis in target.axes]
    runs = [run for run in (sample.datasets if datasets is None else datasets) if run.enabled and run.fit_weight > 0]
    observations, sources, coefficients, ids, variances = [], [], [], [], []
    exposure = np.zeros(target.signal.size)
    numerator = np.zeros(target.signal.size)
    budget, stored = int(max_batch_bytes), 0
    for index, run in enumerate(runs):
        masks = (inherited_masks_by_id or {}).get(run.id,
            inherited_masks if inherited_masks is not None else sample.masks)
        config, data, _, path = _source(sample, run, masks)
        counts = np.asarray(data.measurement_payload["event_signal_numerator"], float)
        normalizer = np.asarray(data.normalization_denominator, float)
        valid = data.valid_mask(require_positive_sigma=False) & np.isfinite(normalizer) & (normalizer > 0)
        identity = (str(path.resolve()), config.get("stream", "spec"))
        namespace = hashlib.sha256(repr(identity).encode()).hexdigest()
        offset = len(ids)
        ids.extend(f"macs:{namespace}:{i}" for i in range(data.size))
        variances.extend(np.where(np.isfinite(counts) & (counts >= 0), counts, 0.))
        coordinates = np.column_stack(data.coordinates())
        # Each batch canonicalizes coincident images before retaining links.
        batch = max(1, min(data.size, budget // max(1, len(operations) * 192)))
        for start in range(0, data.size, batch):
            stop = min(data.size, start + batch)
            inputs, outputs = [], []
            point_source = ArrayRebinSource(np.arange(start, stop), coordinates[start:stop])
            # Use the authoritative coordinate-coincidence rule. Stabilizer
            # operations must not duplicate points on symmetry planes or axes.
            for image in SymmetryRebinSource(point_source, operations).iter_batches():
                raw = np.asarray(image.data, np.int64)
                bins = flat_bin_indices(image.coords @ inverse, edges, target.shape)
                selected = valid[raw] & (bins >= 0)
                selected &= ~target.mask.ravel()[np.maximum(bins, 0)]
                inputs.append(raw[selected])
                outputs.append(bins[selected])
            raw = np.concatenate(inputs)
            output = np.concatenate(outputs)
            # Pair identity is local to this batch; avoid a giant global key space.
            keys, multiplicity = np.unique((raw - start) * target.signal.size + output, return_counts=True)
            raw = keys // target.signal.size + start
            output = keys % target.signal.size
            weights = multiplicity.astype(float) * run.fit_weight
            np.add.at(exposure, output, normalizer[raw] * weights)
            np.add.at(numerator, output, counts[raw] * weights * run.scale_factor * scale)
            observations.append(output)
            sources.append(raw + offset)
            coefficients.append(weights * run.scale_factor * scale / denominator[output])
            stored += len(output)
            if stored * 80 + len(ids) * 180 + target.signal.size * 32 > budget:
                raise SourceReplayRequired("MACS symmetry sensitivities exceed the memory budget; increase the binning budget or use a coarser grid")
        if progress_callback is not None:
            progress_callback(dict(stage="macs_sample_symmetry", iteration=index + 1, total=len(runs),
                                   message=f"Tracking symmetry copies for {run.name}"))
    covered = ~target.mask.ravel() & (denominator > 0)
    if not np.allclose(exposure[covered], denominator[covered], rtol=1e-10, atol=1e-12):
        raise ValueError("MACS source replay does not reproduce the sample histogram exposure")
    if not np.allclose(numerator[covered] / denominator[covered], target.signal.ravel()[covered], rtol=1e-10, atol=1e-12):
        raise ValueError("MACS source replay does not reproduce the sample histogram signal")
    dependencies = SourceDependencies(shape=target.shape, source_ids=tuple(ids), source_variances=np.asarray(variances),
        observation_indices=np.concatenate(observations), source_indices=np.concatenate(sources),
        coefficients=np.concatenate(coefficients), max_bytes=budget)
    metadata = {**target.metadata, "sample_uncertainty": "original_counts_with_shared_symmetry_copies"}
    metadata.pop("poisson_count_model", None)
    variance = dependencies.variances()
    channels = dict(target.auxiliary_channels)
    if EVENT_VARIANCE_NUMERATOR in channels:
        channels[EVENT_VARIANCE_NUMERATOR] = MDHistoChannel(
            variance * np.square(denominator.reshape(target.shape)), label="Event numerator variance")
    if EVENT_STATISTICS_KEY in metadata:
        metadata[EVENT_STATISTICS_KEY] = {**metadata[EVENT_STATISTICS_KEY], "covariance": "tracked_sources"}
    return target.with_updates(errors=np.sqrt(variance), source_dependencies=dependencies,
                               metadata=metadata, auxiliary_channels=channels)
