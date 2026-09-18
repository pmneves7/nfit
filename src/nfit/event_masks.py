"""Apply ordered user masks to normalized event histograms at bin centers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import numpy as np

from .mdhisto import MDHistoChannel, MDHistoData, mdhisto_measured_bins
from .pipeline import DatasetEntry, MaskSpec
from .project_masks import _mdhisto_with_nfit_masks


def reduce_masked_event_runs(
    runs: Sequence[DatasetEntry],
    inherited_masks: Sequence[MaskSpec],
    reduce: Callable[[list[DatasetEntry]], MDHistoData],
    *,
    minimum_samples: float = 0.0,
    zero_count_upper: float,
) -> MDHistoData:
    """Reduce runs sharing a mask once, then combine their valid exposure.

    Masks act on output-bin centers. Partitioning only by distinct ordered
    mask recipes keeps the usual shared-mask case at one native reduction.
    A run excluded from a bin contributes neither counts nor normalization.
    """
    def finish(data: MDHistoData) -> MDHistoData:
        metadata = dict(data.metadata)
        metadata["rebin"] = {**metadata.get("rebin", {}), "minimum_samples": minimum_samples}
        mask = data.mask | (data.num_events < minimum_samples)
        metadata["combined_mask_count"] = int(np.count_nonzero(mask))
        return data.with_updates(mask=mask, metadata=metadata)

    partitions: dict[str, tuple[list[DatasetEntry], list[MaskSpec]]] = {}
    for run in runs:
        masks = [mask for mask in [*inherited_masks, *run.masks] if mask.enabled]
        signature = json.dumps(
            [(mask.type, mask.parameters, mask.invert, mask.additive) for mask in masks],
            sort_keys=True,
        )
        partitions.setdefault(signature, ([], masks))[0].append(run)
    if not partitions:
        return finish(reduce(list(runs)))

    first = None
    for subset, masks in partitions.values():
        data = reduce(subset)
        masked = _mdhisto_with_nfit_masks(DatasetEntry("Event masks", None, masks=masks), data=data)
        if len(partitions) == 1:
            return finish(masked)
        if first is None:
            first = data
            numerator = np.zeros(data.shape)
            variance = np.zeros(data.shape)
            denominator = np.zeros(data.shape)
            events = np.zeros(data.shape)
            all_excluded = np.ones(data.shape, dtype=bool)
        valid = mdhisto_measured_bins(masked)
        exposure = np.where(valid, data.metadata["normalization_denominator"], 0.0)
        numerator += np.where(valid, data.signal, 0.0) * exposure
        variance += np.square(np.where(valid & (data.num_events > 0), data.errors, 0.0) * exposure)
        denominator += exposure
        events += np.where(valid, data.num_events, 0.0)
        all_excluded &= masked.metadata["nfit_mask"]

    measured = denominator > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = numerator / denominator
        errors = np.sqrt(variance) / denominator
    total_events = float(np.sum(events))
    event_weight_rms = float(np.sqrt(np.sum(variance) / total_events)) if total_events else 1.0
    covered_zero = measured & (events == 0)
    errors[covered_zero] = zero_count_upper * event_weight_rms / denominator[covered_zero]
    metadata = dict(first.metadata)
    metadata.update(
        normalization_denominator=denominator,
        event_weight_rms=event_weight_rms,
        nfit_mask=all_excluded,
        nfit_mask_count=int(np.count_nonzero(all_excluded)),
        file_mask=~measured & ~all_excluded,
        combined_mask_count=int(np.count_nonzero(~measured)),
    )
    channels = dict(first.auxiliary_channels)
    channels["normalization_denominator"] = MDHistoChannel(
        denominator, label="Detector-trajectory normalization", unit="arbitrary normalization units"
    )
    return finish(first.with_updates(
        signal=signal, errors=errors, mask=~measured, num_events=events,
        metadata=metadata, auxiliary_channels=channels,
    ))
