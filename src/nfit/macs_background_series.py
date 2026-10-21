"""Energy-matched file backgrounds for one MACS measurement series.

Background files are reduction inputs, not project datasets or viewer choices.
Each matched sample run weights its correction by its share of sample exposure
in the final bin; unmatched energies retain their original contribution.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .backgrounds import subtract_aligned_background
from .histogram_statistics import NORMALIZATION_DENOMINATOR
from .macs import import_macs_nexus
from .macs_background import prepare_macs_background_sample, project_measured_background_macs
from .macs_radial_background import project_isotropic_background_macs
from .macs_reduction import MACS_DEFAULTS
from .mdhisto import MDHistoData
from .measurement_dependencies import combine_source_dependencies, project_source_dependencies
from .pipeline import DatasetEntry, DatasetGroup
from .reduction_recipes import effective_reduction_config
from .source_lineage import merge_source_lineage_metadata


def _condition(path, settings):
    # DIFF coordinates use E=0; matching must still use analyzed acquisition Ei/Ef.
    options = {key: value for key, value in settings.items() if key in MACS_DEFAULTS}
    options.update(stream="spec", vanadium_files=[], background_files=[])
    data = import_macs_nexus(path, options)
    ei = float(data.metadata["incident_energy"])
    ef = float(data.metadata["fixed_final_energy_meV"])
    return ei, ef, ei - ef


def _exposure(data):
    channel = data.auxiliary_channels.get(NORMALIZATION_DENOMINATOR)
    if channel is None:
        raise ValueError("MACS background file series requires retained sample monitor exposure")
    value = np.asarray(channel.values, float)
    return np.where(~data.mask & np.isfinite(value) & (value > 0), value, 0.)


def apply_macs_background_series(
    sample, target, *, prepare_sample, datasets=None, inherited_masks=None,
    inherited_masks_by_id=None, source_signature="", max_batch_bytes=64 * 1024**2,
    progress_callback=None,
):
    """Subtract matching file backgrounds on a combined physical MACS grid.

    ``prepare_sample(runs)`` bins the selected sample runs onto exactly the same
    final axes, using identical masks, scales and exposure pooling. Backgrounds
    inherit each sample run's effective reduction and vanadium calibration.
    Matching uses measured acquisition deltaE and Ef, even for DIFF, with the
    group-owned tolerance in meV. Matching does not shift measured energies.
    Multiple matching backgrounds pool raw counts and monitor exposure.

    Unmatched energies remain unsubtracted. Missing spatial coverage for a
    matched energy masks only output bins containing that sample contribution.
    Shared background observations retain their cross-bin/run covariance.
    Physical, unsymmetrized, exposure-pooled sample bins are required.
    """
    if not isinstance(target, MDHistoData) or len(target.axes) != 4:
        raise ValueError("MACS background file series requires a four-dimensional physical histogram")
    if target.metadata.get("rebin", {}).get("normalize") is False:
        raise ValueError("MACS background series requires normalized exposure pooling")
    settings = effective_reduction_config(sample)
    paths = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in settings.get("background_files", [])))
    if not paths or float(settings.get("background_scale", 1.)) == 0:
        return target
    runs = [d for d in (sample.datasets if datasets is None else datasets) if d.enabled and d.fit_weight > 0]
    sample_paths = {str(Path(d.metadata["source_file"]).resolve()) for d in runs}
    if sample_paths.intersection(paths):
        raise ValueError("MACS sample and background files must be independent acquisitions")
    tolerance = float(settings.get("background_delta_e_tolerance_meV", .05))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("MACS background deltaE tolerance must be finite and nonnegative")
    denominator = _exposure(target)
    condition_cache = {}
    corrections, dependencies, provenance, source_metadata = [], [], [], []
    missing = np.zeros(target.shape, bool)
    method = settings.get("background_method", "replay")
    if method not in {"replay", "isotropic"}:
        raise ValueError("MACS background method must be replay or isotropic")
    for index, run in enumerate(runs):
        config = effective_reduction_config(sample, run)
        condition_options = tuple((key, repr(config.get(key))) for key in (
            "incident_energy_policy", "final_energy_policy", "incident_energy_override",
            "final_energy_override", "mask_misaligned_analyzers", "analyzer_alignment_tolerance_deg",
            "detect_dead_analyzers", "dead_channel_min_nonzero_fraction", "masked_analyzer_channels"))
        def condition(path, condition_options=condition_options, config=config):
            key = str(Path(path).resolve()), condition_options
            if key not in condition_cache:
                condition_cache[key] = _condition(path, config)
            return condition_cache[key]
        ei, ef, transfer = condition(run.metadata["source_file"])
        matches = [path for path in paths if abs(condition(path)[2] - transfer) <= tolerance
                   and abs(condition(path)[1] - ef) <= tolerance]
        record = dict(sample_file=str(Path(run.metadata["source_file"]).resolve()),
                      incident_energy_meV=ei, final_energy_meV=ef,
                      delta_e_meV=transfer, background_files=matches)
        provenance.append(record)
        if not matches:
            continue
        cohort = prepare_sample([run])
        if cohort.shape != target.shape or any(not np.array_equal(a.values, b.values)
            for a, b in zip(cohort.axes, target.axes, strict=True)):
            raise ValueError("MACS sample subset must use the exact final grid")
        fraction = np.divide(_exposure(cohort), denominator, out=np.zeros(target.shape), where=denominator > 0)
        if not np.any(fraction > 0):
            continue
        # Private numerical adapters never enter the saved project or viewer catalog.
        sources = [DatasetEntry(Path(path).name, None, kind=run.kind, data_type=run.data_type,
                    metadata={"source_file": path, "importer": "macs_nexus", "import_options": config})
                   for path in matches]
        background = DatasetGroup("MACS background files", datasets=sources,
            metadata={"macs": dict(config)})
        projector = project_isotropic_background_macs if method == "isotropic" else project_measured_background_macs
        options = ({"q_bin_width": settings.get("background_q_bin_width", .05),
                    "interpolation": "linear"} if method == "isotropic" else {})
        # Matching bounds deltaE and Ef separately, so the derived Ei difference
        # can be up to twice that tolerance. No measured coordinates are changed.
        projected = projector(sample, background, cohort, datasets=[run],
            inherited_masks=inherited_masks, inherited_masks_by_id=inherited_masks_by_id,
            background_inherited_masks=[], energy_tolerance_meV=2 * tolerance + 1e-12,
            max_batch_bytes=max_batch_bytes, progress_callback=progress_callback, **options)
        missing |= (fraction > 0) & projected.mask
        corrections.append(np.where(~projected.mask, projected.signal, 0.) * fraction)
        valid = np.flatnonzero((~projected.mask & (fraction > 0)).ravel())
        dependencies.append(project_source_dependencies(projected.source_dependencies, valid, valid,
            fraction.ravel()[valid], target.shape, max_bytes=max_batch_bytes))
        source_metadata.append(projected)
        if progress_callback is not None:
            progress_callback(dict(stage="macs_background_series", iteration=index + 1,
                total=len(runs), message=f"Matching background for {run.name}"))
    recipe = dict(version=1, method=method, delta_e_tolerance_meV=tolerance,
        final_energy_tolerance_meV=tolerance, files=paths, matches=provenance,
        unmatched_policy="retain_unsubtracted_sample", estimator="sample_exposure_weighted_background")
    if not dependencies:
        return target.with_updates(metadata={**target.metadata, "macs_background_series": recipe})
    sample_target = prepare_macs_background_sample(target, source_signature=source_signature)
    combined = combine_source_dependencies(dependencies, [1.] * len(dependencies), max_bytes=max_batch_bytes)
    signal = np.sum(corrections, axis=0)
    metadata = {**source_metadata[0].metadata, **merge_source_lineage_metadata(*source_metadata),
        "background_projection": {"instrument": "MACS", "mode": "file_series", **recipe}}
    correction = target.with_updates(signal=signal, errors=np.sqrt(combined.variances()),
        mask=missing | target.mask, source_dependencies=combined, counting_dependencies=None,
        auxiliary_channels={}, metadata=metadata)
    result = subtract_aligned_background(sample_target, correction, scale=float(settings.get("background_scale", 1.)))
    final_metadata = {**result.metadata, "macs_background_series": recipe,
        "macs_background_file_signature": hashlib.sha256(repr(paths).encode()).hexdigest()}
    final_metadata.pop("poisson_count_model", None)
    final_metadata.pop("zero_count_variance", None)
    return result.with_updates(metadata=final_metadata)
