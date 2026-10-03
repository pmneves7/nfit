"""Read-only cached-field HYSPEC profile covariance acceptance.

The reference explicitly maps source observations into original cached voxels,
then applies the sample-exposure coefficients of the requested final target.
It does not call production replay accumulation or membership helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from validate_hyspec_background_replay import BASIS, ROOT, compare, parameter_masks

from nfit import (
    bin_mdevent_group,
    clear_cached_background_profile_queries,
    mdevent_dataset_group,
    project_measured_background_mdevent,
    replay_cached_background_profile,
)
from nfit.backgrounds import subtract_aligned_background


def covariance_oracle(sample, source, runs, edges, region_weights):
    shape = tuple(len(edge) - 1 for edge in edges)
    bins = shape[-1]
    fractions = np.array([run.metadata["proton_charge"] * run.fit_weight for run in runs])
    fractions /= fractions.sum()
    inverses, bounds, masks = [], [], []
    ub = np.asarray(sample.metadata["mdevent"]["ub_matrix"])
    with h5py.File(runs[0].metadata["source_file"]) as handle:
        for run in runs:
            experiment = handle[f"MDEventWorkspace/experiment{run.metadata['mdevent_experiment_index']}"]
            gonio = experiment["logs/goniometer/rotation_matrix"][()].reshape(3, 3)
            inverses.append(np.linalg.inv(BASIS)[:3, :3].T @ np.linalg.inv(gonio @ (2 * np.pi * ub)))
            bounds.append(experiment["logs/processed_histogram_bins/value"][()][[0, -1]])
            masks.append(parameter_masks(experiment))
    result = {name: np.zeros(bins) for name in region_weights}
    primitive_count = 0
    digest = hashlib.sha256()
    with h5py.File(source.datasets[0].metadata["source_file"]) as handle:
        experiment = handle["MDEventWorkspace/experiment0"]
        source_bounds = experiment["logs/processed_histogram_bins/value"][()][[0, -1]]
        source_masks = parameter_masks(experiment)
        gonio = experiment["logs/goniometer/rotation_matrix"][()].reshape(3, 3)
        values = handle["MDEventWorkspace/event_data/event_data"]
        for start in range(0, len(values), 32768):
            raw = values[start:start + 32768]
            digest.update(raw.tobytes())
            block = raw.astype(float)
            assert np.all(block[:, 2:4] == 0)
            lab = block[:, 5:8]
            if source.metadata["mdevent"]["dimensions"][0]["frame"] == "QSample":
                lab = lab @ gonio.T
            coefficients = {name: np.zeros(len(block)) for name in region_weights}
            for inverse, limits, masked, fraction in zip(inverses, bounds, masks, fractions, strict=True):
                coordinates = np.column_stack((lab @ inverse.T, block[:, 8]))
                accepted = (~np.isin(block[:, 4], [*source_masks, *masked])
                            & (block[:, 8] >= max(limits[0], source_bounds[0]))
                            & (block[:, 8] <= min(limits[1], source_bounds[1])))
                locations = []
                for dimension, edge in enumerate(edges):
                    index = np.searchsorted(edge, coordinates[:, dimension], side="right") - 1
                    index[coordinates[:, dimension] == edge[-1]] = len(edge) - 2
                    accepted &= (index >= 0) & (index < shape[dimension])
                    locations.append(np.clip(index, 0, shape[dimension] - 1))
                flat = np.ravel_multi_index(tuple(locations), shape)
                for name, weights in region_weights.items():
                    coefficients[name][accepted] += fraction * weights.ravel()[flat[accepted]]
            energy_index = np.searchsorted(edges[-1], block[:, 8], side="right") - 1
            energy_index[block[:, 8] == edges[-1][-1]] = bins - 1
            valid = (energy_index >= 0) & (energy_index < bins)
            for name, coefficients_for_region in coefficients.items():
                result[name] += np.bincount(
                    energy_index[valid], weights=block[valid, 1] * coefficients_for_region[valid]**2,
                    minlength=bins,
                )
            primitive_count += len(block)
    return result, digest.hexdigest(), primitive_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [ROOT / "data" / f"Ei15meV_50K_240Hz_s2_34{suffix}.nxs" for suffix in ("", "_bkg")]
    before = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    sample, source = map(mdevent_dataset_group, paths)
    runs = [sample.datasets[index] for index in np.linspace(0, len(sample.datasets) - 1, 9, dtype=int)]
    requested_edges = [np.linspace(-2, 2, 81), np.linspace(-1, 1, 21),
                       np.linspace(-.5, .62, 9), np.linspace(-12, 12, 49)]
    started = time.perf_counter()
    sample_data = bin_mdevent_group(sample, lower=[edge[0] for edge in requested_edges],
        upper=[edge[-1] for edge in requested_edges], num_bins=[len(edge) - 1 for edge in requested_edges],
        vectors=BASIS, datasets=runs)
    background = project_measured_background_mdevent(sample, source, sample_data, datasets=runs)
    subtracted = subtract_aligned_background(sample_data, background)
    setup_seconds = time.perf_counter() - started
    edges = [axis.values for axis in subtracted.axes]
    ns = sample_data.auxiliary_channels["normalization_denominator"].values
    nb = background.auxiliary_channels["normalization_denominator"].values
    covered = ~subtracted.mask & (ns > 0) & (nb > 0)
    projected_support = covered.any(axis=(1, 2))
    projected_fringe = projected_support & binary_dilation(~projected_support, iterations=2)
    regions = {"all_covered": covered,
               "lowest_exposure_decile": covered & (ns <= np.quantile(ns[covered], .1)),
               "coverage_fringe": covered & binary_dilation(~covered, iterations=2),
               "projected_HHH_energy_fringe": covered & projected_fringe[:, None, None, :]}
    region_weights = {name: np.divide(ns, nb, out=np.zeros(ns.shape), where=region)
                      for name, region in regions.items()}
    started = time.perf_counter()
    reference_variances, digest, event_count = covariance_oracle(sample, source, runs, edges, region_weights)
    oracle_seconds = time.perf_counter() - started
    assignments = np.broadcast_to(np.arange(subtracted.shape[-1]), subtracted.shape)
    sample_variance = sample_data.auxiliary_channels["event_variance_numerator"].values
    background_variance = np.where(background.num_events > 0, np.nan_to_num((background.errors * nb)**2), 0)
    receipts = {}
    for name, region in regions.items():
        clear_cached_background_profile_queries()
        started = time.perf_counter()
        profile = replay_cached_background_profile(subtracted, selected=region,
            indices=assignments, edges=edges[-1])
        cold = time.perf_counter() - started
        started = time.perf_counter()
        warm = replay_cached_background_profile(subtracted, selected=region,
            indices=assignments, edges=edges[-1])
        warm_seconds = time.perf_counter() - started
        exposure = np.where(region, ns, 0).sum(axis=(0, 1, 2))
        numerator = (np.where(region, subtracted.signal, 0) * ns).sum(axis=(0, 1, 2))
        reference_variance = (np.where(region, sample_variance, 0).sum(axis=(0, 1, 2))
                              + reference_variances[name])
        occupied = exposure > 0
        output = profile.data
        assert output.metadata["cached_background_profile"]["sources"][0]["event_sha256"] == digest
        assert not output.metadata["cached_background_profile"]["query_cache_hit"]
        assert warm.data.metadata["cached_background_profile"]["query_cache_hit"]
        diagonal_variance = (np.where(region, sample_variance, 0)
                             + background_variance * region_weights[name]**2).sum(axis=(0, 1, 2))
        ratios = np.sqrt(np.divide(reference_variance, diagonal_variance,
                                  out=np.full(exposure.shape, np.nan), where=diagonal_variance > 0))
        finite_ratios = ratios[np.isfinite(ratios)]
        receipts[name] = {"selected_cells": int(region.sum()),
            "cold_seconds": cold, "warm_seconds": warm_seconds,
            "numerator": compare(output.auxiliary_channels["event_signal_numerator"].values, numerator),
            "variance_numerator": compare(output.auxiliary_channels["event_variance_numerator"].values, reference_variance),
            "signal": compare(output.signal[occupied], numerator[occupied] / exposure[occupied]),
            "variance": compare(output.errors[occupied]**2, reference_variance[occupied] / exposure[occupied]**2),
            "exposure": compare(output.auxiliary_channels["normalization_denominator"].values, exposure),
            "warm_sigma": compare(warm.data.errors[occupied], output.errors[occupied]),
            "exact_to_diagonal_sigma_median": float(np.median(finite_ratios)),
            "exact_to_diagonal_sigma_max": float(np.max(finite_ratios, initial=0))}
        print("HYSPEC_CACHED_PROFILE_VALIDATED", name, cold, warm_seconds, flush=True)
    after = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    assert before == after
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"regions": receipts, "setup_seconds": setup_seconds,
        "oracle_seconds": oracle_seconds, "background_events": event_count,
        "event_sha256": digest, "sample_runs": [run.metadata["run_number"] for run in runs],
        "shape": list(subtracted.shape), "original_inputs_unchanged": True,
        "fringe_definitions": {"coverage_fringe": "Covered original 4D cells within two cells of missing coverage; sparse nine-angle sampling makes this equal all covered cells.",
            "projected_HHH_energy_fringe": "Covered cells within two HHH-energy projected cells of missing projected coverage after integrating other momentum axes."},
        "before_inputs": before, "after_inputs": after,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target": "sample_exposure_weighted_subtracted_field_mean",
        "scope": "50 K 34-degree bank, nine sample angles, complete background, unity normalization, identity symmetry. Sample variance follows the recorded histogram policy; background covariance uses an independent event oracle. Calibration/exposure uncertainty and cross-profile-bin covariance are not reconstructed."}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
