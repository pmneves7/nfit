"""Read-only real two-bank HYSPEC cached-background profile acceptance.

Nine sample angles per bank, complete distinct background streams, nested
public composites and original-grid regular/rotated box queries. The independent
oracle scatters primitive coefficients into an explicit event-by-profile matrix
and squares only after summing their original-voxel contributions. Only scalar
receipts are saved; no project or scientific input is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from validate_hyspec_background_replay import BASIS, ROOT, parameter_masks
from validate_hyspec_cached_background_profiles import covariance_oracle

from nfit import (
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    bin_mdevent_group,
    clear_cached_background_profile_queries,
    composite_dataset_data,
    mdevent_dataset_group,
    project_measured_background_mdevent,
    replay_cached_background_box_profiles,
)
from nfit._parallel import thread_budget
from nfit.backgrounds import subtract_aligned_background
from nfit.project_composites import data_group_composite_config


def compare(actual, expected):
    """Require agreement relative to the actual scientific scale, including zero."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    maximum = float(np.max(np.abs(actual-expected), initial=0))
    scale = float(np.max(np.abs(expected), initial=0))
    assert maximum <= 2e-11*scale, (maximum, scale)
    return dict(cells=int(actual.size), max_absolute=maximum,
                max_relative_to_peak=maximum/scale if scale else 0.)


def module_provenance():
    modules = {}
    for name in ("background_profile_queries", "cached_background_replay", "project_composites",
                 "mdevent", "mdevent_background", "_mdevent_background_numba"):
        module = importlib.import_module(f"nfit.{name}")
        path = Path(module.__file__)
        modules[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return modules


def config(data):
    return dict(enabled=True, auto_rebin=False, mean_weighting="uniform",
        minimum_coverage=0., fractional=True, axes=[dict(name=axis.name,
            lower=float(axis.values[0]), upper=float(axis.values[-1]), mode="edges",
            bin_edges=axis.values.tolist(), vector=BASIS[dim].tolist(),
            auto_lower=False, auto_upper=False, fractional=True)
            for dim, axis in enumerate(data.axes)])


def query_membership(data, extents, angle, display_selected, profiles):
    """Independent center rotation and profile-bin membership, without helpers."""
    x = data.axes[0].centers[:, None, None, None]
    y = data.axes[3].centers[None, None, None, :]
    x0, x1, y0, y1 = extents
    cx, cy = (x0+x1)/2, (y0+y1)/2
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    u, v = cx+(x-cx)*c+(y-cy)*s, cy-(x-cx)*s+(y-cy)*c
    selected = np.broadcast_to((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1), data.shape).copy()
    selected &= display_selected.T[:, None, None, :]
    valid = selected & ~data.mask & np.isfinite(data.signal)
    exposure = data.auxiliary_channels["normalization_denominator"].values
    valid &= np.isfinite(exposure) & (exposure > 0)
    queries = {}
    for side, coordinates, profile in (("x", u, profiles.x_measurement), ("y", v, profiles.y_measurement)):
        edges = profile.data.axes[0].values
        indices = np.clip(np.searchsorted(edges, coordinates, side="right")-1, 0, len(edges)-2)
        queries[side] = dict(selected=selected, valid=valid,
            indices=np.broadcast_to(indices, data.shape), edges=edges,
            profile=profile, bins=len(edges)-1)
    return queries


def projected_covariance_oracle(sample, source, runs, edges, queries, bank_weights):
    """Stream each primitive once; explicit coefficient matrix is independent.

    This generalizes the established energy-only covariance_oracle to regular
    and rotated x/y projections. Membership, masks and transforms are evaluated
    independently; no production event/bin/covariance helpers are called.
    """
    shape = tuple(len(edge)-1 for edge in edges)
    fractions = np.array([run.metadata["proton_charge"]*run.fit_weight for run in runs])
    fractions /= fractions.sum()
    ub = np.asarray(sample.metadata["mdevent"]["ub_matrix"])
    inverses, bounds, masks = [], [], []
    with h5py.File(runs[0].metadata["source_file"], "r") as handle:
        for run in runs:
            experiment = handle[f"MDEventWorkspace/experiment{run.metadata['mdevent_experiment_index']}"]
            gonio = experiment["logs/goniometer/rotation_matrix"][()].reshape(3, 3)
            inverses.append(np.linalg.inv(BASIS)[:3, :3].T @ np.linalg.inv(gonio @ (2*np.pi*ub)))
            bounds.append(experiment["logs/processed_histogram_bins/value"][()][[0, -1]])
            masks.append(parameter_masks(experiment))
    result = {name: np.zeros(query["bins"]) for name, query in queries.items()}
    digest, event_count = hashlib.sha256(), 0
    with h5py.File(source.datasets[0].metadata["source_file"], "r") as handle:
        experiment = handle["MDEventWorkspace/experiment0"]
        source_bounds = experiment["logs/processed_histogram_bins/value"][()][[0, -1]]
        source_masks = parameter_masks(experiment)
        gonio = experiment["logs/goniometer/rotation_matrix"][()].reshape(3, 3)
        values = handle["MDEventWorkspace/event_data/event_data"]
        for start in range(0, len(values), 8192):
            raw = values[start:start+8192]
            digest.update(raw.tobytes())
            block = raw.astype(float)
            assert np.all(block[:, 2:4] == 0)
            lab = block[:, 5:8]
            if source.metadata["mdevent"]["dimensions"][0]["frame"] == "QSample":
                lab = lab @ gonio.T
            locations = []
            for inverse, limits, masked in zip(inverses, bounds, masks, strict=True):
                coordinates = np.column_stack((lab @ inverse.T, block[:, 8]))
                accepted = (~np.isin(block[:, 4], [*source_masks, *masked])
                    & (block[:, 8] >= max(limits[0], source_bounds[0]))
                    & (block[:, 8] <= min(limits[1], source_bounds[1])))
                indices = []
                for dim, edge in enumerate(edges):
                    index = np.searchsorted(edge, coordinates[:, dim], side="right")-1
                    index[coordinates[:, dim] == edge[-1]] = len(edge)-2
                    accepted &= (index >= 0) & (index < shape[dim])
                    indices.append(np.clip(index, 0, shape[dim]-1))
                locations.append((np.ravel_multi_index(tuple(indices), shape), accepted))
            for name, query in queries.items():
                coefficients = np.zeros((len(block), query["bins"]))
                for fraction, (flat, accepted) in zip(fractions, locations, strict=True):
                    valid = accepted & query["valid"].ravel()[flat] & (bank_weights.ravel()[flat] != 0)
                    rows = np.flatnonzero(valid)
                    final = query["indices"].ravel()[flat[valid]]
                    coefficients[rows, final] += fraction*bank_weights.ravel()[flat[valid]]
                result[name] += np.sum(block[:, 1, None]*coefficients**2, axis=0)
            event_count += len(block)
    return result, digest.hexdigest(), event_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = module_provenance()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paths = {bank: [ROOT/"data"/f"Ei15meV_50K_240Hz_s2_{bank}{suffix}.nxs"
                    for suffix in ("", "_bkg")] for bank in (34, 70)}
    before = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for pair in paths.values() for path in pair}
    # Retain the earlier bounded voxel count while covering both detector banks.
    requested_edges = [np.linspace(-3, 3, 81), np.linspace(-2, 2, 21),
                       np.linspace(-2, 2, 9), np.linspace(-12, 12, 49)]
    banks, children = {}, []
    started = time.perf_counter()
    with thread_budget(8):
        for bank, pair in paths.items():
            sample, source = map(mdevent_dataset_group, pair)
            runs = [sample.datasets[index] for index in np.linspace(0, len(sample.datasets)-1, 9, dtype=int)]
            sample_data = bin_mdevent_group(sample, lower=[edge[0] for edge in requested_edges],
                upper=[edge[-1] for edge in requested_edges], num_bins=[len(edge)-1 for edge in requested_edges],
                vectors=BASIS, datasets=runs)
            background = project_measured_background_mdevent(sample, source, sample_data, datasets=runs)
            subtracted = subtract_aligned_background(sample_data, background)
            banks[bank] = dict(sample=sample, source=source, runs=runs,
                               sample_data=sample_data, background=background, subtracted=subtracted)
            child = DatasetGroup(f"50 K bank {bank}", datasets=[DatasetEntry(f"bank {bank}", subtracted, kind="mdhisto")])
            child.metadata["composite"] = config(subtracted)
            children.append(child)
            print("HYSPEC_BANK_READY", bank, "background contributions", int(background.num_events.sum()), flush=True)
        nested = DatasetGroup("Both 50 K banks", subgroups=children, metadata={"composite": config(subtracted)})
        root = DataGroup("HYSPEC acceptance", subgroups=[nested])
        data_group_composite_config(root).update(config(subtracted))
        combined = composite_dataset_data(root, apply_spectral_channels=False)
    assert "cached_background_replay" in combined.metadata, combined.metadata.get("background_replay_unavailable_reason")
    setup_seconds = time.perf_counter()-started
    edges = [axis.values for axis in combined.axes]
    n = combined.auxiliary_channels["normalization_denominator"].values
    support = (~combined.mask & (n > 0)).any(axis=(1, 2)).T
    fringe = support & binary_dilation(~support, iterations=2)
    assert support.any() and fringe.any()
    extents = (-2.5, 2.5, -8., 10.)
    coverage_threshold = .01
    queries, timings = {}, {}
    for region, visible in (("support", support), ("projected_fringe", fringe)):
        for kind, angle in (("regular", 0.), ("rotated", 12.)):
            name = f"{region}_{kind}"
            clear_cached_background_profile_queries()
            started = time.perf_counter()
            profiles = replay_cached_background_box_profiles(combined, x_dim=0, y_dim=3,
                selections={1: (0, combined.shape[1]-1), 2: (0, combined.shape[2]-1)},
                extents=extents, angle=angle, display_selected=visible, coverage_threshold=coverage_threshold)
            timings[name] = time.perf_counter()-started
            for side, query in query_membership(combined, extents, angle, visible, profiles).items():
                queries[f"{name}_{side}"] = query
            print("HYSPEC_BANK_PROFILE_READY", name, timings[name], flush=True)
    # Actual sample exposure determines each bank's contribution to the
    # composite field; missing bank cells have zero coefficient.
    bank_weights, independent, numerator, exposure = {}, np.zeros(combined.shape), np.zeros(combined.shape), np.zeros(combined.shape)
    for bank, data in banks.items():
        sample_data, background, field = data["sample_data"], data["background"], data["subtracted"]
        ns = sample_data.auxiliary_channels["normalization_denominator"].values
        nb = background.auxiliary_channels["normalization_denominator"].values
        valid = ~field.mask & (ns > 0) & (nb > 0)
        bank_weights[bank] = np.divide(ns, nb, out=np.zeros(ns.shape), where=valid)
        independent += np.where(valid, sample_data.auxiliary_channels["event_variance_numerator"].values, 0)
        numerator += np.where(valid, ns*field.signal, 0)
        exposure += np.where(valid, ns, 0)
    assert np.allclose(exposure, n, rtol=1e-12, atol=0)
    assert np.allclose(combined.signal[~combined.mask], numerator[~combined.mask]/n[~combined.mask], rtol=1e-12, atol=0)
    reference = {name: np.zeros(query["bins"]) for name, query in queries.items()}
    sources, energy_oracle_checks = {}, {}
    started = time.perf_counter()
    for bank, data in banks.items():
        variances, digest, count = projected_covariance_oracle(data["sample"], data["source"], data["runs"], edges, queries, bank_weights[bank])
        for name in reference:
            reference[name] += variances[name]
        # Independently cross-check the general projection against the
        # established original-voxel energy covariance oracle.
        energy_queries = {name: bank_weights[bank]*query["valid"] for name, query in queries.items() if "regular_y" in name}
        known, second_digest, second_count = covariance_oracle(data["sample"], data["source"], data["runs"], edges, energy_queries)
        assert digest == second_digest and count == second_count
        energy_oracle_checks[str(bank)] = {}
        for name, energy_variance in known.items():
            selected_energy = np.searchsorted(edges[-1], queries[name]["edges"][:-1], side="left")
            energy_oracle_checks[str(bank)][name] = compare(variances[name], energy_variance[selected_energy])
        totals = {name: float(values.sum()) for name, values in variances.items()}
        assert any(value > 0 for value in totals.values()), f"Bank {bank} contributes no covariance in the selected cuts"
        unmasked_totals, unmasked_counts = {}, {}
        counts = data["background"].num_events
        for name, query in queries.items():
            final_valid = ~query["profile"].data.mask
            unmasked_totals[name] = float(variances[name][final_valid].sum())
            belongs = query["valid"] & (bank_weights[bank] > 0)
            observed = np.bincount(query["indices"][belongs], weights=counts[belongs], minlength=query["bins"])
            unmasked_counts[name] = float(observed[final_valid].sum())
        assert any(value > 0 for value in unmasked_totals.values()), f"Bank {bank} contributes no variance in unmasked final bins"
        assert any(value > 0 for value in unmasked_counts.values()), f"Bank {bank} contributes no observed events in unmasked final bins"
        sources[str(bank)] = dict(event_sha256=digest, background_events=count,
            background_event_contributions=float(data["background"].num_events.sum()),
            covariance_numerator_totals=totals, unmasked_covariance_numerator_totals=unmasked_totals,
            unmasked_cached_cell_event_contributions=unmasked_counts,
            sample_runs=[run.metadata["run_number"] for run in data["runs"]])
    oracle_seconds = time.perf_counter()-started
    receipts = {}
    for name, query in queries.items():
        selected, valid, indices, bins = query["selected"], query["valid"], query["indices"], query["bins"]
        def pool(values, inclusion=valid, indices=indices, bins=bins):
            return np.bincount(indices[inclusion], weights=values[inclusion], minlength=bins)
        denominator = pool(exposure)
        total_variance = pool(independent)+reference[name]
        output = query["profile"].data
        geometric = (np.diff(edges[1])[None, :, None, None]*np.diff(edges[2])[None, None, :, None])
        if "regular_x" in name:
            geometric = geometric*np.diff(edges[3])[None, None, None, :]
        elif "regular_y" in name:
            geometric = geometric*np.diff(edges[0])[:, None, None, None]
        geometric = np.broadcast_to(geometric, combined.shape)
        coverage = np.divide(pool(geometric), pool(geometric, selected),
            out=np.zeros(bins), where=pool(geometric, selected) > 0)
        expected_mask = (denominator <= 0) | (coverage < coverage_threshold)
        np.testing.assert_array_equal(output.mask, expected_mask)
        recorded_sources = {item["source_file"]: item["event_sha256"]
                            for item in output.metadata["cached_background_profile"]["sources"]}
        assert recorded_sources == {str(paths[bank][1]): item["event_sha256"] for bank, item in zip(banks, sources.values(), strict=True)}
        occupied = denominator > 0
        receipts[name] = dict(selected_cells=int(selected.sum()), valid_cells=int(valid.sum()),
            output_bins=bins, masked_bins=int(expected_mask.sum()),
            exposure=compare(output.auxiliary_channels["normalization_denominator"].values, denominator),
            numerator=compare(output.auxiliary_channels["event_signal_numerator"].values, pool(numerator)),
            variance_numerator=compare(output.auxiliary_channels["event_variance_numerator"].values, total_variance),
            signal=compare(output.signal[occupied], pool(numerator)[occupied]/denominator[occupied]),
            variance=compare(output.errors[occupied]**2, total_variance[occupied]/denominator[occupied]**2),
            coverage=compare(output.auxiliary_channels["coverage_fraction"].values, coverage), masks_literal=True)
    after = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for pair in paths.values() for path in pair}
    assert before == after
    assert modules == module_provenance(), "Scientific source changed while validation was running"
    assert script_hash == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(dict(banks=sources, profiles=receipts, profile_seconds=timings,
        setup_seconds=setup_seconds, oracle_seconds=oracle_seconds, original_inputs_unchanged=True,
        before_inputs=before, after_inputs=after, stored_edges=[edge.tolist() for edge in edges],
        basis=BASIS.tolist(), extents=extents, rotated_angle_degrees=12., coverage_threshold=coverage_threshold,
        composite="nested dataset groups; aligned uniform exposure-weighted mean; unity scales and fit weights",
        independent_energy_oracle_checks=energy_oracle_checks,
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        scientific_modules=modules,
        script_sha256=script_hash, scientific_modules_unchanged_during_validation=True,
        scope="Two 50K banks, nine angles each, identity symmetry and unity detector normalization. Full background streams; exact marginal background covariance, recorded sample histogram variance, known exposure. Shared calibration and cross-profile-bin covariance are not reconstructed."), indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
