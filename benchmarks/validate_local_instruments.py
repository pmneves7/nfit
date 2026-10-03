"""Read-only, bounded acceptance checks using optional local MACS/HYSPEC files.

This manual validator is intentionally outside pytest: private scientific inputs
are optional, and no original project or source is saved. It emits scalar audit
results only. MACS tests its recorded compatibility convention and an explicitly
constructed known-exposure Poisson model separately; it changes no defaults.
HYSPEC inputs in this example are already reduced MDEvents, so these checks do
not establish raw HYSPEC DGS-conversion parity with an external reducer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np

from nfit import (
    DataGroup,
    MeasurementContract,
    bin_measurement_points,
    composite_dataset_data,
    export_composite_recipe,
    export_reduction_recipe,
    mdevent_dataset_group,
    replay_composite_recipe,
    replay_measurement_histogram,
    replay_reduction_recipe,
)
from nfit.histogram_reduction import pool_normalized_histogram
from nfit.histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
    selected_event_statistics,
)
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.measurement_profiles import prepare_measurement_profile
from nfit.project_imports import dataset_entry_from_path
from nfit.project_io import NfitProject
from nfit.rebin import rebin_nd
from nfit.source_lineage import source_identity
from nfit.source_selection import SourceSelection
from nfit.source_selection_imports import import_source_selection

BASE = Path.home() / "Library/CloudStorage/OneDrive-JohnsHopkins/_research/LiV2O4"


def project_snapshot(path):
    stat = path.stat()
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("project.json")
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "project_metadata_sha256": hashlib.sha256(payload).hexdigest()}


def entries(payload):
    for group in payload["data_groups"]:
        yield from group_entries(group)


def group_entries(group):
    yield from group.get("datasets", [])
    for child in group.get("subgroups", []):
        yield from group_entries(child)


def assert_close(actual, expected, *, rtol=2e-12, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    actual, expected = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    valid = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[valid] - expected[valid]), initial=0))


def enclosing_edges(points, bins=1):
    result = []
    for coordinate in points.coordinates():
        lo, hi = float(np.nanmin(coordinate)), float(np.nanmax(coordinate))
        margin = max(0.1, (hi-lo)*0.05)
        result.append(np.linspace(lo-margin, hi+margin, bins+1))
    return tuple(result)


def config_for_edges(edges, *, fractional=False):
    return {"enabled": True, "coordinate_mode": "hkle", "fractional": fractional,
            "normalize": True, "mean_weighting": "uniform", "workers": 1,
            "minimum_coverage": 0, "minimum_samples": 0,
            "axes": [{"name": name, "mode": "edges", "bin_edges": edge.tolist(),
                      "lower": float(edge[0]), "upper": float(edge[-1]),
                      "num_bins": len(edge)-1, "fractional": fractional,
                      "vector": np.eye(4)[dim].tolist()}
                     for dim, (name, edge) in enumerate(zip(("H", "K", "L", "DeltaE"), edges, strict=True))],
            "symmetry": {"mode": "none", "expression": ""}}


def validate_macs(project):
    with zipfile.ZipFile(project) as archive:
        payload = json.loads(archive.read("project.json"))
    selected = [entry for entry in entries(payload)
                if entry.get("metadata", {}).get("importer") == "macs_nexus"
                and Path(entry["metadata"]["source_file"]).is_file()]
    # Choose the smallest original file, then exercise its independent streams.
    original = min(selected, key=lambda entry: Path(entry["metadata"]["source_file"]).stat().st_size)
    source = Path(original["metadata"]["source_file"])
    options = original["metadata"]["import_options"]
    reports = []
    for stream in ("spec", "diff"):
        started = time.perf_counter()
        entry = dataset_entry_from_path(source,
            data_type="single_crystal_inelastic" if stream == "spec" else "single_crystal_energy_integrated",
            importer_name="macs_nexus", importer_options={**options, "stream": stream})
        points = entry.data
        with h5py.File(source, "r") as handle:
            counts = np.asarray(handle[f"entry/DAS_logs/{stream}Detector/counts"], float).reshape(-1)
        exposure = points.normalization_denominator
        valid = points.mask & np.isfinite(exposure) & (exposure > 0)
        assert_close(points.intensity[valid]*exposure[valid], counts[valid])
        assert_close((points.sigma[valid]*exposure[valid])**2, np.maximum(counts[valid], 1))
        total_n = float(exposure[valid].sum())
        total_c = float(counts[valid].sum())
        floor_v = float(np.maximum(counts[valid], 1).sum())
        group = DataGroup(f"MACS {stream}", datasets=[entry],
                          metadata={"composite": config_for_edges(enclosing_edges(points))})
        histogram = composite_dataset_data(group, apply_spectral_channels=False)
        signal_difference = assert_close(histogram.signal.item(), total_c/total_n)
        sigma_difference = assert_close(histogram.errors.item(), np.sqrt(floor_v)/total_n)
        root, node = replay_composite_recipe(export_composite_recipe(NfitProject(data_groups=[group]), group.name))
        replayed = composite_dataset_data(root, node=node, apply_spectral_channels=False)
        assert_close(replayed.signal, histogram.signal)
        assert_close(replayed.errors, histogram.errors)
        # Declare the actual raw-count model explicitly; do not infer event counts
        # or overwrite the imported sigma floor / saved scientific configuration.
        model = MeasurementContract(kind="counting", estimator="exposure_pool",
            quantity="monitor-normalized counts", value_units=points.metadata["signal_unit"],
            exposure_units="monitor/target/efficiency")
        counting = points.with_updates(sigma=np.sqrt(np.maximum(counts, 0))/exposure,
            metadata={**points.metadata, "measurement_contract": model.to_dict()},
            measurement_payload={EVENT_SIGNAL_NUMERATOR: counts,
                EVENT_VARIANCE_NUMERATOR: counts, NORMALIZATION_DENOMINATOR: exposure})
        edges = enclosing_edges(points, bins=3)
        legacy_fine = composite_dataset_data(group, config_override=config_for_edges(edges),
                                             apply_spectral_channels=False)
        legacy_view = {**legacy_fine.metadata, "mask": legacy_fine.mask.ravel()}
        legacy_profile = prepare_measurement_profile(legacy_view, legacy_fine.signal.ravel(),
            legacy_fine.errors.ravel(), selected=~legacy_fine.mask.ravel(),
            indices=np.zeros(legacy_fine.signal.size, dtype=int), edges=[0, 1])
        fine = bin_measurement_points(counting, edges)
        stats = selected_event_statistics(fine)
        assert stats is not None
        assert_close(np.sum(stats[0]), total_c)
        assert_close(np.sum(stats[1]), total_c)
        assert_close(np.sum(stats[2]), total_n)
        pooled = pool_normalized_histogram(fine.signal, fine.errors**2, stats[2], axes=tuple(range(4)), mask=fine.mask)
        assert_close(pooled[0], total_c/total_n)
        assert_close(pooled[1], total_c/total_n**2)
        exposed = stats[2] > 0
        assert np.all(fine.mask[~exposed])
        # The existing fractional public binner is audited against an independent
        # normalization identity. Its physical assignment differs from discrete.
        fractional = rebin_nd(points.intensity[valid], np.column_stack(points.coordinates())[valid],
            data_errs=points.sigma[valid], data_weights=exposure[valid],
            lower=[edge[0] for edge in edges], upper=[edge[-1] for edge in edges],
            bin_edges=edges, num_bins=[3]*4, fractional=True, normalize=True,
            mean_weighting="uniform")
        fractional_n = fractional._normalization
        occupied = fractional_n > 0
        assert_close(np.sum(fractional_n), total_n)
        assert_close(np.sum(fractional.binned_data[occupied]*fractional_n[occupied]), total_c)
        if stream == "diff":
            assert np.all(points.E == 0)
            assert "elastic" in points.metadata["coordinate_approximation"]
        reports.append({"energy_analysis": points.metadata["energy_analysis"],
            "coordinate_approximation": points.metadata.get("coordinate_approximation"),
            "stream": stream, "source": source.name, "points": points.size,
            "accepted_points": int(np.sum(valid)), "covered_zero_points": int(np.sum(valid & (counts == 0))),
            "normalization_sum": total_n, "raw_count_sum": total_c,
            "compatibility_variance_sum": floor_v, "max_signal_difference": signal_difference,
            "max_sigma_difference": sigma_difference, "poisson_reference_variance": total_c/total_n**2,
            "unexposed_bins": int(np.sum(~exposed)), "exposed_bins": int(np.sum(exposed)),
            "recipe_replay": "passed", "fractional_count_and_exposure_conservation": "passed",
            "native_contract": points.metadata.get("measurement_contract"),
            "native_histogram_retains_count_payload": selected_event_statistics(legacy_fine) is not None,
            "native_histogram_retains_exposure": NORMALIZATION_DENOMINATOR in legacy_fine.auxiliary_channels,
            "native_profile_estimator": legacy_profile.contract.estimator,
            "native_profile_signal": legacy_profile.data.signal.item(),
            "native_profile_sigma": legacy_profile.data.errors.item(),
            "direct_count_pool_signal": total_c/total_n,
            "direct_count_pool_sigma": np.sqrt(floor_v)/total_n,
            "native_profile_vs_direct_count_pool_difference": legacy_profile.data.signal.item()-total_c/total_n,
            "elapsed_seconds": time.perf_counter()-started})
    # Real source-selection repeat syntax retains alias identity and distinguishes
    # SPEC/DIFF detector streams; independent merging of repeated folders rejects.
    run = int(source.name.split("_")[-1].split(".")[0])
    prefix = source.name[:source.name.rfind(str(run))]
    suffix = source.name[source.name.rfind(str(run))+len(str(run)):]
    repeated = import_source_selection(DataGroup("repeated"),
        SourceSelection(source.parent, prefix, suffix, f"{run}|2|"),
        preserve_groups=True, importer_name="macs_nexus", importer_options=options)
    repeated.metadata["composite"] = config_for_edges(enclosing_edges(points))
    try:
        composite_dataset_data(repeated, apply_spectral_channels=False)
    except SourceReplayRequired:
        duplicate_rejected = True
    else:
        raise AssertionError("Repeated MACS source folders were combined independently")
    assert source_identity(source, importer="macs_nexus", stream="spec") != source_identity(source, importer="macs_nexus", stream="diff")
    return {"streams": reports, "duplicate_guard": duplicate_rejected,
            "limitations": ["known monitor and detector-efficiency normalization; their calibration uncertainty is not represented",
                "native MACS importer retains legacy sqrt(max(counts,1)) errors and no explicit measurement contract",
                "native point histogram drops exposure/count payload; later legacy precision profile estimates a different target than direct exposure pooling",
                "explicit raw-Poisson reference is a validation-only construction, not a scientific default change"]}


def validate_hyspec(project):
    with zipfile.ZipFile(project) as archive:
        payload = json.loads(archive.read("project.json"))
    sources = {Path(entry["metadata"]["source_file"]) for entry in entries(payload)
               if entry.get("kind") == "mdevent" and entry["metadata"].get("source_file")}
    candidates, lab_candidates = [], []
    for path in sources:
        if path.is_file():
            with h5py.File(path, "r") as handle:
                dimension = str(handle["MDEventWorkspace"].attrs["dimension0"])
                if "QSample" in dimension:
                    candidates.append(path)
                elif "QLab" in dimension:
                    lab_candidates.append(path)
    source = min(candidates, key=lambda path: path.stat().st_size)
    group = mdevent_dataset_group(source)
    container_runs = len(group.datasets)
    group.datasets = group.datasets[:2]
    assert group.datasets[0].metadata["instrument_name"] == "HYSPEC"
    group = replay_reduction_recipe(export_reduction_recipe(group))
    c, v, event_count = 0., 0., 0
    with h5py.File(source, "r") as handle:
        data = handle["MDEventWorkspace/event_data/event_data"]
        for start in range(0, data.shape[0], 250_000):
            block = np.asarray(data[start:start+250_000, :3], float)
            block = block[np.isin(block[:, 2].astype(np.int64), [0, 1])]
            c += float(np.sum(block[:, 0]))
            v += float(np.sum(block[:, 1]))
            event_count += len(block)
    options = dict(lower=[-15, -15, -15, -20], upper=[15, 15, 15, 20],
                   num_bins=[120, 120, 1, 40], max_batch_bytes=24*1024**2)
    lab_source = min(lab_candidates, key=lambda path: path.stat().st_size)
    lab_group = mdevent_dataset_group(lab_source)
    try:
        replay_measurement_histogram(lab_group, **options)
    except ValueError as error:
        assert "QSample" in str(error)
    else:
        raise AssertionError("QLab background was silently interpreted as QSample")
    started = time.perf_counter()
    fine = replay_measurement_histogram(group, **options)
    fine_seconds = time.perf_counter()-started
    stats = selected_event_statistics(fine)
    assert stats is not None
    count_diff = assert_close(np.sum(stats[0]), c, rtol=2e-10)
    variance_diff = assert_close(np.sum(stats[1]), v, rtol=2e-10)
    assert_close(np.sum(fine.num_events), event_count)
    exposed = stats[2] > 0
    zeros = exposed & (stats[0] == 0)
    assert np.all(fine.signal[zeros] == 0) and np.all(fine.errors[zeros] == 0)
    assert np.all(fine.mask[~exposed])
    started = time.perf_counter()
    final = replay_measurement_histogram(group, **{**options, "num_bins": [1, 1, 1, 40]})
    final_seconds = time.perf_counter()-started
    pooled = pool_normalized_histogram(fine.signal, fine.errors**2, stats[2], axes=(0, 1, 2), mask=fine.mask)
    signal_diff = assert_close(pooled[0], final.signal.ravel(), rtol=1e-8)
    sigma_diff = assert_close(pooled[1], final.errors.ravel()**2, rtol=2e-8)
    final_stats = selected_event_statistics(final)
    assert_close(pooled[2], final_stats[2].ravel(), rtol=1e-8)
    try:
        replay_measurement_histogram(group, **options, datasets=[group.datasets[0], group.datasets[0]])
    except SourceReplayRequired:
        duplicate_rejected = True
    else:
        raise AssertionError("Duplicate HYSPEC logical run was accepted by source replay")
    threshold = float(np.quantile(stats[2][exposed], 0.1))
    fringe = exposed & (stats[2] <= threshold)
    assert np.any(fringe) and np.any(zeros)
    assert_close(fine.errors[fringe]**2, stats[1][fringe]/stats[2][fringe]**2)
    symmetric = replay_measurement_histogram(group,
        **{**options, "num_bins": [1, 1, 1, 40]},
        symmetry_operations=[np.eye(3), -np.eye(3)])
    symmetric_stats = selected_event_statistics(symmetric)
    for actual, expected in zip(symmetric_stats, final_stats, strict=True):
        assert_close(actual, 2*expected, rtol=2e-8)
    assert_close(symmetric.signal, final.signal, rtol=2e-8)
    assert_close(symmetric.errors**2, final.errors**2/2, rtol=2e-8)
    return {"source": source.name, "instrument": "HYSPEC", "event_count": event_count,
        "event_weighted_numerator_sum": c, "event_variance_numerator_sum": v,
        "normalization_sum": float(np.sum(stats[2])),
        "normalization_treated_as_known": True, "incident_energy_unit": "meV", "t0_unit": "microsecond",
        "run_count": len(group.datasets), "container_runs": container_runs, "max_count_sum_difference": count_diff,
        "max_variance_sum_difference": variance_diff, "max_final_cut_signal_difference": signal_diff,
        "max_final_cut_variance_difference": sigma_diff, "exposed_bins": int(np.sum(exposed)),
        "covered_zero_bins": int(np.sum(zeros)), "unexposed_bins": int(np.sum(~exposed)),
        "lowest_exposure_decile_bins": int(np.sum(fringe)),
        "lowest_exposure_decile_zero_bins": int(np.sum(fringe & zeros)),
        "lowest_exposure_decile_threshold": threshold,
        "qlab_background_frame_guard": "passed",
        "identity_inversion_recorded_independent_copy_policy": "passed",
        "histogram_shape": list(fine.shape),
        "fine_seconds": fine_seconds, "final_replay_seconds": final_seconds,
        "selected_runs": [{key: dataset.metadata[key] for key in
                           ("run_number", "incident_energy", "t0", "duration", "proton_charge")}
                          for dataset in group.datasets],
        "policies": copy.deepcopy(group.metadata["mdevent"]),
        "event_statistics": fine.metadata.get("event_statistics"),
        "duplicate_guard": duplicate_rejected,
        "limitations": ["source is already reduced MDEvent data, not raw HYSPEC DGS events",
                        "no independent HYSPEC Mantid reduction reference was available in this local example",
                        "identity/inversion sigma halving tests recorded independent-copy compatibility convention, not physical covariance accuracy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macs-project", type=Path, default=BASE/"2026_08_MACS/data.nfit")
    parser.add_argument("--hyspec-project", type=Path, default=BASE/"2026_05_HYSPEC/HYSPEC_all.nfit")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/measurement-acceptance-local.json"))
    arguments = parser.parse_args()
    projects = {"macs": arguments.macs_project, "hyspec": arguments.hyspec_project}
    before = {name: project_snapshot(path) for name, path in projects.items()}
    report = {"schema_version": 1, "python": sys.version, "platform": platform.platform(),
              "numpy_version": np.__version__, "nfit_version": importlib.metadata.version("nfit"), "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(), "scientific_defaults_changed": False,
              "project_before": before, "macs": validate_macs(projects["macs"]),
              "hyspec": validate_hyspec(projects["hyspec"])}
    after = {name: project_snapshot(path) for name, path in projects.items()}
    assert before == after, "Original science project changed during validation"
    report["project_after"] = after
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024)
    report["timing_notes"] = "single manual invocation; reducer dispatch timings exclude importer scan and independent reference read; not a performance comparison"
    # Drop bulky source recipes: record only resolved policies and source names.
    report["hyspec"]["policies"] = {key: value for key, value in report["hyspec"]["policies"].items()
                                    if key.endswith("_policy") or key in {"normalization", "format"}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(arguments.output), "macs": "checks passed; native profile target gap recorded", "hyspec": "passed",
                      "original_projects_unchanged": before == after}))


if __name__ == "__main__":
    main()
