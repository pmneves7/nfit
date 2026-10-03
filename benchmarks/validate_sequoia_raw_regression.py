"""Read-only three-run SEQUOIA powder regression after DGS converter changes.

Use the existing ORNL nfit --run-script runtime. Output and tiny metadata-only
reference proxies must live inside IPTS. Existing raw data, Mantid references,
calibrations and projects are never modified. No Mantid process is required.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import runpy
import socket
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

# This shared manual comparison script provides the final-cut/reference helpers
# and selects the current source tree in the frozen runtime before nfit imports.
helpers = runpy.run_path(str(Path(__file__).with_name("validate_sequoia_measurements.py")),
                        run_name="sequoia_reference_helpers")
arrays, difference = (helpers[key] for key in ("arrays", "difference"))

ROOT, REFERENCE, PROJECTS, SYMMETRY = (helpers[key] for key in ("ROOT", "REFERENCE", "PROJECTS", "SYMMETRY"))
RUNS = (392985, 393089, 393387)
MERGED = ROOT / "shared/tempMDE/All_Ei60_T5K_all_617_runs.nxs"
VAN = ROOT / "shared/tempMDE/van382236_D"
RAW = [Path(f"/SNS/SEQ/IPTS-37189/nexus/SEQ_{run}.nxs.h5") for run in RUNS]
MDE = [REFERENCE / f"mantid_mde_{run}.nxs" for run in RUNS]
MODULES = ("raw_dgs", "raw_dgs_geometry", "raw_dgs_geometry_precision", "raw_dgs_hyspec",
           "raw_dgs_cache", "raw_dgs_monitors", "raw_dgs_pulses", "mdevent", "dgs_reduction_policy")


def _module_receipts():
    result = {}
    for name in MODULES:
        module = importlib.import_module("nfit." + name)
        result[name] = {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    return result


def _source_stats():
    return {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in (*PROJECTS, *RAW, *MDE, MERGED, VAN, REFERENCE / "mantid_mde_fine_3barm.h5")}


def _reference_proxy(source, destination):
    """Supply missing UB metadata without copying or changing stored events."""
    with h5py.File(source, "r") as original, h5py.File(destination, "w") as proxy:
        workspace = original["MDEventWorkspace"]
        target = proxy.create_group("MDEventWorkspace")
        target.attrs.update(dict(workspace.attrs))
        for name in workspace:
            if not name.startswith("experiment"):
                target[name] = h5py.ExternalLink(str(source), f"/MDEventWorkspace/{name}")
                continue
            experiment = target.create_group(name)
            experiment.attrs.update(dict(workspace[name].attrs))
            for child in workspace[name]:
                if child != "sample":
                    experiment[child] = h5py.ExternalLink(str(source), f"/MDEventWorkspace/{name}/{child}")
            sample = experiment.create_group("sample")
            if "sample" in workspace[name]:
                sample.attrs.update(dict(workspace[name]["sample"].attrs))
                for child in workspace[name]["sample"]:
                    sample[child] = h5py.ExternalLink(str(source), f"/MDEventWorkspace/{name}/sample/{child}")
            if "oriented_lattice" not in sample:
                sample["oriented_lattice"] = h5py.ExternalLink(str(MERGED), "/MDEventWorkspace/experiment0/sample/oriented_lattice")
    return destination



def _event_comparison(raw, edges):
    # Manual converter audit: no stable public reduced-event iterator exists.
    from nfit import raw_dgs as converter
    from nfit.mdevent import load_detector_normalization
    config = raw.metadata["raw_dgs"]
    norm = load_detector_normalization(VAN)
    mask = converter._combined_detector_mask(config)
    from nfit.dgs_reduction_policy import dgs_powder_coordinates
    result = []
    missing_histograms = {key: np.zeros(tuple(len(edge)-1 for edge in edges))
                          for key in ("numerator", "variance", "counts")}
    for run_index, (source, reference) in enumerate(zip(RAW, MDE, strict=True)):
        info = converter.inspect_raw_dgs_run(source)
        geometry = converter._masked_detector_geometry(converter._detector_geometry(source), norm, mask)
        chunks = converter._iter_reduced_event_chunks(info, config, geometry, info.incident_energy,
                    converter._energy_transfer_bounds(config, info.incident_energy), 128 * 1024**2)
        original = np.concatenate([chunk for chunk, _ in chunks])
        aa = original.astype(np.float32)
        with h5py.File(reference, "r") as handle:
            events = np.asarray(handle["MDEventWorkspace/event_data/event_data"])
        rr = events[:, [5, 6, 7, 8, 0, 1]].astype(np.float32)
        # Void keys retain multiplicity, unlike a plain set comparison.
        def keys(data):
            return np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize*data.shape[1]))).ravel()
        ak, rk = keys(aa[:, [0, 1, 2, 4, 5]]), keys(rr[:, [0, 1, 2, 4, 5]])
        au, ac = np.unique(ak, return_counts=True)
        ru, rc = np.unique(rk, return_counts=True)
        locations = np.searchsorted(au, ru)
        subset = bool(np.all(locations < len(au)) and np.array_equal(au[locations], ru)
                      and np.all(ac[locations] >= rc))
        if not subset:
            mismatched_reference = ~np.isin(rk, ak)
            mismatched_actual = ~np.isin(ak, rk)
            print("EVENT_PAYLOAD_MISMATCH", info.run_number, int(mismatched_reference.sum()), int(mismatched_actual.sum()),
                  "reference", rr[mismatched_reference][:12].tolist(), "actual", aa[mismatched_actual][:12].tolist(), flush=True)
        assert subset, "Stored Mantid QLab/weight/variance multiset is not a subset of current reduction"
        # All boundary omissions in these references have unique event rows.
        missing = ~np.isin(ak, rk)
        np.testing.assert_array_equal(ac[locations], rc)
        expected_missing = (1, 2, 1)[run_index]
        assert len(aa)-len(rr) == int(missing.sum()) == expected_missing
        def order(data):
            return np.lexsort((data[:, 3], data[:, 5], data[:, 4], data[:, 2], data[:, 1], data[:, 0]))
        matched, ordered_reference = aa[~missing][order(aa[~missing])], rr[order(rr)]
        np.testing.assert_array_equal(matched[:, [0, 1, 2, 4, 5]], ordered_reference[:, [0, 1, 2, 4, 5]])
        energy_difference = difference(matched[:, 3], ordered_reference[:, 3])
        np.testing.assert_allclose(matched[:, 3], ordered_reference[:, 3], rtol=0, atol=4e-9)
        q_difference = difference(matched[:, :3], ordered_reference[:, :3])
        np.testing.assert_allclose(matched[:, :3], ordered_reference[:, :3], rtol=0, atol=2e-6)
        omitted = original[missing]
        coordinates, histogram_edges = dgs_powder_coordinates(np.linalg.norm(omitted[:, :3], axis=1), omitted[:, 3], edges)
        for key, weight in (("numerator", omitted[:, 4]), ("variance", omitted[:, 5]),
                            ("counts", np.ones(len(omitted)))):
            missing_histograms[key] += np.histogramdd(coordinates, bins=histogram_edges, weights=weight)[0]
        result.append({"run": info.run_number, "events": len(aa), "reference_events": len(rr),
                       "stored_float32_QLab_weight_variance_multiset_exact_subset": subset,
                       "energy_difference": energy_difference, "energy_absolute_tolerance_meV": 4e-9,
                       "QLab_difference": q_difference, "QLab_absolute_tolerance_inverse_angstrom": 2e-6,
                       "boundary_exclusions": expected_missing,
                       "excluded_float32_events": aa[missing].tolist(),
                       "columns": ["Q_lab_x", "Q_lab_y", "Q_lab_z", "DeltaE", "weight", "variance"]})
    return result, missing_histograms


def _powder_final_cuts(data):
    from nfit import MDHistoSliceViewer, histogram_box_profiles
    from nfit.histogram_statistics import selected_event_statistics
    view = MDHistoSliceViewer(data, x_dim=0, y_dim=1).slice_arrays()
    c, v, n = selected_event_statistics(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(view["signal"], (c / n).T, equal_nan=True, rtol=2e-14)
        np.testing.assert_allclose(view["errors"], (np.sqrt(v) / n).T, equal_nan=True, rtol=2e-14)
    extents = (1.0, 4.0, 3.5, 25.0)
    profiles = histogram_box_profiles(view, view["signal"], view["errors"], extents)
    selected = profiles.selected & ~view["mask"]
    expected = [np.sum(a.T[selected]) for a in (c, v, n)]
    pc, pv, pn = selected_event_statistics(profiles.x_measurement.data)
    np.testing.assert_allclose([pc.sum(), pv.sum(), pn.sum()], expected, rtol=2e-14)
    return {"view_statistics_verified": True, "box_cut_statistics_verified": True,
            "region": list(extents), "numerator": float(expected[0]), "variance_numerator": float(expected[1]),
            "exposure": float(expected[2]), "signal": float(expected[0]/expected[2]),
            "uncertainty": float(np.sqrt(expected[1])/expected[2])}

def main():
    from nfit import mdevent_dataset_group, raw_dgs_dataset_group
    from nfit.mdevent import append_mdevent_file, bin_mdevent_powder_group
    from nfit.raw_dgs import bin_raw_dgs_powder_group

    output = Path(os.environ["NFIT_ACCEPTANCE_OUTPUT"]).resolve()
    if "/IPTS-" not in str(output):
        raise ValueError("SEQUOIA diagnostics must be inside an IPTS directory")
    output.mkdir(parents=True, exist_ok=True)
    before = _source_stats()
    modules = _module_receipts()
    edges = [np.linspace(0, 8, 161), np.linspace(-56, 58, 229)]
    options = dict(lower=[edge[0] for edge in edges], upper=[edge[-1] for edge in edges],
        num_bins=[len(edge) - 1 for edge in edges], bin_edges=edges,
        max_batch_bytes=128 * 1024**2)
    proxies = [_reference_proxy(source, output / f"reference-proxy-{run}.nxs")
               for run, source in zip(RUNS, MDE, strict=True)]
    mde = mdevent_dataset_group(proxies[0], normalization_path=VAN, mask_path=VAN)
    for source in proxies[1:]:
        append_mdevent_file(mde, source)
    assert len(mde.datasets) == 3
    raw = raw_dgs_dataset_group(RAW, normalization_path=VAN, mask_path=VAN)
    raw.metadata["raw_dgs"].update(ub_matrix=mde.metadata["mdevent"]["ub_matrix"], cache_reduced_events=False)
    start = time.perf_counter()
    actual = bin_raw_dgs_powder_group(raw, **options)
    raw_seconds = time.perf_counter() - start
    print("SEQUOIA_RAW_THREE_COMPLETE", flush=True)
    start = time.perf_counter()
    reference = bin_mdevent_powder_group(mde, **options)
    mde_seconds = time.perf_counter() - start
    aa, rr = arrays(actual), arrays(reference)
    covered = rr["norm"] > 0
    low = covered & (rr["norm"] <= np.quantile(rr["norm"][covered], .1))
    fringe = covered & binary_dilation(~covered, iterations=2)
    comparison = {label: {key: difference(aa[key], rr[key], selected) for key in aa}
        for label, selected in (("all", None), ("lowest_exposure_decile", low), ("coverage_fringe", fringe))}
    event_comparison, exclusions = _event_comparison(raw, edges)
    adjusted = {key: aa[key] - exclusions[key] if key in exclusions else aa[key] for key in aa}
    boundary_adjusted = {key: difference(adjusted[key], rr[key]) for key in aa}
    for key in ("numerator", "variance", "counts"):
        np.testing.assert_allclose(adjusted[key], rr[key], rtol=0, atol=2e-12)
    np.testing.assert_array_equal(aa["norm"] > 0, covered)
    np.testing.assert_allclose(aa["norm"][covered], rr["norm"][covered], rtol=3e-8, atol=0)
    after = _source_stats()
    assert before == after
    assert modules == _module_receipts(), "Source modules changed during validation"
    result = {"host": socket.gethostname(), "python": platform.python_version(), "numpy": np.__version__,
        "source_modules": modules, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": list(RUNS), "grid_shape": list(actual.shape), "coordinate_mode": "powder", "symmetry_operations": "none",
        "options": {key: value for key, value in options.items() if key not in {"symmetry_operations", "bin_edges"}},
        "comparison": comparison, "boundary_adjusted_comparison": boundary_adjusted, "event_comparison": event_comparison, "final_cuts": _powder_final_cuts(actual),
        "raw_three_runs_seconds": raw_seconds, "reference_three_runs_seconds": mde_seconds,
        "source_size_mtime": before, "original_sources_and_projects_unchanged": before == after,
        "source_modules_unchanged_during_run": True, "normalization_relative_tolerance": 3e-8,
        "stored_mantid_references": [str(path) for path in MDE],
        "proxy_description": "IPTS-local metadata-only external links; event arrays read from stored per-run Mantid sources; oriented lattice linked from merged reference",
        "timing_scope": "Acceptance timings; concurrent jobs, OS cache and JIT uncontrolled; not a performance comparison",
        "limits": ["Stored independent Mantid reduction references, no fresh Mantid process",
                   "Four recorded per-run ConvertToMD boundary exclusions are audited separately; nfit retains these events",
                   "Three representative SEQUOIA runs, not all617",
                   "Radial Q/E regression plus QLab event audit; not a new HKLE/symmetry certification",
                   "Recorded independent symmetry-copy diagonal variance; calibration covariance not inferred"]}
    destination = output / "sequoia-raw-regression.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("SEQUOIA_REGRESSION_COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
