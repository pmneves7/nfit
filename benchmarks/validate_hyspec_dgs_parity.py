"""Manual HYSPEC raw/MDE comparison with independent stored Mantid references.

Use the existing ORNL nfit executable with --run-script. NFIT_ACCEPTANCE_SOURCE
selects the current source tree; NFIT_ACCEPTANCE_OUTPUT must be an IPTS folder.
This script imports no Mantid, saves no project, and writes aggregate receipts.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

if source := os.environ.get("NFIT_ACCEPTANCE_SOURCE"):
    for module_name in list(sys.modules):
        if module_name == "nfit" or module_name.startswith("nfit."):
            del sys.modules[module_name]
    sys.path.insert(0, source)

import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from nfit import (
    MDHistoSliceViewer,
    mdevent_dataset_group,
    raw_dgs_dataset_group,
    replay_measurement_histogram,
)
from nfit.histogram_statistics import selected_event_statistics
from nfit.raw_dgs import (
    ENERGY_TO_K2,
    _combined_detector_mask,
    _detector_geometry,
    _energy_transfer_bounds,
    _goniometer,
    _iter_reduced_event_chunks,
    _masked_detector_geometry,
    inspect_raw_dgs_run,
)
from nfit.raw_dgs_hyspec import resolved_hyspec_preprocessing

ROOT = Path("/SNS/HYS/IPTS-36860")


def arrays(data):
    c, v, n = selected_event_statistics(data)
    return {"numerator": c, "variance": v, "norm": n, "counts": data.num_events}


def difference(actual, reference, selected=None):
    if selected is not None:
        actual, reference = actual[selected], reference[selected]
    finite = np.isfinite(actual) & np.isfinite(reference)
    a, b = actual[finite], reference[finite]
    nz = b != 0
    return {"cells": int(a.size), "exact": bool(np.array_equal(actual, reference, equal_nan=True)),
            "unequal_cells": int(np.count_nonzero(a != b)),
            "max_absolute": float(np.max(abs(a-b), initial=0)),
            "max_relative": float(np.max(abs((a[nz]-b[nz])/b[nz]), initial=0)),
            "finite_support_equal": bool(np.array_equal(np.isfinite(actual), np.isfinite(reference)))}


def comparisons(actual, reference):
    covered = reference["norm"] > 0
    low = covered & (reference["norm"] <= np.quantile(reference["norm"][covered], .1))
    fringe = covered & binary_dilation(~covered, iterations=2)
    result = {label: {key: difference(actual[key], reference[key], region) for key in actual}
              for label, region in (("all", None), ("lowest_exposure_decile", low), ("coverage_fringe", fringe))}
    result["coverage_support_equal"] = bool(np.array_equal(actual["norm"] > 0, covered))
    result["exposed_zero_count_cells"] = int(np.count_nonzero(covered & (actual["counts"] == 0)))
    result["low_exposure_cells"] = int(low.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        result["signal"] = difference(actual["numerator"]/actual["norm"], reference["numerator"]/reference["norm"])
        result["uncertainty"] = difference(np.sqrt(actual["variance"])/actual["norm"], np.sqrt(reference["variance"])/reference["norm"])
    return result


def final_cut(data, reference):
    viewer = MDHistoSliceViewer(data, x_dim=0, y_dim=3)
    for dim in (1, 2):
        viewer.integrate_checks[dim] = True
        viewer.selections[dim] = (float(data.axes[dim].centers[0]), float(data.axes[dim].centers[-1]))
    view = viewer.slice_arrays()
    covered = reference["norm"] > 0
    c, v, n = (np.sum(np.where(covered, reference[key], 0), axis=(1, 2)).T for key in ("numerator", "variance", "norm"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"signal": difference(view["signal"], c/n), "uncertainty": difference(view["errors"], np.sqrt(v)/n)}


def event_comparison(raw_group, reference, number):
    """Read bounded single-run events for a manual per-event precision audit.

    Detector identities are recovered from kinematics by nearest exact detector
    direction. Event multisets are ordered by detector and energy; repeated
    events are retained. This diagnostic never feeds production binning.
    """
    info = inspect_raw_dgs_run(ROOT/f"nexus/HYS_{number}.nxs.h5")
    config = raw_group.metadata["raw_dgs"]
    with h5py.File(info.path, "r") as handle:
        setup = resolved_hyspec_preprocessing(handle["entry"], config, info.incident_energy, instrument_name=info.instrument_name)
    geom = _detector_geometry(info.path, hyspec_preprocessing=setup)
    geom = _masked_detector_geometry(geom, None, _combined_detector_mask(config))
    started = time.perf_counter()
    chunks = _iter_reduced_event_chunks(info, config, geom, info.incident_energy,
        _energy_transfer_bounds(config, info.incident_energy), 32*1024**2, hyspec_preprocessing=setup)
    events = np.concatenate([chunk for chunk, _ in chunks])
    seconds = time.perf_counter()-started
    ki = np.sqrt(info.incident_energy/ENERGY_TO_K2)
    kf = np.sqrt((info.incident_energy-events[:, 3])/ENERGY_TO_K2)
    direction = np.column_stack((-events[:, 0], -events[:, 1], ki-events[:, 2]))/kf[:, None]
    distance, index = cKDTree(geom.positions/np.linalg.norm(geom.positions, axis=1)[:, None]).query(direction)
    ids = geom.detector_ids[index]
    gonio = _goniometer(info.omega, info.phi, info.chi)
    q_sample = np.zeros_like(events[:, :3])
    for axis in range(3):
        q_sample += events[:, axis, None]*gonio[axis]
    native = np.column_stack((ids, events[:, 3].astype(np.float32), events[:, 4:6].astype(np.float32), q_sample.astype(np.float32)))
    with h5py.File(reference, "r") as handle:
        # SaveMD's event_data is the flat nine-column MD event structure.
        raw = handle["MDEventWorkspace/event_data/event_data"][()]
    mantid = np.column_stack((raw[:, 4], raw[:, 8], raw[:, :2], raw[:, 5:8]))
    native = native[np.lexsort((native[:, 1], native[:, 0]))]
    mantid = mantid[np.lexsort((mantid[:, 1], mantid[:, 0]))]
    equal_size = native.shape == mantid.shape
    result = {"nfit_events": len(native), "mantid_events": len(mantid), "equal_event_count": equal_size,
        "reduction_seconds": seconds, "maximum_detector_direction_distance": float(distance.max()),
        "Ei_meV": info.incident_energy, "T0_us": info.t0, "L1_m": info.l1,
        "instrument_name": info.instrument_name, "geometry_signature": info.geometry_signature,
        "calibration_source": info.calibration_source, "calibration_warning": info.calibration_warning,
        "preprocessing": setup}
    keytype = np.dtype([("detector_id", "<i8"), ("energy", "<f4")])
    def event_keys(values):
        keys = np.empty(len(values), dtype=keytype)
        keys["detector_id"], keys["energy"] = values[:, 0], values[:, 1]
        return keys
    nk, mk = event_keys(native), event_keys(mantid)
    unique, ni, nc = np.unique(nk, return_index=True, return_counts=True)
    other, mi, mc = np.unique(mk, return_index=True, return_counts=True)
    common, nai, mai = np.intersect1d(unique, other, return_indices=True)
    result["common_detector_energy_keys"] = len(common)
    result["common_key_count_mismatches"] = int(np.count_nonzero(nc[nai] != mc[mai]))
    result["common_key_fields"] = {name: difference(native[ni[nai], column], mantid[mi[mai], column]) for column, name in enumerate(("detector_id", "energy", "signal", "variance", "Q_sample_x", "Q_sample_y", "Q_sample_z"))}
    native_only = np.setdiff1d(unique, other)
    mantid_only = np.setdiff1d(other, unique)
    result["native_only_keys"] = [[int(k["detector_id"]), float(k["energy"])] for k in native_only[:20]]
    result["mantid_only_keys"] = [[int(k["detector_id"]), float(k["energy"])] for k in mantid_only[:20]]
    result["native_only_events_detector_energy_signal_variance_Qsample"] = native[ni[np.isin(unique, native_only)]][:20].tolist()
    result["native_only_key_count"] = len(native_only)
    result["mantid_only_key_count"] = len(mantid_only)
    if equal_size:
        result["fields"] = {name: difference(native[:, column], mantid[:, column]) for column, name in enumerate(("detector_id", "energy", "signal", "variance", "Q_sample_x", "Q_sample_y", "Q_sample_z"))}
    return result


def main():
    output = Path(os.environ["NFIT_ACCEPTANCE_OUTPUT"]).resolve()
    if "/IPTS-" not in str(output):
        raise ValueError("ORNL acceptance outputs must be inside an IPTS directory")
    reference_root = output
    receipt = json.loads((reference_root/"mantid_reduction_receipt.json").read_text())
    result = {"python": platform.python_version(), "numpy": np.__version__, "reference": receipt,
              "source_modules": {}, "runs": [], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ("raw_dgs", "raw_dgs_geometry", "raw_dgs_hyspec", "mdevent", "mdevent_detector_masks", "dgs_reduction_policy"):
        module = importlib.import_module("nfit."+name)
        result["source_modules"][name] = {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    for item in receipt["runs"]:
        number = item["run"]
        mde_path = reference_root/f"mantid_mde_{number}.nxs"
        raw_path = ROOT/f"nexus/HYS_{number}.nxs.h5"
        before = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in (raw_path, mde_path)}
        with h5py.File(reference_root/f"mantid_hist_{number}.h5", "r") as handle:
            reference = {key: handle[key][()] for key in ("numerator", "variance", "counts", "norm")}
            options = json.loads(handle.attrs["binning"])
        options["bin_edges"] = [np.linspace(lo, hi, count+1) for lo, hi, count in zip(options["lower"], options["upper"], options["num_bins"], strict=True)]
        options["max_batch_bytes"] = 32*1024**2
        mde = mdevent_dataset_group(mde_path)
        raw_group = raw_dgs_dataset_group([raw_path], mask_path=reference_root/f"hyspec_tip_mask_{number}.nxs")
        raw_group.metadata["raw_dgs"].update(ub_matrix=receipt["UB"], bad_pulse_threshold=0., cache_reduced_events=False)
        started = time.perf_counter()
        mde_data = replay_measurement_histogram(mde, **options)
        mde_seconds = time.perf_counter()-started
        started = time.perf_counter()
        raw_data = replay_measurement_histogram(raw_group, **options)
        raw_seconds = time.perf_counter()-started
        with h5py.File(reference_root/f"mantid_hist_wide_{number}.h5", "r") as handle:
            wide_reference = {key: handle[key][()] for key in reference}
        record = {"run": number, "mde_histogram_seconds": mde_seconds, "raw_histogram_seconds": raw_seconds,
            "mde_ub": mde.metadata["mdevent"]["ub_matrix"], "raw_ub": raw_group.metadata["raw_dgs"]["ub_matrix"],
            "debug_peaks": {k: {"nfit": list(map(int, np.unravel_index(np.argmax(arrays(mde_data)[k]),mde_data.shape))), "mantid": list(map(int,np.unravel_index(np.argmax(reference[k]),mde_data.shape))), "nfit_sum": float(np.sum(arrays(mde_data)[k])), "mantid_sum": float(np.sum(reference[k]))} for k in reference},
            "mde_vs_mantid": comparisons(arrays(mde_data), reference),
            "raw_vs_mantid": comparisons(arrays(raw_data), reference),
            "raw_vs_mde": comparisons(arrays(raw_data), arrays(mde_data)),
            "raw_vs_mantid_physical_energy_bounds": comparisons(arrays(raw_data), wide_reference),
            "raw_final_cut_vs_mantid_physical_energy_bounds": final_cut(raw_data, wide_reference),
            "raw_final_cut_vs_mantid": final_cut(raw_data, reference),
            "events": event_comparison(raw_group, mde_path, number),
            "events_physical_energy_bounds": event_comparison(raw_group, reference_root/f"mantid_mde_wide_{number}.nxs", number)}
        for actual, expected in ((arrays(mde_data), reference), (arrays(raw_data), wide_reference)):
            for key in ("numerator", "variance", "counts"):
                np.testing.assert_array_equal(actual[key], expected[key])
            covered = expected["norm"] > 0
            np.testing.assert_array_equal(actual["norm"] > 0, covered)
            np.testing.assert_allclose(actual["norm"][covered], expected["norm"][covered], rtol=1e-9, atol=0)
        assert record["events_physical_energy_bounds"]["equal_event_count"]
        assert all(value["exact"] for value in record["events_physical_energy_bounds"]["fields"].values())
        assert all(value["max_relative"] < 1e-9 and value["finite_support_equal"]
                   for value in record["raw_final_cut_vs_mantid_physical_energy_bounds"].values())
        record["core_acceptance_assertions_passed"] = True
        after = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in (raw_path, mde_path)}
        assert before == after
        record["original_inputs_unchanged"] = True
        result["runs"].append(record)
        (output/"nfit_hyspec_parity_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print("HYSPEC_PARITY_RUN_COMPLETE", number, json.dumps(record), flush=True)
    print("HYSPEC_PARITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
