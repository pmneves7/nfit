"""Bounded manual raw-HYSPEC acceptance against an existing stored MDE.

Run with the existing ORNL nfit --run-script runtime. Output must be inside an
IPTS directory. No Mantid import, project save, or array export is performed.
An unsupported raw-conversion case is recorded rather than called parity.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import platform
import socket
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

if source := os.environ.get("NFIT_ACCEPTANCE_SOURCE"):
    for module_name in list(sys.modules):
        if module_name == "nfit" or module_name.startswith("nfit."):
            del sys.modules[module_name]
    sys.path.insert(0, source)

import h5py
import numpy as np

from nfit import mdevent_dataset_group, raw_dgs_dataset_group, replay_measurement_histogram
from nfit.histogram_statistics import selected_event_statistics
from nfit.raw_dgs import _detector_geometry, inspect_raw_dgs_run

ROOT = Path("/SNS/HYS/IPTS-36860")
RAW = [ROOT / "nexus" / f"HYS_{run}.nxs.h5" for run in (506277, 506278)]
MDE = ROOT / "shared/Ei15meV_50K_240Hz_s2_70.nxs"


def _scalar(group, key):
    array = np.asarray(group[key][()]).reshape(-1)
    value = array[0]
    return value.decode() if isinstance(value, bytes) else float(value)


def _geometry_evidence(path, experiment):
    with h5py.File(path, "r") as handle:
        entry = handle["entry"]
        root = ET.fromstring(entry["instrument/instrument_xml/data"][()].tobytes())
        xml = [ET.tostring(component, encoding="unicode") for component in root.findall("{*}component")
               if component.get("type") in {"moderator", "Tank"}]
        logs = {name: _scalar(entry, f"DASlogs/{name}/value")
                for name in ("msd", "s2", "s1", "omega", "chi", "phi")}
    geometry = _detector_geometry(path)
    with h5py.File(MDE, "r") as handle:
        detector = handle[f"MDEventWorkspace/experiment{experiment}/instrument/physical_detectors"]
        ids = np.asarray(detector["detector_number"], dtype=np.int64)
        polar = np.deg2rad(np.asarray(detector["polar_angle"], dtype=float))
        azimuth = np.deg2rad(np.asarray(detector["azimuthal_angle"], dtype=float))
        distance = np.asarray(detector["distance"], dtype=float)
    reference = np.column_stack((np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)))
    positions, _, valid = geometry.event_geometry_for_ids(ids)
    direction = positions[valid] / np.linalg.norm(positions[valid], axis=1)[:, None]
    angle = np.rad2deg(np.arccos(np.clip(np.sum(direction * reference[valid], axis=1), -1., 1.)))
    return {"dynamic_idf_components": xml, "logs": logs,
            "source_distance_from_recorded_idf_formula_m": 38.980 + .001 * logs["msd"],
            "matching_detector_ids": int(valid.sum()),
            "direction_difference_degrees": {"min": float(angle.min()), "max": float(angle.max()), "median": float(np.median(angle))},
            "distance_difference_max_m": float(np.max(abs(np.linalg.norm(positions[valid], axis=1) - distance[valid])))}


def _arrays(data):
    c, v, n = selected_event_statistics(data)
    return {"numerator": c, "variance": v, "exposure": n, "counts": data.num_events}


def _summary(data):
    result = {name: float(np.sum(array)) for name, array in _arrays(data).items()}
    result.update(shape=list(data.shape), measured_cells=int(np.count_nonzero(~data.mask)))
    return result


def _difference(actual, reference):
    finite = np.isfinite(actual) & np.isfinite(reference)
    a, b = actual[finite], reference[finite]
    selected = b != 0
    return {"exact": bool(np.array_equal(actual, reference)),
            "max_absolute": float(np.max(abs(a - b), initial=0)),
            "max_relative": float(np.max(abs((a[selected] - b[selected]) / b[selected]), initial=0)),
            "nonzero_support_equal": bool(np.array_equal(actual != 0, reference != 0))}


def main():
    output = Path(os.environ["NFIT_ACCEPTANCE_OUTPUT"]).resolve()
    if "/IPTS-" not in str(output):
        raise ValueError("Acceptance output must be inside an IPTS directory")
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"host": socket.gethostname(), "python": platform.python_version(), "numpy": np.__version__,
               "source_commit": os.environ.get("NFIT_ACCEPTANCE_COMMIT"), "raw_sources": [str(path) for path in RAW],
               "stored_mde": str(MDE), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "limits": ["Stored Mantid/Shiver MDE reference only; no current Mantid process",
                          "Only the first two logical runs, not the whole acquisition",
                          "Native DGS calibration covariance is not represented",
                          "Acceptance timings are not controlled performance comparisons"]}
    destination = output / "hyspec-raw-acceptance.json"

    def save():
        destination.write_text(json.dumps(receipt, indent=2) + "\n")

    receipt["source_modules"] = {}
    for name in ("raw_dgs", "raw_dgs_monitors", "mdevent"):
        module = importlib.import_module("nfit." + name)
        receipt["source_modules"][name] = {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    receipt["raw_metadata"] = []
    for experiment, path in enumerate(RAW):
        info = inspect_raw_dgs_run(path)
        receipt["raw_metadata"].append({"run_number": info.run_number, "event_count": info.event_count,
            "Ei_meV": info.incident_energy, "T0_us": info.t0, "L1_m": info.l1,
            "omega_degrees": info.omega, "charge_uAh": info.proton_charge,
            "calibration_source": info.calibration_source, "calibration_warning": info.calibration_warning,
            "geometry": _geometry_evidence(path, experiment)})
    receipt["reference_metadata"] = []
    with h5py.File(MDE, "r") as handle:
        workspace = handle["MDEventWorkspace"]
        for experiment in (0, 1):
            logs = workspace[f"experiment{experiment}/logs"]
            receipt["reference_metadata"].append({"run_number": _scalar(logs, "run_number/value"),
                "Ei_meV": _scalar(logs, "Ei/value"), "T0_us": _scalar(logs, "CalculatedT0/value"),
                "charge_uAh": _scalar(logs, "gd_prtn_chrg/value"),
                "goniometer_matrix": np.asarray(logs["goniometer/rotation_matrix"]).reshape(3, 3).tolist(),
                "energy_bounds_meV": [float(logs["MDNorm_low/value"][0]), float(logs["MDNorm_high/value"][0])]})
        config = ast.literal_eval(workspace["experiment0/logs/MDEConfig/value"][0].decode())
        receipt["stored_reduction_config"] = {key: value for key, value in config.items()
            if key not in {"filename", "output_dir"} and len(str(value)) < 2000}
    save()
    print("HYSPEC_METADATA_COMPLETE", flush=True)
    options = {"lower": [-15, -15, -15, -20], "upper": [15, 15, 15, 20],
               "num_bins": [120, 120, 1, 40], "max_batch_bytes": 128 * 1024**2}
    receipt["grid"] = options
    mde = mdevent_dataset_group(MDE)
    selected = [dataset for dataset in mde.datasets if dataset.metadata["run_number"] in {"506277", "506278"}]
    start = time.perf_counter()
    reference = replay_measurement_histogram(mde, datasets=selected, **options)
    receipt["mde_seconds"] = time.perf_counter() - start
    receipt["mde_histogram"] = _summary(reference)
    save()
    print("HYSPEC_MDE_COMPLETE", flush=True)
    raw = raw_dgs_dataset_group(RAW)
    raw.metadata["raw_dgs"].update(ub_matrix=mde.metadata["mdevent"]["ub_matrix"],
        incident_energy_override=receipt["reference_metadata"][0]["Ei_meV"],
        t0_override=receipt["reference_metadata"][0]["T0_us"], cache_reduced_events=False)
    start = time.perf_counter()
    try:
        actual = replay_measurement_histogram(raw, **options)
    except (ValueError, KeyError, NotImplementedError) as error:
        receipt["raw_reduction"] = {"status": "unsupported", "exception": type(error).__name__, "message": str(error)}
    else:
        receipt["raw_reduction"] = {"status": "completed", "histogram": _summary(actual),
            "vs_stored_mde": {key: _difference(value, _arrays(reference)[key]) for key, value in _arrays(actual).items()}}
    receipt["raw_seconds"] = time.perf_counter() - start
    save()
    print("HYSPEC_ACCEPTANCE_COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
