"""Manual current-build SEQUOIA acceptance against stored Mantid references.

Run through the existing ORNL executable with ``--run-script``. Set
NFIT_ACCEPTANCE_OUTPUT to an IPTS directory for the aggregate JSON receipt.
NFIT_ACCEPTANCE_SOURCE may select an existing source tree in the bundled runtime.
No Mantid import, project save, or large numerical output is performed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

# The frozen application can have imported its embedded package before dispatch.
if source := os.environ.get("NFIT_ACCEPTANCE_SOURCE"):
    for module_name in list(sys.modules):
        if module_name == "nfit" or module_name.startswith("nfit."):
            del sys.modules[module_name]
    sys.path.insert(0, source)

import h5py
import numpy as np

from nfit import (
    MDHistoSliceViewer,
    export_reduction_recipe,
    histogram_box_profiles,
    mdevent_dataset_group,
    raw_dgs_dataset_group,
    replay_measurement_histogram,
    replay_reduction_recipe,
)
from nfit.histogram_statistics import selected_event_statistics
from nfit.symmetry import SymmetrySpec, resolve_symmetry

ROOT = Path("/SNS/users/paulneves/data/SNS/SEQ/IPTS-37189")
REFERENCE = ROOT / "shared/nfit/.nfit-diagnostics/archive/converter-1d/parity-final"
PROJECTS = [ROOT / "shared/nfit" / name for name in (
    "SEQUOIA_Ei60_NiO_and_sapphire.nfit",
    "SEQUOIA_Ei60_NiO_minus_Al2O3_binning_backgrounds.nfit",
)]
SYMMETRY = "x,y,z;y,z,x;z,x,y;y,x,z;x,z,y;z,y,x;-x,-y,-z;-y,-z,-x;-z,-x,-y;-y,-x,-z;-x,-z,-y;-z,-y,-x"


def arrays(data):
    c, v, n = selected_event_statistics(data)
    return {"numerator": c, "variance": v, "norm": n, "counts": data.num_events}


def difference(actual, reference, selected=None):
    if selected is not None:
        actual, reference = actual[selected], reference[selected]
    finite = np.isfinite(actual) & np.isfinite(reference)
    a, b = actual[finite], reference[finite]
    nonzero = b != 0
    return {
        "cells": int(a.size), "exact": bool(np.array_equal(a, b)),
        "max_absolute": float(np.max(abs(a-b), initial=0)),
        "max_relative": float(np.max(abs((a[nonzero]-b[nonzero])/b[nonzero]), initial=0)),
        "finite_support_equal": bool(np.array_equal(np.isfinite(actual), np.isfinite(reference))),
    }


def final_cuts(data):
    model = MDHistoSliceViewer(data, x_dim=0, y_dim=3)
    for dim in (1, 2):
        model.integrate_checks[dim] = True
        model.selections[dim] = (float(data.axes[dim].centers[0]), float(data.axes[dim].centers[-1]))
    view = model.slice_arrays()
    c, v, n = selected_event_statistics(data)
    measured = (~data.mask) & (n > 0)
    c2, v2, n2 = (np.sum(np.where(measured, a, 0), axis=(1, 2)).T for a in (c, v, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(view["signal"], c2/n2, rtol=2e-14, equal_nan=True)
        np.testing.assert_allclose(view["errors"], np.sqrt(v2)/n2, rtol=2e-14, equal_nan=True)
    extents = (.13, .23, 3.5, 25.)
    profiles = histogram_box_profiles(view, view["signal"], view["errors"], extents)
    selected = profiles.selected & ~view["mask"]
    expected_c, expected_v, expected_n = (np.sum(a[selected]) for a in (c2, v2, n2))
    # Pool the returned profile using its retained additive fields, not means.
    cp, vp, np_ = selected_event_statistics(profiles.x_measurement.data)
    np.testing.assert_allclose([cp.sum(), vp.sum(), np_.sum()], [expected_c, expected_v, expected_n], rtol=2e-14)
    return {"view_pooling_verified": True, "box_cut_statistics_verified": True,
            "region": list(extents), "numerator": float(expected_c),
            "variance_numerator": float(expected_v), "exposure": float(expected_n),
            "signal": float(expected_c/expected_n), "uncertainty": float(np.sqrt(expected_v)/expected_n)}


def main():
    output = Path(os.environ["NFIT_ACCEPTANCE_OUTPUT"]).resolve()
    if "/IPTS-" not in str(output):
        raise ValueError("ORNL acceptance output must be inside an IPTS directory")
    output.mkdir(parents=True, exist_ok=True)
    before = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in PROJECTS}
    van = ROOT / "shared/tempMDE/van382236_D"
    group = mdevent_dataset_group(ROOT / "shared/tempMDE/All_Ei60_T5K_all_617_runs.nxs",
                                 normalization_path=van, mask_path=van)
    with h5py.File(REFERENCE / "mantid_mde_fine_3barm.h5") as handle:
        receipt = json.loads(handle.attrs["receipt"])
        reference = {key: np.asarray(handle[key]) for key in ("numerator", "variance", "norm", "counts")}
    edges = [np.linspace(lo, hi, bins+1) for lo, hi, bins in receipt["requested_edges"]]
    lattice = dict(a=4.17, b=4.17, c=4.17, alpha=90., beta=90., gamma=90.)
    operations = [op.matrix_hkl for op in resolve_symmetry(SymmetrySpec("operations", SYMMETRY), lattice_parameters=lattice)]
    options = dict(lower=[edge[0] for edge in edges], upper=[edge[-1] for edge in edges],
                   num_bins=[len(edge)-1 for edge in edges], bin_edges=edges,
                   vectors=[[1,1,1,0],[1,-1,0,0],[1,1,-2,0],[0,0,0,1]],
                   symmetry_operations=operations, max_batch_bytes=128*1024**2)
    start = time.perf_counter()
    data = replay_measurement_histogram(group, **options)
    seconds = time.perf_counter()-start
    actual = arrays(data)
    for key in ("numerator", "variance", "counts"):
        np.testing.assert_array_equal(actual[key], reference[key])
    covered = reference["norm"] > 0
    np.testing.assert_array_equal(actual["norm"] > 0, covered)
    np.testing.assert_allclose(actual["norm"][covered], reference["norm"][covered], rtol=3e-8, atol=0)
    low = covered & (reference["norm"] <= np.quantile(reference["norm"][covered], .1))
    from scipy.ndimage import binary_dilation
    fringe = covered & binary_dilation(~covered, iterations=2)
    cases = {label: {key: difference(actual[key], reference[key], region) for key in actual}
             for label, region in (("all", None), ("lowest_exposure_decile", low), ("coverage_fringe", fringe))}
    cuts = final_cuts(data)
    print("MDE_PARITY_COMPLETE", flush=True)
    paths = [Path(f"/SNS/SEQ/IPTS-37189/nexus/SEQ_{run}.nxs.h5") for run in (392985,393089,393387)]
    raw = raw_dgs_dataset_group(paths, normalization_path=van, mask_path=van)
    raw.metadata["raw_dgs"].update(ub_matrix=group.metadata["mdevent"]["ub_matrix"], cache_reduced_events=False)
    start = time.perf_counter()
    raw_data = replay_measurement_histogram(raw, **options)
    raw_seconds = time.perf_counter()-start
    # The small per-run diagnostic MDEs have no oriented lattice. Use the
    # authoritative merged reference, retaining its UB and selecting the same
    # three logical runs before streaming; no diagnostic file is modified.
    mde = copy.deepcopy(group)
    mde.datasets = [run for run in mde.datasets if int(run.metadata["run_number"]) in {392985,393089,393387}]
    assert len(mde.datasets) == 3
    mde_data = replay_measurement_histogram(mde, **options)
    ra, ma = arrays(raw_data), arrays(mde_data)
    for key in ("numerator", "variance", "counts"):
        np.testing.assert_array_equal(ra[key], ma[key])
    covered_three = ma["norm"] > 0
    np.testing.assert_allclose(ra["norm"][covered_three], ma["norm"][covered_three], rtol=3e-8, atol=0)
    replayed = replay_reduction_recipe(export_reduction_recipe(raw))
    replayed_data = replay_measurement_histogram(replayed, **options)
    replay_arrays = arrays(replayed_data)
    for key, value in replay_arrays.items():
        if key == "counts":
            np.testing.assert_array_equal(value, ra[key])
        else:
            # Batch/order changes can alter the final floating-point bits.
            np.testing.assert_allclose(value, ra[key], rtol=5e-15, atol=0)
    after = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in PROJECTS}
    assert before == after
    modules = {}
    for name in ("raw_dgs", "raw_dgs_monitors", "mdevent", "plotting_core", "measurement_profiles"):
        module = importlib.import_module("nfit."+name)
        modules[name] = {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    result = {"host": socket.gethostname(), "python": platform.python_version(), "numpy": np.__version__,
              "source_commit": os.environ.get("NFIT_ACCEPTANCE_COMMIT"), "source_modules": modules,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "full_mde_runs": len(group.datasets), "grid_shape": list(data.shape),
              "symmetry_operations": SYMMETRY, "policies": data.metadata.get("dgs_reduction_policies"),
              "stored_mantid_reference": str(REFERENCE / "mantid_mde_fine_3barm.h5"),
              "comparison": cases, "final_cuts": cuts, "mde_seconds": seconds,
              "fresh_raw_three_runs_seconds": raw_seconds,
              "fresh_raw_vs_stored_mantid_mde": {key: difference(ra[key], ma[key]) for key in ra},
              "raw_reduction_recipe_replay": {key: difference(replay_arrays[key], ra[key]) for key in ra},
              "raw_reduction_recipe_replay_verified": True, "original_projects_unchanged": before == after,
              "project_size_mtime": before,
              "timing_scope": "Acceptance timings only; concurrent jobs, OS cache and JIT uncontrolled, not a speed comparison",
              "limits": ["Mantid-compatible independent symmetry-copy diagonal variance; full cross-bin covariance not inferred",
                         "No calibration uncertainty in native DGS exposure", "Only stored Mantid references; no current Mantid process"]}
    destination = output / "sequoia-acceptance.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print("ACCEPTANCE_COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
