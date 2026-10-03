"""Bounded real-MDE projection/accumulation profile; saves scalar receipts only.

Run with the local nfit interpreter. Original inputs are read-only. This isolates
event kernels and does not measure raw reconstruction, trajectory normalization,
project/archive I/O or full histogram preparation.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import platform
import pstats
import time
from pathlib import Path

import h5py
import numpy as np

from nfit.dgs_event_accumulation import accumulate_projected_dgs_events
from nfit.dgs_reduction_policy import prepare_dgs_event_projector
from nfit.event_covariance import accumulate_copy_covariance
from nfit.mdevent import _accumulate_discrete_event_coordinates, _symmetry_matrices
from nfit.symmetry import SymmetrySpec, resolve_symmetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=600_000)
    args = parser.parse_args()
    before = (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    started = time.perf_counter()
    with h5py.File(args.source, "r") as handle:
        workspace = handle["MDEventWorkspace"]
        values = workspace["event_data/event_data"]
        rows = min(args.rows, len(values))
        events = np.asarray(values[:rows], dtype=float)
        ub = workspace["experiment0/sample/oriented_lattice/orientation_matrix"][()]
    read_seconds = time.perf_counter()-started
    q_sample, energy, weights, variances = events[:, 5:8], events[:, 8], events[:, 0], events[:, 1]
    basis = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, 1]])
    symmetry = "x,y,z;y,z,x;z,x,y;y,x,z;x,z,y;z,y,x;-x,-y,-z;-y,-z,-x;-z,-x,-y;-y,-x,-z;-x,-z,-y;-z,-y,-x"
    operations = _symmetry_matrices(operation.matrix_hkl for operation in resolve_symmetry(SymmetrySpec("operations", symmetry)))
    grids = {
        "slab": tuple(np.linspace(low, high, bins+1) for low, high, bins in
                      ((-.5, 1.5, 80), (-2.5, 2.5, 80), (-.03, .03, 1), (-14.25, 14.25, 57))),
        "cube": tuple(np.linspace(low, high, bins+1) for low, high, bins in
                      ((-.5, 1.5, 40), (-2.5, 2.5, 50), (-.5, .5, 25), (-14.25, 14.25, 57))),
    }
    results = []
    # Prepare once to exclude normal first-call JIT from the warm profiles.
    projector = prepare_dgs_event_projector(ub, basis, operations[0], grids["slab"])
    coordinates, edges = projector(q_sample[:2], energy[:2])
    dummy = [np.zeros((80, 80, 1, 57)) for _ in range(3)]
    first = time.perf_counter()
    _accumulate_discrete_event_coordinates(coordinates, weights[:2], variances[:2], edges,
        dummy[0].shape, *dummy, bin_indices=np.full(2, -1, dtype=np.int64))
    compilation_seconds = time.perf_counter()-first
    for label, grid in grids.items():
        shape = tuple(len(edge)-1 for edge in grid)
        for copies in (1, 6, 12):
            projectors = [prepare_dgs_event_projector(ub, basis, operation, grid) for operation in operations[:copies]]
            sums = [np.zeros(shape) for _ in range(3)]
            bins = np.full((copies, rows), -1, dtype=np.int64)
            profile = cProfile.Profile()
            projection_seconds = accumulation_seconds = 0.
            profile.enable()
            for copy, projector in enumerate(projectors):
                t = time.perf_counter()
                coordinates, edges = projector(q_sample, energy)
                projection_seconds += time.perf_counter()-t
                t = time.perf_counter()
                _accumulate_discrete_event_coordinates(coordinates, weights, variances,
                    edges, shape, *sums, bin_indices=bins[copy])
                accumulation_seconds += time.perf_counter()-t
            t = time.perf_counter()
            covariance = accumulate_copy_covariance(bins, variances, sums[1])
            covariance_seconds = time.perf_counter()-t
            profile.disable()
            statistics = pstats.Stats(profile)
            hot = sorted(statistics.stats.items(), key=lambda item: item[1][3], reverse=True)[:12]
            candidate = [np.zeros(shape) for _ in range(3)]
            candidate_bins = np.full_like(bins, -1)
            # Warm the exact strided real-event layout before matched repeats.
            accumulate_projected_dgs_events(projectors[0], q_sample[:2], energy[:2], weights[:2],
                variances[:2], shape, *candidate, fallback_accumulator=_accumulate_discrete_event_coordinates,
                bin_indices=candidate_bins[0, :2])
            timings = {"reference": [], "fused": []}
            for _ in range(5):
                for mode in ("reference", "fused"):
                    for output in candidate:
                        output.fill(0)
                    candidate_bins.fill(-1)
                    started = time.perf_counter()
                    for copy, projector in enumerate(projectors):
                        if mode == "reference":
                            coords, accumulation_edges = projector(q_sample, energy)
                            _accumulate_discrete_event_coordinates(coords, weights, variances,
                                accumulation_edges, shape, *candidate, bin_indices=candidate_bins[copy])
                        else:
                            accumulate_projected_dgs_events(projector, q_sample, energy, weights, variances,
                                shape, *candidate, fallback_accumulator=_accumulate_discrete_event_coordinates,
                                bin_indices=candidate_bins[copy])
                    timings[mode].append(time.perf_counter()-started)
                    np.testing.assert_array_equal(candidate_bins, bins)
                    correction = accumulate_copy_covariance(candidate_bins, variances, candidate[1])
                    assert correction == covariance
                    for actual, expected in zip(candidate, sums, strict=True):
                        np.testing.assert_array_equal(actual, expected)
            results.append({"grid": label, "shape": shape, "copies": copies,
                "projection_seconds": projection_seconds, "accumulation_seconds": accumulation_seconds,
                "covariance_seconds": covariance_seconds, "covariance": covariance,
                "accepted_contributions": int(np.sum(sums[2])),
                "reference_event_seconds": timings["reference"], "fused_event_seconds": timings["fused"],
                "reference_median_seconds": float(np.median(timings["reference"])),
                "fused_median_seconds": float(np.median(timings["fused"])),
                "kernel_speedup": float(np.median(timings["reference"])/np.median(timings["fused"])),
                "C_V_counts_assignments_covariance_literal": True,
                "hot_functions": [{"file": key[0], "line": key[1], "function": key[2],
                    "calls": value[1], "self_seconds": value[2], "cumulative_seconds": value[3]}
                    for key, value in hot]})
            print(json.dumps(results[-1]), flush=True)
    assert before == (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    receipt = {"source": str(args.source), "original_input_unchanged": True,
        "rows": rows, "read_seconds": read_seconds, "first_accumulator_seconds": compilation_seconds,
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": results,
        "source_modules": {}}
    for name in ("dgs_reduction_policy", "dgs_event_accumulation", "_dgs_event_numba", "mdevent"):
        module = importlib.import_module("nfit."+name)
        receipt["source_modules"][name] = {"path": module.__file__,
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
