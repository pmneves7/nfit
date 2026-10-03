"""Read-only DGS histogram timing with real MDE events and trajectory exposure.

Use separate fresh processes for baseline and candidate source trees. This
manual benchmark calls no Mantid and saves scalar receipts only. It includes
all event reads, run selection, normalization and finalization; it does not
measure raw-event reconstruction or project archive persistence. Subsequent
repeats reuse metadata/geometry and operating-system caches, but still stream
all events. Original scientific files are never written.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import time
from pathlib import Path

import numpy as np

from nfit import mdevent_dataset_group
from nfit._parallel import num_threads, thread_budget
from nfit.histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
)
from nfit.measurement_replay import replay_measurement_histogram
from nfit.performance import peak_process_memory_mib
from nfit.symmetry import SymmetrySpec, resolve_symmetry

OPERATIONS = (
    "x,y,z;y,z,x;z,x,y;-x,-y,-z;-y,-z,-x;-z,-x,-y;"
    "y,x,z;x,z,y;z,y,x;-y,-x,-z;-x,-z,-y;-z,-y,-x"
)


def digest(values):
    """Hash bounded array slices without copying an entire histogram."""
    array = np.asarray(values)
    result = hashlib.sha256()
    for block in np.array_split(array.ravel(), max(1, array.size // 100_000)):
        result.update(np.ascontiguousarray(block).tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--instrument", choices=("hyspec", "sequoia"), default="hyspec")
    parser.add_argument("--runs", type=int, default=0, help="0 selects all; otherwise evenly spaced runs")
    parser.add_argument("--copies", type=int, choices=(1, 6, 12), default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--profile", action="store_true", help="Adds cProfile overhead equally to each repeat")
    parser.add_argument("--compare-fallback", action="store_true",
                        help="Also rerun with the fused dispatcher disabled and compare every cell")
    args = parser.parse_args()
    original_stat = (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    stages = []

    def progress(state):
        stage = state.get("stage", "unknown")
        if not stages or stages[-1][1] != stage:
            stages.append((time.perf_counter(), stage))
            print(json.dumps({"label": args.label, "stage": stage,
                              "message": state.get("message")}), flush=True)

    started = time.perf_counter()
    group = mdevent_dataset_group(args.source, progress_callback=progress)
    inspection_seconds = time.perf_counter() - started
    runs = group.datasets
    if 0 < args.runs < len(runs):
        runs = [runs[i] for i in np.linspace(0, len(runs)-1, args.runs, dtype=int)]
    if args.instrument == "hyspec":
        basis = [[1, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, 1]]
        limits = ((-2., 2., 80), (-1., 1., 20), (-.5, .62, 8), (-12., 12., 48))
    else:
        basis = [[1, 1, 1, 0], [1, -1, 0, 0], [1, 1, -2, 0], [0, 0, 0, 1]]
        limits = ((-1., 1., 67), (-2., 2., 134), (-1., 1., 101), (0., 50., 100))
    operations = resolve_symmetry(SymmetrySpec("operations", OPERATIONS))
    recipe = dict(lower=[item[0] for item in limits], upper=[item[1] for item in limits],
                  num_bins=[item[2] for item in limits], vectors=basis, datasets=runs,
                  symmetry_operations=[op.matrix_hkl for op in operations[:args.copies]],
                  progress_callback=progress)
    results = []
    with thread_budget(args.workers):
        for repeat in range(args.repeat):
            stages.clear()
            profiler = cProfile.Profile()
            started = time.perf_counter()
            if args.profile:
                profiler.enable()
            histogram = replay_measurement_histogram(group, **recipe)
            if args.profile:
                profiler.disable()
            stopped = time.perf_counter()
            totals = {}
            transitions = [(started, "setup"), *stages, (stopped, "complete")]
            for (left, stage), (right, _) in zip(transitions, transitions[1:], strict=False):
                totals[stage] = totals.get(stage, 0.) + right-left
            channels = histogram.auxiliary_channels
            names = (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR)
            outputs = {name: digest(channels[name].values) for name in names}
            outputs.update(signal=digest(histogram.signal), errors=digest(histogram.errors),
                           mask=digest(histogram.mask), num_events=digest(histogram.num_events))
            result = {"repeat": repeat, "total_seconds": stopped-started,
                      "first_run_in_process": repeat == 0, "stage_seconds": totals,
                      "shape": histogram.shape, "output_sha256": outputs,
                      "covered_bins": int(np.count_nonzero(~histogram.mask)),
                      "event_contributions": float(np.sum(histogram.num_events)),
                      "peak_process_memory_mib": peak_process_memory_mib(),
                      "cpu_budget": num_threads()}
            if args.profile:
                hot = sorted(pstats.Stats(profiler).stats.items(), key=lambda item: item[1][3], reverse=True)[:25]
                result["hot_functions"] = [{"file": key[0], "line": key[1], "function": key[2],
                                            "calls": value[1], "self_seconds": value[2],
                                            "cumulative_seconds": value[3]} for key, value in hot]
            results.append(result)
            print(json.dumps(result), flush=True)
            if not args.compare_fallback or repeat != args.repeat-1:
                del histogram
        parity = None
        if args.compare_fallback:
            from nfit import dgs_event_accumulation
            compiled = dgs_event_accumulation._compiled
            try:
                dgs_event_accumulation._compiled = None
                reference = replay_measurement_histogram(group, **recipe)
            finally:
                dgs_event_accumulation._compiled = compiled
            parity = {}
            for name in (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR):
                expected = reference.auxiliary_channels[name].values
                actual = histogram.auxiliary_channels[name].values
                assert np.array_equal(expected, actual), name
                parity[name] = {"literal": True}
            for name in ("num_events", "mask"):
                assert np.array_equal(getattr(reference, name), getattr(histogram, name)), name
                parity[name] = {"literal": True}
            n = reference.auxiliary_channels[NORMALIZATION_DENOMINATOR].values
            covered = ~reference.mask
            fringe = covered & (n <= np.quantile(n[covered], .1))
            for name, expected, actual in (
                ("exposure", n, histogram.auxiliary_channels[NORMALIZATION_DENOMINATOR].values),
                ("signal", reference.signal, histogram.signal),
                ("uncertainty", reference.errors, histogram.errors),
            ):
                assert np.allclose(expected, actual, rtol=1e-12, atol=0., equal_nan=True), name
                valid = np.isfinite(expected) & (expected != 0)
                delta = np.zeros(expected.shape)
                np.divide(np.abs(actual-expected), np.abs(expected), out=delta, where=valid)
                parity[name] = {"max_relative_nonzero": float(np.max(delta[valid], initial=0.)),
                                "lowest_exposure_decile_max_relative": float(np.max(delta[valid & fringe], initial=0.))}
    assert original_stat == (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    receipt = {"label": args.label, "source": str(args.source), "original_input_unchanged": True,
               "source_size_bytes": original_stat[0], "source_mtime_ns": original_stat[1],
               "source_events": group.metadata["mdevent"]["event_count"],
               "selected_run_numbers": [run.metadata["run_number"] for run in runs],
               "inspection_seconds": inspection_seconds, "copies": args.copies,
               "grid": limits, "basis": basis, "recorded_policies": {
                   key: value for key, value in group.metadata["mdevent"].items() if key.endswith("policy")},
               "nfit_module": __import__("nfit").__file__, "python": platform.python_version(),
               "numpy": np.__version__, "platform": platform.platform(), "results": results,
               "timing_scope": "MDE stream + histogram + trajectory normalization; no raw reduction or archive I/O",
               "profiled": args.profile, "fused_vs_fallback_parity": parity,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
