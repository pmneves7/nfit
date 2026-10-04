"""Matched float64 event-kernel and optional saved-MDE user-workflow benchmark.

No Mantid dependency. Source files are read only. Outputs must be new paths;
cluster outputs belong inside an IPTS directory. Initial JIT is recorded
separately. The old path is selected only by disabling the fused dispatcher in
this manual harness; both paths use the same scientific settings.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from nfit import dgs_event_accumulation
from nfit.dgs_reduction_policy import prepare_dgs_event_projector
from nfit.mdevent import _accumulate_discrete_event_coordinates
from nfit.symmetry import SymmetrySpec, resolve_symmetry

SYMMETRY = "x,y,z;y,z,x;z,x,y;y,x,z;x,z,y;z,y,x;-x,-y,-z;-y,-z,-x;-z,-x,-y;-y,-x,-z;-x,-z,-y;-z,-y,-x"
BASIS = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, 1]])
EDGES = tuple(np.linspace(lo, hi, bins+1) for lo, hi, bins in
              ((-.5, 1.5, 40), (-2.5, 2.5, 50), (-.5, .5, 25), (-14.25, 14.25, 57)))


@contextlib.contextmanager
def implementation(mode):
    previous = dgs_event_accumulation._compiled_high_precision
    if mode == "reference":
        dgs_event_accumulation._compiled_high_precision = None
    try:
        yield
    finally:
        dgs_event_accumulation._compiled_high_precision = previous


def fingerprint(source):
    stat = source.stat()
    return dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def kernel(source, rows, repeats):
    start = time.perf_counter()
    with h5py.File(source, "r") as handle:
        workspace = handle["MDEventWorkspace"]
        events = np.asarray(workspace["event_data/event_data"][:rows], dtype=float)
        ub = workspace["experiment0/sample/oriented_lattice/orientation_matrix"][()]
    load_seconds = time.perf_counter() - start
    q, energy, weights, variances = events[:, 5:8], events[:, 8], events[:, 0], events[:, 1]
    operations = resolve_symmetry(SymmetrySpec("operations", SYMMETRY))
    projectors = [prepare_dgs_event_projector(ub, BASIS, operation.matrix_hkl, EDGES, "high_precision")
                  for operation in operations]
    shape = tuple(len(edge)-1 for edge in EDGES)
    outputs = [np.zeros(shape) for _ in range(3)]
    def run(mode, copies, stop=None):
        with implementation(mode):
            selection = slice(None, stop)
            for projector in projectors[:copies]:
                dgs_event_accumulation.accumulate_projected_dgs_events(
                    projector, q[selection], energy[selection], weights[selection], variances[selection],
                    shape, *outputs, fallback_accumulator=_accumulate_discrete_event_coordinates)
    warm = {}
    for mode in ("reference", "fused"):
        start = time.perf_counter()
        run(mode, 1, 2)
        warm[mode] = time.perf_counter()-start
    results = []
    for copies in (1, 6, 12):
        times = {mode: [] for mode in warm}
        for repeat in range(repeats):
            for mode in (("reference", "fused") if repeat % 2 == 0 else ("fused", "reference")):
                for array in outputs:
                    array.fill(0)
                start = time.perf_counter()
                run(mode, copies)
                times[mode].append(time.perf_counter()-start)
        summaries = {}
        reference = None
        comparisons = {}
        for mode in warm:
            for array in outputs:
                array.fill(0)
            tracemalloc.start()
            run(mode, copies)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            summaries[mode] = dict(seconds=times[mode], median_seconds=float(np.median(times[mode])),
                                   extra_tracked_peak_bytes=peak, accepted_copies=float(outputs[2].sum()))
            if mode == "reference":
                reference = [array.copy() for array in outputs]
            else:
                for name, old, new in zip(("numerator", "variance", "counts"), reference, outputs, strict=True):
                    comparisons[name] = dict(different_cells=int(np.count_nonzero(old != new)),
                                             max_absolute_difference=float(np.max(np.abs(old-new))))
                    np.testing.assert_array_equal(new, old)
        results.append(dict(copies=copies, implementations=summaries, comparison=comparisons,
                            speedup=summaries["reference"]["median_seconds"]/summaries["fused"]["median_seconds"]))
    return dict(rows=len(events), shape=shape, input_loading_seconds=load_seconds,
                initial_two_event_compile_or_cache_load_seconds=warm, results=results,
                input_bytes=events.nbytes, common_histogram_bytes=sum(array.nbytes for array in outputs),
                scope="Warm event projection/accumulation only; excludes trajectories, source loading, save, clearing. "
                      "Tracked peak is temporary Python/NumPy allocation, not total RSS.")


def workflow(source, root, *, reverse=False):
    from nfit import (
        NfitProject,
        load_project,
        mdevent_dataset_group,
        refresh_composite_dataset,
        save_project,
    )
    from nfit.data_workspace import project_data_workspace
    from nfit.pipeline import DataGroup
    from nfit.project_composites import data_group_composite_config
    from nfit.symmetry import symmetry_config

    results, hashes = {}, {}
    comparison = {}
    for mode in (("fused", "reference") if reverse else ("reference", "fused")):
        destination = root / (mode+".nfit")
        phases = {}
        def timed(label, action, phases=phases):
            start = time.perf_counter()
            result = action()
            phases[label] = time.perf_counter()-start
            return result
        with implementation(mode):
            started = time.perf_counter()
            group = timed("import_metadata", lambda: mdevent_dataset_group(source, event_precision_policy="high_precision"))
            project = NfitProject([DataGroup(mode, subgroups=[group])], settings={"cache_binnings": True})
            config = data_group_composite_config(group)
            config.update(enabled=True, normalize=True, auto_rebin=False, minimum_samples=0,
                          minimum_coverage=0, symmetry=symmetry_config(SymmetrySpec("operations", SYMMETRY)))
            config["axes"] = [dict(name=name, variable=name, vector=vector.tolist(), lower=float(edge[0]),
                upper=float(edge[-1]), num_bins=len(edge)-1, bin_edges=edge.tolist(), mode="edges", fractional=False,
                auto_lower=False, auto_upper=False, auto_step_size=False) for name, vector, edge in
                zip(("H", "K", "L", "E"), BASIS, EDGES, strict=True)]
            with project_data_workspace(project, destination):
                histogram = timed("load_events_trajectory_normalization_and_bin_12_copies",
                    lambda project=project, group=group: refresh_composite_dataset(project.data_groups[0], node=group))
                timed("save", lambda project=project, destination=destination: save_project(project, destination))
            reopened = timed("reopen_metadata", lambda destination=destination: load_project(destination))
            viewed = timed("load_cached_histogram", lambda reopened=reopened: refresh_composite_dataset(
                reopened.data_groups[0], node=reopened.data_groups[0].subgroups[0]))
            elapsed = time.perf_counter()-started
        arrays = dict(signal=viewed.signal, errors=viewed.errors, counts=viewed.num_events, mask=viewed.mask)
        arrays.update({name: viewed.auxiliary_channels[key].values for name, key in
            (("numerator", "event_signal_numerator"), ("variance", "event_variance_numerator"),
             ("exposure", "normalization_denominator"))})
        hashes[mode] = {name: hashlib.sha256(np.asarray(array).tobytes()).hexdigest() for name, array in arrays.items()}
        comparison[mode] = {name: np.array(array) for name, array in arrays.items()}
        np.testing.assert_array_equal(viewed.signal, histogram.signal)
        results[mode] = dict(phases=phases, total_seconds=elapsed, saved_bytes=destination.stat().st_size)
    reference_arrays, fused_arrays = comparison["reference"], comparison["fused"]
    comparison = {}
    for name, old in reference_arrays.items():
        new = fused_arrays[name]
        if name in ("numerator", "variance", "counts", "mask"):
            np.testing.assert_array_equal(new, old)
        else:
            np.testing.assert_allclose(new, old, rtol=1e-12, atol=0, equal_nan=True)
        valid = np.isfinite(old) & np.isfinite(new)
        difference = np.abs(np.asarray(old[valid], dtype=float)-np.asarray(new[valid], dtype=float))
        scale = np.maximum(np.abs(old[valid]), np.abs(new[valid]))
        relative = np.divide(difference, scale, out=np.zeros_like(difference), where=scale>0)
        comparison[name] = dict(different_cells=int(np.count_nonzero(difference)),
            max_absolute_difference=float(np.max(difference)), max_relative_difference=float(np.max(relative)))
    return dict(results=results, array_hashes=hashes, comparison=comparison, speedup=results["reference"]["total_seconds"]/results["fused"]["total_seconds"],
                scope="One complete real saved-MDE input, 12 symmetry copies, native histogram save/reopen/view. "
                      "No raw reduction. Filesystem cache is uncontrolled.", order=list(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=600000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--workflow", action="store_true")
    parser.add_argument("--preset", choices=("hyspec", "sequoia"), default="hyspec")
    parser.add_argument("--reverse-workflow-order", action="store_true")
    args = parser.parse_args()
    global BASIS, EDGES
    if args.preset == "sequoia":
        BASIS = np.array([[1, 1, 1, 0], [1, -1, 0, 0], [1, 1, -2, 0], [0, 0, 0, 1]])
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from benchmark_dgs_settings import center_edges
        EDGES = tuple(np.asarray(center_edges(lo, hi, step)) for lo, hi, step in
                      ((-1., 1., .03), (-2., 2., .03), (-1., 1., .02), (0., 50., .5)))
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.rows < 1 or args.repeats < 1:
        raise ValueError("Positive rows and repeats required")
    output.mkdir(parents=True)
    before = fingerprint(source)
    with threadpool_limits(limits=1, user_api="blas"):
        result = dict(platform=platform.platform(), source=before, preset=args.preset, numpy=np.__version__,
                      kernel=kernel(source, args.rows, args.repeats))
        (output / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
        if args.workflow:
            result["workflow"] = workflow(source, output, reverse=args.reverse_workflow_order)
    assert fingerprint(source) == before
    result["source_unchanged"] = True
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
