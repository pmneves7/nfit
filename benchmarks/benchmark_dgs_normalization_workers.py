"""Manual MDE metadata-only normalization worker sweep; no event reads.

Use the desktop --run-script entry point with NFIT_DGS_WORKER_SWEEP_ARGS
containing a JSON argv list. NFIT_DGS_BENCHMARK_SOURCE selects staged source
through the existing native benchmark helper. Run only after other timed jobs.
All output/temp storage belongs to a NEW directory under IPTS shared/nfit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import marshal
import os
import platform
import re
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True,
                        help="Directory containing the existing native benchmark helpers")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--copies", choices=(1, 6, 12), type=int, default=12)
    argv = json.loads(os.environ["NFIT_DGS_WORKER_SWEEP_ARGS"]) if "NFIT_DGS_WORKER_SWEEP_ARGS" in os.environ else None
    return parser.parse_args(argv)


def ipts(path):
    path = Path(path).expanduser().resolve()
    found = [i for i, part in enumerate(path.parts) if re.fullmatch(r"IPTS-\d+", part)]
    if len(found) != 1:
        raise ValueError(f"Scientific storage must belong to one IPTS: {path}")
    return Path(*path.parts[:found[0] + 1])


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    with path.open("rb") as stream:
        prefix = stream.read(65536)
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                prefix_bytes=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest())


def code_identity(name):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if path.is_file():
        payload, kind = path.read_bytes(), "file"
    else:
        payload, kind = marshal.dumps(module.__loader__.get_code(name)), "frozen_code"
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), hash_kind=kind)


def array_hash(value):
    import numpy as np
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    for block in np.array_split(array.ravel(), max(1, (array.size + 249999) // 250000)):
        digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def compare(actual, reference):
    import numpy as np
    left, right = actual.ravel(), reference.ravel()
    maximum = relative = 0.0
    for start in range(0, left.size, 250000):
        a, b = left[start:start + 250000], right[start:start + 250000]
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError("Exposure must remain finite")
        if not np.array_equal(a > 0.0, b > 0.0):
            raise AssertionError("Exposure support differs")
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)
        delta = np.abs(a - b)
        maximum = max(maximum, float(np.max(delta, initial=0.0)))
        active = b != 0.0
        relative = max(relative, float(np.max(delta[active] / np.abs(b[active]), initial=0.0)))
    return dict(all_cells_pass=True, support_identical=True, rtol=1e-12, atol=0.0,
                maximum_absolute_difference=maximum, maximum_relative_nonzero=relative)


def main():
    args = arguments()
    sys.path.insert(0, str(args.helpers.resolve()))
    import benchmark_dgs_nfit_workflow as native
    from benchmark_dgs_settings import load_settings, output_directory

    settings = load_settings(args.config)
    if settings["preset"] != "sequoia" or len(settings["run_numbers"]) != 617:
        raise ValueError("This manual sweep requires the exact full617 SEQUOIA settings")
    output = output_directory(settings, "nfit", args.tag)
    root = ipts(output)
    inputs = [args.source, args.config, settings["normalization_file"],
              settings["mask_file"], settings["ub_source"]]
    if any(ipts(path) != root for path in inputs if path is not None):
        raise ValueError("Scientific input/config and output must belong to the same IPTS")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = tempfile.tempdir = str(output)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    os.environ["NFIT_PERFORMANCE_FILE"] = str(output / "performance.json")
    os.environ["NUMBA_CACHE_DIR"] = str(output / "numba-cache")
    runtime = native._select_source_tree()
    import h5py
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    from nfit._parallel import num_threads, thread_budget
    from nfit.app_distribution import application_version
    from nfit.dgs_reduction_policy import dgs_histogram_edges, dgs_uses_mantid_trajectory_grid
    from nfit.dgs_trajectory_tasks import pool_trajectory_tasks
    from nfit.mdevent import (
        _trajectory_normalization_from_payloads,
        _trajectory_payloads,
        mdevent_dataset_group,
    )
    from nfit.performance import load_resource_limits, peak_process_memory_mib, save_resource_limits
    from nfit.symmetry import SymmetrySpec, resolve_symmetry

    cpu = min(64, int(settings["threads"]))
    ram = min(200000, int(settings["ram_limit_mib"]))
    save_resource_limits(cpu_limit=cpu, ram_limit_mb=ram)
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in inputs if p is not None))
    before = {p: source_identity(p) for p in paths}
    receipt = dict(status="running", runtime=runtime, nfit_version=application_version(),
                   version_label_note="nfit_version is the reported runtime/build version; staged implementation is identified by complete module hashes",
                   host=socket.gethostname(), python=sys.version, platform=platform.platform(),
                   resource_limits=load_resource_limits(), resolved_cpu_budget=num_threads(),
                   copies=args.copies, trial_order=[32, 64, 64, 32], results=[],
                   scope="Metadata preparation once; normalization-only timed trials; no event reads or project writes",
                   sources_before=before, cache_state="OS/filesystem caches uncontrolled",
                   helper_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__).resolve(), args.helpers / "benchmark_dgs_settings.py",
                                  args.helpers / "benchmark_dgs_nfit_workflow.py")},
                   modules={name: code_identity(name) for name in
                            ("nfit", "nfit.mdevent", "nfit._mdevent_numba", "nfit.dgs_trajectory_tasks",
                             "nfit.dgs_reduction_policy", "nfit.dgs_normalization", "nfit.reduction_runtime",
                             "nfit.mdevent_detector_masks", "nfit._parallel", "nfit.performance")})

    def flush():
        temporary = output / "receipt.pending.json"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output / "receipt.json")

    flush()
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            started = time.perf_counter()
            group = mdevent_dataset_group(args.source, normalization_path=settings["normalization_file"],
                                          mask_path=settings["mask_file"])
            lookup = {int(run.metadata["run_number"]): run for run in group.datasets}
            if len(lookup) != 617 or set(lookup) != set(settings["run_numbers"]):
                raise AssertionError("Merged source differs from the complete617 membership")
            runs = [lookup[number] for number in settings["run_numbers"]]
            with h5py.File(settings["ub_source"], "r") as handle:
                ub = np.asarray(handle["MDEventWorkspace/experiment0/sample/oriented_lattice/orientation_matrix"])
            if not np.array_equal(np.asarray(group.metadata["mdevent"]["ub_matrix"]), ub):
                raise AssertionError("Merged source UB differs from the shared reference")
            basis = np.asarray(native._vectors(settings), dtype=float)
            operations = resolve_symmetry(SymmetrySpec("operations", settings["symmetry_operations"][str(args.copies)]))
            detectors, payloads = _trajectory_payloads(group, runs, np.linalg.inv(basis),
                                                      [op.matrix_hkl for op in operations[:args.copies]])
            requested = tuple(np.asarray(edge, dtype=float) for edge in settings["grid"]["bin_edges"])
            edges = dgs_histogram_edges(requested, "mantid")
            precision = dgs_uses_mantid_trajectory_grid(requested, "mantid")
            shape = tuple(len(edge) - 1 for edge in edges)
            if shape != (67, 134, 101, 101) or not precision:
                raise AssertionError("Unexpected shared full617 grid/convention")
            receipt.update(metadata_preparation_seconds=time.perf_counter() - started, shape=shape,
                           run_numbers=settings["run_numbers"], unique_geometries=len(detectors),
                           prepared_tasks=len(payloads), detector_payload_hashes=[[array_hash(a) for a in row] for row in detectors],
                           run_payload_hashes=[array_hash(np.asarray([row[i] for row in payloads])) for i in range(5)],
                           requested_edges_sha256=[array_hash(edge) for edge in requested],
                           effective_edges_sha256=[array_hash(edge) for edge in edges],
                           mantid_trajectory_precision=precision, threadpools=threadpool_info(),
                           pooled_tasks=sum(len(pool_trajectory_tasks([row for row in payloads if row[4] == i],
                                                                     edges[3], checked_shared_geometry=True))
                                            for i in range(len(detectors))))
            warm = tuple(np.linspace(edge[0], edge[-1], 3) for edge in edges)
            with thread_budget(64):
                _trajectory_normalization_from_payloads(detectors, payloads[:1], warm, (2, 2, 2, 2),
                                                        mantid_precision=precision)
            flush()
            reference = None
            for index, workers in enumerate((32, 64, 64, 32)):
                gc.collect()
                state = {}

                def progress(event, state=state):
                    if "workers" in event:
                        state.update(workers=int(event["workers"]), tasks=int(event["total"]))

                with thread_budget(workers):
                    started = time.perf_counter()
                    result = _trajectory_normalization_from_payloads(detectors, payloads, edges, shape,
                        mantid_precision=precision, progress_callback=progress)
                    elapsed = time.perf_counter() - started
                if reference is None:
                    reference = result
                checks = compare(result, reference)
                trial = dict(index=index, requested_workers=workers, actual_workers=state["workers"],
                             pooled_detector_tasks=state["tasks"], seconds=elapsed, verification=checks,
                             exposure_sha256=array_hash(result), peak_process_memory_mib=peak_process_memory_mib())
                receipt["results"].append(trial)
                print(json.dumps(trial), flush=True)
                del result
                flush()
            medians = {str(w): statistics.median(t["seconds"] for t in receipt["results"]
                                                 if t["requested_workers"] == w) for w in (32, 64)}
            after = {p: source_identity(p) for p in paths}
            if before != after:
                raise AssertionError("Scientific/config inputs changed during the sweep")
            if {name: code_identity(name) for name in receipt["modules"]} != receipt["modules"]:
                raise AssertionError("Scientific modules changed during the sweep")
            if {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in receipt["helper_files"]} != receipt["helper_files"]:
                raise AssertionError("Benchmark helpers changed during the sweep")
            receipt.update(status="complete", sources_after=after, inputs_unchanged=True,
                           median_seconds=medians, speedup_32_over_64=medians["64"] / medians["32"])
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        flush()
        raise
    flush()


if __name__ == "__main__":
    main()
