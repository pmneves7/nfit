"""Manual native DGS import/reduce/cache/save/reopen/rebin benchmark.

Run the installed desktop bundle with ``--run-script`` and set
``NFIT_DGS_BENCHMARK_CONFIG`` to a JSON configuration file. This script never
imports Mantid. All scientific storage stays below the configured IPTS benchmark
root. No original project is opened or edited. Timings intentionally include
native project serialization, unlike diagnostic array exports and comparisons.

The primary workflow creates a 3bar binning, saves its raw-event and histogram
caches, reopens the native project, accesses the saved histogram, creates a
3barm binning from saved events, and saves again. Raw reduction, writing the
reduced-event cache, and event accumulation are interleaved in nfit's public
workflow; their combined interval is reported honestly rather than invented as
three independently measured operations. Optional cached 1/6/12-copy timings follow the primary workflow and do not enter its headline time. The optional
cProfile flag profiles the primary workflow in a separate profiling job; those
measurements must not be used as ordinary timing results.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import marshal
import os
import platform
import socket
import sys
import time
import traceback
import zipfile
from pathlib import Path

# The desktop bootstrap replaces sys.argv; preserve a reproducible env route.
SCRIPT_ENTERED = time.perf_counter()


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset", default="sequoia")
    parser.add_argument("--tag")
    parser.add_argument("--first-runs", type=int)
    parser.add_argument("--runs", help="Comma-separated exact run numbers")
    parser.add_argument("--extra-cached-bins", action="store_true")
    parser.add_argument("--profile", action="store_true")
    argv = json.loads(os.environ["NFIT_DGS_BENCHMARK_ARGS"]) if "NFIT_DGS_BENCHMARK_ARGS" in os.environ else None
    args = parser.parse_args(argv)
    if args.config is None and os.environ.get("NFIT_DGS_BENCHMARK_CONFIG"):
        args.config = Path(os.environ["NFIT_DGS_BENCHMARK_CONFIG"])
    return args


def _plain(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.shape == ():
            return value.item()
        raise TypeError("Numerical arrays belong in diagnostic HDF, never in JSON receipts")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _bounded_metadata(value, *, key="", depth=0):
    """Describe numerical metadata without scanning or materializing arrays.

    Raw histograms retain their normalization array in metadata for compatibility.
    JSON is a scalar receipt, so even tiny arrays are descriptors here. Actual
    numerical values remain in the named diagnostic HDF channels.
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.shape == ():
            return value.item()
        result = {"numerical_payload": "omitted_from_scalar_receipt",
                  "shape": list(value.shape), "dtype": str(value.dtype)}
        if key == "normalization_denominator":
            result["diagnostic_dataset"] = "normalization"
        return result
    if depth > 20:
        return {"metadata_omitted": "depth_limit"}
    if isinstance(value, dict):
        return {str(name): _bounded_metadata(item, key=str(name), depth=depth+1)
                for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > 4096:
            return {"metadata_omitted": "length_limit", "length": len(value)}
        return [_bounded_metadata(item, key=key, depth=depth+1) for item in value]
    return value


def _harness_receipts():
    import benchmark_dgs_settings
    paths = (Path(__file__).resolve(), Path(benchmark_dgs_settings.__file__).resolve())
    return {path.name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths}


class Receipt:
    """Flush partial evidence after every stage, including failures."""

    def __init__(self, output, settings):
        self.output = output
        self.payload = {"status": "running", "engine": "nfit", "settings": settings,
                        "host": socket.gethostname(), "platform": platform.platform(),
                        "python": sys.version, "pid": os.getpid(), "stages": [],
                        "cache_state": "OS filesystem caches are uncontrolled"}
        self.stage = None
        self.transitions = []
        self.last_progress = 0.0
        self.flush()

    def flush(self):
        path = self.output / "receipt.json"
        staging = self.output / "receipt.pending.json"
        staging.write_text(json.dumps(self.payload, indent=2, default=_plain) + "\n")
        staging.replace(path)

    def emit(self, event):
        item = {"wall_time": time.time(), "elapsed_seconds": time.perf_counter() - SCRIPT_ENTERED,
                "operation": self.stage, **event}
        line = json.dumps(item, default=_plain)
        with (self.output / "progress.jsonl").open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def progress(self, state):
        now = time.perf_counter()
        name = str(state.get("stage", "unknown"))
        if name == "raw_dgs_normalization" and state.get("iteration") == state.get("total"):
            name = "histogram_finalize"
        changed = not self.transitions or self.transitions[-1][1] != name
        if changed:
            self.transitions.append((now, name))
        if changed or now - self.last_progress >= 15:
            self.emit(dict(state))
            self.last_progress = now

    @contextlib.contextmanager
    def timed(self, label):
        from nfit.performance import peak_process_memory_mib
        self.stage = label
        self.transitions = []
        start = time.perf_counter()
        self.emit({"event": "started"})
        try:
            yield
        finally:
            stop = time.perf_counter()
            totals = {}
            transitions = [(start, "setup"), *self.transitions, (stop, "complete")]
            for (left, name), (right, _) in zip(transitions, transitions[1:], strict=False):
                totals[name] = totals.get(name, 0.) + right - left
            result = {"label": label, "seconds": stop - start,
                      "progress_intervals_seconds": totals,
                      "peak_process_memory_mib": peak_process_memory_mib()}
            self.payload["stages"].append(result)
            self.emit({"event": "finished", **result})
            self.flush()
            self.stage = None


def _source_receipts(settings):
    paths = [*settings["raw_files"], settings["normalization_file"],
             settings["mask_file"], settings["ub_source"]]
    result = {}
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        stat = path.stat()
        result[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                             "ctime_ns": stat.st_ctime_ns}
    return result


def _module_receipts():
    names = ("nfit", "nfit.raw_dgs", "nfit.raw_dgs_cache", "nfit.raw_dgs_geometry",
             "nfit.raw_dgs_hyspec", "nfit.dgs_event_accumulation", "nfit._dgs_event_numba",
             "nfit.dgs_reduction_policy", "nfit.mdevent", "nfit._mdevent_numba",
             "nfit.project_composites", "nfit.project_rebinning", "nfit.project_io",
             "nfit.project_gui", "nfit.mapped_archive", "nfit.project_archive",
             "nfit.analysis.artifacts", "nfit.array_archive", "nfit.raw_dgs_monitors",
             "nfit.raw_dgs_pulses", "nfit.raw_dgs_geometry_precision")
    result = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            continue
        path = Path(module.__file__).resolve()
        if path.is_file():
            payload, kind = path.read_bytes(), "file"
        elif hasattr(module.__loader__, "get_code"):
            payload, kind = marshal.dumps(module.__loader__.get_code(name)), "frozen_code"
        else:
            raise RuntimeError(f"Cannot fingerprint installed module {name} at {path}")
        result[name] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                        "hash_kind": kind}
    return result


def _vectors(settings):
    """Extend the common Mantid momentum rows to nfit's four-dimensional basis."""
    vectors = settings["grid"]["vectors"]
    if len(vectors) == 3 and all(len(row) == 3 for row in vectors):
        return [[*row, 0] for row in vectors] + [[0, 0, 0, 1]]
    if len(vectors) != 4 or any(len(row) != 4 for row in vectors):
        raise ValueError("Expected three momentum vectors or a full 4D basis")
    return vectors


def _configure(group, settings, copies, binning_id=None):
    from nfit.project_composites import (
        data_group_composite_config,
        data_group_composite_config_by_id,
    )
    from nfit.symmetry import SymmetrySpec, symmetry_config
    config = (data_group_composite_config(group) if binning_id is None else
              data_group_composite_config_by_id(group, binning_id))
    names = settings["grid"].get("axis_names", ["[H,H,H]", "[K,-K,0]", "[L,L,-2L]", "DeltaE"])
    config.update(enabled=True, normalize=True, mean_weighting="uniform",
                  minimum_samples=0, minimum_coverage=0., auto_rebin=False,
                  coordinate_mode="hkle", max_batch_mb=int(settings.get("max_batch_mb", 192)),
                  symmetry=symmetry_config(SymmetrySpec("operations", settings["symmetry_operations"][str(copies)])),
                  stale=True)
    config["axes"] = [dict(name=name, variable=("H", "K", "L", "E")[i], vector=vector,
                           lower=edges[0], upper=edges[-1], num_bins=len(edges)-1,
                           step_size=edges[1]-edges[0], bin_edges=edges, mode="edges",
                           fractional=False, auto_lower=False, auto_upper=False,
                           auto_step_size=False)
                      for i, (name, edges, vector) in enumerate(zip(names, settings["grid"]["bin_edges"],
                                                                    _vectors(settings), strict=True))]
    return config


def _channels(histogram):
    from nfit.histogram_statistics import (
        EVENT_SIGNAL_NUMERATOR,
        EVENT_VARIANCE_NUMERATOR,
        NORMALIZATION_DENOMINATOR,
    )
    return {"C": histogram.auxiliary_channels[EVENT_SIGNAL_NUMERATOR].values,
            "V": histogram.auxiliary_channels[EVENT_VARIANCE_NUMERATOR].values,
            "N": histogram.auxiliary_channels[NORMALIZATION_DENOMINATOR].values,
            "I": histogram.signal, "sigma": histogram.errors,
            "counts": histogram.num_events, "mask": histogram.mask}


def _array_receipts(histogram):
    """Bounded full-array digests, exclusively outside scientific timers."""
    import numpy as np
    result = {}
    for name, array in _channels(histogram).items():
        flat = np.asarray(array).reshape(-1)
        digest = hashlib.sha256()
        finite, total = 0, 0.
        for start in range(0, flat.size, 100_000):
            chunk = flat[start:start+100_000]
            digest.update(chunk.tobytes())
            finite += int(np.count_nonzero(np.isfinite(chunk)))
            total += float(np.sum(chunk, where=np.isfinite(chunk), dtype=np.float64))
        result[name] = {"sha256": digest.hexdigest(), "shape": list(array.shape),
                        "dtype": str(array.dtype), "finite_count": finite, "finite_sum": total}
    return result


def _diagnostic_dump(histogram, path, settings):
    """Common full-array comparator input; not a native workflow save."""
    import h5py
    import numpy as np
    names = {"C": "numerator", "V": "variance", "N": "normalization",
             "I": "signal", "sigma": "errors", "counts": "counts", "mask": "mask"}
    with h5py.File(path, "w") as handle:
        handle.attrs["settings"] = json.dumps(settings, default=_plain)
        handle.attrs["storage_order"] = "nfit axis order; energy last"
        for name, array in _channels(histogram).items():
            target = handle.create_dataset(names[name], shape=array.shape, dtype=array.dtype)
            for i in range(array.shape[0]):
                target[i] = array[i]
        edges = handle.create_group("edges")
        for i, axis in enumerate(histogram.axes):
            if axis.values.size != histogram.shape[i] + 1:
                raise AssertionError("Diagnostic histogram does not retain its bin edges")
            edges.create_dataset(str(i), data=np.asarray(axis.values, dtype=np.float64))
    return str(path)


def _archive_receipt(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        return {"file": str(path), "bytes": path.stat().st_size,
                "member_count": len(members),
                "reduced_cache_bytes": sum(item.file_size for item in members if "reduced" in item.filename),
                "uncompressed_member_bytes": sum(item.file_size for item in members)}


def main():
    args = _arguments()
    # This module is stdlib only and is shared with the independent Mantid job.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_dgs_settings import load_settings, output_directory, select_runs
    settings = load_settings(args.config, preset=args.preset)
    explicit = [int(item) for item in args.runs.split(",")] if args.runs else None
    settings = select_runs(settings, run_numbers=explicit, first_runs=args.first_runs)
    tag = args.tag or settings.get("tag") or time.strftime("%Y%m%d-%H%M%S")
    output = output_directory(settings, "nfit", tag)
    output.mkdir(parents=True, exist_ok=False)
    # Scientific temporary files and isolated performance settings stay in IPTS.
    os.environ["TMPDIR"] = str(output)
    os.environ["NFIT_PERFORMANCE_FILE"] = str(output / "performance.json")
    import tempfile
    tempfile.tempdir = str(output)
    receipt = Receipt(output, settings)
    profiler = None
    try:
        from threadpoolctl import threadpool_info, threadpool_limits

        from nfit import (
            DataGroup,
            NfitProject,
            add_data_group_composite_binning,
            load_project,
            raw_dgs_dataset_group,
            refresh_composite_dataset,
            save_project,
            set_reduction_settings,
        )
        from nfit._parallel import num_threads, thread_budget
        from nfit.app_distribution import application_version
        from nfit.data_workspace import project_data_workspace
        from nfit.performance import load_resource_limits, save_resource_limits
        from nfit.raw_dgs_cache import reduced_event_cache_info
        cpu = int(settings.get("threads", 64))
        ram = int(settings.get("ram_limit_mib", 200_000))
        save_resource_limits(cpu_limit=cpu, ram_limit_mb=ram)
        sources_before = _source_receipts(settings)
        receipt.payload.update(nfit_version=application_version(),
                               modules=_module_receipts(), harnesses=_harness_receipts(), sources_before=sources_before,
                               resource_limits=load_resource_limits(),
                               runtime="installed nfit; no source injection",
                               timing_contract="raw reduction/cache write/event accumulation interleaved; progress intervals are observational")
        receipt.flush()
        project_path = output / "DGS_workflow.nfit"
        primary = int(settings.get("primary_copies", 6))
        additional = int(settings.get("additional_copies", 12))
        histograms = {}
        profiling = args.profile or settings.get("profile", False)
        if profiling:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        receipt.payload["profiled_workflow"] = profiling
        workflow_start = time.perf_counter()
        with thread_budget(cpu), threadpool_limits(limits=1, user_api="blas"):
            receipt.payload["resolved_cpu_budget"] = num_threads()
            receipt.payload["threadpools"] = threadpool_info()
            with receipt.timed("import_load_setup"):
                # The same bounded UB extraction used by the Mantid harness;
                # reading the matrix must not import unrelated MDE experiments.
                import h5py
                import numpy as np
                with h5py.File(settings["ub_source"], "r") as handle:
                    ub = np.asarray(handle["MDEventWorkspace/experiment0/sample/oriented_lattice/orientation_matrix"])
                group = raw_dgs_dataset_group(settings["raw_files"], normalization_path=settings["normalization_file"],
                                              mask_path=settings["mask_file"], name="Native DGS benchmark",
                                              progress_callback=receipt.progress)
                reduction = settings["reduction"]
                updates = dict(ub_matrix=ub.tolist(), cache_reduced_events=True,
                               incident_energy_override=reduction.get("ei_override"),
                               t0_override=reduction.get("t0_override"),
                               energy_min_fraction=reduction.get("emin_fraction", -.95),
                               energy_max_fraction=reduction.get("emax_fraction", .95),
                               bad_pulse_threshold=reduction.get("bad_pulses_threshold", 0.))
                set_reduction_settings(group, updates)
                root = DataGroup(name="DGS workflow benchmark", subgroups=[group])
                project = NfitProject(data_groups=[root], settings={"cache_binnings": True})
                _configure(group, settings, primary)
            with project_data_workspace(project, project_path):
                with receipt.timed(f"raw_reduce_cache_bin_{primary}"):
                    histograms[str(primary)] = refresh_composite_dataset(root, node=group,
                                                                        progress_callback=receipt.progress)
                cache_usage = histograms[str(primary)].metadata.get("reduced_event_cache", {})
                receipt.payload["initial_reduction_cache_usage"] = cache_usage
                if cache_usage != {"hits": 0, "misses": len(group.datasets)}:
                    raise AssertionError(f"Initial binning did not construct all reduced caches: {cache_usage}")
                with receipt.timed(f"native_project_save_{primary}"):
                    save_project(project, project_path, progress_callback=receipt.progress)
            initial_saved = time.perf_counter()
            receipt.payload["initial_dataset_workflow_seconds"] = initial_saved - workflow_start
            with receipt.timed("native_project_reopen_metadata"):
                reopened = load_project(project_path)
                reopened_root = reopened.data_groups[0]
                reopened_group = reopened_root.subgroups[0]
            with receipt.timed(f"saved_histogram_access_{primary}"):
                reopened_histogram = refresh_composite_dataset(reopened_root, node=reopened_group,
                                                               progress_callback=receipt.progress)
            load_intervals = receipt.payload["stages"][-1]["progress_intervals_seconds"]
            if "cache_load" not in load_intervals or any(name.startswith("raw_dgs") for name in load_intervals):
                raise AssertionError(f"Reopen did not exclusively load the saved histogram: {load_intervals}")
            initial_viewed = time.perf_counter()
            receipt.payload["initial_reopen_and_view_seconds"] = initial_viewed - initial_saved
            receipt.payload["reopen"] = {"histogram_object_reused": reopened_histogram is histograms[str(primary)],
                                          "dataset_payloads_loaded": sum(item.data is not None for item in reopened_group.datasets)}
            if reopened_histogram is histograms[str(primary)]:
                raise AssertionError("Saved histogram access reused the live cold histogram")
            histograms[str(primary)] = reopened_histogram
            subsequent_start = time.perf_counter()
            with receipt.timed("additional_binning_setup"):
                binning_id = add_data_group_composite_binning(reopened_group, name=f"{additional} copies")
                _configure(reopened_group, settings, additional, binning_id)
            with receipt.timed(f"saved_event_cache_bin_{additional}"):
                histograms[str(additional)] = refresh_composite_dataset(reopened_root, node=reopened_group,
                                                                        binning_id=binning_id,
                                                                        progress_callback=receipt.progress)
            cache_usage = histograms[str(additional)].metadata.get("reduced_event_cache", {})
            receipt.payload["saved_event_rebin_cache_usage"] = cache_usage
            if cache_usage != {"hits": len(reopened_group.datasets), "misses": 0}:
                raise AssertionError(f"Warm rebin did not exclusively use saved reduced events: {cache_usage}")
            with receipt.timed(f"native_project_save_{primary}_and_{additional}"):
                save_project(reopened, project_path, progress_callback=receipt.progress)
            workflow_stop = time.perf_counter()
            receipt.payload["subsequent_rebin_and_save_seconds"] = workflow_stop - subsequent_start
            receipt.payload["combined_two_stage_user_workflow_seconds"] = workflow_stop - workflow_start
            receipt.payload["primary_workflow_seconds"] = initial_saved - workflow_start
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(str(output / "primary_workflow.prof"))
            receipt.payload["script_enter_to_workflow_complete_seconds"] = time.perf_counter() - SCRIPT_ENTERED
            receipt.payload["combined_workflow_stage_labels"] = [item["label"] for item in receipt.payload["stages"]]
            receipt.payload["primary_workflow_stage_labels"] = ["import_load_setup", f"raw_reduce_cache_bin_{primary}", f"native_project_save_{primary}"]
            receipt.flush()
            if args.extra_cached_bins or settings.get("extra_cached_bins", False):
                from nfit import replay_measurement_histogram
                from nfit.symmetry import SymmetrySpec, resolve_symmetry
                edges = settings["grid"]["bin_edges"]
                for copies in settings.get("rebin_copies", [1, 6, 12]):
                    operations = resolve_symmetry(SymmetrySpec("operations", settings["symmetry_operations"][str(copies)]))
                    options = dict(lower=[edge[0] for edge in edges], upper=[edge[-1] for edge in edges],
                                   num_bins=[len(edge)-1 for edge in edges], bin_edges=edges,
                                   vectors=_vectors(settings),
                                   symmetry_operations=[item.matrix_hkl for item in operations],
                                   progress_callback=receipt.progress)
                    with receipt.timed(f"optional_saved_event_cache_bin_{copies}"):
                        extra = replay_measurement_histogram(reopened_group, **options)
                    del extra
                    gc.collect()
        # All array scans and provenance hashes below are excluded from timers.
        receipt.payload["archive"] = _archive_receipt(project_path)
        receipt.payload["reduced_caches"] = [reduced_event_cache_info(item) for item in reopened_group.datasets]
        receipt.payload["histograms"] = {}
        shape = tuple(len(edge)-1 for edge in settings["grid"]["bin_edges"])
        diagnostic_start = time.perf_counter()
        for copies, histogram in histograms.items():
            if histogram.shape != shape:
                raise AssertionError(f"Unexpected grid: {histogram.shape} != {shape}")
            receipt.payload["histograms"][copies] = {"arrays": _array_receipts(histogram),
                                                     "metadata": _bounded_metadata(histogram.metadata),
                                                     "diagnostic_file": _diagnostic_dump(histogram, output / f"histogram-{copies}.h5", settings)}
        receipt.payload["diagnostic_checks_and_dump_seconds"] = time.perf_counter() - diagnostic_start
        after = _source_receipts(settings)
        if after != sources_before:
            raise AssertionError("Scientific sources changed during benchmark")
        if _harness_receipts() != receipt.payload["harnesses"]:
            raise AssertionError("Benchmark harness or shared settings changed during execution")
        modules_after = _module_receipts()
        if modules_after != receipt.payload["modules"]:
            raise AssertionError("Scientific modules changed during benchmark")
        receipt.payload.update(status="complete", sources_after=after,
                               full_array_checks_excluded_from_workflow_timing=True)
        receipt.emit({"event": "benchmark_complete", "receipt": str(output / "receipt.json")})
        receipt.flush()
    except BaseException as exc:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(output / "incomplete_workflow.prof"))
        receipt.payload.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        receipt.flush()
        raise


if __name__ == "__main__":
    main()
