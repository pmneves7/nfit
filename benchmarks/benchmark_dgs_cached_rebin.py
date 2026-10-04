"""Manual saved-project DGS rebin benchmark, without repeating raw reduction.

Run with the installed nfit ``--run-script`` entry point. CLI arguments can be
passed as JSON in ``NFIT_DGS_BENCHMARK_ARGS`` because the desktop bootstrap
resets argv; ``NFIT_DGS_BENCHMARK_CONFIG`` and
``NFIT_DGS_BENCHMARK_PROJECT`` are equivalent configuration/input routes.
An optional ``NFIT_DGS_BENCHMARK_SOURCE`` selects a staged package using the
same explicit reset as the standard native workflow benchmark.

The headline includes metadata load, saved primary histogram access, one new
cached-event binning, and saving a NEW project. Diagnostic channel scans and
HDF exports are outside that interval. Original input projects are read only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path

SCRIPT_ENTERED = time.perf_counter()


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset", default="sequoia")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--copies", type=int, choices=(1, 6, 12), default=12)
    argv = json.loads(os.environ["NFIT_DGS_BENCHMARK_ARGS"]) if "NFIT_DGS_BENCHMARK_ARGS" in os.environ else None
    args = parser.parse_args(argv)
    if args.config is None and os.environ.get("NFIT_DGS_BENCHMARK_CONFIG"):
        args.config = Path(os.environ["NFIT_DGS_BENCHMARK_CONFIG"])
    if args.project is None and os.environ.get("NFIT_DGS_BENCHMARK_PROJECT"):
        args.project = Path(os.environ["NFIT_DGS_BENCHMARK_PROJECT"])
    if args.project is None:
        parser.error("Provide --project or NFIT_DGS_BENCHMARK_PROJECT")
    return args


def _ipts_root(path):
    resolved = Path(path).expanduser().resolve()
    matches = [index for index, part in enumerate(resolved.parts) if re.fullmatch(r"IPTS-\d+", part)]
    if len(matches) != 1:
        raise ValueError(f"Scientific inputs and outputs must be inside one IPTS: {resolved}")
    return Path(*resolved.parts[:matches[0] + 1])


def _validate_paths(settings, project):
    root = _ipts_root(settings["output_root"])
    paths = [project, *settings["raw_files"], settings["normalization_file"],
             settings["mask_file"], settings["ub_source"]]
    for path in paths:
        if path is not None and _ipts_root(path) != root:
            raise ValueError(f"Scientific input is outside the configured IPTS: {path}")
    project = Path(project).resolve()
    if project.suffix != ".nfit" or not project.is_file():
        raise ValueError("The input must be an existing saved .nfit benchmark project")
    return project


def _input_receipt(path):
    stat = path.stat()
    with zipfile.ZipFile(path) as archive:
        header = archive.read("project.json")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "manifest_sha256": hashlib.sha256(header).hexdigest()}


def _source_headers(sources):
    result = {}
    for path in sources:
        with Path(path).open("rb") as handle:
            payload = handle.read(65536)
        result[path] = {"prefix_bytes": len(payload), "prefix_sha256": hashlib.sha256(payload).hexdigest()}
    return result


def _matching_group(project, settings):
    expected = [str(Path(path).resolve()) for path in settings["raw_files"]]
    matches = []

    def visit(root, group):
        paths = [str(Path(item.metadata.get("source_file", "")).resolve()) for item in group.datasets]
        if "raw_dgs" in group.metadata and paths == expected:
            runs = [int(item.metadata["run_number"]) for item in group.datasets]
            if runs != settings["run_numbers"]:
                raise ValueError("Saved project run membership differs from the shared settings")
            matches.append((root, group))
        for child in group.subgroups:
            visit(root, child)

    for root in project.data_groups:
        visit(root, root)
    if len(matches) != 1:
        raise ValueError("Expected exactly one raw DGS group with the configured ordered source membership")
    return matches[0]


def _validate_reduction(group, settings):
    import h5py
    import numpy as np

    from nfit.reduction_recipes import effective_reduction_config
    expected = settings["reduction"]
    pairs = {"incident_energy_override": expected.get("ei_override"),
             "t0_override": expected.get("t0_override"),
             "energy_min_fraction": expected.get("emin_fraction", -.95),
             "energy_max_fraction": expected.get("emax_fraction", .95),
             "bad_pulse_threshold": expected.get("bad_pulses_threshold", 0.)}
    for dataset in group.datasets:
        if dataset.fit_weight != 1.0 or dataset.scale_factor != 1.0:
            raise ValueError("Cached benchmark requires the original unit source weights and scales")
        config = effective_reduction_config(group, dataset)
        if any(config.get(key) != value for key, value in pairs.items()):
            raise ValueError(f"Saved reduction settings differ for {dataset.name}")
        for key in ("normalization_file", "mask_file"):
            actual, wanted = config.get(key), settings[key]
            if (None if actual is None else Path(actual).resolve()) != (None if wanted is None else Path(wanted).resolve()):
                raise ValueError(f"Saved {key} differs from the shared settings")
    with h5py.File(settings["ub_source"], "r") as handle:
        ub = np.asarray(handle["MDEventWorkspace/experiment0/sample/oriented_lattice/orientation_matrix"])
    if not np.array_equal(np.asarray(group.metadata["raw_dgs"]["ub_matrix"]), ub):
        raise ValueError("Saved UB differs from the shared reference matrix")
    return effective_reduction_config(group)


def _require_saved_caches(group):
    from nfit.raw_dgs_cache import cached_reduction, reduction_signature
    from nfit.reduction_recipes import effective_reduction_config

    for dataset in group.datasets:
        config = effective_reduction_config(group, dataset)
        if not config.get("cache_reduced_events", True) or cached_reduction(dataset, reduction_signature(dataset, config)) is None:
            raise AssertionError(f"A compatible saved reduced cache is required before rebinning: {dataset.name}")


def _validate_histogram_grid(histogram, settings):
    import numpy as np
    edges = settings["grid"]["bin_edges"]
    if histogram.shape != tuple(len(edge)-1 for edge in edges) or any(
        not np.array_equal(axis.values, expected)
        for axis, expected in zip(histogram.axes, edges, strict=True)
    ):
        raise AssertionError("Histogram did not retain the exact shared grid edges")


@contextlib.contextmanager
def _calculate_new_binning(binning_id):
    """Time an explicit rebin, preserving every saved histogram for fair I/O.

    Normally nfit can reuse an equivalent named histogram instantly. This
    single-job manual benchmark disables that optimization for the new request
    only; neither the cached-event algorithm nor project cache contents change.
    """
    from nfit import project_composites

    original = project_composites._matching_composite_base_key
    new_id = binning_id

    def lookup(group, signature, binning_id=None):
        if binning_id == new_id:
            return None
        return original(group, signature, binning_id)

    project_composites._matching_composite_base_key = lookup
    try:
        yield
    finally:
        project_composites._matching_composite_base_key = original


def main():
    args = _arguments()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import benchmark_dgs_nfit_workflow as native
    from benchmark_dgs_settings import load_settings, output_directory

    settings = load_settings(args.config, preset=args.preset)
    input_path = _validate_paths(settings, args.project)
    if args.config is not None and _ipts_root(args.config) != _ipts_root(settings["output_root"]):
        raise ValueError("Configuration must be inside the configured IPTS")
    tag = args.tag or settings.get("tag") or time.strftime("%Y%m%d-%H%M%S")
    output = output_directory(settings, "nfit", tag)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = str(output)
    os.environ["MPLCONFIGDIR"] = str(output / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(output / "runtime-cache")
    os.environ["NFIT_PERFORMANCE_FILE"] = str(output / "performance.json")
    import tempfile
    tempfile.tempdir = str(output)
    receipt = native.Receipt(output, settings)
    try:
        runtime = native._select_source_tree()
        from threadpoolctl import threadpool_info, threadpool_limits

        from nfit import (
            add_data_group_composite_binning,
            load_project,
            refresh_composite_dataset,
            save_project,
        )
        from nfit._parallel import num_threads, thread_budget
        from nfit.app_distribution import application_version
        from nfit.data_workspace import project_data_workspace
        from nfit.performance import load_resource_limits, save_resource_limits
        from nfit.raw_dgs_cache import reduced_event_cache_info

        cpu, ram = int(settings.get("threads", 64)), int(settings.get("ram_limit_mib", 200000))
        save_resource_limits(cpu_limit=cpu, ram_limit_mb=ram)
        before, sources = _input_receipt(input_path), native._source_receipts(settings)
        harnesses = native._harness_receipts()
        harnesses[Path(__file__).name] = {"path": str(Path(__file__).resolve()),
                                        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        receipt.payload.update(workflow="saved_reduced_event_cache_rebin", raw_reduction_performed=False,
            runtime=runtime, nfit_version=application_version(), modules=native._module_receipts(),
            harnesses=harnesses, input_before=before, sources_before=sources,
            source_headers=_source_headers(sources), resource_limits=load_resource_limits(),
            equal_named_histogram_reuse="disabled only for the new explicit rebin; existing archives retained",
            timing_contract="load metadata and saved histogram, one new cached rebin, save new project; no raw reduction")
        receipt.flush()
        project_path = output / "DGS_cached_rebin.nfit"
        workflow_start = time.perf_counter()
        with thread_budget(cpu), threadpool_limits(limits=1, user_api="blas"):
            receipt.payload.update(resolved_cpu_budget=num_threads(), threadpools=threadpool_info())
            with receipt.timed("native_project_reopen_metadata"):
                project = load_project(input_path)
                root, group = _matching_group(project, settings)
                receipt.payload["saved_reduction_settings"] = native._bounded_metadata(_validate_reduction(group, settings))
            primary = int(settings.get("primary_copies", 6))
            with receipt.timed(f"saved_histogram_access_{primary}"):
                saved = refresh_composite_dataset(root, node=group, progress_callback=receipt.progress)
            intervals = receipt.payload["stages"][-1]["progress_intervals_seconds"]
            if "cache_load" not in intervals or any(name.startswith("raw_dgs") for name in intervals):
                raise AssertionError(f"Saved primary histogram access triggered reconstruction: {intervals}")
            receipt.payload["saved_histogram_shape"] = list(saved.shape)
            _validate_histogram_grid(saved, settings)
            del saved
            with project_data_workspace(project, project_path):
                with receipt.timed("additional_binning_setup"):
                    _require_saved_caches(group)
                    binning_id = add_data_group_composite_binning(group, name=f"benchmark {tag}: {args.copies} copies")
                    native._configure(group, settings, args.copies, binning_id)
                with receipt.timed(f"saved_event_cache_bin_{args.copies}"):
                    with _calculate_new_binning(binning_id):
                        histogram = refresh_composite_dataset(root, node=group, binning_id=binning_id,
                                                              progress_callback=receipt.progress)
                intervals = receipt.payload["stages"][-1]["progress_intervals_seconds"]
                if not {"raw_dgs_events", "raw_dgs_normalization"} <= intervals.keys():
                    raise AssertionError(f"Requested rebin did not calculate events and trajectories: {intervals}")
                cache_usage = histogram.metadata.get("reduced_event_cache", {})
                receipt.payload["saved_event_rebin_cache_usage"] = cache_usage
                if cache_usage != {"hits": len(group.datasets), "misses": 0}:
                    raise AssertionError(f"Rebin must exclusively use saved reduced caches: {cache_usage}")
                with receipt.timed(f"native_project_save_{args.copies}"):
                    save_project(project, project_path, progress_callback=receipt.progress)
        receipt.payload["cached_rebin_workflow_seconds"] = time.perf_counter() - workflow_start
        receipt.payload["cached_rebin_workflow_stage_labels"] = [stage["label"] for stage in receipt.payload["stages"]]
        receipt.payload["script_enter_to_workflow_complete_seconds"] = time.perf_counter() - SCRIPT_ENTERED
        receipt.flush()
        # Full-array scans and all fingerprints below are outside user timers.
        diagnostic_start = time.perf_counter()
        _validate_histogram_grid(histogram, settings)
        receipt.payload["archive"] = native._archive_receipt(project_path)
        receipt.payload["reduced_caches"] = [reduced_event_cache_info(item) for item in group.datasets]
        receipt.payload["histograms"] = {str(args.copies): {
            "arrays": native._array_receipts(histogram), "metadata": native._bounded_metadata(histogram.metadata),
            "diagnostic_file": native._diagnostic_dump(histogram, output / f"histogram-{args.copies}.h5", settings)}}
        receipt.payload["diagnostic_checks_and_dump_seconds"] = time.perf_counter() - diagnostic_start
        if _input_receipt(input_path) != before:
            raise AssertionError("Original input project changed during the benchmark")
        if native._source_receipts(settings) != sources:
            raise AssertionError("Scientific source files changed during the benchmark")
        if native._module_receipts() != receipt.payload["modules"]:
            raise AssertionError("Scientific modules changed during the benchmark")
        after_harnesses = native._harness_receipts()
        after_harnesses[Path(__file__).name] = {"path": str(Path(__file__).resolve()),
                                              "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if after_harnesses != harnesses:
            raise AssertionError("Benchmark harnesses changed during execution")
        receipt.payload.update(status="complete", input_after=before,
                               full_array_checks_excluded_from_workflow_timing=True)
        receipt.emit({"event": "benchmark_complete", "receipt": str(output / "receipt.json")})
        receipt.flush()
    except BaseException as exc:
        receipt.payload.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        receipt.flush()
        raise


if __name__ == "__main__":
    main()
