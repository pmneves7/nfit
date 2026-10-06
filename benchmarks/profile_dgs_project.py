"""Profile an entire saved native DGS project, including directional backgrounds.

The source is read-only; execution saves a separate project in a new directory.
Without --execute this prints the plan and touches no output files. This manual
harness never imports Mantid. Profiling overhead makes its times unsuitable for
headline engine comparisons. Generator timings exclude time spent consuming
their yielded blocks. Component intervals are inclusive and must not be summed.
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import hashlib
import importlib
import inspect
import json
import os
import platform
import pstats
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# Only Python entry points are patched. Do not replace a dispatcher referenced
# inside a compiled Numba function: that would alter compilation and execution.
COMPONENTS = {
    "raw_dgs": ["inspect_raw_dgs_run", "_monitor_ei_t0", "_detector_geometry",
                "_DetectorGeometry.event_indices_for_ids",
                "_iter_reduced_event_chunks", "_he3_tube_efficiency_correction",
                "bin_raw_dgs_group", "_trajectory_normalization",
                "_powder_trajectory_normalization"],
    "raw_dgs_cache": ["cache_event_chunks", "iter_cached_event_chunks", "cached_reduction"],
    "raw_dgs_bounds": ["raw_dgs_coordinate_bounds"],
    "dgs_event_accumulation": ["accumulate_projected_dgs_events"],
    "dgs_background_sources": ["sample_exposures", "trajectory_payloads"],
    "mdevent_background": ["project_measured_background_mdevent", "_replay_runs"],
    "mdevent": ["_trajectory_normalization_from_payloads"],
    "_mdevent_background_numba": ["_map_and_combine", "_scatter"],
    "analysis.artifacts": ["write_dataset_artifact"],
    "array_archive": ["_deflate_block"],
    "mapped_archive": ["_inflate_mapped_member"],
    "project_archive": ["_copy_archive_stream", "_rewrite_archive"],
}


class Components:
    """Collect caller-thread CPU and elapsed time without retaining arguments."""

    def __init__(self):
        self.values = {}
        self.completed_reductions = {}
        self.lock = threading.Lock()

    def call(self, label, function, *args, **kwargs):
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            return function(*args, **kwargs)
        finally:
            wall, cpu = time.perf_counter()-wall, time.thread_time()-cpu
            with self.lock:
                value = self.values.setdefault(label, dict(
                    calls=0, inclusive_wall_seconds=0., calling_thread_cpu_seconds=0.,
                    maximum_call_seconds=0.))
                value["calls"] += 1
                value["inclusive_wall_seconds"] += wall
                value["calling_thread_cpu_seconds"] += cpu
                value["maximum_call_seconds"] = max(value["maximum_call_seconds"], wall)

    def wrap(self, label, function):
        if inspect.isgeneratorfunction(function):
            @functools.wraps(function)
            def generator(*args, **kwargs):
                source = function(*args, **kwargs)
                try:
                    while True:
                        try:
                            item = self.call(label, next, source)
                        except StopIteration:
                            if label == "raw_dgs_cache.cache_event_chunks":
                                # Counts completed producers by complete reduction
                                # identity, not just basename or instrument name.
                                signature = args[1] if len(args) > 1 else kwargs["signature"]
                                key = hashlib.sha256(signature.encode()).hexdigest()
                                with self.lock:
                                    self.completed_reductions[key] = self.completed_reductions.get(key, 0) + 1
                            return
                        yield item
                finally:
                    source.close()
            return generator

        @functools.wraps(function)
        def ordinary(*args, **kwargs):
            return self.call(label, function, *args, **kwargs)
        return ordinary

    @contextmanager
    def installed(self):
        replacements = []
        originals = {}
        try:
            for module_name, names in COMPONENTS.items():
                module = importlib.import_module(f"nfit.{module_name}")
                for name in names:
                    owner, attribute = module, name
                    if "." in name:
                        owner_name, attribute = name.rsplit(".", 1)
                        owner = getattr(module, owner_name)
                    function = getattr(owner, attribute)
                    wrapper = self.wrap(f"{module_name}.{name}", function)
                    originals[id(function)] = (function, wrapper)
                    if owner is not module:
                        replacements.append((owner, attribute, function))
                        setattr(owner, attribute, wrapper)
            import h5py

            function = h5py.Dataset.__getitem__
            replacements.append((h5py.Dataset, "__getitem__", function))
            h5py.Dataset.__getitem__ = self.wrap("hdf5.dataset_read", function)
            # Patch explicit re-exports too; later imports see the patched
            # owner. Restore exact identities, including aliases, on failure.
            for name, module in list(sys.modules.items()):
                if name.startswith("nfit.") and module is not None:
                    for attribute, function in list(vars(module).items()):
                        pair = originals.get(id(function))
                        if pair is not None and function is pair[0]:
                            replacements.append((module, attribute, function))
                            setattr(module, attribute, pair[1])
            yield
        finally:
            for module, attribute, function in reversed(replacements):
                setattr(module, attribute, function)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path, help="New directory inside the experiment's IPTS folder")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fresh-reduction", action="store_true",
                        help="Discard caches in the isolated copy; reconstruct all selected raw runs")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    source = args.project.resolve(strict=True)
    output = args.output.resolve()
    ipts = next((path for path in source.parents if path.name.startswith("IPTS-")), None)
    if ipts is not None and not output.is_relative_to(ipts):
        parser.error("output must stay inside the source experiment's IPTS folder")
    if output.exists():
        parser.error("output directory must be new")
    print(json.dumps(dict(source=str(source), output=str(output), workers=args.workers,
                          fresh_reduction=args.fresh_reduction, execute=args.execute)), flush=True)
    if not args.execute:
        return

    from nfit import load_project, save_project
    from nfit._parallel import thread_budget
    from nfit.performance import peak_process_memory_mib
    from nfit.project_caches import clear_project_caches

    output.mkdir(parents=True, exist_ok=False)
    identity = source.stat()
    receipt = dict(status="running", host=platform.node(), python=sys.version,
                   source=str(source), workers=args.workers, fresh_reduction=args.fresh_reduction,
                   scope="Public load/save workflow; all stale configured bins and backgrounds",
                   component_intervals="Inclusive; nested calls and worker intervals overlap",
                   cpu_definition="Component CPU is calling-thread time; total CPU includes all threads",
                   filesystem_cache="Uncontrolled; not a cold-disk measurement")
    components, profiler = Components(), cProfile.Profile()
    started, cpu_started = time.perf_counter(), time.process_time()
    last_event, last_print = None, 0.

    def progress(event):
        nonlocal last_event, last_print
        now = time.perf_counter()
        key = (event.get("stage"), event.get("batch_name"), event.get("rebin_name"),
               event.get("background_run"))
        if key != last_event or now-last_print >= 20:
            print(json.dumps(dict(elapsed_seconds=now-started, **event), default=str), flush=True)
            last_event, last_print = key, now

    try:
        with thread_budget(args.workers), components.installed():
            receipt["module_sha256"] = {
                name: hashlib.sha256(Path(sys.modules[f"nfit.{name}"].__file__).read_bytes()).hexdigest()
                for name in COMPONENTS
            }
            profiler.enable()
            project = components.call("workflow.load_project", load_project, source)
            entries = [dataset for root in project.data_groups for dataset in root.iter_datasets()
                       if dataset.kind == "raw_dgs_nexus"]
            receipt["raw_source_appearances"] = len(entries)
            receipt["unique_raw_files"] = len({
                str(Path(dataset.metadata["source_file"]).resolve()) for dataset in entries})
            if args.fresh_reduction:
                clear_project_caches(project)
            project.settings["cache_binnings"] = True
            components.call("workflow.save_project", save_project, project,
                            output / "profiled.nfit", progress_callback=progress)
            receipt["cached_binnings"] = len(project.settings.get("binning_cache_entries", []))
            receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"], receipt["error"] = "failed", str(error)
        raise
    finally:
        profiler.disable()
        receipt.update(total_wall_seconds=time.perf_counter()-started,
                       total_process_cpu_seconds=time.process_time()-cpu_started,
                       peak_process_memory_mib=peak_process_memory_mib(),
                       components=components.values,
                       completed_reductions_by_signature_sha256=components.completed_reductions)
        current = source.stat()
        receipt["source_identity_unchanged"] = (
            identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns
        ) == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        profiler.dump_stats(str(output / "workflow.prof"))
        stats = pstats.Stats(profiler).stats
        hot = sorted(stats.items(), key=lambda item: item[1][2], reverse=True)[:80]
        receipt["main_thread_profile"] = [dict(
            file=key[0], line=key[1], function=key[2], calls=value[1],
            self_seconds=value[2], cumulative_seconds=value[3]) for key, value in hot]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(status=receipt["status"], receipt=str(output / "receipt.json"))), flush=True)
    if not receipt["source_identity_unchanged"]:
        raise RuntimeError("Source changed during profiling; exclude this measurement")


if __name__ == "__main__":
    main(json.loads(os.environ["NFIT_DGS_PROJECT_PROFILE_ARGS"])
         if "NFIT_DGS_PROJECT_PROFILE_ARGS" in os.environ else None)
