"""Matched directional-background timing on a native project's real runs.

Run sequential source snapshots in the existing desktop runtime. The original
project is read-only; source reduction, diagnostic channels, logs and new saves
live in the configured IPTS output. NFIT_DGS_BENCHMARK_CONFIG supplies JSON;
NFIT_DGS_BENCHMARK_SOURCE optionally selects a frozen source tree.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dgs_nfit_workflow import _select_source_tree  # noqa: E402


def compare_outputs(reference, actual, output, *, allow_rounding=False):
    """Check every cell, edge and coverage boundary using bounded HDF reads."""
    import h5py

    result = {"channels": {}, "literal_parity": True, "numerical_parity": True,
              "relative_tolerance": 1e-12 if allow_rounding else 0.}
    with h5py.File(reference, "r") as baseline, h5py.File(actual, "r") as candidate:
        if set(baseline) != set(candidate):
            raise ValueError("Diagnostic channels differ")
        for key in baseline:
            a, b = baseline[key], candidate[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Diagnostic layout differs: {key}")
            width = max(1, 8*1024**2 // max(1, int(np.prod(a.shape[1:])) * a.dtype.itemsize))
            different = 0
            maximum = 0.
            relative = 0.
            failed = 0
            support_differences = 0
            for start in range(0, a.shape[0], width):
                aa, bb = a[start:start+width], b[start:start+width]
                same = aa == bb
                if aa.dtype.kind == "f":
                    same |= np.isnan(aa) & np.isnan(bb)
                    finite = np.isfinite(aa) & np.isfinite(bb)
                    maximum = max(maximum, float(np.max(np.abs(aa[finite]-bb[finite]), initial=0.)))
                    nonzero = finite & (aa != 0)
                    relative = max(relative, float(np.max(np.abs((bb[nonzero]-aa[nonzero])/aa[nonzero]), initial=0.)))
                different += int(np.count_nonzero(~same))
                close = (np.isclose(aa, bb, rtol=1e-12, atol=0., equal_nan=True)
                         if allow_rounding and key in ("signal", "errors", "N") else same)
                failed += int(np.count_nonzero(~close))
                if key == "N":
                    support_differences += int(np.count_nonzero((aa > 0) != (bb > 0)))
            result["channels"][key] = dict(cells=int(np.prod(a.shape)), different_cells=different,
                                           maximum_absolute_difference=maximum,
                                           maximum_relative_difference=relative,
                                           outside_tolerance=failed, support_differences=support_differences)
            result["literal_parity"] &= different == 0
            result["numerical_parity"] &= failed == 0 and support_differences == 0
    Path(output).write_text(json.dumps(result, indent=2))
    if not result["numerical_parity"]:
        raise AssertionError("Background trial changed numerical cells")
    print(json.dumps(result), flush=True)

def main():
    settings = json.loads(Path(os.environ["NFIT_DGS_BENCHMARK_CONFIG"]).read_text())
    if "comparison" in settings:
        compare_outputs(**settings["comparison"])
        return
    selected_source = _select_source_tree()
    if settings.get("replay_task_limit"):
        from nfit import mdevent_background
        mdevent_background.MAX_REPLAY_TRANSFORM_TASKS = int(settings["replay_task_limit"])
    from profile_dgs_project import Components

    from nfit import load_project, save_project
    from nfit._parallel import thread_budget
    from nfit.data_workspace import bind_data_workspace
    from nfit.performance import peak_process_memory_mib
    from nfit.project_composites import (
        _composite_background_data,
        _composite_scope,
        _raw_dgs_composite_histogram,
    )

    source = Path(settings["project"]).resolve(strict=True)
    output = Path(settings["output"]).resolve()
    ipts = next(p for p in source.parents if p.name.startswith("IPTS-"))
    if not output.is_relative_to(ipts):
        raise ValueError("Scientific output must stay in the source IPTS")
    output.mkdir(parents=True, exist_ok=False)
    identity = source.stat()
    started = time.perf_counter()
    project = load_project(source)
    root = project.data_groups[0]
    sample = next(n for n in root.iter_subgroups() if n.name == settings["sample"])
    link = sample.backgrounds[0]
    dummy = link.source_group
    if settings.get("sample_runs", 0):
        indices = np.linspace(0, len(sample.datasets)-1, settings["sample_runs"], dtype=int)
        sample.datasets = [sample.datasets[i] for i in indices]
    dummy.datasets = dummy.datasets[:settings.get("dummy_runs", 1)]
    for node in (sample, dummy):
        node.enabled = True
        for run in node.datasets:
            run.enabled = True
    # The template may have raw defaults distinct from its copied MDE metadata.
    for node in (sample, dummy):
        node.metadata["raw_dgs"].update(mask_file=None, hyspec_default_mask=True)
    bind_data_workspace(root, output / "result.nfit")
    scope = _composite_scope(root, sample)
    config = copy.deepcopy(settings["binning"])
    last = 0.

    def progress(event):
        nonlocal last
        now = time.perf_counter()
        if now-last > 20:
            print(json.dumps(dict(elapsed_seconds=now-started, **event), default=str), flush=True)
            last = now

    receipt = dict(host=platform.node(), selected_source=selected_source,
                   sample_runs=len(sample.datasets), dummy_runs=len(dummy.datasets),
                   workers=settings["workers"], repeats=[], source_unchanged=None)
    with thread_budget(settings["workers"]):
        before = time.perf_counter()
        target = _raw_dgs_composite_histogram(scope, config, progress)
        receipt["sample_load_reduce_cache_bin_seconds"] = time.perf_counter()-before
        receipt["shape"] = list(target.shape)
        for repeat in range(settings.get("repeats", 2)):
            components = Components()
            before = time.perf_counter()
            if settings.get("component_profile", False):
                with components.installed():
                    background = _composite_background_data(scope, link, target, config=config,
                                                           progress_callback=progress)
            else:
                background = _composite_background_data(scope, link, target, config=config,
                                                       progress_callback=progress)
            duration = time.perf_counter()-before
            item = dict(seconds=duration, components=components.values,
                        completed_reductions_by_signature=components.completed_reductions,
                        source_cache_owned=[getattr(d, "_raw_dgs_reduction_cache", None) is not None
                                            for d in dummy.datasets])
            receipt["repeats"].append(item)
            print(json.dumps(dict(repeat=repeat, **item)), flush=True)
        # Diagnostics and hashes follow timing; no reductions are called here.
        import h5py

        with h5py.File(output / "background.h5", "w") as handle:
            handle.create_dataset("N", data=background.metadata["normalization_denominator"], chunks=True)
            for name in ("signal", "errors", "mask", "num_events"):
                handle.create_dataset(name, data=getattr(background, name), chunks=True)
            for index, axis in enumerate(background.axes):
                handle.create_dataset(f"edge{index}", data=axis.values)
        receipt["raw_cache_publication"] = {
            d.id: bool(getattr(d, "_raw_dgs_reduction_cache", None)) for d in dummy.datasets}
        project.settings["cache_binnings"] = False
        before = time.perf_counter()
        save_project(project, output / "result.nfit")
        receipt["save_seconds"] = time.perf_counter()-before
    current = source.stat()
    receipt.update(source_unchanged=(identity.st_ino, identity.st_size, identity.st_mtime_ns) ==
                   (current.st_ino, current.st_size, current.st_mtime_ns),
                   peak_process_memory_mib=peak_process_memory_mib(),
                   module_sha256={name: hashlib.sha256(Path(sys.modules[f"nfit.{name}"].__file__).read_bytes()).hexdigest()
                                  for name in ("project_composites", "event_masks", "mdevent_background", "dgs_background_sources")})
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    assert receipt["source_unchanged"]
    print("COMPLETE", json.dumps({k: v for k, v in receipt.items() if k != "repeats"}), flush=True)


if __name__ == "__main__":
    main()
