"""Isolated, cancellable rebin benchmarks and editable script export.

Trials use a private snapshot, never the live project's caches or numerical data.
Each fresh process warms the workload once and times two subsequent executions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .performance import available_memory_bytes


class BenchmarkCancelled(Exception):
    """The caller cancelled a benchmark; no recommendation should be applied."""


def _available_memory_mib() -> int | None:
    """Return currently available physical memory, constrained by Linux cgroups."""
    available = available_memory_bytes()
    return None if available is None else max(1, available // 1024**2)


def _power_of_two_floor(value: int) -> int:
    return 1 << max(0, int(value).bit_length() - 1)


def benchmark_candidates() -> list[dict[str, int]]:
    """A bounded sweep scaled to the process's CPU and memory allocation."""
    from ._parallel import num_threads

    available = num_threads()
    if available <= 16:
        workers = {available, *(
            2**power for power in range(int(math.log2(available)) + 1)
        )}
    else:
        workers = {
            1,
            min(8, available),
            _power_of_two_floor(max(1, available // 4)),
            _power_of_two_floor(max(1, available // 2)),
            available,
        }

    available_mib = _available_memory_mib()
    high_memory = 512
    if available_mib is not None:
        high_memory = min(
            65_536,
            max(512, _power_of_two_floor(max(1, available_mib // 8))),
        )
    memory = {32, 192, 512, high_memory}
    if high_memory > 512:
        middle = _power_of_two_floor(int(math.sqrt(512 * high_memory)))
        memory.add(middle)
    return [
        {"max_batch_mb": mb, "workers": worker}
        for mb in sorted(memory)
        for worker in sorted(workers)
    ]


def rebin_benchmark_target(project, *, dataset_id=None, group_name=None, node_id=None):
    """Resolve a dataset or composite and its editable rebin configuration."""
    from .project_data import _CompositeScope, data_group_composite_config, dataset_rebin_config

    for group in project.data_groups:
        if dataset_id is not None:
            for dataset in group.iter_datasets():
                if dataset.id == dataset_id:
                    return dataset, group, dataset_rebin_config(dataset)
        elif group.name == group_name:
            if node_id is None:
                return group, group, data_group_composite_config(group)
            for node in group.iter_subgroups():
                if node.id == node_id:
                    scope = _CompositeScope(group, node)
                    return scope, group, data_group_composite_config(scope)
    raise ValueError("The benchmark target is not in this project.")


def recommend_performance(rows: list[dict]) -> dict[str, int]:
    """Prefer fewer workers and less memory among results within 5% of fastest."""
    if not rows:
        raise ValueError("No successful benchmark trials.")
    best = min(row["seconds"] for row in rows)
    near = [row for row in rows if row["seconds"] <= best * 1.05]
    selected = min(near, key=lambda row: (row["workers"], row["max_batch_mb"]))
    return {key: selected[key] for key in ("max_batch_mb", "workers")}


def benchmark_rebin(project=None, *, dataset_id=None, group_name=None, node_id=None,
                    candidates=None, cancel=None, progress=None) -> dict:
    """Calibrate synthetic workloads, or benchmark a project's exact rebin.

    ``project=None`` selects machine calibration (3-D hard and 4-D fractional
    grids). Otherwise identify a dataset by ID or a composite by group name
    and optional nested node ID. ``cancel`` is a callable returning a boolean;
    ``progress`` receives each completed timing/memory row. Results are advisory
    and do not modify project state or preferences. Memory is process peak RSS,
    including Python, imported libraries, inputs and warmup—not batch memory.
    """
    candidates = benchmark_candidates() if candidates is None else list(candidates)
    if not candidates:
        raise ValueError("At least one candidate is required.")
    for candidate in candidates:
        if int(candidate["max_batch_mb"]) < 1 or int(candidate["workers"]) < 1:
            raise ValueError("Benchmark candidates must have positive memory and workers.")
    rows = []
    with tempfile.TemporaryDirectory(prefix="nfit-benchmark-") as directory:
        root = Path(directory)
        snapshot = root / "input.pickle"
        # Only this locally generated snapshot is unpickled by the private worker.
        with snapshot.open("wb") as stream:
            pickle.dump((project, dict(dataset_id=dataset_id, group_name=group_name, node_id=node_id)), stream)
        for candidate in candidates:
            if cancel and cancel():
                raise BenchmarkCancelled()
            result_path = root / "result.json"
            result_path.unlink(missing_ok=True)
            worker_args = ["--benchmark-worker"] if getattr(sys, "frozen", False) else ["-m", "nfit.performance_benchmark"]
            command = [sys.executable, *worker_args, str(snapshot),
                       str(result_path), str(candidate["max_batch_mb"]), str(candidate["workers"])]
            with (root / "trial.log").open("w+") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                try:
                    while process.poll() is None:
                        if cancel and cancel():
                            raise BenchmarkCancelled()
                        time.sleep(0.1)
                    if process.returncode:
                        log.seek(0)
                        raise RuntimeError("Benchmark trial failed:\n" + log.read()[-4000:])
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
            row = json.loads(result_path.read_text())
            rows.append(row)
            if progress:
                progress(dict(row))
    if cancel and cancel():
        raise BenchmarkCancelled()
    return {"trials": rows, "recommendation": recommend_performance(rows),
            "mode": "machine" if project is None else "rebin"}


def export_benchmark_script(path, project=None, **target) -> None:
    """Export an editable benchmark and, for real rebins, a project snapshot.

    The adjacent .nfit snapshot retains all scientific settings; original source
    files must remain available. Existing sidecars are not overwritten.
    """
    path = Path(path)
    preamble = "project = None\n"
    if project is not None:
        archive = path.with_suffix(".nfit")
        if archive.exists():
            raise FileExistsError(f"Choose a new script name; snapshot already exists: {archive}")
        from .project_archive import write_project_manifest
        from .project_io import _project_to_dict

        snapshot = copy.deepcopy(project)
        asset_source = getattr(snapshot, "_project_path", None)
        write_project_manifest(
            archive,
            _project_to_dict(snapshot),
            asset_source=asset_source,
            preserve_existing=asset_source is not None,
        )
        preamble = f"from nfit import load_project\nproject = load_project(Path(__file__).with_name({archive.name!r}))\n"
    path.write_text(
        '"""Edit the candidate list or project rebin settings before running."""\n'
        "from pathlib import Path\nfrom nfit.performance_benchmark import benchmark_rebin\n"
        + preamble + f"\ncandidates = {benchmark_candidates()!r}\n"
        + f"result = benchmark_rebin(project, candidates=candidates, **{target!r})\nprint(result)\n"
    )


def _trial(snapshot, output, mb, workers):
    import numpy as np

    from ._parallel import num_threads, thread_budget
    from .performance import (
        batch_budget,
        peak_process_memory_mib,
        scientific_memory_limit_bytes,
    )
    from .rebin import rebin_nd

    # Report the resource values actually trialled, including machine limits.
    workers = min(workers, num_threads())
    mb = min(mb, max(1, scientific_memory_limit_bytes() // 1024**2))
    with Path(snapshot).open("rb") as stream:
        project, target = pickle.load(stream)
    if project is None:
        rng = np.random.default_rng(20260905)
        workloads = [(rng.random((300_000, dim)), rng.random(300_000), dim, fractional)
                     for dim, fractional in ((3, False), (4, True))]

        def run():
            for coords, data, dim, fractional in workloads:
                rebin_nd(data, coords, lower=[0] * dim, upper=[1] * dim,
                         num_bins=[24] * dim, fractional=fractional,
                         max_batch_bytes=mb * 1024**2, workers=workers)
    else:
        from .project_data import (
            _viewer_data_before_scale_uncached,
            composite_dataset_data,
            effective_dataset_masks,
        )
        subject, group, config = rebin_benchmark_target(project, **target)
        config.update(max_batch_mb=mb, workers=workers)
        if target["dataset_id"] is not None:
            config["enabled"] = True

            def run():
                result = _viewer_data_before_scale_uncached(
                    subject, extra_masks=effective_dataset_masks(group, subject), force_rebin=True,
                )
                if result is None:
                    raise ValueError("This dataset does not support rebinning.")
                return result
        else:
            def run():
                return composite_dataset_data(subject)
    with thread_budget(workers), batch_budget(mb * 1024**2):
        run()  # compilation and warmup excluded from timing, included in peak RSS
        times = []
        for _ in range(2):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
    peak_mib = peak_process_memory_mib()
    Path(output).write_text(json.dumps(dict(max_batch_mb=mb, workers=workers,
                                          seconds=statistics.median(times), peak_mib=peak_mib)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Private isolated benchmark worker")
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("mb", type=int)
    parser.add_argument("workers", type=int)
    args = parser.parse_args()
    _trial(args.snapshot, args.output, args.mb, args.workers)
