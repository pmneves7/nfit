"""Manual, process-local raw-DGS pipeline trials; production code is unchanged.

Launch with the existing desktop runtime and NFIT_DGS_BENCHMARK_ARGS/CONFIG.
NFIT_DGS_TRIAL_MODE selects one of MODES below. Trials cover chunk threading,
whole-run cache producers, prefetch, coalescing, dense ID lookup and fused
kinematics. The ordinary workflow harness measures
raw reduction/cache/histogram/save and cached reopen/rebin/save. All output stays
in its configured IPTS directory. Numerical comparisons are separate.

The AST extraction below is confined to this experiment: it retains the loaded
reducer's exact reconstruction arithmetic and derives separate reader/compute
functions. It is not an application extension point or a production strategy.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import textwrap
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

# The desktop's --run-script bootstrap does not add the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dgs_settings import RAW_PIPELINE_TRIAL_MODES  # noqa: E402

MODES = RAW_PIPELINE_TRIAL_MODES


@njit(cache=True, fastmath=False, nogil=True)
def compiled_kinematics(indices, exponents, tof, distances, directions, ei, l1,
                        bounds, mantid_precision, tof_constant, energy_constant):
    """Fuse kinematics only; correction ufuncs keep their authoritative path."""
    events = np.empty((len(tof), 6), dtype=np.float64)
    final_k = np.empty(len(tof), dtype=np.float64)
    selected_exponents = np.empty(len(tof), dtype=np.float64)
    incident_tof = tof_constant * l1 / math.sqrt(ei)
    incident_k = math.sqrt(ei / energy_constant)
    used = 0
    for row in range(len(tof)):
        final_tof = tof[row] - incident_tof
        if not final_tof > 0.:
            continue
        index = indices[row]
        ef = (tof_constant * distances[index] / final_tof) ** 2
        energy = ei - ef
        kf_energy = ei - energy if mantid_precision else ef
        kf = math.sqrt(max(kf_energy, 0.) / energy_constant)
        if not (energy >= bounds[0] and energy <= bounds[1]):
            continue
        events[used, 0] = -kf * directions[index, 0]
        events[used, 1] = -kf * directions[index, 1]
        events[used, 2] = incident_k - kf * directions[index, 2]
        events[used, 3] = energy
        final_k[used] = kf
        selected_exponents[used] = exponents[row]
        used += 1
    return events[:used], final_k[:used], selected_exponents[:used]


def dense_geometry_lookup(reference):
    """Trial a geometry-owned dense ID map with an unchanged sparse fallback."""
    def lookup(geometry, ids, *, mantid_precision=False):
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(geometry.detector_ids):
            return reference(geometry, ids, mantid_precision=mantid_precision)
        retained = getattr(geometry, "_trial_dense_id_map", None)
        if retained is None:
            sorted_ids, order = geometry._sorted_ids_and_order
            lower = int(sorted_ids[0])
            size = int(sorted_ids[-1]) - lower + 1
            if size > min(4 * len(sorted_ids), 2_000_000):
                return reference(geometry, ids, mantid_precision=mantid_precision)
            dense = np.full(size, -1, dtype=np.int64)
            _, first = np.unique(sorted_ids, return_index=True)
            dense[sorted_ids[first] - lower] = order[first]
            dense.setflags(write=False)
            retained = lower, dense
            object.__setattr__(geometry, "_trial_dense_id_map", retained)
        lower, dense = retained
        valid = (ids >= lower) & (ids <= lower + len(dense) - 1)
        indices = np.zeros(ids.size, dtype=np.int64)
        indices[valid] = dense[ids[valid].astype(np.int64) - lower]
        valid &= indices >= 0
        indices[~valid] = 0
        exponents = np.zeros(ids.size, dtype=float)
        source = geometry.mantid_he3_exponents if mantid_precision else geometry.he3_exponents
        exponents[valid] = source[indices[valid]]
        if mantid_precision and np.any(~np.isfinite(exponents[valid])):
            raise ValueError(
                "Mantid He-3 correction does not support this detector cylinder shape. "
                "Changing event precision does not change the supported geometry."
            )
        return indices, exponents, valid
    return lookup


def coalesced_chunks(chunks, *, target_bytes=32 * 1024**2):
    """Bounded column blocks, preserving accepted-event order and raw totals."""
    rows = max(1, int(target_bytes) // (6 * 8))
    buffer = np.empty((rows, 6), dtype=np.float64, order="F")
    filled = raw_pending = 0
    try:
        for events, raw_count in chunks:
            if events.ndim != 2 or events.shape[1] != 6:
                raise ValueError("Expected six-column reduced events")
            if not len(events):
                raw_pending += raw_count
                continue
            start = 0
            while start < len(events):
                stop = min(len(events), start + rows - filled)
                buffer[filled:filled + stop - start] = events[start:stop]
                raw_pending += raw_count * stop // len(events) - raw_count * start // len(events)
                filled += stop - start
                start = stop
                if filled == rows:
                    # Returned blocks may outlive the next generator advance.
                    yield buffer.copy(order="F"), raw_pending
                    filled = raw_pending = 0
        if filled or raw_pending:
            yield buffer[:filled].copy(order="F"), raw_pending
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()


def linear_pulse_keep(event_index, pulse_keep, start, stop):
    """Expand only the intersected pulse intervals, without per-event searches."""
    if stop <= start:
        return np.empty(0, dtype=bool)
    # Preserve the authoritative fallback for malformed, unsorted pulse tables.
    if not len(event_index) or not len(pulse_keep) or np.any(np.diff(event_index) < 0):
        indices = np.searchsorted(event_index, np.arange(start, stop), side="right") - 1
        return pulse_keep[np.clip(indices, 0, len(pulse_keep) - 1)]
    left, right = np.maximum(np.searchsorted(event_index, [start, stop - 1], side="right") - 1, 0)
    boundaries = event_index[left + 1:right + 1]
    lengths = np.diff(np.concatenate(([start], boundaries, [stop])))
    indices = np.clip(np.arange(left, right + 1), 0, len(pulse_keep) - 1)
    return np.repeat(pulse_keep[indices], lengths)


def extracted_reconstruction(reference, *, pulse_linear=False):
    """Separate native input reads from the unchanged event arithmetic."""
    source = textwrap.dedent(inspect.getsource(reference))
    node = ast.parse(source).body[0]
    loops = [item for item in ast.walk(node) if isinstance(item, ast.For)
             and isinstance(item.target, ast.Name) and item.target.id == "start"]
    if len(loops) != 1:
        raise ValueError("Unexpected authoritative raw iterator structure")
    loop = loops[0]
    for statement, name in zip(loop.body[1:3], ("event_ids", "raw_tof"), strict=True):
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == name):
            raise ValueError("Unexpected authoritative event input reads")
    if not isinstance(loop.body[-1], ast.Expr) or not isinstance(loop.body[-1].value, ast.Yield):
        raise ValueError("Unexpected authoritative reduced-event return")
    assignments = {statement.targets[0].id: statement for statement in node.body
                   if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)}
    compute = ast.parse("def compute(info, config, geometry, ei, energy_bounds, payload, hyspec_preprocessing=None):\n    pass\n").body[0]
    compute.body = [copy.deepcopy(assignments[name]) for name in ("t0", "mantid_precision")]
    compute.body += ast.parse("event_ids, raw_tof, start, stop, pulse_keep, event_index = payload").body
    compute.body += copy.deepcopy(loop.body[3:-1])
    compute.body.append(ast.Return(value=copy.deepcopy(loop.body[-1].value.value)))
    if pulse_linear:
        replaced = 0
        for statement in ast.walk(compute):
            if (isinstance(statement, ast.If) and statement.body
                and isinstance(statement.body[0], ast.Assign)
                and isinstance(statement.body[0].targets[0], ast.Name)
                and statement.body[0].targets[0].id == "pulse_index"):
                statement.body = ast.parse("valid &= linear_pulse_keep(event_index, pulse_keep, start, stop)").body
                replaced += 1
        if replaced != 1:
            raise ValueError("Unexpected authoritative pulse membership structure")
    reader = copy.deepcopy(node)
    reader.name = "read_tasks"
    reader.decorator_list = []
    reader_loop = next(item for item in ast.walk(reader) if isinstance(item, ast.For)
                       and isinstance(item.target, ast.Name) and item.target.id == "start")
    reader_loop.body = reader_loop.body[:3] + ast.parse(
        "yield event_ids, raw_tof, start, stop, pulse_keep, event_index"
    ).body
    module = ast.fix_missing_locations(ast.Module(body=[reader, compute], type_ignores=[]))
    namespace = dict(reference.__globals__, linear_pulse_keep=linear_pulse_keep)
    exec(compile(module, "<experimental-raw-pipeline>", "exec"), namespace)
    return namespace["read_tasks"], namespace["compute"], hashlib.sha256(source.encode()).hexdigest()


def compiled_reconstruction(reference):
    """Experimentally replace only the native array-pass kinematics section."""
    read_tasks, original_compute, _ = extracted_reconstruction(reference)
    node = ast.parse(textwrap.dedent(inspect.getsource(reference))).body[0]
    loop = next(item for item in ast.walk(node) if isinstance(item, ast.For)
                and isinstance(item.target, ast.Name) and item.target.id == "start")
    filter_end = next(i for i, statement in enumerate(loop.body)
                      if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Tuple)
                      and [x.id for x in statement.targets[0].elts] == ["indices", "exponents", "tof"])
    correction_start = next(i for i, statement in enumerate(loop.body)
                            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
                            and statement.targets[0].id == "weights")
    function = ast.parse("def compute(info, config, geometry, ei, energy_bounds, payload, hyspec_preprocessing=None):\n    pass\n").body[0]
    function.body = ast.parse("if geometry.has_exceptional_positions:\n    return original_compute(info, config, geometry, ei, energy_bounds, payload, hyspec_preprocessing)\n").body
    assignments = {statement.targets[0].id: statement for statement in node.body
                   if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)}
    function.body += [copy.deepcopy(assignments[name]) for name in ("t0", "mantid_precision")]
    function.body += ast.parse("event_ids, raw_tof, start, stop, pulse_keep, event_index = payload").body
    function.body += copy.deepcopy(loop.body[3:filter_end + 1])
    function.body += ast.parse("events, kf, exponents = compiled_kinematics(indices, exponents, tof, geometry.distances, geometry.directions, ei, info.l1, energy_bounds, mantid_precision, TOF_US_PER_M_SQRT_MEV, ENERGY_TO_K2)\nenergy = events[:, 3]\n").body
    function.body += copy.deepcopy(loop.body[correction_start:-1])
    function.body += ast.parse("events[:, 4] = weights\nevents[:, 5] = variances\nreturn events, stop - start\n").body
    namespace = dict(reference.__globals__, original_compute=original_compute,
                     compiled_kinematics=compiled_kinematics)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<experimental-fused-kinematics>", "exec"), namespace)
    compute = namespace["compute"]

    def iterate(info, config, geometry, ei, energy_bounds, max_batch_bytes,
                *, hyspec_preprocessing=None):
        for payload in read_tasks(info, config, geometry, ei, energy_bounds, max_batch_bytes,
                                  hyspec_preprocessing=hyspec_preprocessing):
            yield compute(info, config, geometry, ei, energy_bounds, payload, hyspec_preprocessing)
    return iterate


def threaded_iterator(reference, *, workers, scratch_bytes=64 * 1024**2):
    """One HDF reader, bounded reconstruction workers, original-order delivery."""
    if workers not in (2, 4):
        raise ValueError("This bounded experiment permits two or four workers")
    read_tasks, compute, _ = extracted_reconstruction(reference)
    pending_limit = 2 * workers
    # Account conservatively for live inputs, output, and NumPy compute scratch.
    row_limit = max(1, int(scratch_bytes) // ((pending_limit + workers) * 256))

    def iterate(info, config, geometry, ei, energy_bounds, max_batch_bytes,
                *, hyspec_preprocessing=None):
        tasks = read_tasks(info, config, geometry, ei, energy_bounds,
                           min(max_batch_bytes, row_limit * 96),
                           hyspec_preprocessing=hyspec_preprocessing)
        pending = deque()
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-trial")
        try:
            for payload in tasks:
                pending.append(executor.submit(compute, info, config, geometry, ei,
                                               energy_bounds, payload, hyspec_preprocessing))
                if len(pending) >= pending_limit:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            tasks.close()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    return iterate


def prefetched_chunks(chunks):
    """Overlap one ordered reducer with its cache/histogram consumer."""
    def next_item():
        try:
            return True, next(chunks)
        except StopIteration:
            return False, None

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raw-prefetch-trial")
    future = executor.submit(next_item)
    try:
        while True:
            present, item = future.result()
            if not present:
                break
            future = executor.submit(next_item)
            yield item
    finally:
        future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        chunks.close()


def run_producer_binner(reference, *, workers):
    """Produce private per-run caches, then consume them through native binning."""
    from dataclasses import asdict

    import h5py

    from nfit import raw_dgs
    from nfit.pipeline import DatasetEntry
    from nfit.raw_dgs_cache import cached_reduction, reduction_signature
    from nfit.reduction_recipes import effective_reduction_config

    def bin_group(group, **kwargs):
        candidates = kwargs.get("datasets")
        selected = [dataset for dataset in (group.datasets if candidates is None else candidates)
                    if dataset.enabled and dataset.fit_weight > 0.]
        missing = []
        calibrations = {}
        for dataset in selected:
            config = effective_reduction_config(group, dataset)
            if not config.get("cache_reduced_events", True):
                return reference(group, **kwargs)
            signature = reduction_signature(dataset, config)
            if cached_reduction(dataset, signature) is None:
                missing.append((dataset, config, signature))
                key = config.get("normalization_file"), config.get("mask_file")
                if key not in calibrations:
                    normalization = raw_dgs.load_detector_normalization(key[0]) if key[0] else None
                    calibrations[key] = normalization, raw_dgs._combined_detector_mask(config)
        if not missing:
            return reference(group, **kwargs)
        progress = kwargs.get("progress_callback")

        def report(done):
            if progress is not None:
                progress(dict(stage="raw_dgs_parallel_production", iteration=done,
                              total=len(missing), workers=workers,
                              message=f"reconstructed and cached {done}/{len(missing)} runs"))

        report(0)

        def prepare(item):
            dataset, config, signature = item
            policy = raw_dgs.resolved_dgs_reduction_policies(config)
            source = Path(dataset.metadata["source_file"])
            info = raw_dgs.inspect_raw_dgs_run(source, monitor_variance_policy=policy["monitor_variance_policy"],
                                            bad_pulse_threshold=config.get("bad_pulse_threshold", 95.))
            ei = float(config.get("incident_energy_override") or info.incident_energy)
            preprocessing = None
            if str(info.instrument_name).upper() == "HYSPEC":
                with h5py.File(source, "r") as handle:
                    preprocessing = raw_dgs.resolved_hyspec_preprocessing(
                        handle["entry"], config, ei, instrument_name=info.instrument_name)
            geometry = raw_dgs._detector_geometry(source, hyspec_preprocessing=preprocessing)
            norm, mask = calibrations[config.get("normalization_file"), config.get("mask_file")]
            normalization = raw_dgs._run_normalization_payload(info, geometry, norm, mask, config)
            return dataset, config, signature, info, ei, preprocessing, geometry, norm, mask, normalization

        # Avoid duplicate parsing of the first immutable geometry. Later runs
        # still resolve and check their complete, individual instrument XML.
        first = prepare(missing[0])

        def produce(item, prepared=None):
            dataset, config, signature, info, ei, preprocessing, geometry, norm, mask, normalization = (
                prepare(item) if prepared is None else prepared)
            private = DatasetEntry(dataset.name, None, kind=dataset.kind, data_type=dataset.data_type,
                                   metadata=copy.deepcopy(dataset.metadata), id=dataset.id)
            header = dict(run_info={**asdict(info), "path": str(info.path), "ub_matrix": info.ub_matrix.tolist()},
                          hyspec_preprocessing=preprocessing)
            chunks = raw_dgs._iter_reduced_event_chunks(info, config,
                raw_dgs._masked_detector_geometry(geometry, norm, mask), ei,
                raw_dgs._energy_transfer_bounds(config, ei),
                min(kwargs.get("max_batch_bytes", 192 * 1024**2), 16 * 1024**2),
                hyspec_preprocessing=preprocessing)
            for _ in raw_dgs.cache_event_chunks(private, signature, header, normalization, chunks):
                pass
            return private._raw_dgs_reduction_cache, private.metadata["raw_dgs_reduction_cache"]

        # Futures return only disk references and bounded headers, not complete
        # event arrays. Original datasets are published on the caller thread.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-run-trial") as pool:
            pending = [pool.submit(produce, item, first if i == 0 else None) for i, item in enumerate(missing)]
            results = []
            for future in pending:
                results.append(future.result())
                report(len(results))
        for dataset, config, signature in missing:
            if reduction_signature(dataset, config) != signature:
                raise ValueError("Scientific inputs changed during run cache production")
        for (dataset, _, _), (cache, metadata) in zip(missing, results, strict=True):
            dataset._raw_dgs_reduction_cache = cache
            dataset.metadata["raw_dgs_reduction_cache"] = metadata
        return reference(group, **kwargs)
    return bin_group


def install_trial(mode):
    """Patch only this benchmark process and return a restoration callback."""
    from nfit import raw_dgs

    if mode not in MODES:
        raise ValueError(f"Unknown trial mode {mode!r}")
    original_iterator = raw_dgs._iter_reduced_event_chunks
    original_writer = raw_dgs.cache_event_chunks
    original_lookup = raw_dgs._DetectorGeometry.event_indices_for_ids
    original_binner = raw_dgs.bin_raw_dgs_group
    from nfit import project_composites
    original_composite_binner = project_composites.bin_raw_dgs_group
    previous_preproduce = os.environ.get("NFIT_DGS_TRIAL_PREPRODUCE")
    if mode.startswith("runs"):
        wrapper = run_producer_binner(original_binner, workers=int(mode[4]))
        raw_dgs.bin_raw_dgs_group = project_composites.bin_raw_dgs_group = wrapper
        os.environ["NFIT_DGS_TRIAL_PREPRODUCE"] = "1"
    elif mode.startswith("processes"):
        from trial_dgs_run_processes import process_producer_binner

        wrapper = process_producer_binner(original_binner, workers=int(mode[-1]))
        raw_dgs.bin_raw_dgs_group = project_composites.bin_raw_dgs_group = wrapper
        os.environ["NFIT_DGS_TRIAL_PREPRODUCE"] = "1"
    if mode.startswith("lookup"):
        raw_dgs._DetectorGeometry.event_indices_for_ids = dense_geometry_lookup(original_lookup)
    if mode.startswith("threads"):
        raw_dgs._iter_reduced_event_chunks = threaded_iterator(
            original_iterator, workers=int(mode[7]),
        )
    elif mode == "prefetch":
        def iterator(*args, **kwargs):
            return prefetched_chunks(original_iterator(*args, **kwargs))
        raw_dgs._iter_reduced_event_chunks = iterator
    elif mode.startswith("compiled"):
        raw_dgs._iter_reduced_event_chunks = compiled_reconstruction(original_iterator)
    elif mode == "pulse-linear":
        read_tasks, compute, _ = extracted_reconstruction(original_iterator, pulse_linear=True)

        def iterator(info, config, geometry, ei, energy_bounds, max_batch_bytes,
                     *, hyspec_preprocessing=None):
            for payload in read_tasks(info, config, geometry, ei, energy_bounds, max_batch_bytes,
                                      hyspec_preprocessing=hyspec_preprocessing):
                yield compute(info, config, geometry, ei, energy_bounds, payload, hyspec_preprocessing)
        raw_dgs._iter_reduced_event_chunks = iterator
    if "coalesced" in mode:
        def writer(*args, **kwargs):
            return coalesced_chunks(original_writer(*args, **kwargs))
        raw_dgs.cache_event_chunks = writer

    def restore():
        raw_dgs._iter_reduced_event_chunks = original_iterator
        raw_dgs.cache_event_chunks = original_writer
        raw_dgs._DetectorGeometry.event_indices_for_ids = original_lookup
        raw_dgs.bin_raw_dgs_group = original_binner
        project_composites.bin_raw_dgs_group = original_composite_binner
        if previous_preproduce is None:
            os.environ.pop("NFIT_DGS_TRIAL_PREPRODUCE", None)
        else:
            os.environ["NFIT_DGS_TRIAL_PREPRODUCE"] = previous_preproduce
    return restore


def main():
    import benchmark_dgs_nfit_workflow as workflow
    from benchmark_dgs_settings import load_settings, output_directory

    args = workflow._arguments()
    settings = load_settings(args.config, preset=args.preset)
    if args.tag is None:
        raise ValueError("A unique benchmark tag is required")
    mode = os.environ.get("NFIT_DGS_TRIAL_MODE", "baseline")
    restore = install_trial(mode)
    try:
        workflow.main()
    finally:
        restore()
    output = output_directory(settings, "nfit", args.tag)
    trial = dict(mode=mode, scope="process-local experiment; production code unchanged",
                 prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 supporting_script_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                          for name in ("benchmark_dgs_settings.py", "trial_dgs_run_processes.py")},
                 compute_workers=int(mode[4]) if mode.startswith("runs") else
                 int(mode[-1]) if mode.startswith("processes") else
                 int(mode[7]) if mode.startswith("threads") else 1,
                 live_chunk_limit=8 if mode.startswith("threads4") else
                 4 if mode.startswith("threads2") else
                 int(mode[-1]) if mode.startswith(("runs", "processes")) else 1,
                 run_producer_batch_limit_bytes=16 * 1024**2 if mode.startswith(("runs", "processes")) else None,
                 reconstruction_scratch_target_bytes=64 * 1024**2,
                 coalesced_target_bytes=32 * 1024**2,
                 authoritative_reconstruction_sha256=extracted_reconstruction(
                     __import__("nfit.raw_dgs", fromlist=["_"])._iter_reduced_event_chunks
                 )[2])
    (output / "trial.json").write_text(json.dumps(trial, indent=2) + "\n")
    print(json.dumps(trial), flush=True)


if __name__ == "__main__":
    main()
