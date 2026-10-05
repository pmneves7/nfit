"""Correctness gates for the manual prototype, independent of external engines."""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from nfit import raw_dgs
from tests.test_raw_dgs import _write_raw_dgs


@pytest.fixture(autouse=True)
def benchmark_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks"))


def _prototype():
    # Numba's persistent cache must be able to re-import the defining module.
    return importlib.import_module("trial_dgs_raw_pipeline")


@pytest.mark.parametrize("precision", ["mantid", "high_precision"])
@pytest.mark.parametrize("workers", [2, 4, "compiled"])
def test_parallel_reconstruction_keeps_events_counts_and_geometry(tmp_path, precision, workers):
    prototype = _prototype()
    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path, with_he3=True)
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r+") as handle:
        # Multiple banks and prompt/masked events exercise ordered delivery.
        bank = handle["entry/bank1_events"]
        del bank["event_id"], bank["event_time_offset"]
        bank.create_dataset("event_id", data=[42, 999, 42, 42, 42, 42])
        bank.create_dataset("event_time_offset", data=[9000., 9000., 1., 10000., 11000., 12000.])
        handle.copy(bank, handle["entry"], name="bank2_events")
    info = raw_dgs.inspect_raw_dgs_run(path)
    geometry = raw_dgs._detector_geometry(path)
    config = dict(event_precision_policy=precision, t0_override=0., bad_pulse_threshold=0.)
    options = (info, config, geometry, info.incident_energy, (-19., 19.), 192)
    reference = list(raw_dgs._iter_reduced_event_chunks(*options))
    iterator = (prototype.compiled_reconstruction(raw_dgs._iter_reduced_event_chunks) if workers == "compiled" else
                prototype.threaded_iterator(raw_dgs._iter_reduced_event_chunks, workers=workers))
    candidate = list(iterator(*options))
    assert [n for _, n in candidate] == [n for _, n in reference]
    for (actual, _), (expected, _) in zip(candidate, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert not geometry.positions.flags.writeable


def test_coalesced_blocks_keep_order_raw_totals_and_independent_storage():
    prototype = _prototype()
    blocks = [(np.arange(18.).reshape(3, 6), 5), (np.empty((0, 6)), 7),
              (np.arange(18., 42.).reshape(4, 6), 9), (np.empty((0, 6)), 3)]
    actual = list(prototype.coalesced_chunks(iter(blocks), target_bytes=96))
    np.testing.assert_array_equal(np.concatenate([events for events, _ in actual]),
                                  np.concatenate([events for events, _ in blocks]))
    assert sum(count for _, count in actual) == 24
    assert all(len(events) <= 2 for events, _ in actual)
    assert all(events.flags.f_contiguous for events, _ in actual)
    assert not np.shares_memory(actual[0][0], actual[1][0])


def test_prefetch_preserves_order_and_closes_on_early_exit():
    prototype = _prototype()
    closed = []
    def source():
        try:
            yield from range(100)
        finally:
            closed.append(True)
    iterator = prototype.prefetched_chunks(source())
    assert next(iterator) == 0
    assert next(iterator) == 1
    iterator.close()
    assert closed == [True]


@pytest.mark.parametrize("ids", [np.array([-1, 0, 1, 2, 3, 4, 99]),
                                 np.array([0, 1, 99], dtype=np.uint64)])
@pytest.mark.parametrize("precision", [False, True])
def test_dense_lookup_keeps_holes_duplicates_and_own_geometry(ids, precision):
    prototype = _prototype()
    reference = raw_dgs._DetectorGeometry.event_indices_for_ids
    lookup = prototype.dense_geometry_lookup(reference)
    for detector_ids, exponents in (([3, 0, 1, 1], [3., 1., 2., 4.]),
                                    ([0, 3, 1], [6., 5., 7.])):
        geometry = raw_dgs._DetectorGeometry(np.array(detector_ids),
            np.ones((len(detector_ids), 3)), np.array(exponents))
        for actual, expected in zip(lookup(geometry, ids, mantid_precision=precision),
                                    reference(geometry, ids, mantid_precision=precision), strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert not geometry._trial_dense_id_map[1].flags.writeable


def test_sparse_lookup_falls_back_without_allocating_a_large_map():
    prototype = _prototype()
    geometry = raw_dgs._DetectorGeometry(np.array([0, 100_000_000]),
                                         np.ones((2, 3)), np.ones(2))
    reference = raw_dgs._DetectorGeometry.event_indices_for_ids
    for actual, expected in zip(prototype.dense_geometry_lookup(reference)(geometry, np.array([0, 1, 100_000_000])),
                                reference(geometry, np.array([0, 1, 100_000_000])), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert not hasattr(geometry, "_trial_dense_id_map")


def test_linear_pulse_selection_matches_per_event_search_at_all_boundaries():
    prototype = _prototype()
    rng = np.random.default_rng(48)
    for event_index in (np.sort(rng.integers(0, 100, size=40)),
                        np.array([0, 0, 0, 10, 20, 20, 90]),
                        np.array([10, 0, 50]), np.empty(0, dtype=int)):
        keep = rng.random(max(1, len(event_index) - 2)) > .5
        for start in range(0, 100, 7):
            stop = min(start + 17, 100)
            indices = np.searchsorted(event_index, np.arange(start, stop), side="right") - 1
            expected = keep[np.clip(indices, 0, len(keep) - 1)]
            np.testing.assert_array_equal(prototype.linear_pulse_keep(event_index, keep, start, stop), expected)


@pytest.mark.parametrize("precision", ["mantid", "high_precision"])
def test_linear_pulse_reconstruction_keeps_deadtime_and_bad_pulse_membership(tmp_path, precision):
    prototype = _prototype()
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path, with_he3=True)
    with h5py.File(path, "r+") as handle:
        charge = handle["entry/DASlogs/proton_charge"]
        charge.create_dataset("value", data=[100., 100., 0., 100., 100., 100.])
        times = charge.create_dataset("time", data=np.arange(6.))
        times.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
        pause = handle["entry/DASlogs"].create_group("pause")
        pause.create_dataset("value", data=[0., 1., 0.])
        times = pause.create_dataset("time", data=[0., 3., 4.])
        times.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
        bank = handle["entry/bank1_events"]
        del bank["event_id"], bank["event_time_offset"]
        bank.create_dataset("event_id", data=np.full(12, 42))
        bank.create_dataset("event_time_offset", data=np.linspace(9000., 12000., 12))
        bank.create_dataset("event_index", data=np.arange(0, 12, 2))
        times = bank.create_dataset("event_time_zero", data=np.arange(6.))
        times.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
    info = raw_dgs.inspect_raw_dgs_run(path)
    geometry = raw_dgs._detector_geometry(path)
    config = dict(event_precision_policy=precision, t0_override=0., bad_pulse_threshold=95.)
    options = (info, config, geometry, info.incident_energy, (-19., 19.), 3 * 96)
    expected = list(raw_dgs._iter_reduced_event_chunks(*options))
    reader, compute, _ = prototype.extracted_reconstruction(raw_dgs._iter_reduced_event_chunks, pulse_linear=True)
    actual = [compute(*options[:5], payload) for payload in reader(*options)]
    assert 0 < sum(len(events) for events, _ in actual) < 12
    for (events, raw_count), (reference, count) in zip(actual, expected, strict=True):
        assert raw_count == count
        np.testing.assert_array_equal(events, reference)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_run_producers_preserve_histogram_and_reduction_signatures(tmp_path, workers):
    prototype = _prototype()
    paths = [tmp_path / f"SEQ_{i}.nxs.h5" for i in (42, 43)]
    for path in paths:
        _write_raw_dgs(path, with_he3=True)
    group = raw_dgs.raw_dgs_dataset_group(paths)
    reference = raw_dgs.raw_dgs_dataset_group(paths)
    options = dict(lower=[-10., -10., -10., -19.], upper=[10., 10., 10., 19.],
                   num_bins=[2, 2, 2, 2], vectors=np.eye(4))
    expected = raw_dgs.bin_raw_dgs_group(reference, **options)
    actual = prototype.run_producer_binner(raw_dgs.bin_raw_dgs_group, workers=workers)(group, **options)
    np.testing.assert_array_equal(actual.num_events, expected.num_events)
    np.testing.assert_allclose(actual.signal, expected.signal, rtol=1e-12, atol=0., equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, rtol=1e-12, atol=0., equal_nan=True)
    from nfit.raw_dgs_cache import reduction_signature
    for dataset in group.datasets:
        assert dataset._raw_dgs_reduction_cache.signature == reduction_signature(dataset, group.metadata["raw_dgs"])


def test_run_producers_reject_changed_inputs_before_publishing(tmp_path, monkeypatch):
    prototype = _prototype()
    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path, with_he3=True)
    group = raw_dgs.raw_dgs_dataset_group([path])
    writer = raw_dgs.cache_event_chunks

    def changed_input(*args, **kwargs):
        yield from writer(*args, **kwargs)
        status = path.stat()
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(raw_dgs, "cache_event_chunks", changed_input)
    with pytest.raises(ValueError, match="Scientific inputs changed"):
        prototype.run_producer_binner(lambda *args, **kwargs: None, workers=1)(group, datasets=None)
    assert getattr(group.datasets[0], "_raw_dgs_reduction_cache", None) is None
    assert "raw_dgs_reduction_cache" not in group.datasets[0].metadata


def test_process_worker_publishes_persistent_exact_native_cache(tmp_path, monkeypatch):
    import json

    from nfit.data_workspace import bind_data_workspace
    from nfit.raw_dgs_cache import iter_cached_event_chunks

    scripts = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(scripts))
    from trial_dgs_run_processes import worker_main

    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path, with_he3=True)
    group = raw_dgs.raw_dgs_dataset_group([path])
    bind_data_workspace(group, tmp_path / "benchmark.nfit")
    dataset = group.datasets[0]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(group_metadata=group.metadata,
        datasets=[dict(name=dataset.name, kind=dataset.kind, data_type=dataset.data_type,
                       metadata=dataset.metadata, id=dataset.id)]),
        default=lambda value: value.tolist()))
    monkeypatch.setenv("NFIT_DGS_RUN_MANIFEST", str(manifest))
    worker_main()
    record, = json.loads((tmp_path / "complete.json").read_text())
    info = raw_dgs.inspect_raw_dgs_run(path)
    geometry = raw_dgs._detector_geometry(path)
    config = group.metadata["raw_dgs"]
    native = list(raw_dgs._iter_reduced_event_chunks(
        info, config, geometry, info.incident_energy,
        raw_dgs._energy_transfer_bounds(config, info.incident_energy), 192 * 1024**2))
    # Worker datasets and their staging owners have left scope. Its published
    # disk reference must survive, with the authoritative logical event order.
    with np.load(record["path"], allow_pickle=False) as archive:
        actual = list(iter_cached_event_chunks(archive, 192 * 1024**2))
    np.testing.assert_array_equal(np.concatenate([events for events, _ in actual]),
                                  np.concatenate([events for events, _ in native]))
    assert sum(count for _, count in actual) == sum(count for _, count in native)


@pytest.mark.skipif(os.name != "posix", reason="Manual ORNL process-group prototype")
def test_process_cleanup_joins_worker_and_its_child():
    import subprocess
    import sys

    from trial_dgs_run_processes import stop_owned_jobs

    code = """import signal,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
def finish(*args):
    child.wait(timeout=3)
    sys.exit(0)
signal.signal(signal.SIGTERM,finish)
print('ready',flush=True)
time.sleep(60)
"""
    job = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                           text=True, start_new_session=True)
    try:
        assert job.stdout.readline().strip() == "ready"
        stop_owned_jobs([job])
        # The parent exits cleanly only after it reaps the terminated child.
        assert job.returncode == 0
    finally:
        stop_owned_jobs([job])
        job.stdout.close()


@pytest.mark.parametrize("change", [None, "weights", "exposure", "header", "order"])
def test_saved_cache_gate_checks_payload_and_not_only_histogram(tmp_path, change):
    import io
    import json
    import zipfile

    from compare_dgs_raw_trial_caches import compare

    events = np.arange(18., dtype=float).reshape(3, 6)
    projects = [tmp_path / name for name in ("reference.nfit", "actual.nfit")]
    for index, project in enumerate(projects):
        payload = events.copy()
        direction = np.ones((1, 3))
        header = {"chunk_count": 1, "incident_energy": 20.}
        if index and change == "weights":
            payload[1, 4] += 1e-10
        elif index and change == "exposure":
            direction[0, 0] += 1e-10
        elif index and change == "header":
            header["incident_energy"] = 21.
        elif index and change == "order":
            payload = payload[::-1]
        cache = io.BytesIO()
        np.savez(cache, events_0=payload, raw_count_0=10,
                 detector_ids=[42], direction=direction, solid=[1.], charge=1.,
                 header_json=json.dumps(header))
        manifest = dict(data_groups=[dict(datasets=[dict(metadata={
            "source_file": "SEQ_42.nxs.h5", "raw_dgs_reduction_cache": {
                "member": "cache.npz", "signature": "unchanged"}})])])
        with zipfile.ZipFile(project, "w") as archive:
            archive.writestr("project.json", json.dumps(manifest))
            archive.writestr("cache.npz", cache.getvalue())
    if change:
        with pytest.raises((AssertionError, ValueError)):
            compare(projects[1], projects[0])
    else:
        result = compare(projects[1], projects[0])
        assert result["status"] == "passed"
        assert result["accepted_events"] == 3
        assert result["raw_events"] == 10
