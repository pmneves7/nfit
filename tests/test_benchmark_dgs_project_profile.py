"""Manual profiling preserves streaming, aliases and the source project."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def helper():
    path = Path(__file__).parents[1] / "benchmarks/profile_dgs_project.py"
    spec = importlib.util.spec_from_file_location("manual_dgs_project_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_times_only_producer_and_closes_on_cancel(helper, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(helper.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(helper.time, "thread_time", lambda: clock[0])
    closed = []

    def producer():
        try:
            for value in range(3):
                clock[0] += 2
                yield value
        finally:
            closed.append(True)

    components = helper.Components()
    stream = components.wrap("producer", producer)()
    assert next(stream) == 0
    clock[0] += 100  # Consumer work must not enter the producer's timing.
    assert next(stream) == 1
    stream.close()
    assert closed == [True]
    assert components.values["producer"]["inclusive_wall_seconds"] == 4
    assert components.values["producer"]["calls"] == 2


def test_component_hooks_restore_aliases_after_failure(helper):
    import h5py

    from nfit import dgs_event_accumulation, raw_dgs

    original = raw_dgs.accumulate_projected_dgs_events
    read = h5py.Dataset.__getitem__
    lookup = raw_dgs._DetectorGeometry.event_indices_for_ids
    with pytest.raises(RuntimeError, match="cancelled"):
        with helper.Components().installed():
            assert raw_dgs.accumulate_projected_dgs_events is not original
            assert raw_dgs.accumulate_projected_dgs_events is dgs_event_accumulation.accumulate_projected_dgs_events
            raise RuntimeError("cancelled")
    assert raw_dgs.accumulate_projected_dgs_events is original
    assert dgs_event_accumulation.accumulate_projected_dgs_events is original
    assert h5py.Dataset.__getitem__ is read
    assert raw_dgs._DetectorGeometry.event_indices_for_ids is lookup


def test_partial_hook_install_restores_methods(helper, monkeypatch):
    from nfit import raw_dgs

    lookup = raw_dgs._DetectorGeometry.event_indices_for_ids
    monkeypatch.setattr(helper, "COMPONENTS", {
        "raw_dgs": ["_DetectorGeometry.event_indices_for_ids", "missing_function"]})
    with pytest.raises(AttributeError):
        with helper.Components().installed():
            pass
    assert raw_dgs._DetectorGeometry.event_indices_for_ids is lookup


def test_profiled_background_kernel_preserves_all_channels(helper):
    from tests.test_mdevent_background_numba import _run

    args = (np.array([[.25, .25, .25], [.75, .75, .75]]),
            np.array([.25, .75]), np.zeros(2, dtype=np.int64),
            np.array([2., 3.]), np.array([5., 7.]),
            np.repeat(np.eye(3)[None], 3, axis=0), np.array([1., -1., 4.]),
            tuple(np.array([0., .5, 1.]) for _ in range(4)), (2, 2, 2, 2))
    _, expected = _run(*args, workers=2)
    components = helper.Components()
    with components.installed():
        _, actual = _run(*args, workers=2)
    for reference, candidate in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(candidate, reference)
    assert components.values["_mdevent_background_numba._map_and_combine"]["calls"] == 1
    assert components.values["_mdevent_background_numba._scatter"]["calls"] == 1


def test_plan_does_not_create_output_and_rejects_other_ipts(helper, tmp_path):
    source = tmp_path / "IPTS-1/shared/source.nfit"
    source.parent.mkdir(parents=True)
    source.touch()
    output = source.parent / "new-profile"
    helper.main([str(source), str(output)])
    assert not output.exists()
    with pytest.raises(SystemExit):
        helper.main([str(source), str(tmp_path / "IPTS-2/new-profile")])
    with pytest.raises(SystemExit):
        helper.main([str(source), str(source.parent)])


def test_native_profile_runs_whole_workflow_without_touching_source(helper, tmp_path):
    from nfit import DataGroup, NfitProject, raw_dgs_dataset_group, save_project
    from nfit.project_composites import data_group_composite_config
    from tests.test_raw_dgs import _write_raw_dgs

    root = tmp_path / "IPTS-42"
    root.mkdir()
    raw_file = root / "SEQ_42.nxs.h5"
    _write_raw_dgs(raw_file)
    group = raw_dgs_dataset_group([raw_file])
    config = data_group_composite_config(group)
    config.update(enabled=True, auto_rebin=False, coordinate_mode="hkle", stale=True)
    config["axes"] = [dict(name=name, variable=name, vector=vector, lower=-20., upper=20.,
                           num_bins=2, step_size=20., bin_edges=[-20., 0., 20.], mode="edges",
                           fractional=False, auto_lower=False, auto_upper=False, auto_step_size=False)
                      for name, vector in zip(("H", "K", "L", "E"),
                                              ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]),
                                              strict=True)]
    project = NfitProject(data_groups=[DataGroup("Sample", subgroups=[group])],
                          settings={"cache_binnings": False})
    source = root / "source.nfit"
    save_project(project, source)
    original = source.read_bytes()
    output = root / "profile"
    helper.main([str(source), str(output), "--workers", "1", "--execute"])
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["source_identity_unchanged"]
    assert receipt["module_sha256"]
    assert receipt["raw_source_appearances"] == receipt["unique_raw_files"] == 1
    assert sum(receipt["completed_reductions_by_signature_sha256"].values()) == 1
    assert receipt["components"]["raw_dgs._iter_reduced_event_chunks"]["calls"] > 0
    assert receipt["components"]["dgs_event_accumulation.accumulate_projected_dgs_events"]["calls"] > 0
    assert receipt["components"]["hdf5.dataset_read"]["calls"] > 0
    assert (output / "workflow.prof").is_file()
    assert (output / "profiled.nfit").is_file()
    assert source.read_bytes() == original
