"""Native directional replay retains raw calibration and source covariance."""
import numpy as np
import pytest

from nfit import bin_raw_dgs_group, raw_dgs_dataset_group, replay_cached_background_profile
from nfit.dgs_background_sources import open_event_stream
from nfit.mdevent_background import project_measured_background_mdevent
from nfit.reduction_recipes import set_reduction_settings
from tests.test_raw_dgs import _write_raw_dgs


@pytest.fixture
def groups(tmp_path):
    h5py = pytest.importorskip("h5py")
    paths = [tmp_path/name for name in ("sample0.nxs.h5", "sample1.nxs.h5", "dummy.nxs.h5")]
    for path in paths:
        _write_raw_dgs(path)
    with h5py.File(paths[1], "r+") as f:
        f["entry/DASlogs/omega/average_value"][...] = [90.]
        f["entry/DASlogs/proton_charge/average_value"][...] = [6.]
    sample = raw_dgs_dataset_group(paths[:2])
    dummy = raw_dgs_dataset_group(paths[2:])
    for group in (sample, dummy):
        set_reduction_settings(group, {"bad_pulse_threshold": 0., "ki_kf_normalization": False,
            "he3_detector_efficiency_correction": False, "ub_matrix": np.eye(3).tolist()})
    return sample, dummy


def target(group):
    return bin_raw_dgs_group(group, lower=[-10., -10., -10., -19.],
                            upper=[10., 10., 10., 19.], num_bins=[1, 1, 1, 1])


def test_native_replay_matches_single_primitive_despite_angle_copies(groups):
    sample, dummy = groups
    direct = target(dummy)
    replay = project_measured_background_mdevent(sample, dummy, target(sample))
    np.testing.assert_allclose(replay.signal, direct.signal)
    np.testing.assert_allclose(replay.errors, direct.errors)
    np.testing.assert_allclose(replay.num_events, direct.num_events)
    np.testing.assert_allclose(replay.metadata["normalization_denominator"], direct.metadata["normalization_denominator"])
    assert replay.metadata["cached_background_replay"]["terms"][0]["sources"][0]["source_format"] == "raw-direct-geometry-nexus"
    profile = replay_cached_background_profile(replay, selected=np.ones(replay.shape, bool),
        indices=np.zeros(replay.shape, int), edges=[0., 1.], max_batch_bytes=1)
    assert profile.data.signal.item() == pytest.approx(direct.signal.item())
    assert profile.data.errors.item() == pytest.approx(direct.errors.item())


@pytest.mark.parametrize("coordinate_mask", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_composite_replay_with_inherited_dataset_condition(
    groups, coordinate_mask, compiled, monkeypatch,
):
    from nfit import (
        BackgroundSpec,
        DataGroup,
        dataset_criterion_mask,
        dataset_criterion_preview,
        filter_dataset_criteria,
        mdevent_background,
    )
    from nfit.pipeline import MaskSpec
    from nfit.project_composites import _apply_composite_backgrounds, _composite_scope

    sample, dummy = groups
    if compiled and mdevent_background._REPLAY_NUMBA is None:
        pytest.skip("Numba is unavailable")
    if not compiled:
        monkeypatch.setattr(mdevent_background, "_REPLAY_NUMBA", None)
    for run, value in zip([*sample.datasets, *dummy.datasets], [0., 1., 0.], strict=True):
        run.metadata["selection_value"] = value
    sample.backgrounds.append(BackgroundSpec(
        "Dummy", source_group=dummy, source_group_id=dummy.id, projection="measured_events",
    ))
    root = DataGroup("Parent selection", subgroups=[sample, dummy])
    condition = dataset_criterion_mask(
        root, channel="metadata", source="metadata/selection_value", right_value=0.5,
    )
    if coordinate_mask:
        root.masks.append(MaskSpec("Positive energy", "energy_q_range", {"energy": [0., 19.]}))
    dataset_criterion_preview(root, condition)
    grid = dict(lower=[-10., -10., -10., -19.], upper=[10., 10., 10., 19.],
                num_bins=[1, 1, 1, 2])
    selected = filter_dataset_criteria(root, sample.datasets)
    assert selected == [sample.datasets[0]]
    template = bin_raw_dgs_group(sample, datasets=selected, **grid)
    actual = _apply_composite_backgrounds(_composite_scope(root, sample), template)

    # A whole-run condition must give the same subtraction and uncertainty as
    # explicitly disabling that run, including when coordinate masks coexist.
    assert all(run.enabled for run in [*sample.datasets, *dummy.datasets])
    condition.enabled = False
    sample.datasets[1].enabled = False
    expected = _apply_composite_backgrounds(_composite_scope(root, sample), template)
    for name in ("signal", "errors", "num_events"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), equal_nan=True)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.metadata["normalization_denominator"],
                                  expected.metadata["normalization_denominator"])


def test_raw_replay_keeps_source_detector_ids_and_lab_coordinates(groups):
    _sample, dummy = groups
    with open_event_stream(dummy, dummy.datasets[0], rows=1) as (gonio, _shape, _dtype, blocks):
        events = np.concatenate(list(blocks))
    assert len(events) == 1
    assert events[0, 4] == 42
    np.testing.assert_array_equal(gonio, np.eye(3))
    assert events[0, 5] < 0
    assert events[0, 0] == events[0, 1] == 1


def test_raw_replay_rejects_different_bank_geometry(groups):
    sample, dummy = groups
    h5py = pytest.importorskip("h5py")
    path = dummy.datasets[0].metadata["source_file"]
    with h5py.File(path, "r+") as f:
        payload = f["entry/instrument/instrument_xml/data"][()].tobytes().replace(b'x="1"', b'x="2"')
        del f["entry/instrument/instrument_xml/data"]
        f["entry/instrument/instrument_xml"].create_dataset("data", data=np.frombuffer(payload, dtype="u1"))
    with pytest.raises(ValueError, match="matching incident energy and detector geometry"):
        project_measured_background_mdevent(sample, dummy, target(sample))


def test_raw_profile_retains_per_run_event_correction(groups):
    sample, dummy = groups
    set_reduction_settings(dummy, {"ki_kf_normalization": True}, dataset_ids=[dummy.datasets[0].id])
    replay = project_measured_background_mdevent(sample, dummy, target(sample))
    profile = replay_cached_background_profile(replay, selected=np.ones(replay.shape, bool),
        indices=np.zeros(replay.shape, int), edges=[0., 1.])
    assert profile.data.signal.item() == pytest.approx(replay.signal.item())
    assert profile.data.errors.item() == pytest.approx(replay.errors.item())


def test_native_measured_replay_option_is_visible_in_gui(groups, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit import BackgroundSpec, DataGroup, NfitProject
    from nfit.project_gui import NfitProjectExplorer

    sample, dummy = groups
    link = BackgroundSpec("Dummy", source_group_id=dummy.id, source_group=dummy)
    sample.backgrounds.append(link)
    root = DataGroup("Native DGS", subgroups=[sample, dummy])
    explorer = NfitProjectExplorer(NfitProject([root]))
    monkeypatch.setattr(explorer, "_background_changed", lambda *_: None)
    explorer._set_background_details(root, sample, link)
    projection = explorer.details_widget.findChild(widgets.QComboBox, "background_projection")
    assert projection.findData("measured_events") >= 0
    assert projection.findData("sample_trajectories") < 0
    assert "raw DGS" in projection.toolTip()
    projection.setCurrentIndex(projection.findData("measured_events"))
    assert link.projection == "measured_events"
    explorer.window.close()


def test_composite_replay_publishes_and_reuses_original_run_caches(groups, tmp_path, monkeypatch):
    from nfit import BackgroundSpec, DataGroup, NfitProject, load_project, raw_dgs, save_project
    from nfit.project_composites import _composite_background_data, _composite_scope
    from nfit.raw_dgs_cache import ArchiveMember

    sample, dummy = groups
    link = BackgroundSpec("Dummy", source_group_id=dummy.id, source_group=dummy,
                          projection="measured_events")
    sample.backgrounds.append(link)
    root = DataGroup("Runs", subgroups=[sample, dummy])
    project = NfitProject([root])
    path = tmp_path / "project.nfit"
    save_project(project, path)
    template = target(sample)
    sample_caches = [run._raw_dgs_reduction_cache for run in sample.datasets]
    produced = []
    producer = raw_dgs._iter_reduced_event_chunks

    def record(*args, **kwargs):
        produced.append(args[0].path)
        yield from producer(*args, **kwargs)

    monkeypatch.setattr(raw_dgs, "_iter_reduced_event_chunks", record)
    scope = _composite_scope(root, sample)
    first = _composite_background_data(scope, link, template, config=None, progress_callback=None)
    assert len(produced) == 1  # Only the previously unreduced dummy.
    assert dummy.datasets[0]._raw_dgs_reduction_cache is not None
    assert [run._raw_dgs_reduction_cache for run in sample.datasets] == sample_caches
    second = _composite_background_data(scope, link, template, config=None, progress_callback=None)
    assert len(produced) == 1
    np.testing.assert_array_equal(second.signal, first.signal)
    np.testing.assert_array_equal(second.errors, first.errors)
    save_project(project, path)
    restored = load_project(path)
    root = restored.data_groups[0]
    sample, dummy = root.subgroups
    assert isinstance(dummy.datasets[0]._raw_dgs_reduction_cache.content, ArchiveMember)
    third = _composite_background_data(_composite_scope(root, sample), sample.backgrounds[0],
                                      template, config=None, progress_callback=None)
    assert len(produced) == 1
    np.testing.assert_array_equal(third.signal, first.signal)
    np.testing.assert_array_equal(third.errors, first.errors)
