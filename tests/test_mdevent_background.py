import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import nfit.mdevent_background as mdevent_background
from nfit import (
    BackgroundSpec,
    DataGroup,
    NfitProject,
    bin_mdevent_group,
    mdevent_dataset_group,
    project_measured_background_mdevent,
)
from nfit.mdevent_background import _accumulate_correlated_events
from nfit.pipeline import MaskSpec
from nfit.project_composites import _apply_composite_backgrounds, _CompositeScope
from nfit.project_io import _project_from_dict, _project_to_dict
from tests.test_mdevent import _write_mdevent


def _one_voxel(group):
    return bin_mdevent_group(
        group,
        lower=[-1] * 4,
        upper=[1] * 4,
        num_bins=[1] * 4,
    )


def _assert_replay_results_equal(actual, expected):
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
    np.testing.assert_allclose(actual.num_events, expected.num_events)
    np.testing.assert_allclose(
        actual.metadata["normalization_denominator"],
        expected.metadata["normalization_denominator"],
    )
    assert actual.metadata["event_weight_rms"] == pytest.approx(
        expected.metadata["event_weight_rms"]
    )


def test_replay_same_geometry_matches_native_and_preserves_source_statistics(tmp_path):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    background = mdevent_dataset_group(path)
    target = _one_voxel(sample)
    progress = []
    result = project_measured_background_mdevent(
        sample, background, target, progress_callback=progress.append
    )
    np.testing.assert_allclose(result.signal, target.signal)
    np.testing.assert_allclose(result.errors, target.errors)
    assert not result.mask.item()
    replay_progress = [
        event for event in progress if event["stage"] == "mdevent_background_replay"
    ]
    assert replay_progress
    expected_backend = "numba" if mdevent_background._REPLAY_NUMBA is not None else "numpy"
    assert {event["backend"] for event in replay_progress} == {expected_backend}
    assert all(event["workers"] >= 1 for event in replay_progress)
    expected_message = "compiled parallel" if expected_backend == "numba" else "NumPy fallback"
    assert expected_message in replay_progress[0]["message"]
    # Repeating the same angle cannot manufacture extra independent counts.
    duplicate = [replace(sample.datasets[0]) for _ in range(5)]
    repeated = project_measured_background_mdevent(sample, background, target, datasets=duplicate)
    np.testing.assert_allclose(repeated.signal, result.signal)
    np.testing.assert_allclose(repeated.errors, result.errors)
    small_batches = project_measured_background_mdevent(
        sample, background, target, max_batch_bytes=1
    )
    np.testing.assert_allclose(small_batches.signal, result.signal)
    np.testing.assert_allclose(small_batches.errors, result.errors)


def test_replay_batches_angle_normalization_and_reports_cumulative_progress(
    tmp_path, monkeypatch
):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    background = mdevent_dataset_group(path)
    sample.datasets = [replace(sample.datasets[0]) for _ in range(3)]
    target = _one_voxel(sample)
    progress = []
    calls = []
    normalize = mdevent_background.mdevent._trajectory_normalization_from_payloads

    def record_normalization(
        detectors, payloads, edges, shape, *, progress_callback=None, **kwargs
    ):
        calls.append((len(detectors), len(payloads)))
        return normalize(
            detectors,
            payloads,
            edges,
            shape,
            progress_callback=progress_callback,
            **kwargs,
        )

    monkeypatch.setattr(
        mdevent_background.mdevent,
        "_trajectory_normalization_from_payloads",
        record_normalization,
    )
    project_measured_background_mdevent(
        sample, background, target, progress_callback=progress.append
    )

    setup = [
        event
        for event in progress
        if event["stage"] == "mdevent_background_normalization_setup"
    ]
    assert setup
    assert [event["iteration"] for event in setup] == sorted(
        event["iteration"] for event in setup
    )
    assert setup[0]["iteration"] == 0
    assert setup[-1]["iteration"] == setup[-1]["total"]
    assert "sample angle 3/3" in setup[-1]["message"]

    # All unmasked sample angles share one persistent trajectory accumulator.
    assert len(calls) == len(background.datasets)
    assert all(detectors == 1 and payloads == 3 for detectors, payloads in calls)
    normalization = [
        event
        for event in progress
        if event["stage"] == "mdevent_background_normalization"
    ]
    assert normalization
    assert [event["iteration"] for event in normalization] == sorted(
        event["iteration"] for event in normalization
    )
    assert normalization[0]["iteration"] == 0
    assert normalization[-1]["iteration"] == normalization[-1]["total"]
    assert normalization[-1]["sample_angle"] == 3
    assert normalization[-1]["sample_angles_total"] == 3
    source_total = len(background.datasets)
    assert f"background run {source_total}/{source_total}" in normalization[-1]["message"]
    assert "sample angle 3/3" in normalization[-1]["message"]


def test_replay_does_not_mutate_read_only_normalization_when_masked(
    tmp_path, monkeypatch
):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    background = mdevent_dataset_group(path)
    sample.masks = [
        MaskSpec("masked edge", "energy_q_range", {"energy": [0.5, 1.0]})
    ]
    target = bin_mdevent_group(
        sample,
        lower=[-1, -1, -1, -1],
        upper=[1, 1, 1, 1],
        num_bins=[1, 1, 1, 2],
    )
    normalize = mdevent_background.mdevent._trajectory_normalization_from_payloads

    def readonly_normalization(*args, **kwargs):
        result = normalize(*args, **kwargs)
        result.setflags(write=False)
        return result

    monkeypatch.setattr(
        mdevent_background.mdevent,
        "_trajectory_normalization_from_payloads",
        readonly_normalization,
    )
    result = project_measured_background_mdevent(sample, background, target)
    assert result.shape == target.shape


def test_replay_cancellation_propagates_from_cumulative_setup(tmp_path):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    background = mdevent_dataset_group(path)
    target = _one_voxel(sample)

    def cancel(event):
        if event["stage"] == "mdevent_background_normalization_setup":
            assert event["sample_angle"] == 1
            assert event["sample_angles_total"] == len(sample.datasets)
            raise RuntimeError("cancelled during cumulative setup")

    with pytest.raises(RuntimeError, match="cancelled during cumulative setup"):
        project_measured_background_mdevent(
            sample, background, target, progress_callback=cancel
        )


def test_replay_scales_weights_and_masks(tmp_path):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    background = mdevent_dataset_group(path)
    background.datasets[0].scale_factor = 3
    background.datasets[0].fit_weight = 2
    expected = _one_voxel(background)
    result = project_measured_background_mdevent(sample, background, _one_voxel(sample))
    np.testing.assert_allclose(result.signal, expected.signal)
    np.testing.assert_allclose(result.errors, expected.errors)
    background.masks = [MaskSpec("window", "energy_q_range", {"energy": [10, 11]}, invert=True)]
    sample.backgrounds = [
        BackgroundSpec("replay", source_group=background, projection="measured_events", scale=0.1)
    ]
    root = DataGroup("workspace", subgroups=[sample, background])
    target = _one_voxel(sample)
    subtracted = _apply_composite_backgrounds(_CompositeScope(root, sample), target)
    np.testing.assert_allclose(subtracted.signal, target.signal)
    assert not subtracted.mask.item()
    background.masks.clear()
    subtracted = _apply_composite_backgrounds(_CompositeScope(root, sample), target)
    np.testing.assert_allclose(subtracted.signal, target.signal - 0.1 * expected.signal)


def test_correlated_event_copies_merge_before_squaring():
    numerator, variance, events = (np.zeros(3) for _ in range(3))
    _accumulate_correlated_events(
        np.array([[0, 1, 0], [2, -1, 2]]),
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 5.0]),
        np.array([4.0, 9.0]),
        numerator,
        variance,
        events,
    )
    np.testing.assert_allclose(numerator, [8, 4, 20])
    np.testing.assert_allclose(variance, [64, 16, 144])
    np.testing.assert_allclose(events, [1, 1, 1])


def test_accelerated_replay_matches_numpy_for_masks_collisions_and_small_batches(
    tmp_path, monkeypatch
):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    background = mdevent_dataset_group(path)
    if mdevent_background._REPLAY_NUMBA is None:
        pytest.skip("Numba replay backend is not installed")

    # Repeated sample angles and duplicate symmetry transforms force correlated
    # copies of each source event to collide in the same output bin. Signed
    # calibration and a non-unit fit weight exercise the complete weight path.
    sample.datasets = [
        replace(sample.datasets[0], fit_weight=0.25),
        replace(sample.datasets[0], fit_weight=1.75),
    ]
    for source in background.datasets:
        source.scale_factor = -2.5
        source.fit_weight = 0.4
    sample.masks = [
        MaskSpec("masked edge", "energy_q_range", {"energy": [0.6, 1.0]})
    ]
    target = bin_mdevent_group(
        sample,
        lower=[-1, -1, -1, -1],
        upper=[1, 1, 1, 1],
        num_bins=[1, 1, 1, 4],
        symmetry_operations=[np.eye(3), np.eye(3)],
    )

    numpy_block = mdevent_background._replay_numpy_block

    def forbid_numpy_fallback(*_args, **_kwargs):
        raise AssertionError("compiled replay unexpectedly used the NumPy fallback")

    monkeypatch.setattr(mdevent_background, "_replay_numpy_block", forbid_numpy_fallback)
    accelerated = project_measured_background_mdevent(
        sample, background, target, max_batch_bytes=128
    )
    monkeypatch.setattr(mdevent_background, "_replay_numpy_block", numpy_block)
    monkeypatch.setattr(mdevent_background, "_REPLAY_NUMBA", None)
    fallback_progress = []
    fallback = project_measured_background_mdevent(
        sample,
        background,
        target,
        max_batch_bytes=128,
        progress_callback=fallback_progress.append,
    )

    _assert_replay_results_equal(accelerated, fallback)
    assert np.any((accelerated.num_events == 0) & ~accelerated.mask)
    assert np.any(accelerated.num_events > 0)
    assert np.any(accelerated.signal < 0)
    replay_progress = [
        event
        for event in fallback_progress
        if event["stage"] == "mdevent_background_replay"
    ]
    assert replay_progress
    assert {event["backend"] for event in replay_progress} == {"numpy"}
    assert {event["workers"] for event in replay_progress} == {1}
    assert "NumPy fallback" in replay_progress[0]["message"]


@pytest.mark.parametrize("source_frame", ["QSample", "QLab"])
def test_replay_rotates_lab_frame_and_preserves_detector_anisotropy(tmp_path, source_frame):
    import h5py

    source_path, sample_path, artificial_path = (
        tmp_path / name for name in ("source.nxs", "sample.nxs", "artificial.nxs")
    )
    for path in (source_path, sample_path, artificial_path):
        _write_mdevent(path)
    # Two equally exposed detector directions at the same |Q| and energy,
    # differing in signal by 10x. A radial average would destroy this contrast.
    k = np.sqrt(10 / 2.072124855)
    lab = np.array(
        [[-0.5 * k, 0, k * (1 - np.sqrt(3) / 2)], [0, -0.5 * k, k * (1 - np.sqrt(3) / 2)]]
    )
    source_gonio = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    sample_gonios = [np.eye(3), np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])]
    for path in (source_path, sample_path, artificial_path):
        with h5py.File(path, "r+") as handle:
            workspace = handle["MDEventWorkspace"]
            for i in range(2):
                detector = workspace[f"experiment{i}/instrument/physical_detectors"]
                for name, values in [
                    ("detector_number", [10, 11]),
                    ("polar_angle", [30, 30]),
                    ("azimuthal_angle", [0, 90]),
                ]:
                    del detector[name]
                    detector.create_dataset(name, data=values)
                gonio = source_gonio if path == source_path else sample_gonios[i]
                workspace[f"experiment{i}/logs/goniometer/rotation_matrix"][...] = gonio.ravel()
            rows = []
            for i in [0] if path == source_path else [0, 1]:
                gonio = source_gonio if path == source_path else sample_gonios[i]
                q = lab if path == source_path and source_frame == "QLab" else lab @ gonio
                for detector, intensity in enumerate([10.0, 1.0]):
                    weight = 1 if path == source_path else i + 1
                    rows.append(
                        [
                            intensity * weight,
                            intensity * weight**2,
                            i,
                            0,
                            10 + detector,
                            *q[detector],
                            0,
                        ]
                    )
            del workspace["event_data/event_data"]
            workspace["event_data"].create_dataset("event_data", data=np.array(rows))
    source = mdevent_dataset_group(source_path)
    for axis in source.metadata["mdevent"]["dimensions"][:3]:
        axis["frame"] = source_frame
    source.datasets = source.datasets[:1]
    sample = mdevent_dataset_group(sample_path)
    artificial = mdevent_dataset_group(artificial_path)
    settings = dict(lower=[-2, -2, -2, -0.5], upper=[2, 2, 2, 0.5], num_bins=[17, 17, 17, 1])
    expected = bin_mdevent_group(artificial, **settings)
    result = project_measured_background_mdevent(sample, source, expected)
    np.testing.assert_array_equal(result.mask, expected.mask)
    np.testing.assert_allclose(result.signal, expected.signal, equal_nan=True, atol=1e-12)
    positive = result.signal[~result.mask & (result.num_events > 0)]
    assert positive.max() / positive.min() > 5


def test_replay_projection_persists_and_service_is_gui_independent():
    from nfit.pipeline import DatasetGroup

    source, sample = DatasetGroup("background"), DatasetGroup("sample")
    sample.backgrounds = [
        BackgroundSpec(
            "replay", source_group=source, source_group_id=source.id, projection="measured_events"
        )
    ]
    project = NfitProject([DataGroup("workspace", subgroups=[source, sample])])
    restored = _project_from_dict(_project_to_dict(project))
    background = restored.data_groups[0].subgroups[1].backgrounds[0]
    assert background.projection == "measured_events"
    assert background.source_group is restored.data_groups[0].subgroups[0]
    import nfit.mdevent_background as service

    tree = ast.parse(Path(service.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name in {"project_gui", "project_data"} or "PySide" in name for name in imports)


def test_replay_saved_composite_workflow_matches_public_service(tmp_path):
    from nfit import composite_dataset_data, composite_workflow_script, save_project
    from nfit.project_composites import data_group_composite_config

    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample, source = mdevent_dataset_group(path), mdevent_dataset_group(path)
    sample.backgrounds = [
        BackgroundSpec(
            "replay",
            source_group=source,
            source_group_id=source.id,
            projection="measured_events",
            scale=0.25,
        )
    ]
    root = DataGroup("workspace", subgroups=[sample, source])
    config = data_group_composite_config(_CompositeScope(root, sample))
    config.update(enabled=True, auto_rebin=False, minimum_coverage=0)
    for axis in config["axes"]:
        axis.update(
            lower=-1,
            upper=1,
            auto_lower=False,
            auto_upper=False,
            num_bins=1,
            mode="bins",
            auto_step_size=False,
        )
    project = NfitProject([root])
    save_project(project, tmp_path / "replay.nfit")
    expected = composite_dataset_data(root, node=sample)
    script = composite_workflow_script(project, root.name, node_id=sample.id)
    namespace = {"__name__": "replay_test"}
    exec(compile(script, "<replay workflow>", "exec"), namespace)
    actual = namespace["run"]()
    np.testing.assert_allclose(actual.signal, expected.signal)
    np.testing.assert_allclose(actual.errors, expected.errors)
    assert actual.metadata["background_subtractions"][-1]["projection"]["mode"] == "measured_events"


def test_replay_rejects_mismatched_instrument_and_supports_cancellation(tmp_path):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample, source = mdevent_dataset_group(path), mdevent_dataset_group(path)
    target = _one_voxel(sample)
    source.metadata["mdevent"]["incident_energy_override"] = 20
    with pytest.raises(ValueError, match="matching incident energy"):
        project_measured_background_mdevent(sample, source, target)
    source.metadata["mdevent"]["incident_energy_override"] = None

    def cancel(event):
        if event["stage"] == "mdevent_background_replay":
            assert event["iteration"] == 0
            expected = "numba" if mdevent_background._REPLAY_NUMBA is not None else "numpy"
            assert event["backend"] == expected
            assert event["workers"] >= 1
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        project_measured_background_mdevent(sample, source, target, progress_callback=cancel)
