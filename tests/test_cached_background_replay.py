"""Exact cached-field target, streamed primitive covariance and lazy recipes."""

import ast
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nfit import (
    DataGroup,
    DatasetEntry,
    NfitProject,
    clear_cached_background_profile_queries,
    coarsen_measurement_histogram,
    histogram_box_profiles,
    load_project,
    mdevent_dataset_group,
    prepare_histogram_fit_points,
    project_measured_background_mdevent,
    replay_cached_background_profile,
    save_measurement_profile_csv,
    save_project,
)
from nfit.backgrounds import subtract_aligned_background
from nfit.cached_background_replay import CACHED_BACKGROUND_REPLAY
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.measurement_likelihoods import fit_measurement_residuals
from nfit.measurement_profiles import prepare_measurement_profile
from nfit.pipeline import MaskSpec
from nfit.plotting_core import MDHistoSliceViewer
from nfit.project_dataset_io import _load_nfit_dataset_file, save_dataset_file
from tests.test_mdevent_background_statistics import _directional_fixture, _target


@pytest.fixture
def replayed(tmp_path):
    clear_cached_background_profile_queries()
    sample, source = _directional_fixture(tmp_path)
    target = _target(sample, 2)
    background = project_measured_background_mdevent(sample, source, target)
    return target, background


def _profile(data, **kwargs):
    return replay_cached_background_profile(
        data, selected=np.ones(data.shape, bool), indices=np.zeros(data.shape, int),
        edges=np.array([0., 1.]), **kwargs,
    )


def test_cached_background_profile_matches_full_source_covariance_and_exposure(replayed):
    _, background = replayed
    profile = _profile(background)
    # One primitive with C=V=8 and charge2, replayed with fractions1/4,3/4.
    assert profile.data.signal.item() == pytest.approx(4.)
    assert profile.data.errors.item()**2 == pytest.approx(2.)
    assert profile.data.auxiliary_channels["normalization_denominator"].values.item() == pytest.approx(2.)
    assert profile.contract.dependence == "shared_sources"
    assert profile.data.metadata["profile_covariance"] == "replayed_background_sources"
    assert profile.data.metadata["measurement_target"] == "background_exposure_weighted_field_mean"
    assert not profile.data.metadata["cached_background_profile"]["query_cache_hit"]
    assert profile.data.metadata["cached_background_profile"]["normalizer"] == "known"
    assert not background.signal.flags.writeable


def test_subtracted_field_keeps_sample_exposure_target_and_full_background_covariance(replayed):
    sample, background = replayed
    result = subtract_aligned_background(sample, background)
    profile = _profile(result)
    # Pool the already-subtracted field using sample N=(1,3). Sample C=8,V=8,
    # so I_sample=2,V_sample=.5; reused background has I=4,V=2.
    assert profile.data.signal.item() == pytest.approx(-2.)
    assert profile.data.errors.item()**2 == pytest.approx(2.5)
    assert profile.data.metadata["measurement_target"] == "sample_exposure_weighted_subtracted_field_mean"
    assert profile.data.metadata["cached_background_profile"]["sample_uncertainty"] == "recorded_histogram_variance"


def test_signed_reuse_of_background_combines_coefficients_before_variance(replayed):
    sample, background = replayed
    subtracted = subtract_aligned_background(sample, background)
    restored = subtract_aligned_background(subtracted, background, scale=-1.)
    profile = _profile(restored)
    assert profile.data.signal.item() == pytest.approx(2.)
    assert profile.data.errors.item()**2 == pytest.approx(.5)


def test_selected_pixels_and_missing_coverage_keep_their_declared_support(replayed):
    sample, background = replayed
    result = subtract_aligned_background(sample, background)
    selected = np.zeros(result.shape, bool)
    selected[0] = True
    profile = replay_cached_background_profile(
        result, selected=selected, indices=np.zeros(result.shape, int), edges=[0., 1.],
    )
    assert profile.data.signal.item() == pytest.approx(4.)
    assert profile.data.errors.item()**2 == pytest.approx(10.)
    masked = result.with_updates(mask=np.array([False, True]).reshape(result.shape))
    included = _profile(masked)
    assert included.data.signal.item() == pytest.approx(4.)
    assert included.data.errors.item()**2 == pytest.approx(10.)
    assert included.data.auxiliary_channels["coverage_fraction"].values.item() == .5
    assert _profile(masked, coverage_threshold=.6).data.mask.item()


def test_explicit_background_user_exclusion_has_zero_sensitivity_not_missing_coverage(replayed):
    sample, background = replayed
    excluded = np.array([True, False]).reshape(background.shape)
    masked = background.with_updates(mask=excluded,
        metadata={**background.metadata, "nfit_mask": excluded})
    result = subtract_aligned_background(sample, masked)
    profile = _profile(result)
    assert not result.mask.any()
    assert profile.data.signal.item() == pytest.approx(-1.)
    assert profile.data.errors.item()**2 == pytest.approx(1.625)


def test_differently_masked_source_partitions_retain_each_primitive_acceptance(tmp_path):
    import h5py

    sample, source = _directional_fixture(tmp_path)
    path = source.datasets[0].metadata["source_file"]
    with h5py.File(path, "r+") as handle:
        workspace = handle["MDEventWorkspace"]
        first = workspace["event_data/event_data"][()]
        second = first.copy()
        second[:, 2] = 1
        del workspace["event_data/event_data"]
        workspace["event_data"].create_dataset("event_data", data=np.concatenate((first, second)))
    source = mdevent_dataset_group(path)
    source.datasets[0].masks = [MaskSpec("first cell", "coordinate_range", {"H": [-4., -.75]})]
    background = project_measured_background_mdevent(sample, source, _target(sample, 2))
    profile = _profile(background)
    # Source1 enters only the second cell (.75); independent source2 enters
    # both cells (total coefficient1). Charge2 each gives exposure3.5.
    assert profile.data.signal.item() == pytest.approx(4.)
    assert profile.data.errors.item()**2 == pytest.approx(8 * (.75**2 + 1**2) / 3.5**2)


def test_query_cache_reuses_source_covariance_and_does_not_hide_missing_files(replayed, monkeypatch):
    import h5py

    _, background = replayed
    first = _profile(background)
    real_file = h5py.File
    calls = []
    def count_file(*args, **kwargs):
        calls.append(args[0])
        return real_file(*args, **kwargs)
    monkeypatch.setattr(h5py, "File", count_file)
    second = _profile(background)
    assert not calls
    assert second.data.metadata["cached_background_profile"]["query_cache_hit"]
    np.testing.assert_allclose(second.data.errors, first.data.errors)
    path = Path(background.metadata[CACHED_BACKGROUND_REPLAY]["terms"][0]["sources"][0]["source_file"])
    path.unlink()
    with pytest.raises(SourceReplayRequired, match="unavailable"):
        _profile(background)


def test_source_changes_invalidate_query_cache_and_content_digest_is_checked(replayed):
    import h5py

    _, background = replayed
    _profile(background)
    recipe = background.metadata[CACHED_BACKGROUND_REPLAY]["terms"][0]["sources"][0]
    path = Path(recipe["source_file"])
    stat = path.stat()
    with h5py.File(path, "r+") as handle:
        handle["MDEventWorkspace/event_data/event_data"][0, 1] = 80.
    with pytest.raises(SourceReplayRequired, match="changed"):
        _profile(background)
    # Even if a source is replaced preserving its file metadata, a cold query
    # validates actual event bytes, rather than trusting only timestamps.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    clear_cached_background_profile_queries()
    with pytest.raises(SourceReplayRequired, match="contents differ"):
        _profile(background)


def test_recipe_roundtrip_is_lazy_and_exact_profile_export_records_target(replayed, tmp_path, monkeypatch):
    import h5py

    _, background = replayed
    destination = tmp_path / "cached.npz"
    save_dataset_file(DatasetEntry("cached", background), destination, use_view=False)
    real_file = h5py.File
    def no_event_io(*_args, **_kwargs):
        raise AssertionError("Recipe reload read original scientific events")
    monkeypatch.setattr(h5py, "File", no_event_io)
    restored, _ = _load_nfit_dataset_file(destination)
    assert restored.metadata[CACHED_BACKGROUND_REPLAY] == background.metadata[CACHED_BACKGROUND_REPLAY]
    monkeypatch.setattr(h5py, "File", real_file)
    profile = _profile(restored)
    assert profile.data.errors.item()**2 == pytest.approx(2.)
    path = save_measurement_profile_csv(tmp_path / "profile.csv", profile, coordinate_name="Energy")
    receipt = json.loads(path.with_suffix(".csv.json").read_text())
    assert receipt["metadata"]["background_profile_uncertainty"] == "source_covariance"
    assert receipt["metadata"]["measurement_target"] == "background_exposure_weighted_field_mean"
    assert receipt["metadata"]["cached_background_profile"]["sources"][0]["event_sha256"]


def test_saved_project_replay_recipe_loads_without_opening_original_sources(replayed, tmp_path, monkeypatch):
    import h5py

    from nfit.project_imports import _ensure_dataset_data_loaded, dataset_entry_from_path

    _, background = replayed
    destination = tmp_path / "background.nfit"
    portable = tmp_path / "background.npz"
    save_dataset_file(DatasetEntry("Background", background), portable, use_view=False)
    original_entry = dataset_entry_from_path(portable, dataset_file_loader=_load_nfit_dataset_file)
    save_project(NfitProject([DataGroup("Workspace", datasets=[original_entry])]), destination)
    real_file = h5py.File
    def no_events(*_args, **_kwargs):
        raise AssertionError("Lazy project load opened original event sources")
    monkeypatch.setattr(h5py, "File", no_events)
    restored = load_project(destination)
    entry = restored.data_groups[0].datasets[0]
    assert entry.data is None
    _ensure_dataset_data_loaded(entry, dataset_file_loader=_load_nfit_dataset_file)
    assert entry.data.metadata[CACHED_BACKGROUND_REPLAY] == background.metadata[CACHED_BACKGROUND_REPLAY]
    monkeypatch.setattr(h5py, "File", real_file)
    assert _profile(entry.data).data.errors.item()**2 == pytest.approx(2.)


def test_preview_cuts_label_diagonal_uncertainty_and_exact_request_fails_clearly(replayed):
    _, background = replayed
    view = MDHistoSliceViewer(background, x_dim=0, y_dim=3).slice_arrays()
    result = histogram_box_profiles(view, view["signal"], view["errors"],
                                    (view["x_edges"][0], view["x_edges"][-1], -.5, .5))
    assert "diagonal uncertainty" in result.value_label
    assert result.x_measurement.data.metadata["background_profile_uncertainty"].startswith("diagonal_approximation")
    with pytest.raises(SourceReplayRequired, match="diagonal uncertainty"):
        prepare_measurement_profile(view, view["signal"], view["errors"],
            selected=np.ones(view["signal"].shape, bool), indices=np.zeros(view["signal"].shape, int),
            edges=[0., 1.], require_exact_background_uncertainty=True)


def test_primary_change_drops_recipe_and_missing_recipe_does_not_claim_exact_sigma(replayed):
    _, background = replayed
    changed = background.with_updates(signal=background.signal * 2)
    assert CACHED_BACKGROUND_REPLAY not in changed.metadata
    assert CACHED_BACKGROUND_REPLAY not in background.mutable_copy().metadata
    with pytest.raises(SourceReplayRequired, match="retained replay recipe"):
        _profile(changed)
    axes = (replace(background.axes[0], values=np.array([-2., -.7, .5])), *background.axes[1:])
    assert CACHED_BACKGROUND_REPLAY not in background.with_updates(axes=axes).metadata
    assert CACHED_BACKGROUND_REPLAY not in background.with_updates(auxiliary_channels=dict(background.auxiliary_channels)).metadata
    changed_grid = background.with_updates(axes=axes, metadata=background.metadata)
    with pytest.raises(SourceReplayRequired, match="retained replay recipe"):
        _profile(changed_grid)


def test_cached_background_replay_service_has_no_gui_or_mantid_dependencies():
    import nfit.cached_background_replay as service

    tree = ast.parse(Path(service.__file__).read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("PySide" in module or "mantid" in module or module in {"project_gui", "project_data", "plotting_core", "mdevent", "mdevent_background"}
                   for module in modules)


def test_exact_final_bin_sigmas_do_not_certify_cross_bin_independence(replayed):
    _, background = replayed
    profile = replay_cached_background_profile(
        background, selected=np.ones(background.shape, bool),
        indices=np.arange(background.signal.size).reshape(background.shape), edges=[0., 1., 2.],
    )
    assert profile.data.metadata["cross_profile_bin_covariance"].startswith("not_retained")
    with pytest.raises(SourceReplayRequired, match="Shared-source coarsening"):
        coarsen_measurement_histogram(profile.data, ([0., 2.],))
    points = prepare_histogram_fit_points(profile.data)
    with pytest.raises(SourceReplayRequired, match="shared sources"):
        fit_measurement_residuals(points, np.zeros(points.size), "gaussian")


def test_shared_bin_membership_helper_retains_facade_identity_and_edge_rules():
    import nfit.mdevent as facade
    from nfit.event_bin_indices import flat_bin_indices

    assert facade._flat_bin_indices is flat_bin_indices
    coords = np.array([[-1.], [0.], [1.], [2.], [np.nan]])
    np.testing.assert_array_equal(flat_bin_indices(coords, [np.array([-1., 0., 1.])], (2,)), [0, 1, 1, -1, -1])


def test_copied_metadata_cannot_certify_changed_primary_or_exposure(replayed):
    from nfit.mdhisto import MDHistoChannel
    from nfit.measurement_scaling import scale_measurement_data

    _, background = replayed
    copied = dict(background.metadata)
    changed = background.with_updates(signal=2*background.signal,
        errors=2*background.errors, metadata=copied)
    assert CACHED_BACKGROUND_REPLAY not in changed.metadata
    channels = dict(background.auxiliary_channels)
    channels["normalization_denominator"] = MDHistoChannel(
        2*channels["normalization_denominator"].values)
    changed = background.with_updates(auxiliary_channels=channels, metadata=copied)
    assert CACHED_BACKGROUND_REPLAY not in changed.metadata
    scaled = scale_measurement_data(background, 2.)
    assert CACHED_BACKGROUND_REPLAY not in scaled.metadata
    with pytest.raises(SourceReplayRequired, match="retained replay recipe"):
        _profile(scaled)
    assert CACHED_BACKGROUND_REPLAY in background.with_updates(mask=background.mask.copy(), metadata=copied).metadata


def test_oversized_optional_recipe_preserves_histogram_creation_and_preview(tmp_path, monkeypatch):
    import nfit.cached_background_replay as replay

    sample, source = _directional_fixture(tmp_path)
    target = _target(sample, 2)
    monkeypatch.setattr(replay, "MAX_RECIPE_BYTES", 1)
    background = project_measured_background_mdevent(sample, source, target)
    assert np.isfinite(background.signal).all()
    assert CACHED_BACKGROUND_REPLAY not in background.metadata
    assert "storage budget" in background.metadata["background_replay_unavailable_reason"]
    assert background.metadata["background_profile_uncertainty"].startswith("diagonal_approximation")
    with pytest.raises(SourceReplayRequired, match="storage budget"):
        _profile(background)


def test_oversized_subtraction_and_partition_recipes_preserve_numerical_results(replayed, monkeypatch):
    import nfit.cached_background_replay as replay

    sample, background = replayed
    monkeypatch.setattr(replay, "MAX_RECIPE_BYTES", 1)
    subtracted = subtract_aligned_background(sample, background)
    np.testing.assert_allclose(subtracted.signal, sample.signal-background.signal)
    assert CACHED_BACKGROUND_REPLAY not in subtracted.metadata
    with pytest.raises(SourceReplayRequired, match="storage budget"):
        _profile(subtracted)
    metadata, _ = replay.merge_background_replay_partitions(
        [background, background], dict(background.metadata), dict(background.auxiliary_channels))
    assert CACHED_BACKGROUND_REPLAY not in metadata
    assert "storage budget" in metadata["background_replay_unavailable_reason"]


def test_already_subtracted_background_operand_does_not_claim_untracked_exact_variance(replayed):
    sample, background = replayed
    operand = subtract_aligned_background(sample, background)
    result = subtract_aligned_background(sample, operand)
    np.testing.assert_allclose(result.signal, sample.signal-operand.signal)
    assert CACHED_BACKGROUND_REPLAY not in result.metadata
    assert "independent sample primitives" in result.metadata["background_replay_unavailable_reason"]
    assert result.metadata["background_profile_uncertainty"].startswith("diagonal_approximation")
    with pytest.raises(SourceReplayRequired, match="independent sample primitives"):
        _profile(result)
