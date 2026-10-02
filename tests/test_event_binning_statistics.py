"""Shared-event covariance reaches both native and saved-MDE accumulators."""

import numpy as np
import pytest

from nfit import (
    bin_mdevent_group,
    bin_raw_dgs_group,
    mdevent,
    mdevent_dataset_group,
    raw_dgs_dataset_group,
)
from nfit.histogram_statistics import selected_event_statistics
from tests.test_mdevent import _write_mdevent
from tests.test_raw_dgs import _write_raw_dgs


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("source_kind", ["raw", "mde"])
@pytest.mark.parametrize("variance_policy", ["independent_copies", "within_bin_covariance"])
def test_repeated_symmetry_copy_follows_selected_variance_policy(tmp_path, monkeypatch, compiled, source_kind, variance_policy):
    if compiled and mdevent._MDEVENT_NUMBA is None:
        pytest.skip("Numba backend is unavailable")
    if not compiled:
        monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    if source_kind == "raw":
        source = tmp_path / "SEQ_42.nxs.h5"
        _write_raw_dgs(source)
        group, bin_data = raw_dgs_dataset_group([source]), bin_raw_dgs_group
    else:
        source = tmp_path / "events.nxs"
        _write_mdevent(source)
        group, bin_data = mdevent_dataset_group(source), bin_mdevent_group
    group.metadata["raw_dgs" if source_kind == "raw" else "mdevent"]["symmetry_variance_policy"] = variance_policy
    options = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1]*4)
    original = bin_data(group, **options)
    copied = bin_data(group, symmetry_operations=[np.eye(3), np.eye(3)], **options)
    original_stats = selected_event_statistics(original)
    copied_stats = selected_event_statistics(copied)
    assert original.num_events.item() > 0
    for source_values, copy_values, multiplier in zip(original_stats, copied_stats, (2, 4 if variance_policy == "within_bin_covariance" else 2, 2), strict=True):
        np.testing.assert_allclose(copy_values, multiplier * source_values)
    np.testing.assert_allclose(copied.signal, original.signal)
    np.testing.assert_allclose(copied.errors, original.errors if variance_policy == "within_bin_covariance" else original.errors / np.sqrt(2))
    np.testing.assert_equal(copied.num_events, 2 * original.num_events)
    assert copied.metadata["symmetry_covariance"]["within_bin_pairs_corrected"] == (original.num_events.item() if variance_policy == "within_bin_covariance" else 0)


@pytest.mark.parametrize("compiled", [False, True])
def test_covariance_bin_indices_match_accumulation_at_edges(tmp_path, monkeypatch, compiled):
    if compiled and mdevent._MDEVENT_NUMBA is None:
        pytest.skip("Numba backend is unavailable")
    if not compiled:
        monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    coordinates = np.array([[0., 0.], [.5, .5], [1., 1.], [-.1, .2], [np.nan, .2], [.2, .2]])
    indices = np.full(len(coordinates), -1, dtype=np.int64)
    sums, variances, counts = (np.zeros((2, 2)) for _ in range(3))
    mdevent._accumulate_discrete_event_coordinates(
        coordinates, np.ones(6), None, [np.array([0., .5, 1.])]*2, (2, 2),
        sums, variances, counts, enabled=np.array([True]*5+[False]), bin_indices=indices,
    )
    np.testing.assert_equal(indices, [0, 3, 3, -1, -1, -1])
    np.testing.assert_equal(counts.ravel(), [1, 0, 0, 2])
