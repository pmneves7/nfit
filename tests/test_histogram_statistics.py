from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nfit.histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
    event_statistics_channels,
    has_event_statistics,
    normalized_event_statistics,
    poisson_rate_interval,
    pool_event_statistics,
    scaled_event_statistics_channels,
    selected_event_statistics,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DatasetEntry
from nfit.plotting_core import MDHistoSliceViewer, coarsen_mdhisto_view, smooth_mdhisto_view
from nfit.project_dataset_io import _load_nfit_dataset_file, save_dataset_file


def histogram(numerator, variance, exposure, *, mask=None):
    numerator, variance, exposure = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(variance, dtype=float),
        np.asarray(exposure, dtype=float),
    )
    signal, rate_variance = normalized_event_statistics(numerator, variance, exposure)
    return MDHistoData(
        axes=tuple(
            MDHistoAxis(f"Axis {i}", np.arange(size + 1), "", "unknown")
            for i, size in enumerate(signal.shape)
        ),
        signal=signal,
        errors=np.sqrt(rate_variance),
        mask=np.zeros(signal.shape, dtype=bool) if mask is None else mask,
        num_events=np.maximum(numerator, 0),
        metadata={
            EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA),
            "zero_event_bins_are_measured": True,
            "signal_semantics": "density",
        },
        auxiliary_channels=event_statistics_channels(numerator, variance, exposure),
    )


def test_empty_cells_keep_exposure_and_zero_accumulated_variance():
    for cells in (1, 4, 100):
        statistics = pool_event_statistics(
            np.zeros(cells), np.zeros(cells), np.full(cells, 1 / cells), (0,)
        )
        signal, variance = normalized_event_statistics(*statistics)
        assert signal == 0
        assert variance == 0
        assert statistics[2] == pytest.approx(1)
        lower, upper = poisson_rate_interval(0, statistics[2])
        assert lower == 0
        assert upper == pytest.approx(1.8410216450089925, rel=1e-10)


@pytest.mark.parametrize("expected_counts", [0.2, 1.0, 5.0, 20.0])
def test_final_poisson_intervals_cover_rates_under_repeated_sampling(expected_counts):
    rng = np.random.default_rng(17041)
    counts = rng.poisson(expected_counts, size=30_000)
    exposure, weight, confidence = 7.0, 2.5, 0.9
    true_rate = weight * expected_counts / exposure
    lower, upper = poisson_rate_interval(
        counts, exposure, confidence=confidence, constant_weight=weight,
    )
    coverage = np.mean((lower <= true_rate) & (true_rate <= upper))
    # Discrete central intervals are conservative. Allow four sampling standard
    # errors around nominal coverage, without requiring Gaussian count errors.
    tolerance = 4 * np.sqrt(confidence * (1 - confidence) / counts.size)
    assert coverage >= confidence - tolerance


def test_pooling_excludes_invalid_and_masked_statistics_together():
    result = pool_event_statistics(
        [4.0, 0.0, 100.0, 5.0],
        [4.0, 0.0, 100.0, -1.0],
        [1.0, 100.0, 1.0, 1.0],
        (0,),
        mask=[False, False, True, False],
    )
    np.testing.assert_allclose(result, [4, 4, 101])


def test_direct_and_staged_pooling_are_equal_with_unequal_exposure():
    C = np.array([[4.0, 0.0, 40.0], [0.0, 1.0, 9.0]])
    V = C.copy()
    N = np.array([[1.0, 10.0, 100.0], [2.0, 3.0, 5.0]])
    direct = pool_event_statistics(C, V, N, (0, 1))
    first = pool_event_statistics(C, V, N, (1,))
    staged = pool_event_statistics(*first, (0,))
    np.testing.assert_allclose(direct, staged)
    np.testing.assert_allclose(normalized_event_statistics(*direct), [54 / 121, 54 / 121**2])


def test_confidence_interval_is_separate_and_requires_integer_poisson_counts():
    lower, upper = poisson_rate_interval([0, 4], [2, 2], constant_weight=3.0)
    assert lower[0] == 0
    assert upper[0] > 0
    assert lower[1] < 6 < upper[1]
    unit_lower, unit_upper = poisson_rate_interval([0, 4], [2, 2])
    np.testing.assert_allclose(lower, 3 * unit_lower)
    np.testing.assert_allclose(upper, 3 * unit_upper)
    assert np.isnan(poisson_rate_interval(0, 0)[1])
    with pytest.raises(ValueError, match="integers"):
        poisson_rate_interval(0.5, 1)
    with pytest.raises(ValueError, match="constant_weight"):
        poisson_rate_interval(1, 1, constant_weight=-1)


def test_statistics_are_immutable_and_selected_read_preserves_zeros():
    data = histogram(
        [[4.0, 0.0], [40.0, 0.0]], [[4.0, 0.0], [40.0, 0.0]], [[1.0, 10.0], [100.0, 2.0]]
    )
    assert has_event_statistics(data)
    C, V, N = selected_event_statistics(data, (slice(None), slice(0, 1)))
    np.testing.assert_allclose(C.ravel(), [4, 40])
    np.testing.assert_allclose(V.ravel(), [4, 40])
    np.testing.assert_allclose(N.ravel(), [1, 100])
    assert not data.auxiliary_channels[EVENT_VARIANCE_NUMERATOR].values.flags.writeable
    assert data.signal[0, 1] == data.errors[0, 1] == 0


def test_primary_replacement_drops_stale_statistics_but_metadata_update_preserves():
    data = histogram([[4.0]], [[4.0]], [[1.0]])
    updated = data.with_updates(signal=np.array([[8.0]]))
    assert not has_event_statistics(updated)
    assert EVENT_STATISTICS_KEY not in updated.metadata
    assert EVENT_SIGNAL_NUMERATOR not in updated.auxiliary_channels
    assert NORMALIZATION_DENOMINATOR in updated.auxiliary_channels
    assert has_event_statistics(data.with_updates(metadata=dict(data.metadata, title="new")))
    # dataclasses.replace is another compatibility path. Validation refuses
    # inherited statistics even when that path cannot remove their channels.
    assert selected_event_statistics(replace(data, signal=np.array([[8.0]]))) is None
    background = replace(data, metadata=dict(data.metadata, background_subtractions=[{}]))
    assert selected_event_statistics(background) is None


def test_histogram_archive_round_trip_retains_exact_statistics(tmp_path):
    data = histogram([[4.0, 0.0]], [[4.0, 0.0]], [[1.0, 100.0]])
    path = tmp_path / "events.npz"
    save_dataset_file(DatasetEntry("Events", data), path, use_view=False)
    restored, _ = _load_nfit_dataset_file(path)
    assert has_event_statistics(restored)
    for actual, expected in zip(
        selected_event_statistics(restored), selected_event_statistics(data), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable


def test_slice_and_coarsening_preserve_additive_statistics_and_measured_zeros():
    C = np.array([4.0, 0.0, 40.0, 0.0]).reshape(2, 2, 1, 1)
    V = C.copy()
    N = np.array([1.0, 10.0, 100.0, 2.0]).reshape(C.shape)
    mask = np.zeros(C.shape, dtype=bool)
    mask[0, 0, 0, 0] = True
    data = histogram(C, V, N, mask=mask)
    model = MDHistoSliceViewer(data, x_dim=1, y_dim=3, integrate=True)
    model.selections = {0: (0.5, 1.5), 2: 0}
    view = model.slice_arrays()
    np.testing.assert_allclose(view[EVENT_SIGNAL_NUMERATOR], [[40, 0]])
    np.testing.assert_allclose(view[EVENT_VARIANCE_NUMERATOR], [[40, 0]])
    np.testing.assert_allclose(view[NORMALIZATION_DENOMINATOR], [[100, 12]])
    assert view["errors"][0, 1] == 0
    assert not view["mask"][0, 1]
    coarse = coarsen_mdhisto_view(view, x_step=2)
    assert coarse["signal"].item() == pytest.approx(40 / 112)
    assert coarse["errors"].item() == pytest.approx(np.sqrt(40) / 112)
    assert coarse[EVENT_SIGNAL_NUMERATOR].item() == 40
    assert coarse[EVENT_VARIANCE_NUMERATOR].item() == 40
    assert coarse[NORMALIZATION_DENOMINATOR].item() == 112


def test_stale_statistics_are_not_published_by_slice():
    data = histogram([[4.0, 0.0]], [[4.0, 0.0]], [[1.0, 100.0]])
    stale = replace(data, signal=np.array([[8.0, 0.0]]))
    model = MDHistoSliceViewer(stale, x_dim=1, y_dim=0, channel=EVENT_SIGNAL_NUMERATOR)
    view = model.slice_arrays()
    assert EVENT_SIGNAL_NUMERATOR not in view
    assert EVENT_VARIANCE_NUMERATOR not in view
    assert model.channel == "signal"


def test_smoothing_does_not_claim_independent_event_statistics():
    data = histogram([[4.0, 0.0]], [[4.0, 0.0]], [[1.0, 100.0]])
    view = MDHistoSliceViewer(data, x_dim=1, y_dim=0).slice_arrays()
    smoothed = smooth_mdhisto_view(view, sigma_x=1)
    assert EVENT_STATISTICS_KEY not in smoothed
    assert EVENT_SIGNAL_NUMERATOR not in smoothed
    assert EVENT_VARIANCE_NUMERATOR not in smoothed
    assert EVENT_STATISTICS_KEY in view


def test_histogram_statistics_has_no_gui_or_facade_dependencies():
    source = Path(__file__).parents[1] / "src/nfit/histogram_statistics.py"
    tree = ast.parse(source.read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        name.startswith(("PySide", "PyQt", "project_gui", "project_data", "plotting"))
        for name in modules
    )


def test_known_scaling_uses_squared_factor_for_event_variance():
    data = histogram([[4.0, 0.0]], [[4.0, 0.0]], [[1.0, 100.0]])
    factors = np.array([[2.0, 3.0]])
    channels = scaled_event_statistics_channels(data, factors)
    scaled = data.with_updates(
        signal=data.signal * factors, errors=data.errors * factors, auxiliary_channels=channels
    )
    C, V, N = selected_event_statistics(scaled)
    np.testing.assert_allclose(C, [[8, 0]])
    np.testing.assert_allclose(V, [[16, 0]])
    np.testing.assert_allclose(N, [[1, 100]])
    assert channels[NORMALIZATION_DENOMINATOR] is data.auxiliary_channels[NORMALIZATION_DENOMINATOR]


def test_count_view_labels_distinguish_observed_variance_from_rate_intervals():
    data = histogram([[4.0, 0.0]], [[4.0, 0.0]], [[1.0, 100.0]])
    viewer = MDHistoSliceViewer(data, x_dim=1, y_dim=0)
    assert viewer.CHANNEL_LABELS["errors"] == "Observed event standard error"
    assert viewer.CHANNEL_LABELS["num_events"] == "Event contributions"


def test_stale_masked_statistics_cannot_override_unmasked_view():
    data = histogram([[4.0]], [[4.0]], [[1.0]], mask=np.array([[True]]))
    stale = replace(data, signal=np.array([[8.0]]))
    assert selected_event_statistics(stale) is None
    viewer = MDHistoSliceViewer(stale, x_dim=1, y_dim=0, masked=False)
    view = viewer.slice_arrays()
    assert view["signal"].item() == 8
    assert EVENT_SIGNAL_NUMERATOR not in view


def test_project_catalog_keeps_count_statistics_lazy_until_selected(tmp_path, monkeypatch):
    import nfit.project_gui as project_gui
    import nfit.rebin_cache as rebin_cache
    from nfit.pipeline import DataGroup
    from nfit.project_io import NfitProject

    original = histogram(
        np.array([4.0, 0.0, 40.0, 0.0]).reshape(1, 2, 1, 2),
        np.array([4.0, 0.0, 40.0, 0.0]).reshape(1, 2, 1, 2),
        np.array([1.0, 10.0, 100.0, 2.0]).reshape(1, 2, 1, 2),
    )
    source = tmp_path / "source.npz"
    save_dataset_file(DatasetEntry("Events", original), source, use_view=False)
    dataset = DatasetEntry(
        "Events", original, kind="mdhisto", metadata={"source_file": str(source)}
    )
    dataset.replace_data(original, source_backed=True)
    config = project_gui.dataset_rebin_config(dataset)
    config.update(enabled=True, minimum_coverage=0.0, mean_weighting="uniform")
    expected = project_gui.dataset_for_slice_viewer(dataset)
    assert selected_event_statistics(expected) is not None
    path = tmp_path / "events.nfit"
    project_gui.save_project(
        NfitProject(
            [DataGroup("Counts", datasets=[dataset])],
            settings={project_gui.PROJECT_CACHE_BINNINGS_KEY: True},
        ),
        path,
    )
    project_gui._VIEWER_VIEW_CACHE.clear()
    decoded = []
    reader = rebin_cache.read_project_dataset_artifact

    def read(*args, **kwargs):
        decoded.append(args)
        return reader(*args, **kwargs)

    monkeypatch.setattr(rebin_cache, "read_project_dataset_artifact", read)
    with monkeypatch.context() as catalog_guard:
        catalog_guard.setattr(
            project_gui,
            "read_project_dataset_artifact",
            lambda *args, **kwargs: pytest.fail("project/catalog decoded arrays"),
        )
        restored = project_gui.load_project(path)
        restored_dataset = restored.data_groups[0].datasets[0]
        assert restored_dataset.data is None
        catalog, names = project_gui.slice_viewer_datasets(
            restored.data_groups[0], use_composite=False, preload=False
        )
        assert names == ["Events"]
        assert catalog.cached_indices == ()
        assert not decoded
    actual = catalog[0]
    assert len(decoded) == 1
    assert catalog.cached_indices == (0,)
    for values, wanted in zip(
        selected_event_statistics(actual), selected_event_statistics(expected), strict=True
    ):
        np.testing.assert_array_equal(values, wanted)
        assert not values.flags.writeable
    assert catalog[0] is actual
    assert len(decoded) == 1
    project_gui._VIEWER_VIEW_CACHE.clear()
