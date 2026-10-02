"""Tracked import aliases remain dependent through reduction and fit preparation."""

import numpy as np
import pytest

from nfit import DataGroup, bin_corelli_group, bin_mdevent_group, bin_raw_dgs_group
from nfit.analysis.data_reduction import combine_aligned_histograms
from nfit.backgrounds import subtract_aligned_background
from nfit.fitting import FitDataset, FitProblem, ParameterSpec, fit_problem_least_squares
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_dependencies import SourceDependencies, SourceReplayRequired
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.source_lineage import (
    SOURCE_LINEAGE_KEY,
    merge_source_lineage_metadata,
    source_identity,
    source_lineage_metadata,
    tracked_source_ids,
    with_source_lineage,
)
from nfit.source_selection import SourceSelection
from nfit.source_selection_imports import import_source_selection
from tests.test_corelli import _write_corelli
from tests.test_mdevent import _write_mdevent
from tests.test_raw_dgs import _write_raw_dgs


def histogram(identity=None):
    metadata = {} if identity is None else {SOURCE_LINEAGE_KEY: {"version": 1, "source_ids": [identity]}}
    return MDHistoData((MDHistoAxis("DeltaE", [0, 1, 2], "meV", "energy"),),
                      [2., 3.], [1., 1.], [False, False], [2., 3.], metadata=metadata)


def problem(points, *, second_weight=1):
    return FitProblem([FitDataset("first", points[0]), FitDataset("second", points[1], weight=second_weight)],
                      lambda data, params: np.full(data.size, params["rate"]),
                      [ParameterSpec("rate", 1., vary=False)])


def test_canonical_identity_separates_experiments_and_independent_streams(tmp_path):
    path = tmp_path / "source.nxs"
    path.touch()
    alias = tmp_path / "alias.nxs"
    alias.symlink_to(path)
    assert source_identity(path) == source_identity(alias)
    assert source_identity(path, experiment_index=0) != source_identity(path, experiment_index=1)
    assert source_identity(path, importer="scan", stream="up") != source_identity(path, importer="scan", stream="down")


def test_metadata_is_bounded_explicit_and_immutable(monkeypatch):
    import nfit.source_lineage as lineage

    original = histogram()
    assert with_source_lineage(original, {}) is original
    marked = with_source_lineage(original, histogram("first").metadata)
    assert tracked_source_ids(original) == ()
    assert tracked_source_ids(marked) == ("first",)
    assert not marked.signal.flags.writeable
    assert merge_source_lineage_metadata(marked, histogram("second"))[SOURCE_LINEAGE_KEY]["source_ids"] == ["first", "second"]
    with pytest.raises(ValueError, match="unsupported"):
        tracked_source_ids({SOURCE_LINEAGE_KEY: {"version": 2, "source_ids": []}})
    monkeypatch.setattr(lineage, "MAX_TRACKED_SOURCE_IDS", 1)
    with pytest.raises(SourceReplayRequired, match="budget"):
        merge_source_lineage_metadata(marked, histogram("second"))


@pytest.mark.parametrize("family", ["raw", "mdevent", "corelli"])
def test_native_histograms_and_real_fit_preparation_reject_separate_alias_blocks(tmp_path, family):
    if family == "raw":
        _write_raw_dgs(tmp_path / "run1.nxs")
        reducer = bin_raw_dgs_group
        settings = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 40], num_bins=[1]*4)
    elif family == "mdevent":
        _write_mdevent(tmp_path / "run1.nxs")
        reducer = bin_mdevent_group
        settings = dict(lower=[-1]*4, upper=[1]*4, num_bins=[1]*4)
    else:
        _write_corelli(tmp_path / "run1.nxs")
        reducer = bin_corelli_group
        settings = dict(lower=[-10, -10, -10, -.5], upper=[10, 10, 10, .5], num_bins=[1]*4)
    selected = import_source_selection(DataGroup("sample"), SourceSelection(tmp_path, "run", ".nxs", "1|2|"), preserve_groups=True)
    left, right = selected.subgroups
    histograms = [reducer(group, **settings) for group in (left, right)]
    assert tracked_source_ids(histograms[0]) == tracked_source_ids(histograms[1])
    assert tracked_source_ids(histograms[0])
    points = [prepare_histogram_fit_points(data) for data in histograms]
    assert tracked_source_ids(points[0]) == tracked_source_ids(histograms[0])
    with pytest.raises(SourceReplayRequired, match="another fit dataset"):
        fit_problem_least_squares(problem(points))
    result = fit_problem_least_squares(problem(points, second_weight=0))
    assert result.dataset_weights["second"] == 0
    if family == "corelli":
        # This adapter still processes enabled inputs independently of fit_weight.
        left.datasets[0].fit_weight = 0
        assert tracked_source_ids(reducer(left, **settings))
    for dataset in left.datasets:
        dataset.enabled = False
    assert source_lineage_metadata(left.datasets) == {}


def test_fit_unique_sources_and_unmarked_legacy_are_allowed():
    points = [prepare_histogram_fit_points(histogram(identity)) for identity in ("a", "b")]
    assert fit_problem_least_squares(problem(points)).dataset_sizes == {"first": 2, "second": 2}
    legacy = prepare_histogram_fit_points(histogram())
    assert fit_problem_least_squares(problem([legacy, legacy])).dataset_sizes == {"first": 2, "second": 2}


def test_entry_context_retains_lineage_without_mutating_leaf_points():
    from nfit.pipeline import DatasetEntry
    from nfit.project_gui import _apply_sample_context_to_points

    points = prepare_histogram_fit_points(histogram())
    entry = DatasetEntry("ordinary", points, metadata={"source_selection_lineage": {"version": 1, "source_identity": "ordinary#stream=up"}})
    contextual = _apply_sample_context_to_points(DataGroup("sample"), entry, points)
    assert tracked_source_ids(contextual) == ("ordinary#stream=up",)
    assert tracked_source_ids(points) == ()


def test_arithmetic_rejects_independent_aliases_and_keeps_both_unique_sources():
    left, right = histogram("a"), histogram("b")
    assert tracked_source_ids(combine_aligned_histograms(left, right)) == ("a", "b")
    assert tracked_source_ids(subtract_aligned_background(left, right)) == ("a", "b")
    for operation in (combine_aligned_histograms, subtract_aligned_background):
        with pytest.raises(SourceReplayRequired, match="overlapping tracked sources"):
            operation(left, left)
    assert tracked_source_ids(combine_aligned_histograms(left, left, right_scale=0)) == ("a",)


def test_background_joint_covariance_path_can_subtract_shared_sources():
    data = histogram("a")
    dependencies = SourceDependencies(shape=data.shape, source_ids=("p0", "p1"),
        source_variances=np.ones(2), observation_indices=np.arange(2), source_indices=np.arange(2), coefficients=np.ones(2))
    data = data.with_updates(source_dependencies=dependencies)
    difference = subtract_aligned_background(data, data)
    np.testing.assert_array_equal(difference.signal, [0, 0])
    np.testing.assert_array_equal(difference.errors, [0, 0])
    assert tracked_source_ids(difference) == ("a",)


def test_marked_composite_cache_signature_and_cached_context_include_lineage():
    from nfit import project_composites
    from nfit.pipeline import DatasetEntry

    project_composites._COMPOSITE_DATA_CACHE.clear()
    entry = DatasetEntry("scan", histogram(), kind="mdhisto")
    group = DataGroup("combined", datasets=[entry])
    project_composites.data_group_composite_config(group)["enabled"] = True
    signature = project_composites._composite_cache_signature(group)
    legacy = project_composites._cached_composite_dataset_data(group)
    assert tracked_source_ids(legacy) == ()
    entry.metadata["source_selection_lineage"] = {"version": 1, "source_identity": "first"}
    assert project_composites._composite_cache_signature(group) != signature
    marked = project_composites._cached_composite_dataset_data(group)
    assert tracked_source_ids(marked) == ("first",)
    assert tracked_source_ids(project_composites.composite_dataset_entry(group)) == ("first",)
    assert tracked_source_ids(project_composites._cached_composite_dataset_data(group)) == ("first",)
    assert tracked_source_ids(legacy) == ()


def test_zero_weight_enabled_alias_still_rejected_for_native_coaddition():
    from nfit.pipeline import DatasetEntry
    from nfit.source_selection_imports import validate_source_selection_combination

    entries = [DatasetEntry(name, histogram(), metadata={"source_selection_lineage": {"version": 1, "source_identity": "same"}}) for name in ("a", "b")]
    entries[1].fit_weight = 0
    with pytest.raises(SourceReplayRequired, match="reuse"):
        validate_source_selection_combination(entries)
    entries[1].enabled = False
    validate_source_selection_combination(entries)


def test_public_measurement_pooling_requires_covariance_for_tracked_aliases():
    from nfit.measurement_contracts import MeasurementContract
    from nfit.measurement_rebinning import combine_measurement_histograms

    contract = MeasurementContract(kind="continuous", estimator="uniform_mean",
                                   quantity="response", value_units="a.u.")
    first = histogram("a")
    first = first.with_updates(metadata={**first.metadata, "measurement_contract": contract.to_dict()})
    with pytest.raises(SourceReplayRequired, match="overlapping tracked sources"):
        combine_measurement_histograms([first, first])
    second = first.with_updates(metadata={**first.metadata, SOURCE_LINEAGE_KEY: {
        "version": 1, "source_ids": ["b"],
    }})
    pooled = combine_measurement_histograms([first, second])
    np.testing.assert_allclose(pooled.signal, first.signal)
    np.testing.assert_allclose(pooled.errors, first.errors / np.sqrt(2))
    assert tracked_source_ids(pooled) == ("a", "b")
    dependencies = SourceDependencies(
        shape=first.shape, source_ids=("p0", "p1"),
        source_variances=np.ones(2), observation_indices=np.arange(2),
        source_indices=np.arange(2), coefficients=np.ones(2),
    )
    tracked = first.with_updates(source_dependencies=dependencies)
    correlated = combine_measurement_histograms([tracked, tracked])
    np.testing.assert_allclose(correlated.signal, first.signal)
    np.testing.assert_allclose(correlated.errors, first.errors)
    assert tracked_source_ids(correlated) == ("a",)
