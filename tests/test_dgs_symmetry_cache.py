"""Incremental raw symmetry obeys source guards and additive-statistic semantics."""

import h5py
import numpy as np
import pytest

from nfit import (
    DataGroup,
    MaskSpec,
    add_data_group_composite_binning,
    composite_dataset_data,
    raw_dgs_dataset_group,
    refresh_composite_dataset,
)
from nfit import project_composites as composites
from nfit.dgs_symmetry_cache import (
    symmetry_remainder,
    symmetry_reuse_worthwhile,
    try_expand_raw_dgs_symmetry,
)
from nfit.histogram_statistics import event_statistics_channels
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.project_composites import data_group_composite_config, data_group_composite_config_by_id
from nfit.symmetry import SymmetrySpec, symmetry_config
from tests.test_raw_dgs import _write_raw_dgs

SMALL = "-x,-y,-z;y,z,x"
LARGE = "x,y,z;y,z,x;z,x,y;-x,-y,-z"


def _config(group, expression, *, minimum_samples=0):
    config = data_group_composite_config(group)
    config.update(enabled=True, auto_rebin=True, minimum_coverage=0., minimum_samples=minimum_samples,
                  symmetry=symmetry_config(SymmetrySpec("operations", expression)))
    for dimension, axis in enumerate(config["axes"]):
        axis.update(lower=-10. if dimension < 3 else -19., upper=10. if dimension < 3 else 19.,
                    step_size=10. if dimension < 3 else 9.5, num_bins=2 if dimension < 3 else 4,
                    vector=np.eye(4)[dimension].tolist(), auto_lower=False, auto_upper=False,
                    auto_step_size=False, mode="custom")
    return config


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source, with_he3=True)
    # Unequal corrected event weights and multiple streamed reductions exercise
    # addition association; copies need not form a prefix of the target list.
    with h5py.File(source, "r+") as archive:
        bank = archive["entry/bank1_events"]
        del bank["event_id"], bank["event_time_offset"]
        bank.create_dataset("event_id", data=np.full(37, 42))
        bank.create_dataset("event_time_offset", data=np.linspace(8000., 11500., 37))
    native = raw_dgs_dataset_group([source])
    root = DataGroup("sample", subgroups=[native])
    config = _config(native, SMALL, minimum_samples=50)
    composites._COMPOSITE_DATA_CACHE.clear()
    yield root, native, config, source
    composites._COMPOSITE_DATA_CACHE.clear()


def _compare(actual, expected):
    np.testing.assert_array_equal(actual.num_events, expected.num_events)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    for name in ("event_signal_numerator", "event_variance_numerator", "normalization_denominator"):
        np.testing.assert_allclose(actual.auxiliary_channels[name].values, expected.auxiliary_channels[name].values,
                                   rtol=1e-12, atol=0)
    np.testing.assert_allclose(actual.signal, expected.signal, rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_array_equal(actual.metadata["normalization_denominator"] > 0,
                                  expected.metadata["normalization_denominator"] > 0)


def _expanded(case, monkeypatch):
    root, group, config, _source = case
    cached = refresh_composite_dataset(root, node=group)
    binning = add_data_group_composite_binning(group, name="expanded")
    desired = data_group_composite_config_by_id(group, binning)
    desired["symmetry"] = symmetry_config(SymmetrySpec("operations", LARGE))
    calls = []
    original = composites.bin_raw_dgs_group
    def record(*args, **kwargs):
        calls.append(kwargs["symmetry_operations"])
        return original(*args, **kwargs)
    monkeypatch.setattr(composites, "bin_raw_dgs_group", record)
    return root, group, cached, binning, desired, calls


def test_public_refresh_expands_nonprefix_cache_and_recomputes_sample_and_coverage_masks(case, monkeypatch):
    root, group, cached, binning, desired, calls = _expanded(case, monkeypatch)
    before = cached.num_events.copy()
    actual = refresh_composite_dataset(root, node=group, binning_id=binning)
    assert len(calls) == 1 and len(calls[0]) == 2
    assert actual.metadata["incremental_symmetry"]["computed_operations"] == 2
    expected = composite_dataset_data(root, node=group, config_override=desired)
    _compare(actual, expected)
    assert np.any(cached.mask & ~actual.mask), "Threshold must be applied to the combined counts"
    np.testing.assert_array_equal(cached.num_events, before)
    assert actual.metadata["symmetry_operations_hkl"] == expected.metadata["symmetry_operations_hkl"]
    np.testing.assert_array_equal(actual.auxiliary_channels["coverage_fraction"].values,
                                  expected.auxiliary_channels["coverage_fraction"].values)


@pytest.mark.parametrize("change", ["source", "weight", "scale", "ub", "policy", "mask", "grid", "calibration", "auto"])
def test_guard_changes_force_full_native_replay(case, monkeypatch, change):
    root, group, cached, binning, desired, calls = _expanded(case, monkeypatch)
    source = case[3]
    if change == "source":
        with h5py.File(source, "r+") as archive:
            archive["entry/bank1_events/event_time_offset"][0] += 1
    elif change == "weight":
        group.datasets[0].fit_weight = 2
    elif change == "scale":
        group.datasets[0].scale_factor = 2
    elif change == "ub":
        group.metadata["raw_dgs"]["ub_matrix"][0][0] *= 1.01
    elif change == "policy":
        group.metadata["raw_dgs"]["symmetry_variance_policy"] = "within_bin_covariance"
    elif change == "mask":
        group.datasets[0].masks.append(MaskSpec("box", {"lower": [-1]*4, "upper": [1]*4}))
    elif change == "grid":
        desired["axes"][0]["upper"] = 20.
    elif change == "calibration":
        # Source settings change even when the new file gives the same data.
        calibration = source.parent / "detector.nxs"
        with h5py.File(calibration, "w") as archive:
            entry = archive.create_group("mantid_workspace_1")
            entry.create_dataset("instrument/detector/detector_list", data=[42])
            entry.create_dataset("workspace/values", data=[[1.]])
            entry.create_dataset("workspace/errors", data=[[0.]])
        group.metadata["raw_dgs"]["mask_file"] = str(calibration)
    elif change == "auto":
        desired["axes"][0]["auto_lower"] = True
    actual = refresh_composite_dataset(root, node=group, binning_id=binning)
    assert "incremental_symmetry" not in actual.metadata
    assert len(calls) == 1 and len(calls[0]) == 4


def test_missing_statistics_and_changed_cached_primary_force_replay(case, monkeypatch):
    root, group, cached, binning, desired, calls = _expanded(case, monkeypatch)
    key = composites._composite_cache_key(composites._composite_scope(root, group))
    signature, _ = composites._COMPOSITE_DATA_CACHE.get(key)
    stripped = cached.with_updates(auxiliary_channels={})
    composites._COMPOSITE_DATA_CACHE[key] = (signature, stripped)
    actual = refresh_composite_dataset(root, node=group, binning_id=binning)
    assert len(calls[0]) == 4 and "incremental_symmetry" not in actual.metadata


def test_custom_backend_and_mde_keep_full_replay(case, monkeypatch):
    root, group, config, source = case
    scope = composites._composite_scope(root, group)
    with monkeypatch.context() as context:
        context.setattr(composites, "_backend_value", lambda *args: lambda *args, **kwargs: None)
        assert composites._incremental_raw_dgs_base(scope, config, None) is None
    group.metadata["raw_dgs"]["format"] = "corelli-correlation-nexus"
    assert composites._incremental_raw_dgs_base(scope, config, None) is None
    group.metadata["raw_dgs"]["format"] = "mantid-mdevent"
    assert composites._incremental_raw_dgs_base(scope, config, None) is None


def test_exact_multisets_preserve_duplicates_and_requested_order():
    first, second, third = np.eye(3), -np.eye(3), np.diag([-1., 1., 1.])
    missing = symmetry_remainder([second, first, first], [first, third, first, second, first])
    np.testing.assert_array_equal(missing, [third, first])
    assert symmetry_remainder([first, first], [first, second]) is None
    assert symmetry_remainder([first], [first]) is None
    changed = first.copy()
    changed[0, 0] = np.nextafter(1., 2.)
    assert symmetry_remainder([changed], [first, second]) is None


def _synthetic_histogram(numerator, variance, exposure, counts, operations):
    shape = (1, 1, 1, len(counts))
    c, v, n, counts = (np.asarray(array, dtype=float).reshape(shape) for array in (numerator, variance, exposure, counts))
    with np.errstate(divide="ignore", invalid="ignore"):
        signal, errors = c / n, np.sqrt(v) / n
    return MDHistoData(
        axes=tuple(MDHistoAxis(str(i), np.arange(size + 1), "", "unknown") for i, size in enumerate(shape)),
        signal=signal, errors=errors, mask=n <= 0, num_events=counts,
        metadata={"raw_dgs": {"format": "raw-direct-geometry-nexus"},
                  "event_statistics": {"version": 1}, "symmetry_operations_hkl": operations,
                  "dgs_reduction_policies": {"symmetry_variance_policy": "independent_copies"}},
        auxiliary_channels=event_statistics_channels(c, v, n))


def test_zero_partial_exposure_retains_counts_and_signed_numerator_before_final_normalization():
    first, second = np.eye(3), -np.eye(3)
    base = _synthetic_histogram([3., 0., -1.], [4., 0., 2.], [0., 2., 1.], [1., 0., 1.], [first.tolist()])
    extra = _synthetic_histogram([1., 0., 2.], [1., 0., 4.], [2., 3., 0.], [1., 0., 1.], [second.tolist()])
    actual = try_expand_raw_dgs_symmetry(
        {"symmetry": [first, second], "minimum_samples": 0},
        candidates=[({"symmetry": [first]}, lambda: base)], signature=lambda config: "same",
        operations=lambda config: config["symmetry"], reduce_missing=lambda matrices: extra, finalize=lambda data: data, minimum_samples=0)
    np.testing.assert_array_equal(actual.signal.ravel(), [2., 0., 1.])
    np.testing.assert_array_equal(actual.num_events.ravel(), [2., 0., 2.])
    assert not actual.mask.any()
    assert actual.errors.ravel()[1] == 0


def test_memory_limit_and_stale_statistics_fallback_without_reduction(monkeypatch):
    from nfit import dgs_symmetry_cache as cache
    first, second = np.eye(3), -np.eye(3)
    base = _synthetic_histogram([1.], [1.], [1.], [1.], [first.tolist()])
    kwargs = dict(candidates=[({"symmetry": [first]}, lambda: base)], signature=lambda config: "same",
                  operations=lambda config: config["symmetry"],
                  reduce_missing=lambda matrices: pytest.fail("Do not reduce an unsupported candidate"), finalize=lambda data: data, minimum_samples=0)
    with monkeypatch.context() as context:
        context.setattr(cache, "scientific_memory_limit_bytes", lambda: 1)
        assert try_expand_raw_dgs_symmetry({"symmetry": [first, second]}, **kwargs) is None
    base = base.with_updates(signal=base.signal * 2, auxiliary_channels=base.auxiliary_channels)
    assert try_expand_raw_dgs_symmetry({"symmetry": [first, second]}, **kwargs) is None


def test_saved_project_lazy_base_and_collection_scaling_remain_separate(case, tmp_path, monkeypatch):
    from nfit import NfitProject, configure_composite_scaling, load_project, save_project

    root, group, config, source = case
    configure_composite_scaling(root, node=group, data_scale=3, result_scale=2)
    cached = refresh_composite_dataset(root, node=group)
    project = NfitProject([root], settings={"cache_binnings": True})
    path = tmp_path / "symmetry.nfit"
    save_project(project, path)
    reopened = load_project(path)
    restored_root, restored = reopened.data_groups[0], reopened.data_groups[0].subgroups[0]
    binning = add_data_group_composite_binning(restored, name="expanded")
    desired = data_group_composite_config_by_id(restored, binning)
    desired["symmetry"] = symmetry_config(SymmetrySpec("operations", LARGE))
    calls = []
    original = composites.bin_raw_dgs_group
    def record(*args, **kwargs):
        calls.append(len(kwargs["symmetry_operations"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(composites, "bin_raw_dgs_group", record)
    actual = refresh_composite_dataset(restored_root, node=restored, binning_id=binning)
    assert calls == [2]
    expected = composite_dataset_data(restored_root, node=restored, config_override=desired)
    _compare(actual, expected)
    assert np.all(cached.num_events >= 0)


def test_largest_subset_is_preferred_without_loading_smaller_bases():
    first, second, third = np.eye(3), -np.eye(3), np.diag([-1., 1., 1.])
    base = _synthetic_histogram([2.], [2.], [2.], [2.], [first.tolist(), second.tolist()])
    extra = _synthetic_histogram([1.], [1.], [1.], [1.], [third.tolist()])
    calls = []
    def reduce(matrices):
        calls.append(len(matrices))
        return extra
    actual = try_expand_raw_dgs_symmetry(
        {"symmetry": [first, second, third]},
        candidates=[({"symmetry": [first]}, lambda: pytest.fail("Smaller subset should stay lazy")),
                    ({"symmetry": [first, second]}, lambda: base)],
        signature=lambda config: "same", operations=lambda config: config["symmetry"],
        reduce_missing=reduce, finalize=lambda data: data, minimum_samples=0)
    assert calls == [1]
    assert actual.num_events.item() == 3


@pytest.mark.parametrize("cells, events, operations, expected", [
    (249999, None, 1, True),
    (250000, None, 6, False),
    (250000, 1000000, 2, True),
    (250000, 999999, 2, False),
    (250000, 2000000, 1, True),
    (250000, 2000000, 0, False),
    (0, 2000000, 1, False),
    (np.int64(250000), np.int64(1000000), np.int64(2), True),
    (91584578, 373966474, 6, True),  # Complete SEQUOIA acquisition.
    (91584578, 10200000, 6, False),  # Sparse 24-run pilot on the same grid.
    (1000000, 23600000, 6, True),  # Dense HYSPEC workload.
    (1000000, 780000, 6, False),
])
def test_workload_guard_boundaries_and_representative_density(cells, events, operations, expected):
    assert symmetry_reuse_worthwhile(cells, events, operations) is expected


@pytest.mark.parametrize("events", [None, 0, -1, float("nan"), float("inf"), True, "2000000", 2000000.])
def test_large_grid_requires_reliable_positive_integer_source_count(events):
    assert not symmetry_reuse_worthwhile(250000, events, 6)


def test_workload_predicate_precedes_lazy_candidate_loading():
    first, second = np.eye(3), -np.eye(3)
    calls = []
    assert try_expand_raw_dgs_symmetry(
        {"symmetry": [first, second]},
        candidates=[({"symmetry": [first]}, lambda: pytest.fail("Candidate arrays must remain unloaded"))],
        signature=lambda config: "same", operations=lambda config: config["symmetry"],
        reduce_missing=lambda matrices: pytest.fail("Sparse workload must use full replay"),
        finalize=lambda data: data, minimum_samples=0,
        worthwhile=lambda count: calls.append(count) or False,
    ) is None
    assert calls == [1]


@pytest.mark.parametrize("event_count", [37, None, 0, -1, float("nan")])
def test_project_guard_skips_large_sparse_or_unknown_sources_before_cache_access(case, monkeypatch, event_count):
    import copy

    root, group, cached, binning, desired, calls = _expanded(case, monkeypatch)
    group.datasets[0].metadata["event_count"] = event_count
    group.datasets[0].fit_weight = 1000000000.
    # Neither disabled sources nor zero-weight sources justify a cache load.
    for enabled, weight in ((False, 1.), (True, 0.)):
        inactive = copy.copy(group.datasets[0])
        inactive.metadata = {**inactive.metadata, "event_count": 1000000000}
        inactive.enabled, inactive.fit_weight = enabled, weight
        group.datasets.append(inactive)
    monkeypatch.setattr(composites, "_composite_rebin_bounds", lambda config: ([], [], [67, 134, 101, 101]))
    monkeypatch.setattr(composites, "_matching_composite_base_key",
                        lambda *args: pytest.fail("Sparse candidate must not access a lazy cache"))
    assert composites._incremental_raw_dgs_base(composites._composite_scope(root, group), desired, None) is None
    assert calls == []


def test_project_guard_allows_large_dense_source_cache_lookup_without_arrays(case, monkeypatch):
    root, group, cached, binning, desired, calls = _expanded(case, monkeypatch)
    group.datasets[0].metadata["event_count"] = 400000000
    monkeypatch.setattr(composites, "_composite_rebin_bounds", lambda config: ([], [], [67, 134, 101, 101]))
    lookups = []
    monkeypatch.setattr(composites, "_matching_composite_base_key", lambda *args: lookups.append(args) or None)
    assert composites._incremental_raw_dgs_base(composites._composite_scope(root, group), desired, None) is None
    assert len(lookups) == 1 and calls == []
