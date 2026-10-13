import numpy as np
import pytest

import nfit.analysis.fingerprint as fingerprint
from nfit.dataset import PointData4D, PointListData
from nfit.histogram_statistics import event_statistics_channels
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.measurement_dependencies import (
    CountingDependencies,
    SourceDependencies,
    independent_source_dependencies,
)
from nfit.pipeline import DataGroup, DatasetEntry


@pytest.fixture(autouse=True)
def isolated_fingerprint_caches():
    fingerprint._DATASET_FINGERPRINT_CACHE.clear()
    fingerprint._DATA_ARRAY_HASH_CACHE.clear()
    fingerprint._ARRAY_HASH_CACHE.clear()


def histogram():
    axis = MDHistoAxis("DeltaE", np.array([0., 1., 2.]), "meV", "energy")
    return MDHistoData((axis,), np.ones(2), np.ones(2), np.zeros(2, bool), np.ones(2),
                      metadata={"signal_semantics": "density"})


def fingerprint_pair(first, second):
    dataset = DatasetEntry("input", first)
    group = DataGroup("group", datasets=[dataset])
    previous = fingerprint.dataset_entry_fingerprint(dataset, group)
    dataset.replace_data(second)
    return previous, fingerprint.dataset_entry_fingerprint(dataset, group)


@pytest.mark.parametrize("change", [
    {"axes": (MDHistoAxis("DeltaE", np.array([0., 2., 4.]), "meV", "energy"),)},
    {"metadata": {"signal_semantics": "bin_integral"}},
    {"metadata": {"signal_semantics": "density", "signal_unit": "barn/sr/meV"}},
    {"coordinate_system": 1},
    {"visual_normalization": 1},
])
def test_physical_geometry_or_metadata_changes_invalidate_identical_main_arrays(change):
    data = histogram()
    before, after = fingerprint_pair(data, data.with_updates(**change))
    assert before != after


def test_auxiliary_channel_values_errors_and_units_affect_identity():
    data = histogram().with_updates(auxiliary_channels={"chi": MDHistoChannel(np.ones(2), np.ones(2), unit="mu_B^2/meV")})
    for channel in (
        MDHistoChannel(np.full(2, 2.), np.ones(2), unit="mu_B^2/meV"),
        MDHistoChannel(np.ones(2), np.full(2, 2.), unit="mu_B^2/meV"),
        MDHistoChannel(np.ones(2), np.ones(2), unit="spin^2/meV"),
    ):
        before, after = fingerprint_pair(data, data.with_updates(auxiliary_channels={"chi": channel}))
        assert before != after


def test_primitive_source_identity_changes_even_when_diagonal_errors_agree():
    data = histogram()
    first = data.with_updates(source_dependencies=independent_source_dependencies(np.ones(2), "first acquisition"))
    second = data.with_updates(source_dependencies=independent_source_dependencies(np.ones(2), "second acquisition"))
    before, after = fingerprint_pair(first, second)
    assert before != after


def test_covariance_structure_changes_even_when_source_ids_and_errors_agree():
    data = histogram()
    common = {"shape": (2,), "source_ids": ("a", "b"), "source_variances": np.ones(2),
              "observation_indices": np.array([0, 1]), "coefficients": np.ones(2)}
    independent = SourceDependencies(**common, source_indices=np.array([0, 1]))
    correlated = SourceDependencies(**common, source_indices=np.array([0, 0]))
    before, after = fingerprint_pair(data.with_updates(source_dependencies=independent),
                                    data.with_updates(source_dependencies=correlated))
    assert before != after


def test_counting_exposure_dependencies_affect_identity():
    numerator = independent_source_dependencies(np.ones(2), "counts")
    first = CountingDependencies(numerator_dependencies=numerator, exposure_dependencies=independent_source_dependencies(np.zeros(2), "exposure A"))
    second = CountingDependencies(numerator_dependencies=numerator, exposure_dependencies=independent_source_dependencies(np.zeros(2), "exposure B"))
    data = histogram().with_updates(auxiliary_channels=event_statistics_channels(np.ones(2), np.ones(2), np.ones(2)))
    before, after = fingerprint_pair(data.with_updates(counting_dependencies=first), data.with_updates(counting_dependencies=second))
    assert before != after


def test_point_payload_temperature_field_and_exposure_are_part_of_identity():
    data = PointData4D([0., 1.], [0., 0.], [0., 0.], [1., 1.], [3., 3.], [1., 1.],
                       temperature=[10., 20.], magnetic_field=[0., 0., 1.],
                       normalization_denominator=[1., 1.], measurement_payload={"counts": [3., 3.]})
    for changes in (
        {"temperature": [10., 30.]},
        {"magnetic_field": [0., 0., 2.]},
        {"normalization_denominator": [1., 2.]},
        {"measurement_payload": {"counts": [3., 4.]}},
    ):
        before, after = fingerprint_pair(data, data.with_updates(**changes))
        assert before != after


def test_point_list_quantity_types_affect_identity():
    data = PointListData({"x": [1.], "y": [2.]}, coordinate_names=["x"],
                         channels=[{"label": "value", "value": "y", "error": None}])
    before, after = fingerprint_pair(data, data.with_updates(quantity_types={"y": "dynamic_susceptibility"}))
    assert before != after


def test_nonfinite_metadata_are_deterministic_and_recipes_remain_strict():
    metadata = {"window": [0., np.inf], "fit_diagnostic": np.nan, "lower_bound": -np.inf}
    data = histogram().with_updates(metadata=metadata)
    dataset = DatasetEntry("input", data, metadata={"diagnostic": np.float64(np.nan)})
    group = DataGroup("group", datasets=[dataset])
    first = fingerprint.dataset_entry_fingerprint(dataset, group)
    fingerprint._DATASET_FINGERPRINT_CACHE.clear()
    fingerprint._DATA_ARRAY_HASH_CACHE.clear()
    assert fingerprint.dataset_entry_fingerprint(dataset, group) == first
    dataset.replace_data(data.with_updates(metadata={**metadata, "lower_bound": np.inf}))
    assert fingerprint.dataset_entry_fingerprint(dataset, group) != first
    with pytest.raises(ValueError, match="finite numeric"):
        fingerprint.recipe_hash("analysis", 1, {"invalid": np.nan}, [])


def test_metadata_edits_reuse_array_hashes_without_copying_histogram_payload(monkeypatch):
    data = histogram()
    dataset = DatasetEntry("input", data)
    group = DataGroup("group", datasets=[dataset])
    first = fingerprint.dataset_entry_fingerprint(dataset, group)
    dataset.replace_data(data.with_updates(metadata={"signal_semantics": "density", "temperature": 20.}))
    calls = []
    original = fingerprint.np.ascontiguousarray

    def counted(array):
        calls.append(array)
        return original(array)

    monkeypatch.setattr(fingerprint.np, "ascontiguousarray", counted)
    assert fingerprint.dataset_entry_fingerprint(dataset, group) != first
    assert calls == []


def test_public_hash_does_not_cache_mutable_arrays():
    array = np.array([1., 2.])
    first = fingerprint.array_hash(array)
    array[0] = 3.
    assert fingerprint.array_hash(array) != first
