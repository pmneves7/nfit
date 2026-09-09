from __future__ import annotations

import numpy as np

from nfit import project_composites, project_data
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup, MaskSpec


def _histogram(signal: list[float]) -> MDHistoData:
    values = np.asarray(signal, dtype=float)
    return MDHistoData(
        axes=(MDHistoAxis("Q", np.array([0.0, 1.0, 2.0]), "rlu", "h"),),
        signal=values,
        errors=np.ones_like(values),
        mask=np.zeros_like(values, dtype=bool),
        num_events=np.ones_like(values),
    )


def test_project_data_reexports_composite_service_identities() -> None:
    for name in (
        "_CompositeScope",
        "_composite_scope",
        "data_group_composite_config",
        "data_group_composite_status",
        "_composite_cache_signature",
        "composite_dataset_data",
        "_cached_composite_dataset_data",
        "composite_dataset_entry",
        "materialize_composite_dataset",
    ):
        assert getattr(project_data, name) is getattr(project_composites, name)
    assert (
        project_data._COMPOSITE_DATA_CACHE
        is project_composites._COMPOSITE_DATA_CACHE
    )


def test_nested_composite_scope_preserves_project_object_identity() -> None:
    root_mask = MaskSpec("root mask", "coordinate_range")
    child_mask = MaskSpec("child mask", "coordinate_range")
    dataset = DatasetEntry("scan", _histogram([1.0, 2.0]), kind="mdhisto")
    child = DatasetGroup("child", datasets=[dataset], masks=[child_mask])
    root = DataGroup("root", subgroups=[child], masks=[root_mask])

    scope = project_composites._composite_scope(root, child)

    assert scope.root is root
    assert scope.node is child
    assert scope.masks == [root_mask, child_mask]
    assert list(scope.iter_datasets()) == [dataset]
    assert project_data.effective_dataset_masks(root, dataset) == [
        root_mask,
        child_mask,
    ]


def test_representative_composite_matches_compatibility_facade_without_mutation() -> None:
    first_data = _histogram([1.0, 2.0])
    second_data = _histogram([3.0, 4.0])
    first_signal = first_data.signal.copy()
    second_signal = second_data.signal.copy()
    group = DataGroup(
        "combined",
        datasets=[
            DatasetEntry("first", first_data, kind="mdhisto"),
            DatasetEntry("second", second_data, kind="mdhisto"),
        ],
    )
    project_composites.data_group_composite_config(group)["enabled"] = True

    direct = project_composites.composite_dataset_data(group)
    through_facade = project_data.composite_dataset_data(group)

    np.testing.assert_allclose(through_facade.signal, direct.signal)
    np.testing.assert_allclose(through_facade.errors, direct.errors)
    np.testing.assert_array_equal(first_data.signal, first_signal)
    np.testing.assert_array_equal(second_data.signal, second_signal)
    assert group.datasets[0].data is first_data
    assert group.datasets[1].data is second_data


def test_composite_cache_identity_is_shared_through_facade() -> None:
    project_composites._COMPOSITE_DATA_CACHE.clear()
    group = DataGroup(
        "combined",
        datasets=[DatasetEntry("scan", _histogram([1.0, 2.0]), kind="mdhisto")],
    )
    project_composites.data_group_composite_config(group)["enabled"] = True

    direct = project_composites._cached_composite_dataset_data(group)
    through_facade = project_data._cached_composite_dataset_data(group)

    assert through_facade is direct


def test_configured_backend_tracks_live_project_data_monkeypatches(monkeypatch) -> None:
    dataset = DatasetEntry("scan", _histogram([1.0, 2.0]), kind="mdhisto")
    group = DataGroup("combined", datasets=[dataset])
    monkeypatch.setattr(project_data, "_dataset_data_point_count", lambda _entry: 37)

    assert project_composites._composite_source_points(group) == 37


def test_composite_policy_reads_live_project_data_constants(monkeypatch) -> None:
    group = DataGroup(
        "combined",
        datasets=[
            DatasetEntry("first", _histogram([1.0, 2.0]), kind="mdhisto"),
            DatasetEntry("second", _histogram([3.0, 4.0]), kind="mdhisto"),
        ],
    )
    monkeypatch.setattr(project_data, "REBIN_AUTO_MAX_CONTRIBUTIONS", 1)

    config = project_composites.data_group_composite_config(group)

    assert config["auto_rebin"] is False


def test_named_composite_binnings_are_independent_and_have_one_fit_configuration() -> None:
    group = DataGroup(
        "combined",
        datasets=[DatasetEntry("scan", _histogram([1.0, 2.0]), kind="mdhisto")],
    )
    fit = project_composites.data_group_composite_config(group)
    fit["axes"][0]["step_size"] = 1.0
    auxiliary_id = project_composites.add_data_group_composite_binning(
        group,
        name="Presentation",
        duplicate_from=fit["_binning_id"],
    )
    auxiliary = project_composites.data_group_composite_config_by_id(
        group, auxiliary_id
    )
    auxiliary["axes"][0]["step_size"] = 0.25

    project_composites.make_data_group_fit_binning(group, auxiliary_id)
    binnings = project_composites.data_group_composite_binnings(group)

    assert [(item["name"], item["fit"]) for item in binnings] == [
        ("Presentation", True),
        ("Default", False),
    ]
    assert binnings[0]["config"]["axes"][0]["step_size"] == 0.25
    assert binnings[1]["config"]["axes"][0]["step_size"] == 1.0
    project_composites.rename_data_group_composite_binning(
        group, binnings[1]["id"], "Original"
    )
    assert [
        item["name"] for item in project_composites.data_group_composite_binnings(group)
    ] == ["Presentation", "Original"]
