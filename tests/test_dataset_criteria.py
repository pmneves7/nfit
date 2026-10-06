import copy

import numpy as np
import pytest

from nfit import (
    DataGroup,
    DatasetEntry,
    NfitProject,
    dataset_criterion_mask,
    dataset_criterion_preview,
    filter_dataset_criteria,
    load_project,
    save_project,
)
from nfit.dataset_criteria import criterion_matches
from nfit.dataset_criterion_values import criterion_value
from tests.test_raw_dgs_background import target

pytest_plugins = ["tests.test_raw_dgs_background"]


def metadata_group():
    return DataGroup("runs", datasets=[
        DatasetEntry(str(i), None, metadata={"temperature": i, "run_number": 100+i})
        for i in (1, 2, 3)
    ])


@pytest.mark.parametrize("side,op,expected", [
    ("left", "<", False), ("left", "<=", True),
    ("left", ">", False), ("left", ">=", True),
    ("right", "<", False), ("right", "<=", True),
    ("right", ">", False), ("right", ">=", True),
])
def test_comparison_boundaries(side, op, expected):
    assert criterion_matches(2., {f"{side}_value": 2., f"{side}_operator": op}) == expected


def test_selection_reuses_values_and_preserves_enabled(monkeypatch):
    group = metadata_group()
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature",
                                  right_value=1.5)
    rows = dataset_criterion_preview(group, mask)
    assert [r["number"] for r in rows] == [101, 102, 103]
    assert [d.name for d in group.select()] == ["1"]
    import nfit.dataset_criteria as api
    monkeypatch.setattr(api, "criterion_value", lambda *a, **k: pytest.fail("recalculated"))
    mask.parameters["right_value"] = 2.5
    dataset_criterion_preview(group, mask)
    assert [d.name for d in group.select()] == ["1", "2"]
    mask.invert = True
    assert [d.name for d in group.select()] == ["3"]
    assert all(d.enabled for d in group.datasets)
    mask.parameters["right_value"] = None
    assert group.select() == group.datasets  # Blank limits stay inactive even inverted.


def test_append_computes_only_new_run_and_changed_metadata_invalidates(monkeypatch):
    import nfit.dataset_criteria as api
    group = metadata_group()
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature", right_value=2.)
    dataset_criterion_preview(group, mask)
    original = api.criterion_value
    calls = []
    def counted(owner, entry, *a, **k):
        calls.append(entry.name)
        return original(owner, entry, *a, **k)
    monkeypatch.setattr(api, "criterion_value", counted)
    group.datasets.append(DatasetEntry("new", None, metadata={"temperature": 4.}))
    with pytest.raises(ValueError, match="new or changed"):
        group.select()
    dataset_criterion_preview(group, mask)
    assert calls == ["new"]
    group.datasets[0].metadata["temperature"] = 10.
    dataset_criterion_preview(group, mask)
    assert calls == ["new", "1"]


def test_ordered_additive_and_disabled_rules():
    group = metadata_group()
    first = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature", right_value=1.)
    second = dataset_criterion_mask(group, name="restore", channel="metadata", source="metadata/temperature",
                                    right_operator="<=", right_value=2.)
    second.additive = True
    for mask in group.masks:
        dataset_criterion_preview(group, mask)
    assert [d.name for d in group.select()] == ["1", "2"]
    first.enabled = False
    assert group.select() == group.datasets
    group.datasets[1].enabled = False
    assert [d.name for d in group.select()] == ["1", "3"]


def test_atomic_preview_cancellation_at_final_notification():
    group = metadata_group()
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature")
    before = copy.deepcopy(mask.metadata)
    def cancel(progress):
        if progress["iteration"] == 3:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        dataset_criterion_preview(group, mask, progress_callback=cancel)
    assert mask.metadata == before


def test_metadata_duration_weighted_log_and_extrema(tmp_path):
    import h5py
    path = tmp_path/"run.nxs"
    with h5py.File(path, "w") as f:
        f["entry/start_time"] = np.bytes_("2026-01-01T00:00:00Z")
        f["entry/end_time"] = np.bytes_("2026-01-01T00:00:10Z")
        f["entry/DASlogs/temp/value"] = [2., 10.]
        f["entry/DASlogs/temp/value"].attrs["units"] = "K"
        f["entry/DASlogs/temp/time"] = [0., 9.]
        f["entry/DASlogs/temp/time"].attrs.update(units="second", start="2026-01-01T00:00:00Z")
    entry = DatasetEntry("run", None, metadata={"source_file": str(path)})
    params = dict(channel="metadata", source="/entry/DASlogs/temp/value")
    assert criterion_value(None, entry, dict(params, statistic="time_average"))["value"] == 2.8
    assert criterion_value(None, entry, dict(params, statistic="min"))["value"] == 2.
    assert criterion_value(None, entry, dict(params, statistic="max"))["value"] == 10.
    with h5py.File(path, "r+") as f:
        del f["entry/DASlogs/temp/time"]
    with pytest.raises(ValueError, match="timestamps"):
        criterion_value(None, entry, dict(params, statistic="time_average"))


def test_elastic_fresh_cached_and_selection_remove_counts_and_exposure(groups):
    sample, _dummy = groups
    import h5py
    for entry, charge in zip(sample.datasets, (2., 6.), strict=True):
        with h5py.File(entry.metadata["source_file"], "r+") as f:
            f["entry/proton_charge"] = [charge]
    mask = dataset_criterion_mask(sample, energy_min=-19., energy_max=19.)
    fresh = dataset_criterion_preview(sample, mask)
    target(sample)  # Writes the normal dataset-owned reduced-event caches.
    cached = dataset_criterion_preview(sample, mask, refresh=True)
    np.testing.assert_allclose([r["value"] for r in fresh], [r["value"] for r in cached], rtol=0, atol=0)
    assert fresh[0]["events"] == 1
    assert fresh[0]["value"] == 3*fresh[1]["value"]
    mask.parameters["right_value"] = (fresh[0]["value"]+fresh[1]["value"])/2
    selected = target(sample)
    mask.enabled = False
    sample.datasets[0].enabled = False
    explicit = target(sample)
    for attr in ("signal", "errors", "num_events"):
        np.testing.assert_allclose(getattr(selected, attr), getattr(explicit, attr), equal_nan=True)
    np.testing.assert_array_equal(selected.mask, explicit.mask)


def test_save_reopen_selection_and_portable_composite_replay(tmp_path):
    from nfit import composite_dataset_data
    from nfit.composite_workflow import export_composite_recipe, replay_composite_recipe
    from tests.test_composite_workflow import _grid, _point_source
    group = DataGroup("runs", datasets=[_point_source(tmp_path, "a"), _point_source(tmp_path, "b", (10., 20.))])
    group.datasets[0].metadata["temperature"] = 1.
    group.datasets[1].metadata["temperature"] = 2.
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature", right_value=1.5)
    dataset_criterion_preview(group, mask)
    _grid(group)
    project = NfitProject([group])
    path = tmp_path/"condition.nfit"
    save_project(project, path)
    restored = load_project(path).data_groups[0]
    assert [d.name for d in restored.select()] == [group.datasets[0].name]
    recipe = export_composite_recipe(project, group.name)
    replay, node = replay_composite_recipe(recipe)
    assert node is None
    actual, expected = composite_dataset_data(replay), composite_dataset_data(group)
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    # Dataset conditions are not inverted into point-coordinate masks.
    replay.masks[0].invert = True
    assert replay.select()[0].name == group.datasets[1].name


def test_dataset_condition_services_are_gui_independent_and_registry_is_authoritative():
    import ast
    from pathlib import Path

    from nfit.dataset_criteria import DATASET_CRITERION_DEFINITION
    from nfit.project_gui import MASK_TYPE_DEFINITIONS
    assert MASK_TYPE_DEFINITIONS["dataset_criterion"] is DATASET_CRITERION_DEFINITION
    for name in ("dataset_criteria", "dataset_criterion_values"):
        tree = ast.parse((Path(__file__).parents[1]/"src/nfit"/f"{name}.py").read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(s.startswith(("PySide", "project_gui", "dataset_criterion_gui")) for s in imports)


@pytest.mark.parametrize("powder", [False, True])
def test_metadata_condition_in_native_mdevent_binning(tmp_path, powder):
    from nfit import bin_mdevent_group, bin_mdevent_powder_group, mdevent_dataset_group
    from tests.test_mdevent import _write_mdevent
    source = tmp_path/"events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/proton_charge",
                                  right_operator="<", right_value=1.5)
    # The native public binner prepares the scalar condition itself.
    binner = bin_mdevent_powder_group if powder else bin_mdevent_group
    options = (dict(lower=[0., -1.], upper=[10., 1.], num_bins=[1, 1]) if powder else
               dict(lower=[-10., -10., -10., -1.], upper=[10., 10., 10., 1.], num_bins=[1, 1, 1, 1]))
    actual = binner(group, **options)
    mask.enabled = False
    expected = binner(group, datasets=[group.datasets[1]], **options)
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
    np.testing.assert_array_equal(actual.num_events, expected.num_events)


def test_metadata_condition_for_corelli_does_not_use_dgs_diagnostic(tmp_path, monkeypatch):
    from nfit import bin_corelli_group, corelli_dataset_group, dataset_criterion_values
    from tests.test_corelli import _write_corelli
    sources = [tmp_path/f"CORELLI_{i}.nxs.h5" for i in (1, 2)]
    for source in sources:
        _write_corelli(source)
    group = corelli_dataset_group(sources)
    group.metadata["raw_dgs"].update(timing_offset_ns=0, bad_pulse_threshold=0)
    for entry, temp in zip(group.datasets, (1., 2.), strict=True):
        entry.metadata["temperature"] = temp
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature", right_value=1.5)
    monkeypatch.setattr(dataset_criterion_values, "_elastic_value", lambda *a: pytest.fail("DGS diagnostic"))
    options = dict(lower=[-100., -100., -100., -1.], upper=[100.,100.,100.,1.], num_bins=[1,1,1,1])
    actual = bin_corelli_group(group, **options)
    mask.enabled = False
    expected = bin_corelli_group(group, datasets=[group.datasets[0]], **options)
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_array_equal(actual.num_events, expected.num_events)


def test_fit_preparation_refreshes_condition_without_enabled_mutation(tmp_path):
    from nfit.project_gui import fit_dataset_inputs
    from tests.test_composite_workflow import _point_source
    group = DataGroup("runs", datasets=[_point_source(tmp_path, "a"), _point_source(tmp_path, "b")])
    for entry, temp in zip(group.datasets, (1., 2.), strict=True):
        entry.metadata["temperature"] = temp
    dataset_criterion_mask(group, channel="metadata", source="metadata/temperature", right_value=1.5)
    inputs, _bundles = fit_dataset_inputs(group, purpose="fit")
    assert len(inputs) == 1
    assert all(d.enabled for d in group.datasets)


def test_timestamp_free_series_and_mixed_units_are_rejected_atomically(tmp_path):
    import h5py
    entries = []
    for unit in ("K", "C"):
        source = tmp_path/f"{unit}.nxs"
        with h5py.File(source, "w") as f:
            f["entry/value"] = [1., 2.]
            f["entry/value"].attrs["units"] = unit
        entries.append(DatasetEntry(unit, None, metadata={"source_file": str(source)}))
    group = DataGroup("runs", datasets=entries)
    mask = dataset_criterion_mask(group, channel="metadata", source="entry/value", statistic="max")
    with pytest.raises(ValueError, match="units differ"):
        dataset_criterion_preview(group, mask)
    assert not mask.metadata
    mask.parameters["statistic"] = "time_average"
    with pytest.raises(ValueError, match="timestamps"):
        dataset_criterion_preview(group, mask)


def test_large_hdf_array_is_rejected_before_decode(tmp_path):
    import h5py
    source = tmp_path/"big.nxs"
    with h5py.File(source, "w") as f:
        f.create_dataset("entry/events", shape=(5_000_000,), dtype="f8", chunks=(1000,))
    entry = DatasetEntry("large source", None, metadata={"source_file": str(source)})
    with pytest.raises(ValueError, match="32 MiB"):
        criterion_value(None, entry, dict(channel="metadata", source="entry/events", statistic="max"))


def test_coordinate_mask_evaluator_skips_inverted_dataset_conditions():
    from nfit import PointData4D
    from nfit.project_masks import _nfit_mask_for_point_data
    entry = DatasetEntry("points", PointData4D(H=[0.], K=[0.], L=[0.], E=[0.], intensity=[1.], sigma=[1.]))
    group = DataGroup("runs", datasets=[entry])
    mask = dataset_criterion_mask(group, channel="metadata", source="parameters/temperature", right_value=1.)
    mask.invert = True
    np.testing.assert_array_equal(_nfit_mask_for_point_data(entry, entry.data, extra_masks=[mask]), [False])


def test_preparation_keeps_only_one_current_value_per_run_and_omits_derived_entries():
    group = metadata_group()
    derived = DatasetEntry("derived", None, kind="derived_recipe", metadata={"derived_recipe": {"operation": "subtract"}})
    group.datasets.append(derived)
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature", right_value=2.)
    for value in range(10):
        group.datasets[0].metadata["temperature"] = value
        filter_dataset_criteria(group, group.datasets, compute=True)
        assert len(mask.metadata["dataset_criterion_values"]) == 3
    assert len(dataset_criterion_preview(group, mask)) == 3
