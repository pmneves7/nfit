"""Disabled binnings preserve only current resident scientific results."""

from __future__ import annotations

import numpy as np
import pytest

from nfit import project_composites, project_data, project_viewer_loading
from nfit.binning_lifecycle import binning_is_enabled, set_binning_enabled
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry


def _dataset():
    return DatasetEntry(
        "scan",
        MDHistoData(
            axes=(MDHistoAxis("Q", np.array([0.0, 1.0, 2.0]), "rlu", "h"),),
            signal=np.array([2.0, 4.0]),
            errors=np.ones(2),
            mask=np.zeros(2, dtype=bool),
            num_events=np.ones(2),
        ),
        kind="mdhisto",
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    project_data._VIEWER_VIEW_CACHE.clear()
    project_composites._COMPOSITE_DATA_CACHE.clear()
    yield
    project_data._VIEWER_VIEW_CACHE.clear()
    project_composites._COMPOSITE_DATA_CACHE.clear()


def _forbid(*_args, **_kwargs):
    pytest.fail("disabled binning loaded or calculated numerical data")


def test_toggle_preserves_legacy_enabled_and_freshness():
    config = {"enabled": False, "stale": False}
    assert binning_is_enabled(config)
    set_binning_enabled(config, False)
    assert config == {"enabled": False, "stale": False, "binning_enabled": False}
    set_binning_enabled(config, True)
    assert binning_is_enabled(config)
    assert config["stale"] is False
    assert config["enabled"] is False


@pytest.mark.parametrize("stale", [False, True])
def test_disabled_status_never_promises_automatic_calculation(stale):
    config = {"enabled": True, "binning_enabled": False, "stale": stale}
    dataset = _dataset()
    group = DataGroup("sample", datasets=[dataset])
    dataset_text = project_data._dataset_rebin_status_text(dataset, config)
    composite_text = project_composites._composite_rebin_status_text(group, config)
    assert dataset_text == composite_text
    assert dataset_text.startswith("Binning disabled;")
    assert ("stale" in dataset_text) is stale


@pytest.mark.parametrize("named", [False, True])
def test_disabled_dataset_reuses_current_resident_without_load_or_rebin(monkeypatch, named):
    dataset = _dataset()
    fit = project_data.dataset_rebin_config(dataset)
    fit["enabled"] = True
    other = project_data.dataset_for_slice_viewer(dataset) if named else None
    cache_id = project_data.add_dataset_rebin_binning(dataset) if named else None
    config = project_data.dataset_rebin_config_by_id(dataset, cache_id) if named else fit
    kwargs = {"rebin_config": config, "cache_id": cache_id}
    expected = project_data.dataset_for_slice_viewer(dataset, **kwargs)
    signature = project_data._viewer_view_signature(dataset, None, config)
    set_binning_enabled(config, False)
    assert project_data._viewer_view_signature(dataset, None, config) == signature
    monkeypatch.setattr(project_data, "_ensure_dataset_data_loaded", _forbid)
    monkeypatch.setattr(project_data, "_viewer_data_before_scale_uncached", _forbid)
    monkeypatch.setattr(project_data._VIEWER_VIEW_CACHE, "get", _forbid)
    assert project_data.dataset_for_slice_viewer(dataset, **kwargs) is expected
    assert project_data.rebinned_dataset_data(dataset, config_override=config) is not None
    config["stale"] = True
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_data.dataset_for_slice_viewer(dataset, force_rebin=True, **kwargs)
    key = f"{dataset.id}:{cache_id}" if named else dataset.id
    assert project_data._VIEWER_VIEW_CACHE.peek_resident(key) is None
    assert project_data._VIEWER_VIEW_CACHE.peek_resident(f"{key}:viewer-prepared") is None
    if named:
        assert project_data._VIEWER_VIEW_CACHE.peek_resident(
            f"{dataset.id}:viewer-prepared"
        )[1] is other


@pytest.mark.parametrize("stale", [False, True])
def test_disabled_dataset_never_restores_cache_or_falls_back_to_source(monkeypatch, stale):
    dataset = _dataset()
    config = project_data.dataset_rebin_config(dataset)
    config.update(enabled=True, stale=stale)
    set_binning_enabled(config, False)
    monkeypatch.setattr(project_data, "_ensure_dataset_data_loaded", _forbid)
    monkeypatch.setattr(project_data._VIEWER_VIEW_CACHE, "get", _forbid)
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_data.dataset_for_slice_viewer(dataset)
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_data.rebinned_dataset_data(dataset)


def test_disabled_dataset_scientific_edit_invalidates_even_with_manual_masks(monkeypatch):
    dataset = _dataset()
    config = project_data.dataset_rebin_config(dataset)
    config["enabled"] = True
    project_data.dataset_for_slice_viewer(dataset)
    set_binning_enabled(config, False)
    config["axes"][0]["step_size"] *= 2
    project_data.dataset_mask_application_config(dataset)["auto_apply"] = False
    monkeypatch.setattr(project_data, "_ensure_dataset_data_loaded", _forbid)
    monkeypatch.setattr(project_data._VIEWER_VIEW_CACHE, "get", _forbid)
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_data.dataset_for_slice_viewer(dataset, force_rebin=False, force_masks=False)


@pytest.mark.parametrize("composite", [False, True])
def test_reenabling_reuses_current_result_and_rebuilds_stale_result(composite):
    dataset = _dataset()
    group = DataGroup("sample", datasets=[dataset])
    config = (project_composites.data_group_composite_config(group) if composite else
              project_data.dataset_rebin_config(dataset))
    config["enabled"] = True

    def prepare():
        return (project_composites._cached_composite_dataset_data(group) if composite else
                project_data.dataset_for_slice_viewer(dataset))

    original = prepare()
    set_binning_enabled(config, False)
    assert prepare() is original
    set_binning_enabled(config, True)
    assert prepare() is original
    set_binning_enabled(config, False)
    config["stale"] = True
    if composite:
        assert prepare() is None
    else:
        with pytest.raises(ValueError, match="disabled"):
            prepare()
    set_binning_enabled(config, True)
    assert prepare() is not original
    assert config["stale"] is False


@pytest.mark.parametrize("named", [False, True])
def test_disabled_composite_reuses_current_resident_without_criteria_or_cache_load(monkeypatch, named):
    import nfit.dataset_criteria as criteria

    group = DataGroup("combined", datasets=[_dataset()])
    fit = project_composites.data_group_composite_config(group)
    fit["enabled"] = True
    other = project_composites._cached_composite_dataset_data(group) if named else None
    cache_id = project_composites.add_data_group_composite_binning(group) if named else None
    config = project_composites.data_group_composite_config_by_id(group, cache_id) if named else fit
    kwargs = {"config_override": config, "binning_id": cache_id}
    expected = project_composites._cached_composite_dataset_data(group, **kwargs)
    signature = project_composites._composite_cache_signature(group, **kwargs)
    set_binning_enabled(config, False)
    assert project_composites._composite_cache_signature(group, **kwargs) == signature
    assert project_composites.data_group_composite_enabled(group)
    monkeypatch.setattr(criteria, "prepare_dataset_criteria", _forbid)
    monkeypatch.setattr(project_composites, "composite_dataset_data", _forbid)
    monkeypatch.setattr(project_composites._COMPOSITE_DATA_CACHE, "get", _forbid)
    assert project_composites._cached_composite_dataset_data(group, **kwargs) is expected
    config["stale"] = True
    assert project_composites._cached_composite_dataset_data(group, force_rebin=True, **kwargs) is None
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_composites.composite_dataset_entry(group, **kwargs)
    key = project_composites._composite_cache_key(group, cache_id)
    assert project_composites._COMPOSITE_DATA_CACHE.peek_resident(key) is None
    prepared = (project_composites._composite_cache_key(group), "viewer-finish", cache_id, True)
    assert project_composites._COMPOSITE_DATA_CACHE.peek_resident(prepared) is None
    if named:
        assert project_composites._COMPOSITE_DATA_CACHE.peek_resident(
            project_composites._composite_cache_key(group)
        )[1] is not None
        assert other is not None


def test_disabled_composite_no_current_resident_never_computes_or_restores(monkeypatch):
    import nfit.dataset_criteria as criteria

    group = DataGroup("combined", datasets=[_dataset()])
    config = project_composites.data_group_composite_config(group)
    config["enabled"] = True
    project_composites._cached_composite_dataset_data(group)
    set_binning_enabled(config, False)
    config["axes"][0]["step_size"] *= 2
    monkeypatch.setattr(criteria, "prepare_dataset_criteria", _forbid)
    monkeypatch.setattr(project_composites._COMPOSITE_DATA_CACHE, "get", _forbid)
    monkeypatch.setattr(project_composites, "_composite_dataset_data", _forbid)
    assert project_composites._cached_composite_dataset_data(group, force_rebin=False) is None
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_composites.composite_dataset_data(group)
    project_composites._COMPOSITE_DATA_CACHE.clear()
    assert project_composites._cached_composite_dataset_data(group) is None


def test_disabled_composite_with_no_saved_axes_does_not_resolve_source(monkeypatch):
    group = DataGroup("combined", datasets=[_dataset()])
    group.metadata[project_composites.GROUP_COMPOSITE_KEY] = {
        "enabled": True, "binning_enabled": False,
    }
    monkeypatch.setattr(project_composites, "_composite_reference_data", _forbid)
    assert project_composites._cached_composite_dataset_data(group) is None
    entry = project_composites.composite_dataset_entry(group, allow_disabled_empty=True)
    assert entry.data is None
    assert entry.metadata["composite"]


def test_disabled_composite_viewer_reuses_resident_prepared_alias(monkeypatch):
    group = DataGroup("combined", datasets=[_dataset()])
    config = project_composites.data_group_composite_config(group)
    config["enabled"] = True
    binning = project_composites.data_group_composite_binnings(group)[0]

    def alias():
        entry = project_composites.composite_dataset_entry(group)
        return project_viewer_loading.viewer_binning_entry(
            entry, binning, enabled=True, rebin_key=project_data.DATASET_REBIN_KEY,
            derived_key=project_data.DERIVED_RECIPE_KEY,
        )

    expected = project_data.dataset_for_slice_viewer(alias())
    set_binning_enabled(config, False)
    monkeypatch.setattr(project_composites._COMPOSITE_DATA_CACHE, "get", _forbid)
    monkeypatch.setattr(project_data, "_apply_spectral_channel_view", _forbid)
    assert project_data.dataset_for_slice_viewer(alias()) is expected


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_calculated_disabled_binning_opens_first_viewer_without_rebinning(
    monkeypatch, preload, composite,
):
    from nfit import project_gui

    dataset = _dataset()
    group = DataGroup("combined", datasets=[dataset])
    if composite:
        config = project_composites.data_group_composite_config(group)
        config["enabled"] = True
        expected = project_composites._cached_composite_dataset_data(group)
    else:
        config = project_data.dataset_rebin_config(dataset)
        config["enabled"] = True
        expected = project_data._viewer_data_before_scale(dataset)
    set_binning_enabled(config, False)
    monkeypatch.setattr(project_data, "_ensure_dataset_data_loaded", _forbid)
    monkeypatch.setattr(project_gui, "_ensure_dataset_data_loaded", _forbid)
    monkeypatch.setattr(project_data, "_viewer_data_before_scale_uncached", _forbid)
    monkeypatch.setattr(project_composites, "composite_dataset_data", _forbid)
    monkeypatch.setattr(project_data._VIEWER_VIEW_CACHE, "get", _forbid)
    monkeypatch.setattr(project_composites._COMPOSITE_DATA_CACHE, "get", _forbid)
    datasets, names = project_gui.slice_viewer_datasets(group, preload=preload)
    assert len(names) == 1
    np.testing.assert_array_equal(datasets[0].signal, expected.signal)
    assert project_data._VIEWER_VIEW_CACHE.peek_resident(
        f"{dataset.id}:viewer-prepared"
    ) is None


def test_disabled_dataset_binning_keeps_catalog_but_is_not_refresh_target(monkeypatch):
    from types import SimpleNamespace

    dataset = _dataset()
    group = DataGroup("data", datasets=[dataset])
    config = project_data.dataset_rebin_config(dataset)
    config["enabled"] = True
    set_binning_enabled(config, False)
    monkeypatch.setattr(project_data._VIEWER_VIEW_CACHE, "get", _forbid)
    items = project_viewer_loading.plan_viewer_items(
        group, use_composite=False, dataset_binnings=project_data.dataset_rebin_binnings,
    )
    assert [item.dataset for item in items] == [dataset]
    assert project_viewer_loading.project_binning_targets(
        SimpleNamespace(data_groups=[group]), rebin_key=project_data.DATASET_REBIN_KEY,
        dataset_binnings=project_data.dataset_rebin_binnings,
    ) == []
    assert project_viewer_loading.plan_viewer_items(
        group, use_composite=False, dataset_binnings=project_data.dataset_rebin_binnings,
        disabled_binning_available=lambda _dataset, _scope, _binning: False,
    ) == []
