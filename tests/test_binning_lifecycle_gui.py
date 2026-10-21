"""Explorer binning scheduling must not invalidate scientific results."""

from types import SimpleNamespace

import pytest

from nfit import project_gui


@pytest.mark.parametrize("composite", [False, True])
def test_enabled_toggle_preserves_selected_cache_state(monkeypatch, composite):
    selected = {"binning_enabled": True, "enabled": True, "stale": False}
    other = {"binning_enabled": True, "enabled": True, "stale": False}
    events = []
    monkeypatch.setattr(project_gui, "_resident_binning_available", lambda *_: True)
    group = project_gui.DataGroup("sample")
    dataset = project_gui.DatasetEntry("scan", None)
    explorer = SimpleNamespace(
        _selected_dataset_binning=lambda _: {"config": selected},
        _selected_composite_binning=lambda _: {"config": selected},
        _record_data_group_state_change=lambda owner: events.append(("history", owner)),
        _mark_dirty=lambda: events.append(("dirty",)),
        _refresh_cache_badges=lambda: events.append(("badges",)),
        _set_dataset_details_preserving_scroll=lambda *_: events.append(("details",)),
        _sync_details=lambda: events.append(("details",)),
    )
    def toggle(enabled):
        if composite:
            project_gui.NfitProjectExplorer._set_group_composite_binning_enabled(
                explorer, group, enabled
            )
        else:
            project_gui.NfitProjectExplorer._set_dataset_binning_enabled(
                explorer, dataset, group, enabled
            )

    toggle(False)
    assert selected == {"binning_enabled": False, "enabled": True, "stale": False}
    assert other == {"binning_enabled": True, "enabled": True, "stale": False}
    assert events == [("history", group), ("dirty",), ("badges",), ("details",)]
    events.clear()
    toggle(False)
    assert events == []
    toggle(True)
    assert selected == {"binning_enabled": True, "enabled": True, "stale": False}


@pytest.mark.parametrize("composite", [False, True])
def test_rebin_all_skips_disabled_even_when_stale(monkeypatch, composite):
    group = project_gui.DataGroup("sample")
    dataset = project_gui.DatasetEntry("scan", None)
    enabled = {"enabled": True, "stale": True}
    disabled = {"enabled": True, "stale": True, "binning_enabled": False}
    binnings = [
        {"id": "fit", "name": "Default", "fit": True, "config": enabled},
        {"id": "aux", "name": "Large", "fit": False, "config": disabled},
    ]
    calculated = []
    result = object()
    explorer = SimpleNamespace(
        _interactive=False,
        _confirm_dataset_rebin_memory=lambda *_a, **_k: True,
        _confirm_composite_rebin_memory=lambda *_a, **_k: True,
        _make_rebin_progress_callback=lambda *_a, **_k: None,
        _close_rebin_progress=lambda *_: None,
        _refresh_cache_badges=lambda: None,
        refresh_slice_viewer=lambda *_: None,
        _set_dataset_details_preserving_scroll=lambda *_: None,
        _sync_details=lambda: None,
    )
    if composite:
        monkeypatch.setattr(project_gui, "data_group_composite_binnings", lambda _: binnings)

        def prepare(_group, **kwargs):
            calculated.append(kwargs["config_override"] or enabled)
            return result

        monkeypatch.setattr(project_gui, "_cached_composite_dataset_data", prepare)
        assert project_gui.NfitProjectExplorer.rebin_all_composite_binnings_now(explorer, group)
    else:
        monkeypatch.setattr(project_gui, "dataset_rebin_binnings", lambda _: binnings)
        monkeypatch.setattr(project_gui, "_dataset_can_rebin", lambda _: True)

        def prepare(_dataset, **kwargs):
            calculated.append(kwargs["rebin_config"])
            return result

        monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", prepare)
        assert project_gui.NfitProjectExplorer.rebin_all_dataset_binnings_now(explorer, dataset, group)
    assert calculated == [enabled]
    assert disabled["stale"] is True


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_visualization_binning_works_with_unavailable_disabled_fit(composite, preload):
    from nfit import project_composites, project_data
    from nfit.binning_lifecycle import set_binning_enabled
    from tests.project_gui_test_support import _grid_mdhisto_data

    project_data._VIEWER_VIEW_CACHE.clear()
    project_composites._COMPOSITE_DATA_CACHE.clear()
    dataset = project_gui.DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = project_gui.DataGroup("sample", datasets=[dataset])
    if composite:
        fit = project_composites.data_group_composite_config(group)
        fit.update(enabled=True, minimum_coverage=0.0)
        project_composites.add_data_group_composite_binning(group, name="Overview")
    else:
        fit = project_data.dataset_rebin_config(dataset)
        fit.update(enabled=True, minimum_coverage=0.0)
        project_data.add_dataset_rebin_binning(dataset, name="Overview")
    set_binning_enabled(fit, False)
    datasets, names = project_gui.slice_viewer_datasets(group, preload=preload)
    assert len(names) == 1 and names[0].endswith(" · Overview")
    assert datasets[0].shape == (3, 3)
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_gui.fit_dataset_inputs(group, purpose="fit")


def test_loader_guard_uses_requested_binning_instead_of_editor_selection(monkeypatch):
    from nfit import project_data
    from nfit.binning_lifecycle import set_binning_enabled

    dataset = project_gui.DatasetEntry("scan", None)
    config = project_data.dataset_rebin_config(dataset)
    config["enabled"] = True
    named_id = project_data.add_dataset_rebin_binning(dataset, name="Overview")
    dataset._nfit_selected_binning_id = named_id
    set_binning_enabled(config, False)
    monkeypatch.setattr(
        project_gui, "_ensure_dataset_data_loaded",
        lambda *_a, **_k: pytest.fail("loaded the disabled fit binning's source"),
    )
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_gui.dataset_for_slice_viewer(dataset)


def test_resource_guard_covers_promoted_fit_viewer_cache_keys():
    from nfit import project_composites, project_resources
    from nfit.binning_lifecycle import set_binning_enabled

    group = project_gui.DataGroup("sample")
    named_id = project_composites.add_data_group_composite_binning(group, name="Overview")
    project_composites.make_data_group_fit_binning(group, named_id)
    config = project_composites.data_group_composite_config(group)
    assert config["_binning_id"] == named_id
    set_binning_enabled(config, False)
    disabled = project_resources._disabled_binning_resources(project_gui.NfitProject([group]))
    owner = project_composites._composite_cache_key(group)
    assert disabled("composite", owner)
    assert disabled("composite", (owner, "viewer-prepared", "fit"))
    assert disabled("composite", (owner, "viewer-source", named_id))
