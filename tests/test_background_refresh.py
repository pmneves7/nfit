"""Invalidation and manual-rebin policy for background edits."""

import numpy as np

from nfit import project_composites, project_data, project_gui
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry
from nfit.project_data import dataset_rebin_config
from nfit.project_gui import NfitProject, NfitProjectExplorer


def _histogram(value: float = 1.0) -> MDHistoData:
    return MDHistoData(
        axes=(MDHistoAxis("Q", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),),
        signal=np.full(2, value),
        errors=np.ones(2),
        mask=np.zeros(2, dtype=bool),
        num_events=np.ones(2),
    )


def test_background_change_marks_dataset_stale_without_forcing_rebin(monkeypatch):
    source = DatasetEntry("background", _histogram(2.0), kind="mdhisto")
    sample = DatasetEntry(
        "sample",
        _histogram(5.0),
        kind="mdhisto",
        backgrounds=[BackgroundSpec("bkg", source_dataset_id=source.id, source_entry=source)],
    )
    group = DataGroup("workspace", datasets=[sample, source])
    config = dataset_rebin_config(sample)
    config.update(enabled=True, auto_rebin=True, stale=False)
    explorer = NfitProjectExplorer(NfitProject([group]))
    calls = []
    monkeypatch.setattr(explorer, "_request_overlay_refresh", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(explorer, "_record_data_group_state_change", lambda *_args: None)
    monkeypatch.setattr(explorer, "_refresh_cache_badges", lambda: None)
    monkeypatch.setattr(explorer, "_mark_dirty", lambda: None)

    explorer._background_changed(group, sample)

    assert config["stale"] is True
    assert len(calls) == 1
    assert calls[0][1].get("force_rebin", False) is False


def test_referenced_source_scale_change_is_in_signature():
    source = DatasetEntry("background", _histogram(2.0), kind="mdhisto")
    sample = DatasetEntry(
        "sample",
        _histogram(5.0),
        kind="mdhisto",
        backgrounds=[BackgroundSpec("bkg", source_dataset_id=source.id, source_entry=source)],
    )
    before = project_data._viewer_view_signature(sample, None)
    source.scale_factor = 2.0

    after = project_data._viewer_view_signature(sample, None)

    assert before != after


def test_manual_composite_defers_after_dependency_signature_changes(monkeypatch):
    source = DatasetEntry("source", _histogram(2.0), kind="mdhisto")
    group = DataGroup("workspace", datasets=[source])
    config = project_composites.data_group_composite_config(group)
    config.update(enabled=True, auto_rebin=False, stale=False)
    cached = project_composites._cached_composite_dataset_data(group)
    assert cached is not None
    source.scale_factor = 2.0

    monkeypatch.setattr(
        project_composites,
        "composite_dataset_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("manual rebin unexpectedly ran")),
    )
    deferred = project_composites._cached_composite_dataset_data(group, force_rebin=False)

    assert deferred is cached


def test_interactive_refresh_forwards_progress_and_closes_dialog(monkeypatch):
    group = DataGroup("workspace", datasets=[DatasetEntry("scan", _histogram(), kind="mdhisto")])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._interactive = True

    class Viewer:
        _nfit_use_composite = False
        unmask_model = False
        dataset_combo = None
        binning_combo = None

        def replace_datasets(self, datasets, **kwargs):
            self.replaced = (datasets, kwargs)

    viewer = Viewer()
    explorer._slice_viewers[id(group)] = [viewer]
    callbacks = []
    closed = []

    def make_progress(title, *, aggregate=False):
        def progress(event):
            callbacks.append((title, aggregate, event))

        return progress

    monkeypatch.setattr(explorer, "_make_rebin_progress_callback", make_progress)
    monkeypatch.setattr(
        explorer,
        "_close_rebin_progress",
        lambda callback: closed.append(callback),
    )

    seen = {}

    def fake_slice(group_arg, **kwargs):
        seen.update(kwargs)
        kwargs["progress_callback"]({"stage": "mdevent_events", "iteration": 1, "total": 2})
        return [_histogram()], ["scan"]

    monkeypatch.setattr(project_gui, "slice_viewer_datasets", fake_slice)
    explorer.refresh_slice_viewer(group)

    assert callable(seen["progress_callback"])
    assert callbacks and callbacks[0][2]["stage"] == "mdevent_events"
    assert len(closed) == 1
