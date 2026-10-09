"""Invalidation and manual-rebin policy for background edits."""

import numpy as np
import pytest

from nfit import project_composites, project_data, project_gui
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from nfit.project_data import dataset_rebin_config
from nfit.project_gui import NfitProject, NfitProjectExplorer


@pytest.mark.parametrize("preload", [True, False])
def test_background_scale_updates_open_large_manual_composite_viewers(monkeypatch, preload):
    from PySide6 import QtTest, QtWidgets

    from tests.test_backgrounds import _powder

    background = DatasetEntry(
        "background", _powder(np.full((2, 2), 20.0), np.full((2, 2), 3.0)),
        kind="mdhisto", data_type="powder_inelastic",
    )
    sample = DatasetGroup("sample", datasets=[DatasetEntry(
        "run", _powder(np.full((2, 2), 10.0), np.full((2, 2), 2.0)),
        kind="mdhisto", data_type="powder_inelastic",
    )], backgrounds=[BackgroundSpec("bkg", source_dataset_id=background.id, source_entry=background)])
    root = DataGroup("workspace", datasets=[background], subgroups=[sample])
    scope = project_composites._composite_scope(root, sample)
    config = project_composites.data_group_composite_config(scope)
    config.update(enabled=True, auto_rebin=False, coordinate_mode="powder", axes=[
        dict(name="|Q|", lower=1.0, upper=2.0, step_size=1.0, mode="step",
             auto_lower=False, auto_upper=False),
        dict(name="DeltaE", lower=-1.0, upper=0.0, step_size=1.0, mode="step",
             auto_lower=False, auto_upper=False),
    ])
    project_composites.add_data_group_composite_binning(scope, name="zoom", duplicate_from="fit")
    monkeypatch.setattr(project_data, "_dataset_mask_is_large", lambda _dataset: True)
    monkeypatch.setattr(project_gui, "preload_viewer_data", lambda: preload)
    explorer = NfitProjectExplorer(NfitProject([root]))
    viewers = [explorer.open_slice_viewer(root, selected_dataset_name="sample Composite",
                                         selected_binning_name="zoom") for _ in range(2)]
    for viewer in viewers:
        assert viewer is not None
        np.testing.assert_allclose(viewer._current_slice["signal"], -10.0)
    explorer._set_background_details(root, sample, sample.backgrounds[0])
    scale = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "background_scale")
    assert scale is not None
    monkeypatch.setattr(project_composites, "composite_dataset_data",
                        lambda *_args, **_kwargs: pytest.fail("sample rebin on scale edit"))
    monkeypatch.setattr(project_composites, "_composite_background_data",
                        lambda *_args, **_kwargs: pytest.fail("background rebin on scale edit"))
    explorer._interactive = True
    try:
        for factor in (0.0, 0.25):
            scale.setValue(factor)
            QtTest.QTest.qWait(300)
            for viewer in viewers:
                assert viewer.binning_combo.currentText() == "zoom"
                np.testing.assert_allclose(viewer._current_slice["signal"], 10.0 - 20.0 * factor)
                np.testing.assert_allclose(viewer._current_slice["errors"], np.sqrt(4.0 + 9.0 * factor**2))
    finally:
        explorer._interactive = False
        for viewer in viewers:
            viewer.window.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_prepared_composite_preserves_owned_masks_when_manual_masks_are_deferred(monkeypatch):
    from nfit.project_masks import _mdhisto_with_nfit_masks
    from tests.test_backgrounds import _powder

    data = _powder(np.full((2, 2), 10.0), np.ones((2, 2))).with_updates(
        mask=np.array([[True, False], [False, False]]),
        metadata={"nfit_mask": np.array([[True, False], [False, False]])},
    )
    alias = DatasetEntry("prepared composite", data, kind="mdhisto", metadata={"composite": True})
    alias._viewer_composite_cache_owner = "owner"
    monkeypatch.setattr(project_data, "_dataset_mask_is_large", lambda _dataset: True)
    assert project_data.dataset_mask_application_config(alias)["auto_apply"] is False
    first = project_data.dataset_for_slice_viewer(alias)
    # A new prepared cube must replace the old viewer payload without dropping
    # the masks its composite owner already evaluated.
    alias.replace_data(data.with_updates(signal=np.full((2, 2), 20.0)))
    refreshed = project_data.dataset_for_slice_viewer(alias, force_masks=False, force_rebin=False)
    assert refreshed is not first
    np.testing.assert_array_equal(refreshed.mask, data.mask)
    np.testing.assert_array_equal(refreshed.metadata["nfit_mask"], data.metadata["nfit_mask"])
    np.testing.assert_allclose(refreshed.signal, 20.0)
    # Explicit additional masks still use the normal dataset preparation path.
    from nfit.pipeline import MaskSpec

    mask = MaskSpec("additional", "coordinate_range", parameters={"DeltaE": [-1.5, -0.5]})
    masked = project_data.dataset_for_slice_viewer(alias, extra_masks=[mask])
    expected = _mdhisto_with_nfit_masks(alias, extra_masks=[mask])
    np.testing.assert_array_equal(masked.mask, expected.mask)


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
