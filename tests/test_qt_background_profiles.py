"""Settled source replay stays responsive and never installs stale profiles."""

import time
from dataclasses import replace
from threading import Event, get_ident

import numpy as np
import pytest

from nfit.mdhisto import MDHistoAxis, MDHistoData


def _wait(app, predicate, timeout=4.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        app.processEvents()
        time.sleep(.005)
    assert predicate()


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import nfit.background_profile_queries as replay_module
    from nfit.qt_slice_controls import _qt_app
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    monkeypatch.setattr(replay_module, "background_profile_slice_selection",
        lambda data, **kwargs: kwargs, raising=False)
    data = MDHistoData(
        tuple(MDHistoAxis(name, np.arange(5.), "", "unknown") for name in ("H", "E")),
        np.arange(16.).reshape(4, 4) + 1., np.ones((4, 4)),
        np.zeros((4, 4), bool), np.ones((4, 4)),
        metadata={"cached_background_replay": {"version": 1},
                  "background_profile_uncertainty": "diagonal_approximation"})
    instance = QtMDHistoSliceViewer(data, x_dim=0, y_dim=1)
    instance.hist_axes_check.setChecked(True)
    instance.show_box_check.setChecked(True)
    yield instance, _qt_app()
    instance.window.close()


def _exact(profiles):
    updates = {}
    for name in ("x_measurement", "y_measurement"):
        measurement = getattr(profiles, name)
        if measurement is not None:
            updates[name] = replace(measurement, data=measurement.data.with_updates(
                metadata={**measurement.data.metadata, "background_profile_uncertainty": "source_covariance"}))
    return replace(profiles, **updates)


def test_debounce_replay_renders_on_gui_thread_and_guards_exports(viewer, monkeypatch):
    instance, app = viewer
    controller = instance._background_profiles
    controller._timer.setInterval(15)
    calls, threads = [], []
    preview = instance._current_box_profiles
    def replay(data, **kwargs):
        calls.append(kwargs)
        threads.append(get_ident())
        return _exact(preview)
    controller._replay = replay
    gui_thread = get_ident()
    rendered = []
    original = controller._render
    def render(profiles):
        rendered.append(get_ident())
        original(profiles)
    monkeypatch.setattr(controller, "_render", render)
    instance._set_roi_extents((.1, 1.9, .1, 2.9), update_cuts=True, draw=False)
    instance._set_roi_extents((.2, 1.8, .2, 2.8), update_cuts=True, draw=False)
    assert controller.status == "pending"
    assert not instance.save_x_cut_button.isEnabled()
    assert "replay" in instance.save_x_cut_button.toolTip()
    monkeypatch.setattr(instance, "_csv_export_path", lambda *args: pytest.fail("pending export opened a dialog"))
    instance.save_x_cut()
    _wait(app, lambda: controller.status == "exact")
    assert len(calls) == 1
    assert calls[0]["extents"] == instance._roi_extents
    assert threads[0] != gui_thread
    assert rendered == [gui_thread]
    assert instance.save_x_cut_button.isEnabled()
    assert "background covariance replay" in instance.ax_xcut.get_ylabel()
    assert "background_uncertainty='replay'" in instance._figure_script()
    assert "diagonal uncertainty" in instance.roi_sum_text.get_text()


def test_one_worker_cancels_obsolete_query_and_only_latest_is_installed(viewer):
    instance, app = viewer
    controller = instance._background_profiles
    controller._timer.setInterval(1)
    started, release = Event(), Event()
    calls = []
    preview = instance._current_box_profiles
    def replay(data, **kwargs):
        calls.append(kwargs["extents"])
        if len(calls) == 1:
            started.set()
            release.wait(3.)
        kwargs["progress_callback"]()
        return _exact(preview)
    controller._replay = replay
    instance._set_roi_extents((.1, 1.9, .1, 1.9), update_cuts=True, draw=False)
    _wait(app, started.is_set)
    instance._set_roi_extents((.1, 2.9, .1, 2.9), update_cuts=True, draw=False)
    instance._set_roi_extents((.2, 3.8, .2, 3.8), update_cuts=True, draw=False)
    assert len(calls) == 1
    release.set()
    _wait(app, lambda: controller.status == "exact")
    assert len(calls) == 2
    assert calls[-1] == instance._roi_extents
    assert controller._ready[0] == controller._key


def test_unavailable_source_has_reason_no_retry_loop_and_approximate_export(viewer, monkeypatch, tmp_path):
    from nfit.measurement_dependencies import SourceReplayRequired

    instance, app = viewer
    controller = instance._background_profiles
    controller._timer.setInterval(1)
    calls = []
    def replay(data, **kwargs):
        calls.append(1)
        raise SourceReplayRequired("Background source is unavailable")
    controller._replay = replay
    instance._set_roi_extents((.1, 2.9, .1, 2.9), update_cuts=True, draw=False)
    _wait(app, lambda: controller.status == "approximate")
    instance._update_histogram_cuts_from_extents(instance._roi_extents)
    app.processEvents()
    assert len(calls) == 1
    assert instance.save_x_cut_button.isEnabled()
    assert "unavailable" in instance.save_x_cut_button.toolTip()
    path = tmp_path / "approximate.csv"
    monkeypatch.setattr(instance, "_csv_export_path", lambda *args: str(path))
    instance.save_x_cut()
    declaration = path.with_suffix(".csv.json").read_text()
    assert "diagonal_approximation" in declaration
    assert "Background source is unavailable" in declaration
    assert "background_uncertainty='diagonal'" in instance._figure_script()


@pytest.mark.parametrize("change,reason", [
    (lambda viewer: setattr(viewer, "smoothing_x", 1.), "Smoothed"),
    (lambda viewer: viewer.display_step_factors.update({0: 2}), "coarsening"),
    (lambda viewer: setattr(viewer.model, "masked", False), "Masks"),
    (lambda viewer: setattr(viewer.model, "channel", "errors"), "signal"),
])
def test_unsupported_treatment_is_labelled_and_never_launches(viewer, change, reason):
    instance, app = viewer
    controller = instance._background_profiles
    controller._replay = lambda *args, **kwargs: pytest.fail("unsupported replay")
    change(instance)
    instance._update_background_profiles(instance._roi_extents)
    assert controller.status == "approximate"
    assert reason in controller.reason
    assert "Diagonal" in instance.ax_xcut.get_title()
    app.processEvents()


def test_close_cancels_without_waiting_or_installing_worker_result(viewer):
    instance, app = viewer
    controller = instance._background_profiles
    controller._timer.setInterval(1)
    started, release = Event(), Event()
    preview = instance._current_box_profiles
    def replay(data, **kwargs):
        started.set()
        release.wait(3.)
        return _exact(preview)
    controller._replay = replay
    instance._set_roi_extents((.1, 2.9, .1, 2.9), update_cuts=True, draw=False)
    _wait(app, started.is_set)
    before = time.monotonic()
    instance.window.close()
    assert time.monotonic() - before < .5
    assert controller._closed
    assert controller._cancel.is_set()
    release.set()
    _wait(app, lambda: controller._active is None)
    assert controller.status != "exact"


def test_real_source_replay_matches_public_api_and_exact_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit import bin_mdevent_group, project_measured_background_mdevent
    from nfit.background_profile_queries import replay_cached_background_box_profiles
    from nfit.qt_slice_controls import _qt_app
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    from tests.test_mdevent_background_statistics import _directional_fixture

    sample, source = _directional_fixture(tmp_path)
    target = bin_mdevent_group(sample, lower=[-2., -2., -2., -.5],
        upper=[.5, 2., 2., .5], num_bins=[2, 2, 1, 1])
    background = project_measured_background_mdevent(sample, source, target)
    instance = QtMDHistoSliceViewer(background, x_dim=0, y_dim=1)
    try:
        instance.hist_axes_check.setChecked(True)
        controller = instance._background_profiles
        controller._timer.setInterval(1)
        extents = (-2., .5, -2., 2.)
        instance._set_roi_extents(extents, update_cuts=True, draw=False)
        _wait(_qt_app(), lambda: controller.status in {"exact", "approximate"})
        assert controller.status == "exact", controller.reason
        expected = replay_cached_background_box_profiles(background, x_dim=0, y_dim=1,
            selections=instance.model._normalized_selections(), extents=extents,
            display_selected=np.isfinite(instance._current_slice["signal"]))
        for actual, reference in zip(instance._current_box_profiles.x, expected.x, strict=True):
            np.testing.assert_allclose(actual, reference, equal_nan=True)
        destination = tmp_path / "exact.csv"
        monkeypatch.setattr(instance, "_csv_export_path", lambda *args: str(destination))
        instance.save_x_cut()
        assert "source_covariance" in destination.with_suffix(".csv.json").read_text()
        stale_destination = tmp_path / "stale-dialog.csv"
        def changed_dialog(*args):
            instance.model.channel = "errors"
            return str(stale_destination)
        monkeypatch.setattr(instance, "_csv_export_path", changed_dialog)
        instance.save_x_cut()
        assert not stale_destination.exists()
        instance.model.channel = "signal"
        instance.popout_cuts_check.setChecked(True)
        for side in ("x", "y"):
            popped = instance._cut_viewers[side]
            assert popped.data.metadata["background_profile_uncertainty"] == "source_covariance"
            assert popped.data.metadata["measurement_contract"]["dependence"] == "shared_sources"
            np.testing.assert_allclose(popped.data.errors.ravel(), getattr(expected, side)[2], equal_nan=True)
        instance._set_roi_extents((-2., .5, -1.9, 1.9), update_cuts=True, draw=False)
        _wait(_qt_app(), lambda: controller.status == "exact")
        assert not instance._cut_viewers
        assert instance.popout_cuts_check.isChecked()
        instance._set_roi_extents(extents, update_cuts=True, draw=False)
        _wait(_qt_app(), lambda: controller.status == "exact")
        for side in ("x", "y"):
            np.testing.assert_allclose(instance._cut_viewers[side].data.errors.ravel(),
                getattr(instance._current_box_profiles, side)[2], equal_nan=True)
        instance.popout_cuts_check.setChecked(False)

        import matplotlib.pyplot as plt

        import nfit

        calls = []
        original_plot = nfit.plot_mdhisto_slice
        def plot(data, **kwargs):
            calls.append(kwargs)
            return original_plot(data, **kwargs)
        monkeypatch.setattr(nfit, "plot_mdhisto_slice", plot)
        monkeypatch.setattr(plt, "show", lambda: None)
        script = instance._figure_script()
        # The saved figure script has an editable data placeholder; supply its
        # actual original cached histogram without changing the scientific call.
        data_line = next(line for line in script.splitlines() if line.startswith("data = "))
        namespace = {"actual_data": background}
        exec(script.replace(data_line, "data = actual_data"), namespace)
        assert calls[0]["background_uncertainty"] == "replay"
        figure = namespace["fig"]
        x_axis = next(axis for axis in figure.axes if axis.get_xlabel() == instance.model._axis_label(0)
                      and len(axis.lines))
        np.testing.assert_allclose(x_axis.lines[0].get_ydata(), instance._current_x_cut[1], equal_nan=True)
        segments = x_axis.collections[0].get_segments()
        for segment, value, error in zip(segments, instance._current_x_cut[1], instance._current_x_cut[2], strict=True):
            if np.isfinite(error):
                np.testing.assert_allclose(segment[:, 1], [value - error, value + error])
        plt.close(figure)
    finally:
        instance.window.close()


def test_focused_controller_never_imports_viewer_coordinators():
    import ast
    from pathlib import Path

    import nfit.qt_background_profiles as module

    tree = ast.parse(Path(module.__file__).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {"qt_slice_viewer", "qt_slice_modes", "project_data", "project_gui"} & imports


def test_missing_source_retries_when_file_becomes_available(viewer, tmp_path):
    from nfit.measurement_dependencies import SourceReplayRequired

    instance, app = viewer
    controller = instance._background_profiles
    controller._timer.setInterval(1)
    source = tmp_path / "returned-source.nxs"
    instance.data.metadata["cached_background_replay"]["terms"] = [
        {"sources": [{"source_file": str(source)}]}]
    calls = []
    preview = instance._current_box_profiles
    def replay(data, **kwargs):
        calls.append(1)
        if not source.exists():
            raise SourceReplayRequired("Source unavailable")
        return _exact(preview)
    controller._replay = replay
    instance._set_roi_extents((.1, 2.9, .1, 2.9), update_cuts=True, draw=False)
    _wait(app, lambda: controller.status == "approximate")
    source.write_text("source available")
    instance._update_histogram_cuts_from_extents(instance._roi_extents)
    _wait(app, lambda: controller.status == "exact")
    assert len(calls) == 2


def test_stale_completion_cannot_install_after_channel_change_without_refresh(viewer):
    instance, app = viewer
    controller = instance._background_profiles
    controller._timer.setInterval(1)
    started, release = Event(), Event()
    preview = instance._current_box_profiles
    def replay(data, **kwargs):
        started.set()
        release.wait(3.)
        return _exact(preview)
    controller._replay = replay
    instance._set_roi_extents((.1, 2.9, .1, 2.9), update_cuts=True, draw=False)
    _wait(app, started.is_set)
    instance.model.channel = "errors"
    release.set()
    _wait(app, lambda: controller._active is None)
    assert controller.status == "approximate"
    assert controller._ready is None


def test_hiding_box_panels_cancels_pending_and_fit_overlay_is_explicitly_unsupported(viewer, monkeypatch):
    instance, app = viewer
    controller = instance._background_profiles
    controller._replay = lambda *args, **kwargs: pytest.fail("hidden profile started replay")
    instance._set_roi_extents((.1, 2.9, .1, 2.9), update_cuts=True, draw=False)
    instance.hist_axes_check.setChecked(False)
    assert controller._pending is None
    assert not controller._timer.isActive()
    monkeypatch.setattr(instance, "_fit_panels_active", lambda: True)
    instance.hist_axes_check.setChecked(True)
    controller.update(instance._roi_extents)
    assert "Fit overlays" in controller.reason
    assert controller.status == "approximate"
    app.processEvents()
