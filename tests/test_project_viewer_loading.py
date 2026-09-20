import ast
from pathlib import Path

import numpy as np
import pytest

from nfit import project_data, project_gui, project_viewer_loading
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
from nfit.viewer_data import DeferredViewerDatasets
from tests.project_gui_test_support import _tiny_mdhisto_data


def _group_with_named_binnings():
    first = DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto")
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0), kind="mdhisto")
    for dataset, name in ((first, "Coarse"), (second, "Overview")):
        binning_id = project_data.add_dataset_rebin_binning(dataset, name=name)
        project_data.dataset_rebin_config_by_id(dataset, binning_id)["enabled"] = True
    return DataGroup("Workspace", datasets=[first, second])


def test_deferred_catalog_and_descriptors_do_not_prepare_arrays(monkeypatch):
    group = _group_with_named_binnings()
    prepared = []
    monkeypatch.setattr(
        project_gui,
        "dataset_for_slice_viewer",
        lambda dataset, **_kwargs: prepared.append(dataset.name) or dataset.data,
    )
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)

    datasets, names = project_gui.slice_viewer_datasets(
        group, use_composite=False, preload=False
    )

    assert isinstance(datasets, DeferredViewerDatasets)
    assert names == ["first", "first · Coarse", "second", "second · Overview"]
    assert [item.name for item in datasets.descriptors] == names
    assert [item.source_dataset_name for item in datasets.descriptors] == [
        "first", "first", "second", "second"
    ]
    assert datasets.cached_indices == ()
    assert prepared == []


def test_deferred_initial_selection_loads_only_chosen_later_dataset_binning(monkeypatch):
    group = _group_with_named_binnings()
    prepared = []

    def prepare(dataset, **_kwargs):
        prepared.append(dataset.name)
        return dataset.data

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", prepare)
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    datasets, _names = project_gui.slice_viewer_datasets(
        group,
        use_composite=False,
        preload=False,
        selected_dataset_name="second",
        selected_binning_name="Overview",
    )

    assert datasets.initial_index == 3
    chosen = datasets[datasets.initial_index]
    assert prepared == ["second · Overview"]
    assert datasets.cached_indices == (3,)
    np.testing.assert_allclose(chosen.signal, 2.0)


def test_deferred_payloads_match_eager_results_and_default_stays_eager():
    group = _group_with_named_binnings()

    eager, eager_names = project_gui.slice_viewer_datasets(
        group, use_composite=False
    )
    deferred, deferred_names = project_gui.slice_viewer_datasets(
        group, use_composite=False, preload=False
    )

    assert isinstance(eager, list)
    assert eager_names == deferred_names
    assert deferred.cached_indices == ()
    for index, expected in enumerate(eager):
        actual = deferred[index]
        np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
        np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_allclose(actual.num_events, expected.num_events, equal_nan=True)


def test_named_composite_load_does_not_prepare_fit_or_unrelated_scope(monkeypatch):
    left = DatasetGroup(
        "Left", datasets=[DatasetEntry("left scan", _tiny_mdhisto_data(1.0))]
    )
    right = DatasetGroup(
        "Right", datasets=[DatasetEntry("right scan", _tiny_mdhisto_data(2.0))]
    )
    root = DataGroup("Workspace", subgroups=[left, right])
    scopes = [project_gui._composite_scope(root, node) for node in (left, right)]
    for scope in scopes:
        project_data.data_group_composite_config(scope).update(enabled=True)
        binning_id = project_data.add_data_group_composite_binning(
            scope, name="Overview"
        )
        project_data.data_group_composite_config_by_id(scope, binning_id)[
            "enabled"
        ] = True

    calls = []

    def composite_entry(scope, **kwargs):
        calls.append((scope.node.name, kwargs.get("binning_id"), kwargs.get("config_override")))
        source = scope.node.datasets[0]
        result = source.copy(name=f"{scope.node.name} Composite")
        result.metadata = {**result.metadata, "composite": True}
        return result

    monkeypatch.setattr(project_viewer_loading, "composite_dataset_entry", composite_entry)
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    datasets, names = project_gui.slice_viewer_datasets(
        root,
        preload=False,
        selected_dataset_name="Right Composite",
        selected_binning_name="Overview",
    )

    assert calls == []
    assert names == [
        "Left Composite", "Left Composite · Overview",
        "Right Composite", "Right Composite · Overview",
    ]
    assert datasets.initial_index == 3
    datasets[datasets.initial_index]
    assert len(calls) == 1
    assert calls[0][0] == "Right"
    assert calls[0][1] is not None
    assert calls[0][2] is not None


def test_project_explorer_refresh_preserves_deferred_loading(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.project_gui import NfitProject, NfitProjectExplorer

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = DataGroup(
        "Workspace",
        datasets=[
            DatasetEntry("first", _tiny_mdhisto_data(1.0)),
            DatasetEntry("second", _tiny_mdhisto_data(2.0)),
            DatasetEntry("third", _tiny_mdhisto_data(3.0)),
        ],
    )
    prepared = []

    def prepare(dataset, **_kwargs):
        prepared.append(dataset.name)
        return dataset.data

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", prepare)
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    explorer = NfitProjectExplorer(NfitProject([group]))

    class Combo:
        def currentText(self):
            return "second"

    class Viewer:
        dataset_combo = Combo()
        binning_combo = None
        unmask_model = False
        _nfit_use_composite = False
        _nfit_preload_data = False

        def replace_datasets(self, datasets, **kwargs):
            self.datasets = datasets
            self.replacement = kwargs

    viewer = Viewer()
    explorer._slice_viewers[id(group)] = [viewer]
    try:
        assert explorer.refresh_slice_viewer(group) is viewer
        assert isinstance(viewer.datasets, DeferredViewerDatasets)
        assert viewer.replacement["selected_dataset_name"] == "second"
        assert viewer.datasets.initial_index == 1
        assert viewer.datasets.cached_indices == (1,)
        assert prepared == ["second"]
    finally:
        explorer._slice_viewers.clear()
        explorer.window.close()


@pytest.mark.parametrize(
    ("preload", "initial_prepared"),
    [(False, ["second"]), (True, ["first", "first · Coarse", "second", "second · Overview"])],
)
def test_real_project_viewer_honors_preload_and_caches_switches(
    monkeypatch, preload, initial_prepared
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.project_gui import NfitProject, NfitProjectExplorer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = _group_with_named_binnings()
    prepared = []
    original_prepare = project_gui.dataset_for_slice_viewer

    def prepare(dataset, **kwargs):
        prepared.append(dataset.name)
        return original_prepare(dataset, **kwargs)

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", prepare)
    monkeypatch.setattr(project_gui, "preload_viewer_data", lambda: preload)
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    explorer = NfitProjectExplorer(NfitProject([group]))
    viewer = None
    try:
        viewer = explorer.open_slice_viewer(
            group, selected_dataset_name="second", use_composite=False
        )
        assert viewer is not None
        assert viewer.dataset_combo.currentText() == "second"
        assert prepared == initial_prepared

        if not preload:
            viewer.dataset_combo.setCurrentIndex(viewer.dataset_combo.findText("first"))
            app.processEvents()
            assert prepared == ["second", "first"]
            viewer.binning_combo.setCurrentIndex(
                viewer.binning_combo.findText("Coarse")
            )
            app.processEvents()
            assert prepared == ["second", "first", "first · Coarse"]
            viewer.dataset_combo.setCurrentIndex(viewer.dataset_combo.findText("second"))
            viewer.dataset_combo.setCurrentIndex(viewer.dataset_combo.findText("first"))
            viewer.binning_combo.setCurrentIndex(
                viewer.binning_combo.findText("Coarse")
            )
            app.processEvents()
            assert prepared == ["second", "first", "first · Coarse"]
    finally:
        if viewer is not None:
            viewer.window.close()
        explorer.window.close()


def test_lazy_open_cancellation_does_not_create_viewer(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.project_gui import NfitProject, NfitProjectExplorer

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = DataGroup(
        "Workspace", datasets=[DatasetEntry("scan", _tiny_mdhisto_data(1.0))]
    )
    monkeypatch.setattr(project_gui, "preload_viewer_data", lambda: False)

    def cancel(*_args, **_kwargs):
        raise project_gui.RebinCancellationRequested("cancel selected load")

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", cancel)
    explorer = NfitProjectExplorer(NfitProject([group]))
    created = []
    monkeypatch.setattr(
        explorer,
        "_create_slice_viewer",
        lambda *_args, **_kwargs: created.append(True),
    )
    try:
        assert explorer.open_slice_viewer(
            group, selected_dataset_name="scan", use_composite=False
        ) is None
        assert created == []
        assert id(group) not in explorer._slice_viewers
    finally:
        explorer.window.close()


def test_lazy_open_memory_preflight_is_limited_to_selected_binning(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.project_gui import NfitProject, NfitProjectExplorer

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = _group_with_named_binnings()
    project_data.dataset_rebin_config(group.datasets[1])["enabled"] = True
    monkeypatch.setattr(project_gui, "preload_viewer_data", lambda: False)
    monkeypatch.setattr(project_gui, "_project_binning_is_current", lambda *_args: False)
    explorer = NfitProjectExplorer(NfitProject([group]))
    checked = []
    monkeypatch.setattr(
        explorer,
        "_confirm_dataset_rebin_memory",
        lambda items: checked.append([(item["name"], item["id"]) for item in items]) or True,
    )
    monkeypatch.setattr(
        explorer,
        "_confirm_viewer_rebin_memory",
        lambda *_args, **_kwargs: pytest.fail("lazy loading must not preflight every binning"),
    )
    viewer = None
    try:
        viewer = explorer.open_slice_viewer(
            group, selected_dataset_name="second", use_composite=False
        )
        assert viewer is not None
        assert len(checked) == 1
        assert checked[0][0][0] == "Default"
        assert len(checked[0]) == 1
    finally:
        if viewer is not None:
            viewer.window.close()
        explorer.window.close()


def test_real_viewer_preserves_named_binning_for_initial_and_new_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.project_gui import NfitProject, NfitProjectExplorer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = _group_with_named_binnings()
    prepared = []
    original_prepare = project_gui.dataset_for_slice_viewer

    def prepare(dataset, **kwargs):
        prepared.append(dataset.name)
        return original_prepare(dataset, **kwargs)

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", prepare)
    monkeypatch.setattr(project_gui, "preload_viewer_data", lambda: False)
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    explorer = NfitProjectExplorer(NfitProject([group]))
    viewer = child = None
    try:
        viewer = explorer.open_slice_viewer(
            group,
            selected_dataset_name="second",
            selected_binning_name="Overview",
            use_composite=False,
        )
        assert viewer is not None
        assert viewer.dataset_combo.currentText() == "second"
        assert viewer.binning_combo.currentText() == "Overview"
        assert viewer.dataset_names[viewer.dataset_index] == "second · Overview"
        assert prepared == ["second · Overview"]

        child = viewer.open_new_viewer()
        app.processEvents()
        assert child is not None
        assert child.dataset_combo.currentText() == "second"
        assert child.binning_combo.currentText() == "Overview"
        assert child.dataset_names[child.dataset_index] == "second · Overview"
        assert prepared == ["second · Overview", "second · Overview"]
    finally:
        if child is not None:
            child.window.close()
        if viewer is not None:
            viewer.window.close()
        explorer.window.close()


def test_real_named_composite_matches_eager_payload():
    project_data._COMPOSITE_DATA_CACHE.clear()
    group = DataGroup(
        "Workspace",
        datasets=[
            DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto"),
            DatasetEntry("second", _tiny_mdhisto_data(3.0), kind="mdhisto"),
        ],
    )
    fit = project_data.data_group_composite_config(group)
    fit.update(enabled=True, minimum_coverage=0.0)
    named_id = project_data.add_data_group_composite_binning(
        group, name="Overview"
    )
    named = project_data.data_group_composite_config_by_id(group, named_id)
    named.update(enabled=True, minimum_coverage=0.0)

    eager, names = project_gui.slice_viewer_datasets(group)
    deferred, deferred_names = project_gui.slice_viewer_datasets(
        group,
        preload=False,
        selected_dataset_name="Workspace Composite",
        selected_binning_name="Overview",
    )

    assert names == deferred_names == [
        "Workspace Composite", "Workspace Composite · Overview"
    ]
    assert deferred.initial_index == 1
    assert deferred.cached_indices == ()
    actual = deferred[deferred.initial_index]
    expected = eager[1]
    assert deferred.cached_indices == (1,)
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.num_events, expected.num_events, equal_nan=True)


def test_viewer_loading_services_remain_gui_independent():
    package = Path(project_viewer_loading.__file__).parent
    forbidden_modules = {"project_gui", "project_data"}
    for filename in ("project_viewer_loading.py", "viewer_data.py"):
        tree = ast.parse((package / filename).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.startswith(("PySide", "PyQt")) for name in imports)
        assert not any(name.rsplit(".", 1)[-1] in forbidden_modules for name in imports)
