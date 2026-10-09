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


def test_viewer_availability_does_not_validate_conditions_or_touch_sources(monkeypatch):
    from nfit import dataset_criteria, dataset_criterion_mask, dataset_criterion_preview

    node = DatasetGroup("Runs", datasets=[
        DatasetEntry(str(i), None, kind="raw_dgs_nexus",
                     metadata={"source_file": f"/unavailable/run{i}.nxs", "temperature": i})
        for i in range(100)
    ], metadata={"raw_dgs": {"format": "raw-direct-geometry-nexus"},
                 "composite": {"enabled": True}})
    root = DataGroup("Workspace", subgroups=[node])
    mask = dataset_criterion_mask(node, channel="metadata", source="metadata/temperature",
                                  right_value=50.)
    dataset_criterion_preview(node, mask)
    # A pending source change must not force validation merely to enable a
    # navigation button. Scientific preparation still checks it on execution.
    node.datasets[0].metadata["temperature"] = 101.
    with monkeypatch.context() as guarded:
        guarded.setattr(dataset_criteria, "value_signature",
                        lambda *_args: pytest.fail("validated diagnostic while browsing"))
        guarded.setattr(Path, "stat", lambda *_args, **_kwargs: pytest.fail("stat while browsing"))
        assert project_gui._has_slice_viewer_candidates(root)
    assert all(dataset.data is None for dataset in node.datasets)
    with pytest.raises(ValueError, match="new or changed"):
        root.select()


@pytest.mark.parametrize("metadata_key,source_format,kind", [
    ("raw_dgs", "raw-direct-geometry-nexus", "raw_dgs_nexus"),
    ("raw_dgs", "corelli-correlation-nexus", "raw_dgs_nexus"),
    ("mdevent", "mantid-mdevent", "mdevent"),
])
@pytest.mark.parametrize("use_composite", [False, True])
def test_unbinned_event_inputs_are_hidden_in_eager_and_deferred_viewers(
    monkeypatch, metadata_key, source_format, kind, use_composite,
):
    sources = DatasetGroup("Background sources", datasets=[
        DatasetEntry(f"run {run}", None, kind=kind) for run in range(32)
    ], metadata={metadata_key: {"format": source_format}, "composite": {"enabled": False}})
    ordinary = DatasetEntry("Imported histogram", _tiny_mdhisto_data(1.0), kind="mdhisto")
    root = DataGroup("Workspace", datasets=[ordinary], subgroups=[sources])
    prepared = []

    def prepare(dataset, **_kwargs):
        assert dataset.kind == "mdhisto"
        prepared.append(dataset.name)
        return dataset.data

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", prepare)
    monkeypatch.setattr(project_gui, "current_model_channels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    eager, names = project_gui.slice_viewer_datasets(root, use_composite=use_composite)
    deferred, deferred_names = project_gui.slice_viewer_datasets(
        root, use_composite=use_composite, preload=False,
    )
    assert names == deferred_names == [ordinary.name]
    assert len(eager) == 1
    assert deferred.cached_indices == ()
    assert prepared == [ordinary.name]
    assert project_gui._effective_dataset_entry_count(root, root, use_composite=use_composite) == 1
    assert project_gui._viewer_progress_work_counts(root, root, use_composite=use_composite) == (1, 0, 1)
    assert len(sources.datasets) == 32
    assert all(dataset.data is None for dataset in sources.datasets)
    assert not project_viewer_loading.has_viewer_candidates(DataGroup("Only events", subgroups=[sources]))


@pytest.mark.parametrize("metadata_key", ["raw_dgs", "mdevent"])
def test_binned_event_collections_keep_their_histogram_and_named_binnings(metadata_key):
    node = DatasetGroup("Sample", datasets=[DatasetEntry("run", None)],
                        metadata={metadata_key: {}, "composite": {"enabled": True}})
    root = DataGroup("Workspace", subgroups=[node])
    scope = project_gui._composite_scope(root, node)
    binning_id = project_data.add_data_group_composite_binning(scope, name="Overview")
    project_data.data_group_composite_config_by_id(scope, binning_id)["enabled"] = True
    items = project_viewer_loading.plan_viewer_items(
        root, use_composite=True, dataset_binnings=lambda _dataset: [],
    )
    assert [item.name for item in items] == ["Sample Composite", "Sample Composite · Overview"]
    assert all(item.dataset is None for item in items)
    assert project_gui._effective_dataset_entry_count(root, root, use_composite=True) == 1
    assert project_gui._viewer_progress_work_counts(root, root, use_composite=True) == (1, 1, 2)


def test_viewer_policy_preserves_independent_results_and_lazy_sources(monkeypatch):
    from nfit.dataset import PointListData

    curve = PointListData({"T": [2.0, 3.0], "chi": [1.0, 2.0]},
                          coordinate_names=["T"], channels=[{"label": "chi", "value": "chi", "error": None}])
    visible = [
        DatasetEntry("Excluded histogram", _tiny_mdhisto_data(1.0), kind="mdhisto", enabled=False, fit_weight=0),
        DatasetEntry("Susceptibility", curve, data_type="magnetization"),
        DatasetEntry("Unloaded import", None, metadata={"source_file": "/missing/scan.npz"}),
        DatasetEntry("Analysis output", None, kind="analysis", metadata={"analysis_artifact_path": "assets/output.npz"}),
        DatasetEntry("Live result", None, metadata={"derived_recipe": {"analysis_id": "operation"}}),
    ]
    hidden = [
        DatasetEntry("Placeholder", None),
        DatasetEntry("Raw acquisition", None, kind="raw_dgs_nexus", metadata={"source_file": "/missing/raw.h5"}),
        DatasetEntry("MDE acquisition", None, kind="mdevent", metadata={"source_file": "/missing/events.nxs"}),
    ]
    root = DataGroup("Workspace", datasets=[*visible, *hidden])
    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", lambda *_args, **_kwargs: _tiny_mdhisto_data(1.0))
    monkeypatch.setattr(project_gui, "current_model_channels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    eager, names = project_gui.slice_viewer_datasets(root, use_composite=False)
    deferred, deferred_names = project_gui.slice_viewer_datasets(root, use_composite=False, preload=False)
    assert names == deferred_names == [dataset.name for dataset in visible]
    assert len(eager) == len(visible)
    assert deferred.cached_indices == ()
    assert project_gui._effective_dataset_entry_count(root, root, use_composite=False) == len(visible)
    assert project_gui._viewer_progress_work_counts(root, root, use_composite=False) == (len(visible), 0, len(visible))
    for dataset in visible:
        assert project_viewer_loading.has_viewer_candidates(DataGroup("Visible", datasets=[dataset]))
    for dataset in hidden:
        assert not project_viewer_loading.has_viewer_candidates(DataGroup("Hidden", datasets=[dataset]))


def test_viewer_catalog_uses_collection_ownership_not_fit_participation(monkeypatch):
    member = DatasetEntry("Independent source", _tiny_mdhisto_data(1.0), kind="mdhisto", enabled=False)
    child = DatasetGroup("Child", datasets=[member], enabled=False)
    parent = DatasetGroup("Collection", subgroups=[child], enabled=False,
                          metadata={"composite": {"enabled": True}})
    root = DataGroup("Workspace", subgroups=[parent])
    monkeypatch.setattr(project_gui, "current_model_channels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    calls = []

    def composite(scope, **_kwargs):
        calls.append(scope.node.name)
        return DatasetEntry("Collection Composite", _tiny_mdhisto_data(1.0), metadata={"composite": True})

    monkeypatch.setattr(project_gui, "composite_dataset_entry", composite)
    monkeypatch.setattr(project_viewer_loading, "composite_dataset_entry", composite)
    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", lambda dataset, **_kwargs: dataset.data)
    eager, names = project_gui.slice_viewer_datasets(root)
    deferred, deferred_names = project_gui.slice_viewer_datasets(root, preload=False)
    assert names == deferred_names == ["Collection Composite"]
    assert len(eager) == 1
    assert calls == ["Collection"]
    assert deferred.cached_indices == ()
    assert project_viewer_loading.plan_viewer_items(root, use_composite=False, dataset_binnings=lambda _ds: [])[0].dataset is member


@pytest.mark.parametrize("metadata_key,kind", [("raw_dgs", "raw_dgs_nexus"), ("mdevent", "mdevent")])
def test_prepared_results_in_source_collections_follow_their_current_representation(
    monkeypatch, metadata_key, kind,
):
    source = DatasetEntry("Acquisition", None, kind=kind, metadata={"source_file": "/missing/source.nxs"})
    prepared = DatasetEntry("Prepared run", _tiny_mdhisto_data(1.0), kind=kind)
    artifact = DatasetEntry("Saved result", None, metadata={"project_artifact_path": "assets/result.npz"})
    child = DatasetGroup("Prepared outputs", datasets=[artifact])
    native = DatasetGroup("Source collection", datasets=[source, prepared], subgroups=[child],
                          metadata={metadata_key: {}, "composite": {"enabled": False}})
    root = DataGroup("Workspace", subgroups=[native])
    monkeypatch.setattr(project_gui, "current_model_channels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", lambda *_args, **_kwargs: _tiny_mdhisto_data(1.0))
    eager, names = project_gui.slice_viewer_datasets(root)
    deferred, deferred_names = project_gui.slice_viewer_datasets(root, preload=False)
    assert names == deferred_names == ["Prepared run", "Saved result"]
    assert len(eager) == 2
    assert deferred.cached_indices == ()
    assert project_gui._effective_dataset_entry_count(root, root, use_composite=True) == 2
    assert project_gui._viewer_progress_work_counts(root, root, use_composite=True) == (2, 0, 2)


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


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("selection", ["dataset", "root_composite", "nested_composite", "owned_dataset"])
def test_show_selected_rebin_opens_dropdown_binning(monkeypatch, preload, selection):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.project_gui import NfitProject, NfitProjectExplorer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0), kind="mdhisto")
    if selection == "nested_composite":
        node = DatasetGroup("Collection", datasets=[dataset])
        group = DataGroup("Workspace", subgroups=[node])
    else:
        group = DataGroup("Workspace", datasets=[dataset])
        node = group
    if selection == "dataset":
        binning_id = project_data.add_dataset_rebin_binning(dataset, name="Chosen view")
        project_data.dataset_rebin_config_by_id(dataset, binning_id)["enabled"] = True
    else:
        scope = project_gui._composite_scope(group, node)
        project_data.data_group_composite_config(scope).update(enabled=True, minimum_coverage=0.0)
        binning_id = project_data.add_data_group_composite_binning(scope, name="Chosen view")
        project_data.data_group_composite_config_by_id(scope, binning_id).update(
            enabled=True, minimum_coverage=0.0
        )
    monkeypatch.setattr(project_gui, "preload_viewer_data", lambda: preload)
    monkeypatch.setattr(project_gui, "current_model_channel", lambda *_args, **_kwargs: None)
    explorer = NfitProjectExplorer(NfitProject([group]))
    viewer = None

    def find_item(target):
        iterator = QtWidgets.QTreeWidgetItemIterator(explorer.tree)
        while item := iterator.value():
            item_group, entry, _mask, _model, role = explorer._objects_for_item(item)
            if target is dataset and role == "dataset" and entry is dataset:
                return item
            if target is group and role == "datasets" and item_group is group:
                return item
            if target is node and role == "dataset_group" and explorer._dataset_group_for_item(item) is node:
                return item
            iterator += 1
        raise AssertionError("missing tree selection")

    try:
        explorer.tree.setCurrentItem(find_item(dataset if selection == "dataset" else node))
        selector = explorer.window.findChild(
            QtWidgets.QComboBox,
            "dataset_rebin_binning" if selection == "dataset" else "group_composite_binning",
        )
        assert selector is not None
        selector.setCurrentIndex(selector.findData(binning_id))
        app.processEvents()
        if selection == "owned_dataset":
            explorer.tree.setCurrentItem(find_item(dataset))
        viewer = explorer.open_slice_viewer_for_selection()
        assert viewer is not None
        assert viewer.binning_combo.currentText() == "Chosen view"
        assert viewer.data.metadata["binning_id"] == binning_id
        if not preload:
            assert viewer.datasets.cached_indices == (viewer.datasets.initial_index,)
    finally:
        if viewer is not None:
            viewer.window.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()
