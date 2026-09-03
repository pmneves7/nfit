import numpy as np
import pytest

import nfit.project_gui as project_gui
from nfit.analysis import (
    AnalysisEntry,
    AnalysisOutputRef,
    AnalysisResultRecord,
    analysis_definition,
    composite_analysis_source_id,
    default_analysis_parameters,
)
from nfit.analysis.artifacts import dataset_artifact_bytes
from nfit.analysis.bragg import integrate_bragg_peaks
from nfit.analysis.fingerprint import dataset_entry_fingerprint, recipe_hash
from nfit.dataset import PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
from nfit.project_archive import replace_analysis_artifacts
from nfit.project_gui import NfitProject, NfitProjectExplorer, save_project


def test_analysis_tree_and_playground_controls_have_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace1", datasets=[DatasetEntry("scan", None)])
    explorer = NfitProjectExplorer(NfitProject([group]))
    workspace = explorer.tree.topLevelItem(0)
    assert [workspace.child(i).text(0) for i in range(workspace.childCount())] == ["Datasets", "Models", "Fits", "Analyses", "Plots"]

    explorer.tree.setCurrentItem(workspace)
    playground = explorer.open_data_playground_for_selection()
    assert playground is not None
    controls = [
        playground.analysis_combo, playground.new_button, playground.duplicate_button,
        playground.delete_button, playground.dataset_combo, playground.secondary_dataset_combo,
        playground.additional_inputs_button, playground.operation_combo,
        playground.name_edit, playground.result_tabs, playground.results,
        playground.diagnostics, playground.provenance, playground.run_button, playground.export_bragg_button,
        *playground.parameter_widgets.values(),
    ]
    assert controls
    assert all(control.toolTip() for control in controls)
    playground.window.close()


def test_analysis_window_config_and_output_panels_are_resizable(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace1", datasets=[DatasetEntry("scan", None)])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    playground = explorer.open_data_playground_for_selection()
    assert playground is not None

    splitter = playground.config_output_splitter
    assert splitter.objectName() == "analysis_config_output_splitter"
    assert splitter.widget(0) is playground.parameter_scroll
    assert splitter.widget(1) is playground.result_tabs
    assert not splitter.childrenCollapsible()
    assert splitter.orientation().name == "Vertical"
    assert splitter.toolTip()
    assert playground.parameter_scroll.minimumHeight() == 240
    assert playground.result_tabs.minimumHeight() == 180

    playground.window.show()
    QtWidgets.QApplication.processEvents()
    assert splitter.sizes()[0] > splitter.sizes()[1]
    playground.window.close()


def test_derived_dataset_details_edit_sources_and_arithmetic_at_top(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    axis = MDHistoAxis("H", np.array([0.0, 1.0]), "rlu", "momentum")

    def source_data(value):
        return MDHistoData(
            (axis,),
            np.array([value], dtype=float),
            np.array([1.0]),
            np.array([False]),
            np.array([1.0]),
        )

    cold = DatasetGroup(
        "1.8 K",
        datasets=[DatasetEntry("cold", source_data(2.0), kind="mdhisto")],
    )
    warm = DatasetGroup(
        "50 K",
        datasets=[DatasetEntry("warm", source_data(1.0), kind="mdhisto")],
    )
    group = DataGroup("Workspace1", subgroups=[cold, warm])
    for source in (cold, warm):
        config = project_gui.data_group_composite_config(
            project_gui._composite_scope(group, source)
        )
        config.update({"enabled": True, "fractional": False})
    analysis = AnalysisEntry(
        "Low minus 50 K",
        "histogram_arithmetic",
        [
            composite_analysis_source_id(group, cold),
            composite_analysis_source_id(group, warm),
        ],
        {"operation": "subtract", "right_scale": 1.0},
    )
    derived = project_gui.create_derived_analysis_dataset(group, analysis)
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer._set_dataset_details(derived, group)

    tabs = explorer.details_widget.findChild(
        QtWidgets.QTabWidget, "dataset_details_tabs"
    )
    processing_tab = explorer.details_widget.findChild(
        QtWidgets.QScrollArea, "dataset_details_processing"
    )
    recipe_panel = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "derived_recipe_controls"
    )
    assert tabs is not None and processing_tab is not None
    assert recipe_panel is not None and processing_tab.isAncestorOf(recipe_panel)
    left = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "derived_recipe_source_0"
    )
    right = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "derived_recipe_source_1"
    )
    operation = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "derived_recipe_operation"
    )
    scale = explorer.details_widget.findChild(
        QtWidgets.QDoubleSpinBox, "derived_recipe_right_scale"
    )
    stage = explorer.details_widget.findChild(
        QtWidgets.QLabel, "derived_recipe_source_stage"
    )
    controls = [left, right, operation, scale, stage]
    assert all(control is not None and control.toolTip() for control in controls)
    assert left.currentData() == composite_analysis_source_id(group, cold)
    assert right.currentData() == composite_analysis_source_id(group, warm)

    left.setCurrentIndex(left.findData(composite_analysis_source_id(group, warm)))
    operation.setCurrentIndex(operation.findData("add"))
    scale.setValue(2.5)
    scale.editingFinished.emit()

    assert analysis.input_dataset_ids[0] == composite_analysis_source_id(group, warm)
    assert analysis.parameters == {"operation": "add", "right_scale": 2.5}
    assert derived.metadata["derived_recipe"]["input_source_ids"] == analysis.input_dataset_ids
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_analyses_branch_new_analysis_button_opens_fresh_recipe(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace1", datasets=[DatasetEntry("scan", None)])
    explorer = NfitProjectExplorer(NfitProject([group]))
    analyses_item = explorer.tree.topLevelItem(0).child(3)
    explorer.tree.setCurrentItem(analyses_item)

    open_button = explorer.open_analysis_button
    new_button = explorer.new_analysis_button
    assert not open_button.isHidden()
    assert open_button.toolTip()
    assert explorer.create_group_button.isHidden()
    assert explorer.tree_action_layout.indexOf(open_button) + 1 == (
        explorer.tree_action_layout.indexOf(explorer.delete_button)
    )
    assert not new_button.isHidden()
    assert new_button.toolTip()
    open_button.click()

    playground = explorer._analysis_window
    assert playground is not None
    assert playground.group is group
    assert playground.window.windowTitle() == "nfit Analysis Window"
    new_button.click()
    assert playground.analysis_combo.currentData() is None
    assert playground.name_edit.text() == "Analysis"
    assert explorer.context_menu_action_names(analyses_item) == [
        "Open Analysis Window",
        "New analysis",
    ]
    playground.window.close()


def test_analysis_workflow_script_can_be_copied_from_tree(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("scan", None)
    analysis = AnalysisEntry("Powder average", "spherical_average", [dataset.id], {})
    group = DataGroup("Workspace1", datasets=[dataset], analyses=[analysis])
    explorer = NfitProjectExplorer(NfitProject([group]))
    analysis_item = explorer.tree.topLevelItem(0).child(3).child(0)
    explorer.tree.setCurrentItem(analysis_item)
    monkeypatch.setattr(
        project_gui,
        "analysis_workflow_script",
        lambda project, analysis_id: f"# analysis {analysis_id}\n",
    )

    actions = explorer.context_menu_action_names(analysis_item)
    assert "Copy workflow script" in actions
    assert "Save workflow script..." in actions
    assert explorer.copy_workflow_script_for_selection()
    assert QtWidgets.QApplication.clipboard().text() == f"# analysis {analysis.id}\n"


def test_delete_key_removes_selected_analysis_and_linked_datasets(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")
    pytest.importorskip("PySide6.QtWidgets")

    analysis = AnalysisEntry("Peaks", "bragg_integration", [], {})
    linked = DatasetEntry(
        "Integrated peaks",
        None,
        metadata={"derived_from_analysis": {"analysis_id": analysis.id}},
    )
    unrelated = DatasetEntry("Keep me", None)
    subgroup = DatasetGroup("Derived data", datasets=[linked, unrelated])
    group = DataGroup("Workspace1", subgroups=[subgroup], analyses=[analysis])
    explorer = NfitProjectExplorer(NfitProject([group]))
    analysis_item = explorer.tree.topLevelItem(0).child(3).child(0)
    explorer.tree.setCurrentItem(analysis_item)
    explorer.tree.setFocus()

    assert not explorer.open_analysis_button.isHidden()
    assert explorer.create_group_button.isHidden()
    explorer.open_analysis_button.click()
    assert explorer._analysis_window is not None
    assert explorer._analysis_window.analysis_combo.currentData() == analysis.id
    explorer._analysis_window.window.close()
    assert explorer.delete_button.isEnabled()
    assert "Delete" in explorer.context_menu_action_names(analysis_item)
    QtTest.QTest.keyClick(explorer.tree, QtCore.Qt.Key.Key_Delete)

    assert group.analyses == []
    assert subgroup.datasets == [unrelated]
    assert explorer.has_unsaved_changes


def test_analysis_tree_marks_changed_inputs_stale(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("scan", None)
    group = DataGroup("Workspace1", datasets=[dataset])
    parameters = default_analysis_parameters("bragg_integration")
    analysis = AnalysisEntry("Peaks", "bragg_integration", [dataset.id], parameters)
    analysis.result = AnalysisResultRecord(
        recipe_hash("bragg_integration", analysis_definition("bragg_integration").version, parameters, [dataset.id]),
        {dataset.id: dataset_entry_fingerprint(dataset, group)},
        [AnalysisOutputRef("count", "Count", "scalar", scalar_value=1.0)], "success", "now",
    )
    group.analyses.append(analysis)
    explorer = NfitProjectExplorer(NfitProject([group]))
    assert "[fresh]" in explorer.tree.topLevelItem(0).child(3).child(0).text(0)
    dataset.scale_factor = 2.0
    explorer._refresh_tree()
    assert "[stale]" in explorer.tree.topLevelItem(0).child(3).child(0).text(0)


def test_analysis_fingerprint_reuses_array_hashes_for_setting_changes(monkeypatch):
    import nfit.analysis.fingerprint as fingerprint

    fingerprint._DATASET_FINGERPRINT_CACHE.clear()
    fingerprint._DATA_ARRAY_HASH_CACHE.clear()
    dataset = DatasetEntry("scan", _bragg_volume_for_gui())
    group = DataGroup("Workspace1", datasets=[dataset])
    calls = 0
    original = fingerprint.array_hash

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(fingerprint, "array_hash", counted)
    first = dataset_entry_fingerprint(dataset, group)
    first_calls = calls
    dataset.scale_factor = 2.0
    second = dataset_entry_fingerprint(dataset, group)

    assert second != first
    assert calls == first_calls


def test_analysis_fingerprint_hashes_replacement_data_instead_of_stale_source(tmp_path):
    source = tmp_path / "scan.nxs"
    source.write_bytes(b"source")
    dataset = DatasetEntry(
        "scan",
        _bragg_volume_for_gui(),
        metadata={"source_file": str(source)},
    )
    group = DataGroup("Workspace1", datasets=[dataset])
    first = dataset_entry_fingerprint(dataset, group)

    editable = dataset.data.mutable_copy()
    editable.signal.flat[0] += 1.0
    dataset.replace_data(editable)
    second = dataset_entry_fingerprint(dataset, group)

    assert second != first


def test_bragg_analysis_loads_a_lazy_primary_dataset(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("lazy scan", None, metadata={"source_file": "scan.nxs"})
    group = DataGroup("Workspace1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    analysis_window = explorer.open_data_playground_for_selection()
    assert analysis_window is not None
    loaded = MDHistoData(
        axes=(MDHistoAxis("E", np.array([0.0, 1.0]), "meV", "energy"),),
        signal=np.array([1.0]),
        errors=np.array([1.0]),
        mask=np.array([False]),
        num_events=np.array([1.0]),
        metadata={},
    )
    calls = []

    def load(entry, **kwargs):
        calls.append((entry, kwargs))
        entry.replace_data(loaded, source_backed=True)
        return loaded

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", load)
    assert analysis_window._primary_analysis_data(dataset) is loaded
    assert calls == [
        (
            dataset,
            {
                "extra_masks": [],
                "force_rebin": True,
                "force_masks": True,
                "progress_callback": None,
            },
        )
    ]
    analysis_window.window.close()


def _bragg_volume_for_gui():
    edges = np.linspace(-1.5, 1.5, 13)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, k, l = np.meshgrid(centers, centers, centers, indexing="ij")
    signal = 1.0 + 8.0 * np.exp(-0.5 * (h**2 + k**2 + l**2) / 0.2**2)
    axes = tuple(MDHistoAxis(name, edges, "r.l.u.", "momentum") for name in ("H", "K", "L"))
    return MDHistoData(
        axes,
        signal,
        np.full(signal.shape, 0.1),
        np.zeros(signal.shape, bool),
        np.ones(signal.shape),
        metadata={
            "signal_semantics": "density",
            "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi},
        },
    )


def test_bragg_parameter_groups_hide_irrelevant_controls(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace1", datasets=[DatasetEntry("scan", _bragg_volume_for_gui())])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    window = explorer.open_data_playground_for_selection()
    window.operation_combo.setCurrentIndex(window.operation_combo.findData("bragg_integration"))

    groups = window.parameter_panel.findChildren(QtWidgets.QGroupBox)
    assert {box.title() for box in groups} >= {
        "Peak selection", "Elastic volume", "Integration region", "Background",
        "Quality filters", "Gaussian fit", "Numerics",
    }
    assert window.parameter_widgets["method"].currentData() == "gaussian_fit"
    assert window.parameter_widgets["background_mode"].currentData() == "shell"
    assert not window.parameter_widgets["gaussian_max_nfev"].isHidden()
    window.parameter_widgets["method"].setCurrentIndex(
        window.parameter_widgets["method"].findData("box_sum")
    )
    assert window.parameter_widgets["gaussian_max_nfev"].isHidden()
    assert all(widget.toolTip() for widget in window.parameter_widgets.values())
    window.window.close()


def test_saved_bragg_result_renders_tables_diagnostics_and_materializes_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    source = _bragg_volume_for_gui()
    dataset = DatasetEntry("scan", source, data_type="single_crystal_elastic")
    group = DataGroup("Workspace1", datasets=[dataset])
    parameters = default_analysis_parameters("bragg_integration")
    analysis = AnalysisEntry("Nuclear peaks", "bragg_integration", [dataset.id], parameters)
    table = integrate_bragg_peaks(
        source,
        [[0.0, 0.0, 0.0], [1.45, 1.45, 1.45]],
        method="ellipsoid_sum",
        ellipsoid_semiaxes=[0.2] * 3,
        background_mode="shell",
        minimum_peak_coverage=0.8,
    )
    project_path = tmp_path / "project.nfit"
    project = NfitProject([group])
    save_project(NfitProject(), project_path)
    artifact_path = replace_analysis_artifacts(
        project_path,
        analysis.id,
        {"peak_table.npz": dataset_artifact_bytes(table)},
    )["peak_table.npz"]
    output = AnalysisOutputRef(
        "peak_table",
        "Integrated Bragg peaks",
        "table",
        artifact_path=artifact_path,
        dataset_id="bragg-table-id",
        metadata={"data_type": "bragg_reflections", "fit_enabled": False},
    )
    analysis.result = AnalysisResultRecord(
        "recipe",
        {dataset.id: "fingerprint"},
        [output],
        "success",
        "now",
        diagnostics={"generated_peaks": 2, "accepted_peaks": table.metadata["accepted_count"]},
    )
    group.analyses.append(analysis)
    explorer = NfitProjectExplorer(project)
    explorer.project_path = project_path
    explorer._refresh_tree(select_group=group)
    window = explorer.open_data_playground_for_selection()
    window.analysis_combo.setCurrentIndex(window.analysis_combo.findData(analysis.id))
    window._select_analysis()

    assert window.result_table.rowCount() == 2
    assert window.bragg_diagnostics.peak_table.rowCount() == 2
    assert window.result_table.isSortingEnabled()
    assert window.bragg_diagnostics.peak_table.isSortingEnabled()
    headers = [
        window.bragg_diagnostics.peak_table.horizontalHeaderItem(column).text()
        for column in range(window.bragg_diagnostics.peak_table.columnCount())
    ]
    assert headers[:7] == ["Accepted", "H", "K", "L", "I", "dI", "I/dI"]
    h_column = headers.index("H")
    window.bragg_diagnostics.peak_table.sortItems(
        h_column, QtCore.Qt.SortOrder.DescendingOrder
    )
    QtWidgets.QApplication.processEvents()
    assert float(window.bragg_diagnostics.peak_table.item(0, h_column).text()) >= float(
        window.bragg_diagnostics.peak_table.item(1, h_column).text()
    )
    assert len(window.bragg_diagnostics.figure.axes) == 6
    assert "accepted_peaks" in window.diagnostic_summary.toPlainText()
    assert not group.subgroups

    assert window.add_current_output_to_datasets()
    materialized = next(group.iter_subgroups()).datasets[0]
    assert materialized.data_type == "bragg_reflections"
    assert materialized.enabled is False
    assert materialized.fit_weight == 0.0

    analyses_item = explorer.tree.topLevelItem(0).child(3)
    output_item = analyses_item.child(0).child(0)
    explorer.tree.setCurrentItem(output_item)
    output_table = explorer.details_widget.findChild(QtWidgets.QTableWidget, "analysis_output_table")
    metadata_tree = explorer.details_widget.findChild(QtWidgets.QTreeWidget, "analysis_output_metadata_tree")
    assert output_table is not None and output_table.rowCount() == 2
    assert metadata_tree is not None
    window.window.close()


def test_bragg_result_export_writes_accepted_reflections_as_int(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    source = _bragg_volume_for_gui()
    dataset = DatasetEntry("scan", source, data_type="single_crystal_elastic")
    group = DataGroup("Workspace1", datasets=[dataset])
    table = integrate_bragg_peaks(
        source,
        [[0.0, 0.0, 0.0], [1.45, 1.45, 1.45]],
        method="ellipsoid_sum",
        ellipsoid_semiaxes=[0.2] * 3,
        background_mode="shell",
        minimum_peak_coverage=0.8,
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    window = explorer.open_data_playground_for_selection()
    window._current_result_data = table
    output_path = tmp_path / "peaks.int"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "Intensity files (*.int)"),
    )

    assert window.export_current_bragg_int()
    rows = [line.split() for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert all(len(row) == 5 for row in rows)
    assert len(rows) == int(np.count_nonzero(table.column("Accepted")))
    window.window.close()


def test_bragg_int_export_rounds_hkl_and_preserves_intensity(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    window = explorer.open_data_playground_for_selection()
    window._current_result_data = PointListData(
        columns={
            "H": [1.49, -1.51, 8.2],
            "K": [-2.49, 2.51, 8.2],
            "L": [0.2, -0.8, 8.2],
            "I": [12.345, 67.89, 100.0],
            "dI": [0.123, 0.456, 1.0],
            "Accepted": [1.0, 1.0, 0.0],
        },
        metadata={"analysis_kind": "bragg_peak_integration"},
    )
    output_path = tmp_path / "rounded.int"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "Intensity files (*.int)"),
    )

    assert window.export_current_bragg_int()
    rows = [line.split() for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        ["1", "-2", "0", "12.345", "0.123"],
        ["-2", "3", "-1", "67.89", "0.456"],
    ]
    window.window.close()


def test_data_viewer_marks_accepted_and_rejected_bragg_peaks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    source = _bragg_volume_for_gui()
    table = integrate_bragg_peaks(
        source,
        [[0.0, 0.0, 0.0], [1.45, 1.45, 1.45]],
        method="ellipsoid_sum",
        ellipsoid_semiaxes=[0.2] * 3,
        minimum_peak_coverage=0.8,
    )
    viewer = QtMDHistoSliceViewer(source, x_dim=0, y_dim=1)
    viewer.set_bragg_peak_overlay(table)

    assert viewer.bragg_peak_overlay is not None
    assert np.count_nonzero(viewer.bragg_peak_overlay["accepted"]) == 1
    assert len(viewer.ax_image.collections) >= 3
    viewer.window.close()
