import pytest

import nfit.project_gui as project_gui
from nfit.analysis import (
    AnalysisEntry,
    AnalysisOutputRef,
    AnalysisResultRecord,
    analysis_definition,
    default_analysis_parameters,
)
from nfit.analysis.fingerprint import dataset_entry_fingerprint, recipe_hash
from nfit.pipeline import DataGroup, DatasetEntry
from nfit.project_gui import NfitProject, NfitProjectExplorer


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
        playground.diagnostics, playground.provenance, playground.run_button,
        *playground.parameter_widgets.values(),
    ]
    assert controls
    assert all(control.toolTip() for control in controls)
    playground.window.close()


def test_analyses_branch_new_analysis_button_opens_fresh_recipe(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace1", datasets=[DatasetEntry("scan", None)])
    explorer = NfitProjectExplorer(NfitProject([group]))
    analyses_item = explorer.tree.topLevelItem(0).child(3)
    explorer.tree.setCurrentItem(analyses_item)

    button = explorer.new_analysis_button
    assert button is not None
    assert not button.isHidden()
    assert button.toolTip()
    button.click()

    playground = explorer._analysis_window
    assert playground is not None
    assert playground.group is group
    assert playground.window.windowTitle() == "nfit Analysis Window"
    assert playground.analysis_combo.currentData() is None
    assert playground.name_edit.text() == "Analysis"
    assert explorer.context_menu_action_names(analyses_item) == [
        "Open Analysis Window",
        "New analysis",
    ]
    playground.window.close()


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


def test_bragg_analysis_loads_a_lazy_primary_dataset(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("lazy scan", None, metadata={"source_file": "scan.nxs"})
    group = DataGroup("Workspace1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    analysis_window = explorer.open_data_playground_for_selection()
    assert analysis_window is not None
    loaded = object()
    calls = []

    def load(entry):
        calls.append(entry)
        entry.data = loaded
        return loaded

    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", load)
    assert analysis_window._primary_analysis_data(dataset) is loaded
    assert calls == [dataset]
    analysis_window.window.close()
