import pytest

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
        playground.delete_button, playground.dataset_combo, playground.secondary_dataset_combo, playground.operation_combo,
        playground.name_edit, playground.result_tabs, playground.results,
        playground.diagnostics, playground.provenance, playground.run_button,
        *playground.parameter_widgets.values(),
    ]
    assert controls
    assert all(control.toolTip() for control in controls)
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
