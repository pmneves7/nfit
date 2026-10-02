"""Collection workflow sections retain lazy state and existing public scientific controls."""

import numpy as np
import pytest

pytest_plugins = ["tests.test_reduction_recipe_gui"]


def test_collection_sections_separate_reduction_and_histogram_settings(recipe_explorer):
    from PySide6 import QtWidgets

    from nfit.reduction_recipes import reduction_settings_schema

    explorer, group = recipe_explorer
    tabs = explorer.details_widget.findChild(QtWidgets.QTabWidget, "collection_workflow_tabs")
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Sources", "Reduction", "Binning and combination", "Plots and cuts"
    ]
    assert all(tabs.tabToolTip(index) for index in range(tabs.count()))
    for field in reduction_settings_schema(group):
        target = tabs.widget(2 if field.scope == "histogram" else 1)
        editor = target.findChild(QtWidgets.QWidget, f"raw_dgs_{field.key}")
        assert editor is not None and editor.toolTip()
    assert tabs.widget(0).findChild(QtWidgets.QTableWidget, "group_datasets_table") is not None
    assert tabs.widget(3).findChild(QtWidgets.QPushButton, "collection_open_viewer").toolTip()
    assert all(dataset.data is None for dataset in group.datasets)
    tabs.setCurrentIndex(2)
    explorer._set_dataset_collection_details(explorer.project.data_groups[0], group)
    rebuilt = explorer.details_widget.findChild(QtWidgets.QTabWidget, "collection_workflow_tabs")
    assert rebuilt.tabText(rebuilt.currentIndex()) == "Binning and combination"


def test_collection_shared_values_report_mixed_overrides(recipe_explorer):
    from PySide6 import QtWidgets

    from nfit.reduction_recipes import set_reduction_settings

    explorer, group = recipe_explorer
    set_reduction_settings(group, {"t0_override": -15.0}, dataset_ids=[group.datasets[0].id])
    explorer._set_dataset_collection_details(explorer.project.data_groups[0], group)
    summary = explorer.details_widget.findChild(QtWidgets.QLabel, "raw_dgs_mixed_reduction_values")
    assert "Mixed effective values: T0 override" in summary.text()
    assert "1 of 2 enabled runs" in summary.text()
    assert summary.toolTip()
    automatic = explorer.details_widget.findChild(QtWidgets.QCheckBox, "raw_dgs_t0_override_automatic")
    assert automatic.isChecked()  # Shared editor remains inherited automatic, not -15.
    assert all(dataset.data is None for dataset in group.datasets)


def test_resolved_symmetry_lists_the_actual_dual_hkl_matrices():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtWidgets

    from nfit.project_rebin_panels import rebin_symmetry_operations_widget
    from nfit.symmetry import resolve_symmetry, symmetry_spec_from_config

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    config = {"symmetry": {"mode": "operations", "expression": "x,y,z;y,z,x;z,x,y;-x,-y,-z;-y,-z,-x;-z,-x,-y"}}
    widget = rebin_symmetry_operations_widget(config, object_prefix="test")
    tree = widget.findChild(QtWidgets.QTreeWidget, "test_resolved_symmetry_operations")
    expected = resolve_symmetry(symmetry_spec_from_config(config["symmetry"]))
    assert tree.topLevelItemCount() == len(expected) == 6
    for index, operation in enumerate(expected):
        assert operation.label in tree.topLevelItem(index).text(0)
        assert np.array2string(operation.matrix_hkl, precision=8, separator=", ", max_line_width=1000) == tree.topLevelItem(index).text(1)
    assert all(label.toolTip() for label in widget.findChildren(QtWidgets.QLabel))
    config["symmetry"]["mode"] = "none"
    widget.refresh_operations()
    assert tree.topLevelItemCount() == 1
    config["symmetry"] = {"mode": "operations", "expression": "nonsense"}
    widget.refresh_operations()
    assert tree.topLevelItem(0).text(0) == "Invalid symmetry"
    widget.close()
    assert app is not None


def test_measurement_average_choices_retain_saved_keys_and_explain_target():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtCore, QtWidgets

    from nfit.project_rebin_panels import measurement_average_choices

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = QtWidgets.QComboBox()
    measurement_average_choices(combo)
    assert [combo.itemData(index) for index in range(combo.count())] == ["inverse_variance", "uniform"]
    assert "Native DGS/MDE" in combo.toolTip()
    assert all(combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole) for index in range(combo.count()))
    combo.close()
    assert app is not None


def test_collection_complete_workflow_button_uses_public_export(recipe_explorer):
    from PySide6 import QtWidgets

    from nfit.workflow import composite_workflow_script

    explorer, group = recipe_explorer
    root = explorer.project.data_groups[0]
    button = explorer.details_widget.findChild(QtWidgets.QPushButton, "collection_copy_workflow")
    assert button is not None and button.toolTip()
    button.click()
    assert QtWidgets.QApplication.clipboard().text() == composite_workflow_script(
        explorer.project, root.name, node_id=group.id
    )
    assert all(dataset.data is None for dataset in group.datasets)


def test_collection_workflow_export_reports_unsupported_source(recipe_explorer, monkeypatch):
    from PySide6 import QtWidgets

    from nfit import workflow

    explorer, _group = recipe_explorer
    messages = []

    def unsupported(*_args, **_kwargs):
        raise workflow.WorkflowValidationError("Original acquisition sources are required.")

    monkeypatch.setattr(workflow, "composite_workflow_script", unsupported)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *_args: messages.append(_args[-1]))
    button = explorer.details_widget.findChild(QtWidgets.QPushButton, "collection_copy_workflow")
    button.click()
    assert messages == ["This workflow cannot yet be exported:\nOriginal acquisition sources are required."]
