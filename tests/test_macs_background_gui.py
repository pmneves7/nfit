"""Background source rows expose recipes without becoming viewer datasets."""

import pytest

from nfit import NfitProject, NfitProjectExplorer
from tests.test_macs_background_series import _collection, _source


@pytest.fixture
def background_explorer(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    sample = _source(tmp_path / "sample.nxs")
    paths = [_source(tmp_path / "background-410.nxs"), _source(tmp_path / "background-413.nxs")]
    root, spec, _ = _collection([sample], paths)
    _, diff, _ = _collection([sample], paths, stream="diff")
    root.subgroups.append(diff)
    explorer = NfitProjectExplorer(NfitProject([root]))
    explorer._refresh_tree(select_dataset_group=spec)
    yield explorer, root, spec, diff, paths
    explorer.has_unsaved_changes = False
    explorer.window.close()


def _rows(explorer, role, node=None):
    result = []

    def visit(item):
        if explorer._objects_for_item(item)[4] == role and (
            node is None or explorer._dataset_group_for_item(item) is node
        ):
            result.append(item)
        for index in range(item.childCount()):
            visit(item.child(index))

    for index in range(explorer.tree.topLevelItemCount()):
        visit(explorer.tree.topLevelItem(index))
    return result


def test_background_folder_rows_are_visible_without_dataset_or_viewer_entries(background_explorer):
    from nfit.project_viewer_loading import plan_viewer_items

    explorer, root, spec, diff, paths = background_explorer
    for node in (spec, diff):
        folder, = _rows(explorer, "macs_backgrounds", node)
        assert folder.text(0) == "Backgrounds"
        assert folder.toolTip(0)
        assert [folder.child(index).text(0) for index in range(folder.childCount())] == [path.name for path in paths]
        for index in range(folder.childCount()):
            item = folder.child(index)
            assert item.toolTip(0)
            assert explorer._objects_for_item(item)[1] is None
    items = plan_viewer_items(root, use_composite=True, dataset_binnings=lambda _dataset: [])
    assert [item.source_name for item in items] == ["MACS SPEC Composite", "MACS DIFF Composite"]
    assert len(list(root.iter_datasets())) == 2


def test_source_panels_never_open_background_files(background_explorer, monkeypatch):
    import h5py

    explorer, _root, spec, _diff, _paths = background_explorer

    def unexpected_open(*args, **kwargs):
        pytest.fail("Opening a background panel must not read source arrays")

    monkeypatch.setattr(h5py, "File", unexpected_open)
    explorer._refresh_tree(select_dataset_group=spec, refresh_viewers=False)
    explorer.tree.setCurrentItem(_rows(explorer, "macs_backgrounds", spec)[0])
    explorer.tree.setCurrentItem(_rows(explorer, "macs_background_file", spec)[0])


def test_folder_edits_shared_subtraction_controls_and_has_tooltips(background_explorer):
    from PySide6 import QtWidgets

    from nfit.reduction_recipes import effective_reduction_config

    explorer, _root, spec, _diff, _paths = background_explorer
    explorer.tree.setCurrentItem(_rows(explorer, "macs_backgrounds", spec)[0])
    for name in ("macs_background_files", "macs_background_method", "macs_background_scale", "macs_background_delta_e_tolerance_meV", "macs_background_q_bin_width", "macs_background_sources_enabled", "macs_background_copy_recipe"):
        widget = explorer.details_widget.findChild(QtWidgets.QWidget, name)
        assert widget is not None and widget.toolTip()
    method = explorer.details_widget.findChild(QtWidgets.QComboBox, "macs_background_method")
    method.setCurrentIndex(method.findData("isotropic"))
    explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_scale").setValue(1.5)
    config = effective_reduction_config(spec)
    assert config["background_method"] == "isotropic"
    assert config["background_scale"] == 1.5


def test_individual_source_scale_and_inherited_reduction_edits(background_explorer):
    from PySide6 import QtWidgets

    from nfit.macs_background_sources import list_macs_background_sources
    from nfit.reduction_recipes import reduction_settings_schema

    explorer, _root, spec, _diff, _paths = background_explorer
    explorer.tree.setCurrentItem(_rows(explorer, "macs_background_file", spec)[0])
    assert explorer.view_slice_button.isHidden()
    for field in reduction_settings_schema(spec):
        editor = explorer.details_widget.findChild(QtWidgets.QWidget, f"macs_background_run_{field.key}")
        assert (editor is not None) == field.per_run
        if field.per_run:
            inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, f"macs_background_run_{field.key}_inherit")
            assert editor.toolTip() and inherit.toolTip()
            assert inherit.isChecked() and not editor.isEnabled()
    scale = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_source_scale_factor")
    assert scale.toolTip()
    scale.setValue(.25)
    inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_run_monitor_target_inherit")
    inherit.setChecked(False)
    target = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_run_monitor_target")
    target.setValue(1234.)
    first, second = list_macs_background_sources(spec)
    assert first["scale_factor"] == .25
    assert first["reduction_overrides"]["monitor_target"] == 1234.
    assert second["scale_factor"] == 1.
    inherit.setChecked(True)
    first = list_macs_background_sources(spec)[0]
    assert first["reduction_overrides"] == {}
    assert first["scale_factor"] == .25


def test_numeric_inheritance_roundtrip_refreshes_displayed_effective_value(background_explorer):
    from PySide6 import QtWidgets

    from nfit.macs_background_sources import effective_macs_background_config

    explorer, _root, spec, _diff, paths = background_explorer
    explorer.tree.setCurrentItem(_rows(explorer, "macs_background_file", spec)[0])
    inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_run_monitor_target_inherit")
    inherit.setChecked(False)
    target = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_run_monitor_target")
    target.setValue(1234.)
    inherit.setChecked(True)
    target = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_run_monitor_target")
    assert target.value() == effective_macs_background_config(spec, paths[0])["monitor_target"] == 1e6
    assert not target.isEnabled()
    inherit.setChecked(False)
    target = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_run_monitor_target")
    assert target.isEnabled()
    assert target.value() == effective_macs_background_config(spec, paths[0])["monitor_target"] == 1e6


@pytest.mark.parametrize("shared, overridden", [(None, 2.5), (3.5, None)])
def test_automatic_inheritance_roundtrip_refreshes_numeric_and_automatic_state(background_explorer, shared, overridden):
    from PySide6 import QtWidgets

    from nfit.macs_background_sources import effective_macs_background_config
    from nfit.reduction_recipes import set_reduction_settings

    explorer, _root, spec, _diff, paths = background_explorer
    set_reduction_settings(spec, {"incident_energy_override": shared})
    explorer.tree.setCurrentItem(_rows(explorer, "macs_background_file", spec)[0])
    inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_run_incident_energy_override_inherit")
    inherit.setChecked(False)
    automatic = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_run_incident_energy_override_automatic")
    automatic.setChecked(overridden is None)
    numeric = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_run_incident_energy_override")
    if overridden is not None:
        numeric.setValue(overridden)
    assert effective_macs_background_config(spec, paths[0])["incident_energy_override"] == overridden
    inherit.setChecked(True)
    inherit.setChecked(False)
    automatic = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_run_incident_energy_override_automatic")
    numeric = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "macs_background_run_incident_energy_override")
    assert automatic.isChecked() == (shared is None)
    assert numeric.isEnabled() == (shared is not None)
    if shared is not None:
        assert numeric.value() == shared
    assert effective_macs_background_config(spec, paths[0])["incident_energy_override"] == shared


def test_enable_refresh_preserves_source_selection_and_folder_expansion(background_explorer):
    from PySide6 import QtCore, QtWidgets

    from nfit.macs_background_sources import list_macs_background_sources

    explorer, _root, spec, _diff, _paths = background_explorer
    folder, = _rows(explorer, "macs_backgrounds", spec)
    folder.setExpanded(True)
    explorer.tree.setCurrentItem(folder.child(1))
    key = explorer._tree_item_state_key(explorer._current_item())
    enabled = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_source_enabled")
    assert enabled.toolTip()
    enabled.setChecked(False)
    QtWidgets.QApplication.processEvents()
    assert explorer._tree_item_state_key(explorer._current_item()) == key
    assert _rows(explorer, "macs_backgrounds", spec)[0].isExpanded()
    assert [source["enabled"] for source in list_macs_background_sources(spec)] == [True, False]
    explorer.tree.setCurrentItem(_rows(explorer, "macs_backgrounds", spec)[0])
    bulk = explorer.details_widget.findChild(QtWidgets.QCheckBox, "macs_background_sources_enabled")
    assert bulk.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    bulk.setCheckState(QtCore.Qt.CheckState.Checked)
    QtWidgets.QApplication.processEvents()
    assert all(source["enabled"] for source in list_macs_background_sources(spec))


def test_shared_recipe_file_list_refreshes_visible_sources(background_explorer):
    from PySide6 import QtWidgets

    explorer, _root, spec, _diff, paths = background_explorer
    assert explorer._set_reduction_recipe_value(spec, "background_files", [str(paths[1])])
    QtWidgets.QApplication.processEvents()
    folder, = _rows(explorer, "macs_backgrounds", spec)
    assert [folder.child(index).text(0) for index in range(folder.childCount())] == [paths[1].name]


def test_focused_background_gui_does_not_import_project_coordinator():
    import ast
    from pathlib import Path

    from nfit import macs_background_gui

    tree = ast.parse(Path(macs_background_gui.__file__).read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported.intersection({"project_gui", "project_data"})


def test_delete_sources_leaves_editable_empty_folder(background_explorer):
    from PySide6 import QtWidgets

    from nfit.macs_background_sources import list_macs_background_sources

    explorer, root, spec, _diff, paths = background_explorer
    item = _rows(explorer, "macs_background_file", spec)[0]
    assert explorer._can_delete_tree_item(item)
    assert explorer._delete_tree_item(item) == (True, root)
    assert [source["source_file"] for source in list_macs_background_sources(spec)] == [str(paths[1])]
    explorer._refresh_tree(select_dataset_group=spec, refresh_viewers=False)
    folder, = _rows(explorer, "macs_backgrounds", spec)
    assert explorer._delete_tree_item(folder) == (True, root)
    explorer._refresh_tree(select_dataset_group=spec, refresh_viewers=False)
    folder, = _rows(explorer, "macs_backgrounds", spec)
    assert folder.childCount() == 0
    explorer.tree.setCurrentItem(folder)
    assert explorer.details_widget.findChild(QtWidgets.QWidget, "macs_background_files") is not None


def test_legacy_histogram_links_are_distinct_from_source_files(background_explorer):
    from nfit import BackgroundSpec

    explorer, _root, spec, _diff, _paths = background_explorer
    spec.backgrounds.append(BackgroundSpec(name="Histogram correction"))
    explorer._refresh_tree(select_dataset_group=spec, refresh_viewers=False)
    source_folder, = _rows(explorer, "macs_backgrounds", spec)
    histogram_folder, = _rows(explorer, "group_backgrounds", spec)
    assert source_folder.text(0) == "Backgrounds"
    assert histogram_folder.text(0) == "Background histograms"


def test_rejected_source_setting_restores_displayed_value(background_explorer, monkeypatch):
    from PySide6 import QtWidgets

    from nfit import list_macs_background_sources

    explorer, _root, spec, _diff, _paths = background_explorer
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    explorer.tree.setCurrentItem(_rows(explorer, "macs_background_file", spec)[0])
    name = "macs_background_run_monitor_target"
    inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, f"{name}_inherit")
    inherit.setChecked(False)
    editor = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, name)
    previous = editor.value()
    editor.setValue(0.)
    assert len(warnings) == 1
    QtWidgets.QApplication.processEvents()
    restored = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, name)
    assert restored.value() == previous
    assert list_macs_background_sources(spec)[0]["reduction_overrides"]["monitor_target"] == previous
