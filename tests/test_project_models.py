# ruff: noqa: F401, F403, F405
import copy
from types import SimpleNamespace

from nfit import electronic_model_from_component
from tests.project_gui_test_support import *
from tests.project_gui_test_support import (
    _explorer_with_fit_result,
    _grid_mdhisto_data,
    _he_mdhisto_data,
    _points_for_dynamic_writeback,
    _rpa_overlay_group,
    _standard_shortcut_text,
    _tiny_mdhisto_data,
    _tree_items_with_children,
)


def test_tight_binding_model_editor_exposes_scriptable_plot_actions(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Electronic")
    model = create_model_component(group, "bands", type="tight_binding")
    model.config["crystal"]["sites"].append(
        {
            "label": "M1",
            "element": "Fe",
            "position": [0.0, 0.0, 0.0],
            "ion": "",
        }
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_model=model)

    actions = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "tight_binding_actions_group"
    )
    assert actions is not None
    assert explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_crystal_group"
    )
    assert explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_crystal_sites_group"
    )
    assert explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "tight_binding_orbitals_group"
    )
    assert explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "tight_binding_onsite_group"
    )
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QGroupBox, "model_bonds_group"
        )
        is None
    )
    structure_status = explorer.model_parameter_widget.findChild(
        QtWidgets.QLabel, "tight_binding_structure_status"
    )
    assert structure_status is not None and structure_status.toolTip()
    for key in ("bands", "dos", "fermi_surface"):
        plot_button = explorer.model_parameter_widget.findChild(
            QtWidgets.QPushButton, f"model_plot_{key}"
        )
        script_button = explorer.model_parameter_widget.findChild(
            QtWidgets.QPushButton, f"model_plot_script_{key}"
        )
        assert plot_button is not None and plot_button.toolTip()
        assert script_button is not None and script_button.toolTip()
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QPushButton, "tight_binding_import_wannier90"
        ).toolTip()
    )
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QPushButton, "tight_binding_structure_script"
        ).toolTip()
    )
    path_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_path_convention",
    )
    assert path_combo is not None
    assert path_combo.findData("setyawan_curtarolo") >= 0
    assert "ASE" in path_combo.toolTip()
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QPushButton,
            "tight_binding_generate_standard_path",
        )
        is None
    )
    path_combo.setCurrentIndex(path_combo.findData("manual"))
    assert model.config["band_path_convention"] == "manual"
    unit_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_energy_unit"
    )
    chemical_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_chemical_potential_meV"
    )
    band_density_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit,
        "model_config_band_points_per_inv_angstrom",
    )
    dos_min_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_dos_energy_min_meV"
    )
    assert unit_combo is not None and unit_combo.currentData() == "eV"
    assert "canonical" in unit_combo.toolTip()
    assert chemical_editor is not None and chemical_editor.toolTip()
    assert band_density_editor is None
    assert dos_min_editor is not None and dos_min_editor.toolTip()
    backend_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_electronic_backend",
    )
    assert backend_combo is not None and backend_combo.currentData() == "auto"
    assert "CuPy" in backend_combo.toolTip()
    dos_method_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_dos_method",
    )
    assert dos_method_combo is None
    dos_sampling_mode = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_dos_sampling_mode",
    )
    dos_sampling_accuracy = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_dos_sampling_accuracy",
    )
    dos_sampling_mesh = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit,
        "model_config_dos_mesh",
    )
    dos_sampling_button = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton,
        "tight_binding_dos_certify_sampling",
    )
    dos_auto_range = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox,
        "tight_binding_dos_auto_energy_range",
    )
    dos_energy_points = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit,
        "model_config_dos_energy_points",
    )
    dos_min_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_dos_energy_min_meV"
    )
    assert dos_sampling_mode is not None and dos_sampling_mode.toolTip()
    assert dos_sampling_mode.currentData() == "automatic"
    assert dos_sampling_accuracy is not None
    assert dos_sampling_accuracy.isEnabled()
    assert dos_sampling_mesh is not None and not dos_sampling_mesh.isEnabled()
    assert dos_sampling_button is not None and dos_sampling_button.toolTip()
    assert dos_auto_range is not None and dos_auto_range.toolTip()
    assert dos_energy_points is not None and dos_energy_points.toolTip()
    assert dos_min_editor is not None and dos_min_editor.toolTip()
    assert dos_min_editor.isEnabled() is not dos_auto_range.isChecked()
    backend_combo.setCurrentIndex(backend_combo.findData("threaded"))
    assert model.config["electronic_backend"] == "threaded"
    chemical_editor.setText("0.0125")
    chemical_editor.editingFinished.emit()
    assert model.config["chemical_potential_meV"] == pytest.approx(12.5)

    unit_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_energy_unit"
    )
    unit_combo.setCurrentIndex(unit_combo.findData("meV"))
    assert model.config["electronic_energy_unit"] == "meV"
    chemical_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_chemical_potential_meV"
    )
    dos_min_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_dos_energy_min_meV"
    )
    assert float(chemical_editor.text()) == pytest.approx(12.5)
    assert dos_min_editor is not None
    tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget,
        "tight_binding_builder_tabs",
    )
    assert tabs is not None
    assert tabs.toolTip()
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Structure and basis",
        "Hamiltonian",
        "Calculate and inspect",
        "Advanced",
    ]
    summary = explorer.model_parameter_widget.findChild(
        QtWidgets.QLabel,
        "tight_binding_model_summary",
    )
    assert summary is not None and summary.toolTip()
    advanced = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox,
        "model_config_group",
    )
    assert advanced.title() == "Advanced model and execution"
    primitive = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox,
        "tight_binding_use_primitive_cell",
    )
    assert primitive.isChecked() and "fallback" in primitive.toolTip()
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QGroupBox,
            "model_fit_parameters_group",
        )
        is None
    )
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QGroupBox,
            "model_dataset_scope_group",
        )
        is None
    )
    element = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_crystal_site_element_0"
    )
    assert element is not None and element.text() == "Fe" and element.toolTip()
    model.config["crystal"]["provenance"] = {
        "source": "cif",
        "path": "/stale/source.cif",
        "sha256": "0" * 64,
    }
    element.setText("Co")
    element.editingFinished.emit()
    assert model.config["crystal"]["sites"][0]["element"] == "Co"
    assert model.config["crystal"]["provenance"] == {"source": "manual"}
    from nfit import tight_binding_structure_script

    script = tight_binding_structure_script(
        model.config["crystal"],
        model.config["periodic_axes"] or [0, 1, 2],
        electronic_energy_unit=model.config["electronic_energy_unit"],
    )
    assert "crystal_from_cif" not in script
    assert "'element': 'Co'" in script
    assert "set_electronic_energy_unit(model, 'meV')" in script
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_tight_binding_viewer_settings_update_canonical_plot_config(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Electronic")
    model = create_model_component(group, "bands", type="tight_binding")
    scripted = copy.deepcopy(model)
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_model=model)

    assert explorer._apply_tight_binding_plot_settings(
        model,
        "dos",
        {
            "dos_method": "tetrahedron",
            "dos_symmetry": "auto",
            "dos_auto_energy_range": "true",
            "dos_energy_min_meV": "-0.25",
            "dos_energy_max_meV": "0.75",
            "dos_energy_points": "501",
            "dos_broadening_meV": "0.004",
        },
    )
    assert model.config["dos_method"] == "tetrahedron"
    assert model.config["dos_symmetry"] == "full"
    assert model.config["dos_auto_energy_range"] is True
    assert model.config["dos_mesh"] == [40, 40, 40]
    assert model.config["dos_energy_min_meV"] == pytest.approx(-250.0)
    assert model.config["dos_energy_max_meV"] == pytest.approx(750.0)
    assert model.config["dos_broadening_meV"] == pytest.approx(4.0)
    assert model.config["dos_energy_points"] == 501
    from nfit import configure_tight_binding_plot

    configure_tight_binding_plot(
        scripted,
        "dos",
        dos_method="tetrahedron",
        dos_symmetry="full",
        dos_auto_energy_range=True,
        dos_energy_min_meV=-250.0,
        dos_energy_max_meV=750.0,
        dos_broadening_meV=4.0,
        dos_energy_points=501,
    )
    for name in (
        "dos_method",
        "dos_symmetry",
        "dos_auto_energy_range",
        "dos_energy_min_meV",
        "dos_energy_max_meV",
        "dos_broadening_meV",
        "dos_energy_points",
    ):
        assert model.config[name] == scripted.config[name]
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_fermi_surface_spacing_policy_updates_model_configuration(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Electronic")
    model = create_model_component(group, "bands", type="tight_binding")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_model=model)

    assert explorer._apply_tight_binding_plot_settings(
        model,
        "fermi_surface",
        {
            "fermi_mesh_mode": "spacing",
            "fermi_spacing_inv_angstrom": "0.02",
            "fermi_mesh": "[64, 64, 64]",
            "fermi_energy_meV": "0.01",
        },
    )
    assert model.config["fermi_mesh_mode"] == "spacing"
    assert model.config["fermi_spacing_inv_angstrom"] == pytest.approx(0.02)
    assert model.config["fermi_energy_meV"] == pytest.approx(10.0)
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_model_plot_recalculation_updates_existing_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from matplotlib.figure import Figure

    model = ModelComponentSpec(name="bands", type="tight_binding", config={})
    explorer = NfitProjectExplorer(NfitProject())
    calculations = []
    renders = []

    def calculate(_model):
        calculations.append(_model)
        return SimpleNamespace(dimension=1)

    def render(result):
        renders.append(result)
        figure = Figure()
        figure.add_subplot(111)
        return figure, figure.axes[0]

    plot = SimpleNamespace(
        key="bands",
        calculate=calculate,
        context_calculate=None,
        render=render,
    )
    monkeypatch.setattr(
        "nfit.project_gui.model_plot_definitions",
        lambda _model_type: (plot,),
    )
    monkeypatch.setattr(
        explorer,
        "_apply_tight_binding_plot_settings",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        explorer,
        "_tight_binding_plot_settings",
        lambda *_args, **_kwargs: {"band_points_per_inv_angstrom": "120"},
    )

    class Window:
        def __init__(self):
            self.replacements = []
            self.settings = []
            self.close_count = 0

        def _nfit_replace_figure(self, figure):
            self.replacements.append(figure)

        def _nfit_update_calculation_settings(self, settings):
            self.settings.append(settings)

        def close(self):
            self.close_count += 1

    window = Window()
    callbacks = []

    def show_figure(_figure, **kwargs):
        callbacks.append(kwargs["on_apply_settings"])
        return window

    monkeypatch.setattr(
        "nfit.qt_electronic_viewer.show_electronic_figure",
        show_figure,
    )

    assert explorer._open_model_plot(model, "bands")
    assert len(calculations) == 1
    callbacks[0]({"band_points_per_inv_angstrom": "120"})

    assert len(calculations) == 2
    assert len(renders) == 2
    assert len(window.replacements) == 1
    assert window.settings == [{"band_points_per_inv_angstrom": "120"}]
    assert window.close_count == 0
    assert explorer._plot_windows[f"model:{id(model)}:bands"] is window
    explorer.window.close()


def test_lindhard_editor_selects_a_sibling_electronic_model(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Electronic")
    source = create_model_component(group, "bands", type="tight_binding")
    response = create_model_component(group, "response", type="lindhard")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_model=response)

    selector = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "lindhard_electronic_component",
    )
    assert selector is not None and selector.toolTip()
    tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget,
        "lindhard_builder_tabs",
    )
    assert tabs is not None and tabs.toolTip()
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Response",
        "Sampling",
        "Experimental coupling",
        "Calculate and inspect",
        "Advanced",
    ]
    mesh = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit,
        "model_config_response_mesh",
    )
    q_accuracy = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "lindhard_q_accuracy",
    )
    sampling_mode = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "lindhard_sampling_mode",
    )
    sampling_accuracy = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "lindhard_sampling_accuracy",
    )
    sampling_button = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton,
        "lindhard_certify_sampling",
    )
    sampling_status = explorer.model_parameter_widget.findChild(
        QtWidgets.QLabel,
        "lindhard_convergence_status",
    )
    formula_mode = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "lindhard_formula_units_mode",
    )
    assert mesh is not None and mesh.toolTip()
    assert not mesh.isEnabled()
    assert sampling_mode is not None and sampling_mode.currentData() == "automatic"
    assert sampling_accuracy is not None and sampling_accuracy.currentData() == "standard"
    assert sampling_button is not None and sampling_button.toolTip()
    assert sampling_status is not None and sampling_status.toolTip()
    assert q_accuracy is not None and q_accuracy.currentData() == "exact"
    assert formula_mode is not None and formula_mode.currentData() == "auto"
    assert selector.findData(source.name) >= 0
    selector.setCurrentIndex(selector.findData(source.name))
    assert response.config["electronic_component"] == source.name

    plot_button = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton,
        "model_plot_complex_energy_scan",
    )
    script_button = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton,
        "model_plot_script_complex_energy_scan",
    )
    assert plot_button is not None and plot_button.toolTip()
    assert script_button is not None and script_button.toolTip()
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_tight_binding_orbital_onsite_and_geometry_gui_are_scriptable(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Electronic")
    model = create_model_component(group, "bands", type="tight_binding")
    model.config["crystal"]["sites"] = [
        {
            "label": "M1",
            "element": "Fe",
            "position": [0.0, 0.0, 0.0],
            "ion": "",
        },
        {
            "label": "X1",
            "element": "O",
            "position": [0.5, 0.5, 0.5],
            "ion": "",
        },
    ]
    model.config["crystal"]["spacegroup"] = "P m -3 m"
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_model=model)

    site = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_orbital_site"
    )
    preset = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_orbital_preset"
    )
    add = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_orbital_add"
    )
    point_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QLabel, "tight_binding_orbital_site_symmetry"
    )
    use_site_symmetry = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "tight_binding_orbital_use_site_symmetry"
    )
    submanifold = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_orbital_submanifold"
    )
    assert site.currentData() == "M1"
    assert point_group.text() == "Site symmetry: m-3m"
    assert use_site_symmetry.toolTip() and submanifold.toolTip()
    preset.setCurrentIndex(preset.findData("d"))
    assert submanifold.count() == 2
    use_site_symmetry.setChecked(True)
    add.click()

    assert len(model.config["orbital_manifolds"]) == 1
    assert len(model.config["onsite_terms"]) == 1
    assert model.config["orbital_manifolds"][0]["site_point_group"] == "m-3m"
    assert len(model.config["orbital_manifolds"][0]["orbitals"]) in {2, 3}
    assert model.config["model_stale"] is True
    assert model.config["model_digest"] == ""
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QPushButton,
            "tight_binding_onsite_generate",
        )
        is None
    )
    frame = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "tight_binding_manifold_frame_0"
    )
    value = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "tight_binding_onsite_value_0"
    )
    fit = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "tight_binding_onsite_fit_0"
    )
    assert frame is not None and "orthonormal" in frame.toolTip()
    assert value is not None and "many-body" in value.toolTip()
    assert fit is not None and "compatible electronic-response" in fit.toolTip()
    value.setText("0.025")
    value.editingFinished.emit()
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QLineEdit,
            "tight_binding_onsite_value_0",
        )
        is value
    )
    assert model.config["onsite_terms"][0]["value_meV"] == pytest.approx(25.0)
    onsite_identifier = model.config["onsite_terms"][0]["identifier"]
    assert model.parameters[onsite_identifier] == pytest.approx(25.0)
    onsite_sharing = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_onsite_0_sharing"
    )
    assert onsite_sharing is not None and onsite_sharing.toolTip()
    assert not onsite_sharing.isHidden()
    onsite_sharing.setCurrentIndex(onsite_sharing.findData("per_dataset"))
    assert model.sharing[onsite_identifier]["mode"] == "per_dataset"

    cutoff = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "tight_binding_hopping_cutoff"
    )
    parameterization = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_hopping_parameterization",
    )
    generate = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_hopping_generate"
    )
    assert parameterization.currentData() == "slater_koster"
    assert "General symmetry matrices" == parameterization.itemText(1)
    assert parameterization.toolTip()
    cutoff.setText("5.1")
    generate.click()
    assert model.config["spatial_orbits"]
    assert model.config["hopping_candidates"]
    assert model.config["hopping_terms"] == []
    suggestions = explorer.model_parameter_widget.findChild(
        QtWidgets.QTableWidget, "tight_binding_hopping_suggestions"
    )
    add_hopping = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_hopping_add_selected"
    )
    assert suggestions is not None and suggestions.rowCount() > 0
    assert suggestions.horizontalHeaderItem(4).text() == "From orbital(s)"
    assert "M1_d_subspace" in suggestions.item(0, 4).text()
    assert "←" in suggestions.item(0, 0).text()
    suggestions.selectRow(0)
    add_hopping.click()
    assert len(model.config["hopping_terms"]) == 1
    hopping_value = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "tight_binding_hopping_value_0"
    )
    assert hopping_value is not None and "canonical meV" in hopping_value.toolTip()
    hopping_sharing = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "tight_binding_hopping_0_sharing"
    )
    assert hopping_sharing is not None and hopping_sharing.toolTip()
    assert not hopping_sharing.isHidden()
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QCheckBox,
            "tight_binding_hopping_show_details",
        )
        is None
    )
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QCheckBox,
            "tight_binding_onsite_show_details",
        )
        is None
    )
    remove_hopping = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_hopping_remove_0"
    )
    assert remove_hopping is not None and remove_hopping.toolTip()

    config_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_config_group"
    )
    electronic_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "tight_binding_actions_group"
    )
    tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget,
        "tight_binding_builder_tabs",
    )
    assert tabs.indexOf(config_group.parentWidget()) == 3
    assert tabs.indexOf(electronic_group.parentWidget()) == 2
    matrix_action = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_matrix_inspector"
    )
    fermi_action = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "model_plot_fermi_surface"
    )
    action_layout = electronic_group.layout()
    matrix_row = action_layout.getItemPosition(action_layout.indexOf(matrix_action))[0]
    fermi_row = action_layout.getItemPosition(action_layout.indexOf(fermi_action))[0]
    assert matrix_row > fermi_row

    copied = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_structure_script"
    )
    copied.click()
    copied_script = QtWidgets.QApplication.clipboard().text()
    assert "configure_tight_binding_builder" in copied_script
    assert "set_tight_binding_parameter_state" in copied_script
    assert "hopping_parameterization" in copied_script

    opened = []
    monkeypatch.setattr(
        "nfit.qt_model_geometry_viewer.open_model_geometry_viewer",
        lambda component, parent=None: opened.append((component, parent)) or object(),
    )
    viewer = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "model_geometry_viewer"
    )
    script = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "model_geometry_script"
    )
    assert viewer.toolTip() and script.toolTip()
    viewer.click()
    assert opened and opened[0][0] is model
    script.click()
    assert "model_geometry_scene" in QtWidgets.QApplication.clipboard().text()

    zones = []
    monkeypatch.setattr(
        "nfit.qt_brillouin_zone_viewer.show_brillouin_zone_scene",
        lambda scene, parent=None: zones.append((scene, parent)) or object(),
    )
    zone = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_brillouin_zone"
    )
    zone_script = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "tight_binding_brillouin_zone_script"
    )
    assert zone.toolTip() and zone_script.toolTip()
    primitive = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox,
        "tight_binding_use_primitive_cell",
    )
    path_convention = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_path_convention",
    )
    assert primitive.isChecked() and primitive.toolTip()
    assert path_convention.currentData() == "hinuma"
    assert path_convention.toolTip()
    zone.click()
    assert zones and zones[0][0].path_nodes
    zone_script.click()
    assert "build_brillouin_zone_scene" in QtWidgets.QApplication.clipboard().text()

    spin_treatment = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "tight_binding_spin_treatment",
    )
    soc_enabled = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox,
        "tight_binding_soc_enabled_0",
    )
    assert spin_treatment.currentData() == "auto"
    assert spin_treatment.toolTip()
    assert soc_enabled.isEnabled() and soc_enabled.toolTip()
    soc_enabled.click()
    assert len(model.config["soc_terms"]) == 1
    assert electronic_model_from_component(model).n_basis == (
        2 * len(model.config["orbital_manifolds"][0]["orbitals"])
    )

    matrix_windows = []
    monkeypatch.setattr(
        "nfit.qt_electronic_matrix_viewer.show_electronic_matrix_catalog",
        lambda catalog, parent=None: matrix_windows.append((catalog, parent))
        or object(),
    )
    matrix_button = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton,
        "tight_binding_matrix_inspector",
    )
    matrix_script = explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton,
        "tight_binding_matrix_script",
    )
    assert matrix_button.toolTip() and matrix_script.toolTip()
    matrix_button.click()
    assert matrix_windows and matrix_windows[0][0]
    matrix_script.click()
    assert (
        "show_electronic_matrix_catalog"
        in QtWidgets.QApplication.clipboard().text()
    )
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_tight_binding_cif_structure_project_and_script_round_trip(tmp_path):
    from nfit import (
        import_cif_into_model,
        tight_binding_structure_script,
    )

    cif = tmp_path / "structure.cif"
    cif.write_text(
        """
data_structure
_cell_length_a 4.0
_cell_length_b 5.0
_cell_length_c 6.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
M1 Fe 0.125 0.250 0.375
"""
    )
    group = DataGroup("Electronic")
    model = create_model_component(group, "bands", type="tight_binding")
    imported = import_cif_into_model(model, str(cif), group=group)

    assert model.config["crystal"] == imported
    assert model.config["periodic_axes"] == [0, 1, 2]
    assert model.config["band_path_convention"] == "hinuma"
    assert model.config["band_path_metadata"]["provider"] == "seekpath"
    assert model.config["band_path_metadata"]["convention"] == "HPKOT"
    assert model.config["band_path_metadata"]["provider_version"]
    assert imported["sites"][0]["element"] == "Fe"
    assert group.metadata["crystal"] == imported

    path = tmp_path / "structure.nfit"
    save_project(NfitProject([group]), path)
    restored = load_project(path)
    restored_model = restored.data_groups[0].models["bands"]
    assert restored_model.config["crystal"] == imported
    assert restored_model.config["periodic_axes"] == [0, 1, 2]
    assert (
        restored_model.config["band_path_metadata"]
        == model.config["band_path_metadata"]
    )

    script = tight_binding_structure_script(
        restored_model.config["crystal"],
        restored_model.config["periodic_axes"],
        group_name="Electronic",
        model_name="bands",
        electronic_energy_unit=restored_model.config["electronic_energy_unit"],
    )
    namespace = {}
    exec(compile(script, "<tight-binding-structure>", "exec"), namespace)
    scripted = namespace["model"]
    assert scripted.config["crystal"] == imported
    assert scripted.config["periodic_axes"] == [0, 1, 2]
    assert scripted.config["electronic_energy_unit"] == "eV"


def test_heisenberg_rpa_editor_generates_orbits_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("gemmi")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    category = explorer.model_category_combo
    category.setCurrentIndex(category.findData("spin_fluctuation"))
    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("heisenberg_rpa"))
    assert model.type == "heisenberg_rpa"

    crystal_box = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_crystal_group"
    )
    bonds_box = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_bonds_group")
    sites_box = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_crystal_sites_group"
    )
    assert crystal_box is not None and bonds_box is not None and sites_box is not None
    assert explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "model_geometry_viewer"
    ).toolTip()
    assert explorer.model_parameter_widget.findChild(
        QtWidgets.QPushButton, "model_geometry_script"
    ).toolTip()

    # configure an FCC crystal through the handlers
    for name in ("a", "b", "c"):
        explorer._set_model_crystal_lattice(name, "4.0")
    explorer._set_model_crystal_spacegroup("F m -3 m")
    explorer._add_model_crystal_site()
    explorer._set_model_crystal_site(0, "label", "Ni1")
    explorer._set_model_crystal_site(0, "ion", "Ni2")
    explorer._set_model_site_magnetic(0, True)

    cutoff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_bonds_cutoff"
    )
    assert cutoff_editor is not None
    cutoff_editor.setText("3.0")
    explorer._generate_selected_model_bond_orbits()

    assert [orbit["label"] for orbit in model.config["orbits"]] == ["J1"]
    assert len(model.config["site_positions"]) == 4
    assert model.parameters["J1"] == 0.0
    assert model.fit_parameters["J1"] is False

    j1_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_parameter_min_J1"
    )
    assert j1_editor is not None  # J1 row exists in the fit-parameter grid

    orbit_table = explorer.model_parameter_widget.findChild(
        QtWidgets.QTableWidget, "model_bonds_table"
    )
    assert orbit_table is not None
    assert orbit_table.rowCount() == 1
    assert orbit_table.item(0, 0).text() == "J1"
    assert orbit_table.item(0, 2).text() == "24"

    cutoff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_bonds_cutoff"
    )
    assert cutoff_editor is not None
    cutoff_editor.setText("4.1")
    explorer._generate_selected_model_bond_orbits()

    assert model.config["bond_cutoff_angstrom"] == 4.1
    assert [orbit["label"] for orbit in model.config["orbits"]] == ["J1", "J2"]

    explorer._set_model_parameter("J1", "0.15")
    explorer._set_model_fit_parameter("J1", True)

    path = tmp_path / "project.json"
    save_project(NfitProject([group]), path)
    loaded = load_project(path)
    loaded_model = next(iter(loaded.data_groups[0].models.values()))
    assert [orbit["label"] for orbit in loaded_model.config["orbits"]] == ["J1", "J2"]
    assert loaded_model.parameters["J1"] == 0.15
    assert loaded_model.fit_parameters["J1"] is True
    assert loaded_model.config["site_positions"] == model.config["site_positions"]


def test_heisenberg_rpa_editor_toggles_tensor_interactions(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("gemmi")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    category = explorer.model_category_combo
    category.setCurrentIndex(category.findData("spin_fluctuation"))
    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("heisenberg_rpa"))

    for name in ("a", "b", "c"):
        explorer._set_model_crystal_lattice(name, "4.0")
    explorer._set_model_crystal_spacegroup("F m -3 m")
    explorer._add_model_crystal_site()
    explorer._set_model_crystal_site(0, "label", "Ni1")
    explorer._set_model_crystal_site(0, "ion", "Ni2")
    explorer._set_model_site_magnetic(0, True)

    cutoff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_bonds_cutoff"
    )
    cutoff_editor.setText("3.0")
    explorer._generate_selected_model_bond_orbits()

    interactions = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_interactions_group"
    )
    assert interactions is not None

    # Every interaction toggle carries a tooltip (audit convention).
    for kind in ("anisotropy", "sia", "dipole", "zeeman"):
        box = explorer.model_parameter_widget.findChild(
            QtWidgets.QCheckBox, f"model_interaction_{kind}"
        )
        assert box is not None and box.toolTip().strip()

    # Dipole seeds the physical strength as a fit parameter.
    dipole_box = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_interaction_dipole"
    )
    dipole_box.setChecked(True)
    assert model.config["dipole"]["enabled"] is True
    assert "D_dip" in model.parameters
    assert model.parameters["D_dip"] > 0.0

    # Zeeman exposes g_factor and the transverse ratios.
    zeeman_box = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_interaction_zeeman"
    )
    zeeman_box.setChecked(True)
    assert model.config["zeeman"]["enabled"] is True
    for name in ("g_factor", "chi_perp_ratio", "gamma_perp_ratio"):
        assert name in model.parameters

    # Anisotropy snapshots the symmetry-allowed exchange tensors.
    aniso_box = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_interaction_anisotropy"
    )
    aniso_box.setChecked(True)
    assert model.config.get("anisotropy")

    path = tmp_path / "project.json"
    save_project(NfitProject([group]), path)
    loaded = load_project(path)
    loaded_model = next(iter(loaded.data_groups[0].models.values()))
    assert loaded_model.config["dipole"]["enabled"] is True
    assert loaded_model.config["zeeman"]["enabled"] is True
    assert loaded_model.config.get("anisotropy")
    assert loaded_model.parameters["D_dip"] == model.parameters["D_dip"]

    # Disabling clears the config section and its dynamic parameters.
    dipole_box = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_interaction_dipole"
    )
    dipole_box.setChecked(False)
    assert model.config["dipole"]["enabled"] is False
    assert "D_dip" not in model.parameters


def test_heisenberg_rpa_editor_closure_controls(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("gemmi")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()
    category = explorer.model_category_combo
    category.setCurrentIndex(category.findData("spin_fluctuation"))
    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("heisenberg_rpa"))

    tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget, "heisenberg_builder_tabs"
    )
    assert tabs is not None and tabs.toolTip().strip()
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Structure and exchange",
        "Response and fit",
        "Advanced",
    ]
    summary = explorer.model_parameter_widget.findChild(
        QtWidgets.QLabel, "heisenberg_model_summary"
    )
    assert summary is not None and "bare RPA" in summary.text()
    powder = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_powder_orientations"
    )
    assert powder is not None and powder.toolTip().strip()
    powder.setText("72")
    powder.editingFinished.emit()
    assert model.config["powder_orientations"] == 72
    numerical_status = explorer.model_parameter_widget.findChild(
        QtWidgets.QLabel, "heisenberg_numerical_status"
    )
    assert numerical_status is not None
    assert "does not use" in numerical_status.text()
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_closure_bz_grid"
        )
        is None
    )

    for name in ("a", "b", "c"):
        explorer._set_model_crystal_lattice(name, "4.0")
    explorer._set_model_crystal_spacegroup("F m -3 m")
    explorer._add_model_crystal_site()
    explorer._set_model_crystal_site(0, "label", "Ni1")
    explorer._set_model_crystal_site(0, "ion", "Ni2")
    explorer._set_model_site_magnetic(0, True)
    explorer.model_parameter_widget.findChild(QtWidgets.QLineEdit, "model_bonds_cutoff").setText(
        "3.0"
    )
    explorer._generate_selected_model_bond_orbits()

    closure_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_closure_group"
    )
    assert closure_group is not None
    mode_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_closure_mode"
    )
    assert mode_combo is not None and mode_combo.toolTip().strip()
    # No closure by default: config has none, no closure params.
    assert "closure" not in model.config
    assert "m2_total" not in model.parameters

    # Select the fitted-target Onsager closure.
    mode_combo.setCurrentIndex(mode_combo.findData("onsager"))
    assert model.config["closure"]["mode"] == "onsager"
    grid = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_closure_bz_grid"
    )
    assert grid is not None and grid.toolTip().strip()
    grid.setText("20")
    grid.editingFinished.emit()
    assert model.config["closure"]["bz_grid"] == 20
    moment_check = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_closure_moment_mode"
    )
    assert moment_check is not None and moment_check.toolTip().strip()
    moment_check.setChecked(True)
    assert model.config["closure"]["moment_mode"] == "fitted"
    assert "m2_total" in model.parameters
    fit_check = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_parameter_fit_m2_total"
    )
    assert fit_check is not None and fit_check.toolTip().strip()
    fit_check.setChecked(True)
    assert model.fit_parameters["m2_total"] is True

    cutoff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_closure_energy_cutoff_mev"
    )
    assert cutoff_editor is not None and cutoff_editor.toolTip().strip()
    cutoff_editor.setText("50.0")
    cutoff_editor.editingFinished.emit()
    assert model.config["closure"]["energy_cutoff_mev"] == 50.0

    # Switch to SCR: exposes mode_coupling_u, drops m2_total.
    mode_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_closure_mode"
    )
    mode_combo.setCurrentIndex(mode_combo.findData("scr"))
    assert "mode_coupling_u" in model.parameters
    assert "m2_total" not in model.parameters
    scr_fit_check = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_parameter_fit_mode_coupling_u"
    )
    assert scr_fit_check is not None and scr_fit_check.toolTip().strip()
    scr_fit_check.setChecked(True)
    assert model.fit_parameters["mode_coupling_u"] is True

    # TAC exposes its conserved total amplitude through the same fit controls.
    mode_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_closure_mode"
    )
    mode_combo.setCurrentIndex(mode_combo.findData("tac"))
    assert "mode_coupling_u" not in model.parameters
    assert "total_amplitude" in model.parameters
    tac_fit_check = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_parameter_fit_total_amplitude"
    )
    assert tac_fit_check is not None and tac_fit_check.toolTip().strip()
    tac_fit_check.setChecked(True)
    assert model.fit_parameters["total_amplitude"] is True
    chi0_fit_check = explorer.model_parameter_widget.findChild(
        QtWidgets.QCheckBox, "model_parameter_fit_chi0"
    )
    assert chi0_fit_check is not None
    assert not chi0_fit_check.isEnabled()
    assert "root-search seed" in chi0_fit_check.toolTip()

    # Round-trips through save/load.
    path = tmp_path / "closure.json"
    save_project(NfitProject([group]), path)
    loaded_model = next(iter(load_project(path).data_groups[0].models.values()))
    assert loaded_model.config["closure"]["mode"] == "tac"
    assert loaded_model.config["closure"]["energy_cutoff_mev"] == 50.0
    assert loaded_model.config["closure"]["bz_grid"] == 20
    assert loaded_model.config["powder_orientations"] == 72
    assert "total_amplitude" in loaded_model.parameters

    # Back to none: config section and closure params are cleared.
    mode_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_closure_mode"
    )
    mode_combo.setCurrentIndex(mode_combo.findData("none"))
    assert "closure" not in model.config
    assert "total_amplitude" not in model.parameters


def test_fit_result_context_menu_offers_report_export(monkeypatch):
    explorer, result_item = _explorer_with_fit_result(monkeypatch)
    names = explorer.context_menu_action_names(result_item)
    assert "Export report..." in names
    # Not offered for non-fit items.
    assert "Export report..." not in explorer.context_menu_action_names(
        explorer.tree.topLevelItem(0)
    )
    # The fit details pane carries the export button, tooltipped.
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    button = explorer.window.findChild(QtWidgets.QPushButton, "fit_export_report_button")
    assert button is not None and button.toolTip().strip()


def test_export_fit_report_writes_tex(monkeypatch, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    explorer, _item = _explorer_with_fit_result(monkeypatch)
    target = tmp_path / "report.tex"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "LaTeX source (*.tex)")),
    )
    assert explorer.export_fit_report_for_selection() is True
    tex = target.read_text()
    assert tex.startswith("\\documentclass")
    assert "Fit summary" in tex


def test_export_fit_report_pdf_falls_back_to_tex_on_compile_error(monkeypatch, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import nfit.report as report_module

    explorer, _item = _explorer_with_fit_result(monkeypatch)
    target = tmp_path / "report.pdf"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "PDF report (*.pdf)")),
    )

    def boom(_tex, _path):
        raise report_module.LatexCompileError("no engine", engine=None, log_tail="")

    monkeypatch.setattr(report_module, "compile_latex_pdf", boom)
    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda _parent, _title, message, *a, **k: warnings.append(message)),
    )
    assert explorer.export_fit_report_for_selection() is False
    assert warnings and "no engine" in warnings[0]
    # The LaTeX source was written next to the requested PDF.
    fallback = target.with_suffix(".tex")
    assert fallback.exists()
    assert fallback.read_text().startswith("\\documentclass")


def test_export_fit_report_pdf_opens_viewer_on_success(monkeypatch, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtGui = pytest.importorskip("PySide6.QtGui")
    import nfit.report as report_module

    explorer, _item = _explorer_with_fit_result(monkeypatch)
    target = tmp_path / "report.pdf"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "PDF report (*.pdf)")),
    )
    monkeypatch.setattr(
        report_module,
        "compile_latex_pdf",
        lambda _tex, path: Path(path).write_bytes(b"%PDF-1.4 fake"),
    )
    opened: list[str] = []
    monkeypatch.setattr(
        QtGui.QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url.toLocalFile()) or True),
    )
    assert explorer.export_fit_report_for_selection() is True
    assert opened == [str(target)]


def test_fit_details_show_diagnostics_table(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.pipeline import FitTimelineEntry

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    entry = FitTimelineEntry(
        name="fit",
        kind="result",
        snapshot={},
        created_at="now",
        goodness={"status": "converged"},
        metadata={
            "diagnostics": {
                "scan": {
                    "temperature": 10.0,
                    "mu_eff_sq": 1.23,
                    "chi_static_q0": 0.7,
                    "chi_static_qpeak": 0.9,
                    "chi0_gamma0": 0.6,
                    "stability_margin": 0.3,
                    "stability_ratio": 0.7,
                    "lambda_shift": 0.05,
                    "chi0_eff": 0.3,
                }
            }
        },
    )
    box = explorer._fit_diagnostics_group_box(entry)
    assert box is not None
    table = box.findChild(QtWidgets.QTableWidget, "fit_diagnostics_table")
    assert table is not None
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "scan"
    # A header tooltip exists on every metric column (audit convention).
    for column in range(1, table.columnCount()):
        assert table.horizontalHeaderItem(column).toolTip().strip()

    # No diagnostics metadata -> no table.
    empty = FitTimelineEntry(name="f", kind="result", snapshot={}, created_at="now", metadata={})
    assert explorer._fit_diagnostics_group_box(empty) is None


def test_heisenberg_rpa_editor_scrolls_while_fit_parameters_grow(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    category = explorer.model_category_combo
    category.setCurrentIndex(category.findData("spin_fluctuation"))
    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("heisenberg_rpa"))
    model.config["orbits"] = [{"label": f"J{index}", "bonds": []} for index in range(1, 25)]
    project_gui.reconcile_model_orbit_parameters(model)
    explorer._rebuild_model_parameter_editor(model)

    fit_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_fit_parameters_group"
    )
    model_scroll = explorer.window.findChild(QtWidgets.QScrollArea, "model_parameter_scroll")
    orbit_table = explorer.model_parameter_widget.findChild(
        QtWidgets.QTableWidget, "model_bonds_table"
    )
    assert fit_group is not None
    assert model_scroll is not None
    assert orbit_table is not None
    assert not model_scroll.isHidden()
    assert model_scroll.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert model_scroll.widget().minimumSizeHint().height() > model_scroll.minimumHeight()
    assert fit_group.findChild(QtWidgets.QScrollArea, "model_fit_parameters_scroll") is None
    assert fit_group.minimumSizeHint().height() > model_scroll.minimumHeight()
    assert orbit_table.maximumHeight() == orbit_table.minimumHeight()
    assert orbit_table.maximumHeight() >= orbit_table.horizontalHeader().height() + 4 * 24
    assert (
        explorer.model_parameter_widget.findChild(QtWidgets.QLineEdit, "model_parameter_min_J24")
        is not None
    )


def test_import_cif_into_model_populates_config_and_group(tmp_path):
    pytest.importorskip("gemmi")
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import generate_model_bond_orbits, import_cif_into_model

    cif = tmp_path / "fcc.cif"
    cif.write_text(
        """
data_fcc
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
"""
    )
    model = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.1, "gamma0": 5.0},
    )
    group = DataGroup("Datagroup1")
    imported = import_cif_into_model(model, str(cif), group=group)
    assert model.config["crystal"]["lattice"]["a"] == pytest.approx(4.0)
    assert group.lattice_parameters["a"] == pytest.approx(4.0)
    assert group.metadata["crystal"] == imported

    model.config["magnetic_sites"] = ["Ni1"]
    model.config["bond_cutoff_angstrom"] = 3.0
    labels = generate_model_bond_orbits(model)
    assert labels == ["J1"]
    assert model.parameters["J1"] == 0.0


def test_workspace_spacegroup_editor_updates_bragg_analysis_context(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Datagroup1", spacegroup="F d -3 m:2")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

    editor = explorer.window.findChild(QtWidgets.QLineEdit, "group_spacegroup_editor")
    assert editor is not None
    assert editor.text() == "F d -3 m:2"
    assert editor.toolTip()
    editor.setText("227")
    editor.editingFinished.emit()
    assert group.spacegroup == "227"


def test_generate_model_bond_orbits_requires_magnetic_site():
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import generate_model_bond_orbits, model_crystal_config

    model = ModelComponentSpec(name="rpa", type="heisenberg_rpa", parameters={})
    model_crystal_config(model)["sites"].append(
        {"label": "X1", "position": [0.0, 0.0, 0.0], "ion": ""}
    )
    with pytest.raises(ValueError, match="magnetic site"):
        generate_model_bond_orbits(model)


def test_reconcile_model_orbit_parameters_keeps_and_drops():
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import reconcile_model_orbit_parameters

    model = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.1, "gamma0": 5.0, "J1": 0.2, "J9": 0.7},
        fit_parameters={"J1": True, "J9": True},
        limits={"J9": [0.0, 1.0]},
    )
    model.config["orbits"] = [
        {"label": "J1", "bonds": []},
        {"label": "J2", "bonds": []},
    ]
    reconcile_model_orbit_parameters(model)
    assert model.parameters["J1"] == 0.2
    assert model.parameters["J2"] == 0.0
    assert "scale" not in model.parameters
    assert "J9" not in model.parameters
    assert "J9" not in model.fit_parameters
    assert "J9" not in model.limits
    assert model.fit_parameters["J1"] is True


def test_heisenberg_rpa_overlay_evaluates_powder_inelastic_grid():
    q_edges = np.linspace(0.2, 1.0, 5)
    energy_edges = np.linspace(0.5, 3.5, 4)
    signal = np.ones((4, 3), dtype=float)
    data = MDHistoData(
        axes=(
            MDHistoAxis("Q", q_edges, "1/angstrom", "q_modulus"),
            MDHistoAxis("DeltaE", energy_edges, "meV", "energy"),
        ),
        signal=signal,
        errors=np.full_like(signal, 0.1),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )
    dataset = DatasetEntry(
        "powder",
        data,
        kind="mdhisto",
        data_type="powder_inelastic",
        parameters={
            "temperature": 10.0,
            project_gui.SPECTRAL_CHANNEL_CONFIG_KEY: (
                project_gui.default_spectral_channel_config()
            ),
        },
    )
    model = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={
            "scale": 1.0,
            "chi0": 0.3,
            "gamma0": 2.0,
            "J1": 0.05,
        },
        fit_parameters={},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}],
                }
            ],
            "powder_orientations": 12,
            "crystal": {
                "lattice": {
                    "a": 8.0,
                    "b": 8.0,
                    "c": 8.0,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                }
            },
        },
    )
    group = DataGroup(
        "Workspace1",
        datasets=[dataset],
        lattice_parameters={
            "a": 8.0,
            "b": 8.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
    )
    group.models[model.name] = model

    channels = project_gui.current_model_channels(group)

    assert "powder" in channels
    fit = np.asarray(channels["powder"]["fit"], dtype=float)
    assert fit.shape == signal.shape
    assert np.all(np.isfinite(fit))


def test_overlay_cache_reuses_compiled_problem_across_parameter_edits():
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, _dataset, model = _rpa_overlay_group()

    first = project_gui.current_model_channels(group)
    assert "scan" in first
    cached = project_gui._MODEL_OVERLAY_CACHE[id(group)]
    compiled_first = cached["compiled"]

    # A parameter-value edit must reuse the cached bundles/compiled problem...
    model.parameters["J1"] = 0.25
    second = project_gui.current_model_channels(group)
    assert project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"] is compiled_first
    # ...while still reflecting the new value in the overlay.
    a = np.asarray(first["scan"]["fit"], dtype=float)
    b = np.asarray(second["scan"]["fit"], dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.any()
    assert np.nanmax(np.abs(a[finite] - b[finite])) > 0.0


def test_overlay_cache_invalidates_on_structural_change():
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, _dataset, model = _rpa_overlay_group()
    project_gui.current_model_channels(group)
    compiled_first = project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"]

    # Changing the bond network (structure) must rebuild, not reuse.
    model.config["orbits"] = [
        {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
        {"label": "J2", "bonds": [{"site_i": 0, "site_j": 0, "offset": [0, 0, 1]}]},
    ]
    model.parameters["J2"] = 0.05
    project_gui.current_model_channels(group)
    assert project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"] is not compiled_first


def test_overlay_cache_invalidates_when_rebin_basis_changes():
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, dataset, _model = _rpa_overlay_group()
    project_gui.current_model_channels(group)
    compiled_first = project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"]

    dataset.parameters[project_gui.DATASET_REBIN_KEY] = {
        "enabled": False,
        "axes": [{"vector": [1.0, 1.0, 0.0, 0.0]}],
    }
    dataset.data.metadata["rebin"] = {"vectors": [[1.0, 1.0, 0.0, 0.0]]}

    project_gui.current_model_channels(group)
    assert project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"] is not compiled_first


def test_overlay_cache_invalidates_when_dataset_data_is_replaced():
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, dataset, _model = _rpa_overlay_group()
    project_gui.current_model_channels(group)
    compiled_first = project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"]

    editable = dataset.data.mutable_copy()
    editable.signal.flat[0] += 1.0
    dataset.replace_data(editable)
    project_gui.current_model_channels(group)

    assert project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"] is not compiled_first


def test_dataset_activation_evaluation_creates_current_state_with_channels(
    monkeypatch,
):
    group = DataGroup(
        "Datagroup1",
        datasets=[DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")],
    )
    project_gui.ensure_fit_history(group)
    payload = {
        "scan": {
            "kind": "grid",
            "fit": np.ones((2, 2)),
            "residual": np.zeros((2, 2)),
        }
    }
    monkeypatch.setattr(project_gui, "current_model_channels", lambda _group: payload)

    current = project_gui.evaluate_current_state_model(group)

    assert current.kind == "current"
    assert current.channels is payload
    assert current.metadata["model_evaluation_status"] == "evaluated"
    assert current.metadata["model_evaluation_datasets"] == ["scan"]
    assert group.fits[-1] is current


def test_overlay_evaluates_only_valid_points(monkeypatch):
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, dataset, _model = _rpa_overlay_group()
    # Mask one grid cell; the overlay there must be NaN, and the model must not
    # be evaluated over the masked point.
    editable = dataset.data.mutable_copy()
    editable.mask[0, 0] = True
    dataset.replace_data(editable)

    seen_sizes = []
    original = project_gui.evaluate_problem_model

    def spy(problem, name, params, data=None):
        seen_sizes.append(data.size if data is not None else None)
        return original(problem, name, params, data=data)

    monkeypatch.setattr(project_gui, "evaluate_problem_model", spy)
    channels = project_gui.current_model_channels(group)
    fit = np.asarray(channels["scan"]["fit"], dtype=float)
    assert not np.isfinite(fit[0, 0])  # masked cell is NaN
    # exactly the unmasked points were evaluated (fewer than the full grid)
    assert seen_sizes and seen_sizes[0] == int(np.isfinite(fit).sum())


def test_request_overlay_refresh_coalesces_without_event_loop(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group, _dataset, _model = _rpa_overlay_group()
    explorer = NfitProjectExplorer(NfitProject([group]))

    refreshed = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda g: refreshed.append(g))
    # No viewer open -> request is a no-op.
    explorer._request_overlay_refresh(group)
    assert refreshed == []

    explorer._slice_viewers[id(group)] = object()

    # Headless/non-interactive: refresh happens synchronously so callers and
    # tests observe the update immediately.
    assert explorer._interactive is False
    explorer._request_overlay_refresh(group)
    assert refreshed == [group]

    # Interactive: rapid requests coalesce into the pending set + a debounce
    # timer instead of refreshing on every call.
    refreshed.clear()
    explorer._interactive = True
    explorer._request_overlay_refresh(group)
    explorer._request_overlay_refresh(group)
    assert refreshed == []
    assert id(group) in explorer._pending_overlay_groups
    assert explorer._overlay_refresh_timer is not None
    # Firing the debounced slot runs exactly one refresh and clears the queue.
    explorer._run_pending_overlay_refresh()
    assert refreshed == [group]
    assert not explorer._pending_overlay_groups


def test_viewer_view_cache_reuses_and_invalidates():
    project_gui._VIEWER_VIEW_CACHE.clear()
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    DataGroup("Datagroup1", datasets=[dataset])

    first = project_gui._viewer_data_before_scale(dataset)
    second = project_gui._viewer_data_before_scale(dataset)
    assert first is second  # identical masked view reused

    # A mask change must invalidate the cache (new object, new masking).
    dataset.masks = [MaskSpec(name="m", type="coordinate_range", parameters={"E": [0.0, 5.0]})]
    third = project_gui._viewer_data_before_scale(dataset)
    assert third is not second
    # ...and is reused again until the next change.
    assert project_gui._viewer_data_before_scale(dataset) is third


def test_viewer_lazy_loads_once_then_reuses_canonical_data(monkeypatch, tmp_path):
    project_gui._VIEWER_VIEW_CACHE.clear()
    source = tmp_path / "scan.nxs"
    source.write_bytes(b"placeholder")
    dataset = DatasetEntry(
        "scan",
        None,
        kind="nxs",
        metadata={"source_file": str(source), "import_status": "pending"},
    )
    calls = []

    def load(path, *, copy_metadata=False):
        calls.append((path, copy_metadata))
        return _grid_mdhisto_data()

    monkeypatch.setattr(project_gui, "load_mantid_mdhisto_nxs", load)
    first = project_gui._viewer_data_before_scale(dataset)
    second = project_gui._viewer_data_before_scale(dataset)

    assert calls == [(source, False)]
    assert dataset.data_revision == 1
    assert dataset.data_matches_source
    assert second is first


def test_viewer_view_cache_invalidates_when_dataset_data_is_replaced():
    project_gui._VIEWER_VIEW_CACHE.clear()
    dataset = DatasetEntry(
        "scan",
        _grid_mdhisto_data(),
        kind="mdhisto",
        data_type="single_crystal_inelastic",
    )
    first = project_gui._viewer_data_before_scale(dataset)

    editable = dataset.data.mutable_copy()
    editable.signal.flat[0] += 1.0
    dataset.replace_data(editable)
    second = project_gui._viewer_data_before_scale(dataset)

    assert second is not first
    assert second.signal.flat[0] == pytest.approx(editable.signal.flat[0])


def test_viewer_view_cache_uses_lru_eviction_instead_of_clear_all(monkeypatch):
    project_gui._VIEWER_VIEW_CACHE.clear()
    monkeypatch.setattr(project_gui, "_VIEWER_VIEW_CACHE_LIMIT", 3)
    datasets = [
        DatasetEntry(f"scan-{index}", _grid_mdhisto_data(), kind="mdhisto") for index in range(4)
    ]

    for dataset in datasets[:3]:
        project_gui._viewer_data_before_scale(dataset)
    project_gui._viewer_data_before_scale(datasets[0])
    project_gui._viewer_data_before_scale(datasets[3])

    assert list(project_gui._VIEWER_VIEW_CACHE) == [
        datasets[2].id,
        datasets[0].id,
        datasets[3].id,
    ]


def test_viewer_view_cache_does_not_retain_entry_over_byte_budget(monkeypatch):
    project_gui._VIEWER_VIEW_CACHE.clear()
    monkeypatch.setattr(project_gui, "_VIEWER_VIEW_CACHE_MAX_BYTES", 1)
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")

    result = project_gui._viewer_data_before_scale(dataset)

    assert isinstance(result, MDHistoData)
    assert dataset.id not in project_gui._VIEWER_VIEW_CACHE


def test_large_dataset_manual_masks_defer_passive_evaluation_and_force_for_fit(monkeypatch):
    project_gui._VIEWER_VIEW_CACHE.clear()
    monkeypatch.setattr(project_gui, "MASK_AUTO_MAX_POINTS", 1)
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    group = DataGroup("Datagroup1", datasets=[dataset])
    mask = create_mask(dataset)
    mask.parameters["E"] = [0.0, 5.0]
    config = project_gui.dataset_mask_application_config(dataset)
    assert config["auto_apply"] is False
    assert config["stale"] is False

    evaluations = []
    original = project_gui._evaluate_mdhisto_mask

    def record_evaluation(data, spec):
        evaluations.append(spec.name)
        return original(data, spec)

    monkeypatch.setattr(project_gui, "_evaluate_mdhisto_mask", record_evaluation)

    passive = dataset_for_slice_viewer(dataset, force_rebin=False, force_masks=False)
    assert evaluations == []
    assert passive.metadata["mask_application_pending"] is True
    assert config["stale"] is False

    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle is not None
    assert evaluations == [mask.name]
    assert config["stale"] is False

    mask.parameters["E"] = [5.0, 10.0]
    config["stale"] = True
    cached = dataset_for_slice_viewer(dataset, force_rebin=False, force_masks=False)
    assert evaluations == [mask.name]
    np.testing.assert_array_equal(cached.mask, bundle.view.mask)

    config["auto_apply"] = True
    automatic = dataset_for_slice_viewer(dataset, force_rebin=False, force_masks=False)
    assert automatic is not None
    assert evaluations == [mask.name, mask.name]
    assert config["stale"] is False


def test_large_mask_editor_defaults_manual_and_apply_now_clears_pending(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setattr(project_gui, "MASK_AUTO_MAX_POINTS", 1)
    project_gui._VIEWER_VIEW_CACHE.clear()

    dataset = DatasetEntry(
        "scan",
        _grid_mdhisto_data(),
        kind="mdhisto",
        data_type="single_crystal_inelastic",
    )
    mask = create_mask(dataset)
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_mask=mask)

    auto = explorer.window.findChild(QtWidgets.QCheckBox, "mask_auto_apply")
    apply_now = explorer.window.findChild(QtWidgets.QPushButton, "mask_apply_now")
    status = explorer.window.findChild(QtWidgets.QLabel, "mask_application_status")
    assert auto is not None and not auto.isChecked()
    assert apply_now is not None and apply_now.toolTip()
    assert status is not None and status.toolTip()
    assert "pending" in status.text().lower()

    explorer._set_mask_parameter("E", "[0.25, 5.0]")
    config = project_gui.dataset_mask_application_config(dataset)
    assert config["stale"] is True
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert (
        "pending"
        in explorer.window.findChild(QtWidgets.QLabel, "mask_application_status").text().lower()
    )

    assert explorer.apply_masks_now_for_selection()
    assert config["stale"] is False
    assert (
        "current"
        in explorer.window.findChild(QtWidgets.QLabel, "mask_application_status").text().lower()
    )


def test_apply_masks_switches_to_rebin_progress_when_rebinning(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    mask = create_mask(dataset)
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_mask=mask)
    dataset_rebin_config(dataset)["enabled"] = True

    titles = []
    events = []
    closed = []

    def make_progress(title):
        titles.append(title)

        def callback(event):
            events.append(event)

        return callback

    def fake_viewer_data(entry, **kwargs):
        callback = kwargs["progress_callback"]
        assert callback is not None
        callback({"iteration": 1, "total": 1, "message": "Rebinning batch 1 of 1"})
        return entry.data

    monkeypatch.setattr(explorer, "_make_rebin_progress_callback", make_progress)
    monkeypatch.setattr(explorer, "_close_rebin_progress", lambda callback: closed.append(callback))
    monkeypatch.setattr(project_gui, "dataset_for_slice_viewer", fake_viewer_data)

    assert explorer.apply_masks_now_for_selection()
    assert titles == ["Rebinning scan..."]
    assert events and events[0]["message"] == "Rebinning batch 1 of 1"
    assert len(closed) == 1


def test_mdhisto_fit_bin_count_matches_point_based_count():
    data = _grid_mdhisto_data()
    editable = data.mutable_copy()
    editable.mask[0, 0] = True
    editable.signal[1, 1] = np.nan
    data = editable.immutable_copy()
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    DataGroup("Datagroup1", datasets=[dataset])
    view = project_gui.dataset_for_slice_viewer(dataset)
    fast = project_gui._mdhisto_fit_bin_count(view)
    reference = int(np.count_nonzero(project_gui._point_data_from_mdhisto_view(view).valid_mask()))
    assert fast == reference


def test_fit_points_carry_magnetic_field_from_dataset_parameters():
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={
            "a": 3.0,
            "b": 5.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
    )

    # No field set -> nothing stamped.
    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle.points.magnetic_field is None
    assert project_gui.effective_dataset_field(group, dataset) is None

    dataset.parameters["magnetic_field"] = {
        "magnitude_T": 2.0,
        "direction": [1, 1, 0],
        "frame": "uvw",
    }
    bundle = project_gui.fit_data_bundle(group, dataset)
    expected = 2.0 * np.array([3.0, 5.0, 0.0]) / np.linalg.norm([3.0, 5.0, 0.0])
    np.testing.assert_allclose(bundle.points.magnetic_field, expected, atol=1e-12)

    # hkl frame differs for this orthorhombic lattice.
    dataset.parameters["magnetic_field"]["frame"] = "hkl"
    bundle = project_gui.fit_data_bundle(group, dataset)
    expected_hkl = 2.0 * np.array([1 / 3.0, 1 / 5.0, 0.0]) / np.linalg.norm([1 / 3.0, 1 / 5.0, 0.0])
    np.testing.assert_allclose(bundle.points.magnetic_field, expected_hkl, atol=1e-12)

    # Zero magnitude means no field.
    dataset.parameters["magnetic_field"]["magnitude_T"] = 0.0
    assert project_gui.effective_dataset_field(group, dataset) is None

    # Without lattice parameters the direction cannot be oriented.
    dataset.parameters["magnetic_field"]["magnitude_T"] = 2.0
    bare_group = DataGroup("Bare", datasets=[dataset])
    assert project_gui.effective_dataset_field(bare_group, dataset) is None


def test_sample_environment_panel_hosts_temperature_and_field(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={
            "a": 3.0,
            "b": 5.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    # Panel exists, is titled correctly, and sits between Dataset and Axes.
    panel = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "dataset_sample_environment_group"
    )
    assert panel is not None
    titles = [box.title() for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)]
    assert titles.index("Conditions") == titles.index("Dataset") + 1
    assert titles.index("Conditions") < titles.index("Axes")

    # Temperature control relocated but keeps its objectName + behavior.
    temp = explorer.window.findChild(QtWidgets.QDoubleSpinBox, "dataset_temperature")
    assert temp is explorer.dataset_temperature_spin
    explorer._set_selected_dataset_temperature(12.5)
    assert dataset.parameters["temperature"] == 12.5
    for name in ("dataset_fixed_q", "dataset_fixed_energy"):
        editor = explorer.window.findChild(QtWidgets.QLineEdit, name)
        assert editor is not None
        assert editor.toolTip()

    kinematic = explorer.window.findChild(QtWidgets.QCheckBox, "dataset_kf_ki_included")
    assert kinematic is explorer.dataset_kf_ki_included_check
    assert kinematic.isChecked()
    kinematic.setChecked(False)
    assert dataset.parameters[project_gui.KINEMATIC_KF_KI_INCLUDED_KEY] is False

    # Field controls write the structured parameter.
    explorer.dataset_field_magnitude_spin.setValue(2.0)
    explorer.dataset_field_direction_edit.setText("1 1 0")
    explorer._set_selected_dataset_field_direction()
    field = dataset.parameters["magnetic_field"]
    assert field["magnitude_T"] == 2.0
    assert field["direction"] == [1.0, 1.0, 0.0]
    assert field["frame"] == "uvw"

    # Switching the frame updates the payload; the panel re-syncs on reselect.
    idx = explorer.dataset_field_frame_combo.findData("hkl")
    explorer.dataset_field_frame_combo.setCurrentIndex(idx)
    assert dataset.parameters["magnetic_field"]["frame"] == "hkl"

    # Zeroing the magnitude clears the field entirely.
    explorer.dataset_field_magnitude_spin.setValue(-1.0)
    assert "magnetic_field" not in dataset.parameters

    # Field survives the data-group state snapshot/restore round-trip (the same
    # mechanism the fit timeline and project files serialize through).
    dataset.parameters["magnetic_field"] = {
        "magnitude_T": 3.0,
        "direction": [0, 0, 1],
        "frame": "hkl",
    }
    snapshot = project_gui.snapshot_data_group_state(group)
    dataset.parameters.pop("magnetic_field")
    project_gui.restore_data_group_state(group, snapshot)
    restored = list(group.iter_datasets())[0]
    assert restored.parameters["magnetic_field"] == {
        "magnitude_T": 3.0,
        "direction": [0, 0, 1],
        "frame": "hkl",
    }


def test_inelastic_dataset_panel_creates_typed_cross_section_and_chipp_channels(
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    dataset.data_type = "single_crystal_inelastic"
    dataset.parameters["temperature"] = 25.0
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    panel = next(
        box
        for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)
        if box.title() == "INS representations"
    )
    controls = (
        "ins_channels_enabled",
        "ins_source_representation",
        "ins_source_unit",
        "ins_fit_representation",
        "ins_normalization_basis",
        "ins_normalization_label",
        "ins_signal_per_mbarn",
        "ins_form_factor_ion",
        "ins_polarization_mode",
        "ins_polarization_scalar",
        "ins_moment_unit",
        "ins_g_factor",
        "ins_kf_ki_state",
        "ins_incident_energy",
        "ins_final_energy",
    )
    for name in controls:
        widget = panel.findChild(QtWidgets.QWidget, name)
        assert widget is not None
        assert widget.toolTip()

    fit_combo = panel.findChild(QtWidgets.QComboBox, "ins_fit_representation")
    fit_combo.setCurrentIndex(fit_combo.findData("chi_double_prime"))
    view = project_gui.dataset_for_slice_viewer(dataset)
    assert view.channel_quantity_type() == "dynamic_susceptibility"
    assert view.channel_quantity_type("scattering_cross_section") == "differential_cross_section"
    assert "dynamic_susceptibility" in view.auxiliary_channels


def test_saved_spectral_view_reimports_without_double_conversion(tmp_path):
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    dataset.data_type = "single_crystal_inelastic"
    dataset.parameters["temperature"] = 25.0
    config = project_gui.default_spectral_channel_config()
    config.update(
        {
            "source_unit": "mbarn/sr/meV/f.u.",
            "fit_representation": "chi_double_prime",
        }
    )
    dataset.parameters[project_gui.SPECTRAL_CHANNEL_CONFIG_KEY] = config
    expected = project_gui.dataset_for_slice_viewer(dataset)
    path = tmp_path / "spectral_view.npz"
    project_gui.save_dataset_file(dataset, path)

    restored = project_gui.dataset_entry_from_path(path, data_type="single_crystal_inelastic")
    actual = project_gui.dataset_for_slice_viewer(restored)
    np.testing.assert_allclose(actual.signal, expected.signal)
    assert actual.channel_unit() == expected.channel_unit()


def test_isaw_ub_round_trip_uses_transposed_file_convention(tmp_path):
    path = tmp_path / "orientation.mat"
    path.write_text(
        " 0.07074842 -0.05057369  0.08492118\n"
        " 0.06997787 -0.04812406 -0.08695870\n"
        " 0.06980259  0.09950387  0.00110520\n"
        " 8.2270 8.2270 8.2270 90.0000 90.0000 90.0000 556.8324\n"
        " 0 0 0 0 0 0 0\n"
    )
    ub, lattice = project_gui.read_isaw_ub(path)
    np.testing.assert_allclose(ub[:, 0], [0.07074842, -0.05057369, 0.08492118])
    output = tmp_path / "roundtrip.mat"
    project_gui.write_isaw_ub(output, ub, lattice)
    restored, restored_lattice = project_gui.read_isaw_ub(output)
    np.testing.assert_allclose(restored, ub)
    assert restored_lattice == pytest.approx(lattice)
    assert float(output.read_text().splitlines()[3].split()[6]) == pytest.approx(8.227**3, rel=1e-4)


def test_ub_from_lattice_orientation_uses_beam_x_and_vertical_z():
    lattice = {"a": 4.0, "b": 5.0, "c": 6.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    ub = project_gui.ub_from_lattice_orientation(lattice, [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(ub @ [1, 0, 0], [0.25, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ub @ [0, 1, 0], [0.0, 0.2, 0.0], atol=1e-12)
    assert np.cross(ub @ [1, 0, 0], ub @ [0, 1, 0])[2] > 0.0


def test_single_crystal_dataset_and_group_expose_ub_setup(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0), data_type="single_crystal_inelastic")
    subgroup = DatasetGroup("runs", datasets=[dataset])
    group = DataGroup("sample", subgroups=[subgroup])
    explorer = project_gui.NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group, select_dataset=dataset)
    assert explorer.details_widget.findChild(QtWidgets.QPushButton, "open_ub_setup") is not None
    explorer._refresh_tree(select_dataset_group=subgroup)
    assert explorer.details_widget.findChild(QtWidgets.QPushButton, "open_ub_setup") is not None


def test_dataset_importing_numor_parser_accepts_ranges_steps_and_gaps():
    assert project_gui.parse_dataset_numors("409981:409995") == list(range(409981, 409996))
    assert project_gui.parse_dataset_numors("409981:3:409995") == [
        409981,
        409984,
        409987,
        409990,
        409993,
    ]
    assert project_gui.parse_dataset_numors("409981:409982,409984:409985") == [
        409981,
        409982,
        409984,
        409985,
    ]
    assert project_gui.parse_dataset_numors("5,5,3:1") == [5, 3, 2, 1]
    with pytest.raises(ValueError, match="invalid run range"):
        project_gui.parse_dataset_numors("409981:0:409995")


def test_dataset_importing_panel_builds_paths_and_clears_nested_data(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    for number in (409981, 409982, 409984, 409985):
        (tmp_path / f"SEQ_{number}.nxs.h5").touch()
    group = DataGroup("sample")
    explorer = project_gui.NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_group=group)
    enabled = explorer.details_widget.findChild(QtWidgets.QCheckBox, "dataset_importing_enabled")
    controls = explorer.details_widget.findChild(QtWidgets.QWidget, "dataset_importing_controls")
    assert enabled is not None and controls is not None and not controls.isEnabled()
    enabled.setChecked(True)
    assert group.metadata["dataset_importing"]["enabled"] is True

    captured = []
    monkeypatch.setattr(
        explorer,
        "import_dataset_paths",
        lambda _group, paths, **_kwargs: captured.extend(paths) or [],
    )
    config = explorer._dataset_importing_config(group)
    config.update(
        {
            "path": str(tmp_path),
            "prefix": "SEQ_",
            "suffix": ".nxs.h5",
            "numors": "409981:409982,409984:409985",
        }
    )
    explorer._import_dataset_importing_range(group)
    assert [path.name for path in captured] == [
        "SEQ_409981.nxs.h5",
        "SEQ_409982.nxs.h5",
        "SEQ_409984.nxs.h5",
        "SEQ_409985.nxs.h5",
    ]

    group.datasets.append(DatasetEntry("direct", _tiny_mdhisto_data(1.0)))
    group.subgroups.append(
        DatasetGroup("nested", datasets=[DatasetEntry("nested-data", _tiny_mdhisto_data(1.0))])
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    explorer._clear_imported_datasets(group)
    assert not group.datasets and not group.subgroups


def test_interactive_dataset_import_stages_work_before_main_thread_attach(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("sample")
    explorer = project_gui.NfitProjectExplorer(NfitProject([group]))
    explorer._interactive = True

    def staged_import(staging, _paths, **_kwargs):
        entry = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
        staging.datasets.append(entry)
        return [entry]

    def run_now(**kwargs):
        result = kwargs["task"](lambda _event: None)
        kwargs["on_success"](result)
        return True

    monkeypatch.setattr(project_gui, "import_dataset_paths", staged_import)
    monkeypatch.setattr(explorer, "_start_background_task", run_now)

    assert explorer._request_dataset_import(group, ["scan.dat"])
    assert [dataset.name for dataset in group.datasets] == ["scan"]


def test_raw_dgs_nexus_import_creates_a_file_backed_reduction_group(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "SEQ_409981.nxs.h5"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("bank1_events")
    group = DataGroup("sample")

    entries = project_gui.import_dataset_paths(
        group, [source], data_type="single_crystal_inelastic"
    )

    assert entries[0].kind == "raw_dgs_nexus"
    assert len(group.subgroups) == 1
    assert "raw_dgs" in group.subgroups[0].metadata
    assert group.subgroups[0].metadata["raw_dgs"]["energy_min_fraction"] == -0.95
    assert group.subgroups[0].metadata["raw_dgs"]["energy_max_fraction"] == 0.95
    ok, _message = project_gui.data_group_composite_status(group.subgroups[0])
    assert ok
    assert project_gui.slice_viewer_datasets(group) == ([], [])
