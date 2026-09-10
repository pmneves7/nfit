import copy

import numpy as np
import pytest

import nfit
from nfit.composite_spectral import composite_spectral_config
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
from nfit.project_composites import (
    _cached_composite_dataset_data,
    _composite_scope,
    composite_dataset_entry,
    data_group_composite_config,
)
from nfit.spectral_channels import with_paired_spectral_channels


def source(t, value=10):
    data = nfit.PointData4D(
        H=[0.], K=[0.], L=[0.], E=[1.], intensity=[value], sigma=[2.], temperature=t
    )
    return DatasetEntry(str(t), data, data_type="single_crystal_inelastic", parameters={"temperature": t})


def collection(temperatures):
    group = DataGroup("scans", datasets=[source(t) for t in temperatures])
    config = data_group_composite_config(group)
    config.update(enabled=True, fractional=True, minimum_coverage=0, mean_weighting="uniform")
    for i, axis in enumerate(config["axes"]):
        axis.update(bin_edges=[-.5, .5] if i < 3 else [.5, 1.5], mode="edges",
                    auto_lower=False, auto_upper=False, auto_step_size=False)
    return group, config


@pytest.mark.parametrize("axis", [False, True])
def test_conversion_matches_public_spectral_api_and_cached_fit_entry(axis):
    group, config = collection([5, 20] if axis else [5, 5])
    if axis:
        nfit.set_metadata_dimensions(group, [nfit.MetadataDimension("Temperature", "parameters/temperature", "K")])
    raw = nfit.composite_dataset_data(group)
    config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    result = nfit.composite_dataset_data(group)
    expected = with_paired_spectral_channels(
        raw, config["spectral_channels"], temperature_K=np.array([5, 20]) if axis else 5
    )
    np.testing.assert_allclose(result.signal, expected.signal)
    np.testing.assert_allclose(result.errors, expected.errors)
    np.testing.assert_allclose(_cached_composite_dataset_data(group).signal, result.signal)
    entry = composite_dataset_entry(group)
    np.testing.assert_allclose(nfit.dataset_for_slice_viewer(entry).signal, result.signal)
    assert "dynamic_susceptibility" in result.auxiliary_channels
    assert group.datasets[0].data.intensity[0] == 10


@pytest.mark.parametrize("temperatures", [[5, 20], [5, None]])
def test_mixed_or_missing_temperature_requires_axis_or_explicit_nominal(temperatures):
    group, config = collection([5, 20])
    if temperatures[1] is None:
        group.datasets[1].replace_data(group.datasets[1].data.with_updates(temperature=None))
        group.datasets[1].parameters.pop("temperature")
    config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    with pytest.raises(ValueError, match="temperature metadata dimension"):
        nfit.composite_dataset_data(group)
    config["spectral_channels"]["temperature_K"] = 7
    assert np.isfinite(nfit.composite_dataset_data(group).signal).all()


def test_metadata_temperature_takes_precedence_over_nominal():
    group, config = collection([5, 20])
    nfit.set_metadata_dimensions(group, [nfit.MetadataDimension("Temperature", "parameters/temperature", "K")])
    config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime", "temperature_K": 999}
    values = nfit.composite_dataset_data(group).signal.ravel()
    assert values[0] > values[1]


def test_nested_composite_does_not_convert_child_twice():
    group, config = collection([5, 5])
    child = DatasetGroup("child", datasets=group.datasets)
    group.datasets = []
    group.subgroups.append(child)
    scope = _composite_scope(group, child)
    child_config = data_group_composite_config(scope)
    child_config.update(copy.deepcopy(config))
    child_config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    baseline = nfit.composite_dataset_data(group)
    config["spectral_channels"] = dict(child_config["spectral_channels"])
    result = nfit.composite_dataset_data(group)
    expected = with_paired_spectral_channels(baseline, config["spectral_channels"], temperature_K=5)
    np.testing.assert_allclose(result.signal, expected.signal)


def test_energy_and_conventions_inferred_from_all_sources():
    group, config = collection([5, 5])
    for entry in group.datasets:
        entry.parameters["spectral_channels"] = {"kf_ki_state": "included", "final_energy_meV": 3.7}
    inferred = composite_spectral_config(config, group.datasets)
    assert inferred["kf_ki_state"] == "included"
    assert inferred["final_energy_meV"] == 3.7
    group.datasets[1].parameters["spectral_channels"]["final_energy_meV"] = 5
    assert composite_spectral_config(config, group.datasets)["final_energy_meV"] is None


def test_composite_gui_uses_saved_recipe_and_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    group, config = collection([5, 5])
    explorer = nfit.NfitProjectExplorer(nfit.NfitProject([group]))
    monkeypatch.setattr(explorer, "_after_group_composite_changed", lambda g: None)
    panel = explorer._group_composite_group_box(group)
    try:
        enabled = panel.findChild(QtWidgets.QCheckBox, "ins_channels_enabled")
        enabled.setChecked(True)
        selector = panel.findChild(QtWidgets.QComboBox, "ins_fit_representation")
        selector.setCurrentIndex(selector.findData("chi_double_prime"))
        assert config["spectral_channels"]["fit_representation"] == "chi_double_prime"
        assert config["spectral_channels"]["enabled"]
        for widget in panel.findChildren(QtWidgets.QWidget):
            if widget.objectName().startswith(("ins_", "composite_temperature")):
                assert widget.toolTip()
        assert np.isfinite(nfit.composite_dataset_data(group).signal).all()
    finally:
        panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_saved_composite_workflow_replays_spectral_settings(tmp_path):
    group, config = collection([5, 5])
    for index, entry in enumerate(group.datasets):
        single, _ = collection([5])
        raw = nfit.composite_dataset_data(single)
        path = tmp_path / f"source{index}.npz"
        nfit.save_dataset_file(entry.copy(data=raw), path, use_view=False)
        group.datasets[index] = nfit.dataset_entry_from_path(path)
        group.datasets[index].parameters["temperature"] = 5
    config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    expected = nfit.composite_dataset_data(group)
    path = tmp_path / "project.nfit"
    nfit.save_project(nfit.NfitProject([group]), path)
    project = nfit.load_project(path)
    script = nfit.composite_workflow_script(project, group.name)
    assert "spectral_channels" in script
    namespace = {"__name__": "test_workflow"}
    exec(compile(script, "composite.py", "exec"), namespace)
    output = namespace["run"]()
    np.testing.assert_allclose(output.signal, expected.signal)
    np.testing.assert_allclose(output.errors, expected.errors)
    restored = composite_dataset_entry(project.data_groups[0])
    np.testing.assert_allclose(restored.data.signal, expected.signal)


def test_dataset_metadata_fills_tab_and_physics_controls_are_compact(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    group, _ = collection([5])
    explorer = nfit.NfitProjectExplorer(nfit.NfitProject([group]))
    try:
        explorer._set_dataset_details(group.datasets[0], group)
        tabs = explorer.window.findChild(QtWidgets.QTabWidget, "dataset_details_tabs")
        tabs.setParent(None)
        tabs.resize(1100, 800)
        tabs.show()
        tabs.setCurrentIndex(3)
        QtWidgets.QApplication.processEvents()
        tree = tabs.findChild(QtWidgets.QTreeWidget, "dataset_metadata_tree")
        assert tree.height() > tabs.height() * .8
        assert tree.maximumHeight() > 10000
        tabs.setCurrentIndex(1)
        QtWidgets.QApplication.processEvents()
        scalar = tabs.findChild(QtWidgets.QLineEdit, "ins_polarization_scalar")
        assert not scalar.isEnabled()
        assert not tabs.findChild(QtWidgets.QLineEdit, "ins_g_factor").isEnabled()
    finally:
        tabs.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_background_is_subtracted_before_conversion():
    from nfit.pipeline import BackgroundSpec

    template, config = collection([5])
    sample = DatasetGroup("sample", datasets=template.datasets, metadata={"composite": config})
    background = DatasetGroup("background", datasets=[source(50, 2)], enabled=False)
    root = DataGroup("workspace", subgroups=[sample, background])
    group = _composite_scope(root, sample)
    background_config = data_group_composite_config(_composite_scope(root, background))
    background_config.update(copy.deepcopy(config))
    background_config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    sample.backgrounds.append(BackgroundSpec(
        "background", source_group_id=background.id, source_group=background
    ))
    raw = nfit.composite_dataset_data(group)
    np.testing.assert_allclose(raw.signal, 8)
    config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    result = nfit.composite_dataset_data(group)
    expected = with_paired_spectral_channels(raw, config["spectral_channels"], temperature_K=5)
    np.testing.assert_allclose(result.signal, expected.signal)
    np.testing.assert_allclose(result.errors, expected.errors)


def test_invalid_temperature_and_missing_kinematics_do_not_silently_fall_back():
    group, config = collection([5])
    config["spectral_channels"] = {"enabled": True, "temperature_K": -1}
    with pytest.raises(ValueError, match="positive and finite"):
        nfit.composite_dataset_data(group)
    config["spectral_channels"] = {"enabled": True, "kf_ki_state": "included"}
    with pytest.raises(ValueError, match="energy|Ei|Ef"):
        nfit.composite_dataset_data(group)


def test_materialized_composite_keeps_underlying_signal_and_conversion_recipe(tmp_path):
    group, config = collection([5])
    config["spectral_channels"] = {"enabled": True, "fit_representation": "chi_double_prime"}
    expected = nfit.composite_dataset_data(group)
    source_path = tmp_path / "source.npz"
    raw = nfit.composite_dataset_data(group, apply_spectral_channels=False)
    nfit.save_dataset_file(group.datasets[0].copy(data=raw), source_path, use_view=False)
    group.datasets[0] = nfit.dataset_entry_from_path(source_path)
    group.datasets[0].parameters["temperature"] = 5
    path = tmp_path / "materialized.nfit"
    nfit.save_project(nfit.NfitProject([group]), path)
    entry = nfit.materialize_composite_dataset(path, group)
    np.testing.assert_allclose(entry.data.signal, 10)
    assert entry.parameters["temperature"] == 5
    np.testing.assert_allclose(nfit.dataset_for_slice_viewer(entry).signal, expected.signal)
