import numpy as np
import pytest

from nfit.backgrounds import subtract_powder_background
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from nfit.project_gui import NfitProject, _project_from_dict, _project_to_dict


def _powder(signal, errors):
    signal = np.asarray(signal, dtype=float)
    return MDHistoData(
        axes=(
            MDHistoAxis("|Q|", np.array([0.5, 1.5, 2.5]), "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", np.array([-1.5, -0.5, 0.5]), "meV", "energy"),
        ),
        signal=signal,
        errors=np.asarray(errors, dtype=float),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={"signal_semantics": "density"},
    )


def test_powder_background_subtraction_propagates_scaled_variance():
    data = _powder([[10.0, 12.0], [14.0, 16.0]], np.ones((2, 2)))
    background = _powder([[2.0, 4.0], [6.0, 8.0]], np.full((2, 2), 2.0))

    result = subtract_powder_background(data, background, scale=0.5)

    np.testing.assert_allclose(result.signal, [[9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_allclose(result.errors, np.sqrt(2.0))
    assert not np.any(result.mask)


def test_powder_background_projects_onto_hkle_using_lattice_metadata():
    target = MDHistoData(
        axes=(
            MDHistoAxis("H", np.array([0.5, 1.5, 2.5]), "r.l.u.", "momentum"),
            MDHistoAxis("K", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("L", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", np.array([-1.5, -0.5, 0.5]), "meV", "energy"),
        ),
        signal=np.full((2, 1, 1, 2), 10.0),
        errors=np.ones((2, 1, 1, 2)),
        mask=np.zeros((2, 1, 1, 2), dtype=bool),
        num_events=np.ones((2, 1, 1, 2)),
        metadata={
            "signal_semantics": "density",
            "lattice_parameters": {
                "a": 2.0 * np.pi,
                "b": 2.0 * np.pi,
                "c": 2.0 * np.pi,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
        },
    )
    background = _powder([[1.0, 2.0], [3.0, 4.0]], np.ones((2, 2)))

    result = subtract_powder_background(target, background)

    np.testing.assert_allclose(
        result.signal[:, 0, 0, :],
        [[9.0, 8.0], [7.0, 6.0]],
    )
    np.testing.assert_allclose(result.errors, np.sqrt(2.0))


def test_background_specs_round_trip_and_relink_by_dataset_id():
    source = DatasetEntry(
        "Powder background",
        None,
        data_type="powder_inelastic",
        metadata={"source_file": "background.npz"},
    )
    target = DatasetEntry(
        "Single crystal",
        None,
        data_type="single_crystal_inelastic",
        metadata={"source_file": "data.npz"},
    )
    target.backgrounds.append(
        BackgroundSpec("Environment", source.id, scale=1.25, source_entry=source)
    )
    payload = _project_to_dict(NfitProject(data_groups=[DataGroup("Workspace", [target, source])]))

    restored = _project_from_dict(payload)
    restored_target, restored_source = restored.data_groups[0].datasets
    restored_background = restored_target.backgrounds[0]

    assert restored_background.scale == 1.25
    assert restored_background.source_dataset_id == restored_source.id
    assert restored_background.source_entry is restored_source


def test_group_background_specs_round_trip_and_relink_by_dataset_id():
    source = DatasetEntry(
        "Powder background",
        None,
        data_type="powder_inelastic",
        metadata={"source_file": "background.npz"},
    )
    target = DatasetEntry(
        "Single crystal",
        None,
        data_type="single_crystal_inelastic",
        metadata={"source_file": "data.npz"},
    )
    subgroup = DatasetGroup("Angles", datasets=[target])
    subgroup.backgrounds.append(
        BackgroundSpec("Environment", source.id, scale=0.75, source_entry=source)
    )
    payload = _project_to_dict(
        NfitProject(data_groups=[DataGroup("Workspace", [source], [subgroup])])
    )

    restored = _project_from_dict(payload)
    restored_group = restored.data_groups[0]
    restored_source = restored_group.datasets[0]
    restored_background = restored_group.subgroups[0].backgrounds[0]

    assert restored_background.scale == 0.75
    assert restored_background.source_dataset_id == restored_source.id
    assert restored_background.source_entry is restored_source


def test_composite_background_is_excluded_from_inputs_and_subtracted_once():
    source = DatasetEntry(
        "Powder background",
        _powder(np.full((2, 2), 2.0), np.ones((2, 2))),
        kind="mdhisto",
        data_type="powder_inelastic",
    )
    target = DatasetEntry(
        "Crystal",
        _powder(np.full((2, 2), 10.0), np.ones((2, 2))),
        kind="mdhisto",
        data_type="single_crystal_inelastic",
    )
    group = DataGroup("Workspace", [target, source])
    group.backgrounds.append(
        BackgroundSpec("Environment", source.id, scale=0.5, source_entry=source)
    )
    from nfit import project_gui

    config = project_gui.data_group_composite_config(group)
    config["enabled"] = True
    config["fractional"] = False
    result = project_gui.composite_dataset_data(group)

    np.testing.assert_allclose(result.signal, 9.0)
    np.testing.assert_allclose(result.errors, np.sqrt(1.25))
    assert result.metadata["background_subtractions"][0]["scale"] == 0.5


def test_project_tree_exposes_background_controls_with_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import NfitProjectExplorer

    source = DatasetEntry("Powder", _powder(np.ones((2, 2)), np.ones((2, 2))), data_type="powder_inelastic")
    target = DatasetEntry("Crystal", _powder(np.full((2, 2), 5.0), np.ones((2, 2))), data_type="single_crystal_inelastic")
    background = BackgroundSpec("Environment", source.id, source_entry=source)
    target.backgrounds.append(background)
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace", [target, source])]))

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    backgrounds_item = dataset_item.child(1)
    assert backgrounds_item.text(0) == "Backgrounds"
    explorer.tree.setCurrentItem(backgrounds_item.child(0))
    source_combo = explorer.details_widget.findChild(QtWidgets.QComboBox, "background_source_dataset")
    scale = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "background_scale")
    assert source_combo is not None and source_combo.toolTip()
    assert scale is not None and scale.toolTip()
    scale.setValue(0.5)
    assert background.scale == 0.5


def test_project_tree_exposes_group_background_controls_with_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import NfitProjectExplorer

    source = DatasetEntry(
        "Powder",
        _powder(np.ones((2, 2)), np.ones((2, 2))),
        data_type="powder_inelastic",
    )
    target = DatasetEntry(
        "Crystal",
        _powder(np.full((2, 2), 5.0), np.ones((2, 2))),
        data_type="single_crystal_inelastic",
    )
    subgroup = DatasetGroup("Angles", datasets=[target])
    background = BackgroundSpec("Environment", source.id, source_entry=source)
    subgroup.backgrounds.append(background)
    explorer = NfitProjectExplorer(
        NfitProject([DataGroup("Workspace", [source], [subgroup])])
    )

    datasets_item = explorer.tree.topLevelItem(0).child(0)
    subgroup_item = next(
        datasets_item.child(index)
        for index in range(datasets_item.childCount())
        if datasets_item.child(index).text(0) == "Angles"
    )
    backgrounds_item = next(
        subgroup_item.child(index)
        for index in range(subgroup_item.childCount())
        if subgroup_item.child(index).text(0) == "Backgrounds"
    )
    assert backgrounds_item.toolTip(0)
    explorer.tree.setCurrentItem(backgrounds_item.child(0))
    source_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "background_source_dataset"
    )
    scale = explorer.details_widget.findChild(
        QtWidgets.QDoubleSpinBox, "background_scale"
    )
    assert source_combo is not None and source_combo.toolTip()
    assert scale is not None and scale.toolTip()
