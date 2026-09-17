import numpy as np
import pytest

from nfit.backgrounds import subtract_background, subtract_powder_background
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
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


def test_aligned_single_crystal_background_requires_matching_grid_and_propagates_variance():
    axes = (
        MDHistoAxis("H", np.array([-1.0, 0.0, 1.0]), "r.l.u.", "momentum"),
        MDHistoAxis("DeltaE", np.array([-1.0, 0.0, 1.0]), "meV", "energy"),
    )
    data = MDHistoData(
        axes,
        np.full((2, 2), 10.0),
        np.ones((2, 2)),
        np.zeros((2, 2), dtype=bool),
        np.ones((2, 2)),
    )
    background = MDHistoData(
        axes,
        np.full((2, 2), 4.0),
        np.full((2, 2), 2.0),
        np.zeros((2, 2), dtype=bool),
        np.ones((2, 2)),
    )

    result = subtract_background(data, background, scale=0.5)

    np.testing.assert_allclose(result.signal, 8.0)
    np.testing.assert_allclose(result.errors, np.sqrt(2.0))
    assert result.metadata["background_subtractions"][0]["interpolation"] == "aligned"

    shifted = background.with_updates(
        axes=(
            MDHistoAxis("H", np.array([-1.0, 0.1, 1.0]), "r.l.u.", "momentum"),
            axes[1],
        )
    )
    with pytest.raises(ValueError, match="identical axis"):
        subtract_background(data, shifted)


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


def test_linear_background_interpolation_uses_squared_uncertainty_weights():
    target = MDHistoData(
        axes=(
            MDHistoAxis("|Q|", np.array([1.0, 2.0]), "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", np.array([-1.0, 0.0]), "meV", "energy"),
        ),
        signal=np.array([[10.0]]),
        errors=np.array([[0.0]]),
        mask=np.zeros((1, 1), dtype=bool),
        num_events=np.ones((1, 1)),
        metadata={"signal_semantics": "density"},
    )
    background = _powder(
        [[1.0, 1.0], [1.0, 1.0]],
        np.ones((2, 2)),
    )

    result = subtract_powder_background(target, background, interpolation="linear")

    assert result.signal.item() == 9.0
    assert result.errors.item() == 0.5


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
    subgroup = DatasetGroup("Angles", datasets=[target], metadata={"mdevent": {}})
    subgroup.backgrounds.append(
        BackgroundSpec(
            "Environment",
            source.id,
            scale=0.75,
            projection="sample_trajectories",
            source_entry=source,
        )
    )
    payload = _project_to_dict(
        NfitProject(data_groups=[DataGroup("Workspace", [source], [subgroup])])
    )

    restored = _project_from_dict(payload)
    restored_group = restored.data_groups[0]
    restored_source = restored_group.datasets[0]
    restored_background = restored_group.subgroups[0].backgrounds[0]

    assert restored_background.scale == 0.75
    assert restored_background.projection == "sample_trajectories"
    assert restored_background.source_dataset_id == restored_source.id
    assert restored_background.source_entry is restored_source


def test_group_background_specs_round_trip_and_relink_by_composite_group_id():
    target = DatasetGroup("Sample")
    source = DatasetGroup("Powder background")
    target.backgrounds.append(
        BackgroundSpec(
            "Environment",
            source_group_id=source.id,
            source_group=source,
            scale=0.8,
        )
    )
    payload = _project_to_dict(
        NfitProject(data_groups=[DataGroup("Workspace", subgroups=[target, source])])
    )

    restored = _project_from_dict(payload)
    restored_target, restored_source = restored.data_groups[0].subgroups
    restored_background = restored_target.backgrounds[0]

    assert restored_background.source_dataset_id == ""
    assert restored_background.source_group_id == restored_source.id
    assert restored_background.source_group is restored_source


def test_parent_composite_recomputes_child_and_live_background_group_recipes():
    from nfit import project_gui

    sample_data = _powder(np.full((2, 2), 10.0), np.ones((2, 2)))
    background_data = _powder(np.full((2, 2), 2.0), np.ones((2, 2)))
    sample = DatasetGroup(
        "34 sample",
        datasets=[DatasetEntry("sample", sample_data, kind="mdhisto")],
    )
    background = DatasetGroup(
        "34 powder background",
        datasets=[DatasetEntry("background", background_data, kind="mdhisto")],
        enabled=False,
    )
    parent = DatasetGroup("Low temperature", subgroups=[sample, background])
    group = DataGroup("Workspace", subgroups=[parent])
    for node in (sample, background, parent):
        config = project_gui.data_group_composite_config(
            project_gui._composite_scope(group, node)
        )
        config.update({"enabled": True, "fractional": False})
    sample.backgrounds.append(
        BackgroundSpec(
            "Powder subtraction",
            source_group_id=background.id,
            source_group=background,
            scale=0.5,
        )
    )

    result = project_gui.composite_dataset_data(
        project_gui._composite_scope(group, parent)
    )
    np.testing.assert_allclose(result.signal, 9.0)

    sample.backgrounds[0].scale = 1.5
    updated = project_gui.composite_dataset_data(
        project_gui._composite_scope(group, parent)
    )
    np.testing.assert_allclose(updated.signal, 7.0)


def test_live_group_background_dependency_cycle_is_rejected():
    from nfit import project_gui

    first = DatasetGroup(
        "first",
        datasets=[DatasetEntry("first data", _powder(np.ones((2, 2)), np.ones((2, 2))), kind="mdhisto")],
    )
    second = DatasetGroup(
        "second",
        datasets=[DatasetEntry("second data", _powder(np.ones((2, 2)), np.ones((2, 2))), kind="mdhisto")],
    )
    group = DataGroup("Workspace", subgroups=[first, second])
    for node in (first, second):
        project_gui.data_group_composite_config(
            project_gui._composite_scope(group, node)
        ).update({"enabled": True, "fractional": False})
    first.backgrounds.append(
        BackgroundSpec("second", source_group_id=second.id, source_group=second)
    )
    second.backgrounds.append(
        BackgroundSpec("first", source_group_id=first.id, source_group=first)
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        project_gui.composite_dataset_data(project_gui._composite_scope(group, first))


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


def test_mdhisto_composite_automatically_uses_saved_normalization_denominator():
    first_data = _powder(np.full((2, 2), 2.0), np.ones((2, 2)))
    second_data = _powder(np.full((2, 2), 4.0), np.ones((2, 2)))
    first_data = first_data.with_updates(
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(np.ones((2, 2)))
        }
    )
    second_data = second_data.with_updates(
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(np.full((2, 2), 3.0))
        }
    )
    group = DataGroup(
        "Workspace",
        [
            DatasetEntry("first", first_data, kind="mdhisto"),
            DatasetEntry("second", second_data, kind="mdhisto"),
        ],
    )
    from nfit import project_gui

    config = project_gui.data_group_composite_config(group)
    config.update(
        {
            "enabled": True,
            "fractional": False,
            "mean_weighting": "uniform",
        }
    )

    result = project_gui.composite_dataset_data(group)

    np.testing.assert_allclose(result.signal, 3.5)
    np.testing.assert_allclose(result.errors, np.sqrt(10.0) / 4.0)
    assert result.metadata["rebin"]["mean_weighting"] == "uniform"
    assert result.metadata["rebin"]["weighted_by_normalization_denominator"] is True


def test_mdhisto_inverse_variance_also_uses_normalization_denominator():
    first_data = _powder(np.full((2, 2), 2.0), np.ones((2, 2)))
    second_data = _powder(np.full((2, 2), 4.0), np.full((2, 2), 2.0))
    first_data = first_data.with_updates(
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(np.ones((2, 2)))
        }
    )
    second_data = second_data.with_updates(
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(np.full((2, 2), 3.0))
        }
    )
    group = DataGroup(
        "Workspace",
        [
            DatasetEntry("first", first_data, kind="mdhisto"),
            DatasetEntry("second", second_data, kind="mdhisto"),
        ],
    )
    from nfit import project_gui

    config = project_gui.data_group_composite_config(group)
    config.update(
        {
            "enabled": True,
            "fractional": False,
            "mean_weighting": "inverse_variance",
        }
    )

    result = project_gui.composite_dataset_data(group)

    np.testing.assert_allclose(result.signal, 20.0 / 7.0)
    np.testing.assert_allclose(result.errors, np.sqrt(13.0) / 3.5)


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
    backgrounds_item = dataset_item.child(0)
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
    subgroup = DatasetGroup("Angles", datasets=[target], metadata={"mdevent": {}})
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
    projection = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "background_projection"
    )
    assert source_combo is not None and source_combo.toolTip()
    assert scale is not None and scale.toolTip()
    assert projection is not None and projection.toolTip()
    projection.setCurrentIndex(projection.findData("sample_trajectories"))
    assert background.projection == "sample_trajectories"
