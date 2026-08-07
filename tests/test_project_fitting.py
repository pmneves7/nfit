# ruff: noqa: F401, F403, F405
import copy

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


def test_project_explorer_fit_history_creates_results_branches_and_restores(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("PySide6.QtCore")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 0.0]
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 0.5
    explorer = NfitProjectExplorer(NfitProject([group]))

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    initial_item = fits_item.child(0)
    assert fits_item.text(0) == "Fits"
    assert initial_item.text(0) == "Initial"

    explorer.tree.setCurrentItem(initial_item)
    result = explorer.fit_now_for_selection()

    assert result is not None
    assert result.name == "Fit Result1"
    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result1"]
    assert group.fits[1].goodness["status"] in {"converged", "not converged"}
    assert "first" in group.fits[1].channels

    result.goodness["diagnostics"] = {"gradient_norm": 0.01, "history": list(range(12))}
    result.metadata["fit_pipeline"] = {"version": "test", "notes": ["prepared", "solved"]}
    explorer._set_fit_details(result)

    def current_detail_boxes():
        boxes = []
        seen = set()

        def collect(widget):
            if isinstance(widget, QtWidgets.QGroupBox) and id(widget) not in seen:
                seen.add(id(widget))
                boxes.append(widget)
            for child in widget.findChildren(QtWidgets.QGroupBox):
                if id(child) not in seen:
                    seen.add(id(child))
                    boxes.append(child)

        for index in range(explorer.details_layout.count()):
            widget = explorer.details_layout.itemAt(index).widget()
            if widget is not None:
                collect(widget)
        return boxes

    panel_titles = [box.title() for box in current_detail_boxes()]
    parameter_splitter = explorer.details_widget.findChild(
        QtWidgets.QSplitter, "fit_details_parameter_splitter"
    )
    assert parameter_splitter is not None
    assert "Fit results" in panel_titles
    assert "Goodness of fit" in panel_titles
    assert "Stored fit channels" in panel_titles
    assert "Metadata" in panel_titles

    boxes_by_title = {box.title(): box for box in current_detail_boxes()}
    results_table = boxes_by_title["Fit results"].findChild(
        QtWidgets.QTableWidget, "fit_results_table"
    )
    goodness_tree = boxes_by_title["Goodness of fit"].findChild(
        QtWidgets.QTreeWidget, "fit_goodness_tree"
    )
    channels_tree = boxes_by_title["Stored fit channels"].findChild(
        QtWidgets.QTreeWidget, "fit_channels_tree"
    )
    metadata_tree = boxes_by_title["Metadata"].findChild(QtWidgets.QTreeWidget, "fit_metadata_tree")
    assert results_table is not None
    assert parameter_splitter.widget(0).title() == "Fit results"
    assert goodness_tree is not None
    assert channels_tree is not None
    assert metadata_tree is not None
    assert results_table.columnCount() == 6
    assert results_table.rowCount() >= 1
    assert goodness_tree.maximumHeight() == 260
    goodness_items = {
        goodness_tree.topLevelItem(index).text(0): goodness_tree.topLevelItem(index)
        for index in range(goodness_tree.topLevelItemCount())
    }
    assert goodness_items["diagnostics"].text(1) == "2 field(s)"
    metadata_items = {
        metadata_tree.topLevelItem(index).text(0): metadata_tree.topLevelItem(index)
        for index in range(metadata_tree.topLevelItemCount())
    }
    assert metadata_items["fit_pipeline"].text(1) == "2 field(s)"
    assert channels_tree.topLevelItemCount() >= 1

    dataset.masks[0].parameters["H"] = [2.0, 3.0]
    model.parameters["constant"] = 9.0
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    assert dataset.masks[0].parameters["H"] == [0.0, 0.0]
    assert model.parameters["constant"] == 0.5

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    restored_result_item = explorer.tree.topLevelItem(0).child(2).child(1)
    assert explorer.tree.currentItem() is model_item
    assert restored_result_item.font(0).italic()
    assert restored_result_item.font(0).bold()
    assert "Active fit state" in restored_result_item.toolTip(0)

    explorer._set_model_parameter("constant", "1.25")

    assert group.fits[1].children == []
    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result1", "Current state"]
    assert group.fits[2].kind == "current"
    assert group.fits[2].snapshot["models"][0]["parameters"]["constant"] == 1.25
    edit_current_item = explorer.tree.topLevelItem(0).child(2).child(2)
    assert explorer._fit_entry_for_item(edit_current_item) is group.fits[2]
    assert explorer.tree.currentItem().text(0) == "Model1"
    assert edit_current_item.font(0).italic()
    assert edit_current_item.font(0).bold()

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(2))
    explorer.fit_branch_check.setChecked(True)
    branched = explorer.fit_now_for_selection()

    assert branched is not None
    explicit_branch = group.fits[2].children[-1]
    assert explicit_branch.kind == "timeline"
    assert explicit_branch.children == [branched]
    assert explorer.tree.currentItem().text(0) == branched.name
    assert explorer.tree.currentItem().font(0).italic()
    assert explorer.tree.currentItem().font(0).bold()
    assert not explorer.fit_now_button.isHidden()
    assert explorer.import_dataset_button.isHidden()


def test_selecting_historic_fit_preserves_saved_current_state_masks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
    create_mask(dataset, "Mask1")
    group = DataGroup("Datagroup1", datasets=[dataset])
    one_mask_snapshot = project_gui.snapshot_data_group_state(group)

    create_mask(dataset, "Mask2")
    two_mask_snapshot = project_gui.snapshot_data_group_state(group)
    group.fits = [
        FitTimelineEntry("Initial", kind="initial", snapshot=one_mask_snapshot),
        FitTimelineEntry("Fit Result1", kind="result", snapshot=one_mask_snapshot),
        FitTimelineEntry("Fit Result2", kind="result", snapshot=two_mask_snapshot),
        FitTimelineEntry("Current state", kind="current", snapshot=two_mask_snapshot),
    ]
    explorer = NfitProjectExplorer(NfitProject([group]))

    fits_item = explorer.tree.topLevelItem(0).child(2)
    explorer.tree.setCurrentItem(fits_item.child(1))

    assert [entry.name for entry in group.fits] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Current state",
    ]
    assert len(group.datasets[0].masks) == 1
    assert len(group.fits[2].snapshot["datasets"][0]["masks"]) == 2

    fits_item = explorer.tree.topLevelItem(0).child(2)
    explorer.tree.setCurrentItem(fits_item.child(3))

    assert [entry.name for entry in group.fits] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Current state",
    ]
    assert len(group.datasets[0].masks) == 2


def test_open_selects_active_fit_without_overwriting_authoritative_live_state(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    initial_snapshot = project_gui.snapshot_data_group_state(group)
    group.fits = [
        FitTimelineEntry("Initial", kind="initial", snapshot=initial_snapshot)
    ]
    group.active_fit_path = [0]
    model.parameters["constant"] = 2.0
    path = tmp_path / "live-state.nfit"
    save_project(NfitProject([group]), path)

    explorer = NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    loaded_group = explorer.project.data_groups[0]

    assert loaded_group.models[model.name].parameters["constant"] == 2.0
    assert explorer._fit_entry_for_item(explorer.tree.currentItem()).name == "Initial"
    assert explorer.has_unsaved_changes is False

    # Restoring history remains available as an explicit selection action.
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))
    assert loaded_group.models[model.name].parameters["constant"] == 1.0
    assert explorer.has_unsaved_changes is True


def test_external_project_edit_reconciles_mutable_and_historical_fit_state(tmp_path):
    group = DataGroup("Datagroup1")
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    initial_snapshot = project_gui.snapshot_data_group_state(group)
    initial = FitTimelineEntry("Initial", kind="initial", snapshot=initial_snapshot)
    group.fits = [initial]
    group.active_fit_path = [0]

    model.parameters["constant"] = 2.0
    issues = project_gui.project_state_issues(NfitProject([group]))
    assert [issue.code for issue in issues] == ["active_fit_snapshot_differs"]
    with pytest.raises(project_gui.ProjectStateConsistencyError):
        project_gui.validate_project_state(NfitProject([group]))

    reconciled = project_gui.reconcile_external_project_edit(group)
    assert reconciled is initial
    assert reconciled.snapshot["models"][0]["parameters"]["constant"] == 2.0
    assert project_gui.project_state_issues(NfitProject([group])) == ()

    result_snapshot = copy.deepcopy(reconciled.snapshot)
    result = FitTimelineEntry("Fit Result1", kind="result", snapshot=result_snapshot)
    group.fits.append(result)
    group.active_fit_path = [1]
    model.parameters["constant"] = 3.0
    preserved_result = copy.deepcopy(result.snapshot)

    current = project_gui.reconcile_external_project_edit(group)

    assert current.kind == "current"
    assert group.active_fit_path == [2]
    assert result.snapshot == preserved_result
    assert current.snapshot["models"][0]["parameters"]["constant"] == 3.0
    assert project_gui.project_state_issues(NfitProject([group])) == ()

    path = tmp_path / "scripted-edit.nfit"
    save_project(NfitProject([group]), path)
    with project_gui.edit_project_file(path) as project:
        project.data_groups[0].models[model.name].parameters["constant"] = 4.0
    restored = load_project(path)
    restored_group = restored.data_groups[0]
    assert restored_group.models[model.name].parameters["constant"] == 4.0
    assert project_gui.project_state_issues(restored) == ()


def test_clear_fit_history_uses_selected_fit_as_new_initial_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    first_snapshot = project_gui.snapshot_data_group_state(group)
    model.parameters["constant"] = 2.0
    second_snapshot = project_gui.snapshot_data_group_state(group)
    group.fits = [
        FitTimelineEntry("Initial", kind="initial", snapshot=first_snapshot),
        FitTimelineEntry("Fit Result1", kind="result", snapshot=first_snapshot),
        FitTimelineEntry("Fit Result2", kind="result", snapshot=second_snapshot),
        FitTimelineEntry("Current state", kind="current", snapshot=second_snapshot),
    ]
    explorer = NfitProjectExplorer(NfitProject([group]))
    fits_item = explorer.tree.topLevelItem(0).child(2)
    latest_item = fits_item.child(2)
    explorer.tree.setCurrentItem(latest_item)
    fits_item = explorer.tree.topLevelItem(0).child(2)
    explorer.tree.setCurrentItem(fits_item)

    button = explorer.details_widget.findChild(QtWidgets.QPushButton, "clear_fit_history_button")
    assert button is not None and button.toolTip()
    assert explorer.clear_fit_history()

    assert [entry.name for entry in group.fits] == ["Initial"]
    assert group.fits[0].snapshot == second_snapshot
    assert group.models[model.name].parameters["constant"] == 2.0
    assert group.active_fit_path == [0]


def test_clear_fit_history_confirms_when_selected_fit_is_not_latest(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("scan", _tiny_mdhisto_data(1.0))])
    first_snapshot = project_gui.snapshot_data_group_state(group)
    group.fits = [
        FitTimelineEntry("Initial", kind="initial", snapshot=first_snapshot),
        FitTimelineEntry("Fit Result1", kind="result", snapshot=first_snapshot),
        FitTimelineEntry("Fit Result2", kind="result", snapshot=first_snapshot),
    ]
    explorer = NfitProjectExplorer(NfitProject([group]))
    fits_item = explorer.tree.topLevelItem(0).child(2)
    explorer.tree.setCurrentItem(fits_item.child(1))
    fits_item = explorer.tree.topLevelItem(0).child(2)
    explorer.tree.setCurrentItem(fits_item)
    prompts = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: prompts.append(args) or QtWidgets.QMessageBox.StandardButton.Cancel,
    )

    assert not explorer.clear_fit_history()
    assert [entry.name for entry in group.fits] == ["Initial", "Fit Result1", "Fit Result2"]
    assert prompts


def test_project_explorer_deletes_a_range_of_selected_fits(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    group.fits.extend(FitTimelineEntry(name=f"Fit Result{index}") for index in range(1, 5))
    explorer._refresh_tree(select_group=group)

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    assert fits_item.text(0) == "Fits"
    fits_item.setExpanded(True)
    assert [group_item.child(2).child(i).text(0) for i in range(fits_item.childCount())] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Fit Result3",
        "Fit Result4",
    ]

    # A shift/ctrl selection suppresses the per-fit state restore (and its tree
    # rebuild) so the growing multi-selection survives.
    monkeypatch.setattr(explorer, "_is_multi_select_gesture", lambda: True)

    # Select a contiguous run of result fits and delete them in one action.
    range_items = [fits_item.child(index) for index in (1, 2, 3)]
    explorer.tree.setCurrentItem(range_items[-1])
    for item in range_items:
        item.setSelected(True)

    explorer.delete_selected()

    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result4"]


def test_project_explorer_range_selection_stays_within_role(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    group.fits.append(FitTimelineEntry(name="Fit Result1"))
    explorer._refresh_tree(select_group=group)

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    fits_item.setExpanded(True)
    fit_item = fits_item.child(1)
    assert fit_item.text(0) == "Fit Result1"

    # Emulate a shift-click whose visual range sweeps in the workspace root and
    # folder headers: the current (clicked) item is a fit, so the non-fit rows
    # must be pruned back out of the selection.
    monkeypatch.setattr(explorer, "_is_multi_select_gesture", lambda: True)
    explorer.tree.setCurrentItem(fit_item)
    fit_item.setSelected(True)
    group_item.setSelected(True)
    fits_item.setSelected(True)

    selected_roles = {explorer._objects_for_item(item)[4] for item in explorer.tree.selectedItems()}
    assert selected_roles == {"fit"}


def test_fit_now_updates_live_model_parameters_and_editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _tiny_mdhisto_data(3.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))
    result = explorer.fit_now_for_selection()

    assert result is not None
    assert model.parameters["constant"] == pytest.approx(3.0, abs=1e-6)
    assert result.snapshot["models"][0]["parameters"]["constant"] == pytest.approx(3.0, abs=1e-6)

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    tooltip = model_parameter_tooltip("constant_background", "constant")
    parameter_editor = next(
        editor
        for editor in explorer.model_parameter_widget.findChildren(QtWidgets.QLineEdit)
        if editor.toolTip() == tooltip
    )
    assert float(parameter_editor.text()) == pytest.approx(3.0, abs=1e-6)


def test_write_back_fitted_parameters_includes_dynamic_orbit_parameters():
    model = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"chi0": 0.1, "gamma0": 5.0, "J1": 0.0},
        fit_parameters={"chi0": True, "gamma0": True, "J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}],
                }
            ],
        },
    )
    compiled = project_gui.compile_fit_problem(
        [model],
        [
            project_gui.FitDatasetInput(
                "scan", _points_for_dynamic_writeback(), data_type="single_crystal_inelastic"
            )
        ],
    )
    result = type(
        "Result",
        (),
        {"params": {"rpa.chi0": 0.2, "rpa.gamma0": 4.0, "rpa.J1": 1.25}},
    )()

    project_gui._write_back_fitted_parameters(DataGroup("Datagroup1"), [model], compiled, result)

    assert model.parameters["J1"] == pytest.approx(1.25)


def test_slice_viewer_datasets_attach_current_model_before_fit():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 7.0

    datasets, names = project_gui.slice_viewer_datasets(group)

    assert names == ["scan"]
    assert "fit" in datasets[0].metadata
    assert "residual" in datasets[0].metadata
    np.testing.assert_allclose(datasets[0].metadata["fit"], np.full(dataset.data.shape, 7.0))
    np.testing.assert_allclose(datasets[0].metadata["residual"], dataset.data.signal - 7.0)


def test_slice_viewer_unmasked_model_evaluates_masked_grid_bins():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    editable = dataset.data.mutable_copy()
    editable.mask[0, 0] = True
    dataset.replace_data(editable)
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 7.0

    masked, _names = project_gui.slice_viewer_datasets(group)
    assert np.isnan(masked[0].metadata["fit"][0, 0])

    unmasked, _names = project_gui.slice_viewer_datasets(group, unmask_model=True)
    assert unmasked[0].metadata["fit"][0, 0] == pytest.approx(7.0)
    assert np.isfinite(unmasked[0].metadata["residual"][0, 0])


def test_slice_viewer_datasets_prefer_current_model_over_stored_fit_channels():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 7.0
    group.fits = [
        FitTimelineEntry(
            "Fit Result1",
            kind="result",
            channels={
                "scan": {
                    "fit": np.full(dataset.data.shape, 3.0),
                    "residual": np.full(dataset.data.shape, -2.0),
                }
            },
        )
    ]

    datasets, names = project_gui.slice_viewer_datasets(group)

    assert names == ["scan"]
    np.testing.assert_allclose(datasets[0].metadata["fit"], np.full(dataset.data.shape, 7.0))
    np.testing.assert_allclose(datasets[0].metadata["residual"], dataset.data.signal - 7.0)


def test_live_overlay_keeps_successful_datasets_and_records_failures(monkeypatch):
    project_gui._MODEL_OVERLAY_CACHE.clear()
    project_gui._MODEL_OVERLAY_ERRORS.clear()
    group = DataGroup(
        "Datagroup1",
        datasets=[
            DatasetEntry("good", _grid_mdhisto_data(), kind="mdhisto"),
            DatasetEntry("bad", _grid_mdhisto_data(), kind="mdhisto"),
        ],
    )
    model = create_model_component(group)
    model.parameters["constant"] = 7.0
    original = project_gui.evaluate_problem_model

    def fail_one_dataset(problem, name, params, data=None):
        if name == "bad":
            raise RuntimeError("deliberate overlay failure")
        return original(problem, name, params, data=data)

    monkeypatch.setattr(project_gui, "evaluate_problem_model", fail_one_dataset)

    channels = project_gui.current_model_channels(group)
    current = project_gui.evaluate_current_state_model(group)

    assert set(channels) == {"good"}
    assert current.metadata["model_evaluation_status"] == "partially failed"
    assert current.metadata["model_evaluation_datasets"] == ["good"]
    assert "deliberate overlay failure" in current.metadata[
        "model_evaluation_errors"
    ]["bad"]


def test_point_fit_overlay_is_only_mapped_to_its_fitted_channel():
    from nfit.plotting import MDHistoSliceViewer

    temperature = np.array([10.0, 20.0, 30.0])
    view = PointListData(
        columns={
            "Temperature": temperature,
            "Moment": np.array([1.0, 0.8, 0.6]),
            "Susceptibility": np.array([0.1, 0.08, 0.06]),
        },
        units={"Temperature": "K", "Moment": "emu", "Susceptibility": "emu/Oe"},
        coordinate_names=["Temperature"],
        channels=[
            {"label": "Moment", "value": "Moment", "error": None},
            {
                "label": "Susceptibility",
                "value": "Susceptibility",
                "error": None,
            },
        ],
    )
    group = DataGroup("Datagroup1")
    view = project_gui.attach_fit_channels_to_view(
        group,
        "scan",
        view,
        fallback_payload={
            "kind": "points",
            "fit_channel": "Susceptibility",
            "fit": np.array([0.11, 0.09, 0.07]),
            "residual": np.array([-1.0, -1.0, -1.0]),
        },
    )

    viewer = MDHistoSliceViewer(view)
    assert viewer.channel == "Moment"
    assert viewer.point_overlay_channel("fit") is None
    assert "fit" not in viewer.slice_arrays()
    viewer.channel = "Susceptibility"
    assert viewer.point_overlay_channel("fit") == "fit"
    np.testing.assert_allclose(viewer.slice_arrays()["fit"], [0.11, 0.09, 0.07])


def test_point_fit_channel_round_trips_with_saved_fit_channels():
    encoded = project_gui._fit_channels_to_dict(
        {
            "scan": {
                "kind": "points",
                "fit_channel": "Susceptibility",
                "fit": np.array([1.0, 2.0]),
                "residual": np.array([0.0, 0.0]),
            }
        }
    )

    decoded = project_gui._fit_channels_from_dict(encoded)
    assert decoded["scan"]["fit_channel"] == "Susceptibility"


def test_project_explorer_fit_pipeline_controls_have_tooltips_and_update_config(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    controls = [
        explorer.fit_loss_combo,
        explorer.fit_covariance_mode_combo,
        explorer.fit_f_scale_spin,
        explorer.fit_finite_difference_workers_spin,
        explorer.fit_de_check,
        explorer.fit_de_maxiter_spin,
        explorer.fit_de_popsize_spin,
        explorer.fit_de_workers_spin,
        explorer.fit_emcee_check,
        explorer.fit_emcee_walkers_spin,
        explorer.fit_emcee_steps_spin,
        explorer.fit_emcee_burn_spin,
        explorer.fit_emcee_thin_spin,
        explorer.fit_emcee_seed_spin,
        explorer.fit_emcee_workers_spin,
        explorer.fit_optimizer_config_editor,
        explorer.fit_branch_check,
        explorer.fit_now_button,
        explorer.fit_corner_button,
    ]
    assert all(control.toolTip() for control in controls)
    assert explorer.fit_branch_check.parentWidget() is explorer.fit_editor_widget
    assert explorer.fit_settings_panel.parentWidget() is explorer.details_widget
    assert (
        explorer.details_widget.findChild(QtWidgets.QGroupBox, "fit_optimizer_settings_group")
        is not None
    )
    assert (
        explorer.details_widget.findChild(QtWidgets.QGroupBox, "fit_de_settings_group") is not None
    )
    assert (
        explorer.details_widget.findChild(QtWidgets.QGroupBox, "fit_posterior_settings_group")
        is not None
    )
    assert explorer.fit_emcee_workers_spin.value() == -1
    assert explorer.fit_finite_difference_workers_spin.value() == -1

    explorer.fit_loss_combo.setCurrentText("soft_l1")
    explorer.fit_covariance_mode_combo.setCurrentIndex(
        explorer.fit_covariance_mode_combo.findData("residual")
    )
    explorer.fit_f_scale_spin.setValue(2.0)
    explorer.fit_finite_difference_workers_spin.setValue(3)
    explorer.fit_de_check.setChecked(True)
    explorer.fit_de_maxiter_spin.setValue(11)
    explorer.fit_de_popsize_spin.setValue(4)
    explorer.fit_de_workers_spin.setValue(-1)
    explorer.fit_emcee_check.setChecked(True)
    explorer.fit_emcee_walkers_spin.setValue(16)
    explorer.fit_emcee_steps_spin.setValue(25)
    explorer.fit_emcee_burn_spin.setValue(5)
    explorer.fit_emcee_thin_spin.setValue(2)
    explorer.fit_emcee_workers_spin.setValue(3)

    fit_entry = group.fits[0]
    assert fit_entry.optimizer_config["loss"] == "soft_l1"
    assert fit_entry.optimizer_config["f_scale"] == 2.0
    assert fit_entry.optimizer_config["covariance_mode"] == "residual"
    assert fit_entry.optimizer_config["finite_difference_workers"] == 3
    assert fit_entry.optimizer_config["initialization"] == {
        "enabled": True,
        "method": "differential_evolution",
        "maxiter": 11,
        "popsize": 4,
        "workers": -1,
    }
    assert fit_entry.optimizer_config["sampler"] == {
        "enabled": True,
        "method": "emcee",
        "n_walkers": 16,
        "n_steps": 25,
        "burn_in": 5,
        "thin": 2,
        "workers": 3,
    }


def test_sampling_result_serializes_raw_chain_and_rewindows():
    result = project_gui.SamplingResult(
        samples=np.zeros((1, 1), dtype=float),
        variable_names=["level"],
        log_probability=np.zeros(1, dtype=float),
        metadata={"method": "emcee", "n_steps": 4, "n_walkers": 2, "burn_in": 0, "thin": 1},
        chain=np.arange(8, dtype=float).reshape(4, 2, 1),
        log_probability_chain=np.arange(8, dtype=float).reshape(4, 2),
    )

    restored = project_gui._sampling_result_from_dict(project_gui._sampling_result_to_dict(result))
    assert restored is not None
    assert restored.chain is not None
    np.testing.assert_allclose(restored.chain, result.chain)

    rewindowed = project_gui._sampling_result_with_window(restored, burn_in=1, thin=2)
    np.testing.assert_allclose(rewindowed.samples[:, 0], [2.0, 3.0, 6.0, 7.0])
    assert rewindowed.metadata["burn_in"] == 1
    assert rewindowed.metadata["thin"] == 2
    assert rewindowed.metadata["samples"] == 4


def test_fit_pipeline_saves_partial_emcee_when_sampler_is_cancelled(monkeypatch):
    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    chain = np.arange(4, dtype=float).reshape(2, 2, 1)
    partial = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        metadata={
            "method": "emcee",
            "n_walkers": 2,
            "n_steps": 2,
            "requested_n_steps": 10,
            "burn_in": 0,
            "thin": 1,
            "cancelled": True,
            "completed": False,
        },
        chain=chain,
        log_probability_chain=np.zeros((2, 2), dtype=float),
    )

    def cancel_sampling(*_args, **_kwargs):
        raise project_gui.SamplingCancelled("cancelled", partial)

    monkeypatch.setattr(project_gui, "sample_problem_parameters", cancel_sampling)

    outcome = project_gui.perform_group_fit(
        group,
        optimizer_config={"sampler": {"enabled": True, "n_walkers": 2, "n_steps": 10}},
    )

    assert outcome["goodness"]["status"] == "cancelled"
    assert "partial posterior" in outcome["goodness"]["message"]
    stored = project_gui._sampling_result_from_dict(outcome["metadata"].get("posterior_samples"))
    assert stored is not None
    assert stored.metadata["cancelled"] is True
    assert stored.chain is not None
    np.testing.assert_allclose(stored.chain, chain)


def test_fit_details_posterior_sampler_controls_update_burn_without_timeline(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    chain = np.arange(8, dtype=float).reshape(4, 2, 1)
    sampling = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        metadata={"method": "emcee", "n_steps": 4, "n_walkers": 2, "burn_in": 0, "thin": 1},
        chain=chain,
    )
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
        metadata={"posterior_samples": project_gui._sampling_result_to_dict(sampling)},
        snapshot=project_gui.snapshot_data_group_state(group),
    )
    group.fits = [result]
    explorer = NfitProjectExplorer(NfitProject([group]))

    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    controls = [
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_walkers_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_steps_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_burn_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_thin_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_seed_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_workers_spin"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_apply_button"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_rerun_button"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_append_button"),
        explorer.window.findChild(QtWidgets.QCheckBox, "fit_posterior_use_uncertainties_check"),
        explorer.window.findChild(QtWidgets.QCheckBox, "fit_posterior_use_best_sample_check"),
    ]
    assert all(control is not None and control.toolTip() for control in controls)
    workers_spin = explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_workers_spin")
    assert workers_spin.value() == -1
    posterior_group = explorer.window.findChild(QtWidgets.QGroupBox, "fit_posterior_settings_group")
    assert posterior_group is not None
    assert (
        explorer.window.findChild(QtWidgets.QWidget, "fit_posterior_result_actions").parentWidget()
        is posterior_group
    )

    burn_spin = explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_burn_spin")
    thin_spin = explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_thin_spin")
    burn_spin.setValue(1)
    thin_spin.setValue(2)
    before_count = len(group.fits)

    assert explorer.apply_posterior_sampling_window(result, burn_spin.value(), thin_spin.value())

    assert len(group.fits) == before_count
    posterior = result.goodness["posterior"]
    assert posterior["burn_in"] == 1
    assert posterior["thin"] == 2
    assert posterior["samples"] == 4


def test_posterior_rerun_confirms_before_replacing_existing_samples(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    sampling = project_gui.SamplingResult(
        samples=np.zeros((1, 1), dtype=float),
        variable_names=["model1.constant"],
        metadata={"method": "emcee"},
        chain=np.zeros((1, 1, 1), dtype=float),
    )
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
        metadata={"posterior_samples": project_gui._sampling_result_to_dict(sampling)},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    starts = []
    prompts = []
    monkeypatch.setattr(
        explorer,
        "start_posterior_sampler_for_fit",
        lambda *args, **kwargs: starts.append((args, kwargs)) or True,
    )

    def answer_prompt(parent, title, text, buttons, default):
        prompts.append((parent, title, text, buttons, default))
        return QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", answer_prompt)
    kwargs = dict(n_walkers=8, n_steps=20, burn_in=2, thin=1, random_seed=None, workers=1)

    assert explorer.confirm_and_start_posterior_rerun(group, result, **kwargs) is False
    assert starts == []
    assert prompts[0][1] == "Replace emcee samples?"
    assert "raw chain" in prompts[0][2]
    assert prompts[0][4] == QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    assert explorer.confirm_and_start_posterior_rerun(group, result, **kwargs) is True
    assert starts[0][0] == (group, result)
    assert starts[0][1] == {**kwargs, "append": False}


def test_posterior_rerun_without_existing_samples_does_not_prompt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: pytest.fail("unexpected overwrite confirmation"),
    )
    starts = []
    monkeypatch.setattr(
        explorer,
        "start_posterior_sampler_for_fit",
        lambda *args, **kwargs: starts.append((args, kwargs)) or True,
    )

    assert (
        explorer.confirm_and_start_posterior_rerun(
            group,
            result,
            n_walkers=8,
            n_steps=20,
            burn_in=2,
            thin=1,
            random_seed=None,
            workers=1,
        )
        is True
    )
    assert len(starts) == 1


def test_best_posterior_sample_updates_live_model_and_restores_lm_values(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(2.0))
    disabled = DatasetEntry("disabled", _tiny_mdhisto_data(8.0), enabled=False)
    group = DataGroup("Datagroup1", datasets=[dataset, disabled])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    model.fit_parameters["constant"] = True
    model.limits["constant"] = [0.0, 2.0]
    parameter_name = f"{model.name}.constant"
    chain = np.array([[[1.0]], [[2.0]]], dtype=float)
    log_probability_chain = np.array([[-0.5], [0.0]], dtype=float)
    sampling = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=[parameter_name],
        log_probability=log_probability_chain.reshape(-1),
        metadata={"method": "emcee", "n_steps": 2, "n_walkers": 1, "burn_in": 0, "thin": 1},
        chain=chain,
        log_probability_chain=log_probability_chain,
    )
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={
            "chi2": 1.0,
            "parameters": {parameter_name: 1.0},
            "posterior": project_gui._posterior_summary(sampling),
        },
        metadata={"posterior_samples": project_gui._sampling_result_to_dict(sampling)},
        snapshot=project_gui.snapshot_data_group_state(group),
        optimizer="custom_optimizer",
        optimizer_config={
            "initialization": {"enabled": True, "method": "differential_evolution", "maxiter": 12},
            "sampler": {"enabled": True, "method": "emcee", "n_steps": 25},
        },
    )
    group.fits = [result]
    explorer = NfitProjectExplorer(NfitProject([group]))
    refreshed_groups = []
    monkeypatch.setattr(
        explorer,
        "_request_overlay_refresh",
        lambda refreshed_group: refreshed_groups.append(refreshed_group),
    )

    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)
    use_best_sample = explorer.window.findChild(
        QtWidgets.QCheckBox, "fit_posterior_use_best_sample_check"
    )

    assert use_best_sample is not None
    assert use_best_sample.isEnabled()
    use_uncertainties = explorer.window.findChild(
        QtWidgets.QCheckBox, "fit_posterior_use_uncertainties_check"
    )
    assert use_uncertainties is not None
    assert use_uncertainties.isEnabled()
    use_uncertainties.setChecked(True)
    use_best_sample = explorer.window.findChild(
        QtWidgets.QCheckBox, "fit_posterior_use_best_sample_check"
    )
    assert use_best_sample is not None
    use_best_sample.setChecked(True)

    assert model.parameters["constant"] == pytest.approx(2.0)
    assert [fit.kind for fit in group.fits] == ["result"]
    display = result.metadata[project_gui.POSTERIOR_DISPLAY_KEY]
    assert display["use_best_sample"] is True
    assert display["use_posterior_uncertainties"] is True
    assert display["best_sample"]["parameters"][parameter_name] == pytest.approx(2.0)
    disabled_channels = project_gui.current_model_channels(group)["disabled"]
    assert disabled_channels["fit"] == pytest.approx(
        np.full(disabled.data.shape, 2.0)
    )
    assert project_gui._fit_results_rows(result)[0]["value"] == "2"
    result_table = explorer.window.findChild(QtWidgets.QTableWidget, "fit_results_table")
    assert result_table.horizontalHeaderItem(2).text() == "68% error"
    assert result_table.item(0, 1).foreground().color() == QtGui.QColor("#ff9c94")
    assert explorer._active_fit_entry(group) is result
    assert refreshed_groups and all(
        refreshed_group is group for refreshed_group in refreshed_groups
    )
    refresh_count = len(refreshed_groups)

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    value_editor = explorer.window.findChild(QtWidgets.QLineEdit, "model_parameter_value_constant")
    assert value_editor is not None
    assert "#c0392b" in value_editor.styleSheet()

    explorer._set_posterior_display_option(group, result, "use_best_sample", False)
    assert model.parameters["constant"] == pytest.approx(1.0)
    assert len(refreshed_groups) == refresh_count + 1


def test_posterior_display_uses_asymmetric_intervals_and_correlations():
    sampling = project_gui.SamplingResult(
        samples=np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        variable_names=["model1.a", "model1.b"],
    )
    entry = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={
            "parameters": {"model1.a": 1.5, "model1.b": 3.0},
            "stderr": {"model1.a": 0.2, "model1.b": 0.3},
            "posterior": project_gui._posterior_summary(sampling),
        },
        metadata={
            "posterior_samples": project_gui._sampling_result_to_dict(sampling),
            project_gui.POSTERIOR_DISPLAY_KEY: {"use_posterior_uncertainties": True},
        },
    )

    rows = project_gui._fit_results_rows(entry)
    assert rows[0]["uncertainty"].startswith("-")
    matrix, names, title = project_gui._covariance_matrix_from_fit_entry(entry)
    assert names == ["model1.a", "model1.b"]
    assert title == "Posterior correlation"
    np.testing.assert_allclose(matrix, np.ones((2, 2)))


def test_posterior_corner_density_panel_draws_contours():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    from nfit.project_gui import _draw_corner_density_panel

    rng = np.random.default_rng(12)
    x = rng.normal(0.0, 1.0, 1500)
    y = 0.7 * x + rng.normal(0.0, 0.35, 1500)
    fig = Figure()
    ax = fig.subplots()

    _draw_corner_density_panel(ax, x, y)

    assert ax.images
    assert ax.collections


def test_fit_diagnostics_detects_covariance_without_posterior_samples():
    from nfit.project_gui import _covariance_matrix_from_fit_entry, _fit_entry_has_diagnostic_plots

    entry = FitTimelineEntry(
        "Fit Result1",
        goodness={
            "covariance": {
                "variables": ["a", "b"],
                "matrix": [[4.0, -1.0], [-1.0, 9.0]],
            }
        },
    )

    matrix, names, title = _covariance_matrix_from_fit_entry(entry)
    np.testing.assert_allclose(matrix, [[4.0, -1.0], [-1.0, 9.0]])
    assert names == ["a", "b"]
    assert title == "Covariance"
    assert _fit_entry_has_diagnostic_plots(entry)


def test_fit_result_summary_places_chi_squared_after_duration():
    entry = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        duration_seconds=12.5,
        goodness={"chi2": 8.0, "reduced_chi2": 1.25},
    )

    lines = project_gui.fit_details_text(entry).splitlines()

    duration_index = lines.index("Duration: 12.5 s")
    assert lines[duration_index + 1 : duration_index + 3] == [
        "Chi^2: 8",
        "Reduced Chi^2: 1.25",
    ]


def test_fit_diagnostics_matrix_heatmap_draws_image():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    from nfit.project_gui import _draw_matrix_heatmap

    fig = Figure()
    ax = fig.subplots()

    _draw_matrix_heatmap(
        ax,
        [[1.0, 0.5], [0.5, 2.0]],
        ["a", "b"],
        labels={"a": "$A$", "b": "p2"},
        title="Covariance",
    )

    assert ax.images
    assert ax.get_title() == "Covariance"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["$A$", "p2"]


def test_fit_diagnostics_matrix_heatmap_is_centered_in_figure():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    from nfit.project_gui import _draw_centered_matrix_heatmap

    fig = Figure(figsize=(8, 6))
    ax = _draw_centered_matrix_heatmap(
        fig,
        [[1.0, 0.2], [0.2, 3.0]],
        ["a", "b"],
        labels={"a": "p1", "b": "p2"},
        title="Covariance",
    )

    bbox = ax.get_position()
    assert abs((bbox.x0 + bbox.x1) / 2.0 - 0.5) < 0.08
    assert abs((bbox.y0 + bbox.y1) / 2.0 - 0.5) < 0.08


def test_corner_histogram_panel_draws_step_histogram_and_reference_lines():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    from nfit.project_gui import _draw_corner_histogram_panel

    fig = Figure()
    ax = fig.subplots()

    _draw_corner_histogram_panel(
        ax,
        np.linspace(0.0, 1.0, 200),
        "amplitude",
        {"best": 0.5, "low": 0.4, "high": 0.7},
    )
    ax.set_ylabel("Count")

    assert ax.patches
    assert not ax.patches[0].get_fill()
    assert len(ax.lines) >= 3
    assert r"\pm" in ax.get_title()
    assert ax.get_ylabel() == "Count"


def test_corner_reference_lines_are_solid_only():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    from nfit.project_gui import _draw_corner_reference_lines

    fig = Figure()
    ax = fig.subplots()

    _draw_corner_reference_lines(
        ax,
        {"best": 1.0, "low": 0.5, "high": 1.5},
        {"best": 2.0, "low": 1.5, "high": 2.5},
    )

    assert ax.lines
    assert all(line.get_linestyle() != ":" for line in ax.lines)


def test_trace_panel_draws_walkers_and_burn_in_marker():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    from nfit.project_gui import _draw_trace_panel

    chain = np.stack(
        [
            np.column_stack([np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6)]),
            np.column_stack([np.linspace(0.5, 1.5, 6), np.linspace(1.5, 2.5, 6)]),
        ],
        axis=1,
    )
    result = project_gui.SamplingResult(
        samples=chain.reshape(-1, 2),
        variable_names=["a", "b"],
        metadata={"burn_in": 2},
        chain=chain,
    )
    fig = Figure()
    ax = fig.subplots()

    _draw_trace_panel(ax, result, parameter_index=0, summary={"best": 0.75})

    assert len(ax.lines) == 4
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    assert [text.get_text() for text in ax.texts] == ["burn-in"]


def test_fit_diagnostics_label_table_updates_plot_labels(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import _FitDiagnosticsPlotWindow

    entry = FitTimelineEntry(
        "Fit Result1",
        goodness={
            "parameters": {"a": 1.0, "b": 2.0},
            "covariance": {
                "variables": ["a", "b"],
                "matrix": [[1.0, 0.1], [0.1, 2.0]],
            },
        },
        metadata={
            "posterior_samples": {
                "variable_names": ["a", "b"],
                "samples": project_gui._encode_float_array(
                    np.column_stack(
                        [
                            np.linspace(0.8, 1.2, 64),
                            np.linspace(1.8, 2.2, 64),
                        ]
                    )
                ),
            }
        },
    )
    explorer = NfitProjectExplorer(NfitProject())

    window = _FitDiagnosticsPlotWindow(entry, explorer)
    table = window.window.findChild(QtWidgets.QTableWidget, "fit_diagnostics_label_table")
    splitter = window.window.findChild(QtWidgets.QSplitter, "fit_diagnostics_plot_label_splitter")

    assert table is not None
    assert splitter is not None
    assert splitter.orientation() == QtCore.Qt.Orientation.Vertical
    assert splitter.widget(0) is window.tabs
    assert splitter.widget(1) is table
    assert table.item(0, 1).text() == "p1"
    corner_index = [window.tabs.tabText(index) for index in range(window.tabs.count())].index(
        "Corner"
    )
    window.tabs.setCurrentIndex(corner_index)
    table.item(0, 1).setText(r"$\Gamma$")

    assert entry.metadata["parameter_labels"]["a"] == r"$\Gamma$"
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Corner"


def test_fit_diagnostics_trace_uses_chain_steps_when_available(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import _FitDiagnosticsPlotWindow

    chain = np.arange(24, dtype=float).reshape(6, 2, 2)
    entry = FitTimelineEntry(
        "Fit Result1",
        metadata={
            "posterior_samples": project_gui._sampling_result_to_dict(
                project_gui.SamplingResult(
                    samples=chain.reshape(-1, 2),
                    variable_names=["a", "b"],
                    metadata={"burn_in": 2},
                    chain=chain,
                )
            )
        },
    )
    explorer = NfitProjectExplorer(NfitProject())

    window = _FitDiagnosticsPlotWindow(entry, explorer)
    trace_index = [window.tabs.tabText(index) for index in range(window.tabs.count())].index(
        "Trace"
    )
    canvas = window.tabs.widget(trace_index)

    assert canvas.figure.axes[-1].get_xlabel() == "MCMC step"


def test_fit_progress_dialog_uses_parameter_table_and_resets(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    from nfit.project_gui import _FitProgressDialog

    explorer = NfitProjectExplorer(NfitProject())
    dialog = _FitProgressDialog(explorer)

    dialog.update_progress(
        {
            "stage": "least_squares",
            "iteration": 10,
            "cost": 12.5,
            "seconds_per_step": 0.125,
            "parameters": {
                "model.constant[a]": 1.25,
                "model.constant[b]": 2.5,
            },
        }
    )

    table = dialog.dialog.findChild(QtWidgets.QTableWidget, "fit_progress_parameter_table")
    splitter = dialog.dialog.findChild(QtWidgets.QSplitter, "fit_progress_panel_splitter")
    log = dialog.dialog.findChild(QtWidgets.QPlainTextEdit, "fit_progress_log")
    assert table is not None
    assert splitter is not None
    assert log is dialog.log
    assert splitter.orientation() == QtCore.Qt.Orientation.Vertical
    assert splitter.indexOf(table) == 0
    assert splitter.indexOf(log) == 1
    assert table.rowCount() == 2
    assert dialog.cancel_button.text() == "Terminate"
    assert "lowest-objective parameter set evaluated so far" in dialog.cancel_button.toolTip()
    assert "model.constant" not in dialog.log.toPlainText()
    assert "Least-squares fit" in dialog.stage_label.text()
    assert "125 ms/step" in dialog.status_label.text()
    assert "125 ms/step" in dialog.log.toPlainText()

    dialog.update_progress(
        {
            "stage": "bragg_integration",
            "completed": 3,
            "total": 12,
            "accepted_count": 2,
            "rejected_count": 1,
            "message": "Reflection 3/12: (1, 1, 1) accepted; coverage 100%; I/dI 8.2",
        }
    )
    assert dialog.stage_label.text() == "Bragg integration"
    assert "Reflection 3 of 12" in dialog.status_label.text()
    assert "2 accepted" in dialog.status_label.text()
    assert "(1, 1, 1) accepted" in dialog.log.toPlainText()

    dialog.finish(
        "Fit pipeline finished.",
        parameters={"model.constant[a]": 1.0, "model.constant[b]": 2.5},
        limit_hits=[{"name": "model.constant[a]", "side": "upper", "bound": 1.0}],
        summary_lines=["emcee integrated autocorrelation time (steps): [12.3]."],
    )
    assert "limit reached" in dialog.status_label.text()
    assert "fit parameter limit reached" in dialog.log.toPlainText()
    assert "integrated autocorrelation time" in dialog.log.toPlainText()
    assert table.item(0, 1).foreground().color() == QtGui.QColor("#c0392b")
    assert table.item(1, 1).foreground().color() != QtGui.QColor("#c0392b")

    dialog.reset()

    assert table.rowCount() == 0
    assert dialog.log.toPlainText() == ""

    summary = project_gui._progress_event_summary(
        {
            "stage": "emcee",
            "iteration": 3,
            "elapsed_seconds": 1.5,
            "seconds_per_step": 0.5,
        }
    )
    assert summary["elapsed_seconds"] == 1.5
    assert summary["seconds_per_step"] == 0.5


def test_fit_limit_hits_are_saved_and_rendered_in_fit_results(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.fitting import ParameterSpec

    hits = project_gui._fit_parameter_limit_hits(
        [
            ParameterSpec("constant.offset", 1.0, min=0.0, max=1.0),
            ParameterSpec("constant.near", -99.9957, min=-100.0, max=100.0),
            ParameterSpec("constant.edgeward", -99.9, min=-100.0, max=100.0),
            ParameterSpec("constant.free", 0.4, min=0.0, max=1.0),
            ParameterSpec("constant.fixed", 1.0, min=0.0, max=1.0, vary=False),
        ],
        {
            "constant.offset": 1.0,
            "constant.near": -99.9957,
            "constant.edgeward": -99.9,
            "constant.free": 0.4,
            "constant.fixed": 1.0,
        },
    )
    assert hits == [
        {"name": "constant.offset", "side": "upper", "bound": 1.0},
        {"name": "constant.near", "side": "lower", "bound": -100.0},
    ]

    entry = FitTimelineEntry(
        name="Fit Result1",
        kind="result",
        goodness={
            "parameters": {"constant.offset": 1.0, "constant.free": 0.4},
            "parameters_at_limits": hits,
        },
    )
    explorer = NfitProjectExplorer(NfitProject())
    box = explorer._fit_results_group_box(entry)
    table = box.findChild(QtWidgets.QTableWidget, "fit_results_table")
    assert table is not None
    assert all(
        table.item(0, column).background().color() == QtGui.QColor("#5a2929") for column in range(6)
    )
    assert table.item(1, 1).background().color() != QtGui.QColor("#5a2929")

def test_model_editor_marks_effectively_boundary_pinned_parameter_red(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    model = ModelComponentSpec(
        "Model1",
        parameters={"constant": -99.9957},
        fit_parameters={"constant": True},
        sharing={"constant": {"mode": "global", "groups": {}}},
        limits={"constant": [-100.0, 100.0]},
    )
    group = DataGroup("Datagroup1", models={model.name: model})
    explorer = NfitProjectExplorer(NfitProject([group]))
    model_item = explorer.tree.topLevelItem(0).child(1).child(0)

    explorer.tree.setCurrentItem(model_item)

    editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_parameter_value_constant"
    )
    assert editor is not None
    assert "#c0392b" in editor.styleSheet()
    assert "lower bound" in editor.toolTip()


def test_project_explorer_reuses_fit_progress_dialog(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    def fake_run_group_fit(group, parent, *, branch_timeline=False, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "least_squares",
                    "iteration": 1,
                    "parameters": {"model.constant": 1.0},
                }
            )
        return parent

    monkeypatch.setattr(project_gui, "run_group_fit", fake_run_group_fit)
    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    explorer.fit_now_for_selection()
    first_dialog = explorer._fit_progress_dialog
    assert first_dialog is not None
    assert first_dialog.dialog.isVisible()
    assert first_dialog.close_button.isEnabled()
    explorer.fit_now_for_selection()

    assert explorer._fit_progress_dialog is first_dialog
    assert first_dialog.dialog.isVisible()


def test_project_explorer_start_fit_runs_in_background_worker(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))
    calls = []

    def fake_run_group_fit(group, parent, *, branch_timeline=False, progress_callback=None):
        calls.append("run")
        if progress_callback is not None:
            progress_callback({"stage": "least_squares", "iteration": 1, "total": 1})
        return project_gui.create_fit_result_entry(
            group,
            parent,
            branch_timeline=branch_timeline,
            goodness={"status": "converged", "parameters": {}},
        )

    monkeypatch.setattr(project_gui, "run_group_fit", fake_run_group_fit)
    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    assert explorer.start_fit_for_selection()
    assert explorer._fit_worker_thread is not None

    deadline = time.monotonic() + 3.0
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()

    assert calls == ["run"]
    assert explorer._fit_worker_thread is None
    assert explorer._fit_progress_dialog is not None
    assert explorer._fit_progress_dialog.dialog.isVisible()
    assert explorer._fit_progress_dialog.close_button.isEnabled()


def test_background_posterior_cancel_saves_partial_chain_and_reenables_gui(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    result_entry = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
        snapshot=project_gui.snapshot_data_group_state(group),
    )
    group.fits = [result_entry]
    explorer = NfitProjectExplorer(NfitProject([group]))
    chain = np.arange(6, dtype=float).reshape(3, 2, 1)
    partial = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        log_probability=np.zeros(6, dtype=float),
        metadata={
            "method": "emcee",
            "n_walkers": 2,
            "n_steps": 3,
            "requested_n_steps": 20,
            "burn_in": 0,
            "thin": 1,
            "cancelled": True,
            "completed": False,
        },
        chain=chain,
        log_probability_chain=np.zeros((3, 2), dtype=float),
    )

    def cancel_sampler(_group, _fit_entry, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback({"stage": "emcee", "iteration": 3, "total": 20})
        raise project_gui.SamplingCancelled("cancelled", partial)

    monkeypatch.setattr(explorer, "_posterior_sampler_result_for_fit", cancel_sampler)

    assert explorer.start_posterior_sampler_for_fit(
        group,
        result_entry,
        n_walkers=2,
        n_steps=20,
        burn_in=0,
        thin=1,
        random_seed=None,
        workers=1,
    )
    assert not explorer.tree.isEnabled()
    assert explorer._fit_worker_thread is not None

    deadline = time.monotonic() + 3.0
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()

    assert explorer._fit_worker_thread is None
    assert explorer.tree.isEnabled()
    assert explorer.details_scroll.isEnabled()
    stored = project_gui._sampling_result_from_dict(result_entry.metadata.get("posterior_samples"))
    assert stored is not None
    assert stored.metadata["cancelled"] is True
    assert stored.metadata["n_steps"] == 3
    assert stored.chain is not None
    np.testing.assert_allclose(stored.chain, chain)
    assert result_entry.goodness["posterior"]["samples"] == 6
    assert explorer._fit_progress_dialog is not None
    assert explorer._fit_progress_dialog.close_button.isEnabled()


def test_partial_emcee_chain_does_not_overwrite_requested_steps_in_fit_editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    chain = np.arange(6, dtype=float).reshape(3, 2, 1)
    partial = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        log_probability=np.zeros(6, dtype=float),
        metadata={
            "method": "emcee",
            "n_walkers": 2,
            "n_steps": 3,
            "requested_n_steps": 20,
            "burn_in": 0,
            "thin": 1,
            "cancelled": True,
            "completed": False,
        },
        chain=chain,
        log_probability_chain=np.zeros((3, 2), dtype=float),
    )
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
        optimizer_config={
            "sampler": {
                "enabled": True,
                "method": "emcee",
                "n_walkers": 2,
                "n_steps": 20,
                "burn_in": 0,
                "thin": 1,
            }
        },
        metadata={"posterior_samples": project_gui._sampling_result_to_dict(partial)},
        snapshot=project_gui.snapshot_data_group_state(group),
    )
    group.fits = [result]
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))

    assert explorer.fit_emcee_steps_spin.value() == 20
    assert result.optimizer_config["sampler"]["n_steps"] == 20
    stored = project_gui._sampling_result_from_dict(result.metadata["posterior_samples"])
    assert stored is not None
    assert stored.metadata["n_steps"] == 3


def test_background_task_failure_reenables_gui(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    explorer = NfitProjectExplorer(NfitProject())
    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, message: messages.append((parent, title, message)),
    )

    def fail(_progress_callback):
        raise ValueError("analysis failure")

    assert explorer._start_background_task(
        title="Running analysis",
        failure_title="Analysis failed",
        task=fail,
        on_success=lambda _result: True,
        success_message="Analysis complete",
    )
    deadline = time.monotonic() + 3.0
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()

    assert explorer._fit_worker_thread is None
    assert explorer.tree.isEnabled()
    assert explorer.details_scroll.isEnabled()
    assert messages == [(explorer.window, "Analysis failed", "analysis failure")]


def test_background_task_can_keep_completed_analysis_log_open(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    explorer = NfitProjectExplorer(NfitProject())

    def complete(progress_callback):
        progress_callback(
            {
                "stage": "bragg_integration",
                "completed": 1,
                "total": 1,
                "accepted_count": 1,
                "rejected_count": 0,
                "message": "Reflection 1/1: (1, 1, 1) accepted; coverage 100%",
            }
        )
        return {"complete": True}

    assert explorer._start_background_task(
        title="Preparing bragg integration...",
        failure_title="Analysis failed",
        task=complete,
        on_success=lambda _result: True,
        success_message="Analysis complete.",
        close_on_success=False,
        completion_summary=lambda _result: ["Integrated 1: 1 accepted, 0 rejected."],
        progress_window_title="Analysis progress",
    )
    deadline = time.monotonic() + 3.0
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()

    progress = explorer._fit_progress_dialog
    assert progress is not None
    assert progress.dialog.windowTitle() == "Analysis progress"
    assert progress.dialog.isVisible()
    assert progress.close_button.isEnabled()
    assert "Reflection 1/1" in progress.log.toPlainText()
    assert "Integrated 1: 1 accepted, 0 rejected." in progress.log.toPlainText()
    progress.close()


def test_project_explorer_edits_initial_state_in_place_without_results(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    initial_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(initial_item)
    assert [fit.name for fit in group.fits] == ["Initial"]

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    explorer._set_model_fit_parameter("constant", True)
    explorer._set_model_sharing_mode("constant", "per_dataset")

    assert model.fit_parameters["constant"] is True
    assert model.sharing["constant"]["mode"] == "per_dataset"
    assert [fit.name for fit in group.fits] == ["Initial"]
    assert group.fits[0].children == []
    snapshot_model = group.fits[0].snapshot["models"][0]
    assert snapshot_model["fit_parameters"]["constant"] is True
    assert snapshot_model["sharing"]["constant"]["mode"] == "per_dataset"


def test_project_explorer_fit_now_from_earlier_result_creates_nested_timeline(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))
    explorer.fit_now_for_selection()
    for _index in range(4):
        fits_item = explorer.tree.topLevelItem(0).child(2)
        explorer.tree.setCurrentItem(fits_item.child(fits_item.childCount() - 1))
        explorer.fit_now_for_selection()

    assert [fit.name for fit in group.fits] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Fit Result3",
        "Fit Result4",
        "Fit Result5",
    ]

    fit_result3_item = explorer.tree.topLevelItem(0).child(2).child(3)
    explorer.tree.setCurrentItem(fit_result3_item)
    branched = explorer.fit_now_for_selection()

    assert branched is not None
    assert [fit.name for fit in group.fits] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Fit Result3",
        "Fit Result4",
        "Fit Result5",
    ]
    assert group.fits[3].children[0].kind == "timeline"
    assert group.fits[3].children[0].children[0] is branched
    assert group.fits[3].children[0].children == [branched]
    assert explorer.tree.currentItem().text(0) == branched.name
    assert not explorer.fit_now_button.isHidden()
    assert explorer.import_dataset_button.isHidden()

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    explorer._set_model_parameter("constant", "2.5")

    timeline = group.fits[3].children[0]
    assert [entry.kind for entry in timeline.children] == ["result", "current"]
    assert timeline.children[1].snapshot["models"][0]["parameters"]["constant"] == 2.5
    assert timeline.children[0].children == []
