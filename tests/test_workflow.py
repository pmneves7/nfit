from __future__ import annotations

import json

import numpy as np
import pytest

from nfit import (
    AnalysisEntry,
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    FitTimelineEntry,
    MaskSpec,
    MDHistoAxis,
    MDHistoData,
    NfitProject,
    NfitProjectExplorer,
    PointData4D,
    WorkflowNode,
    WorkflowOutput,
    WorkflowPlan,
    WorkflowValidationError,
    analysis_workflow_plan,
    analysis_workflow_script,
    create_model_component,
    dataset_entry_from_path,
    dataset_for_slice_viewer,
    dataset_workflow_plan,
    dataset_workflow_script,
    fit_data_bundle,
    fit_workflow_plan,
    fit_workflow_script,
    prepare_analysis_input,
    prepare_analysis_inputs,
    run_project_analysis,
    save_dataset_file,
)
from nfit.project_gui import effective_dataset_masks


def _grid(value: float) -> MDHistoData:
    q = MDHistoAxis("Q", np.array([0.0, 1.0, 2.0]), "A^-1", "momentum")
    energy = MDHistoAxis("E", np.array([0.0, 10.0, 20.0]), "meV", "energy")
    signal = value * np.array([[1.0, 2.0], [3.0, 4.0]])
    return MDHistoData(
        axes=(q, energy),
        signal=signal,
        errors=np.full_like(signal, value),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )


def test_workflow_plan_round_trips_and_orders_dependencies():
    source = WorkflowNode(
        id="source:a",
        kind="source_dataset",
        operation="load",
        outputs=(WorkflowOutput("dataset", "entry:a", "dataset_entry"),),
    )
    prepared = WorkflowNode(
        id="dataset:a",
        kind="prepared_dataset",
        operation="prepare",
        dependencies=("source:a",),
        outputs=(WorkflowOutput("data", "a", "prepared_dataset"),),
    )
    unused = WorkflowNode(id="source:unused", kind="source_dataset", operation="load")
    plan = WorkflowPlan(nodes=(unused, prepared, source), targets=("dataset:a",))

    plan.validate()
    assert [node.id for node in plan.topological_nodes()] == ["source:a", "dataset:a"]
    assert WorkflowPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan


def test_workflow_plan_reports_cycles_and_duplicate_outputs():
    cyclic = WorkflowPlan(
        nodes=(
            WorkflowNode(
                id="a",
                kind="source_dataset",
                operation="a",
                dependencies=("b",),
            ),
            WorkflowNode(
                id="b",
                kind="prepared_dataset",
                operation="b",
                dependencies=("a",),
            ),
        ),
        targets=("a",),
    )
    with pytest.raises(WorkflowValidationError, match="cycle"):
        cyclic.validate()

    duplicate_output = WorkflowPlan(
        nodes=(
            WorkflowNode(
                id="a",
                kind="source_dataset",
                operation="a",
                outputs=(WorkflowOutput("x", "same", "dataset_entry"),),
            ),
            WorkflowNode(
                id="b",
                kind="source_dataset",
                operation="b",
                outputs=(WorkflowOutput("x", "same", "dataset_entry"),),
            ),
        ),
        targets=("a",),
    )
    with pytest.raises(WorkflowValidationError, match="produced by both"):
        duplicate_output.validate()


def test_dataset_workflow_script_rebuilds_prepared_dataset(tmp_path):
    target_path = tmp_path / "target.npz"
    background_path = tmp_path / "background.npz"
    save_dataset_file(DatasetEntry("raw target", _grid(2.0)), target_path, use_view=False)
    save_dataset_file(
        DatasetEntry("raw background", _grid(0.25)),
        background_path,
        use_view=False,
    )

    target = dataset_entry_from_path(
        target_path,
        data_type="single_crystal_inelastic",
    )
    target.name = "Measured signal"
    target.scale_factor = 1.5
    target.scale_factor_vary = True
    target.scale_factor_group = "same_run"
    target.masks = [
        MaskSpec(
            "Exclude high H",
            parameters={"Q": [1.0, 2.0]},
        )
    ]
    background = dataset_entry_from_path(
        background_path,
        data_type="single_crystal_inelastic",
    )
    background.name = "Empty can"
    target.backgrounds = [
        BackgroundSpec(
            "Subtract empty can",
            background.id,
            scale=0.5,
            source_entry=background,
        )
    ]
    group = DataGroup(
        "Experiment",
        datasets=[target, background],
        masks=[
            MaskSpec(
                "Exclude high energy",
                parameters={"E": [10.0, 20.0]},
            )
        ],
    )
    project = NfitProject([group])

    script = dataset_workflow_script(project, target.id)
    compile(script, "<nfit-workflow>", "exec")
    namespace = {"__name__": "imported_workflow"}
    exec(script, namespace)
    result = namespace["build_workflow"]()
    expected = dataset_for_slice_viewer(
        target,
        extra_masks=effective_dataset_masks(group, target),
    )
    actual = result.prepared_datasets[target.id]

    assert set(result.source_datasets) == {target.id, background.id}
    rebuilt_target = result.source_datasets[target.id]
    assert rebuilt_target.scale_factor_vary is True
    assert rebuilt_target.scale_factor_group == "same_run"
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    assert "from nfit import" in script
    assert "load_project" not in script
    assert "PySide" not in script
    assert "def load_sources" in script
    assert "def prepare_datasets" in script


def test_dataset_workflow_rejects_non_source_and_grouped_reduction_datasets(tmp_path):
    derived = DatasetEntry(
        "Derived",
        _grid(1.0),
        metadata={"derived_from_analysis": "analysis-id"},
    )
    project = NfitProject([DataGroup("Experiment", datasets=[derived])])
    with pytest.raises(WorkflowValidationError, match="derived from an analysis"):
        dataset_workflow_plan(project, derived.id)

    path = tmp_path / "grouped.npz"
    save_dataset_file(DatasetEntry("raw", _grid(1.0)), path, use_view=False)
    grouped = dataset_entry_from_path(path, data_type="single_crystal_inelastic")
    grouped.kind = "raw_dgs_nexus"
    project = NfitProject([DataGroup("Experiment", datasets=[grouped])])
    with pytest.raises(WorkflowValidationError, match="grouped reduction"):
        dataset_workflow_plan(project, grouped.id)


def test_dataset_workflow_can_be_copied_and_saved_from_gui(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    source_path = tmp_path / "source.npz"
    save_dataset_file(DatasetEntry("raw", _grid(1.0)), source_path, use_view=False)
    dataset = dataset_entry_from_path(
        source_path,
        data_type="single_crystal_inelastic",
    )
    explorer = NfitProjectExplorer(
        NfitProject([DataGroup("Experiment", datasets=[dataset])])
    )
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    assert explorer.copy_dataset_workflow_script_for_selection()
    copied = QtWidgets.QApplication.clipboard().text()
    assert "def build_workflow" in copied

    output = tmp_path / "saved_workflow.py"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output), "Python scripts (*.py)"),
    )
    assert explorer.save_dataset_workflow_script_for_selection()
    assert output.read_text(encoding="utf-8") == copied


def test_analysis_workflow_rebuilds_prepared_inputs_and_runs_analysis(tmp_path):
    cold_path = tmp_path / "cold.npz"
    warm_path = tmp_path / "warm.npz"
    save_dataset_file(DatasetEntry("cold raw", _grid(1.0)), cold_path, use_view=False)
    save_dataset_file(DatasetEntry("warm raw", _grid(1.5)), warm_path, use_view=False)
    cold = dataset_entry_from_path(
        cold_path,
        data_type="powder_inelastic",
    )
    warm = dataset_entry_from_path(
        warm_path,
        data_type="powder_inelastic",
    )
    cold.name = "10 K"
    warm.name = "100 K"
    cold.parameters["temperature"] = 10.0
    warm.parameters["temperature"] = 100.0
    cold.scale_factor = 2.0
    cold.masks = [
        MaskSpec("Exclude high Q", parameters={"Q": [1.0, 2.0]})
    ]
    analysis = AnalysisEntry(
        "Separate elastic signal",
        "bose_elastic_separation",
        [cold.id, warm.id],
        {},
    )
    group = DataGroup(
        "Experiment",
        datasets=[cold, warm],
        analyses=[analysis],
    )
    project = NfitProject([group])

    prepared_inputs = prepare_analysis_inputs(group, analysis)
    viewed_cold = dataset_for_slice_viewer(
        cold,
        extra_masks=effective_dataset_masks(group, cold),
    )
    np.testing.assert_allclose(
        prepared_inputs[0].data.signal,
        viewed_cold.signal,
        equal_nan=True,
    )
    np.testing.assert_array_equal(prepared_inputs[0].data.mask, viewed_cold.mask)

    plan = analysis_workflow_plan(project, analysis.id)
    assert plan.targets == (f"analysis:{analysis.id}",)
    assert {
        node.id for node in plan.topological_nodes()
    } >= {
        f"source:{cold.id}",
        f"dataset:{cold.id}",
        f"source:{warm.id}",
        f"dataset:{warm.id}",
        f"analysis:{analysis.id}",
    }

    script = analysis_workflow_script(project, analysis.id)
    compile(script, "<nfit-analysis-workflow>", "exec")
    namespace = {"__name__": "imported_workflow"}
    exec(script, namespace)
    actual = namespace["build_workflow"]().analysis_results[analysis.id]
    expected = run_project_analysis(group, analysis)

    assert actual.outputs.keys() == expected.outputs.keys()
    for key in actual.outputs:
        np.testing.assert_allclose(
            actual.outputs[key].data.signal,
            expected.outputs[key].data.signal,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            actual.outputs[key].data.mask,
            expected.outputs[key].data.mask,
        )


def test_point_data_view_analysis_and_fit_share_preparation():
    points = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[5.0, 5.0],
        intensity=[2.0, 3.0],
        sigma=[0.2, 0.3],
    )
    dataset = DatasetEntry(
        "points",
        points,
        data_type="single_crystal_inelastic",
        scale_factor=4.0,
        masks=[MaskSpec("Exclude second", parameters={"H": [0.5, 1.5]})],
    )
    group = DataGroup("Experiment", datasets=[dataset])

    viewed = dataset_for_slice_viewer(dataset)
    analysis_input = prepare_analysis_input(group, dataset)
    fit_bundle = fit_data_bundle(group, dataset)

    assert isinstance(viewed, PointData4D)
    assert fit_bundle is not None
    np.testing.assert_allclose(viewed.intensity, [8.0, 12.0])
    np.testing.assert_allclose(analysis_input.data.intensity, viewed.intensity)
    np.testing.assert_allclose(fit_bundle.points.intensity, viewed.intensity)
    np.testing.assert_array_equal(viewed.mask, [True, False])
    np.testing.assert_array_equal(analysis_input.data.mask, viewed.mask)
    np.testing.assert_array_equal(fit_bundle.points.mask, viewed.mask)


def test_fit_workflow_rebuilds_and_fits_live_state(tmp_path):
    path = tmp_path / "fit_data.npz"
    save_dataset_file(DatasetEntry("raw", _grid(1.0)), path, use_view=False)
    dataset = dataset_entry_from_path(path, data_type="powder_inelastic")
    dataset.name = "scan"
    group = DataGroup("Fit workspace", datasets=[dataset])
    model = create_model_component(group, type="constant_background")
    model.fit_parameters["constant"] = True
    group.fits = [
        FitTimelineEntry(
            "Current state",
            kind="current",
            optimizer_config={"loss": "linear"},
        )
    ]
    group.active_fit_path = [0]
    project = NfitProject([group])

    plan = fit_workflow_plan(project, group.name)
    fit_node = plan.topological_nodes()[-1]
    assert fit_node.kind == "fit"
    assert fit_node.config["optimizer_config"] == {"loss": "linear"}

    script = fit_workflow_script(project, group.name)
    compile(script, "<nfit-fit-workflow>", "exec")
    namespace = {"__name__": "imported_workflow"}
    exec(script, namespace)
    result = namespace["build_workflow"]()
    outcome = result.fit_results[f"fit:{group.name}"]

    assert outcome["goodness"]["status"] == "converged"
    assert outcome["goodness"]["parameters"]["Model1.constant"] == pytest.approx(2.5)
    assert "load_project" not in script
