from __future__ import annotations

import json

import numpy as np
import pytest

import nfit.project_gui as project_gui
from nfit import (
    AnalysisEntry,
    AnalysisOutputRef,
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    DatasetGroup,
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
    composite_analysis_source_id,
    create_derived_analysis_dataset,
    create_model_component,
    dataset_entry_from_path,
    dataset_for_slice_viewer,
    dataset_workflow_plan,
    dataset_workflow_script,
    derived_analysis_dataset_data,
    fit_data_bundle,
    fit_workflow_plan,
    fit_workflow_script,
    prepare_analysis_input,
    prepare_analysis_inputs,
    run_project_analysis,
    save_dataset_file,
    upsert_analysis_output_dataset,
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


def test_histogram_arithmetic_analysis_accepts_live_group_composites(tmp_path):
    low = DatasetGroup("Low temperature", datasets=[DatasetEntry("low", _grid(5.0), kind="mdhisto")])
    high = DatasetGroup("50 K", datasets=[DatasetEntry("high", _grid(2.0), kind="mdhisto")])
    group = DataGroup("Experiment", subgroups=[low, high])
    for node in (low, high):
        config = project_gui.data_group_composite_config(
            project_gui._composite_scope(group, node)
        )
        config.update({"enabled": True, "fractional": False})
        for axis_config in config["axes"]:
            axis_config.update({"auto_lower": False, "auto_upper": False})
    analysis = AnalysisEntry(
        "Low minus 50 K",
        "histogram_arithmetic",
        [
            composite_analysis_source_id(group, low),
            composite_analysis_source_id(group, high),
        ],
        {"operation": "subtract", "right_scale": 0.5},
    )
    group.analyses.append(analysis)
    project = NfitProject([group])
    project._project_path = tmp_path / "experiment.nfit"

    inputs = prepare_analysis_inputs(group, analysis)
    execution = run_project_analysis(group, analysis)

    assert [item.dataset_name for item in inputs] == [
        "Low temperature composite",
        "50 K composite",
    ]
    np.testing.assert_allclose(execution.outputs["histogram"].data.signal, _grid(4.0).signal)
    script = analysis_workflow_script(project, analysis.id)
    compile(script, "<live-composite-analysis>", "exec")
    assert "run_project_analysis" in script


def test_dataset_clone_tracks_live_composite_and_defaults_to_comparison_only(tmp_path):
    source = DatasetGroup(
        "1.8 K",
        datasets=[DatasetEntry("run", _grid(5.0), kind="mdhisto")],
    )
    group = DataGroup("Experiment", subgroups=[source])
    config = project_gui.data_group_composite_config(
        project_gui._composite_scope(group, source)
    )
    config.update({"enabled": True, "fractional": False})
    for axis_config in config["axes"]:
        axis_config.update({"auto_lower": False, "auto_upper": False})
    analysis = AnalysisEntry(
        "1.8 K comparison clone",
        "dataset_clone",
        [composite_analysis_source_id(group, source)],
        {},
    )
    group.analyses.append(analysis)
    project = NfitProject([group])
    project._project_path = tmp_path / "experiment.nfit"

    execution = run_project_analysis(group, analysis)
    output = execution.outputs["clone"]

    np.testing.assert_allclose(output.data.signal, _grid(5.0).signal)
    assert output.metadata["fit_enabled"] is False
    assert output.metadata["fit_weight"] == 0.0
    assert output.metadata["clone_source_id"] == composite_analysis_source_id(
        group, source
    )
    script = analysis_workflow_script(project, analysis.id)
    compile(script, "<live-composite-clone>", "exec")


def test_refreshing_analysis_output_preserves_independent_dataset_settings(tmp_path):
    source = DatasetEntry("source", _grid(1.0))
    group = DataGroup("Experiment", datasets=[source])
    analysis = AnalysisEntry("Clone", "dataset_clone", [source.id], {})
    output = AnalysisOutputRef(
        "clone",
        "source clone",
        "dataset",
        artifact_path="assets/analyses/a/clone.npz",
        dataset_id="clone-id",
        metadata={
            "data_type": "powder_inelastic",
            "fit_enabled": False,
            "fit_weight": 0.0,
        },
    )

    clone = upsert_analysis_output_dataset(
        group,
        analysis,
        output,
        _grid(1.0),
        project_path=tmp_path / "experiment.nfit",
    )
    clone.parameters[project_gui.DATASET_REBIN_KEY] = {"enabled": True}
    clone.masks.append(MaskSpec("comparison mask"))
    clone.enabled = True
    clone.fit_weight = 0.25

    refreshed = upsert_analysis_output_dataset(
        group,
        analysis,
        output,
        _grid(2.0),
        project_path=tmp_path / "experiment.nfit",
    )

    assert refreshed is clone
    np.testing.assert_allclose(refreshed.data.signal, _grid(2.0).signal)
    assert refreshed.parameters[project_gui.DATASET_REBIN_KEY] == {"enabled": True}
    assert [mask.name for mask in refreshed.masks] == ["comparison mask"]
    assert refreshed.enabled is True
    assert refreshed.fit_weight == 0.25


def test_live_clone_bins_underlying_points_on_its_own_grid(tmp_path):
    points = PointData4D(
        H=[0.2, 0.8],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[0.0, 0.0],
        intensity=[1.0, 3.0],
        sigma=[1.0, 1.0],
    )
    source = DatasetGroup(
        "1.8 K",
        datasets=[
            DatasetEntry(
                "run",
                points,
                kind="point",
                data_type="single_crystal_inelastic",
            )
        ],
    )
    group = DataGroup("Experiment", subgroups=[source])
    source_config = project_gui.data_group_composite_config(
        project_gui._composite_scope(group, source)
    )
    source_config.update(
        {
            "enabled": True,
            "mean_weighting": "uniform",
            "minimum_coverage": 0.0,
        }
    )
    for index, axis in enumerate(source_config["axes"]):
        axis.update(
            {
                "lower": 0.0 if index == 0 else -0.5,
                "upper": 1.0 if index == 0 else 0.5,
                "auto_lower": False,
                "auto_upper": False,
                "num_bins": 1,
                "step_size": 1.0,
                "mode": "bins",
            }
        )
    coarse = project_gui.composite_dataset_data(
        project_gui._composite_scope(group, source)
    )
    np.testing.assert_allclose(coarse.signal[~coarse.mask], [2.0])
    source.datasets[0].masks.append(
        MaskSpec("Fit-only exclusion", parameters={"H": [0.0, 0.5]})
    )
    masked_source = project_gui.composite_dataset_data(
        project_gui._composite_scope(group, source)
    )
    np.testing.assert_allclose(masked_source.signal[~masked_source.mask], [3.0])

    analysis = AnalysisEntry(
        "1.8 K comparison",
        "dataset_clone",
        [composite_analysis_source_id(group, source)],
        {},
    )
    output_config = json.loads(json.dumps(source_config))
    output_config["axes"][0].update(
        {"num_bins": 2, "step_size": 0.5, "mode": "discrete"}
    )
    clone = create_derived_analysis_dataset(
        group,
        analysis,
        rebin_config=output_config,
    )

    result = derived_analysis_dataset_data(clone)

    assert isinstance(result, MDHistoData)
    np.testing.assert_allclose(result.signal[~result.mask], [1.0, 3.0])
    assert clone.data is None
    assert clone.enabled is False
    assert clone.fit_weight == 0.0
    assert clone.metadata["derived_recipe"]["source_stage"] == "underlying_data"
    project = NfitProject([group])
    project._project_path = tmp_path / "experiment.nfit"
    script = analysis_workflow_script(project, analysis.id)
    compile(script, "<source-linked-clone>", "exec")
    assert "derived_analysis_dataset_data" in script


def test_live_histogram_arithmetic_reduces_both_sources_to_output_grid():
    axis = MDHistoAxis("H", np.array([0.0, 0.5, 1.0]), "rlu", "momentum")

    def histogram(values):
        return MDHistoData(
            (axis,),
            np.asarray(values, dtype=float),
            np.ones(2),
            np.zeros(2, dtype=bool),
            np.ones(2),
        )

    left = DatasetGroup(
        "1.8 K",
        datasets=[DatasetEntry("cold", histogram([1.0, 3.0]), kind="mdhisto")],
    )
    right = DatasetGroup(
        "50 K",
        datasets=[DatasetEntry("warm", histogram([0.5, 1.0]), kind="mdhisto")],
    )
    group = DataGroup("Experiment", subgroups=[left, right])
    for source in (left, right):
        config = project_gui.data_group_composite_config(
            project_gui._composite_scope(group, source)
        )
        config.update(
            {
                "enabled": True,
                "mean_weighting": "uniform",
                "minimum_coverage": 0.0,
            }
        )
        config["axes"][0].update(
            {
                "lower": 0.0,
                "upper": 1.0,
                "auto_lower": False,
                "auto_upper": False,
                "num_bins": 1,
                "step_size": 1.0,
                "mode": "bins",
            }
        )
    output_config = json.loads(
        json.dumps(
            project_gui.data_group_composite_config(
                project_gui._composite_scope(group, left)
            )
        )
    )
    output_config["axes"][0].update(
        {"num_bins": 2, "step_size": 0.5, "mode": "discrete"}
    )
    analysis = AnalysisEntry(
        "Low minus 50 K",
        "histogram_arithmetic",
        [
            composite_analysis_source_id(group, left),
            composite_analysis_source_id(group, right),
        ],
        {"operation": "subtract", "right_scale": 1.0},
    )
    derived = create_derived_analysis_dataset(
        group,
        analysis,
        rebin_config=output_config,
    )

    result = derived_analysis_dataset_data(derived)

    assert isinstance(result, MDHistoData)
    assert result.shape == (2,)
    np.testing.assert_allclose(result.signal, [0.5, 2.0])


def test_live_histogram_arithmetic_resolves_one_grid_from_all_auto_bounds():
    def histogram(edges, values):
        axis = MDHistoAxis("H", np.asarray(edges, dtype=float), "rlu", "momentum")
        values = np.asarray(values, dtype=float)
        return MDHistoData(
            (axis,), values, np.ones_like(values), np.zeros_like(values, dtype=bool),
            np.ones_like(values),
        )

    left = DatasetEntry("left", histogram([-1.25, -0.75, -0.25, 0.25], [1, 2, 3]), kind="mdhisto")
    right = DatasetEntry("right", histogram([-0.25, 0.25, 0.75, 1.25], [4, 5, 6]), kind="mdhisto")
    group = DataGroup("Experiment", datasets=[left, right])
    config = project_gui.dataset_rebin_config(left)
    config.update(enabled=True, minimum_coverage=0.0)
    axis = config["axes"][0]
    axis.update(
        lower=-1.0, upper=0.0, step_size=0.5, num_bins=3, mode="step",
        auto_lower=True, auto_lower_value=-1.0,
        auto_upper=True, auto_upper_value=0.0,
        auto_step_size=False,
    )
    analysis = AnalysisEntry(
        "shared auto grid", "histogram_arithmetic", [left.id, right.id],
        {"operation": "subtract", "right_scale": 1.0},
    )
    derived = create_derived_analysis_dataset(group, analysis, rebin_config=config)

    result = derived_analysis_dataset_data(derived)

    assert result.shape == (7,)
    np.testing.assert_allclose(
        result.axes[0].centers, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )
    assert np.any(result.mask)
    assert np.any(~result.mask)
    assert not result.mask[3]
    assert result.signal[3] == pytest.approx(-1.0)
    assert result.errors[3] == pytest.approx(np.sqrt(2.0))
    saved_axis = project_gui.dataset_rebin_config(derived)["axes"][0]
    assert saved_axis["auto_lower"] and saved_axis["auto_upper"]


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
