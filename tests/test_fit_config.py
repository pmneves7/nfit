from __future__ import annotations

import numpy as np
import pytest

from metallix.dataset import PointData4D, PointListData
from metallix.fit_config import (
    FitDatasetInput,
    compile_fit_problem,
    instanced_parameter_name,
    model_supports_data_type,
    qualified_parameter_name,
)
from metallix.fitting import fit_problem_least_squares
from metallix.mdhisto import MDHistoAxis, MDHistoData
from metallix.pipeline import DataGroup, DatasetEntry, ModelComponentSpec
from metallix.plotting import MDHistoSliceViewer
from metallix.project_gui import (
    MetallixProject,
    attach_fit_channels_to_view,
    create_mask,
    create_model_component,
    dataset_for_slice_viewer,
    ensure_fit_history,
    fit_data_bundle,
    latest_fit_channels,
    load_project,
    perform_group_fit,
    run_group_fit,
    save_project,
    slice_viewer_datasets,
)


RNG = np.random.default_rng(7)


def _points(level: float, n: int = 64) -> PointData4D:
    return PointData4D(
        H=np.zeros(n),
        K=np.zeros(n),
        L=np.zeros(n),
        E=np.linspace(0.0, 10.0, n),
        intensity=level + RNG.normal(0.0, 0.01, n),
        sigma=np.full(n, 0.01),
    )


def _constant_component(**overrides) -> ModelComponentSpec:
    spec = ModelComponentSpec(
        name="bg",
        type="constant_background",
        parameters={"constant": 0.5},
        fit_parameters={"constant": True},
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _grid_mdhisto(signal: np.ndarray) -> MDHistoData:
    ny, nx = signal.shape
    axis_h = MDHistoAxis("H", np.linspace(0.0, 1.0, ny + 1), "rlu", "momentum")
    axis_e = MDHistoAxis("E", np.linspace(0.0, 10.0, nx + 1), "meV", "energy")
    return MDHistoData(
        axes=(axis_h, axis_e),
        signal=signal,
        errors=np.full_like(signal, 0.1),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )


def test_global_sharing_emits_one_parameter():
    compiled = compile_fit_problem(
        [_constant_component()],
        [FitDatasetInput("a", _points(1.0)), FitDatasetInput("b", _points(3.0))],
    )
    names = [spec.name for spec in compiled.problem.parameter_specs]
    assert names == ["bg.constant"]
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["bg.constant"] == pytest.approx(2.0, abs=0.05)


def test_per_dataset_sharing_emits_one_parameter_per_dataset():
    component = _constant_component(sharing={"constant": {"mode": "per_dataset"}})
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("a", _points(1.0)), FitDatasetInput("b", _points(3.0))],
    )
    names = sorted(spec.name for spec in compiled.problem.parameter_specs)
    assert names == ["bg.constant[a]", "bg.constant[b]"]
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["bg.constant[a]"] == pytest.approx(1.0, abs=0.05)
    assert result.params["bg.constant[b]"] == pytest.approx(3.0, abs=0.05)


def test_grouped_sharing_ties_named_datasets_and_frees_the_rest():
    component = _constant_component(
        sharing={"constant": {"mode": "grouped", "groups": {"a": "low", "b": "low"}}}
    )
    compiled = compile_fit_problem(
        [component],
        [
            FitDatasetInput("a", _points(1.0)),
            FitDatasetInput("b", _points(3.0)),
            FitDatasetInput("c", _points(7.0)),
        ],
    )
    names = sorted(spec.name for spec in compiled.problem.parameter_specs)
    assert names == ["bg.constant[c]", "bg.constant[low]"]
    instances = compiled.instances_for("bg", "constant")
    by_scope = {instance.scope: instance.datasets for instance in instances}
    assert by_scope == {"low": ("a", "b"), "c": ("c",)}
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["bg.constant[low]"] == pytest.approx(2.0, abs=0.05)
    assert result.params["bg.constant[c]"] == pytest.approx(7.0, abs=0.05)


def test_limits_and_vary_flow_into_parameter_specs():
    component = _constant_component(
        limits={"constant": [0.0, 2.0]},
        fit_parameters={"constant": False},
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", _points(1.0))])
    (spec,) = compiled.problem.parameter_specs
    assert (spec.min, spec.max, spec.vary) == (0.0, 2.0, False)


def test_bound_pins_fit_at_limit():
    component = _constant_component(limits={"constant": [2.5, None]})
    component.parameters["constant"] = 3.0
    compiled = compile_fit_problem([component], [FitDatasetInput("a", _points(1.0))])
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["bg.constant"] == pytest.approx(2.5, abs=1e-6)


def test_constraint_reparameterizes_and_holds():
    component = _constant_component(
        constraints=[{"parameter": "constant", "op": ">=", "reference": 2.5}]
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", _points(1.0))])
    names = {spec.name for spec in compiled.problem.parameter_specs}
    assert names == {"bg.constant__offset"}
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["bg.constant"] == pytest.approx(2.5, abs=1e-6)
    assert result.params["bg.constant__offset"] == pytest.approx(0.0, abs=1e-6)


def test_constraint_between_two_component_parameters():
    floor = ModelComponentSpec(
        name="floor",
        type="constant_background",
        parameters={"constant": 2.0},
        fit_parameters={"constant": False},
    )
    top = ModelComponentSpec(
        name="top",
        type="constant_background",
        parameters={"constant": 0.0},
        fit_parameters={"constant": True},
        applies_to=["a"],
        constraints=[{"parameter": "constant", "op": ">=", "reference": "floor.constant"}],
    )
    compiled = compile_fit_problem(
        [floor, top],
        [FitDatasetInput("a", _points(4.0)), FitDatasetInput("b", _points(2.0))],
    )
    result = fit_problem_least_squares(compiled.problem)
    # dataset b is fit by "floor" alone at 2.0; on dataset a the extra "top"
    # component must stay at or above the shared floor value.
    assert result.params["top.constant"] >= result.params["floor.constant"] - 1e-9
    assert result.params["top.constant"] == pytest.approx(2.0, abs=0.05)


def test_constraint_requires_global_mode():
    component = _constant_component(
        sharing={"constant": {"mode": "per_dataset"}},
        constraints=[{"parameter": "constant", "op": ">=", "reference": 1.0}],
    )
    with pytest.raises(ValueError, match="global sharing"):
        compile_fit_problem([component], [FitDatasetInput("a", _points(1.0))])


def test_applies_to_and_data_type_gating_skip_datasets():
    assert model_supports_data_type("constant_background", "magnetization")
    assert not model_supports_data_type("single_q_paramagnon", "magnetization")
    only_b = _constant_component(applies_to=["b"])
    compiled = compile_fit_problem(
        [only_b],
        [FitDatasetInput("a", _points(1.0)), FitDatasetInput("b", _points(3.0))],
    )
    assert compiled.skipped_datasets == ["a"]
    assert [dataset.name for dataset in compiled.problem.datasets] == ["b"]
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["bg.constant"] == pytest.approx(3.0, abs=0.05)


def test_component_matching_no_dataset_emits_no_parameters():
    paramagnon = ModelComponentSpec(
        name="pm",
        type="single_q_paramagnon",
        parameters={
            "amplitude": 1.0,
            "q0_h": 0.0,
            "q0_k": 0.0,
            "q0_l": 0.0,
            "kappa": 1.0,
            "omega_sf": 1.0,
        },
        fit_parameters={"amplitude": True},
    )
    compiled = compile_fit_problem(
        [paramagnon, _constant_component()],
        [FitDatasetInput("m", _points(1.0), data_type="magnetization")],
    )
    names = [spec.name for spec in compiled.problem.parameter_specs]
    assert names == ["bg.constant"]


def test_disabled_component_is_ignored():
    disabled = _constant_component()
    disabled.enabled = False
    with pytest.raises(ValueError, match="no dataset is matched"):
        compile_fit_problem([disabled], [FitDatasetInput("a", _points(1.0))])


def test_qualified_and_instanced_names():
    assert qualified_parameter_name("bg", "constant") == "bg.constant"
    assert instanced_parameter_name("bg", "constant", "low") == "bg.constant[low]"


def _fit_ready_group(levels: dict[str, float]) -> DataGroup:
    group = DataGroup("g")
    for name, level in levels.items():
        signal = np.full((4, 5), level)
        group.datasets.append(DatasetEntry(name, _grid_mdhisto(signal)))
    return group


def test_run_group_fit_recovers_constant_and_stores_channels():
    group = _fit_ready_group({"first": 1.5})
    model = create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"
    assert model.parameters["constant"] == pytest.approx(1.5, abs=1e-6)
    channels = entry.channels["first"]
    assert channels["kind"] == "grid"
    assert channels["fit"].shape == (4, 5)
    assert channels["fit"] == pytest.approx(np.full((4, 5), 1.5), abs=1e-6)
    assert channels["residual"] == pytest.approx(np.zeros((4, 5)), abs=1e-5)
    # the snapshot captures the fitted value so restoring reproduces the fit
    assert entry.snapshot["models"][0]["parameters"]["constant"] == pytest.approx(1.5, abs=1e-6)


def test_run_group_fit_honors_masks_disabled_datasets_and_scales():
    group = _fit_ready_group({"first": 1.0, "second": 9.0, "third": 2.0})
    group.get_dataset("second").enabled = False
    third = group.get_dataset("third")
    third.scale_factor = 2.0  # data at 2.0 scaled to 4.0

    # mask out the high-H half of "first" after poisoning it
    first = group.get_dataset("first")
    poisoned = np.full((4, 5), 1.0)
    poisoned[2:, :] = 100.0
    first.data = _grid_mdhisto(poisoned)
    mask = create_mask(first)
    mask.parameters["H"] = [0.0, 0.49]

    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    model.sharing["constant"] = {"mode": "per_dataset"}
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"
    fitted = model.metadata["fitted_values"]["constant"]
    assert set(fitted) == {"first", "third"}
    assert fitted["first"] == pytest.approx(1.0, abs=1e-6)
    assert fitted["third"] == pytest.approx(4.0, abs=1e-6)
    assert "second" not in entry.channels


def test_run_group_fit_weights_change_global_compromise():
    group = _fit_ready_group({"low": 1.0, "high": 3.0})
    group.get_dataset("high").fit_weight = 1e6
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"
    assert model.parameters["constant"] == pytest.approx(3.0, abs=0.01)


def test_run_group_fit_failure_is_recorded_not_raised():
    group = _fit_ready_group({"first": 1.0})
    ensure_fit_history(group)
    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "failed"
    assert "model" in entry.goodness["message"]
    assert entry.channels == {}


def test_fit_channels_attach_to_mdhisto_view_and_viewer_channels():
    group = _fit_ready_group({"first": 1.5})
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)
    run_group_fit(group, group.fits[0])

    assert latest_fit_channels(group, "first") is not None
    datasets, names = slice_viewer_datasets(group)
    view = datasets[names.index("first")]
    assert view.metadata["fit"].shape == view.shape
    assert view.metadata["residual"].shape == view.shape

    viewer = MDHistoSliceViewer(view)
    assert "fit" in viewer.CHANNELS and "residual" in viewer.CHANNELS
    viewer.channel = "fit"
    arrays = viewer.slice_arrays()
    assert arrays["fit"] == pytest.approx(np.full((4, 5), 1.5), abs=1e-6)


def test_stale_fit_channels_are_skipped_after_shape_change():
    group = _fit_ready_group({"first": 1.5})
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)
    run_group_fit(group, group.fits[0])

    group.get_dataset("first").data = _grid_mdhisto(np.full((3, 3), 1.5))
    datasets, names = slice_viewer_datasets(group)
    view = datasets[names.index("first")]
    assert "fit" not in view.metadata


def test_fit_channels_attach_to_point_list_view():
    columns = {
        "T": np.linspace(1.0, 10.0, 8),
        "M": np.full(8, 2.0),
        "dM": np.full(8, 0.1),
    }
    data = PointListData(
        columns=columns,
        coordinate_names=["T"],
        channels=[{"label": "M", "value": "M", "error": "dM"}],
    )
    dataset = DatasetEntry("mag", data, data_type="magnetization")
    group = DataGroup("g", datasets=[dataset])
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"
    assert model.parameters["constant"] == pytest.approx(2.0, abs=1e-6)
    assert entry.channels["mag"]["kind"] == "points"

    view = dataset_for_slice_viewer(dataset)
    attach_fit_channels_to_view(group, "mag", view)
    assert "fit" in view.channel_labels
    assert view.channel_values("fit") == pytest.approx(np.full(8, 2.0), abs=1e-6)
    assert "residual" in view.channel_labels


def test_fit_bundle_masks_invalid_and_masked_points():
    group = _fit_ready_group({"first": 1.0})
    dataset = group.get_dataset("first")
    mask = create_mask(dataset)
    mask.parameters["E"] = [0.0, 4.9]
    bundle = fit_data_bundle(group, dataset)
    assert bundle.grid_shape == (4, 5)
    assert bundle.points.size == 20
    kept = int(np.count_nonzero(bundle.points.mask))
    assert kept < 20
    assert np.all(bundle.points.E[np.asarray(bundle.points.mask, dtype=bool)] <= 4.9)


def test_project_round_trip_preserves_fit_channels_and_model_fields(tmp_path):
    group = _fit_ready_group({"first": 1.5})
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    model.sharing["constant"] = {"mode": "grouped", "groups": {"first": "low"}}
    model.limits["constant"] = [0.0, 10.0]
    model.constraints = []
    model.applies_to = ["first"]
    ensure_fit_history(group)
    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"

    # replace in-memory data with a lazy source reference so the project saves
    dataset = group.get_dataset("first")
    dataset.data = None
    dataset.metadata["source_file"] = str(tmp_path / "missing.nxs")

    path = tmp_path / "project.json"
    save_project(MetallixProject([group]), path)
    loaded = load_project(path)
    loaded_group = loaded.data_groups[0]
    loaded_model = next(iter(loaded_group.models.values()))
    assert loaded_model.sharing["constant"]["mode"] == "grouped"
    assert loaded_model.sharing["constant"]["groups"] == {"first": "low"}
    assert loaded_model.limits["constant"] == [0.0, 10.0]
    assert loaded_model.applies_to == ["first"]

    payload = latest_fit_channels(loaded_group, "first")
    assert payload is not None
    stored = np.asarray(payload["fit"], dtype=float)
    assert stored.shape == (4, 5)
    assert stored == pytest.approx(np.full((4, 5), 1.5), rel=1e-6)
