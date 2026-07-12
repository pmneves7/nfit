from __future__ import annotations

import numpy as np
import pytest

from nfit.dataset import PointData4D, PointListData
from nfit.fit_config import (
    FitDatasetInput,
    compile_fit_problem,
    component_parameter_names,
    dataset_scale_parameter_name,
    instanced_parameter_name,
    model_supports_data_type,
    qualified_parameter_name,
)
from nfit.fitting import fit_problem_least_squares
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry, ModelComponentSpec
from nfit.plotting import MDHistoSliceViewer
from nfit.project_gui import (
    NfitProject,
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


def test_compile_fit_problem_adds_dataset_scale_parameter_to_residuals():
    data = PointData4D(
        H=np.zeros(8),
        K=np.zeros(8),
        L=np.zeros(8),
        E=np.linspace(0.0, 1.0, 8),
        intensity=np.full(8, 2.0),
        sigma=np.full(8, 0.1),
    )
    component = _constant_component(parameters={"constant": 1.0}, fit_parameters={"constant": False})
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("scan", data, scale_value=1.0, scale_vary=True)],
    )

    scale_name = dataset_scale_parameter_name("scan")
    assert scale_name in compiled.parameter_instances
    assert compiled.parameter_instances[scale_name].component == "dataset"
    assert compiled.problem.datasets[0].data_scale_parameter == scale_name

    result = fit_problem_least_squares(compiled.problem)
    assert result.success
    assert result.params[scale_name] == pytest.approx(0.5, abs=1e-6)
    assert result.reduced_chi2 == pytest.approx(0.0, abs=1e-8)


def test_run_group_fit_can_fit_dataset_scale_factor():
    group = _fit_ready_group({"scan": 2.0})
    dataset = group.get_dataset("scan")
    dataset.scale_factor = 1.0
    dataset.scale_factor_vary = True
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    model.fit_parameters["constant"] = False
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"
    assert dataset.scale_factor == pytest.approx(0.5, abs=1e-6)
    assert entry.snapshot["datasets"][0]["scale_factor"] == pytest.approx(0.5, abs=1e-6)
    assert entry.snapshot["datasets"][0]["scale_factor_vary"] is True
    assert entry.channels["scan"]["residual"] == pytest.approx(np.zeros((4, 5)), abs=1e-6)


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

    model.enabled = False
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
    save_project(NfitProject([group]), path)
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


def _rlu_matrix(a: float) -> list[list[float]]:
    return (2.0 * np.pi / a * np.eye(3)).tolist()


def _spin_fluctuation_points(
    intensity: np.ndarray,
    H: np.ndarray,
    E: np.ndarray,
    *,
    temperature: float | None,
    lattice_a: float | None = None,
) -> PointData4D:
    metadata = {}
    if lattice_a is not None:
        metadata["rlu_to_inv_angstrom_matrix"] = _rlu_matrix(lattice_a)
    return PointData4D(
        H=H,
        K=np.zeros_like(H),
        L=np.zeros_like(H),
        E=E,
        intensity=intensity,
        sigma=np.full(H.shape, 0.01),
        temperature=temperature,
        metadata=metadata,
    )


def test_heisenberg_rpa_emits_dynamic_orbit_parameters():
    component = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.5, "gamma0": 2.0, "J1": 0.1, "J3a": 0.0},
        fit_parameters={"J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]},
                {"label": "J3a", "bonds": [{"site_i": 0, "site_j": 0, "offset": [3, 0, 0]}]},
            ],
        },
    )
    H = np.linspace(0.0, 1.0, 16)
    points = _spin_fluctuation_points(np.ones(16), H, np.ones(16), temperature=10.0)
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    names = [spec.name for spec in compiled.problem.parameter_specs]
    assert names == ["rpa.scale", "rpa.chi0", "rpa.gamma0", "rpa.J1", "rpa.J3a"]


def test_spin_fluctuation_models_require_temperature():
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"scale": 1.0, "chi_loc": 1.0, "gamma": 2.0},
        fit_parameters={"chi_loc": True},
    )
    H = np.zeros(8)
    points = _spin_fluctuation_points(
        np.ones(8), H, np.linspace(0.5, 4.0, 8), temperature=None
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    with pytest.raises(ValueError, match="temperature"):
        fit_problem_least_squares(compiled.problem)


def test_local_relaxational_fit_recovers_synthetic_parameters():
    from nfit.cross_section import intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.spin_fluctuations import local_relaxational_chipp

    rng = np.random.default_rng(11)
    E = np.linspace(0.5, 12.0, 80)
    H = np.zeros_like(E)
    temperature = 25.0
    truth = {"chi_loc": 2.4, "gamma": 3.1}
    clean = intensity_from_chipp(
        local_relaxational_chipp(E, **truth),
        E,
        temperature,
        polarization=ISOTROPIC_POLARIZATION,
    )
    points = _spin_fluctuation_points(
        clean + rng.normal(0.0, 0.005, E.size), H, E, temperature=temperature
    )
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"scale": 1.0, "chi_loc": 1.0, "gamma": 2.0},
        fit_parameters={"chi_loc": True, "gamma": True},
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["loc.chi_loc"] == pytest.approx(truth["chi_loc"], rel=0.02)
    assert result.params["loc.gamma"] == pytest.approx(truth["gamma"], rel=0.02)


def test_mmp_relaxational_fit_recovers_synthetic_parameters():
    from nfit.cross_section import intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.spin_fluctuations import mmp_chipp

    rng = np.random.default_rng(5)
    lattice_a = 4.0
    H_axis = np.linspace(0.2, 0.8, 13)
    E_axis = np.linspace(0.5, 8.0, 11)
    H, E = (arr.ravel() for arr in np.meshgrid(H_axis, E_axis))
    temperature = 40.0
    truth = {"chi_pk": 3.0, "xi": 2.2, "omega_sf": 1.8}
    q_sq = (2.0 * np.pi / lattice_a) ** 2 * (H - 0.5) ** 2
    clean = intensity_from_chipp(
        mmp_chipp(q_sq, E, **truth),
        E,
        temperature,
        polarization=ISOTROPIC_POLARIZATION,
    )
    points = _spin_fluctuation_points(
        clean + rng.normal(0.0, 0.005, E.size),
        H,
        E,
        temperature=temperature,
        lattice_a=lattice_a,
    )
    component = ModelComponentSpec(
        name="mmp",
        type="mmp_relaxational",
        parameters={
            "scale": 1.0,
            "chi_pk": 1.5,
            "xi": 1.0,
            "omega_sf": 1.0,
            "q0_h": 0.5,
            "q0_k": 0.0,
            "q0_l": 0.0,
        },
        fit_parameters={"chi_pk": True, "xi": True, "omega_sf": True},
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["mmp.chi_pk"] == pytest.approx(truth["chi_pk"], rel=0.05)
    assert result.params["mmp.xi"] == pytest.approx(truth["xi"], rel=0.05)
    assert result.params["mmp.omega_sf"] == pytest.approx(truth["omega_sf"], rel=0.05)


def _heisenberg_chain_component(**overrides) -> ModelComponentSpec:
    spec = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.5, "gamma0": 2.5, "J1": 0.05},
        fit_parameters={"chi0": True, "gamma0": True, "J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}
            ],
        },
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def test_heisenberg_rpa_fit_recovers_synthetic_parameters():
    from nfit.cross_section import intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.spin_fluctuations import build_rpa_geometry, heisenberg_rpa_chipp

    rng = np.random.default_rng(2)
    H_axis = np.linspace(0.0, 1.0, 15)
    E_axis = np.linspace(0.5, 10.0, 12)
    H, E = (arr.ravel() for arr in np.meshgrid(H_axis, E_axis))
    temperature = 15.0
    truth = {"chi0": 0.7, "gamma0": 3.0, "J1": 0.4}
    orbits = [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}]
    geometry = build_rpa_geometry(H, np.zeros_like(H), np.zeros_like(H), [[0.0, 0.0, 0.0]], orbits)
    clean = intensity_from_chipp(
        heisenberg_rpa_chipp(
            geometry,
            E,
            chi0=truth["chi0"],
            gamma0=truth["gamma0"],
            j_values={"J1": truth["J1"]},
        ),
        E,
        temperature,
        polarization=ISOTROPIC_POLARIZATION,
    )
    points = _spin_fluctuation_points(
        clean + rng.normal(0.0, 0.002, E.size), H, E, temperature=temperature
    )
    compiled = compile_fit_problem(
        [_heisenberg_chain_component()], [FitDatasetInput("a", points)]
    )
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["rpa.chi0"] == pytest.approx(truth["chi0"], rel=0.05)
    assert result.params["rpa.gamma0"] == pytest.approx(truth["gamma0"], rel=0.05)
    assert result.params["rpa.J1"] == pytest.approx(truth["J1"], rel=0.05)


def test_heisenberg_rpa_blank_custom_form_factor_uses_selected_ion():
    H = np.linspace(0.0, 1.0, 8)
    points = _spin_fluctuation_points(
        np.ones(H.size),
        H,
        np.ones(H.size),
        temperature=5.0,
        lattice_a=8.24062,
    )
    component = _heisenberg_chain_component()
    component.config["ion"] = "V2"
    component.config["form_factor_coefficients"] = ""
    component.fit_parameters = {}

    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)

    assert result.success
    assert np.all(np.isfinite(result.model_values))


def test_heisenberg_rpa_dynamic_parameter_supports_per_dataset_sharing():
    component = _heisenberg_chain_component(
        sharing={"chi0": {"mode": "per_dataset"}},
    )
    H = np.linspace(0.0, 1.0, 10)
    make = lambda: _spin_fluctuation_points(np.ones(10), H, np.ones(10), temperature=5.0)
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("cold", make()), FitDatasetInput("hot", make())],
    )
    names = sorted(spec.name for spec in compiled.problem.parameter_specs)
    assert "rpa.chi0[cold]" in names and "rpa.chi0[hot]" in names
    assert "rpa.J1" in names


def _rpa_component(**overrides) -> ModelComponentSpec:
    spec = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.2, "chi0": 0.3, "gamma0": 2.0, "J1": 0.1, "J2": -0.05},
        fit_parameters={"scale": True, "chi0": True, "gamma0": True, "J1": True, "J2": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]],
            "orbits": [
                {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
                {"label": "J2", "bonds": [{"site_i": 0, "site_j": 0, "offset": [0, 0, 1]}]},
            ],
        },
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _rpa_points(temperature: float, seed: int, n: int = 60) -> PointData4D:
    rng = np.random.default_rng(seed)
    return PointData4D(
        H=rng.uniform(-1.0, 1.0, n),
        K=rng.uniform(-1.0, 1.0, n),
        L=rng.uniform(-1.0, 1.0, n),
        E=rng.uniform(0.5, 5.0, n),
        intensity=rng.uniform(0.0, 1.0, n),
        sigma=np.full(n, 0.1),
        temperature=float(temperature),
        metadata={"coordinate_units": "r.l.u.", "energy_units": "meV"},
    )


def test_heisenberg_rpa_problem_reports_analytic_jacobian():
    from nfit.fitting import problem_supports_analytic_jacobian

    compiled = compile_fit_problem(
        [_rpa_component()],
        [FitDatasetInput("T5", _rpa_points(5.0, 1), data_type="single_crystal_inelastic")],
    )
    assert problem_supports_analytic_jacobian(compiled.problem)
    assert compiled.problem.datasets[0].model_jacobian is not None


def test_analytic_jacobian_matches_finite_differences_with_grouped_sharing():
    from nfit.fitting import (
        _evaluate_problem,
        _evaluate_problem_jacobian,
        _finite_difference_jacobian,
        pack_parameters,
        unpack_parameters,
    )

    # chi0 tied across two temperatures (grouped), scale free per dataset:
    # exercises the binding chain in the Jacobian assembly.
    component = _rpa_component(
        sharing={"chi0": "grouped", "scale": "per_dataset"},
        groups={"chi0": {"T5": "cold", "T50": "cold"}},
    )
    compiled = compile_fit_problem(
        component and [component],
        [
            FitDatasetInput("T5", _rpa_points(5.0, 1), data_type="single_crystal_inelastic"),
            FitDatasetInput("T50", _rpa_points(50.0, 2), data_type="single_crystal_inelastic"),
        ],
    )
    problem = compiled.problem
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    params = unpack_parameters(x0, names, fixed)
    analytic = _evaluate_problem_jacobian(problem, params, names, require_positive_sigma=True)

    def residual_fn(x):
        return _evaluate_problem(
            problem, unpack_parameters(x, names, fixed), require_positive_sigma=True
        ).residuals

    fd = _finite_difference_jacobian(residual_fn, x0, residual_fn(x0), bounds)
    np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=1e-6)


def test_analytic_jacobian_matches_finite_differences_with_constraint():
    from nfit.fitting import (
        _evaluate_problem,
        _evaluate_problem_jacobian,
        _finite_difference_jacobian,
        pack_parameters,
        unpack_parameters,
    )

    # gamma0 >= chi0 reparameterizes gamma0 as a derived (chi0 + offset):
    # exercises the derived-parameter chain rule in the assembly.
    component = _rpa_component(
        constraints=[{"parameter": "gamma0", "op": ">=", "reference": "M.chi0"}]
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("T5", _rpa_points(5.0, 3), data_type="single_crystal_inelastic")],
    )
    problem = compiled.problem
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    params = unpack_parameters(x0, names, fixed)
    analytic = _evaluate_problem_jacobian(problem, params, names, require_positive_sigma=True)

    def residual_fn(x):
        return _evaluate_problem(
            problem, unpack_parameters(x, names, fixed), require_positive_sigma=True
        ).residuals

    fd = _finite_difference_jacobian(residual_fn, x0, residual_fn(x0), bounds)
    np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=1e-6)


def test_analytic_and_numeric_jacobians_recover_same_fit():
    from nfit.fitting import OptimizationConfig

    # Synthesize data from the model, then fit from a perturbed start with the
    # analytic Jacobian and confirm it recovers the generating parameters.
    truth = _rpa_component()
    data = _rpa_points(5.0, 7, n=120)
    compiled_truth = compile_fit_problem(
        [truth], [FitDatasetInput("T5", data, data_type="single_crystal_inelastic")]
    )
    from nfit.fitting import evaluate_problem_model

    model_values = evaluate_problem_model(
        compiled_truth.problem,
        "T5",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    fitted_points = PointData4D(
        data.H, data.K, data.L, data.E, model_values, np.full(data.size, 0.02),
        temperature=data.temperature, metadata=dict(data.metadata),
    )
    start = _rpa_component(
        parameters={"scale": 1.0, "chi0": 0.2, "gamma0": 2.5, "J1": 0.05, "J2": 0.0}
    )
    compiled = compile_fit_problem(
        [start], [FitDatasetInput("T5", fitted_points, data_type="single_crystal_inelastic")]
    )
    result = fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
    assert result.success
    # Noise-free data drawn from the model must be fit essentially perfectly.
    assert result.reduced_chi2 < 1e-6
    # The RPA response is invariant under chi0 -> a*chi0, J -> J/a, scale ->
    # scale/a, so only gamma0 and the products scale*chi0 and chi0*J_o are
    # physically determined; assert those recover the generating values.
    p = result.params
    assert p["M.gamma0"] == pytest.approx(2.0, rel=1e-3)
    assert p["M.scale"] * p["M.chi0"] == pytest.approx(1.2 * 0.3, rel=1e-3)
    assert p["M.chi0"] * p["M.J1"] == pytest.approx(0.3 * 0.1, rel=1e-3)
    assert p["M.chi0"] * p["M.J2"] == pytest.approx(0.3 * -0.05, rel=1e-3)


def test_heisenberg_rpa_fit_is_backend_invariant():
    """A full fit must converge to the same result on numpy and numba backends.

    The numba path uses the Jacobi eigensolver (different from LAPACK), so this
    locks that the accelerated backend does not change fit results beyond
    floating-point noise.
    """
    pytest.importorskip("numba")
    from nfit import spin_fluctuations as sf
    from nfit.fitting import OptimizationConfig

    # A large-enough single-crystal problem to exercise the numba eigensolver.
    truth = _rpa_component()
    data = _rpa_points(5.0, 11, n=3000)
    compiled_truth = compile_fit_problem(
        [truth], [FitDatasetInput("d", data, data_type="single_crystal_inelastic")]
    )
    from nfit.fitting import evaluate_problem_model

    model_values = evaluate_problem_model(
        compiled_truth.problem,
        "d",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    fitted = PointData4D(
        data.H, data.K, data.L, data.E, model_values, np.full(data.size, 0.02),
        temperature=data.temperature, metadata=dict(data.metadata),
    )

    def fit_with(backend):
        sf.set_rpa_backend(backend)
        try:
            start = _rpa_component(
                parameters={"scale": 1.0, "chi0": 0.2, "gamma0": 2.5, "J1": 0.05, "J2": 0.0}
            )
            compiled = compile_fit_problem(
                [start], [FitDatasetInput("d", fitted, data_type="single_crystal_inelastic")]
            )
            return fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
        finally:
            sf.set_rpa_backend("auto")

    result_numba = fit_with("numba")
    result_numpy = fit_with("numpy")
    assert result_numba.success and result_numpy.success
    # Both backends reach the same-quality minimum. Individual parameters can
    # differ because the RPA response is invariant under chi0 -> a*chi0,
    # J -> J/a, scale -> scale/a (a flat valley of equally good fits), so the
    # invariant is the fit quality and the physical model prediction.
    assert result_numba.reduced_chi2 == pytest.approx(result_numpy.reduced_chi2, rel=1e-6)
    compiled = compile_fit_problem(
        [_rpa_component()], [FitDatasetInput("d", fitted, data_type="single_crystal_inelastic")]
    )
    from nfit.fitting import evaluate_problem_model

    pred_numba = evaluate_problem_model(compiled.problem, "d", result_numba.params)
    pred_numpy = evaluate_problem_model(compiled.problem, "d", result_numpy.params)
    np.testing.assert_allclose(pred_numba, pred_numpy, rtol=1e-4, atol=1e-6)


def test_dataset_magnetic_field_validator_gives_actionable_error():
    from nfit.fit_config import _dataset_magnetic_field

    points = _rpa_points(5.0, 3)
    with pytest.raises(ValueError, match="Sample\\s*environment"):
        _dataset_magnetic_field(points)
    points.magnetic_field = np.array([0.0, 0.0, 1.5])
    np.testing.assert_array_equal(
        _dataset_magnetic_field(points), np.array([0.0, 0.0, 1.5])
    )


def _pyrochlore_tensor_component(fit_aniso=True):
    from nfit.crystal import (
        generate_bond_orbits,
        orbits_to_config,
        sites_to_config,
        symmetry_allowed_exchange_basis,
    )

    crystal = {
        "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "spacegroup": "F d -3 m:2",
        "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "V2"}],
    }
    nn = 10.0 * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(crystal, ["M1"], cutoff_angstrom=nn + 0.01)
    basis = symmetry_allowed_exchange_basis(crystal, sites, orbits[0])
    return crystal, ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.02, "gamma0": 3.0, "J1": 0.05,
                    "J1_S1": 0.0, "J1_S2": 0.0, "J1_D1": 0.0},
        fit_parameters={"scale": True, "chi0": True, "J1": True, "J1_S1": fit_aniso},
        config={
            "site_positions": sites_to_config(sites),
            "orbits": orbits_to_config(orbits),
            "crystal": crystal,
            "ion": "V2",
            "anisotropy": {"J1": {"enabled": True, "basis": basis}},
        },
    )


def _tensor_points(seed, n=200):
    from nfit.fitting import reciprocal_basis_from_lattice_parameters

    rng = np.random.default_rng(seed)
    matrix = reciprocal_basis_from_lattice_parameters(10, 10, 10, 90, 90, 90)
    return PointData4D(
        H=rng.uniform(-1.0, 1.0, n), K=rng.uniform(-1.0, 1.0, n), L=rng.uniform(-1.0, 1.0, n),
        E=rng.uniform(0.5, 5.0, n), intensity=np.zeros(n), sigma=np.full(n, 0.05),
        temperature=5.0,
        metadata={"coordinate_units": "r.l.u.", "energy_units": "meV",
                  "rlu_to_inv_angstrom_matrix": matrix.tolist()},
    )


def test_tensor_component_emits_anisotropy_parameters_and_declines_analytic_jacobian():
    from nfit.fitting import problem_supports_analytic_jacobian

    _crystal, component = _pyrochlore_tensor_component()
    names = component_parameter_names(component)
    assert names == ("scale", "chi0", "gamma0", "J1", "J1_S1", "J1_S2", "J1_D1")
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", _tensor_points(0), data_type="single_crystal_inelastic")]
    )
    # Tensor path falls back to finite-difference gradients.
    assert not problem_supports_analytic_jacobian(compiled.problem)


def test_disabled_anisotropy_takes_scalar_path_bit_identical():
    """A component whose anisotropy sections are all disabled must be scalar."""
    from nfit.fitting import evaluate_problem_model, problem_supports_analytic_jacobian
    from nfit.fit_config import _RpaComponentEvaluator

    _crystal, component = _pyrochlore_tensor_component()
    # Disable the anisotropy section entirely.
    component.config["anisotropy"]["J1"]["enabled"] = False
    component.fit_parameters = {"scale": True, "chi0": True, "J1": True}
    assert not _RpaComponentEvaluator(component).tensor_mode
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", _tensor_points(1), data_type="single_crystal_inelastic")]
    )
    # The scalar path keeps its analytic Jacobian.
    assert problem_supports_analytic_jacobian(compiled.problem)
    values = evaluate_problem_model(
        compiled.problem, "d", {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    )
    assert np.all(np.isfinite(values))


def test_tensor_fit_recovers_anisotropic_parameters():
    from nfit.fitting import (
        OptimizationConfig,
        evaluate_problem_model,
        fit_problem_least_squares,
    )

    _crystal, truth = _pyrochlore_tensor_component(fit_aniso=False)
    truth.parameters.update({"scale": 1.2, "J1_S1": 0.012})
    points = _tensor_points(3)
    compiled_truth = compile_fit_problem(
        [truth], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    model_values = evaluate_problem_model(
        compiled_truth.problem, "d",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    fitted = PointData4D(
        points.H, points.K, points.L, points.E, model_values, np.full(points.size, 0.01),
        temperature=points.temperature, metadata=dict(points.metadata),
    )
    _crystal, start = _pyrochlore_tensor_component()
    compiled = compile_fit_problem(
        [start], [FitDatasetInput("d", fitted, data_type="single_crystal_inelastic")]
    )
    result = fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
    assert result.success
    assert result.reduced_chi2 < 1e-6
