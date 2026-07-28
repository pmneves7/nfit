# ruff: noqa: F401, F403, F405
from tests.fit_config_test_support import *
from tests.fit_config_test_support import (
    _closure_points,
    _closure_scalar_component,
    _constant_component,
    _evaluate,
    _fit_ready_group,
    _grid_mdhisto,
    _heisenberg_chain_component,
    _magnetization_component,
    _magnetization_points,
    _points,
    _pyrochlore_tensor_component,
    _rlu_matrix,
    _rpa_component,
    _rpa_points,
    _spin_fluctuation_points,
    _tensor_points,
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


def test_exact_expression_constraint_removes_dependent_parameter_and_sets_dof():
    first = ModelComponentSpec(
        name="first",
        type="constant_background",
        parameters={"constant": 1.0},
        fit_parameters={"constant": True},
    )
    second = ModelComponentSpec(
        name="second",
        type="constant_background",
        parameters={"constant": 2.0},
        fit_parameters={"constant": True},
        constraints=[
            {
                "parameter": "constant",
                "op": "=",
                "expression": "10 / `first.constant`",
            }
        ],
    )
    compiled = compile_fit_problem([first, second], [FitDatasetInput("a", _points(6.0, n=12))])

    assert [spec.name for spec in compiled.problem.parameter_specs] == ["first.constant"]
    result = fit_problem_least_squares(compiled.problem)

    assert result.params["second.constant"] == pytest.approx(10.0 / result.params["first.constant"])
    assert result.reduced_chi2 == pytest.approx(result.chi2 / 11.0)


def test_exact_constraints_can_depend_on_other_exact_constraints():
    components = [
        ModelComponentSpec(
            name=name,
            type="constant_background",
            parameters={"constant": value},
            fit_parameters={"constant": True},
            constraints=constraints,
        )
        for name, value, constraints in (
            ("a", 1.0, []),
            ("b", 2.0, [{"parameter": "constant", "op": "=", "expression": "2 * `a.constant`"}]),
            ("c", 3.0, [{"parameter": "constant", "op": "=", "expression": "`b.constant` + 1"}]),
        )
    ]
    compiled = compile_fit_problem(components, [FitDatasetInput("scan", _points(7.0))])
    resolved = compiled.problem.resolve_parameters({"a.constant": 2.0})

    assert resolved["b.constant"] == pytest.approx(4.0)
    assert resolved["c.constant"] == pytest.approx(5.0)


def test_exact_constraint_rejects_cycles():
    first = ModelComponentSpec(
        name="a",
        type="constant_background",
        parameters={"constant": 1.0},
        fit_parameters={"constant": True},
        constraints=[{"parameter": "constant", "op": "=", "expression": "`b.constant`"}],
    )
    second = ModelComponentSpec(
        name="b",
        type="constant_background",
        parameters={"constant": 1.0},
        fit_parameters={"constant": True},
        constraints=[{"parameter": "constant", "op": "=", "expression": "`a.constant`"}],
    )

    with pytest.raises(ValueError, match="cyclic derived parameter constraints"):
        compile_fit_problem([first, second], [FitDatasetInput("scan", _points(2.0))])


def test_exact_constraint_with_no_independent_variables_uses_all_points_for_dof():
    fixed = ModelComponentSpec(
        name="fixed",
        type="constant_background",
        parameters={"constant": 1.0},
        fit_parameters={"constant": False},
    )
    dependent = ModelComponentSpec(
        name="dependent",
        type="constant_background",
        parameters={"constant": 1.0},
        fit_parameters={"constant": True},
        constraints=[{"parameter": "constant", "op": "=", "expression": "`fixed.constant`"}],
    )
    compiled = compile_fit_problem(
        [fixed, dependent], [FitDatasetInput("scan", _points(2.1, n=10))]
    )
    result = fit_problem_least_squares(compiled.problem)

    assert result.variable_names == []
    assert result.reduced_chi2 == pytest.approx(result.chi2 / 10.0)


def test_constraint_requires_global_mode():
    component = _constant_component(
        sharing={"constant": {"mode": "per_dataset"}},
        constraints=[{"parameter": "constant", "op": ">=", "reference": 1.0}],
    )
    with pytest.raises(ValueError, match="global sharing"):
        compile_fit_problem([component], [FitDatasetInput("a", _points(1.0))])


def test_applies_to_and_data_type_gating_skip_datasets():
    assert model_supports_data_type("constant_background", "magnetization")
    assert not model_supports_data_type("mmp_relaxational", "magnetization")
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
    mmp = ModelComponentSpec(
        name="mmp",
        type="mmp_relaxational",
        parameters={
            "scale": 1.0,
            "chi_pk": 1.0,
            "xi": 1.0,
            "omega_sf": 1.0,
            "q0_h": 0.5,
            "q0_k": 0.0,
            "q0_l": 0.0,
        },
        fit_parameters={"chi_pk": True},
    )
    compiled = compile_fit_problem(
        [mmp, _constant_component()],
        [FitDatasetInput("m", _points(1.0), data_type="magnetization")],
    )
    names = [spec.name for spec in compiled.problem.parameter_specs]
    assert names == ["bg.constant"]


def test_physical_model_parameters_receive_default_nonnegative_bounds():
    component = ModelComponentSpec(
        name="local",
        type="local_relaxational",
        parameters={"scale": 1.0, "chi_loc": 1.0, "gamma": 2.0},
        fit_parameters={"scale": True, "chi_loc": True, "gamma": True},
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("scan", _points(1.0))],
    )

    assert {spec.name: spec.min for spec in compiled.problem.parameter_specs} == {
        "local.scale": 0.0,
        "local.chi_loc": 0.0,
        "local.gamma": 0.0,
    }


def test_disabled_component_is_ignored():
    disabled = _constant_component()
    disabled.enabled = False
    with pytest.raises(ValueError, match="no dataset is matched"):
        compile_fit_problem([disabled], [FitDatasetInput("a", _points(1.0))])


def test_qualified_and_instanced_names():
    assert qualified_parameter_name("bg", "constant") == "bg.constant"
    assert instanced_parameter_name("bg", "constant", "low") == "bg.constant[low]"


def test_run_group_fit_recovers_constant_and_stores_channels():
    group = _fit_ready_group({"first": 1.5})
    model = create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] == "converged"
    assert model.parameters["constant"] == pytest.approx(1.5, abs=1e-6)
    # Per-dataset point counts are persisted for the fit report.
    assert entry.goodness["dataset_n_points"] == {"first": 20}
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
    mask.parameters["H"] = [0.5, 1.0]

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
    assert entry.channels["second"]["visualization_only"] is True


def test_compile_fit_problem_adds_dataset_scale_parameter_to_residuals():
    data = PointData4D(
        H=np.zeros(8),
        K=np.zeros(8),
        L=np.zeros(8),
        E=np.linspace(0.0, 1.0, 8),
        intensity=np.full(8, 2.0),
        sigma=np.full(8, 0.1),
    )
    component = _constant_component(
        parameters={"constant": 1.0}, fit_parameters={"constant": False}
    )
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


def test_zero_weight_dataset_is_evaluated_only_for_post_fit_visualization():
    from nfit.project_gui import (
        _fit_channels_from_dict,
        _fit_channels_to_dict,
        fit_dataset_inputs,
    )

    group = _fit_ready_group({"fit_octant": 1.0, "full_volume": 9.0})
    group.get_dataset("full_volume").fit_weight = 0.0
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    fit_inputs, fit_bundles = fit_dataset_inputs(group, purpose="fit")
    view_inputs, view_bundles = fit_dataset_inputs(group, purpose="visualization")
    assert [item.name for item in fit_inputs] == ["fit_octant"]
    assert list(fit_bundles) == ["fit_octant"]
    assert [item.name for item in view_inputs] == ["full_volume"]
    assert list(view_bundles) == ["full_volume"]

    entry = run_group_fit(group, group.fits[0])

    assert entry.goodness["status"] == "converged"
    assert model.parameters["constant"] == pytest.approx(1.0, abs=0.01)
    assert entry.goodness["dataset_n_points"] == {"fit_octant": 20}
    assert set(entry.channels) == {"fit_octant", "full_volume"}
    assert entry.channels["full_volume"]["visualization_only"] is True
    assert entry.channels["full_volume"]["fit"] == pytest.approx(np.ones((4, 5)), abs=0.01)
    assert entry.metadata["visualization_only_datasets"] == ["full_volume"]
    restored = _fit_channels_from_dict(_fit_channels_to_dict(entry.channels))
    assert restored["full_volume"]["visualization_only"] is True


def test_disabled_dataset_gets_post_fit_model_without_affecting_fit():
    from nfit.project_gui import fit_dataset_inputs

    group = _fit_ready_group({"fit_octant": 1.0, "disabled_volume": 9.0})
    disabled = group.get_dataset("disabled_volume")
    disabled.enabled = False
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    fit_inputs, _fit_bundles = fit_dataset_inputs(group, purpose="fit")
    view_inputs, _view_bundles = fit_dataset_inputs(
        group,
        purpose="visualization",
    )

    assert [item.name for item in fit_inputs] == ["fit_octant"]
    assert [item.name for item in view_inputs] == ["disabled_volume"]
    assert view_inputs[0].weight == 0.0

    entry = run_group_fit(group, group.fits[0])

    assert entry.goodness["dataset_n_points"] == {"fit_octant": 20}
    assert set(entry.channels) == {"fit_octant", "disabled_volume"}
    assert entry.channels["disabled_volume"]["visualization_only"] is True
    assert entry.channels["disabled_volume"]["fit"] == pytest.approx(
        np.ones((4, 5)),
        abs=0.01,
    )
    datasets, names = slice_viewer_datasets(group)
    disabled_view = datasets[names.index("disabled_volume")]
    assert "fit" in disabled_view.metadata


def test_disabled_dataset_group_is_visualization_only():
    from nfit.project_gui import fit_dataset_inputs

    group = _fit_ready_group({"fit_octant": 1.0, "disabled_volume": 9.0})
    disabled = group.get_dataset("disabled_volume")
    group.datasets.remove(disabled)
    group.subgroups.append(
        DatasetGroup("disabled_group", datasets=[disabled], enabled=False)
    )
    model = create_model_component(group)
    model.fit_parameters["constant"] = True
    ensure_fit_history(group)

    fit_inputs, _ = fit_dataset_inputs(group, purpose="fit")
    view_inputs, _ = fit_dataset_inputs(group, purpose="visualization")
    all_inputs, _ = fit_dataset_inputs(group, purpose="overlay")

    assert [item.name for item in fit_inputs] == ["fit_octant"]
    assert [item.name for item in view_inputs] == ["disabled_volume"]
    assert view_inputs[0].weight == 0.0
    assert [item.name for item in all_inputs] == ["fit_octant", "disabled_volume"]

    entry = run_group_fit(group, group.fits[0])

    assert entry.goodness["dataset_n_points"] == {"fit_octant": 20}
    assert entry.channels["disabled_volume"]["visualization_only"] is True
    assert entry.channels["disabled_volume"]["fit"] == pytest.approx(
        np.ones((4, 5)),
        abs=0.01,
    )
    live = current_model_channels(group)
    assert live["disabled_volume"]["fit"] == pytest.approx(
        np.ones((4, 5)),
        abs=0.01,
    )


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
    view = attach_fit_channels_to_view(group, "mag", view)
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
    assert np.all(bundle.points.E[np.asarray(bundle.points.mask, dtype=bool)] > 4.9)


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
    dataset.metadata["source_file"] = str(tmp_path / "missing.nxs")
    dataset.unload_data()

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
