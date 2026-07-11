import numpy as np

from nfit import (
    DataGroup,
    DatasetEntry,
    FitModelSession,
    ParameterSpec,
    PointData4D,
    make_constant_intensity_model,
)


def test_data_group_selects_data_and_fit_model_session_tracks_history():
    data_a = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    data_b = PointData4D([0.0], [0.0], [0.0], [1.0], [4.0], [0.1])
    data_b_prepared = PointData4D(
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [4.0],
        [0.1],
        mask=[True],
    )
    group = DataGroup(
        name="field_series",
        lattice_parameters={"a": 3.8, "b": 3.8, "c": 12.0},
        spacegroup="P4/mmm",
        datasets=[
            DatasetEntry("low_field", data_a, kind="neutron", parameters={"field": 0.0}),
            DatasetEntry(
                "high_field",
                data_b,
                kind="neutron",
                parameters={"field": 9.0},
                transforms=[lambda _data: data_b_prepared],
            ),
        ],
    )
    session = FitModelSession(
        name="constant_by_field",
        model=make_constant_intensity_model("constant"),
        parameter_specs=[
            ParameterSpec("low_background", 1.0),
            ParameterSpec("high_background", 1.0),
        ],
        parameter_bindings_by_dataset={
            "low_field": {"constant": "low_background"},
            "high_field": {"constant": "high_background"},
        },
    )
    group.add_model(session)

    result = session.fit(group)

    assert group.dataset_names == ["low_field", "high_field"]
    assert group.data_sequence(["high_field"]) == [data_b]
    assert result.success
    np.testing.assert_allclose(result.params["low_background"], 2.0, atol=1e-8)
    np.testing.assert_allclose(result.params["high_background"], 4.0, atol=1e-8)
    assert group.prepared_sequence(["high_field"]) == [data_b_prepared]
    assert len(session.history) == 1
    assert session.history[0].dataset_names == ("low_field", "high_field")
    np.testing.assert_allclose(session.current_parameters()["low_background"], 2.0, atol=1e-8)

    session.parameter_specs[0] = ParameterSpec("low_background", 99.0)
    session.rollback(0, use_after=True)

    np.testing.assert_allclose(session.current_parameters()["low_background"], 2.0, atol=1e-8)


def test_fit_model_session_skips_disabled_datasets_and_uses_dataset_fit_weight():
    data_a = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    data_b = PointData4D([0.0], [0.0], [0.0], [1.0], [4.0], [0.1])
    group = DataGroup(
        name="field_series",
        datasets=[
            DatasetEntry("enabled", data_a, fit_weight=2.5),
            DatasetEntry("disabled", data_b, enabled=False, fit_weight=9.0),
        ],
    )
    session = FitModelSession(
        name="constant",
        model=make_constant_intensity_model("constant"),
        parameter_specs=[ParameterSpec("constant", 1.0)],
    )

    problem = session.build_problem(group)

    assert [dataset.name for dataset in group.select()] == ["enabled"]
    assert [dataset.name for dataset in problem.datasets] == ["enabled"]
    assert problem.datasets[0].weight == 2.5
    assert group.select(["enabled", "disabled"]) == [group.datasets[0]]
