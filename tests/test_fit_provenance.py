from dataclasses import replace

import numpy as np
import pytest

from nfit.dataset import PointData4D
from nfit.fit_provenance import fit_problem_fingerprint
from nfit.fitting import DerivedParameter, FitDataset, FitProblem, ParameterSpec


def _problem():
    points = PointData4D(
        H=[0, 1],
        K=[0, 0],
        L=[0, 0],
        E=[1, 2],
        intensity=[3, 4],
        sigma=[1, 2],
        temperature=10.0,
        metadata={"coordinate_units": "rlu"},
    )
    return FitProblem(
        datasets=[FitDataset("scan", points)],
        model=lambda data, params: np.full(data.size, params["a"]),
        parameter_specs=[
            ParameterSpec("a", 1.0, min=0.0, max=10.0),
            ParameterSpec("fixed", 2.0, vary=False),
        ],
        metadata={"fit_model_configuration": [{"type": "constant"}]},
    )


def test_target_fingerprint_ignores_varied_start_but_includes_fixed_values_and_bounds():
    problem = _problem()
    original = fit_problem_fingerprint(problem)
    a, fixed = problem.parameter_specs
    assert (
        fit_problem_fingerprint(replace(problem, parameter_specs=[replace(a, value=9.0), fixed]))
        == original
    )
    for specs in ([a, replace(fixed, value=3.0)], [replace(a, max=11.0), fixed]):
        assert fit_problem_fingerprint(replace(problem, parameter_specs=specs)) != original
    assert (
        fit_problem_fingerprint(
            replace(
                problem,
                derived_parameters=[
                    DerivedParameter("derived", base=0.0, expression="2 * a", dependencies=("a",)),
                ],
            )
        )
        != original
    )


@pytest.mark.parametrize(
    "change",
    [
        {"intensity": [3.0, 5.0]},
        {"sigma": [1.0, 3.0]},
        {"H": [0.0, 2.0]},
        {"temperature": 20.0},
        {"metadata": {"coordinate_units": "inv_angstrom"}},
        {"mask": [True, False]},
        {"normalization_denominator": [1.0, 2.0]},
    ],
)
def test_target_fingerprint_detects_changed_scientific_inputs(change):
    problem = _problem()
    dataset = problem.datasets[0]
    modified = replace(
        problem, datasets=[replace(dataset, data=dataset.data.with_updates(**change))]
    )
    assert fit_problem_fingerprint(modified) != fit_problem_fingerprint(problem)


def test_target_fingerprint_detects_objective_weight_model_and_binding_changes():
    problem = _problem()
    original = fit_problem_fingerprint(problem)
    for changes in (
        {"weight": 2.0},
        {"likelihood": "gaussian_gls"},
        {"parameter_bindings": {"local": "a"}},
        {"data_scale_parameter": "a"},
    ):
        altered = replace(problem, datasets=[replace(problem.datasets[0], **changes)])
        assert fit_problem_fingerprint(altered) != original
    altered = replace(problem, metadata={"fit_model_configuration": [{"type": "linear"}]})
    assert fit_problem_fingerprint(altered) != original


def test_target_fingerprint_requires_declared_custom_model_identity():
    problem = replace(_problem(), metadata={})
    with pytest.raises(ValueError, match="declared model configuration"):
        fit_problem_fingerprint(problem)
    assert fit_problem_fingerprint(problem, model_configuration={"type": "custom", "revision": 1})


def test_target_fingerprint_hashes_strided_arrays_and_mapping_order_identically():
    import hashlib

    from nfit.fit_provenance import _hash_value

    array = np.arange(40.0).reshape(10, 4)[:, ::2]
    left, right = hashlib.sha256(), hashlib.sha256()
    _hash_value(left, {"b": array, "a": 3})
    _hash_value(right, {"a": 3, "b": array.copy()})
    assert left.hexdigest() == right.hexdigest()


def test_electronic_evaluation_certificates_do_not_change_fit_target():
    from nfit import FitDatasetInput, compile_fit_problem
    from nfit.fitting import evaluate_problem_model
    from tests.test_electronic_pipeline_sampling import _pipeline_components

    bands, response, dressing = _pipeline_components()
    response.config.update(
        response_q_evaluation="auto",
        response_mesh=[32],
        response_q_interpolation_atol=1e-8,
        response_q_interpolation_mesh=[8],
        response_q_validation_points=2,
    )
    points = PointData4D(
        H=[0.137, 0.283],
        K=[0, 0],
        L=[0, 0],
        E=[1, 2],
        intensity=[1, 1],
        sigma=[1, 1],
        temperature=30.0,
    )
    components = [bands, response, dressing]
    datasets = [FitDatasetInput("scan", points, data_type="single_crystal_inelastic")]

    def fingerprint():
        return fit_problem_fingerprint(compile_fit_problem(components, datasets).problem)

    problem = compile_fit_problem(components, datasets).problem
    original = fit_problem_fingerprint(problem)
    evaluate_problem_model(
        problem, "scan", {spec.name: spec.value for spec in problem.parameter_specs}
    )
    assert response.config["response_q_interpolation_certificates"]["scan"]["status"] == "certified"
    assert fingerprint() == original
    response.config["response_sampling_certificate"] = {"status": "converged", "mesh": [32]}
    assert fingerprint() == original
    response.config["response_q_interpolation_rtol"] *= 0.5
    assert fingerprint() != original
    response.config["response_q_interpolation_rtol"] *= 2.0
    response.config["response_mesh"] = [36]
    assert fingerprint() != original
