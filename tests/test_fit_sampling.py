"""Scientific sample integrity and a separate tempered optimization target."""

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit import (
    ExplorationConfig,
    FitDataset,
    FitProblem,
    OptimizationConfig,
    ParameterSpec,
    PointData4D,
    SamplerConfig,
    SamplingResult,
    best_exploration_candidate,
    evaluate_problem_objective,
    explore_problem_parameters,
    sample_problem_parameters,
)
from nfit.fit_sampling import (
    best_sampling_sample,
    combined_sampling_result,
    sampling_diagnostics,
    sampling_result_from_dict,
    sampling_result_is_posterior,
    sampling_result_to_dict,
    sampling_result_with_window,
)


def constant_problem(observations=(0.0, 0.0, 1.0), *, lower=-10.0, upper=10.0):
    size = len(observations)
    data = PointData4D(
        np.arange(size), np.zeros(size), np.zeros(size), np.ones(size), observations, np.ones(size)
    )
    return FitProblem(
        [FitDataset("data", data)],
        lambda data, params: np.full(data.size, params["level"]),
        [ParameterSpec("level", 0.0, min=lower, max=upper)],
    )


def test_scientific_sampling_codec_preserves_small_spread_and_scores():
    samples = np.array([[1.0e9 - 1], [1.0e9], [1.0e9 + 1]])
    scores = np.array([-3333.333333333333, -3333.333333333332, -3333.333333333334])
    source = SamplingResult(
        samples,
        ["level"],
        scores,
        chain=samples.reshape(3, 1, 1),
        log_probability_chain=scores.reshape(3, 1),
    )
    stored = sampling_result_to_dict(source)
    assert stored["samples"]["dtype"] == "float64"
    decoded = sampling_result_from_dict(stored)
    for name in ("samples", "log_probability", "chain", "log_probability_chain"):
        np.testing.assert_array_equal(getattr(decoded, name), getattr(source, name))
    assert np.std(decoded.samples) > 0


def test_sampling_codec_reads_historical_float32_payloads():
    from nfit.project_io import _encode_float_array

    decoded = sampling_result_from_dict(
        {"samples": _encode_float_array([[1.0, 2.0]]), "variable_names": ["a", "b"]}
    )
    np.testing.assert_array_equal(decoded.samples, [[1.0, 2.0]])


def test_sampler_seed_controls_transitions_independently_of_global_rng():
    pytest.importorskip("emcee")
    problem = constant_problem()
    settings = SamplerConfig(n_walkers=8, n_steps=16, random_seed=17)
    np.random.seed(1)
    first = sample_problem_parameters(problem, settings)
    np.random.seed(2)
    second = sample_problem_parameters(problem, settings)
    np.testing.assert_array_equal(first.chain, second.chain)
    np.testing.assert_array_equal(first.log_probability_chain, second.log_probability_chain)
    assert sampling_result_is_posterior(first)


def test_initial_sampler_and_rewindow_use_identical_raw_steps():
    pytest.importorskip("emcee")
    result = sample_problem_parameters(
        constant_problem(), SamplerConfig(n_walkers=8, n_steps=12, burn_in=2, thin=3, random_seed=5)
    )
    expected = result.chain[2::3].reshape(-1, 1)
    np.testing.assert_array_equal(result.samples, expected)
    updated = sampling_result_with_window(result, 2, 3)
    np.testing.assert_array_equal(result.samples, updated.samples)
    np.testing.assert_array_equal(result.log_probability, updated.log_probability)
    empty = sampling_result_with_window(result, 12, 3)
    assert empty.samples.shape == (0, 1)
    assert empty.metadata["burn_in"] == 12


def test_appending_recomputes_complete_chain_diagnostics_and_acceptance():
    pytest.importorskip("emcee")
    rng = np.random.default_rng(7)
    a = rng.normal(size=(30, 8, 1))
    b = np.cumsum(rng.normal(size=(20, 8, 1)), axis=0)
    original = SamplingResult(
        a.reshape(-1, 1), ["x"], metadata={"acceptance_fraction": [0.2] * 8}, chain=a
    )
    appended = SamplingResult(
        b.reshape(-1, 1), ["x"], metadata={"acceptance_fraction": [0.7] * 8}, chain=b
    )
    combined = combined_sampling_result(original, appended, burn_in=3, thin=2)
    diagnostics = sampling_diagnostics(np.concatenate([a, b]), burn_in=3)
    assert combined.metadata["n_steps"] == 50
    for name, expected in diagnostics.items():
        assert combined.metadata[name] == expected
    assert combined.metadata["acceptance_fraction_mean"] == pytest.approx(0.4)
    np.testing.assert_array_equal(combined.samples, np.concatenate([a, b])[3::2].reshape(-1, 1))


def test_appending_cannot_mix_hot_search_with_posterior():
    chain = np.zeros((2, 4, 1))
    posterior = SamplingResult(chain.reshape(-1, 1), ["x"], chain=chain)
    hot = SamplingResult(
        chain.reshape(-1, 1),
        ["x"],
        chain=chain,
        metadata={"purpose": "exploration", "temperature": 10.0},
    )
    with pytest.raises(ValueError, match="different sampling targets"):
        combined_sampling_result(posterior, hot, burn_in=0, thin=1)
    assert not sampling_result_is_posterior(hot)


def test_best_sample_honors_posterior_window_but_search_can_keep_burnin():
    result = SamplingResult(
        np.array([[2.0]]),
        ["x"],
        np.array([-2.0]),
        chain=np.array([[[1.0]], [[2.0]]]),
        log_probability_chain=np.array([[-1.0], [-2.0]]),
    )
    assert best_sampling_sample(result)[0] == {"x": 2.0}
    assert best_sampling_sample(result, include_burn_in=True)[0] == {"x": 1.0}


@pytest.mark.parametrize("temperature", [0.0, 0.5, np.inf, np.nan])
def test_exploration_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        explore_problem_parameters(constant_problem(), ExplorationConfig(temperature=temperature))


def test_broad_exploration_has_explicit_finite_bounds_requirement():
    with pytest.raises(ValueError, match="finite bounds"):
        explore_problem_parameters(constant_problem(lower=None, upper=None), ExplorationConfig())


def test_exploration_target_uses_original_robust_cost_divided_by_temperature(monkeypatch):
    import nfit.fit_exploration as module

    problem = constant_problem((0.0, 0.0, 100.0))
    optimizer = OptimizationConfig(kwargs={"loss": "cauchy", "f_scale": 2.0})

    def inspect_target(target, walkers, names, config, **kwargs):
        for value in (0.0, 3.0, 8.0):
            cost = evaluate_problem_objective(problem, {"level": value}, config=optimizer)
            assert target(np.array([value])) == pytest.approx(-cost / 12.0)
        assert target(np.array([11.0])) == -np.inf
        return SamplingResult(
            walkers,
            names,
            metadata={"purpose": kwargs["purpose"], "temperature": kwargs["temperature"]},
        )

    monkeypatch.setattr(module, "sample_parameter_vectors", inspect_target)
    result = explore_problem_parameters(
        problem, ExplorationConfig(temperature=12.0, random_seed=3), optimization_config=optimizer
    )
    assert result.metadata["objective_loss"] == "cauchy"
    assert not sampling_result_is_posterior(result)


def test_candidate_comparison_rechecks_original_cost_and_rejects_rounded_fake_improvement():
    problem = constant_problem((0.0, 0.0, 100.0), lower=-100, upper=100)
    baseline = {"level": 100 / 3}
    result = SamplingResult(
        np.array([[np.float32(100 / 3)]]),
        ["level"],
        np.array([1.0e10]),
        metadata={"purpose": "exploration", "temperature": 100.0},
    )
    assert best_exploration_candidate(problem, result, baseline_params=baseline) is None
    good = SamplingResult(
        np.array([[100 / 3]]),
        ["level"],
        np.array([-1000.0]),
        metadata={"purpose": "exploration", "temperature": 100.0},
    )
    candidate = best_exploration_candidate(problem, good, baseline_params={"level": 0.0})
    assert candidate["objective"] == pytest.approx(10000 / 3)
    assert candidate["baseline_objective"] == 5000.0
    assert candidate["parameters"]["level"] == pytest.approx(100 / 3)


def test_robust_candidate_cannot_win_only_by_lowering_unrobust_chisquared():
    problem = constant_problem((0.0, 0.0, 100.0), lower=-100, upper=100)
    result = SamplingResult(
        np.array([[100 / 3]]),
        ["level"],
        np.array([-1.0]),
        metadata={"purpose": "exploration", "temperature": 10.0},
    )
    assert (
        best_exploration_candidate(
            problem,
            result,
            baseline_params={"level": 0.005},
            optimization_config=OptimizationConfig(kwargs={"loss": "cauchy"}),
        )
        is None
    )


def test_bounded_search_discovers_remote_lower_minimum_and_keeps_posterior_separate():
    pytest.importorskip("emcee")
    data = PointData4D([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    problem = FitProblem(
        [FitDataset("d", data)],
        lambda d, p: np.array([p["x"] ** 2 - 9.0, 0.1 * (p["x"] - 3.0)]),
        [ParameterSpec("x", -3.0, min=-5.0, max=5.0)],
    )
    settings = ExplorationConfig(temperature=10.0, n_walkers=12, n_steps=40, random_seed=19)
    result = explore_problem_parameters(problem, settings)
    candidate = best_exploration_candidate(problem, result, baseline_params={"x": -3.0})
    assert candidate is not None
    assert candidate["parameters"]["x"] > 0
    assert candidate["objective"] < 0.18
    assert result.metadata["purpose"] == "exploration"
    assert not sampling_result_is_posterior(result)


def test_sampling_services_are_gui_independent_and_facade_types_keep_identity():
    import nfit.fit_sampling as sampling
    import nfit.fitting as fitting

    assert fitting.SamplerConfig is sampling.SamplerConfig
    assert fitting.SamplingResult is sampling.SamplingResult
    assert fitting.SamplingCancelled is sampling.SamplingCancelled
    for name in ("fit_sampling.py", "fit_exploration.py"):
        tree = ast.parse((Path(fitting.__file__).parent / name).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any("PySide" in name or "project_gui" in name for name in imports)


def test_candidate_preserves_fixed_parameters_and_fills_missing_baseline_values():
    original = constant_problem()
    problem = FitProblem(
        original.datasets,
        lambda data, params: np.full(data.size, params["level"] + params["fixed"]),
        [ParameterSpec("level", 0.0, min=-10.0, max=10.0), ParameterSpec("fixed", 2.0, vary=False)],
    )
    result = SamplingResult(
        np.array([[-5 / 3]]),
        ["level"],
        np.array([-1.0]),
        metadata={"purpose": "exploration", "temperature": 10.0},
    )
    candidate = best_exploration_candidate(problem, result, baseline_params={"fixed": 1000.0})
    assert candidate["parameters"]["fixed"] == 2.0
    assert candidate["baseline_objective"] == pytest.approx(4.5)
    result.variable_names = ["unknown"]
    with pytest.raises(ValueError, match="parameter names"):
        best_exploration_candidate(problem, result, baseline_params={})


def test_candidate_rejects_out_of_bounds_stored_values():
    result = SamplingResult(
        np.array([[11.0]]),
        ["level"],
        np.array([1.0]),
        metadata={"purpose": "exploration", "temperature": 10.0},
    )
    with pytest.raises(ValueError, match="outside bounds"):
        best_exploration_candidate(constant_problem(), result, baseline_params={})


def test_candidate_recovers_saved_robust_objective_and_rejects_changed_loss():
    result = SamplingResult(
        np.array([[100 / 3]]),
        ["level"],
        np.array([-1.0]),
        metadata={
            "purpose": "exploration",
            "temperature": 10.0,
            "objective_loss": "cauchy",
            "objective_f_scale": 1.0,
        },
    )
    problem = constant_problem((0.0, 0.0, 100.0), lower=-100, upper=100)
    assert best_exploration_candidate(problem, result, baseline_params={"level": 0.005}) is None
    with pytest.raises(ValueError, match="objective settings"):
        best_exploration_candidate(
            problem,
            result,
            baseline_params={"level": 0.005},
            optimization_config=OptimizationConfig(),
        )


def test_exactly_redundant_parameters_do_not_report_finite_covariance_errors():
    from nfit import fit_problem_least_squares

    data = PointData4D(
        np.arange(5), np.zeros(5), np.zeros(5), np.ones(5), np.full(5, 2.0), np.ones(5)
    )
    dataset = FitDataset(
        "d",
        data,
        model_jacobian=lambda data, params: {"a": np.ones(data.size), "b": np.ones(data.size)},
    )
    problem = FitProblem(
        [dataset],
        lambda data, params: np.full(data.size, params["a"] + params["b"]),
        [ParameterSpec("a", 0.0), ParameterSpec("b", 0.0)],
    )
    result = fit_problem_least_squares(problem)
    assert result.success
    assert result.params["a"] + result.params["b"] == pytest.approx(2.0)
    assert result.covariance is None
    assert result.stderr is None
    assert result.covariance_interpretation == "unidentifiable_rank_deficient"
    assert "rank deficient" in result.message


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf])
def test_fit_problem_rejects_nonfinite_dataset_weights(weight):
    problem = constant_problem()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        FitProblem(
            [FitDataset("data", problem.datasets[0].data, weight=weight)],
            problem.model,
            problem.parameter_specs,
        )


def test_differential_evolution_probes_use_parent_robust_objective(monkeypatch):
    from types import SimpleNamespace

    import nfit.fitting as module

    problem = constant_problem((0.0, 0.0, 100.0), lower=-100.0, upper=100.0)
    config = OptimizationConfig(
        kwargs={
            "loss": "cauchy",
            "f_scale": 2.0,
            "initialization": {"method": "differential_evolution"},
        }
    )
    probed = []

    def fake_de(objective, bounds, **kwargs):
        for value in (0.0, 100 / 3):
            observed = objective(np.array([value]))
            expected = evaluate_problem_objective(problem, {"level": value}, config=config)
            assert observed == pytest.approx(expected)
            probed.append(observed)
        return SimpleNamespace(x=np.array([0.0]))

    monkeypatch.setattr(module, "_scipy_differential_evolution", fake_de)
    module.fit_problem_least_squares(problem, config=config)
    assert len(probed) == 2
    assert probed[0] < probed[1]


def test_no_variable_fit_reports_robust_cost_separately_from_raw_chisquared():
    from nfit import fit_problem_least_squares

    original = constant_problem((0.0, 0.0, 100.0))
    problem = FitProblem(
        original.datasets, original.model, [ParameterSpec("level", 0.0, vary=False)]
    )
    config = OptimizationConfig(kwargs={"loss": "cauchy"})
    result = fit_problem_least_squares(problem, config=config)
    assert result.chi2 == 10000.0
    assert result.cost == pytest.approx(0.5 * np.log1p(10000.0))


def test_resuming_saved_rng_state_matches_uninterrupted_chain():
    import json

    pytest.importorskip("emcee")
    problem = constant_problem()
    settings = SamplerConfig(n_walkers=8, n_steps=12, random_seed=9)
    first = sample_problem_parameters(problem, settings)
    restored = sampling_result_from_dict(json.loads(json.dumps(sampling_result_to_dict(first))))
    second = sample_problem_parameters(
        problem,
        settings,
        initial_walkers=restored.chain[-1],
        initial_random_state=restored.metadata["random_state"],
    )
    full = sample_problem_parameters(problem, SamplerConfig(n_walkers=8, n_steps=24, random_seed=9))
    np.testing.assert_array_equal(np.concatenate([first.chain, second.chain]), full.chain)
    np.testing.assert_array_equal(
        np.concatenate([first.log_probability_chain, second.log_probability_chain]),
        full.log_probability_chain,
    )


@pytest.mark.parametrize("mode", ["posterior", "exploration_local", "exploration_custom"])
def test_invalid_initial_models_fail_before_recording_samples(mode):
    pytest.importorskip("emcee")
    original = constant_problem()

    def invalid_model(data, params):
        raise ValueError("conserved E=0 configuration needs a finite slope")

    problem = FitProblem(original.datasets, invalid_model, original.parameter_specs)
    if mode == "posterior":
        with pytest.raises(ValueError, match="finite slope"):
            sample_problem_parameters(problem, SamplerConfig(n_walkers=8, n_steps=2))
    else:
        walkers = np.linspace(-1.0, 1.0, 8).reshape(-1, 1) if mode == "exploration_custom" else None
        with pytest.raises(ValueError, match="finite slope"):
            explore_problem_parameters(
                problem,
                ExplorationConfig(initialization="local", n_walkers=8, n_steps=2),
                initial_walkers=walkers,
            )
