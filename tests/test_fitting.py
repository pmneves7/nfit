import numpy as np
import pytest

from nfit import (
    FitDataset,
    FitProblem,
    ModelSpec,
    ParameterSpec,
    PointData4D,
    ResolutionSpec,
    fit_least_squares,
    fit_problem_least_squares,
    OptimizationConfig,
    make_mask_transform,
    rebin_point_data,
    SamplerConfig,
    sample_problem_parameters,
)
from nfit.cross_section import intensity_from_chipp
from nfit.models import paramagnon_chipp


def measured_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
    chipp = paramagnon_chipp(
        data.H,
        data.K,
        data.L,
        data.E,
        amplitude=params["amplitude"],
        q0=(params["q0_h"], params["q0_k"], params["q0_l"]),
        kappa=params["kappa"],
        omega_sf=params["omega_sf"],
    )
    return intensity_from_chipp(
        chipp,
        data.E,
        params["temperature"],
        scale=params["scale"],
        background=params["background"],
    )


def test_fit_least_squares_recovers_synthetic_single_q_parameters():
    rng = np.random.default_rng(123)
    H = np.linspace(0.25, 0.75, 26)
    E = np.linspace(1.0, 10.0, 22)
    HH, EE = np.meshgrid(H, E, indexing="ij")
    H_flat = HH.ravel()
    E_flat = EE.ravel()
    K_flat = np.full_like(H_flat, 0.5)
    L_flat = np.zeros_like(H_flat)

    true = {
        "amplitude": 5.0,
        "q0_h": 0.5,
        "q0_k": 0.5,
        "q0_l": 0.0,
        "kappa": 0.18,
        "omega_sf": 4.0,
        "scale": 2.0,
        "background": 0.15,
        "temperature": 50.0,
    }
    empty = PointData4D(H_flat, K_flat, L_flat, E_flat, np.zeros_like(H_flat), np.ones_like(H_flat))
    clean = measured_model(empty, true)
    sigma = np.full_like(clean, 0.03)
    data = PointData4D(
        H_flat,
        K_flat,
        L_flat,
        E_flat,
        clean + rng.normal(0.0, sigma),
        sigma,
        temperature=true["temperature"],
    )
    specs = [
        ParameterSpec("amplitude", 4.5, min=0.0),
        ParameterSpec("q0_h", 0.48, min=0.0, max=1.0),
        ParameterSpec("q0_k", 0.5, vary=False),
        ParameterSpec("q0_l", 0.0, vary=False),
        ParameterSpec("kappa", 0.22, min=0.02, max=1.0),
        ParameterSpec("omega_sf", 3.5, min=0.5, max=12.0),
        ParameterSpec("scale", 2.0, vary=False),
        ParameterSpec("background", 0.12),
        ParameterSpec("temperature", 50.0, vary=False),
    ]
    result = fit_least_squares(data, measured_model, specs)
    assert result.success
    np.testing.assert_allclose(result.params["q0_h"], 0.5, atol=0.01)
    np.testing.assert_allclose(result.params["kappa"], 0.18, rtol=0.08)
    np.testing.assert_allclose(result.params["omega_sf"], 4.0, rtol=0.08)
    np.testing.assert_allclose(result.params["amplitude"], 5.0, rtol=0.08)
    assert result.covariance is not None


def test_fit_problem_combines_weighted_dataset_objectives():
    data_a = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[1.0, 1.0],
        intensity=[2.0, 2.0],
        sigma=[1.0, 1.0],
    )
    data_b = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[1.0, 1.0],
        intensity=[4.0, 4.0],
        sigma=[1.0, 1.0],
    )

    def constant_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, params["level"], dtype=float)

    problem = FitProblem(
        datasets=[
            FitDataset("low", data_a, weight=1.0),
            FitDataset("high", data_b, weight=3.0),
        ],
        model=ModelSpec("constant", constant_model),
        parameter_specs=[ParameterSpec("level", 3.0)],
    )

    result = fit_problem_least_squares(problem)

    assert result.success
    np.testing.assert_allclose(result.params["level"], 3.5, atol=1e-8)
    assert set(result.dataset_chi2) == {"low", "high"}
    np.testing.assert_allclose(result.dataset_chi2["high"], result.dataset_chi2["low"] / 3.0)
    np.testing.assert_allclose(result.chi2, sum(result.dataset_chi2.values()))


def test_least_squares_progress_reports_time_per_step():
    data = PointData4D([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 3.0], [1.0, 1.0])
    events = []

    def constant_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, params["level"], dtype=float)

    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=constant_model,
        parameter_specs=[ParameterSpec("level", 1.0)],
    )

    fit_problem_least_squares(problem, progress_callback=events.append)

    timed = [event for event in events if event.get("stage") == "least_squares"]
    assert timed
    assert all(event["elapsed_seconds"] >= 0.0 for event in timed)
    assert all(event["seconds_per_step"] >= 0.0 for event in timed)


def test_fit_dataset_applies_mask_transform_and_resolution():
    data = PointData4D(
        H=[0.0, 1.0, 2.0],
        K=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0],
        E=[1.0, 1.0, 1.0],
        intensity=[10.0, 20.0, 30.0],
        sigma=[1.0, 1.0, 1.0],
    )

    def model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, params["level"], dtype=float)

    def resolution(
        data: PointData4D,
        model_values: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        return model_values + params["offset"]

    problem = FitProblem(
        datasets=[
            FitDataset(
                "masked",
                data,
                transforms=[make_mask_transform(H=(1.0, None))],
                resolution=ResolutionSpec("offset", resolution),
            )
        ],
        model=model,
        parameter_specs=[
            ParameterSpec("level", 10.0, vary=False),
            ParameterSpec("offset", 5.0, vary=False),
        ],
    )

    result = fit_problem_least_squares(problem)

    assert result.dataset_sizes["masked"] == 2
    np.testing.assert_allclose(result.dataset_residuals["masked"], [5.0, 15.0])
    np.testing.assert_allclose(result.dataset_model_values["masked"], [15.0, 15.0])


def test_rebin_point_data_returns_masked_regular_grid():
    data = PointData4D(
        H=[0.25, 0.75],
        K=[0.25, 0.25],
        L=[0.25, 0.25],
        E=[0.25, 0.25],
        intensity=[1.0, 3.0],
        sigma=[1.0, 1.0],
    )

    rebinned = rebin_point_data(
        data,
        lower=[0.0, 0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0, 1.0],
        num_bins=[2, 1, 1, 1],
    )

    assert rebinned.size == 2
    np.testing.assert_allclose(rebinned.intensity, [1.0, 3.0])
    np.testing.assert_allclose(rebinned.sigma, [1.0, 1.0])
    assert np.all(rebinned.mask)


def test_rebin_point_data_respects_existing_mask():
    data = PointData4D(
        H=[0.25, 0.75],
        K=[0.25, 0.25],
        L=[0.25, 0.25],
        E=[0.25, 0.25],
        intensity=[1.0, 100.0],
        sigma=[1.0, 1.0],
        mask=[True, False],
    )

    rebinned = rebin_point_data(
        data,
        lower=[0.0, 0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0, 1.0],
        num_bins=[1, 1, 1, 1],
    )

    np.testing.assert_allclose(rebinned.intensity, [1.0])
    np.testing.assert_allclose(rebinned.sigma, [1.0])
    assert np.all(rebinned.mask)


def test_robust_loss_reduces_outlier_pull():
    data = PointData4D(
        H=np.arange(6),
        K=np.zeros(6),
        L=np.zeros(6),
        E=np.ones(6),
        intensity=[2.0, 2.0, 2.0, 2.0, 2.0, 30.0],
        sigma=np.ones(6),
    )

    def constant_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, params["level"], dtype=float)

    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=constant_model,
        parameter_specs=[ParameterSpec("level", 1.0)],
    )

    linear = fit_problem_least_squares(problem)
    robust = fit_problem_least_squares(
        problem,
        config=OptimizationConfig(kwargs={"loss": "soft_l1", "f_scale": 1.0}),
    )

    assert abs(robust.params["level"] - 2.0) < abs(linear.params["level"] - 2.0)


def test_differential_evolution_initialization_finds_better_basin():
    data = PointData4D([0.0], [0.0], [0.0], [1.0], [0.0], [1.0])

    def multimodal_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        del data
        x = params["x"]
        return np.asarray([(x - 3.0) * (x + 3.0)])

    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=multimodal_model,
        parameter_specs=[ParameterSpec("x", 0.1, min=-5.0, max=5.0)],
    )

    result = fit_problem_least_squares(
        problem,
        config=OptimizationConfig(
            kwargs={
                "initialization": {
                    "method": "differential_evolution",
                    "maxiter": 20,
                    "popsize": 5,
                    "seed": 12,
                }
            }
        ),
    )

    assert result.success
    assert min(abs(result.params["x"] - 3.0), abs(result.params["x"] + 3.0)) < 1.0e-5


def test_differential_evolution_reports_starting_model_failure():
    data = PointData4D([0.0], [0.0], [0.0], [1.0], [0.0], [1.0])

    def bad_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        del data, params
        raise ValueError("bad model configuration")

    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=bad_model,
        parameter_specs=[ParameterSpec("x", 0.1, min=-5.0, max=5.0)],
    )

    with pytest.raises(RuntimeError, match="starting parameters.*bad model configuration"):
        fit_problem_least_squares(
            problem,
            config=OptimizationConfig(
                kwargs={
                    "initialization": {
                        "method": "differential_evolution",
                        "maxiter": 1,
                        "popsize": 5,
                    }
                }
            ),
        )


def test_emcee_sampling_reports_posterior_samples():
    pytest.importorskip("emcee")
    data = PointData4D(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.1, 1.9],
        [0.1, 0.1, 0.1],
    )

    def constant_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, params["level"], dtype=float)

    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=constant_model,
        parameter_specs=[ParameterSpec("level", 2.0, min=0.0, max=4.0)],
    )

    result = sample_problem_parameters(
        problem,
        initial_params={"level": 2.0},
        config=SamplerConfig(n_walkers=8, n_steps=12, burn_in=2, random_seed=5),
    )

    assert result.samples.shape[1] == 1
    assert result.variable_names == ["level"]
    assert result.metadata["method"] == "emcee"
    assert result.chain is not None
    assert result.chain.shape == (12, 8, 1)
    assert result.log_probability_chain is not None
    assert result.log_probability_chain.shape == (12, 8)


def test_parallel_worker_auto_uses_conservative_cpu_count(monkeypatch):
    from nfit.fitting import _resolve_parallel_workers

    monkeypatch.setattr("nfit.fitting.os.cpu_count", lambda: 12)

    assert _resolve_parallel_workers(-1) == 8
    assert _resolve_parallel_workers(1) == 1
    assert _resolve_parallel_workers(4) == 4


def test_emcee_sampling_requires_variable_parameters():
    data = PointData4D([0.0], [0.0], [0.0], [1.0], [1.0], [1.0])
    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=lambda d, p: np.ones(d.size),
        parameter_specs=[],
    )

    with pytest.raises(ValueError, match="at least one variable"):
        sample_problem_parameters(problem)


def test_dataset_parameter_bindings_share_values_by_group():
    def constant_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(data.size, params["constant"], dtype=float)

    data_a = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    data_b = PointData4D([0.0], [0.0], [0.0], [1.0], [2.2], [0.1])
    data_c = PointData4D([0.0], [0.0], [0.0], [1.0], [5.0], [0.1])
    problem = FitProblem(
        datasets=[
            FitDataset("a", data_a, parameter_bindings={"constant": "group_low"}),
            FitDataset("b", data_b, parameter_bindings={"constant": "group_low"}),
            FitDataset("c", data_c, parameter_bindings={"constant": "group_high"}),
        ],
        model=constant_model,
        parameter_specs=[
            ParameterSpec("group_low", 1.0),
            ParameterSpec("group_high", 4.0),
        ],
    )

    result = fit_problem_least_squares(problem)

    assert result.success
    np.testing.assert_allclose(result.params["group_low"], 2.1, atol=1e-8)
    np.testing.assert_allclose(result.params["group_high"], 5.0, atol=1e-8)
