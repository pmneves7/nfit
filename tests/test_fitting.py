from types import SimpleNamespace

import numpy as np
import pytest

import nfit.fitting as fitting_module
from nfit import (
    FitCancellationRequested,
    FitDataset,
    FitProblem,
    ModelSpec,
    OptimizationConfig,
    ParameterSpec,
    PointData4D,
    ResolutionSpec,
    SamplerConfig,
    SamplingCancelled,
    fit_least_squares,
    fit_problem_least_squares,
    make_mask_transform,
    rebin_point_data,
    sample_problem_parameters,
)
from nfit.cross_section import MAGNETIC_GAMMA0_PER_MU_B, intensity_from_chipp
from nfit.models import relaxational_chipp


def measured_model(data: PointData4D, params: dict[str, float]) -> np.ndarray:
    q_profile = params["amplitude"] / (1.0 + np.square((data.H - params["q0_h"]) / params["kappa"]))
    chipp = relaxational_chipp(q_profile, params["omega_sf"], data.E)
    return intensity_from_chipp(
        chipp,
        data.E,
        params["temperature"],
        scale=params["scale"],
        background=params["background"],
    )


def test_fit_least_squares_recovers_synthetic_peak_parameters():
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
    sigma = np.full_like(clean, 0.03 * MAGNETIC_GAMMA0_PER_MU_B**2 / np.pi)
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


def test_covariance_defaults_to_absolute_uncertainties_and_can_use_residual_scale():
    data = PointData4D(
        H=[0.0, 1.0, 2.0],
        K=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0],
        E=[1.0, 1.0, 1.0],
        intensity=[0.0, 2.0, 4.0],
        sigma=[1.0, 1.0, 1.0],
    )

    def constant_model(data, params):
        return np.full(data.size, params["level"])

    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=ModelSpec("constant", constant_model),
        parameter_specs=[ParameterSpec("level", 1.0)],
    )

    absolute = fit_problem_least_squares(problem)
    residual = fit_problem_least_squares(
        problem,
        config=OptimizationConfig(covariance_mode="residual"),
    )

    assert absolute.covariance_mode == "absolute"
    assert absolute.covariance_scale_factor == 1.0
    np.testing.assert_allclose(absolute.covariance, [[1.0 / 3.0]])
    assert residual.covariance_mode == "residual"
    np.testing.assert_allclose(residual.covariance_scale_factor, 4.0)
    np.testing.assert_allclose(residual.covariance, absolute.covariance * 4.0)


def test_covariance_mode_is_validated():
    data = PointData4D([0.0], [0.0], [0.0], [1.0], [1.0], [1.0])
    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=ModelSpec("constant", lambda data, params: np.ones(data.size)),
        parameter_specs=[],
    )

    with pytest.raises(ValueError, match="covariance_mode"):
        fit_problem_least_squares(
            problem,
            config=OptimizationConfig(covariance_mode="unknown"),
        )


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


def test_least_squares_can_parallelize_numerical_derivative_evaluations(
    monkeypatch,
):
    observed = {}

    def fake_least_squares(
        fun,
        *,
        x0,
        bounds,
        workers=None,
        **kwargs,
    ):
        del bounds, kwargs
        assert callable(workers)
        points = [
            np.asarray(x0, dtype=float) + offset
            for offset in np.eye(len(x0))
        ]
        values = workers(fun, points)
        observed["values"] = values
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            jac=np.eye(len(x0)),
            success=True,
            message="parallel derivative probe",
            cost=0.0,
        )

    monkeypatch.setattr(
        fitting_module,
        "_scipy_least_squares",
        fake_least_squares,
    )
    result = fitting_module._run_least_squares(
        lambda values: np.asarray(values, dtype=float),
        x0=np.zeros(3, dtype=float),
        bounds=(
            np.full(3, -np.inf, dtype=float),
            np.full(3, np.inf, dtype=float),
        ),
        kwargs={"finite_difference_workers": 3},
        names=("a", "b", "c"),
    )

    assert result.success is True
    assert len(observed["values"]) == 3
    np.testing.assert_array_equal(
        np.asarray(observed["values"]),
        np.eye(3),
    )


def test_parallel_numerical_derivatives_preserve_fit_result():
    coordinate = np.linspace(-1.0, 1.0, 9)
    data = PointData4D(
        H=coordinate,
        K=np.zeros_like(coordinate),
        L=np.zeros_like(coordinate),
        E=np.ones_like(coordinate),
        intensity=1.7 * coordinate - 0.4,
        sigma=np.full_like(coordinate, 0.1),
    )
    problem = FitProblem(
        datasets=[FitDataset("line", data)],
        model=lambda values, params: (
            params["slope"] * values.H + params["offset"]
        ),
        parameter_specs=[
            ParameterSpec("slope", 1.0),
            ParameterSpec("offset", 0.0),
        ],
    )

    serial = fit_problem_least_squares(
        problem,
        config=OptimizationConfig(
            kwargs={"finite_difference_workers": 1}
        ),
    )
    parallel = fit_problem_least_squares(
        problem,
        config=OptimizationConfig(
            kwargs={"finite_difference_workers": 2}
        ),
    )

    assert serial.success and parallel.success
    assert parallel.params == pytest.approx(serial.params, rel=1.0e-12)
    assert parallel.cost == pytest.approx(serial.cost, abs=1.0e-20)


def test_cancelled_least_squares_returns_lowest_objective_point(monkeypatch):
    data = PointData4D(
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [1.0, 1.0],
    )
    problem = FitProblem(
        datasets=[FitDataset("data", data)],
        model=lambda data, params: np.full(data.size, params["level"]),
        parameter_specs=[ParameterSpec("level", 0.0)],
    )

    def fake_least_squares(fun, *, x0, bounds, **kwargs):
        del x0, bounds, kwargs
        for level in (0.0, 1.0, 1.8, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
            fun(np.asarray([level], dtype=float))
        raise AssertionError("termination should unwind the optimizer")

    def terminate(event):
        if event["iteration"] >= 10:
            raise FitCancellationRequested("stop")

    monkeypatch.setattr(fitting_module, "_scipy_least_squares", fake_least_squares)

    result = fit_problem_least_squares(problem, progress_callback=terminate)

    assert result.cancelled is True
    assert result.success is False
    assert result.params["level"] == pytest.approx(1.8)
    assert result.cost == pytest.approx(0.04)
    assert result.chi2 == pytest.approx(0.08)
    assert result.covariance is None
    assert result.stderr is None
    assert "lowest-objective parameter set" in result.message


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
        lower=[0.25, 0.0, 0.0, 0.0],
        upper=[0.75, 1.0, 1.0, 1.0],
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


def test_emcee_sampling_reports_posterior_samples(capsys):
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
    assert "emcee integrated autocorrelation time" in capsys.readouterr().out
    assert "autocorrelation_recommended_steps" in result.metadata
    assert "autocorrelation_recommended_additional_steps" in result.metadata


def test_emcee_sampling_cancellation_returns_partial_chain():
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

    def cancel_after_three(event: dict[str, object]) -> None:
        if event.get("stage") == "emcee" and event.get("iteration") == 3:
            raise RuntimeError("stop")

    with pytest.raises(SamplingCancelled) as caught:
        sample_problem_parameters(
            problem,
            initial_params={"level": 2.0},
            config=SamplerConfig(n_walkers=8, n_steps=12, burn_in=0, random_seed=5),
            progress_callback=cancel_after_three,
        )

    result = caught.value.result
    assert result.metadata["cancelled"] is True
    assert result.metadata["completed"] is False
    assert result.metadata["n_steps"] == 3
    assert result.metadata["requested_n_steps"] == 12
    assert result.chain is not None
    assert result.chain.shape == (3, 8, 1)
    assert result.log_probability_chain is not None
    assert result.log_probability_chain.shape == (3, 8)


def test_parallel_worker_auto_uses_conservative_cpu_count(monkeypatch):
    from nfit.fitting import _resolve_parallel_workers

    monkeypatch.setattr(
        "nfit.fitting._parallel.detect_cpu_budget",
        lambda: 12,
    )

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


def test_magnetic_field_vector_frames_agree_for_cubic():
    from nfit.fitting import magnetic_field_vector

    lattice = {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    uvw = magnetic_field_vector(2.0, [1, 1, 1], "uvw", lattice)
    hkl = magnetic_field_vector(2.0, [1, 1, 1], "hkl", lattice)
    np.testing.assert_allclose(uvw, hkl, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(uvw), 2.0, rtol=1e-12)
    np.testing.assert_allclose(uvw, 2.0 * np.ones(3) / np.sqrt(3.0), atol=1e-12)


def test_magnetic_field_vector_frames_differ_for_orthorhombic():
    from nfit.fitting import magnetic_field_vector

    lattice = {"a": 3.0, "b": 5.0, "c": 8.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    # Direct [110] direction is along a*x + b*y ~ (3, 5, 0); reciprocal (110)
    # is along x/a + y/b ~ (1/3, 1/5, 0). Hand-checked normalization.
    uvw = magnetic_field_vector(1.0, [1, 1, 0], "uvw", lattice)
    hkl = magnetic_field_vector(1.0, [1, 1, 0], "hkl", lattice)
    np.testing.assert_allclose(
        uvw, np.array([3.0, 5.0, 0.0]) / np.linalg.norm([3.0, 5.0, 0.0]), atol=1e-12
    )
    np.testing.assert_allclose(
        hkl, np.array([1 / 3.0, 1 / 5.0, 0.0]) / np.linalg.norm([1 / 3.0, 1 / 5.0, 0.0]), atol=1e-12
    )
    assert not np.allclose(uvw, hkl)

    with pytest.raises(ValueError, match="frame"):
        magnetic_field_vector(1.0, [1, 0, 0], "cartesian", lattice)
    with pytest.raises(ValueError, match="direction"):
        magnetic_field_vector(1.0, [0, 0, 0], "uvw", lattice)


def test_point_data_magnetic_field_propagates_through_pipeline():
    from nfit.fitting import _copy_point_data

    field = np.array([0.0, 0.0, 2.0])
    data = PointData4D(
        H=np.array([0.1, 0.2, np.nan]),
        K=np.zeros(3),
        L=np.zeros(3),
        E=np.array([1.0, 2.0, 3.0]),
        intensity=np.ones(3),
        sigma=np.ones(3),
        temperature=4.0,
        magnetic_field=field,
    )
    np.testing.assert_array_equal(data.magnetic_field, field)
    valid = data.valid()
    assert valid.size == 2
    np.testing.assert_array_equal(valid.magnetic_field, field)
    copied = _copy_point_data(data, mask=np.array([True, True, False]))
    np.testing.assert_array_equal(copied.magnetic_field, field)
    # a non-3-vector is rejected
    with pytest.raises(ValueError, match="magnetic_field"):
        PointData4D(
            H=np.zeros(1),
            K=np.zeros(1),
            L=np.zeros(1),
            E=np.zeros(1),
            intensity=np.zeros(1),
            sigma=np.ones(1),
            magnetic_field=[1.0, 2.0],
        )
