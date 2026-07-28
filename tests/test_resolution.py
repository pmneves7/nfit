import numpy as np

from nfit import (
    FitDataset,
    FitProblem,
    ParameterSpec,
    PointData4D,
    constant_fwhm_energy_resolution,
    fit_problem_least_squares,
    polynomial_fwhm_energy_resolution,
)


def _energy_data(energy):
    energy = np.asarray(energy, dtype=float)
    return PointData4D(
        H=np.zeros_like(energy),
        K=np.zeros_like(energy),
        L=np.zeros_like(energy),
        E=energy,
        intensity=np.zeros_like(energy),
        sigma=np.ones_like(energy),
    )


def test_constant_fwhm_resolution_leaves_constant_model_unchanged():
    data = _energy_data(np.linspace(-2.0, 2.0, 9))

    def model(eval_data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(eval_data.size, params["constant"], dtype=float)

    resolution = constant_fwhm_energy_resolution("fwhm", oversampling=4)
    values = resolution.evaluate_model(data, model, {"constant": 3.0, "fwhm": 0.8})

    np.testing.assert_allclose(values, 3.0, atol=1e-10)


def test_energy_resolution_broadens_delta_like_model():
    data = _energy_data(np.linspace(-2.0, 2.0, 17))

    def narrow_peak(eval_data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.exp(-0.5 * (eval_data.E / params["sigma"]) ** 2)

    resolution = constant_fwhm_energy_resolution(1.0, oversampling=8)
    raw = narrow_peak(data, {"sigma": 0.08})
    broadened = resolution.evaluate_model(data, narrow_peak, {"sigma": 0.08})

    assert broadened[8] < raw[8]
    assert broadened[7] > raw[7]


def test_energy_resolution_convolves_single_energy_point():
    data = _energy_data([0.0])

    def narrow_peak(eval_data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.exp(-0.5 * (eval_data.E / params["sigma"]) ** 2)

    resolution = constant_fwhm_energy_resolution(1.0, oversampling=8)
    broadened = resolution.evaluate_model(data, narrow_peak, {"sigma": 0.08})

    assert 0.0 < broadened[0] < 1.0


def test_polynomial_fwhm_can_use_fitted_parameters_in_fit_problem():
    energy = np.linspace(-2.0, 2.0, 9)
    data = PointData4D(
        H=np.zeros_like(energy),
        K=np.zeros_like(energy),
        L=np.zeros_like(energy),
        E=energy,
        intensity=np.full_like(energy, 4.0),
        sigma=np.full_like(energy, 0.1),
    )

    def constant_model(eval_data: PointData4D, params: dict[str, float]) -> np.ndarray:
        return np.full(eval_data.size, params["constant"], dtype=float)

    problem = FitProblem(
        datasets=[
            FitDataset(
                "data",
                data,
                resolution=polynomial_fwhm_energy_resolution(["fwhm0", "fwhm1"], oversampling=3),
            )
        ],
        model=constant_model,
        parameter_specs=[
            ParameterSpec("constant", 3.0),
            ParameterSpec("fwhm0", 0.5, min=0.01, vary=False),
            ParameterSpec("fwhm1", 0.0, vary=False),
        ],
    )

    result = fit_problem_least_squares(problem)

    assert result.success
    np.testing.assert_allclose(result.params["constant"], 4.0, atol=1e-8)
