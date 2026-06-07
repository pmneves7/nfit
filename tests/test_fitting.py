import numpy as np

from metallix import PointData4D, ParameterSpec, fit_least_squares
from metallix.cross_section import intensity_from_chipp
from metallix.models import paramagnon_chipp


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
