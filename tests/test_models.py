import numpy as np
import pytest

from nfit.models import (
    compound_additive_model,
    linear_background,
    make_constant_intensity_model,
    multi_q_paramagnon_chipp,
    paramagnon_chipp,
    quadratic_distance_rlu,
    relaxational_chipp,
)


def test_quadratic_distance_rlu_is_zero_at_q0():
    q2 = quadratic_distance_rlu([0.5], [0.5], [0.0], q0=(0.5, 0.5, 0.0), kappa=0.2)
    np.testing.assert_allclose(q2, [0.0])


def test_paramagnon_chipp_shape_and_odd_energy():
    E = np.array([-2.0, 0.0, 2.0])
    out = paramagnon_chipp(
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        E,
        amplitude=2.0,
        q0=(0.0, 0.0, 0.0),
        kappa=0.2,
        omega_sf=4.0,
    )
    assert out.shape == E.shape
    assert out[0] < 0
    assert out[1] == pytest.approx(0.0)
    assert out[2] > 0


def test_paramagnon_peak_at_omega_sf_for_q0():
    E = np.linspace(0.1, 10.0, 500)
    omega_sf = 4.0
    out = paramagnon_chipp(
        np.zeros_like(E),
        np.zeros_like(E),
        np.zeros_like(E),
        E,
        amplitude=1.0,
        q0=(0.0, 0.0, 0.0),
        kappa=0.2,
        omega_sf=omega_sf,
    )
    assert E[np.argmax(out)] == pytest.approx(omega_sf, abs=0.03)


def test_multi_q_sum_matches_single_for_one_center():
    E = np.array([1.0, 2.0, 3.0])
    single = paramagnon_chipp(
        [0.0, 0.1, 0.2],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        E,
        amplitude=1.5,
        q0=(0.0, 0.0, 0.0),
        kappa=0.3,
        omega_sf=4.0,
    )
    multi = multi_q_paramagnon_chipp(
        [0.0, 0.1, 0.2],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        E,
        centers=[(0.0, 0.0, 0.0)],
        amplitude=1.5,
        kappa=0.3,
        omega_sf=4.0,
    )
    np.testing.assert_allclose(multi, single)


def test_relaxational_chipp_and_background():
    out = relaxational_chipp(chi_q=[2.0], gamma_q=[3.0], E=[3.0])
    np.testing.assert_allclose(out, [1.0])
    np.testing.assert_allclose(linear_background([1.0, 2.0], 0.5, 2.0), [2.5, 4.5])


def test_constant_and_compound_measured_intensity_models():
    from nfit import PointData4D

    data = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[1.0, 2.0],
        intensity=[0.0, 0.0],
        sigma=[1.0, 1.0],
    )
    background = make_constant_intensity_model("background")
    offset = make_constant_intensity_model(value=2.0)
    model = compound_additive_model(background, offset)

    np.testing.assert_allclose(model(data, {"background": 3.0}), [5.0, 5.0])
