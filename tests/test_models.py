import numpy as np

from nfit.models import (
    compound_additive_model,
    linear_background,
    make_constant_intensity_model,
    relaxational_chipp,
)


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
