import numpy as np
import pytest

from nfit import PointData4D, from_arrays


def test_point_data_valid_masks_nonfinite_and_bad_sigma():
    data = PointData4D(
        H=[0.0, 1.0, np.nan, 3.0],
        K=[0.0, 0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0, 0.0],
        E=[1.0, 2.0, 3.0, 4.0],
        intensity=[1.0, 2.0, 3.0, np.inf],
        sigma=[0.1, -1.0, 0.1, 0.1],
    )
    np.testing.assert_array_equal(data.valid_mask(), [True, False, False, False])
    assert data.valid().size == 1


def test_from_arrays_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        from_arrays(
            H=[0.0, 1.0],
            K=[0.0],
            L=[0.0, 0.0],
            E=[1.0, 2.0],
            intensity=[1.0, 2.0],
            sigma=[0.1, 0.1],
        )


def test_temperature_array_is_masked_with_data():
    data = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[1.0, 2.0],
        intensity=[1.0, 2.0],
        sigma=[0.1, -1.0],
        temperature=[10.0, 20.0],
    )
    valid = data.valid()
    np.testing.assert_allclose(valid.temperature, [10.0])

