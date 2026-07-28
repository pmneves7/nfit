import numpy as np
import pytest

from nfit import PointData4D, PointListData, from_arrays


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



def test_per_point_magnetic_field_validated_and_masked():
    n = 4
    field = np.zeros((n, 3))
    field[:, 2] = [1.0, 2.0, np.nan, 4.0]
    data = PointData4D(
        H=np.zeros(n), K=np.zeros(n), L=np.zeros(n), E=np.zeros(n),
        intensity=np.ones(n), sigma=np.full(n, 0.1),
        magnetic_field=field,
    )
    # A NaN field component drops that point.
    mask = data.valid_mask()
    assert mask.tolist() == [True, True, False, True]
    valid = data.valid()
    assert valid.magnetic_field.shape == (3, 3)
    np.testing.assert_allclose(valid.magnetic_field[:, 2], [1.0, 2.0, 4.0])


def test_magnetic_field_shape_validation():
    with pytest.raises(ValueError, match="per-point"):
        PointData4D(
            H=np.zeros(2), K=np.zeros(2), L=np.zeros(2), E=np.zeros(2),
            intensity=np.ones(2), sigma=np.ones(2),
            magnetic_field=np.zeros((3, 3)),  # wrong point count
        )
    # A single shared 3-vector is still accepted.
    data = PointData4D(
        H=np.zeros(2), K=np.zeros(2), L=np.zeros(2), E=np.zeros(2),
        intensity=np.ones(2), sigma=np.ones(2), magnetic_field=[0.0, 0.0, 1.5],
    )
    assert data.magnetic_field.shape == (3,)


def test_point_data_numeric_payload_requires_explicit_mutable_copy():
    data = PointData4D(
        H=[0.0],
        K=[0.0],
        L=[0.0],
        E=[1.0],
        intensity=[2.0],
        sigma=[0.1],
        temperature=[10.0],
        magnetic_field=[[0.0, 0.0, 1.0]],
    )

    for array in (
        data.H,
        data.intensity,
        data.mask,
        data.temperature,
        data.magnetic_field,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        data.intensity[0] = 3.0

    editable = data.mutable_copy()
    editable.intensity[0] = 3.0
    editable.temperature[0] = 20.0
    replacement = editable.immutable_copy()

    assert data.intensity[0] == 2.0
    assert replacement.intensity[0] == 3.0
    assert replacement.temperature[0] == 20.0
    assert not replacement.intensity.flags.writeable


def test_point_list_columns_require_explicit_mutable_copy():
    data = PointListData(
        columns={"temperature": [10.0], "moment": [1.0]},
        coordinate_names=["temperature"],
        channels=[{"label": "M", "value": "moment", "error": None}],
    )

    assert not data.columns["moment"].flags.writeable
    editable = data.mutable_copy()
    editable.columns["moment"][0] = 2.0
    replacement = editable.immutable_copy()

    assert data.columns["moment"][0] == 1.0
    assert replacement.columns["moment"][0] == 2.0
    assert not replacement.columns["moment"].flags.writeable
