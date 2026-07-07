import numpy as np
import pytest

from metallix import NDRebin, rebin_nd


def test_rebin_nd_averages_values_and_propagates_errors():
    result = rebin_nd(
        data=[1.0, 3.0, 10.0],
        coords=[0.25, 0.75, 1.75],
        data_errs=[1.0, 1.0, 2.0],
        lower=0.0,
        upper=2.0,
        num_bins=2,
    )

    np.testing.assert_allclose(result.binned_data, [2.0, 10.0])
    np.testing.assert_allclose(result.binned_data_errs, [np.sqrt(2.0) / 2.0, 2.0])
    np.testing.assert_allclose(result.n_samples, [2.0, 1.0])
    np.testing.assert_allclose(result.bin_centers_list[0], [0.5, 1.5])


def test_rebin_nd_sums_when_normalize_is_false():
    result = rebin_nd(
        data=[1.0, 3.0, 10.0],
        coords=[0.25, 0.75, 1.75],
        data_errs=[1.0, 1.0, 2.0],
        lower=0.0,
        upper=2.0,
        num_bins=2,
        normalize=False,
    )

    np.testing.assert_allclose(result.binned_data, [4.0, 10.0])
    np.testing.assert_allclose(result.binned_data_errs, [np.sqrt(2.0), 2.0])


def test_fractional_rebin_distributes_between_neighboring_bins():
    result = rebin_nd(
        data=[10.0],
        coords=[1.0],
        lower=0.0,
        upper=2.0,
        num_bins=2,
        fractional=True,
        normalize=False,
    )

    np.testing.assert_allclose(result.binned_data, [5.0, 5.0])
    np.testing.assert_allclose(result.n_samples, [0.5, 0.5])


def test_rebin_accepts_last_axis_coordinate_dimension():
    x = np.array([0.25, 0.75])
    y = np.array([0.25, 0.75])
    coords = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = rebin_nd(
        data=data,
        coords=coords,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        num_bins=[2, 2],
    )

    np.testing.assert_allclose(result.binned_data, data)


def test_rebin_requires_bin_definition():
    rebin = NDRebin(data=[1.0], coords=[0.0])
    with pytest.raises(ValueError, match="Either step_size or num_bins"):
        rebin.run()

