"""Independent scalar references for fractional point-cloud deposition."""

from itertools import product

import numpy as np
import pytest

from nfit import ArrayRebinSource, RebinBatch, rebin_nd, rebin_nd_stream


def _axis_support(value, edges, fractional):
    """Evaluate the center hat functions directly in physical coordinates."""
    if not np.isfinite(value) or not edges[0] <= value <= edges[-1]:
        return []
    centers = (edges[:-1] + edges[1:]) / 2
    if fractional:
        if value <= centers[0]:
            return [(0, 1.0)]
        if value >= centers[-1]:
            return [(len(centers) - 1, 1.0)]
        for left in range(len(centers) - 1):
            if centers[left] <= value <= centers[left + 1]:
                upper_fraction = (value - centers[left]) / (centers[left + 1] - centers[left])
                return [(left, 1 - upper_fraction), (left + 1, upper_fraction)]
    else:
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if left <= value < right or (index == len(centers) - 1 and value == right):
                return [(index, 1.0)]
    raise AssertionError("accepted coordinate has no bin support")


def _reference(data, coords, errors, weights, edges, fractional_axes, normalize, mean_weighting):
    shape = tuple(len(axis) - 1 for axis in edges)
    numerator = np.zeros(shape)
    variance = np.zeros(shape)
    denominator = np.zeros(shape)
    samples = np.zeros(shape)
    inverse_variance = normalize and mean_weighting == "inverse_variance" and errors is not None
    for row, (value, coordinate, statistical_weight) in enumerate(zip(data, coords, weights, strict=True)):
        if not np.isfinite(statistical_weight) or statistical_weight <= 0:
            continue
        error = 0.0 if errors is None else errors[row]
        if inverse_variance and (not np.isfinite(error) or error <= 0):
            continue
        mean_weight = statistical_weight / error**2 if inverse_variance else statistical_weight
        supports = [_axis_support(x, axis_edges, fractional)
                    for x, axis_edges, fractional in zip(coordinate, edges, fractional_axes, strict=True)]
        for corner in product(*supports):
            index = tuple(component[0] for component in corner)
            spatial_weight = np.prod([component[1] for component in corner])
            if spatial_weight == 0:
                continue
            total_weight = spatial_weight * mean_weight
            numerator[index] += total_weight * value
            variance[index] += (total_weight * error)**2
            denominator[index] += total_weight
            samples[index] += spatial_weight
    values = numerator.copy()
    uncertainty = np.sqrt(variance)
    if normalize:
        np.divide(values, denominator, out=values, where=denominator > 0)
        np.divide(uncertainty, denominator, out=uncertainty, where=denominator > 0)
    values[samples == 0] = np.nan
    uncertainty[samples == 0] = np.nan
    return values, uncertainty, samples


@pytest.mark.parametrize("route", ["numpy", "numba", "numba_dense", "numba_sparse",
                                   "stream_numpy", "stream_numba"])
@pytest.mark.parametrize("normalize,mean_weighting", [
    (False, "uniform"), (True, "uniform"), (True, "inverse_variance"),
])
@pytest.mark.parametrize("grid", ["uniform", "mixed", "explicit"])
@pytest.mark.parametrize("fractional_axes", [[True, False, True], [True, True, True]])
def test_fractional_cloud_matches_physical_hat_reference(
    route, normalize, mean_weighting, grid, fractional_axes,
):
    if "numba" in route:
        pytest.importorskip("numba")
    edges = [np.array([-1.5, -0.5, 0.5, 1.5]),
             np.array([0.0, 1.0, 2.0]),
             np.array([-2.0, 0.0, 2.0])]
    lower, upper = [-1.0, 0.5, -1.0], [1.0, 1.5, 1.0]
    if grid != "uniform":
        edges[1] = np.array([0.0, 0.25, 2.0])
        edges[2] = np.array([-2.0, -0.75, 2.0])
    bins = None if grid == "uniform" else [None, edges[1], edges[2]]
    if grid == "explicit":
        bins = edges
    rng = np.random.default_rng(451)
    coords = rng.uniform([-1.75, -0.2, -2.25], [1.75, 2.2, 2.25], size=(73, 3))
    coords[:7] = [[-1.5, 0, -2], [1.5, 2, 2], [-0.5, 0.25, 0],
                  [0.0, 0.5, 0.5], [np.nan, 0.5, 0.5],
                  [0.0, np.inf, 0.5], [0.0, 0.5, -np.inf]]
    data = rng.normal(size=coords.shape[0])
    errors = rng.uniform(0.2, 3.0, size=data.size)
    errors[8] = 0.0
    weights = rng.uniform(0.1, 4.0, size=data.size)
    weights[9:12] = [0.0, -1.0, np.nan]
    strategy = route.split("_")[-1] if route in {"numba_dense", "numba_sparse"} else "serial"
    kwargs = dict(lower=lower, upper=upper, num_bins=[3, 2, 2], bin_edges=bins,
                  fractional_axes=fractional_axes, normalize=normalize,
                  mean_weighting=mean_weighting, workers=2, parallel_strategy=strategy)
    if route.startswith("stream_"):
        result = rebin_nd_stream(
            ArrayRebinSource(data, coords, data_errs=errors, data_weights=weights, batch_size=11),
            backend=route.removeprefix("stream_"), **kwargs,
        )
    else:
        result = rebin_nd(data, coords, data_errs=errors, data_weights=weights,
                          backend=route.split("_")[0], batch_size=11, **kwargs)
    expected = _reference(data, coords, errors, weights, edges, fractional_axes, normalize, mean_weighting)
    for actual, reference in zip((result.binned_data, result.binned_data_errs, result.n_samples),
                                 expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13, equal_nan=True)


@pytest.mark.parametrize("ndim", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("backend,strategy", [("numpy", "serial"), ("numba", "serial"),
                                             ("numba", "dense"), ("numba", "sparse")])
def test_fractional_tensor_preserves_mass_position_and_diagonal_variance(ndim, backend, strategy):
    if backend == "numba":
        pytest.importorskip("numba")
    coordinate = np.linspace(0.1, 0.9, ndim)[None, :]
    result = rebin_nd([12.0], coordinate, data_errs=[3.0], data_weights=[2.0],
                      lower=np.zeros(ndim), upper=np.ones(ndim), num_bins=[2] * ndim,
                      normalize=False, backend=backend, workers=2, parallel_strategy=strategy)
    corners = np.array(list(product((0, 1), repeat=ndim)))
    expected_weights = np.array([
        np.prod([coordinate[0, axis] if corner[axis] else 1 - coordinate[0, axis]
                 for axis in range(ndim)])
        for corner in corners
    ]).reshape((2,) * ndim)
    np.testing.assert_allclose(result.n_samples, expected_weights, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.binned_data, 24 * expected_weights, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.binned_data_errs, 6 * expected_weights, rtol=2e-13, atol=2e-13)
    assert np.sum(result.n_samples) == pytest.approx(1.0)
    assert np.sum(result.binned_data) == pytest.approx(24.0)
    for axis in range(ndim):
        projected_center = np.sum(result.n_samples.ravel() * corners[:, axis])
        assert projected_center == pytest.approx(coordinate[0, axis])


@pytest.mark.parametrize("route", ["numpy", "numba", "stream_numpy", "stream_numba"])
@pytest.mark.parametrize("normalize", [False, True])
def test_fractional_one_bin_axes_do_not_duplicate_signal_or_variance(route, normalize):
    if "numba" in route:
        pytest.importorskip("numba")
    data, errors, weights = [4.0, 12.0], [2.0, 3.0], [1.0, 2.0]
    coordinates = [[0.25, -2.0, 10.0], [0.75, 3.0, 20.0]]
    kwargs = dict(lower=[0.0, -2.0, 10.0], upper=[1.0, 3.0, 20.0],
                  step_size=[1.0, np.inf, np.inf], fractional=True,
                  normalize=normalize, workers=1)
    if route.startswith("stream_"):
        result = rebin_nd_stream(
            ArrayRebinSource(data, coordinates, data_errs=errors, data_weights=weights, batch_size=1),
            backend=route.removeprefix("stream_"), **kwargs,
        )
    else:
        result = rebin_nd(data, coordinates, data_errs=errors, data_weights=weights,
                          backend=route, **kwargs)
    denominator = np.array([1.25, 1.75]) if normalize else np.ones(2)
    np.testing.assert_allclose(result.binned_data[:, 0, 0], np.array([9.0, 19.0]) / denominator)
    np.testing.assert_allclose(result.binned_data_errs[:, 0, 0],
                               np.sqrt([4.5, 20.5]) / denominator)
    np.testing.assert_allclose(result.n_samples[:, 0, 0], [1.0, 1.0])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("fractional", [False, True])
def test_integrated_axes_skip_nonfinite_coordinates(backend, fractional):
    if backend == "numba":
        pytest.importorskip("numba")
    data = [2.0, 100.0, 200.0, 300.0, 4.0]
    coordinates = np.array([0.25, np.nan, np.inf, -np.inf, 0.75])[:, None]
    kwargs = dict(lower=[0.0], upper=[1.0], step_size=[np.inf],
                  fractional=fractional, normalize=False, backend=backend, workers=1)
    for result in (rebin_nd(data, coordinates, **kwargs),
                   rebin_nd_stream(ArrayRebinSource(data, coordinates, batch_size=2), **kwargs)):
        np.testing.assert_allclose(result.binned_data, [6.0])
        np.testing.assert_allclose(result.n_samples, [2.0])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_mixed_fractional_grid_accepts_roundoff_next_to_last_physical_edge(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    discrete_edges = -1.0 + (np.arange(22) - 0.5) * 0.1
    coordinate = [[0.25, np.nextafter(discrete_edges[-1], -np.inf)]]
    kwargs = dict(lower=[0.0, -1.0], upper=[1.0, 1.0], step_size=[1.0, 0.1],
                  fractional_axes=[True, False], normalize=False, backend=backend, workers=1)
    for result in (rebin_nd([8.0], coordinate, data_errs=[4.0], **kwargs),
                   rebin_nd_stream(ArrayRebinSource([8.0], coordinate, data_errs=[4.0]), **kwargs)):
        np.testing.assert_allclose(result.binned_data[:, -1], [6.0, 2.0])
        np.testing.assert_allclose(result.binned_data_errs[:, -1], [3.0, 1.0])
        np.testing.assert_allclose(result.n_samples[:, -1], [0.75, 0.25])
        assert np.sum(result.n_samples[:, :-1]) == 0.0


@pytest.mark.parametrize("route", ["numpy", "numba", "stream_numpy", "stream_numba"])
@pytest.mark.parametrize("discrete_lower", [0.0, -1.0, -100.0])
def test_mixed_fractional_grid_respects_exact_discrete_edges(route, discrete_lower):
    if "numba" in route:
        pytest.importorskip("numba")
    edges = [np.array([-0.5, 0.5, 1.5]), discrete_lower + (np.arange(12) - 0.5) * 0.1]
    discrete_coordinates = np.stack([
        np.nextafter(edges[1], -np.inf), edges[1], np.nextafter(edges[1], np.inf),
    ], axis=1).ravel()
    coords = np.column_stack([np.resize([0.25, 0.75], discrete_coordinates.size), discrete_coordinates])
    data = np.arange(1.0, discrete_coordinates.size + 1)
    errors = np.linspace(1.0, 3.0, data.size)
    weights = np.linspace(0.5, 2.0, data.size)
    kwargs = dict(lower=[0.0, discrete_lower], upper=[1.0, discrete_lower + 1.0],
                  step_size=[1.0, 0.1], fractional_axes=[True, False], workers=1)
    if route.startswith("stream_"):
        result = rebin_nd_stream(
            ArrayRebinSource(data, coords, data_errs=errors, data_weights=weights, batch_size=7),
            backend=route.removeprefix("stream_"), **kwargs,
        )
    else:
        result = rebin_nd(data, coords, data_errs=errors, data_weights=weights,
                          backend=route, batch_size=7, **kwargs)
    expected = _reference(data, coords, errors, weights, edges, [True, False], True, "uniform")
    for actual, reference in zip((result.binned_data, result.binned_data_errs, result.n_samples),
                                 expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13, equal_nan=True)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_automatic_limits_ignore_rows_with_incomplete_coordinates(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    data = [2.0, 4.0, 100.0, 200.0]
    coordinates = [[0.0, 0.0], [1.0, 1.0], [100.0, np.nan], [-100.0, np.inf]]
    kwargs = dict(num_bins=[2, 2], backend=backend, workers=1)
    direct = rebin_nd(data, coordinates, **kwargs)
    streamed = rebin_nd_stream(ArrayRebinSource(data, coordinates, batch_size=2), **kwargs)
    for result in (direct, streamed):
        np.testing.assert_allclose(result.bin_centers_list, [[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(result.binned_data, [[2.0, np.nan], [np.nan, 4.0]], equal_nan=True)
        np.testing.assert_allclose(result.n_samples, [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("route", ["numpy", "numba", "stream_numpy", "stream_numba"])
@pytest.mark.parametrize("center_lower", [0.0, -1.0, -100.0])
def test_exact_physical_centers_do_not_create_fractional_ghost_bins(route, center_lower):
    if "numba" in route:
        pytest.importorskip("numba")
    edges = center_lower + (np.arange(12) - 0.5) * 0.1
    centers = (edges[:-1] + edges[1:]) / 2.0
    kwargs = dict(lower=[center_lower], upper=[center_lower + 1.0],
                  step_size=[0.1], fractional=True, workers=1)
    for index, center in enumerate(centers):
        if route.startswith("stream_"):
            result = rebin_nd_stream(ArrayRebinSource([4.0], [[center]], data_errs=[2.0]),
                                     backend=route.removeprefix("stream_"), **kwargs)
        else:
            result = rebin_nd([4.0], [[center]], data_errs=[2.0], backend=route, **kwargs)
        expected_samples = np.zeros(11)
        expected_samples[index] = 1.0
        np.testing.assert_array_equal(result.n_samples, expected_samples)
        assert np.isfinite(result.binned_data).sum() == 1
        assert np.isfinite(result.binned_data_errs).sum() == 1
        assert result.binned_data[index] == pytest.approx(4.0)
        assert result.binned_data_errs[index] == pytest.approx(2.0)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_stream_limit_discovery_accepts_empty_batches(backend):
    if backend == "numba":
        pytest.importorskip("numba")

    class Source:
        ndim = 1
        n_points = 2

        def iter_batches(self):
            yield RebinBatch([], np.empty((0, 1)))
            yield RebinBatch([2.0, 4.0], [[0.0], [1.0]])
            yield RebinBatch([], np.empty((0, 1)))

    result = rebin_nd_stream(Source(), num_bins=[2], backend=backend, workers=1)
    np.testing.assert_allclose(result.binned_data, [2.0, 4.0])
    np.testing.assert_allclose(result.n_samples, [1.0, 1.0])


@pytest.mark.parametrize("route", ["numpy", "numba", "stream_numpy", "stream_numba"])
@pytest.mark.parametrize("fractional_axes", [[False, False], [True, False], [True, True]])
def test_explicit_grid_with_no_finite_points_returns_empty_bins(route, fractional_axes):
    if "numba" in route:
        pytest.importorskip("numba")
    data = [2.0, 4.0]
    coordinates = [[np.nan, np.nan], [np.inf, -np.inf]]
    kwargs = dict(bin_edges=[[0.0, 1.0, 3.0], [-2.0, 0.0, 2.0]],
                  fractional_axes=fractional_axes, workers=1)
    if route.startswith("stream_"):
        result = rebin_nd_stream(ArrayRebinSource(data, coordinates, batch_size=1),
                                 backend=route.removeprefix("stream_"), **kwargs)
    else:
        result = rebin_nd(data, coordinates, backend=route, **kwargs)
    assert np.isnan(result.binned_data).all()
    assert np.isnan(result.binned_data_errs).all()
    np.testing.assert_array_equal(result.n_samples, np.zeros((2, 2)))
