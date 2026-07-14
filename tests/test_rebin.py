import numpy as np
import pytest

from nfit import ArrayRebinSource, NDRebin, RebinBatch, rebin_nd, rebin_nd_stream


def test_rebin_nd_averages_values_and_propagates_errors():
    result = rebin_nd(
        data=[1.0, 3.0, 10.0],
        coords=[0.25, 0.75, 1.75],
        data_errs=[1.0, 1.0, 2.0],
        lower=0.0,
        upper=2.0,
        num_bins=2,
        fractional=False,
    )

    np.testing.assert_allclose(result.binned_data, [2.0, 10.0])
    np.testing.assert_allclose(result.binned_data_errs, [np.sqrt(2.0) / 2.0, 2.0])
    np.testing.assert_allclose(result.n_samples, [2.0, 1.0])
    np.testing.assert_allclose(result.bin_centers_list[0], [0.5, 1.5])


def test_rebin_nd_defaults_to_inverse_variance_weighted_mean():
    result = rebin_nd(
        data=[0.0, 10.0],
        coords=[0.25, 0.75],
        data_errs=[1.0, 3.0],
        lower=0.0,
        upper=1.0,
        num_bins=1,
    )

    np.testing.assert_allclose(result.binned_data, [1.0])
    np.testing.assert_allclose(result.binned_data_errs, [1.0 / np.sqrt(1.0 + 1.0 / 9.0)])
    np.testing.assert_allclose(result.n_samples, [2.0])


def test_rebin_nd_inverse_variance_respects_extra_data_weights():
    result = rebin_nd(
        data=[0.0, 10.0],
        coords=[0.25, 0.75],
        data_errs=[1.0, 1.0],
        data_weights=[1.0, 3.0],
        lower=0.0,
        upper=1.0,
        num_bins=1,
        fractional=False,
    )

    np.testing.assert_allclose(result.binned_data, [7.5])
    np.testing.assert_allclose(result.binned_data_errs, [0.5])
    np.testing.assert_allclose(result.n_samples, [2.0])


def test_rebin_nd_can_use_uniform_mean_for_legacy_averaging():
    result = rebin_nd(
        data=[0.0, 10.0],
        coords=[0.25, 0.75],
        data_errs=[1.0, 3.0],
        lower=0.0,
        upper=1.0,
        num_bins=1,
        mean_weighting="uniform",
    )

    np.testing.assert_allclose(result.binned_data, [5.0])
    np.testing.assert_allclose(result.binned_data_errs, [np.sqrt(10.0) / 2.0])
    np.testing.assert_allclose(result.n_samples, [2.0])


def test_rebin_nd_sums_when_normalize_is_false():
    result = rebin_nd(
        data=[1.0, 3.0, 10.0],
        coords=[0.25, 0.75, 1.75],
        data_errs=[1.0, 1.0, 2.0],
        lower=0.0,
        upper=2.0,
        num_bins=2,
        fractional=False,
        normalize=False,
    )

    np.testing.assert_allclose(result.binned_data, [4.0, 10.0])
    np.testing.assert_allclose(result.binned_data_errs, [np.sqrt(2.0), 2.0])


def test_rebin_nd_defaults_to_fractional_binning():
    result = rebin_nd(
        data=[10.0],
        coords=[1.0],
        lower=0.0,
        upper=2.0,
        num_bins=2,
        normalize=False,
    )

    np.testing.assert_allclose(result.binned_data, [5.0, 5.0])
    np.testing.assert_allclose(result.n_samples, [0.5, 0.5])


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


def test_fractional_rebin_batching_matches_full_accumulation():
    coords = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [1.25, 0.75],
            [1.75, 1.75],
        ]
    )
    data = np.array([1.0, 2.0, 8.0, 16.0])
    errors = np.array([1.0, 2.0, 1.0, 4.0])
    kwargs = dict(
        data=data,
        coords=coords,
        data_errs=errors,
        lower=[0.0, 0.0],
        upper=[2.0, 2.0],
        num_bins=[2, 2],
        fractional=True,
    )

    full = rebin_nd(**kwargs, batch_size=10_000)
    batched = rebin_nd(**kwargs, batch_size=1)

    np.testing.assert_allclose(batched.binned_data, full.binned_data)
    np.testing.assert_allclose(batched.binned_data_errs, full.binned_data_errs)
    np.testing.assert_allclose(batched.n_samples, full.n_samples)


def test_rebin_reports_batch_progress():
    events = []
    result = rebin_nd(
        data=np.arange(6.0),
        coords=np.arange(6.0),
        num_bins=[3],
        batch_size=2,
        fractional=False,
        progress_callback=events.append,
    )

    assert result.binned_data is not None
    assert [event["iteration"] for event in events] == [2, 4, 6]
    assert [event["total"] for event in events] == [6, 6, 6]
    assert all(event["stage"] == "rebin" for event in events)


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
        fractional=False,
    )

    np.testing.assert_allclose(result.binned_data, data)


def test_rebin_requires_bin_definition():
    rebin = NDRebin(data=[1.0], coords=[0.0])
    with pytest.raises(ValueError, match="Either step_size or num_bins"):
        rebin.run()


@pytest.mark.parametrize("ndim", [1, 2, 4, 5])
@pytest.mark.parametrize("fractional", [False, True])
@pytest.mark.parametrize("mean_weighting", ["uniform", "inverse_variance"])
@pytest.mark.parametrize("normalize", [False, True])
def test_numba_rebin_backend_matches_numpy(ndim, fractional, mean_weighting, normalize):
    pytest.importorskip("numba")
    rng = np.random.default_rng(12345)
    coords = rng.uniform(-0.25, 2.25, size=(257, ndim))
    coords[0] = np.zeros(ndim)
    coords[1] = np.full(ndim, 2.0)
    coords[2, 0] = np.nan
    data = rng.normal(size=257)
    errors = rng.uniform(0.1, 2.0, size=257)
    errors[3] = 0.0
    errors[4] = np.nan
    weights = rng.uniform(0.1, 3.0, size=257)
    weights[5] = 0.0
    kwargs = dict(
        data=data,
        coords=coords,
        data_errs=errors,
        data_weights=weights,
        lower=np.zeros(ndim),
        upper=np.full(ndim, 2.0),
        num_bins=[2 + index % 3 for index in range(ndim)],
        fractional=fractional,
        mean_weighting=mean_weighting,
        normalize=normalize,
        batch_size=31,
    )

    numpy_result = rebin_nd(**kwargs, backend="numpy")
    numba_result = rebin_nd(**kwargs, backend="numba")

    assert numba_result.resolved_backend == "numba"
    np.testing.assert_allclose(numba_result.binned_data, numpy_result.binned_data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        numba_result.binned_data_errs,
        numpy_result.binned_data_errs,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(numba_result.n_samples, numpy_result.n_samples, rtol=1e-12, atol=1e-12)


def test_numba_rebin_without_uncertainties_matches_numpy_uniform_fallback():
    pytest.importorskip("numba")
    kwargs = dict(
        data=[1.0, 3.0, 9.0],
        coords=[0.25, 0.75, 1.75],
        lower=[0.0],
        upper=[2.0],
        num_bins=[2],
        fractional=True,
        mean_weighting="inverse_variance",
    )

    numpy_result = rebin_nd(**kwargs, backend="numpy")
    numba_result = rebin_nd(**kwargs, backend="numba")

    np.testing.assert_allclose(numba_result.binned_data, numpy_result.binned_data)
    np.testing.assert_allclose(numba_result.binned_data_errs, numpy_result.binned_data_errs)
    np.testing.assert_allclose(numba_result.n_samples, numpy_result.n_samples)


def test_rebin_auto_backend_keeps_tiny_jobs_on_numpy(monkeypatch):
    import nfit.rebin as rebin_module

    monkeypatch.setattr(rebin_module, "NUMBA_REBIN_MIN_POINTS", 10)
    tiny = rebin_nd(data=np.ones(9), coords=np.arange(9.0), num_bins=[3])
    large = rebin_nd(data=np.ones(10), coords=np.arange(10.0), num_bins=[3])

    assert tiny.resolved_backend == "numpy"
    assert large.resolved_backend == ("numba" if rebin_module._NUMBA_REBIN is not None else "numpy")


def test_fused_backend_does_not_allocate_full_bin_index_array():
    pytest.importorskip("numba")
    result = rebin_nd(
        data=np.ones(100),
        coords=np.linspace(0.0, 1.0, 100),
        num_bins=[10],
        backend="numba",
    )

    assert result.bin_inds is None
    assert result.timings.keys() == {"prepare", "accumulate", "normalize", "total"}


def test_dense_threaded_rebin_matches_serial_and_respects_memory_budget():
    pytest.importorskip("numba")
    rng = np.random.default_rng(99)
    kwargs = dict(
        data=rng.normal(size=1000),
        coords=rng.random((1000, 3)),
        data_errs=rng.uniform(0.5, 2.0, size=1000),
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0],
        num_bins=[8, 9, 10],
        backend="numba",
        batch_size=211,
    )
    serial = rebin_nd(**kwargs, workers=1)
    threaded = rebin_nd(**kwargs, workers=4, parallel_strategy="dense")
    constrained = rebin_nd(
        **kwargs,
        workers=4,
        parallel_strategy="dense",
        max_parallel_bytes=1,
    )

    assert threaded.resolved_workers == 4
    assert threaded.resolved_parallel_strategy == "dense"
    assert constrained.resolved_workers == 1
    assert constrained.resolved_parallel_strategy == "serial"
    np.testing.assert_allclose(threaded.binned_data, serial.binned_data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(threaded.binned_data_errs, serial.binned_data_errs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(threaded.n_samples, serial.n_samples, rtol=1e-12, atol=1e-12)


def test_sparse_threaded_rebin_matches_serial_for_large_sparse_grid():
    pytest.importorskip("numba")
    rng = np.random.default_rng(101)
    coords = rng.uniform(0.0, 0.02, size=(701, 3))
    kwargs = dict(
        data=rng.normal(size=701),
        coords=coords,
        data_errs=rng.uniform(0.5, 1.5, size=701),
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0],
        num_bins=[100, 100, 100],
        backend="numba",
        batch_size=173,
    )
    serial = rebin_nd(**kwargs, workers=1)
    sparse = rebin_nd(**kwargs, workers=3, parallel_strategy="sparse")
    automatic = rebin_nd(
        **kwargs,
        workers=3,
        parallel_strategy="auto",
        max_parallel_bytes=2 * 1024 * 1024,
    )

    assert sparse.resolved_workers == 3
    assert sparse.resolved_parallel_strategy == "sparse"
    assert automatic.resolved_parallel_strategy == "sparse"
    np.testing.assert_allclose(sparse.binned_data, serial.binned_data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sparse.binned_data_errs, serial.binned_data_errs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sparse.n_samples, serial.n_samples, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(automatic.binned_data, serial.binned_data, rtol=1e-12, atol=1e-12)


def test_auto_parallel_plan_prefers_serial_when_sparse_occupancy_is_too_high():
    pytest.importorskip("numba")
    result = rebin_nd(
        data=np.ones(1000),
        coords=np.random.default_rng(4).random((1000, 2)),
        lower=[0.0, 0.0], upper=[1.0, 1.0], num_bins=[20, 20],
        backend="numba", workers=8, max_parallel_bytes=1,
    )

    assert result.resolved_workers == 1
    assert result.resolved_parallel_strategy == "serial"


@pytest.mark.parametrize("fractional", [False, True])
def test_streaming_rebin_matches_in_memory_with_automatic_limits(fractional):
    pytest.importorskip("numba")
    rng = np.random.default_rng(2026)
    coords = rng.normal(size=(503, 3))
    data = rng.normal(size=503)
    errors = rng.uniform(0.2, 2.0, size=503)
    weights = rng.uniform(0.1, 3.0, size=503)
    axes = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    source = ArrayRebinSource(
        data, coords, data_errs=errors, data_weights=weights, batch_size=47
    )
    kwargs = dict(
        axes=axes,
        num_bins=[7, 8, 9],
        fractional=fractional,
        mean_weighting="inverse_variance",
    )

    expected = rebin_nd(
        data, coords, data_errs=errors, data_weights=weights,
        backend="numpy", **kwargs,
    )
    streamed = rebin_nd_stream(
        source, backend="numba", workers=3, parallel_strategy="dense", **kwargs
    )

    assert streamed.bin_inds is None
    assert streamed.resolved_workers == 3
    assert streamed.resolved_parallel_strategy == "dense"
    np.testing.assert_allclose(streamed.binned_data, expected.binned_data, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(streamed.binned_data_errs, expected.binned_data_errs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(streamed.n_samples, expected.n_samples, rtol=1e-12, atol=1e-12)


def test_streaming_explicit_limits_need_only_one_source_pass():
    pytest.importorskip("numba")

    class CountingSource:
        ndim = 1
        n_points = 4

        def __init__(self):
            self.passes = 0

        def iter_batches(self):
            self.passes += 1
            yield RebinBatch([1.0, 3.0], [[0.0], [0.25]], [1.0, 1.0])
            yield RebinBatch([5.0, 7.0], [[0.75], [1.0]], [1.0, 1.0])

    source = CountingSource()
    result = rebin_nd_stream(source, lower=[0.0], upper=[1.0], num_bins=[2])

    assert source.passes == 1
    assert result.Nvals == 4
    assert result.resolved_backend == "numpy"


def test_array_rebin_source_preserves_memmap_batches(tmp_path):
    path = tmp_path / "points.dat"
    values = np.memmap(path, dtype="float64", mode="w+", shape=(100, 3))
    values[:] = np.arange(300.0).reshape(100, 3)
    source = ArrayRebinSource(values[:, 0], values[:, 1:], batch_size=17)

    batches = list(source.iter_batches())

    assert len(batches) == 6
    assert sum(np.asarray(batch.data).size for batch in batches) == 100
