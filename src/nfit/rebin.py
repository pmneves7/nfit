from __future__ import annotations

from itertools import product
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

MeanWeighting = Literal["inverse_variance", "uniform"]


class NDRebin:
    """N-dimensional rebinning into regular bins.

    This is adapted from the ``NDRebin`` implementation originally written for
    SasView's ``sasdata.transforms.NDrebin`` module, with the SasView
    ``Quantity`` dependency removed so the class accepts ordinary array-like
    inputs.

    Parameters
    ----------
    data:
        Data values in any array shape.
    coords:
        Coordinates for each data point. One axis must have length equal to the
        dimensionality of the coordinate space; all remaining axes must match
        ``data`` after flattening.
    data_errs:
        Optional one-sigma uncertainties with the same shape as ``data``.
    data_weights:
        Optional positive statistical weights with the same shape as ``data``.
        These multiply the averaging weights. With inverse-variance averaging,
        the effective point weight is ``data_weights / data_errs**2``.
    axes:
        Optional coordinate axes to project onto before binning. Defaults to the
        identity basis.
    upper, lower:
        Optional upper and lower limits for each coordinate dimension. Missing,
        NaN, or infinite limits are replaced with the corresponding data limits.
    step_size:
        Optional bin width for each coordinate dimension. Takes precedence over
        ``num_bins``.
    num_bins:
        Optional number of bins for each coordinate dimension. Required when
        ``step_size`` is not supplied.
    fractional:
        If true, distribute each point to neighboring bins using multilinear
        weights. If false, each point contributes to a single bin. Defaults to
        true.
    normalize:
        If true, return averages. If false, return sums, which is useful for
        integrations.
    mean_weighting:
        Averaging mode used when ``normalize`` is true. ``"inverse_variance"``
        uses ``1 / data_errs**2`` weights when uncertainties are available;
        points with non-positive or non-finite uncertainties are skipped because
        they do not define inverse-variance weights. ``"uniform"`` keeps the
        previous simple mean behavior. Fractional binning multiplies either mean
        weight by the fractional spatial contribution.
    batch_size:
        Optional maximum number of source points processed in one accumulation
        batch. ``None`` chooses a batch size from ``max_batch_bytes``.
    max_batch_bytes:
        Approximate per-batch working-memory target used when ``batch_size`` is
        not supplied. The rebinner accumulates directly into output arrays, so
        fractional mode no longer materializes the full expanded contribution
        list for all source points at once.

    Notes
    -----
    The source implementation is distributed by SasView under the BSD-3-Clause
    license; see ``THIRD_PARTY_LICENSES.md`` for the full notice.
    """

    def __init__(
        self,
        data: ArrayLike,
        coords: ArrayLike,
        data_errs: ArrayLike | None = None,
        data_weights: ArrayLike | None = None,
        axes: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        lower: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
        num_bins: ArrayLike | None = None,
        fractional: bool = True,
        normalize: bool = True,
        mean_weighting: MeanWeighting = "inverse_variance",
        batch_size: int | None = None,
        max_batch_bytes: int = 192 * 1024 * 1024,
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.coords = np.asarray(coords, dtype=float)
        self.data_errs = None if data_errs is None else np.asarray(data_errs, dtype=float)
        self.data_weights = None if data_weights is None else np.asarray(data_weights, dtype=float)
        self.axes = None if axes is None else np.asarray(axes, dtype=float)
        self.upper = upper
        self.lower = lower
        self.step_size = step_size
        self.num_bins = num_bins
        self.fractional = fractional
        self.normalize = normalize
        self.mean_weighting = mean_weighting
        self.batch_size = batch_size
        self.max_batch_bytes = int(max_batch_bytes)

        self.Nvals: int | None = None
        self.Ndims: int | None = None
        self.dim_axis: int | None = None
        self.has_data_errs = data_errs is not None
        self.data_flat: FloatArray | None = None
        self.errors_flat: FloatArray | None = None
        self.weights_flat: FloatArray | None = None
        self.coords_flat: FloatArray | None = None
        self.bins_list: list[FloatArray] | None = None
        self.bin_centers_list: list[FloatArray] | None = None
        self.bin_inds: FloatArray | None = None
        self.binned_data: FloatArray | None = None
        self.binned_data_errs: FloatArray | None = None
        self.n_samples: FloatArray | None = None
        self._normalization: FloatArray | None = None
        self._prepared = False

    def __call__(self) -> None:
        self.run()

    def run(self) -> None:
        """Bin the data into the defined grid."""

        if not self._prepared:
            self._prepare()

        if self.fractional:
            self._calculate_fractional_bins()
        else:
            self._calculate_bins()
        self._norm_data()

    def _prepare(self) -> None:
        if self._prepared:
            return

        self._check_data_coords()
        self._check_data_errs()
        self._check_data_weights()
        self._check_options()
        self._flatten_coords()

        if self.axes is None:
            self._make_axes()
        else:
            self._project_axes()

        self._build_limits()
        self._make_bins()
        self._create_bin_inds()
        self._prepared = True

    def _check_data_coords(self) -> None:
        self.Nvals = int(self.data.size)
        if self.Nvals == 0:
            raise ValueError("data must contain at least one value")

        ndims = self.coords.size / self.Nvals
        if not float(ndims).is_integer():
            raise ValueError(
                "coords must have the same shape as data plus one coordinate dimension"
            )
        self.Ndims = int(ndims)
        if self.Ndims < 1:
            raise ValueError("coords must describe at least one coordinate dimension")

    def _check_data_errs(self) -> None:
        self.data_flat = self.data.reshape(-1)
        if self.data_errs is None:
            self.errors_flat = np.zeros_like(self.data_flat)
        else:
            self.errors_flat = self.data_errs.reshape(-1)

        if self.errors_flat.shape != self.data_flat.shape:
            raise ValueError("data_errs must have the same shape as data")

    def _check_data_weights(self) -> None:
        assert self.data_flat is not None
        if self.data_weights is None:
            self.weights_flat = np.ones(self.data_flat.shape, dtype=float)
            return
        self.weights_flat = np.asarray(self.data_weights, dtype=float).ravel()
        if self.weights_flat.shape != self.data_flat.shape:
            raise ValueError("data_weights must have the same shape as data")

    def _check_options(self) -> None:
        if self.mean_weighting not in {"inverse_variance", "uniform"}:
            raise ValueError("mean_weighting must be 'inverse_variance' or 'uniform'")
        if self.batch_size is not None and int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive")

    def _flatten_coords(self) -> None:
        assert self.Ndims is not None

        if self.Ndims == 1:
            self.coords = self.coords.reshape(-1, 1)

        if self.coords.ndim == self.data.ndim + 1 and self.coords.shape[-1] == self.Ndims:
            if self.coords.shape[:-1] == self.data.shape:
                self.dim_axis = -1
            elif self.coords.shape[0] == self.Ndims and self.coords.shape[1:] == self.data.shape:
                self.dim_axis = 0
            else:
                self.dim_axis = None
        elif self.coords.shape[0] == self.Ndims:
            self.dim_axis = 0
        elif self.coords.shape[-1] == self.Ndims:
            self.dim_axis = -1
        else:
            self.dim_axis = next(
                (i for i, size in enumerate(self.coords.shape) if size == self.Ndims),
                None,
            )

        if self.dim_axis is None:
            raise ValueError("coords must have one axis with length equal to Ndims")

        moved = np.moveaxis(self.coords, self.dim_axis, 0)
        self.coords_flat = moved.reshape(self.Ndims, -1).T

    def _make_axes(self) -> None:
        assert self.Ndims is not None
        self.axes = np.eye(self.Ndims)

    def _project_axes(self) -> None:
        assert self.coords_flat is not None
        assert self.Ndims is not None
        if self.axes is None or self.axes.shape != (self.Ndims, self.Ndims):
            raise ValueError("axes must be a square array with shape (Ndims, Ndims)")
        axes_inv = np.linalg.inv(self.axes)
        self.coords_flat = np.tensordot(self.coords_flat, axes_inv, axes=([1], [0]))

    def _build_limits(self) -> None:
        assert self.coords_flat is not None
        assert self.Ndims is not None

        mins = np.min(self.coords_flat, axis=0)
        maxs = np.max(self.coords_flat, axis=0)
        lower = mins if self.lower is None else self.lower
        upper = maxs if self.upper is None else self.upper

        lower_arr = np.atleast_1d(lower).astype(float, copy=True)
        upper_arr = np.atleast_1d(upper).astype(float, copy=True)

        if lower_arr.size != self.Ndims:
            raise ValueError("lower must be None or a 1D iterable of length Ndims")
        if upper_arr.size != self.Ndims:
            raise ValueError("upper must be None or a 1D iterable of length Ndims")

        lower_arr = np.where(np.isfinite(lower_arr), lower_arr, mins)
        upper_arr = np.where(np.isfinite(upper_arr), upper_arr, maxs)

        self.lower = np.minimum(lower_arr, upper_arr)
        self.upper = np.maximum(lower_arr, upper_arr)

    def _make_bins(self) -> None:
        if self.step_size is None and self.num_bins is None:
            raise ValueError("Either step_size or num_bins must be provided")

        self.bins_list = []
        self.bin_centers_list = []
        if self.step_size is None:
            self._step_size_from_num_bins()
        else:
            self._num_bins_from_step_size()

    def _step_size_from_num_bins(self) -> None:
        assert self.Ndims is not None
        assert self.bins_list is not None
        assert self.bin_centers_list is not None
        assert self.lower is not None
        assert self.upper is not None

        num_bins = np.atleast_1d(self.num_bins).astype(int, copy=False)
        if num_bins.size != self.Ndims:
            raise ValueError("num_bins must be a 1D iterable of length Ndims")
        if np.any(num_bins < 1):
            raise ValueError("num_bins values must be positive")

        step_size: list[float] = []
        for ind in range(self.Ndims):
            these_bins = np.linspace(self.lower[ind], self.upper[ind], int(num_bins[ind]) + 1)
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_step_size = these_bins[1] - these_bins[0]

            self.bins_list.append(these_bins)
            self.bin_centers_list.append(these_centers)
            step_size.append(float(this_step_size))

        self.num_bins = num_bins
        self.step_size = np.asarray(step_size, dtype=float)

    def _num_bins_from_step_size(self) -> None:
        assert self.Ndims is not None
        assert self.bins_list is not None
        assert self.bin_centers_list is not None
        assert self.lower is not None
        assert self.upper is not None

        step_size = np.atleast_1d(self.step_size).astype(float, copy=False)
        if step_size.size != self.Ndims:
            raise ValueError("step_size must be a 1D iterable of length Ndims")
        if np.any(step_size <= 0):
            raise ValueError("step_size values must be positive")

        num_bins: list[int] = []
        for ind in range(self.Ndims):
            if self.lower[ind] == self.upper[ind]:
                these_bins = np.array([self.lower[ind], self.lower[ind]])
            elif np.isinf(step_size[ind]):
                these_bins = np.array([self.lower[ind], self.upper[ind]])
            else:
                these_bins = np.arange(self.lower[ind], self.upper[ind], step_size[ind])
                if these_bins.size == 0 or these_bins[0] != self.lower[ind]:
                    these_bins = np.insert(these_bins, 0, self.lower[ind])
                if these_bins[-1] != self.upper[ind]:
                    these_bins = np.append(these_bins, self.upper[ind])

            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_num_bins = int(these_bins.size - 1)

            self.bins_list.append(these_bins)
            self.bin_centers_list.append(these_centers)
            num_bins.append(this_num_bins)

        self.step_size = step_size
        self.num_bins = np.asarray(num_bins, dtype=int)

    def _create_bin_inds(self) -> None:
        assert self.Nvals is not None
        assert self.Ndims is not None
        assert self.coords_flat is not None
        assert self.bins_list is not None
        assert self.step_size is not None
        assert self.num_bins is not None

        step_size = np.asarray(self.step_size, dtype=float)
        num_bins = np.asarray(self.num_bins, dtype=int)
        self.bin_inds = np.zeros((self.Nvals, self.Ndims), dtype=float)

        for ind in range(self.Ndims):
            this_min = self.bins_list[ind][0]
            this_step = step_size[ind]
            if np.isinf(this_step):
                self.bin_inds[:, ind] = 0
            else:
                self.bin_inds[:, ind] = (self.coords_flat[:, ind] - this_min) / this_step
            self.bin_inds[self.coords_flat[:, ind] < self.bins_list[ind][0], ind] = np.nan
            self.bin_inds[self.coords_flat[:, ind] == self.bins_list[ind][-1], ind] = (
                num_bins[ind] - 1
            )
            self.bin_inds[self.coords_flat[:, ind] > self.bins_list[ind][-1], ind] = np.nan

    def _calculate_bins(self) -> None:
        assert self.bin_inds is not None
        assert self.num_bins is not None

        num_bins = np.asarray(self.num_bins, dtype=int)
        size = int(np.prod(num_bins))
        bd_sum, err_sum, norm_sum, ns_sum = self._empty_accumulators(size)

        for start, stop in self._batch_ranges(fractional=False):
            inds = self.bin_inds[start:stop]
            valid = ~np.isnan(inds).any(axis=1)
            if not np.any(valid):
                continue
            inds_int = inds[valid].astype(int)
            flat_idx = np.ravel_multi_index(inds_int.T, dims=tuple(num_bins))
            point_indices = np.arange(start, stop, dtype=int)[valid]
            spatial_weights = np.ones(point_indices.size, dtype=float)
            self._accumulate_contributions(
                flat_idx,
                point_indices,
                spatial_weights,
                bd_sum,
                err_sum,
                norm_sum,
                ns_sum,
            )

        self._store_accumulators(bd_sum, err_sum, norm_sum, ns_sum)

    def _calculate_fractional_bins(self) -> None:
        assert self.Ndims is not None
        assert self.bin_inds is not None
        assert self.num_bins is not None

        num_bins = np.asarray(self.num_bins, dtype=int)
        size = int(np.prod(num_bins))
        bd_sum, err_sum, norm_sum, ns_sum = self._empty_accumulators(size)

        for start, stop in self._batch_ranges(fractional=True):
            valid_inds = self.bin_inds[start:stop] - 0.5
            valid = ~np.isnan(valid_inds).any(axis=1)
            if not np.any(valid):
                continue
            valid_inds = valid_inds[valid]
            point_indices = np.arange(start, stop, dtype=int)[valid]
            partial_weights = 1.0 - np.mod(valid_inds, 1)
            edge_masks = [
                ~np.logical_or(
                    valid_inds[:, ind] < 0,
                    valid_inds[:, ind] > num_bins[ind] - 1,
                )
                for ind in range(self.Ndims)
            ]

            for offsets in product((0, 1), repeat=self.Ndims):
                contribution_valid = np.ones(valid_inds.shape[0], dtype=bool)
                spatial_weights = np.ones(valid_inds.shape[0], dtype=float)
                contribution_inds = np.empty(valid_inds.shape, dtype=int)
                for ind, offset in enumerate(offsets):
                    edge_mask = edge_masks[ind]
                    if offset:
                        contribution_valid &= edge_mask
                        spatial_weights *= 1.0 - partial_weights[:, ind]
                        dim_inds = valid_inds[:, ind] + 1.0
                    else:
                        spatial_weights *= np.where(edge_mask, partial_weights[:, ind], 1.0)
                        dim_inds = valid_inds[:, ind]
                    dim_inds = np.clip(dim_inds, 0.0, float(num_bins[ind] - 1))
                    contribution_inds[:, ind] = dim_inds.astype(int)
                contribution_valid &= spatial_weights != 0.0
                if not np.any(contribution_valid):
                    continue
                flat_idx = np.ravel_multi_index(
                    contribution_inds[contribution_valid].T,
                    dims=tuple(num_bins),
                )
                self._accumulate_contributions(
                    flat_idx,
                    point_indices[contribution_valid],
                    spatial_weights[contribution_valid],
                    bd_sum,
                    err_sum,
                    norm_sum,
                    ns_sum,
                )

        self._store_accumulators(bd_sum, err_sum, norm_sum, ns_sum)

    def _empty_accumulators(self, size: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        return (
            np.zeros(size, dtype=float),
            np.zeros(size, dtype=float),
            np.zeros(size, dtype=float),
            np.zeros(size, dtype=float),
        )

    def _batch_ranges(self, *, fractional: bool):
        assert self.Nvals is not None
        batch_size = self._resolved_batch_size(fractional=fractional)
        for start in range(0, self.Nvals, batch_size):
            yield start, min(start + batch_size, self.Nvals)

    def _resolved_batch_size(self, *, fractional: bool) -> int:
        assert self.Nvals is not None
        assert self.Ndims is not None
        if self.batch_size is not None:
            return min(int(self.batch_size), self.Nvals)
        if self.max_batch_bytes <= 0:
            return self.Nvals
        contribution_factor = 2 ** self.Ndims if fractional else 1
        bytes_per_point = 8 * max(8, self.Ndims * (6 + contribution_factor))
        return max(1, min(self.Nvals, int(self.max_batch_bytes // bytes_per_point)))

    def _store_accumulators(
        self,
        bd_sum: FloatArray,
        err_sum: FloatArray,
        norm_sum: FloatArray,
        ns_sum: FloatArray,
    ) -> None:
        assert self.num_bins is not None
        shape = tuple(np.asarray(self.num_bins, dtype=int))
        self.binned_data = bd_sum.reshape(shape)
        self.binned_data_errs = err_sum.reshape(shape)
        self._normalization = norm_sum.reshape(shape)
        self.n_samples = ns_sum.reshape(shape)

    def _accumulate_contributions(
        self,
        flat_idx: NDArray[np.int_],
        point_indices: NDArray[np.int_],
        spatial_weights: FloatArray,
        bd_sum: FloatArray,
        err_sum: FloatArray,
        norm_sum: FloatArray,
        ns_sum: FloatArray,
    ) -> None:
        assert self.data_flat is not None
        assert self.errors_flat is not None
        assert self.weights_flat is not None

        valid = np.isfinite(spatial_weights)
        statistical_weights = self.weights_flat[point_indices]
        valid &= np.isfinite(statistical_weights) & (statistical_weights > 0.0)
        if self._use_inverse_variance_weights():
            errors = self.errors_flat[point_indices]
            valid &= np.isfinite(errors) & (errors > 0.0)
            if not np.any(valid):
                return
            inv_var = statistical_weights[valid] / (errors[valid] ** 2)
            mean_weights = spatial_weights[valid] * inv_var
            data_weights = mean_weights * self.data_flat[point_indices[valid]]
            err_weights = mean_weights
            norm_weights = mean_weights
            sample_weights = spatial_weights[valid]
            flat_idx = flat_idx[valid]
        else:
            if not np.any(valid):
                return
            errors = self.errors_flat[point_indices[valid]]
            sample_weights = spatial_weights[valid]
            norm_weights = sample_weights * statistical_weights[valid]
            data_weights = norm_weights * self.data_flat[point_indices[valid]]
            err_weights = (norm_weights**2) * (errors**2)
            flat_idx = flat_idx[valid]

        minlength = bd_sum.size
        bd_sum += np.bincount(flat_idx, weights=data_weights, minlength=minlength)
        err_sum += np.bincount(flat_idx, weights=err_weights, minlength=minlength)
        norm_sum += np.bincount(flat_idx, weights=norm_weights, minlength=minlength)
        ns_sum += np.bincount(flat_idx, weights=sample_weights, minlength=minlength)

    def _use_inverse_variance_weights(self) -> bool:
        return bool(
            self.normalize
            and self.mean_weighting == "inverse_variance"
            and self.has_data_errs
        )

    def _norm_data(self) -> None:
        assert self.binned_data is not None
        assert self.binned_data_errs is not None
        assert self.n_samples is not None
        assert self._normalization is not None

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.normalize:
                self.binned_data = np.divide(self.binned_data, self._normalization)
                self.binned_data_errs = np.divide(np.sqrt(self.binned_data_errs), self._normalization)
            else:
                self.binned_data_errs = np.sqrt(self.binned_data_errs)

        mask = self.n_samples == 0
        if self.normalize:
            mask |= self._normalization == 0
        self.binned_data[mask] = np.nan
        self.binned_data_errs[mask] = np.nan


def rebin_nd(*args: Any, **kwargs: Any) -> NDRebin:
    """Create, run, and return an :class:`NDRebin` instance."""

    rebin = NDRebin(*args, **kwargs)
    rebin.run()
    return rebin
