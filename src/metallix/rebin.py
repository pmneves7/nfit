from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


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
        weights. If false, each point contributes to a single bin.
    normalize:
        If true, return weighted averages. If false, return weighted sums, which
        is useful for integrations.

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
        axes: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        lower: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
        num_bins: ArrayLike | None = None,
        fractional: bool = False,
        normalize: bool = True,
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.coords = np.asarray(coords, dtype=float)
        self.data_errs = None if data_errs is None else np.asarray(data_errs, dtype=float)
        self.axes = None if axes is None else np.asarray(axes, dtype=float)
        self.upper = upper
        self.lower = lower
        self.step_size = step_size
        self.num_bins = num_bins
        self.fractional = fractional
        self.normalize = normalize

        self.Nvals: int | None = None
        self.Ndims: int | None = None
        self.dim_axis: int | None = None
        self.data_flat: FloatArray | None = None
        self.errors_flat: FloatArray | None = None
        self.coords_flat: FloatArray | None = None
        self.bins_list: list[FloatArray] | None = None
        self.bin_centers_list: list[FloatArray] | None = None
        self.bin_inds: FloatArray | None = None
        self.binned_data: FloatArray | None = None
        self.binned_data_errs: FloatArray | None = None
        self.n_samples: FloatArray | None = None
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
        assert self.data_flat is not None
        assert self.errors_flat is not None

        valid = ~np.isnan(self.bin_inds).any(axis=1)
        inds_int = self.bin_inds[valid].astype(int)
        num_bins = np.asarray(self.num_bins, dtype=int)
        flat_idx = np.ravel_multi_index(inds_int.T, dims=tuple(num_bins))
        size = int(np.prod(num_bins))

        bd_sum = np.bincount(flat_idx, weights=self.data_flat[valid], minlength=size)
        err_sum = np.bincount(flat_idx, weights=self.errors_flat[valid] ** 2, minlength=size)
        ns_sum = np.bincount(flat_idx, minlength=size)

        self.binned_data = bd_sum.reshape(tuple(num_bins))
        self.binned_data_errs = err_sum.reshape(tuple(num_bins))
        self.n_samples = ns_sum.reshape(tuple(num_bins))

    def _calculate_fractional_bins(self) -> None:
        assert self.Ndims is not None
        assert self.bin_inds is not None
        assert self.num_bins is not None
        assert self.data_flat is not None
        assert self.errors_flat is not None

        num_bins = np.asarray(self.num_bins, dtype=int)
        bin_inds_frac = self.bin_inds - 0.5
        valid = ~np.isnan(bin_inds_frac).any(axis=1)
        valid_inds = bin_inds_frac[valid]
        partial_weights = 1.0 - np.mod(valid_inds, 1)
        data_valid = self.data_flat[valid]
        errs_valid = self.errors_flat[valid]

        for ind in range(self.Ndims):
            edge_mask = ~np.logical_or(
                valid_inds[:, ind] < 0,
                valid_inds[:, ind] > num_bins[ind] - 1,
            )
            partial_weights[~edge_mask, ind] = 1.0

            shifted_inds = valid_inds[edge_mask].copy()
            shifted_inds[:, ind] += 1.0
            valid_inds = np.vstack([valid_inds, shifted_inds])

            shifted_weights = partial_weights[edge_mask].copy()
            shifted_weights[:, ind] = 1.0 - shifted_weights[:, ind]
            partial_weights = np.vstack([partial_weights, shifted_weights])

            data_valid = np.concatenate([data_valid, data_valid[edge_mask]])
            errs_valid = np.concatenate([errs_valid, errs_valid[edge_mask]])

        for ind in range(self.Ndims):
            valid_inds[valid_inds[:, ind] < 0, ind] = 0
            valid_inds[valid_inds[:, ind] > num_bins[ind] - 1, ind] = num_bins[ind] - 1

        weights = np.prod(partial_weights, axis=1)
        inds_int = valid_inds.astype(int)
        flat_idx = np.ravel_multi_index(inds_int.T, dims=tuple(num_bins))
        size = int(np.prod(num_bins))

        bd_sum = np.bincount(flat_idx, weights=weights * data_valid, minlength=size)
        err_sum = np.bincount(
            flat_idx,
            weights=(weights**2) * (errs_valid**2),
            minlength=size,
        )
        ns_sum = np.bincount(flat_idx, weights=weights, minlength=size)

        self.binned_data = bd_sum.reshape(tuple(num_bins))
        self.binned_data_errs = err_sum.reshape(tuple(num_bins))
        self.n_samples = ns_sum.reshape(tuple(num_bins))

    def _norm_data(self) -> None:
        assert self.binned_data is not None
        assert self.binned_data_errs is not None
        assert self.n_samples is not None

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.normalize:
                self.binned_data = np.divide(self.binned_data, self.n_samples)
                self.binned_data_errs = np.divide(np.sqrt(self.binned_data_errs), self.n_samples)
            else:
                self.binned_data_errs = np.sqrt(self.binned_data_errs)

        mask = self.n_samples == 0
        self.binned_data[mask] = np.nan
        self.binned_data_errs[mask] = np.nan


def rebin_nd(*args: Any, **kwargs: Any) -> NDRebin:
    """Create, run, and return an :class:`NDRebin` instance."""

    rebin = NDRebin(*args, **kwargs)
    rebin.run()
    return rebin
