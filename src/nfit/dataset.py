from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class PointData4D:
    """Flattened 4D neutron data for fitting.

    Parameters
    ----------
    H, K, L
        Reciprocal lattice coordinates in reciprocal lattice units (RLU).
    E
        Energy transfer in meV. Positive E is neutron energy loss.
    intensity
        Measured neutron intensity in arbitrary or absolute units.
    sigma
        One-standard-deviation uncertainty of ``intensity`` in the same units.
    mask
        Optional boolean mask. True means the point is eligible for analysis.
    temperature
        Sample temperature in K, either scalar or one value per point.
    metadata
        Free-form metadata. Use this for units, sample, scan, and normalization
        notes; do not hide physics assumptions in code.
    """

    H: ArrayLike
    K: ArrayLike
    L: ArrayLike
    E: ArrayLike
    intensity: ArrayLike
    sigma: ArrayLike
    mask: ArrayLike | None = None
    temperature: float | ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.H = _as_float_1d("H", self.H)
        self.K = _as_float_1d("K", self.K)
        self.L = _as_float_1d("L", self.L)
        self.E = _as_float_1d("E", self.E)
        self.intensity = _as_float_1d("intensity", self.intensity)
        self.sigma = _as_float_1d("sigma", self.sigma)

        shape = self.H.shape
        for name in ("K", "L", "E", "intensity", "sigma"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match H shape {shape}")

        if self.mask is None:
            self.mask = np.ones(shape, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != shape:
                raise ValueError(f"mask shape {self.mask.shape} does not match H shape {shape}")

        if self.temperature is not None and not np.isscalar(self.temperature):
            temp = _as_float_1d("temperature", self.temperature)
            if temp.shape != shape:
                raise ValueError(
                    f"temperature shape {temp.shape} does not match H shape {shape}"
                )
            self.temperature = temp

    @property
    def size(self) -> int:
        """Number of flattened data points."""

        return int(self.H.size)

    def valid_mask(self, require_positive_sigma: bool = True) -> BoolArray:
        """Return mask of finite, usable points.

        By default, nonpositive uncertainties are excluded. The original mask is
        respected and never modified in place.
        """

        mask = np.asarray(self.mask, dtype=bool).copy()
        mask &= np.isfinite(self.H)
        mask &= np.isfinite(self.K)
        mask &= np.isfinite(self.L)
        mask &= np.isfinite(self.E)
        mask &= np.isfinite(self.intensity)
        mask &= np.isfinite(self.sigma)
        if require_positive_sigma:
            mask &= self.sigma > 0.0
        if isinstance(self.temperature, np.ndarray):
            mask &= np.isfinite(self.temperature)
        return mask

    def valid(self, require_positive_sigma: bool = True) -> PointData4D:
        """Return a new data object containing only valid points."""

        mask = self.valid_mask(require_positive_sigma=require_positive_sigma)
        temp: float | FloatArray | None
        if isinstance(self.temperature, np.ndarray):
            temp = self.temperature[mask]
        else:
            temp = self.temperature
        return PointData4D(
            H=self.H[mask],
            K=self.K[mask],
            L=self.L[mask],
            E=self.E[mask],
            intensity=self.intensity[mask],
            sigma=self.sigma[mask],
            mask=np.ones(int(np.count_nonzero(mask)), dtype=bool),
            temperature=temp,
            metadata=dict(self.metadata),
        )

    def coordinates(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return ``(H, K, L, E)`` arrays."""

        return self.H, self.K, self.L, self.E


def from_arrays(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    E: ArrayLike,
    intensity: ArrayLike,
    sigma: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    temperature: float | ArrayLike | None = None,
    metadata: dict[str, Any] | None = None,
) -> PointData4D:
    """Construct :class:`PointData4D` from array-like inputs."""

    return PointData4D(
        H=H,
        K=K,
        L=L,
        E=E,
        intensity=intensity,
        sigma=sigma,
        mask=mask,
        temperature=temperature,
        metadata={} if metadata is None else dict(metadata),
    )


@dataclass
class PointListData:
    """A list of measured points with named coordinate and channel columns.

    This container holds non-gridded, tabular data such as MPMS magnetization or
    powder diffraction: every raw column from the source file is retained in
    ``columns``, with units parsed per column. ``coordinate_names`` designates the
    independent variables and ``channels`` designates the dependent (value,
    error) pairs. Roles are editable later and do not restrict what is stored.

    Parameters
    ----------
    columns
        Mapping of column name to a one-dimensional float array. All columns must
        share the same length ``N`` (the number of points).
    units
        Optional mapping of column name to a unit string (e.g. ``"K"``, ``"Oe"``,
        ``"emu"``). Columns without an entry are treated as unitless.
    coordinate_names
        Names of the columns to treat as independent coordinates.
    channels
        Dependent-variable definitions. Each entry is a dict with keys
        ``"label"`` (display name), ``"value"`` (a column name), and ``"error"``
        (a column name or ``None``).
    metadata
        Free-form metadata, including the source-file header block.
    """

    columns: dict[str, ArrayLike]
    units: dict[str, str] = field(default_factory=dict)
    coordinate_names: list[str] = field(default_factory=list)
    channels: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        columns: dict[str, FloatArray] = {}
        length: int | None = None
        for name, values in self.columns.items():
            arr = _as_float_1d(str(name), values)
            if length is None:
                length = arr.size
            elif arr.size != length:
                raise ValueError(
                    f"column {name!r} has length {arr.size}, expected {length}"
                )
            columns[str(name)] = arr
        self.columns = columns
        self.units = {str(key): str(value) for key, value in dict(self.units).items()}
        for name in self.coordinate_names:
            if name not in self.columns:
                raise ValueError(f"coordinate {name!r} is not a known column")
        for channel in self.channels:
            value_name = channel.get("value")
            if value_name not in self.columns:
                raise ValueError(f"channel value column {value_name!r} is not a known column")
            error_name = channel.get("error")
            if error_name is not None and error_name not in self.columns:
                raise ValueError(f"channel error column {error_name!r} is not a known column")
            channel.setdefault("label", str(value_name))

    @property
    def size(self) -> int:
        """Number of points."""

        if not self.columns:
            return 0
        return int(next(iter(self.columns.values())).size)

    @property
    def column_names(self) -> list[str]:
        """All stored column names in insertion order."""

        return list(self.columns)

    def column(self, name: str) -> FloatArray:
        """Return one column by name."""

        return self.columns[name]

    def coordinate(self, name: str) -> FloatArray:
        """Return one coordinate column by name."""

        if name not in self.coordinate_names:
            raise KeyError(f"{name!r} is not a coordinate")
        return self.columns[name]

    @property
    def channel_labels(self) -> list[str]:
        """Labels of the defined channels."""

        return [str(channel["label"]) for channel in self.channels]

    def channel(self, label: str) -> dict[str, Any]:
        """Return the channel definition with the given label."""

        for channel in self.channels:
            if str(channel["label"]) == label:
                return channel
        raise KeyError(f"unknown channel {label!r}")

    def channel_values(self, label: str) -> FloatArray:
        """Return a channel's value column."""

        return self.columns[self.channel(label)["value"]]

    def channel_errors(self, label: str) -> FloatArray | None:
        """Return a channel's error column, or ``None`` when undefined."""

        error_name = self.channel(label).get("error")
        return None if error_name is None else self.columns[error_name]

    def unit(self, name: str) -> str:
        """Return the unit for a column, or an empty string when unknown."""

        return self.units.get(name, "")

    def rebin_to_histogram(
        self,
        coordinate_names: list[str],
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        num_bins: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
        fractional: bool = False,
        normalize: bool = True,
    ) -> "PointListData":
        """Bin the points onto a regular grid, returning occupied bin centers.

        Each channel's value and error are binned over the chosen coordinates
        using :func:`nfit.rebin.rebin_nd`. Only bins that received at least
        one point are kept, so the result is a smaller point list suitable for
        reducing the number of fit points.
        """

        from .rebin import rebin_nd

        if not coordinate_names:
            raise ValueError("at least one coordinate is required to rebin")
        for name in coordinate_names:
            if name not in self.columns:
                raise ValueError(f"coordinate {name!r} is not a known column")
        coords = np.column_stack([self.columns[name] for name in coordinate_names])
        finite = np.all(np.isfinite(coords), axis=1)
        if not np.any(finite):
            raise ValueError("no finite coordinate points remain before rebinning")

        n_samples: FloatArray | None = None
        bin_centers_list: list[FloatArray] | None = None
        binned_channels: list[tuple[dict[str, Any], FloatArray, FloatArray | None]] = []
        for channel in self.channels:
            value = self.columns[channel["value"]][finite]
            error_name = channel.get("error")
            errors = self.columns[error_name][finite] if error_name is not None else None
            result = rebin_nd(
                value,
                coords[finite],
                data_errs=errors,
                lower=lower,
                upper=upper,
                num_bins=num_bins,
                step_size=step_size,
                fractional=fractional,
                normalize=normalize,
            )
            if result.binned_data is None or result.n_samples is None or result.bin_centers_list is None:
                raise RuntimeError("rebinning did not produce binned data")
            if n_samples is None:
                n_samples = np.asarray(result.n_samples, dtype=float)
                bin_centers_list = [np.asarray(c, dtype=float) for c in result.bin_centers_list]
            binned_channels.append(
                (
                    channel,
                    np.asarray(result.binned_data, dtype=float),
                    None if result.binned_data_errs is None else np.asarray(result.binned_data_errs, dtype=float),
                )
            )

        assert n_samples is not None and bin_centers_list is not None
        occupied = n_samples > 0.0
        center_grids = np.meshgrid(*bin_centers_list, indexing="ij")
        new_columns: dict[str, FloatArray] = {}
        new_units: dict[str, str] = {}
        for name, grid in zip(coordinate_names, center_grids, strict=True):
            new_columns[name] = np.asarray(grid, dtype=float)[occupied]
            if name in self.units:
                new_units[name] = self.units[name]
        new_channels: list[dict[str, Any]] = []
        for channel, binned_value, binned_error in binned_channels:
            value_name = str(channel["value"])
            new_columns[value_name] = binned_value[occupied]
            if value_name in self.units:
                new_units[value_name] = self.units[value_name]
            error_name = channel.get("error")
            if error_name is not None and binned_error is not None:
                new_columns[str(error_name)] = binned_error[occupied]
                if error_name in self.units:
                    new_units[str(error_name)] = self.units[str(error_name)]
            new_channels.append(dict(channel))

        metadata = dict(self.metadata)
        metadata["rebin"] = {
            "coordinate_names": list(coordinate_names),
            "num_bins": list(n_samples.shape),
            "fractional": bool(fractional),
            "normalize": bool(normalize),
        }
        return PointListData(
            columns=new_columns,
            units=new_units,
            coordinate_names=list(coordinate_names),
            channels=new_channels,
            metadata=metadata,
        )


def _as_float_1d(name: str, value: ArrayLike) -> FloatArray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got shape {arr.shape}")
    return arr

