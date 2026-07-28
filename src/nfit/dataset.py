from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
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
    magnetic_field
        Applied magnetic field in Tesla (crystal Cartesian frame, a along x),
        as either a single 3-vector shared by all points or a per-point
        ``(n, 3)`` array (a field sweep, e.g. magnetization vs field).
        ``None`` when no field was applied or recorded.
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
    magnetic_field: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _mutable: InitVar[bool] = False
    _arrays_mutable: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self, _mutable: bool) -> None:
        object.__setattr__(self, "H", _as_float_1d("H", self.H, mutable=_mutable))
        object.__setattr__(self, "K", _as_float_1d("K", self.K, mutable=_mutable))
        object.__setattr__(self, "L", _as_float_1d("L", self.L, mutable=_mutable))
        object.__setattr__(self, "E", _as_float_1d("E", self.E, mutable=_mutable))
        object.__setattr__(
            self,
            "intensity",
            _as_float_1d("intensity", self.intensity, mutable=_mutable),
        )
        object.__setattr__(
            self,
            "sigma",
            _as_float_1d("sigma", self.sigma, mutable=_mutable),
        )

        shape = self.H.shape
        for name in ("K", "L", "E", "intensity", "sigma"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match H shape {shape}")

        if self.mask is None:
            mask = np.ones(shape, dtype=bool)
            if not _mutable:
                mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)
        else:
            mask = _as_array(self.mask, dtype=bool, mutable=_mutable)
            if mask.shape != shape:
                raise ValueError(f"mask shape {mask.shape} does not match H shape {shape}")
            object.__setattr__(self, "mask", mask)

        if self.temperature is not None and not np.isscalar(self.temperature):
            temp = _as_float_1d(
                "temperature",
                self.temperature,
                mutable=_mutable,
            )
            if temp.shape != shape:
                raise ValueError(
                    f"temperature shape {temp.shape} does not match H shape {shape}"
                )
            object.__setattr__(self, "temperature", temp)

        if self.magnetic_field is not None:
            field_vector = _as_array(
                self.magnetic_field,
                dtype=float,
                mutable=_mutable,
            )
            if field_vector.shape != (3,) and field_vector.shape != shape + (3,):
                raise ValueError(
                    "magnetic_field must be a Cartesian 3-vector in Tesla or a "
                    f"per-point ({shape[0]}, 3) array; got shape "
                    f"{field_vector.shape}"
                )
            object.__setattr__(self, "magnetic_field", field_vector)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "_arrays_mutable", bool(_mutable))

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
        if isinstance(self.magnetic_field, np.ndarray) and self.magnetic_field.ndim == 2:
            mask &= np.all(np.isfinite(self.magnetic_field), axis=1)
        return mask

    def valid(self, require_positive_sigma: bool = True) -> PointData4D:
        """Return a new data object containing only valid points."""

        mask = self.valid_mask(require_positive_sigma=require_positive_sigma)
        temp: float | FloatArray | None
        if isinstance(self.temperature, np.ndarray):
            temp = self.temperature[mask]
        else:
            temp = self.temperature
        if isinstance(self.magnetic_field, np.ndarray) and self.magnetic_field.ndim == 2:
            field = self.magnetic_field[mask]
        elif self.magnetic_field is None:
            field = None
        else:
            field = np.array(self.magnetic_field)
        return PointData4D(
            H=self.H[mask],
            K=self.K[mask],
            L=self.L[mask],
            E=self.E[mask],
            intensity=self.intensity[mask],
            sigma=self.sigma[mask],
            mask=np.ones(int(np.count_nonzero(mask)), dtype=bool),
            temperature=temp,
            magnetic_field=field,
            metadata=dict(self.metadata),
        )

    def coordinates(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return ``(H, K, L, E)`` arrays."""

        return self.H, self.K, self.L, self.E

    def mutable_copy(self) -> PointData4D:
        """Return an isolated copy whose numeric arrays may be edited in place."""

        return PointData4D(
            H=self.H,
            K=self.K,
            L=self.L,
            E=self.E,
            intensity=self.intensity,
            sigma=self.sigma,
            mask=self.mask,
            temperature=self.temperature,
            magnetic_field=self.magnetic_field,
            metadata=copy.deepcopy(self.metadata),
            _mutable=True,
        )

    def immutable_copy(self) -> PointData4D:
        """Return an immutable copy, or ``self`` when already immutable."""

        arrays = (
            self.H,
            self.K,
            self.L,
            self.E,
            self.intensity,
            self.sigma,
            self.mask,
            self.temperature,
            self.magnetic_field,
        )
        if not self._arrays_mutable and not any(
            isinstance(array, np.ndarray) and array.flags.writeable
            for array in arrays
        ):
            return self
        return self.with_updates()

    def with_updates(self, **changes: Any) -> PointData4D:
        """Return an immutable container with selected fields replaced."""

        values = {
            "H": self.H,
            "K": self.K,
            "L": self.L,
            "E": self.E,
            "intensity": self.intensity,
            "sigma": self.sigma,
            "mask": self.mask,
            "temperature": self.temperature,
            "magnetic_field": self.magnetic_field,
            "metadata": self.metadata,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unknown PointData4D field(s): {', '.join(sorted(unknown))}")
        values.update(changes)
        return PointData4D(**values)


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
    magnetic_field: ArrayLike | None = None,
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
        magnetic_field=magnetic_field,
        metadata={} if metadata is None else dict(metadata),
    )


@dataclass(frozen=True)
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
    quantity_types: dict[str, str] = field(default_factory=dict)
    _mutable: InitVar[bool] = False
    _arrays_mutable: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self, _mutable: bool) -> None:
        columns: dict[str, FloatArray] = {}
        length: int | None = None
        for name, values in self.columns.items():
            arr = _as_float_1d(str(name), values, mutable=_mutable)
            if length is None:
                length = arr.size
            elif arr.size != length:
                raise ValueError(
                    f"column {name!r} has length {arr.size}, expected {length}"
                )
            columns[str(name)] = arr
        object.__setattr__(self, "columns", columns)
        units = {str(key): str(value) for key, value in dict(self.units).items()}
        from .quantities import infer_quantity_type, normalize_unit

        units = {
            name: normalize_unit(units.get(name, "")) for name in self.columns
        }
        declared = {str(key): str(value) for key, value in self.quantity_types.items()}
        quantity_types = {
            name: declared.get(name) or infer_quantity_type(name, units.get(name, ""))
            for name in self.columns
        }
        coordinate_names = list(self.coordinate_names)
        channels = [dict(channel) for channel in self.channels]
        for name in coordinate_names:
            if name not in self.columns:
                raise ValueError(f"coordinate {name!r} is not a known column")
        for channel in channels:
            value_name = channel.get("value")
            if value_name not in self.columns:
                raise ValueError(f"channel value column {value_name!r} is not a known column")
            error_name = channel.get("error")
            if error_name is not None and error_name not in self.columns:
                raise ValueError(f"channel error column {error_name!r} is not a known column")
            channel.setdefault("label", str(value_name))
            channel.setdefault("quantity_type", quantity_types.get(value_name, "unknown"))
            channel.setdefault("unit", units.get(value_name, ""))
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "quantity_types", quantity_types)
        object.__setattr__(self, "coordinate_names", coordinate_names)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "_arrays_mutable", bool(_mutable))

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

    def quantity_type(self, name: str) -> str:
        """Return the physical quantity type declared for a column."""

        return self.quantity_types.get(name, "unknown")

    def channel_quantity_type(self, label: str) -> str:
        """Return the physical quantity type of a dependent channel."""

        channel = self.channel(label)
        return str(channel.get("quantity_type") or self.quantity_type(channel["value"]))

    def mutable_copy(self) -> PointListData:
        """Return an isolated copy whose column arrays may be edited in place."""

        return PointListData(
            columns=self.columns,
            units=dict(self.units),
            coordinate_names=list(self.coordinate_names),
            channels=copy.deepcopy(self.channels),
            metadata=copy.deepcopy(self.metadata),
            quantity_types=dict(self.quantity_types),
            _mutable=True,
        )

    def immutable_copy(self) -> PointListData:
        """Return an immutable copy, or ``self`` when already immutable."""

        if not self._arrays_mutable and not any(
            array.flags.writeable for array in self.columns.values()
        ):
            return self
        return self.with_updates()

    def with_updates(self, **changes: Any) -> PointListData:
        """Return an immutable container with selected fields replaced."""

        values = {
            "columns": self.columns,
            "units": self.units,
            "coordinate_names": self.coordinate_names,
            "channels": self.channels,
            "metadata": self.metadata,
            "quantity_types": self.quantity_types,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unknown PointListData field(s): {', '.join(sorted(unknown))}")
        values.update(changes)
        return PointListData(**values)

    def rebin_to_histogram(
        self,
        coordinate_names: list[str],
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        num_bins: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
        fractional: bool = True,
        normalize: bool = True,
        mean_weighting: str = "inverse_variance",
        max_batch_bytes: int = 192 * 1024 * 1024,
        symmetry_operations: list[ArrayLike] | tuple[ArrayLike, ...] | None = None,
    ) -> PointListData:
        """Bin the points onto a regular grid, returning occupied bin centers.

        Each channel's value and error are binned over the chosen coordinates
        using :func:`nfit.rebin.rebin_nd`. Only bins that received at least
        one point are kept, so the result is a smaller point list suitable for
        reducing the number of fit points.
        """

        from .rebin import rebin_nd, rebin_nd_symmetry

        if not coordinate_names:
            raise ValueError("at least one coordinate is required to rebin")
        for name in coordinate_names:
            if name not in self.columns:
                raise ValueError(f"coordinate {name!r} is not a known column")
        coords = np.column_stack([self.columns[name] for name in coordinate_names])
        if symmetry_operations is not None and coordinate_names[:3] != ["H", "K", "L"]:
            raise ValueError("point-list symmetry requires H, K, and L as the first three rebin coordinates")
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
            kwargs = dict(
                data_errs=errors,
                lower=lower,
                upper=upper,
                num_bins=num_bins,
                step_size=step_size,
                fractional=fractional,
                normalize=normalize,
                mean_weighting=mean_weighting,
                max_batch_bytes=max_batch_bytes,
            )
            result = (
                rebin_nd_symmetry(value, coords[finite], symmetry_operations, **kwargs)
                if symmetry_operations is not None
                else rebin_nd(value, coords[finite], **kwargs)
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
            "mean_weighting": str(mean_weighting),
            "max_batch_bytes": int(max_batch_bytes),
        }
        return PointListData(
            columns=new_columns,
            units=new_units,
            coordinate_names=list(coordinate_names),
            channels=new_channels,
            metadata=metadata,
            quantity_types={
                name: self.quantity_type(name) for name in new_columns
            },
        )


def _as_array(value: ArrayLike, *, dtype: Any, mutable: bool) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if mutable:
        return np.array(arr, dtype=dtype, copy=True)
    if arr.flags.writeable:
        arr = np.array(arr, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _as_float_1d(name: str, value: ArrayLike, *, mutable: bool = False) -> FloatArray:
    arr = _as_array(value, dtype=float, mutable=mutable)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got shape {arr.shape}")
    return arr
