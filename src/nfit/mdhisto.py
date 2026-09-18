from __future__ import annotations

import copy
import re
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .axes import AxisRole, infer_axis_role

AxisKind = Literal["momentum", "energy", "unknown"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MDHistoChannel:
    values: FloatArray
    errors: FloatArray | None = None
    label: str = ""
    unit: str = ""
    quantity_type: str = "unknown"
    _mutable: InitVar[bool] = False
    _arrays_mutable: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self, _mutable: bool) -> None:
        object.__setattr__(
            self,
            "values",
            _as_array(self.values, dtype=float, mutable=_mutable),
        )
        if self.errors is not None:
            object.__setattr__(
                self,
                "errors",
                _as_array(self.errors, dtype=float, mutable=_mutable),
            )
        from .quantities import QUANTITY_TYPES, normalize_unit

        object.__setattr__(self, "unit", normalize_unit(self.unit))
        if self.quantity_type not in QUANTITY_TYPES:
            raise ValueError(f"unknown channel quantity type {self.quantity_type!r}")
        object.__setattr__(self, "_arrays_mutable", bool(_mutable))

    def mutable_copy(self) -> MDHistoChannel:
        return MDHistoChannel(
            self.values,
            self.errors,
            self.label,
            self.unit,
            self.quantity_type,
            _mutable=True,
        )

    def immutable_copy(self) -> MDHistoChannel:
        if (
            not self._arrays_mutable
            and not self.values.flags.writeable
            and (self.errors is None or not self.errors.flags.writeable)
        ):
            return self
        return MDHistoChannel(
            self.values,
            self.errors,
            self.label,
            self.unit,
            self.quantity_type,
        )


@dataclass(frozen=True)
class MDHistoAxis:
    """One binned reduced-data axis from a Mantid MDHistoWorkspace."""

    name: str
    values: FloatArray
    units: str
    kind: AxisKind
    frame: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize legacy metadata without changing stable axis names."""

        object.__setattr__(
            self,
            "values",
            _as_array(self.values, dtype=float, mutable=False),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        compact_name = self.name.casefold().replace("_", "").replace(" ", "")
        compact_unit = self.units.casefold().replace("_", "").replace(" ", "")
        if (
            self.kind == "energy" or compact_name in {"deltae", "energytransfer"}
        ) and compact_unit in {"deltae", "energy", "energytransfer"}:
            object.__setattr__(self, "units", "meV")

    @property
    def centers(self) -> FloatArray:
        """Return bin centers when ``values`` are bin boundaries."""

        if "discrete_centers" in self.metadata:
            centers = np.asarray(self.metadata["discrete_centers"], dtype=float)
            if (
                centers.shape != (self.values.size - 1,)
                or not np.all(np.isfinite(centers))
                or np.any(np.diff(centers) <= 0)
            ):
                raise ValueError(
                    "discrete axis coordinates must be finite, increasing, and match the bins"
                )
            return centers.copy()
        if self.values.size < 2:
            return self.values.copy()
        return 0.5 * (self.values[:-1] + self.values[1:])

    @property
    def role(self) -> AxisRole:
        """Reduced-data role inferred from axis name, units, and kind."""

        if "metadata_dimension" in self.metadata:
            return "unknown"
        return infer_axis_role(self.name, units=self.units, kind=self.kind)

    def immutable_copy(self) -> MDHistoAxis:
        """Return an axis with a read-only coordinate array."""

        if not self.values.flags.writeable:
            return self
        return MDHistoAxis(
            self.name,
            self.values,
            self.units,
            self.kind,
            self.frame,
            self.path,
            copy.deepcopy(self.metadata),
        )


@dataclass(frozen=True)
class MDHistoData:
    """Imported binned reduced data from a Mantid MDHistoWorkspace.

    ``axes`` are ordered to match the array dimensions of ``signal``, ``errors``,
    ``mask``, and ``num_events``. Mantid stores the mask as bin flags; this class
    preserves those flags as booleans rather than inverting their meaning.
    Importers may additionally mask invalid numerical bins.
    """

    axes: tuple[MDHistoAxis, ...]
    signal: FloatArray
    errors: FloatArray
    mask: BoolArray
    num_events: FloatArray
    coordinate_system: int | None = None
    visual_normalization: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary_channels: dict[str, MDHistoChannel] = field(default_factory=dict)
    _mutable: InitVar[bool] = False
    _arrays_mutable: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self, _mutable: bool) -> None:
        object.__setattr__(
            self,
            "signal",
            _as_array(self.signal, dtype=float, mutable=_mutable),
        )
        object.__setattr__(
            self,
            "errors",
            _as_array(self.errors, dtype=float, mutable=_mutable),
        )
        object.__setattr__(
            self,
            "mask",
            _as_array(self.mask, dtype=bool, mutable=_mutable),
        )
        object.__setattr__(
            self,
            "num_events",
            _as_array(self.num_events, dtype=float, mutable=_mutable),
        )
        object.__setattr__(
            self,
            "axes",
            tuple(axis.immutable_copy() for axis in self.axes),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        channels = {
            str(key): (
                channel.mutable_copy()
                if _mutable
                else channel.immutable_copy()
            )
            for key, channel in self.auxiliary_channels.items()
        }
        object.__setattr__(self, "auxiliary_channels", channels)

        shape = self.signal.shape
        for name in ("errors", "mask", "num_events"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match signal shape {shape}")
        if len(self.axes) != self.signal.ndim:
            raise ValueError(
                f"{len(self.axes)} axes supplied for signal with {self.signal.ndim} dimensions"
            )
        for axis, size in zip(self.axes, self.signal.shape, strict=True):
            if axis.values.size not in (size, size + 1):
                raise ValueError(
                    f"axis {axis.name!r} has {axis.values.size} values for dimension size {size}"
                )
        for key, channel in self.auxiliary_channels.items():
            if channel.values.shape != shape:
                raise ValueError(f"auxiliary channel {key!r} shape does not match signal shape")
            if channel.errors is not None and channel.errors.shape != shape:
                raise ValueError(f"auxiliary channel {key!r} error shape does not match signal shape")
        object.__setattr__(self, "_arrays_mutable", bool(_mutable))

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the imported binned arrays."""

        return self.signal.shape

    def channel_unit(self, name: str = "signal") -> str:
        """Return the explicit unit for the primary or an auxiliary channel."""

        if name == "signal":
            return str(self.metadata.get("signal_unit", ""))
        return self.auxiliary_channels[name].unit

    def channel_quantity_type(self, name: str = "signal") -> str:
        """Return the physical quantity type for one channel."""

        if name == "signal":
            return str(self.metadata.get("signal_quantity_type", "unknown"))
        return self.auxiliary_channels[name].quantity_type

    def mutable_copy(self) -> MDHistoData:
        """Return an isolated copy whose channel arrays may be edited in place."""

        return MDHistoData(
            axes=self.axes,
            signal=self.signal,
            errors=self.errors,
            mask=self.mask,
            num_events=self.num_events,
            coordinate_system=self.coordinate_system,
            visual_normalization=self.visual_normalization,
            metadata=copy.deepcopy(self.metadata),
            auxiliary_channels=self.auxiliary_channels,
            _mutable=True,
        )

    def immutable_copy(self) -> MDHistoData:
        """Return an immutable copy, or ``self`` when already immutable."""

        arrays = (self.signal, self.errors, self.mask, self.num_events)
        channels_are_immutable = all(
            channel.immutable_copy() is channel
            for channel in self.auxiliary_channels.values()
        )
        if (
            not self._arrays_mutable
            and not any(array.flags.writeable for array in arrays)
            and channels_are_immutable
            and all(axis.immutable_copy() is axis for axis in self.axes)
        ):
            return self
        return self.with_updates()

    def with_updates(self, **changes: Any) -> MDHistoData:
        """Return an immutable container with selected fields replaced."""

        values = {
            "axes": self.axes,
            "signal": self.signal,
            "errors": self.errors,
            "mask": self.mask,
            "num_events": self.num_events,
            "coordinate_system": self.coordinate_system,
            "visual_normalization": self.visual_normalization,
            "metadata": self.metadata,
            "auxiliary_channels": self.auxiliary_channels,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unknown MDHistoData field(s): {', '.join(sorted(unknown))}")
        values.update(changes)
        return MDHistoData(**values)


def _as_array(value: Any, *, dtype: Any, mutable: bool) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if mutable:
        return np.array(arr, dtype=dtype, copy=True)
    if arr.flags.writeable:
        arr = np.array(arr, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def mdhisto_measured_bins(data: MDHistoData) -> BoolArray:
    """Return bins measured by the instrument, including covered zero counts."""

    coverage = data.metadata.get("normalization_denominator")
    return mdhisto_measured_bins_from_arrays(
        data.mask,
        data.num_events,
        zero_event_bins_are_measured=bool(
            data.metadata.get("zero_event_bins_are_measured", False)
        ),
        normalization_denominator=(
            coverage
            if isinstance(coverage, np.ndarray) and coverage.shape == data.shape
            else None
        ),
    )


def mdhisto_measured_bins_from_arrays(
    mask: BoolArray,
    num_events: FloatArray,
    *,
    zero_event_bins_are_measured: bool,
    normalization_denominator: FloatArray | None = None,
) -> BoolArray:
    """Return measured bins for complete arrays or matching bounded chunks."""

    unmasked = ~np.asarray(mask, dtype=bool)
    if zero_event_bins_are_measured:
        if normalization_denominator is not None:
            coverage = np.asarray(normalization_denominator, dtype=float)
            return np.isfinite(coverage) & (coverage > 0.0) & unmasked
        return unmasked
    return (np.asarray(num_events, dtype=float) > 0.0) & unmasked


def mdhisto_coverage_fraction(
    data: MDHistoData,
    selection: tuple[Any, ...] | None = None,
) -> FloatArray:
    """Return each bin's measured fraction of its requested geometric support.

    New reductions store coverage as an auxiliary channel so it remains
    independently viewable and serializable. Older data fall back to binary
    measured-bin coverage; that fallback can distinguish covered from empty
    bins but cannot reconstruct partial support inside a native bin. When a
    basic NumPy ``selection`` is supplied, only that region is materialized.
    """

    index = (...,) if selection is None else selection
    channel = data.auxiliary_channels.get("coverage_fraction")
    if channel is not None and channel.values.shape == data.shape:
        values = np.asarray(channel.values[index], dtype=float)
        return np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, 1.0)
    stored = data.metadata.get("coverage_fraction")
    if isinstance(stored, np.ndarray) and stored.shape == data.shape:
        values = np.asarray(stored[index], dtype=float)
        return np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, 1.0)
    if bool(data.metadata.get("zero_event_bins_are_measured", False)):
        denominator = data.metadata.get("normalization_denominator")
        if isinstance(denominator, np.ndarray) and denominator.shape == data.shape:
            selected = denominator[index]
            return np.asarray(
                np.isfinite(selected) & (selected > 0.0),
                dtype=float,
            )
        return np.ones(np.shape(data.signal[index]), dtype=float)
    return np.asarray(
        np.asarray(data.num_events[index], dtype=float) > 0.0,
        dtype=float,
    )


def load_mantid_mdhisto_nxs(
    path: str | Path,
    *,
    workspace_path: str = "/MDHistoWorkspace",
    copy_metadata: bool = False,
) -> MDHistoData:
    """Load the important arrays and axes from a Mantid ``SaveMD`` NeXus file.

    The importer reads ``/MDHistoWorkspace/data`` by default, including axis
    names, axis values, units, signal, one-sigma errors, Mantid mask flags, and
    event counts. ``errors`` are returned as ``sqrt(errors_squared)`` because
    Mantid stores variances in the file. Non-finite values, invalid variances,
    and invalid event counts are added to the saved mask. Bulky non-data
    metadata groups such as ``experiment0`` are not copied unless
    ``copy_metadata=True``.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ImportError("load_mantid_mdhisto_nxs requires h5py") from exc

    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        workspace = handle[workspace_path]
        data = workspace["data"]
        signal_dataset = data["signal"]
        axis_names = _signal_axis_names(signal_dataset)
        axes = tuple(_read_axis(data, axis_name) for axis_name in axis_names)

        signal = np.asarray(signal_dataset[()], dtype=float)
        errors_squared = np.asarray(data["errors_squared"][()], dtype=float)
        mask = np.asarray(data["mask"][()], dtype=bool)
        num_events = np.asarray(data["num_events"][()], dtype=float)
        invalid_variance = ~np.isfinite(errors_squared) | (errors_squared < 0.0)
        invalid_bins = (
            ~np.isfinite(signal)
            | invalid_variance
            | ~np.isfinite(num_events)
            | (num_events < 0.0)
        )
        mask = mask | invalid_bins
        errors_squared[invalid_variance] = np.nan
        errors = np.sqrt(errors_squared, out=errors_squared)
        metadata: dict[str, Any] = {
            "source_file": str(file_path),
            "workspace_path": workspace_path,
            "data_path": f"{workspace_path.rstrip('/')}/data",
            "signal_axes": axis_names,
            "workspace_attrs": _decode_attrs(workspace.attrs),
            "data_attrs": _decode_attrs(data.attrs),
            "signal_attrs": _decode_attrs(signal_dataset.attrs),
            # Reduced MD histograms are normally normalized/averaged values.
            # Treat them as densities by default; users can still declare a
            # genuinely pre-integrated source explicitly in the dataset GUI.
            "signal_semantics": "density",
            "signal_semantics_source": "mantid_mdhisto_workspace",
        }

        coordinate_system = _read_optional_scalar(workspace, "coordinate_system")
        visual_normalization = _read_optional_scalar(workspace, "visual_normalization")
        if copy_metadata:
            metadata["nexus"] = _copy_metadata_tree(workspace)
        oriented_lattice = _read_oriented_lattice_metadata(workspace)
        if oriented_lattice:
            metadata["oriented_lattice"] = oriented_lattice
            if "orientation_matrix" in oriented_lattice:
                metadata["ub_matrix"] = copy.deepcopy(
                    oriented_lattice["orientation_matrix"]
                )
            lattice_parameters = {
                key: float(oriented_lattice[key])
                for key in ("a", "b", "c", "alpha", "beta", "gamma")
                if key in oriented_lattice
            }
            if len(lattice_parameters) == 6:
                metadata["lattice_parameters"] = lattice_parameters
        metadata.update(_read_mantid_reduction_metadata(workspace, file_path, axes))
        invalid_count = int(np.count_nonzero(invalid_bins))
        if invalid_count:
            metadata["invalid_bin_count"] = invalid_count
            metadata["invalid_bins_masked"] = True

        return MDHistoData(
            axes=axes,
            signal=signal,
            errors=errors,
            mask=mask,
            num_events=num_events,
            coordinate_system=coordinate_system,
            visual_normalization=visual_normalization,
            metadata=metadata,
        )


def _signal_axis_names(signal_dataset: Any) -> tuple[str, ...]:
    axes_attr = signal_dataset.attrs.get("axes")
    if axes_attr is None:
        return tuple(f"D{i}" for i in range(signal_dataset.ndim))
    axes_text = _decode_value(axes_attr)
    return tuple(part for part in str(axes_text).split(":") if part)


def _read_axis(data_group: Any, axis_name: str) -> MDHistoAxis:
    dataset = data_group[axis_name]
    attrs = _decode_attrs(dataset.attrs)
    long_name = str(attrs.get("long_name", axis_name))
    units = str(attrs.get("units", ""))
    frame = attrs.get("frame")
    return MDHistoAxis(
        name=long_name,
        values=np.asarray(dataset[()], dtype=float),
        units=units,
        kind=_axis_kind(long_name, units, frame),
        frame=None if frame is None else str(frame),
        path=dataset.name,
        metadata=attrs,
    )


def _axis_kind(name: str, units: str, frame: Any) -> AxisKind:
    text = f"{name} {units} {frame}".lower()
    if "deltae" in text or "energy" in text or "mev" in text:
        return "energy"
    if "hkl" in text or "r.l.u" in text or "rlu" in text:
        return "momentum"
    return "unknown"


def _read_optional_scalar(group: Any, name: str) -> int | None:
    if name not in group:
        return None
    value = np.asarray(group[name][()]).reshape(-1)
    if value.size == 0:
        return None
    return int(value[0])


def _read_oriented_lattice_metadata(workspace: Any) -> dict[str, Any]:
    """Read common Mantid oriented-lattice fields when present."""

    aliases = {
        "orientation_matrix": "orientation_matrix",
        "ub_matrix": "ub_matrix",
        "a": "a",
        "b": "b",
        "c": "c",
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "unit_cell_a": "a",
        "unit_cell_b": "b",
        "unit_cell_c": "c",
        "unit_cell_alpha": "alpha",
        "unit_cell_beta": "beta",
        "unit_cell_gamma": "gamma",
    }
    found: dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        parts = name.lower().split("/")
        if "oriented_lattice" not in parts:
            return
        source_key = parts[-1]
        key = aliases.get(source_key)
        if key is None:
            return
        if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
            return
        raw = np.asarray(obj[()])
        value = _decode_value(raw.item()) if raw.size == 1 else _decode_value(raw)
        found[key] = value
        found[f"{key}_path"] = obj.name

    workspace.visititems(visit)
    return found


def _read_mantid_reduction_metadata(
    workspace: Any,
    file_path: Path,
    axes: tuple[MDHistoAxis, ...],
) -> dict[str, Any]:
    """Return compact instrument and reduction provenance from a saved workspace."""

    metadata: dict[str, Any] = {}
    instrument, source = _mantid_instrument_name(workspace, file_path)
    if instrument:
        metadata["instrument_name"] = instrument
        metadata["instrument_name_source"] = source

    q_convention = _decode_value(workspace.attrs.get("QConvention", ""))
    if str(q_convention).strip():
        metadata["q_convention"] = str(q_convention).strip()

    has_energy = any(axis.kind == "energy" for axis in axes)
    momentum_count = sum(axis.kind == "momentum" for axis in axes)
    if not has_energy and momentum_count >= 2:
        metadata["suggested_data_type"] = "single_crystal_elastic"

    wavelength = _mantid_log_scalar(workspace, "wavelength")
    if wavelength is not None and wavelength > 0.0:
        metadata["incident_wavelength"] = wavelength
        metadata["incident_wavelength_unit"] = "angstrom"

    reduction: dict[str, Any] = {"format": "Mantid MDHistoWorkspace"}
    stem = file_path.stem.casefold()
    if instrument == "CORELLI" and re.search(r"(?:^|_)cc(?:_|$)", stem):
        reduction.update(
            {
                "scattering_mode": "elastic",
                "elastic_discrimination": "correlation_chopper",
                "correlation_chopper": True,
                "correlation_chopper_source": "source_filename",
            }
        )
    elif instrument == "WAND²" and not has_energy:
        reduction["scattering_mode"] = "monochromatic_elastic"
    metadata["reduction_provenance"] = reduction
    return metadata


def _mantid_instrument_name(workspace: Any, file_path: Path) -> tuple[str, str]:
    for name in sorted(workspace):
        if not name.casefold().startswith("experiment"):
            continue
        experiment = workspace[name]
        try:
            value = experiment["instrument/name"][()]
        except (KeyError, TypeError):
            continue
        text = _first_text(value).strip()
        if text:
            return _canonical_mantid_instrument(text), "nexus"

    path_text = str(file_path).upper()
    if "CORELLI" in path_text:
        return "CORELLI", "source_path"
    if "WAND" in path_text or "HB2C" in path_text:
        return "WAND²", "source_path"
    return "", ""


def _canonical_mantid_instrument(name: str) -> str:
    compact = str(name).strip().upper()
    if compact in {"WAND", "WAND2", "WAND²", "HB2C"}:
        return "WAND²"
    return compact


def _first_text(value: Any) -> str:
    decoded = _decode_value(value)
    while isinstance(decoded, list):
        if not decoded:
            return ""
        decoded = decoded[0]
    return str(decoded).replace("\x00", "")


def _mantid_log_scalar(workspace: Any, log_name: str) -> float | None:
    for name in sorted(workspace):
        if not name.casefold().startswith("experiment"):
            continue
        try:
            values = np.asarray(
                workspace[name][f"logs/{log_name}/value"][()], dtype=float
            )
        except (KeyError, TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(finite.reshape(-1)[0])
    return None


def _copy_metadata_tree(group: Any, *, max_dataset_items: int = 16) -> dict[str, Any]:
    tree: dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        entry: dict[str, Any] = {"attrs": _decode_attrs(obj.attrs)}
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            entry["shape"] = tuple(int(size) for size in obj.shape)
            entry["dtype"] = str(obj.dtype)
            if int(np.prod(obj.shape, dtype=np.int64)) <= max_dataset_items:
                entry["value"] = _decode_value(obj[()])
        else:
            entry["type"] = "group"
        tree[name] = entry

    group.visititems(visit)
    return tree


def _decode_attrs(attrs: Any) -> dict[str, Any]:
    return {key: _decode_value(value) for key, value in attrs.items()}


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_value(value.item())
        if value.dtype.kind == "S":
            return [_decode_value(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_decode_value(item) for item in value]
    return value
