from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .axes import AxisRole, infer_axis_role

AxisKind = Literal["momentum", "energy", "unknown"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class MDHistoChannel:
    values: FloatArray
    errors: FloatArray | None = None
    label: str = ""
    unit: str = ""
    quantity_type: str = "unknown"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float)
        from .quantities import QUANTITY_TYPES, normalize_unit

        self.unit = normalize_unit(self.unit)
        if self.quantity_type not in QUANTITY_TYPES:
            raise ValueError(f"unknown channel quantity type {self.quantity_type!r}")


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

        compact_name = self.name.casefold().replace("_", "").replace(" ", "")
        compact_unit = self.units.casefold().replace("_", "").replace(" ", "")
        if (
            self.kind == "energy" or compact_name in {"deltae", "energytransfer"}
        ) and compact_unit in {"deltae", "energy", "energytransfer"}:
            object.__setattr__(self, "units", "meV")

    @property
    def centers(self) -> FloatArray:
        """Return bin centers when ``values`` are bin boundaries."""

        if self.values.size < 2:
            return self.values.copy()
        return 0.5 * (self.values[:-1] + self.values[1:])

    @property
    def role(self) -> AxisRole:
        """Reduced-data role inferred from axis name, units, and kind."""

        return infer_axis_role(self.name, units=self.units, kind=self.kind)


@dataclass
class MDHistoData:
    """Imported binned reduced data from a Mantid MDHistoWorkspace.

    ``axes`` are ordered to match the array dimensions of ``signal``, ``errors``,
    ``mask``, and ``num_events``. Mantid stores the mask as bin flags; this class
    preserves those flags as booleans rather than inverting their meaning.
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

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        self.errors = np.asarray(self.errors, dtype=float)
        self.mask = np.asarray(self.mask, dtype=bool)
        self.num_events = np.asarray(self.num_events, dtype=float)

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


def mdhisto_measured_bins(data: MDHistoData) -> BoolArray:
    """Return bins measured by the instrument, including covered zero counts."""

    if bool(data.metadata.get("zero_event_bins_are_measured", False)):
        coverage = data.metadata.get("normalization_denominator")
        if isinstance(coverage, np.ndarray) and coverage.shape == data.shape:
            return np.isfinite(coverage) & (coverage > 0.0) & ~np.asarray(data.mask, dtype=bool)
        return ~np.asarray(data.mask, dtype=bool)
    return (np.asarray(data.num_events, dtype=float) > 0.0) & ~np.asarray(data.mask, dtype=bool)


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
    Mantid stores variances in the file. Bulky non-data metadata groups such as
    ``experiment0`` are not copied unless ``copy_metadata=True``.
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

        errors_squared = np.asarray(data["errors_squared"][()], dtype=float)
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

        return MDHistoData(
            axes=axes,
            signal=np.asarray(signal_dataset[()], dtype=float),
            errors=np.sqrt(errors_squared),
            mask=np.asarray(data["mask"][()], dtype=bool),
            num_events=np.asarray(data["num_events"][()], dtype=float),
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

    wanted = {
        "orientation_matrix",
        "ub_matrix",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
    }
    found: dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        parts = name.lower().split("/")
        if "oriented_lattice" not in parts:
            return
        key = parts[-1]
        if key not in wanted:
            return
        if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
            return
        value = _decode_value(obj[()])
        found[key] = value
        found[f"{key}_path"] = obj.name

    workspace.visititems(visit)
    return found


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
