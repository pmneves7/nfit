from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


AxisKind = Literal["momentum", "energy", "unknown"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MDHistoAxis:
    """One binned Mantid MDHistoWorkspace axis."""

    name: str
    values: FloatArray
    units: str
    kind: AxisKind
    frame: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def centers(self) -> FloatArray:
        """Return bin centers when ``values`` are bin boundaries."""

        if self.values.size < 2:
            return self.values.copy()
        return 0.5 * (self.values[:-1] + self.values[1:])


@dataclass
class MDHistoData:
    """Imported Mantid MDHistoWorkspace data with full binned axes.

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

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the imported binned arrays."""

        return self.signal.shape


def load_mantid_mdhisto_nxs(
    path: str | Path,
    *,
    workspace_path: str = "/MDHistoWorkspace",
    copy_metadata: bool = True,
) -> MDHistoData:
    """Load the important arrays and axes from a Mantid ``SaveMD`` NeXus file.

    The importer reads ``/MDHistoWorkspace/data`` by default, including axis
    names, axis values, units, signal, one-sigma errors, Mantid mask flags, and
    event counts. ``errors`` are returned as ``sqrt(errors_squared)`` because
    Mantid stores variances in the file.
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
        }

        coordinate_system = _read_optional_scalar(workspace, "coordinate_system")
        visual_normalization = _read_optional_scalar(workspace, "visual_normalization")
        if copy_metadata:
            metadata["nexus"] = _copy_metadata_tree(workspace)

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
