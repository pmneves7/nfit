from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .axes import AxisRole, infer_axis_role
from .dataset import PointData4D


AxisKind = Literal["momentum", "energy", "unknown"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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


def point_data_from_hyspec_hhl(
    data: MDHistoData,
    *,
    max_q_trajectories: int | None = None,
    random_seed: int = 12345,
    temperature: float | None = None,
) -> tuple[PointData4D, dict[str, Any]]:
    """Flatten a HYSPEC HHL ``MDHistoData`` volume into point data.

    The expected axes are ``DeltaE``, ``[H,-H,0]``, ``[0,0,L]``, and
    ``[H,H,0]``. A point at coordinates ``u=[H,H,0]`` and ``v=[H,-H,0]`` maps to
    physical reciprocal-lattice coordinates ``H = u + v`` and ``K = u - v``.

    ``max_q_trajectories`` can be used to choose a reproducible subset of
    complete Q trajectories for faster examples or exploratory fits.
    """

    e_dim = _axis_index(data, "DeltaE")
    hmh_dim = _axis_index(data, "[H,-H,0]")
    l_dim = _axis_index(data, "[0,0,L]")
    hh_dim = _axis_index(data, "[H,H,0]")
    if (e_dim, hmh_dim, l_dim, hh_dim) != (0, 1, 2, 3):
        raise ValueError("HYSPEC HHL axes must be in array order (DeltaE, [H,-H,0], [0,0,L], [H,H,0])")

    valid = np.isfinite(data.signal)
    valid &= np.isfinite(data.errors)
    valid &= data.errors > 0.0
    valid &= ~data.mask
    valid &= data.num_events > 0.0

    trajectory_has_data = np.any(valid, axis=e_dim)
    trajectory_indices = np.argwhere(trajectory_has_data)
    total_trajectories = int(trajectory_indices.shape[0])

    if max_q_trajectories is not None and total_trajectories > max_q_trajectories:
        rng = np.random.default_rng(random_seed)
        chosen = np.sort(rng.choice(total_trajectories, size=max_q_trajectories, replace=False))
        trajectory_indices = trajectory_indices[chosen]

    n_traj = int(trajectory_indices.shape[0])
    e_centers = data.axes[e_dim].centers
    hmh_centers = data.axes[hmh_dim].centers
    l_centers = data.axes[l_dim].centers
    hh_centers = data.axes[hh_dim].centers
    n_energy = int(e_centers.size)

    e_idx = np.tile(np.arange(n_energy), n_traj)
    hmh_idx = np.repeat(trajectory_indices[:, 0], n_energy)
    l_idx = np.repeat(trajectory_indices[:, 1], n_energy)
    hh_idx = np.repeat(trajectory_indices[:, 2], n_energy)

    hmh = hmh_centers[hmh_idx]
    hh = hh_centers[hh_idx]
    l = l_centers[l_idx]
    point_mask = valid[e_idx, hmh_idx, l_idx, hh_idx]

    metadata: dict[str, Any] = {
        "source_file": data.metadata.get("source_file"),
        "coordinate_units": "r.l.u.",
        "energy_units": "meV",
        "hyspec_hhl_axis_map": {
            "H": "[H,H,0] + [H,-H,0]",
            "K": "[H,H,0] - [H,-H,0]",
            "L": "[0,0,L]",
            "E": "DeltaE",
        },
    }
    if "oriented_lattice" in data.metadata:
        metadata["oriented_lattice"] = data.metadata["oriented_lattice"]

    point_data = PointData4D(
        H=hh + hmh,
        K=hh - hmh,
        L=l,
        E=e_centers[e_idx],
        intensity=data.signal[e_idx, hmh_idx, l_idx, hh_idx],
        sigma=data.errors[e_idx, hmh_idx, l_idx, hh_idx],
        mask=point_mask,
        temperature=temperature,
        metadata=metadata,
    )
    summary = {
        "total_q_trajectories_with_data": total_trajectories,
        "used_q_trajectories": n_traj,
        "candidate_points": point_data.size,
        "initial_valid_points": int(np.count_nonzero(point_mask)),
    }
    return point_data, summary


def hyspec_hhl_point_indices(
    data: MDHistoData,
    points: PointData4D,
    *,
    tolerance: float | None = None,
) -> tuple[NDArray[np.intp], ...]:
    """Return MDHisto bin indices for HYSPEC HHL point data.

    This is the inverse grid lookup for :func:`point_data_from_hyspec_hhl`.
    Points are mapped from physical reciprocal-lattice coordinates back to the
    HYSPEC axes using ``[H,H,0] = (H + K) / 2`` and
    ``[H,-H,0] = (H - K) / 2``.
    """

    e_dim = _axis_index(data, "DeltaE")
    hmh_dim = _axis_index(data, "[H,-H,0]")
    l_dim = _axis_index(data, "[0,0,L]")
    hh_dim = _axis_index(data, "[H,H,0]")

    indices: list[NDArray[np.intp] | None] = [None] * data.signal.ndim
    indices[e_dim] = _nearest_axis_center_indices(
        data.axes[e_dim],
        np.asarray(points.E, dtype=float),
        tolerance=tolerance,
    )
    indices[hmh_dim] = _nearest_axis_center_indices(
        data.axes[hmh_dim],
        0.5 * (np.asarray(points.H, dtype=float) - np.asarray(points.K, dtype=float)),
        tolerance=tolerance,
    )
    indices[l_dim] = _nearest_axis_center_indices(
        data.axes[l_dim],
        np.asarray(points.L, dtype=float),
        tolerance=tolerance,
    )
    indices[hh_dim] = _nearest_axis_center_indices(
        data.axes[hh_dim],
        0.5 * (np.asarray(points.H, dtype=float) + np.asarray(points.K, dtype=float)),
        tolerance=tolerance,
    )
    return tuple(index for index in indices if index is not None)


def _signal_axis_names(signal_dataset: Any) -> tuple[str, ...]:
    axes_attr = signal_dataset.attrs.get("axes")
    if axes_attr is None:
        return tuple(f"D{i}" for i in range(signal_dataset.ndim))
    axes_text = _decode_value(axes_attr)
    return tuple(part for part in str(axes_text).split(":") if part)


def _axis_index(data: MDHistoData, name: str) -> int:
    for index, axis in enumerate(data.axes):
        if axis.name == name:
            return index
    raise ValueError(f"axis {name!r} not found")


def _nearest_axis_center_indices(
    axis: MDHistoAxis,
    values: FloatArray,
    *,
    tolerance: float | None,
) -> NDArray[np.intp]:
    centers = np.asarray(axis.centers, dtype=float)
    if centers.size == 0:
        raise ValueError(f"axis {axis.name!r} has no centers")
    positions = np.searchsorted(centers, values)
    right = np.clip(positions, 0, centers.size - 1)
    left = np.clip(positions - 1, 0, centers.size - 1)
    use_right = np.abs(centers[right] - values) <= np.abs(centers[left] - values)
    indices = np.where(use_right, right, left).astype(np.intp)

    if tolerance is None:
        if centers.size > 1:
            scale = float(np.nanmax(np.abs(np.diff(centers))))
        else:
            scale = max(abs(float(centers[0])), 1.0)
        tolerance = max(scale * 1.0e-6, 1.0e-10)
    mismatch = ~np.isfinite(values) | (np.abs(centers[indices] - values) > tolerance)
    if np.any(mismatch):
        first = int(np.flatnonzero(mismatch)[0])
        raise ValueError(
            f"value {values[first]!r} does not map onto axis {axis.name!r} "
            f"within tolerance {tolerance:g}"
        )
    return indices


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
