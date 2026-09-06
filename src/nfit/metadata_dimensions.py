"""Discrete coordinates derived from sample metadata, without interpolation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PointData4D
from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData


@dataclass(frozen=True)
class MetadataDimension:
    """One discrete coordinate. ``tolerance`` has the same units as ``centers``.

    ``source`` is a ``metadata/...`` or ``parameters/...`` path, ``temperature``
    (the loaded point temperatures), or an HDF5 path such as
    ``entry/data/temp/average_value``. ``>`` separators are also accepted.
    ``sampling`` is ``dataset_mean``, ``dataset_median``, or ``per_point``.
    Explicit centers use nearest assignment within tolerance; ties and values
    outside tolerance raise. Without centers, exact unique values are retained.
    """

    name: str
    source: str
    units: str = ""
    sampling: str = "dataset_mean"
    centers: Sequence[float] | None = None
    tolerance: float = 0.5

    def __post_init__(self):
        if not self.name.strip() or not self.source.strip():
            raise ValueError("dimension name and metadata channel are required")
        if self.sampling not in {"dataset_mean", "dataset_median", "per_point"}:
            raise ValueError("sampling must be dataset_mean, dataset_median, or per_point")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("assignment tolerance must be finite and nonnegative")
        if self.centers is not None:
            values = np.asarray(self.centers, dtype=float)
            if (
                values.ndim != 1
                or not values.size
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError("discrete coordinates must be finite and strictly increasing")
            object.__setattr__(self, "centers", tuple(float(v) for v in values))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.centers is not None:
            result["centers"] = list(self.centers)
        return result


def _path(source: str) -> str:
    return "/".join(part.strip() for part in source.replace(">", "/").split("/") if part.strip())


def _lookup(mapping, path):
    if path in mapping:
        return mapping[path]
    first, _, rest = path.partition("/")
    value = mapping[first]
    return _lookup(value, rest) if rest else value


def metadata_channel_values(dataset, source: str) -> tuple[np.ndarray, str]:
    """Read one numeric channel from an entry or its source file, without edits."""
    path = _path(source)
    data = dataset.data
    unit = ""
    if path == "temperature":
        value = getattr(data, "temperature", None)
        unit = "K"
    elif path.startswith("parameters/"):
        value = _lookup(dataset.parameters, path.removeprefix("parameters/"))
        unit = "K" if path == "parameters/temperature" else ""
    elif path.startswith("metadata/"):
        metadata = {**getattr(data, "metadata", {}), **dataset.metadata}
        value = _lookup(metadata, path.removeprefix("metadata/"))
    else:
        import h5py

        filename = dataset.metadata.get("source_file") or getattr(data, "metadata", {}).get(
            "source_file"
        )
        if not filename or not h5py.is_hdf5(filename):
            raise ValueError(f"{dataset.name}: this channel requires an accessible HDF5 source")
        with h5py.File(filename, "r") as handle:
            channel = handle[path]
            if not isinstance(channel, h5py.Dataset) or channel.dtype.kind not in "iuf":
                raise ValueError(f"{path} is not a numeric metadata channel")
            value = channel[()]
            unit = channel.attrs.get("units", "")
            if isinstance(unit, bytes):
                unit = unit.decode()
    if value is None:
        raise ValueError(f"{dataset.name}: no values in {source}")
    result = np.asarray(value, dtype=float)
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{dataset.name}: {source} must contain finite numeric values")
    return result, str(unit)


def metadata_dimension_coordinates(dataset, dimension: MetadataDimension) -> np.ndarray:
    """Resolve a scalar or an explicitly aligned coordinate per measured point.

    MACS scan channels under ``entry/data`` are repeated over detector channels
    using the importer's recorded scan shape. Timestamped logs are never aligned
    by guessing their sampling rate or interpolating time.
    """
    values, units = metadata_channel_values(dataset, dimension.source)
    if units and dimension.units and units != dimension.units:
        raise ValueError(
            f"{dataset.name}: channel units {units!r} differ from {dimension.units!r}; convert the source first"
        )
    if dimension.sampling == "dataset_mean":
        return np.asarray([float(np.mean(values))])
    if dimension.sampling == "dataset_median":
        return np.asarray([float(np.median(values))])
    data = dataset.data
    size = data.size if isinstance(data, PointData4D) else data.signal.size
    path = _path(dimension.source)
    if path.startswith("entry/") and not path.startswith("entry/data/"):
        raise ValueError(
            "pointwise HDF5 channels must be aligned entry/data columns; time logs need an explicit alignment adapter"
        )
    if path.startswith("entry/data/") and path.endswith("/value"):
        import h5py

        filename = dataset.metadata.get("source_file") or data.metadata.get("source_file")
        with h5py.File(filename, "r") as handle:
            if "time" in handle[path].parent:
                raise ValueError(
                    "timestamped value logs need an explicit alignment adapter; choose an aligned scan column instead"
                )
    if values.size == 1:
        return values.reshape(-1)
    if values.shape == (size,) or (isinstance(data, MDHistoData) and values.shape == data.shape):
        return values.reshape(-1)
    scan_shape = data.metadata.get("scan_point_shape")
    if (
        path.startswith("entry/data/")
        and data.metadata.get("importer") == "macs_nexus"
        and scan_shape
        and tuple(values.shape) == (scan_shape[0],)
        and int(np.prod(scan_shape)) == size
    ):
        return np.repeat(values, int(np.prod(scan_shape[1:])))
    raise ValueError(
        f"{dataset.name}: {values.shape} channel values are not aligned with {size} points"
    )


def metadata_dimension_indices(values, centers, tolerance: float) -> np.ndarray:
    """Assign each value to exactly one center, rejecting ties and outliers."""
    values = np.asarray(values, dtype=float).reshape(-1)
    centers = np.asarray(centers, dtype=float)
    right = np.searchsorted(centers, values).clip(0, len(centers) - 1)
    left = (right - 1).clip(0)
    dl, dr = np.abs(values - centers[left]), np.abs(values - centers[right])
    if np.any((left != right) & np.isclose(dl, dr, atol=1e-12, rtol=0)):
        raise ValueError("metadata value is equidistant from two discrete coordinates")
    indices = np.where(dl < dr, left, right)
    if np.any(np.abs(values - centers[indices]) > tolerance + 1e-12):
        bad = values[np.abs(values - centers[indices]) > tolerance + 1e-12][0]
        raise ValueError(f"metadata value {bad:g} is outside assignment tolerance {tolerance:g}")
    return indices


def metadata_channels(dataset) -> dict[str, str]:
    """List numeric metadata paths and units without reading HDF5 payloads."""
    channels = {}
    if getattr(dataset.data, "temperature", None) is not None:
        channels["temperature"] = "K"

    def mapping_channels(value, path):
        if isinstance(value, Mapping):
            for key, child in value.items():
                mapping_channels(child, f"{path}/{key}")
        else:
            try:
                array = np.asarray(value)
            except (ValueError, TypeError):
                return
            if array.dtype.kind in "iuf" and array.size:
                channels[path] = "K" if path == "parameters/temperature" else ""

    mapping_channels(dataset.parameters, "parameters")
    mapping_channels({**getattr(dataset.data, "metadata", {}), **dataset.metadata}, "metadata")
    filename = dataset.metadata.get("source_file")
    if filename and Path(filename).is_file():
        import h5py

        if h5py.is_hdf5(filename):
            with h5py.File(filename, "r") as handle:
                # Preserve NeXus aliases under entry/data, which visititems skips.
                def visit(group, prefix, ancestors):
                    identity = hash(group.id)
                    if identity in ancestors:
                        return
                    for key, item in group.items():
                        path = f"{prefix}/{key}".lstrip("/")
                        if isinstance(item, h5py.Group):
                            visit(item, path, ancestors | {identity})
                        elif (
                            isinstance(item, h5py.Dataset)
                            and item.dtype.kind in "iuf"
                            and item.ndim <= 1
                        ):
                            unit = item.attrs.get("units", "")
                            channels[path] = unit.decode() if isinstance(unit, bytes) else str(unit)

                visit(handle, "", set())
    return dict(sorted(channels.items()))


def discrete_metadata_axis(dimension: MetadataDimension, centers) -> MDHistoAxis:
    """Create display boundaries while retaining the exact physical coordinates."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        edges = np.array([centers[0] - 0.5, centers[0] + 0.5])
    else:
        edges = np.r_[
            centers[0] - (centers[1] - centers[0]) / 2,
            (centers[:-1] + centers[1:]) / 2,
            centers[-1] + (centers[-1] - centers[-2]) / 2,
        ]
    return MDHistoAxis(
        dimension.name,
        edges,
        dimension.units,
        "unknown",
        metadata={
            "discrete_centers": centers.tolist(),
            "metadata_dimension": dimension.to_dict(),
            "interpolation": "none",
        },
    )


def metadata_temperature_grid(data: MDHistoData) -> np.ndarray | None:
    """Return broadcastable temperatures in kelvin from a discrete K axis."""
    dimensions = [
        i
        for i, axis in enumerate(data.axes)
        if "metadata_dimension" in axis.metadata and axis.units == "K"
    ]
    if len(dimensions) > 1:
        raise ValueError(
            "more than one temperature axis in kelvin; choose one temperature dimension"
        )
    if not dimensions:
        return None
    dimension = dimensions[0]
    shape = [1] * data.signal.ndim
    shape[dimension] = data.shape[dimension]
    return data.axes[dimension].centers.reshape(shape)


def stack_metadata_histograms(template, slices, dimensions, centers) -> MDHistoData:
    """Stack independent histograms; missing combinations remain masked."""
    shape = template.shape + tuple(len(c) for c in centers)
    signal = np.full(shape, np.nan)
    errors = np.full(shape, np.nan)
    mask = np.ones(shape, dtype=bool)
    events = np.zeros(shape)
    channels = {
        key: (np.zeros(shape), None if channel.errors is None else np.full(shape, np.nan))
        for key, channel in template.auxiliary_channels.items()
    }
    for indices, data in slices.items():
        target = (slice(None),) * len(template.shape) + indices
        signal[target], errors[target], mask[target], events[target] = (
            data.signal,
            data.errors,
            data.mask,
            data.num_events,
        )
        for key, (values, uncertainty) in channels.items():
            values[target] = data.auxiliary_channels[key].values
            if uncertainty is not None:
                uncertainty[target] = data.auxiliary_channels[key].errors
    metadata = copy.deepcopy(template.metadata)
    metadata.pop("temperature", None)
    metadata["metadata_dimensions"] = [d.to_dict() for d in dimensions]
    return template.with_updates(
        axes=template.axes
        + tuple(discrete_metadata_axis(d, c) for d, c in zip(dimensions, centers, strict=True)),
        signal=signal,
        errors=errors,
        mask=mask,
        num_events=events,
        metadata=metadata,
        auxiliary_channels={
            key: MDHistoChannel(
                values,
                uncertainty,
                template.auxiliary_channels[key].label,
                template.auxiliary_channels[key].unit,
                template.auxiliary_channels[key].quantity_type,
            )
            for key, (values, uncertainty) in channels.items()
        },
    )
