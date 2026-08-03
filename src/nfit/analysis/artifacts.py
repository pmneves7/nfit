from __future__ import annotations

import json
import tempfile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ..dataset import PointListData
from ..mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from ..project_archive import read_project_artifact
from .core import DatasetOutput, TableOutput


def write_dataset_artifact(data: MDHistoData | PointListData, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(data)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **payload)
    try:
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def dataset_artifact_bytes(data: MDHistoData | PointListData) -> bytes:
    """Serialize an analysis dataset for storage inside a project archive."""

    stream = BytesIO()
    np.savez_compressed(stream, **_payload(data))
    return stream.getvalue()


def read_project_dataset_artifact(
    project_path: str | Path,
    artifact_path: str,
) -> MDHistoData | PointListData:
    """Read an analysis dataset stored inside an nfit project."""

    return read_dataset_artifact(read_project_artifact(project_path, artifact_path))


def read_dataset_artifact(
    source: str | PathLike[str] | bytes | BinaryIO,
) -> MDHistoData | PointListData:
    stream: str | PathLike[str] | BinaryIO
    stream = BytesIO(source) if isinstance(source, bytes) else source
    with np.load(stream, allow_pickle=False) as archive:
        kind = str(np.asarray(archive["container"]).item())
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        if kind == "point_list":
            names = json.loads(str(np.asarray(archive["column_names_json"]).item()))
            return PointListData(
                {name: np.asarray(archive[f"column_{i}"], dtype=float) for i, name in enumerate(names)},
                units=json.loads(str(np.asarray(archive["units_json"]).item())),
                coordinate_names=json.loads(str(np.asarray(archive["coordinate_names_json"]).item())),
                channels=json.loads(str(np.asarray(archive["channels_json"]).item())), metadata=metadata,
                quantity_types=(json.loads(str(np.asarray(archive["quantity_types_json"]).item())) if "quantity_types_json" in archive else {}),
            )
        if kind != "mdhisto":
            raise ValueError(f"unknown analysis artifact container {kind!r}")
        count = int(np.asarray(archive["axis_count"]).item())
        axes = tuple(
            MDHistoAxis(
                str(np.asarray(archive[f"axis_{i}_name"]).item()), np.asarray(archive[f"axis_{i}_values"], dtype=float),
                str(np.asarray(archive[f"axis_{i}_units"]).item()), str(np.asarray(archive[f"axis_{i}_kind"]).item()),
                frame=str(np.asarray(archive[f"axis_{i}_frame"]).item()) or None,
                path=str(np.asarray(archive[f"axis_{i}_path"]).item()) or None,
                metadata=json.loads(str(np.asarray(archive[f"axis_{i}_metadata"]).item())),
            ) for i in range(count)
        )
        channel_names = json.loads(str(np.asarray(archive["auxiliary_names"]).item())) if "auxiliary_names" in archive else []
        channels = {
            name: MDHistoChannel(
                np.asarray(archive[f"aux_{i}_values"]),
                np.asarray(archive[f"aux_{i}_errors"])
                if f"aux_{i}_errors" in archive
                else None,
                str(np.asarray(archive[f"aux_{i}_label"]).item()),
                str(np.asarray(archive[f"aux_{i}_unit"]).item()),
                str(np.asarray(archive[f"aux_{i}_quantity_type"]).item())
                if f"aux_{i}_quantity_type" in archive
                else "unknown",
            )
            for i, name in enumerate(channel_names)
        }
        coordinate = int(np.asarray(archive["coordinate_system"]).item()) if "coordinate_system" in archive else -1
        visual = int(np.asarray(archive["visual_normalization"]).item()) if "visual_normalization" in archive else -1
        return MDHistoData(axes, archive["signal"], archive["errors"], archive["mask"], archive["num_events"], coordinate_system=None if coordinate < 0 else coordinate, visual_normalization=None if visual < 0 else visual, metadata=metadata, auxiliary_channels=channels)


def output_data(output: DatasetOutput | TableOutput) -> MDHistoData | PointListData:
    return output.data


def _payload(data: MDHistoData | PointListData) -> dict[str, Any]:
    common: dict[str, Any] = {"format": np.asarray("nfit-analysis-artifact"), "version": np.asarray(1), "metadata_json": np.asarray(json.dumps(_json_metadata(data.metadata), sort_keys=True))}
    if isinstance(data, PointListData):
        common.update({"container": np.asarray("point_list"), "column_names_json": np.asarray(json.dumps(data.column_names)), "units_json": np.asarray(json.dumps(data.units)), "quantity_types_json": np.asarray(json.dumps(data.quantity_types)), "coordinate_names_json": np.asarray(json.dumps(data.coordinate_names)), "channels_json": np.asarray(json.dumps(data.channels))})
        common.update({f"column_{i}": data.column(name) for i, name in enumerate(data.column_names)})
        return common
    common.update({"container": np.asarray("mdhisto"), "signal": data.signal, "errors": data.errors, "mask": data.mask, "num_events": data.num_events, "axis_count": np.asarray(len(data.axes)), "coordinate_system": np.asarray(-1 if data.coordinate_system is None else data.coordinate_system), "visual_normalization": np.asarray(-1 if data.visual_normalization is None else data.visual_normalization)})
    for i, axis in enumerate(data.axes):
        common.update({f"axis_{i}_name": np.asarray(axis.name), f"axis_{i}_values": axis.values, f"axis_{i}_units": np.asarray(axis.units), f"axis_{i}_kind": np.asarray(axis.kind), f"axis_{i}_frame": np.asarray(axis.frame or ""), f"axis_{i}_path": np.asarray(axis.path or ""), f"axis_{i}_metadata": np.asarray(json.dumps(_json_metadata(axis.metadata), sort_keys=True))})
    common["auxiliary_names"] = np.asarray(json.dumps(list(data.auxiliary_channels)))
    for i, channel in enumerate(data.auxiliary_channels.values()):
        common.update({f"aux_{i}_values": channel.values, f"aux_{i}_label": np.asarray(channel.label), f"aux_{i}_unit": np.asarray(channel.unit), f"aux_{i}_quantity_type": np.asarray(channel.quantity_type)})
        if channel.errors is not None:
            common[f"aux_{i}_errors"] = channel.errors
    return common


def _json_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 256 else {"omitted_array_shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
