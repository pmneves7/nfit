from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ..array_archive import write_array_archive
from ..background_channel_io import background_metadata_payload, restore_background_metadata
from ..data_workspace import record_temporary_file
from ..dataset import PointData4D, PointListData
from ..mapped_archive import read_mapped_array_archive
from ..mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from ..operation_control import report_operation
from ..performance import operation_worker_count
from ..point_data_archive import point_data_archive_payload, restore_point_data_archive
from ..project_archive import _project_artifact_reader_factory, open_project_artifact
from ..resource_budget import reserve_memory
from ..storage_budget import reserve_disk_space
from .core import DatasetOutput, TableOutput

# Mapping is an explicit scripting option; project viewers load whole cubes.
_MAPPED_MEMBER_MIN_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class ArtifactCapacity:
    """Expanded array storage and conservative decode peak, from NPZ headers."""

    expanded_bytes: int
    peak_bytes: int
    member_bytes: tuple[tuple[str, int], ...]


def _array_header(stream: BinaryIO):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    if version == (3, 0):
        # NumPy exposes public readers for versions 1/2 only; version 3 uses
        # UTF-8 field names and must not be decoded with the Latin-1 reader.
        return np.lib.format._read_array_header(stream, version)
    raise ValueError(f"unsupported NumPy member version {version!r}")


def _decoded_array_dtype(name: str, dtype: np.dtype) -> np.dtype:
    """Account for dtype normalization performed by the dataset codecs."""

    if name == "mask":
        return np.dtype(bool)
    if name in {
        "signal", "errors", "num_events", "H", "K", "L", "E",
        "intensity", "sigma", "temperature", "magnetic_field",
        "normalization_denominator",
    }:
        return np.dtype(float)
    for prefix in ("column_", "measurement_payload_"):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return np.dtype(float)
    for prefix in ("axis_", "aux_", "auxiliary_"):
        if name.startswith(prefix):
            index, _, field = name[len(prefix):].partition("_")
            if index.isdigit() and field in {"values", "errors"}:
                return np.dtype(float)
    for prefix in (
        "source_dependencies_", "counting_numerator_dependencies_",
        "counting_exposure_dependencies_",
    ):
        if name.startswith(prefix):
            field = name[len(prefix):]
            if field in {"source_variances", "coefficients"}:
                return np.dtype(float)
            if field in {"observation_indices", "source_indices"}:
                return np.dtype(np.int64)
    # Optional numerical payloads can also require float64 normalization.
    return np.dtype(float) if dtype.kind == "f" else dtype


def _archive_capacity(archive: Any) -> ArtifactCapacity:
    sizes = []
    conversion = 0
    for info in archive.zip.infolist():
        if not info.filename.endswith(".npy"):
            continue
        with archive.zip.open(info) as stream:
            shape, _fortran, dtype = _array_header(stream)
            if dtype.hasobject:
                raise ValueError("object arrays are not supported")
            count = int(np.prod(shape, dtype=object))
            size = count * dtype.itemsize
            if size != info.file_size - stream.tell():
                raise ValueError(f"array member {info.filename!r} has an invalid declared size")
            decoded_dtype = _decoded_array_dtype(info.filename[:-4], dtype)
            if dtype != decoded_dtype:
                # A cast can create a writable array, then the immutable
                # container isolates it. Integer columns and byte masks need
                # this allowance just as non-float64 numerical arrays do.
                conversion += 2 * count * decoded_dtype.itemsize
            sizes.append((info.filename[:-4], size))
    expanded = sum(size for _name, size in sizes)
    return ArtifactCapacity(expanded, expanded + conversion + 16 * 1024**2, tuple(sizes))


def dataset_artifact_capacity(
    source: str | PathLike[str] | bytes | BinaryIO,
) -> ArtifactCapacity:
    """Inspect an artifact without decoding any numerical array payload."""

    stream = BytesIO(source) if isinstance(source, bytes) else source
    position = stream.tell() if hasattr(stream, "tell") else None
    try:
        with np.load(stream, allow_pickle=False) as archive:
            return _archive_capacity(archive)
    finally:
        if position is not None:
            stream.seek(position)


def project_dataset_artifact_capacity(project_path: str | Path, artifact_path: str) -> ArtifactCapacity:
    """Inspect a project member without loading its histogram."""

    with open_project_artifact(project_path, artifact_path) as stream:
        return dataset_artifact_capacity(stream)


def write_dataset_artifact(
    data: MDHistoData | PointData4D | PointListData, destination: str | Path, *, compressed: bool = True
) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(data)
    temporary = None
    try:
        expanded = sum(np.asarray(value).nbytes for value in payload.values())
        # DEFLATE can expand incompressible input slightly; headers and ZIP
        # directory entries also occupy space in the atomic sibling file.
        with reserve_disk_space(target.parent, expanded + expanded // 100 + 1024**2):
            report_operation("Saving dataset artifact")
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                if compressed:
                    write_array_archive(stream, payload)
                else:
                    np.savez(stream, **payload)
            report_operation("Publishing dataset artifact")
            temporary.replace(target)
            record_temporary_file(target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def dataset_artifact_bytes(data: MDHistoData | PointData4D | PointListData) -> bytes:
    """Serialize an analysis dataset for storage inside a project archive."""

    payload = _payload(data)
    expanded = sum(np.asarray(value).nbytes for value in payload.values())
    # Unlike the normal disk writer, this public helper retains the compressed
    # archive in RAM. Allow incompressible output and a possible getvalue copy.
    with reserve_memory(2 * (expanded + expanded // 100 + 1024**2),
                        operation="Serializing in-memory dataset artifact"):
        stream = BytesIO()
        write_array_archive(stream, payload)
        return stream.getvalue()


def read_project_dataset_artifact(
    project_path: str | Path,
    artifact_path: str,
    *,
    memory_map: bool | None = False,
) -> MDHistoData | PointData4D | PointListData:
    """Read an analysis dataset stored inside an nfit project."""

    with open_project_artifact(project_path, artifact_path) as stream:
        return read_dataset_artifact(
            stream, memory_map=memory_map, temp_dir=Path(project_path).absolute().parent
        )


def read_dataset_artifact(
    source: str | PathLike[str] | bytes | BinaryIO,
    *,
    memory_map: bool | None = False,
    temp_dir: str | PathLike[str] | None = None,
) -> MDHistoData | PointData4D | PointListData:
    """Read immutable data after reserving its expanded RAM footprint.

    ``True`` explicitly requests mapping for batch workflows. ``None`` retains
    ordinary whole-cube RAM loading. File-backed mappings expand beside their
    source unless ``temp_dir`` is supplied. Mapping failures propagate: a disk
    failure must never silently trigger a much larger resident allocation.
    """

    stream: str | PathLike[str] | BinaryIO
    stream = BytesIO(source) if isinstance(source, bytes) else source
    if temp_dir is None and isinstance(source, (str, PathLike)):
        temp_dir = Path(source).absolute().parent
    capacity = dataset_artifact_capacity(stream)
    position = stream.tell() if hasattr(stream, "tell") else None
    with reserve_memory(capacity.peak_bytes, operation="Loading saved dataset"):
        if memory_map is True:
            with np.load(stream, allow_pickle=False) as archive:
                histogram = str(np.asarray(archive["container"]).item()) in {"mdhisto", "point4d"}
            if hasattr(stream, "seek"):
                stream.seek(position or 0)
            if histogram:
                payload = read_mapped_array_archive(
                    stream, mapped_min_bytes=_MAPPED_MEMBER_MIN_BYTES, temp_dir=temp_dir
                )
                return dataset_artifact_from_payload(payload)
        with np.load(stream, allow_pickle=False) as archive:
            return dataset_artifact_from_payload(_OwnedArchiveArrays(archive))


class _OwnedArchiveArrays:
    """Transfer newly decoded arrays to immutable containers without copying.

    Only this private reader owns its inputs. The public mapping-based decoder
    must continue isolating writable arrays supplied by a caller.
    """

    def __init__(self, archive: Any) -> None:
        self.archive = archive
        self.files = set(archive.files)
        self.loaded: dict[str, np.ndarray] = {}
        self.independent_reader_factory = None
        self.parallel_names: frozenset[str] = frozenset()
        kind = str(np.asarray(archive["container"]).item())
        if kind == "mdhisto":
            names = ["signal", "errors", "mask", "num_events"]
            channels = (
                json.loads(str(np.asarray(archive["auxiliary_names"]).item()))
                if "auxiliary_names" in self.files else []
            )
            names += [
                f"aux_{index}_{field}"
                for index in range(len(channels))
                for field in ("values", "errors")
                if f"aux_{index}_{field}" in self.files
            ]
        else:
            names = []
        sizes = {name: archive.zip.getinfo(f"{name}.npy").file_size for name in names}
        largest = max(sizes.values(), default=0)
        workers = min(len(names), operation_worker_count(
            sum(sizes.values()),
            bytes_per_worker=max(16 * 1024**2, 2 * largest),
            min_parallel_bytes=32 * 1024**2,
        ))
        if workers > 1:
            # Stored project members can share the validated open descriptor
            # with independent positional readers. Generic streams and older
            # compressed outer members retain ZipFile's shared seek lock.
            self.independent_reader_factory = _project_artifact_reader_factory(archive.zip.fp)
            self.parallel_names = frozenset(names)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(copy_context().run, self._read, name) for name in names]
                try:
                    self.loaded.update((name, future.result()) for name, future in zip(names, futures, strict=True))
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

    def _read(self, name: str) -> np.ndarray:
        if self.independent_reader_factory is not None and name in self.parallel_names:
            with self.independent_reader_factory() as stream:
                with np.load(stream, allow_pickle=False) as archive:
                    array = _decode_array_member(archive, name)
        else:
            array = _decode_array_member(self.archive, name)
        array.setflags(write=False)
        return array

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.loaded:
            return self.loaded.pop(name)
        return self._read(name)


def _decode_array_member(archive: Any, name: str) -> np.ndarray:
    """Decode a single member in bounded cancellable chunks, without copies."""

    with archive.zip.open(f"{name}.npy") as stream:
        shape, fortran, dtype = _array_header(stream)
        if dtype.hasobject:
            raise ValueError("object arrays are not supported")
        report_operation(f"Decoding {name}")
        array = np.empty(shape, dtype=dtype, order="F" if fortran else "C")
        raw = memoryview(array.ravel(order="K")).cast("B")
        interval = 64 * 1024**2
        chunk = 4 * 1024**2
        for offset in range(0, len(raw), chunk):
            if offset % interval == 0:
                report_operation(f"Decoding {name}", completed=offset, total=len(raw))
            view = raw[offset:offset + chunk]
            filled = 0
            while filled < len(view):
                count = stream.readinto(view[filled:])
                if not count:
                    raise EOFError(f"array member {name!r} ended before its declared size")
                filled += count
        report_operation(f"Decoded {name}", completed=len(raw), total=len(raw))
        return array


def dataset_artifact_from_payload(archive: Any) -> MDHistoData | PointData4D | PointListData:
    """Reconstruct an artifact from an NPZ reader or in-memory array mapping."""

    kind = str(np.asarray(archive["container"]).item())
    metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if kind == "point4d":
        return restore_point_data_archive(archive, metadata)
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
    metadata = restore_background_metadata(metadata, archive)
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
    if "normalization_denominator" in archive:
        metadata["normalization_denominator"] = np.asarray(
            archive["normalization_denominator"], dtype=float
        )
    elif "normalization_denominator_auxiliary_channel" in archive:
        name = str(np.asarray(archive["normalization_denominator_auxiliary_channel"]).item())
        metadata["normalization_denominator"] = channels[name].values
    from ..measurement_dependencies import (
        restore_counting_dependencies,
        restore_source_dependencies,
    )

    return MDHistoData(axes, archive["signal"], archive["errors"], archive["mask"], archive["num_events"], coordinate_system=None if coordinate < 0 else coordinate, visual_normalization=None if visual < 0 else visual, metadata=metadata, auxiliary_channels=channels, source_dependencies=restore_source_dependencies(archive), counting_dependencies=restore_counting_dependencies(archive))


def output_data(output: DatasetOutput | TableOutput) -> MDHistoData | PointListData:
    return output.data


def _payload(data: MDHistoData | PointData4D | PointListData) -> dict[str, Any]:
    metadata = dict(data.metadata)
    denominator = metadata.get("normalization_denominator")
    if isinstance(denominator, np.ndarray):
        metadata.pop("normalization_denominator")
    metadata, background_arrays = background_metadata_payload(metadata)
    common: dict[str, Any] = {"format": np.asarray("nfit-analysis-artifact"), "version": np.asarray(1), "metadata_json": np.asarray(json.dumps(_json_metadata(metadata), sort_keys=True))}
    common.update(background_arrays)
    if isinstance(data, PointData4D):
        common.update({"container": np.asarray("point4d"), **point_data_archive_payload(data)})
        return common
    if isinstance(data, PointListData):
        common.update({"container": np.asarray("point_list"), "column_names_json": np.asarray(json.dumps(data.column_names)), "units_json": np.asarray(json.dumps(data.units)), "quantity_types_json": np.asarray(json.dumps(data.quantity_types)), "coordinate_names_json": np.asarray(json.dumps(data.coordinate_names)), "channels_json": np.asarray(json.dumps(data.channels))})
        common.update({f"column_{i}": data.column(name) for i, name in enumerate(data.column_names)})
        return common
    common.update({"container": np.asarray("mdhisto"), "signal": data.signal, "errors": data.errors, "mask": data.mask, "num_events": data.num_events, "axis_count": np.asarray(len(data.axes)), "coordinate_system": np.asarray(-1 if data.coordinate_system is None else data.coordinate_system), "visual_normalization": np.asarray(-1 if data.visual_normalization is None else data.visual_normalization)})
    from ..measurement_dependencies import (
        counting_dependency_archive_payload,
        source_dependency_archive_payload,
    )

    common.update(source_dependency_archive_payload(data.source_dependencies))
    common.update(counting_dependency_archive_payload(data.counting_dependencies))
    for i, axis in enumerate(data.axes):
        common.update({f"axis_{i}_name": np.asarray(axis.name), f"axis_{i}_values": axis.values, f"axis_{i}_units": np.asarray(axis.units), f"axis_{i}_kind": np.asarray(axis.kind), f"axis_{i}_frame": np.asarray(axis.frame or ""), f"axis_{i}_path": np.asarray(axis.path or ""), f"axis_{i}_metadata": np.asarray(json.dumps(_json_metadata(axis.metadata), sort_keys=True))})
    common["auxiliary_names"] = np.asarray(json.dumps(list(data.auxiliary_channels)))
    for i, channel in enumerate(data.auxiliary_channels.values()):
        common.update({f"aux_{i}_values": channel.values, f"aux_{i}_label": np.asarray(channel.label), f"aux_{i}_unit": np.asarray(channel.unit), f"aux_{i}_quantity_type": np.asarray(channel.quantity_type)})
        if channel.errors is not None:
            common[f"aux_{i}_errors"] = channel.errors
    if isinstance(denominator, np.ndarray):
        channel = data.auxiliary_channels.get("normalization_denominator")
        if channel is not None and np.shares_memory(channel.values, denominator):
            common["normalization_denominator_auxiliary_channel"] = np.asarray(
                "normalization_denominator"
            )
        else:
            common["normalization_denominator"] = denominator
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
