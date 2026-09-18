"""Read large NumPy archives without retaining their arrays in the Python heap.

NPZ members are compressed streams and therefore cannot normally be memory
mapped.  This module expands selected members into private temporary ``.npy``
files and maps those files read-only.  On POSIX the names are removed as soon
as the mapping is open; the operating system releases the storage when the
last array view releases the mapping.
"""

from __future__ import annotations

import io
import mmap
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

import numpy as np

from .performance import operation_worker_count

_COPY_CHUNK_BYTES = 4 * 1024**2
_MEMORY_BACKED_FILESYSTEMS = frozenset({"tmpfs", "ramfs"})
_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")


class MappedArchiveError(ValueError):
    """An archive cannot safely be loaded as a mapped array payload."""


class MappedWorkspaceError(OSError):
    """Temporary storage is unavailable for a mapped archive payload."""


@dataclass(frozen=True)
class ArrayStorageBytes:
    """Distinct array bytes split by their primary storage location."""

    heap: int = 0
    mapped: int = 0

    @property
    def total(self) -> int:
        return self.heap + self.mapped


def read_mapped_array_archive(
    source: str | os.PathLike[str] | bytes | BinaryIO,
    *,
    mapped_min_bytes: int,
    temp_dir: str | os.PathLike[str] | None = None,
) -> Mapping[str, np.ndarray]:
    """Return an NPZ payload with large members backed by read-only mappings.

    The caller may pass the returned mapping directly to an existing payload
    decoder.  Failures leave no named temporary files behind and never alter
    ``source``. Callers may fall back to resident loading after
    ``MappedWorkspaceError``. Archive corruption and invalid array contents
    propagate separately and should not trigger a second decoding attempt.
    """

    threshold = max(int(mapped_min_bytes), 0)
    stream: str | os.PathLike[str] | BinaryIO
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        with ZipFile(stream, "r") as archive:
            members = _array_members(archive)
            mapped_size = sum(
                info.file_size for info in members.values()
                if info.file_size >= threshold
            )
            if mapped_size and not _supports_unlinked_mappings():
                raise MappedWorkspaceError(
                    "mapped archive extraction requires POSIX unlink semantics"
                )
            _require_disk_space(mapped_size, temp_dir=temp_dir)
            result: dict[str, np.ndarray] = {}
            mapped_arrays: dict[str, np.ndarray] = {}
            try:
                mapped_members = [
                    (name, info) for name, info in members.items()
                    if info.file_size >= threshold
                ]
                mapped_arrays = _inflate_mapped_members(
                    archive,
                    mapped_members,
                    mapped_size=mapped_size,
                    temp_dir=temp_dir,
                )
                for name, info in members.items():
                    if name in mapped_arrays:
                        result[name] = mapped_arrays.pop(name)
                    else:
                        with archive.open(info, "r") as member:
                            result[name] = _load_array(io.BytesIO(member.read()))
                return result
            except BaseException:
                _close_mapped_arrays(mapped_arrays.values())
                _close_mapped_arrays(result.values())
                raise
    except BadZipFile:
        raise


def mapped_array_backing(value: Any) -> mmap.mmap | None:
    """Return the mmap retained by an ndarray or one of its base arrays."""

    current = value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, mmap.mmap):
            return current
        mapping = getattr(current, "_mmap", None)
        if isinstance(mapping, mmap.mmap):
            return mapping
        current = getattr(current, "base", None)
    return None


def is_mapped_array(value: Any) -> bool:
    """Return whether an ndarray ultimately reads from an mmap."""

    return isinstance(value, np.ndarray) and mapped_array_backing(value) is not None


def array_storage_nbytes(value: Any) -> ArrayStorageBytes:
    """Count distinct ndarray bytes, separating heap and mapped storage."""

    heap, mapped = _storage_nbytes(value, seen=set(), seen_storage=set())
    return ArrayStorageBytes(heap=heap, mapped=mapped)


def _array_members(archive: ZipFile) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith(".npy"):
            continue
        name = info.filename[:-4]
        if not name or "/" in name or "\\" in name:
            raise MappedArchiveError("array archive contains an invalid member name")
        if name in result:
            raise MappedArchiveError(f"array archive contains duplicate member {name!r}")
        result[name] = info
    if not result:
        raise MappedArchiveError("array archive contains no NumPy members")
    return result


def _supports_unlinked_mappings() -> bool:
    """Return whether open temporary mappings can be unlinked safely."""

    return os.name == "posix"


def _require_disk_space(size: int, *, temp_dir: str | os.PathLike[str] | None) -> None:
    if size <= 0:
        return
    root = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
    _require_disk_backed_temp(root)
    try:
        free = int(shutil.disk_usage(root).free)
    except OSError as exc:
        raise MappedWorkspaceError(
            f"cannot inspect mapped-archive temporary storage at {root}"
        ) from exc
    reserve = max(1024**3, free // 10)
    if size > max(0, free - reserve):
        raise MappedWorkspaceError(
            f"mapped archive requires {size} temporary bytes plus {reserve} bytes "
            f"of free-space headroom, but only {free} are free"
        )


def _mountinfo_text() -> str | None:
    """Read Linux mount metadata when the proc filesystem exposes it."""

    try:
        return Path("/proc/self/mountinfo").read_text()
    except OSError:
        return None


def _decode_mountinfo_path(value: str) -> str:
    """Decode the octal escapes used for mountinfo path fields."""

    return _MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _resolved_temp_path(root: Path) -> Path:
    """Resolve the selected workspace for mount-point comparison."""

    return root.resolve()


def _require_disk_backed_temp(root: Path) -> None:
    """Reject Linux temporary directories backed by volatile memory."""

    content = _mountinfo_text()
    if not content:
        return
    resolved = _resolved_temp_path(root)
    selected: tuple[int, str] | None = None
    for line in content.splitlines():
        before, separator, after = line.partition(" - ")
        fields = before.split()
        filesystem_fields = after.split()
        if not separator or len(fields) < 5 or not filesystem_fields:
            continue
        mountpoint = Path(_decode_mountinfo_path(fields[4]))
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        candidate = (len(mountpoint.parts), filesystem_fields[0])
        if selected is None or candidate[0] >= selected[0]:
            selected = candidate
    if selected is not None and selected[1] in _MEMORY_BACKED_FILESYSTEMS:
        raise MappedWorkspaceError(
            f"mapped archive temporary storage at {resolved} uses "
            f"memory-backed {selected[1]}"
        )


def _inflate_mapped_members(
    archive: ZipFile,
    members: list[tuple[str, Any]],
    *,
    mapped_size: int,
    temp_dir: str | os.PathLike[str] | None,
) -> dict[str, np.ndarray]:
    """Inflate mapped members concurrently within shared CPU/RAM limits."""

    if not members:
        return {}
    workers = min(
        len(members),
        operation_worker_count(
            mapped_size,
            bytes_per_worker=3 * _COPY_CHUNK_BYTES,
            min_parallel_bytes=32 * 1024**2,
        ),
    )
    if workers <= 1:
        completed: dict[str, np.ndarray] = {}
        try:
            for name, info in members:
                completed[name] = _inflate_mapped_member(
                    archive, info, temp_dir=temp_dir
                )
        except BaseException:
            _close_mapped_arrays(completed.values())
            raise
        return completed

    completed = {}
    first_error: BaseException | None = None
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="nfit-mapped-archive"
    ) as executor:
        futures = [
            (name, executor.submit(
                _inflate_mapped_member, archive, info, temp_dir=temp_dir
            ))
            for name, info in members
        ]
        for name, future in futures:
            try:
                completed[name] = future.result()
            except CancelledError:
                continue
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    for _other_name, other in futures:
                        if other is not future:
                            other.cancel()
    if first_error is not None:
        _close_mapped_arrays(completed.values())
        raise first_error
    return completed


def _inflate_mapped_member(
    archive: ZipFile,
    info: Any,
    *,
    temp_dir: str | os.PathLike[str] | None,
) -> np.ndarray:
    try:
        descriptor, raw_path = tempfile.mkstemp(suffix=".npy", dir=temp_dir)
    except OSError as exc:
        raise MappedWorkspaceError(
            "cannot create mapped-archive temporary storage"
        ) from exc
    path = Path(raw_path)
    array: np.ndarray | None = None
    try:
        try:
            destination_stream = os.fdopen(descriptor, "wb")
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise MappedWorkspaceError(
                "cannot open mapped-archive temporary storage"
            ) from exc
        with destination_stream as destination:
            with archive.open(info, "r") as source:
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    try:
                        destination.write(chunk)
                    except OSError as exc:
                        raise MappedWorkspaceError(
                            "cannot write mapped-archive temporary storage"
                        ) from exc
            try:
                destination.flush()
            except OSError as exc:
                raise MappedWorkspaceError(
                    "cannot flush mapped-archive temporary storage"
                ) from exc
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except OSError as exc:
            raise MappedWorkspaceError(
                "cannot map archive temporary storage"
            ) from exc
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise MappedArchiveError("object arrays are not supported")
        array.setflags(write=False)
        backing = mapped_array_backing(array)
        if backing is None:
            raise MappedArchiveError("NumPy did not create a file mapping")
        try:
            path.unlink()
        except OSError as exc:
            raise MappedWorkspaceError(
                "cannot unlink mapped-archive temporary storage"
            ) from exc
        return array
    except BaseException:
        if array is not None:
            backing = mapped_array_backing(array)
            if backing is not None:
                backing.close()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Preserve the classified extraction error. A failed unlink is
            # already represented by MappedWorkspaceError on the success path.
            pass
        raise


def _close_mapped_arrays(values: Any) -> None:
    """Release distinct mappings owned by an incomplete archive result."""

    seen: set[int] = set()
    for value in values:
        backing = mapped_array_backing(value)
        if backing is None or id(backing) in seen:
            continue
        seen.add(id(backing))
        backing.close()


def _load_array(source: BinaryIO) -> np.ndarray:
    array = np.load(source, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise MappedArchiveError("object arrays are not supported")
    array.setflags(write=False)
    return array


def _storage_nbytes(
    value: Any, *, seen: set[int], seen_storage: set[tuple[str, int]]
) -> tuple[int, int]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return 0, 0
    identity = id(value)
    if identity in seen:
        return 0, 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        backing = mapped_array_backing(value)
        if backing is not None:
            owner = _array_owner(value, mapped=True)
            token = ("mapped", id(backing))
            if token in seen_storage:
                return 0, 0
            seen_storage.add(token)
            return 0, int(owner.nbytes)
        owner = _array_owner(value, mapped=False)
        token = ("heap", id(owner))
        if token in seen_storage:
            return 0, 0
        seen_storage.add(token)
        return int(owner.nbytes), 0
    if hasattr(value, "__cuda_array_interface__") and hasattr(value, "nbytes"):
        # GPU allocations are not file mappings. Preserve the existing cache
        # accounting without importing an optional accelerator package.
        return int(value.nbytes), 0
    if isinstance(value, memoryview):
        return int(value.nbytes), 0
    children: Any = ()
    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    elif isinstance(value, FunctionType):
        children = (cell.cell_contents for cell in (value.__closure__ or ()))
    elif is_dataclass(value) and not isinstance(value, type):
        # A FitDataBundle's DatasetEntry is project-owned. Charging it here
        # would count source arrays as if the result cache retained them.
        children = (
            getattr(value, field.name)
            for field in fields(value)
            if not (
                type(value).__name__ == "FitDataBundle"
                and field.name == "dataset"
            )
        )
    heap = mapped = 0
    for child in children:
        child_heap, child_mapped = _storage_nbytes(
            child, seen=seen, seen_storage=seen_storage
        )
        heap += child_heap
        mapped += child_mapped
    return heap, mapped


def _array_owner(value: np.ndarray, *, mapped: bool) -> np.ndarray:
    owner = value
    current = value
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            owner = current
        base = getattr(current, "base", None)
        if not isinstance(base, np.ndarray):
            break
        owner = base
        current = base
    return owner
