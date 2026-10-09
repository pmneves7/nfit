"""Incremental ZIP snapshots with bounded, explicitly committed readers.

The fixed prefix owns publication; ordinary ZIP readers are not recovery
readers. Numerical members remain ordinary ZIP members, and older generations
are retained until a separate compaction operation replaces the file.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import io
import os
import shutil
import struct
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .operation_control import report_operation

_PAGE = 4096
_PREFIX_BYTES = 3 * _PAGE
_MAGIC = b"NFITZIP2"
_SLOT_MAGIC = b"NFCOMMIT"
_HEADER = struct.Struct("<8sI16s")
_SLOT = struct.Struct("<8s16sQQQQ")
_VERSION = 1
_MAX_INDEX_BYTES = 256 * 1024**2
_HASH_BLOCK = 1024**2
_PINNED_READERS = ContextVar("nfit_pinned_project_readers", default=None)


class StoreConflictError(OSError):
    """The destination no longer has the revision the caller edited."""


class StoreBusyError(OSError):
    """Another cooperating process is writing this file."""


class StoreRecoveryError(ValueError, zipfile.BadZipFile):
    """The file requires recovery or a recovered snapshot cannot be edited."""


class StoreCommitUncertainError(OSError):
    """Publication began, but its durability could not be confirmed."""


@dataclass(frozen=True)
class StoreRevision:
    file_uuid: str
    generation: int
    committed_size: int


@dataclass(frozen=True)
class StoreInfo:
    format: str
    revision: StoreRevision | None
    recovered: bool = False


@dataclass(frozen=True)
class _Commit:
    revision: StoreRevision
    index_offset: int
    index_size: int
    index_digest: bytes
    slot: int


class _BoundedFile(io.RawIOBase):
    """A cursor over one pinned file descriptor, ending at a committed EOF."""

    def __init__(self, stream: BinaryIO, size: int):
        self._stream = stream
        self._size = size
        self._position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def fileno(self):
        return self._stream.fileno()

    def tell(self):
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self._position
        elif whence == os.SEEK_END:
            offset += self._size
        elif whence != os.SEEK_SET:
            raise ValueError("invalid seek origin")
        if offset < 0:
            raise ValueError("negative seek position")
        self._position = offset
        return offset

    def readinto(self, buffer):
        count = min(len(buffer), max(0, self._size - self._position))
        if not count:
            return 0
        if hasattr(os, "pread"):
            block = os.pread(self.fileno(), count, self._position)
        else:
            previous = self._stream.tell()
            self._stream.seek(self._position)
            block = self._stream.read(count)
            self._stream.seek(previous)
        buffer[:len(block)] = block
        self._position += len(block)
        return len(block)


def _checkpoint(stage: str) -> None:
    """Private fault-injection seam; never a cancellation point after publish."""


def _read_at(stream, offset, count):
    stream.seek(offset)
    return stream.read(count)


def _digest_range(stream, offset, size):
    digest = hashlib.sha256()
    stream.seek(offset)
    while size:
        block = stream.read(min(size, _HASH_BLOCK))
        if not block:
            raise StoreRecoveryError("Truncated project index")
        digest.update(block)
        size -= len(block)
    return digest.digest()


def _page(payload):
    signed = payload + hashlib.sha256(payload).digest()
    return signed.ljust(_PAGE, b"\0")


def _validate_page(page, payload_size):
    if len(page) != _PAGE:
        return None
    payload = page[:payload_size]
    if hashlib.sha256(payload).digest() != page[payload_size:payload_size + 32]:
        return None
    if any(page[payload_size + 32:]):
        return None
    return payload


def _validate_members(archive, index_offset):
    names = set()
    for item in archive.infolist():
        name = item.filename
        parts = name.removesuffix("/").split("/")
        if (not name or "\\" in name or "\0" in name
                or any(part in {"", ".", ".."} for part in parts) or name in names):
            raise StoreRecoveryError("Invalid or duplicate project member name")
        names.add(name)
        if (item.header_offset < _PREFIX_BYTES or item.compress_size < 0
                or item.header_offset + 30 + item.compress_size > index_offset):
            raise StoreRecoveryError("Project member lies outside its committed data")


def _open_snapshot(stream, commit):
    view = io.BufferedReader(_BoundedFile(stream, commit.revision.committed_size))
    try:
        archive = zipfile.ZipFile(view)
        try:
            if archive.start_dir != commit.index_offset:
                raise StoreRecoveryError("Project index does not match its commit")
            _validate_members(archive, commit.index_offset)
        except BaseException:
            archive.close()
            raise
        return archive, view
    except BaseException:
        view.close()
        raise


def _inspect(stream):
    header_page = _read_at(stream, 0, _PAGE)
    if header_page[:4] in {b"PK\x03\x04", b"PK\x05\x06"}:
        return StoreInfo("legacy", None), None
    header = _validate_page(header_page, _HEADER.size)
    if header is None:
        raise StoreRecoveryError("Invalid nfit storage header")
    magic, version, identity = _HEADER.unpack(header)
    if magic != _MAGIC or version != _VERSION:
        raise StoreRecoveryError("Unsupported nfit storage header")
    file_size = os.fstat(stream.fileno()).st_size
    valid = []
    invalid = False
    for slot in range(2):
        page = _read_at(stream, (slot + 1) * _PAGE, _PAGE)
        if page == bytes(_PAGE):
            continue
        payload = _validate_page(page, _SLOT.size + 32)
        if payload is None:
            invalid = True
            continue
        marker, slot_uuid, generation, end, offset, size = _SLOT.unpack(payload[:_SLOT.size])
        digest = payload[_SLOT.size:]
        if (marker != _SLOT_MAGIC or slot_uuid != identity or generation < 1
                or not _PREFIX_BYTES <= offset < end <= file_size
                or size != end - offset or not 22 <= size <= _MAX_INDEX_BYTES):
            invalid = True
            continue
        commit = _Commit(StoreRevision(str(uuid.UUID(bytes=identity)), generation, end),
                         offset, size, digest, slot)
        try:
            if _digest_range(stream, offset, size) != digest:
                raise StoreRecoveryError("Invalid project index checksum")
            archive, view = _open_snapshot(stream, commit)
            archive.close()
            view.close()
        except (StoreRecoveryError, zipfile.BadZipFile, EOFError, ValueError):
            invalid = True
            continue
        valid.append(commit)
    if not valid:
        raise StoreRecoveryError("No valid committed nfit generation")
    if len(valid) == 2 and valid[0].revision.generation == valid[1].revision.generation:
        raise StoreRecoveryError("Ambiguous nfit generation slots")
    commit = max(valid, key=lambda item: item.revision.generation)
    return StoreInfo("incremental", commit.revision, invalid), commit


def inspect_project_store(path: str | Path) -> StoreInfo:
    """Inspect committed storage; incomplete appended bytes are never scanned."""
    with Path(path).open("rb") as stream:
        info, _ = _inspect(stream)
        return info


def _file_identity(stream):
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@contextmanager
def open_project_zip(path: str | Path) -> Iterator[zipfile.ZipFile]:
    """Open a pinned committed ZIP, or a legacy project ZIP.

    The yielded archive exposes ``nfit_revision`` and ``nfit_recovered``.
    Its file descriptor remains pinned through atomic pathname replacement.
    """
    pinned = (_PINNED_READERS.get() or {}).get(str(Path(path).resolve()))
    if pinned is not None:
        yield pinned
        return
    with Path(path).open("rb") as stream:
        # Capture before reading commit slots: a writer can publish between
        # selecting a generation and opening its index. Load verification must
        # compare against that earlier snapshot, never a later file's stat.
        identity = _file_identity(stream)
        info, commit = _inspect(stream)
        if commit is None:
            with zipfile.ZipFile(stream) as archive:
                archive.nfit_revision = None
                archive.nfit_recovered = False
                archive.nfit_file_identity = identity
                yield archive
        else:
            archive, view = _open_snapshot(stream, commit)
            try:
                archive.nfit_revision = info.revision
                archive.nfit_recovered = info.recovered
                archive.nfit_file_identity = identity
                yield archive
            finally:
                archive.close()
                view.close()


def active_snapshot_identity(path):
    """Return the file identity pinned by a coordinated metadata-load operation."""
    pinned = (_PINNED_READERS.get() or {}).get(str(Path(path).resolve()))
    return None if pinned is None else pinned.nfit_file_identity


@contextmanager
def project_read_snapshot(path):
    """Keep manifest and lazy-cache binding on one opened project generation."""
    with open_project_zip(path) as archive:
        readers = dict(_PINNED_READERS.get() or {})
        readers[str(Path(path).resolve())] = archive
        token = _PINNED_READERS.set(readers)
        try:
            yield archive
        finally:
            _PINNED_READERS.reset(token)


@contextmanager
def _writer_lock(stream):
    if os.name == "nt":
        import msvcrt
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise StoreBusyError("Project is already being saved") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StoreBusyError("Project is already being saved") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _check_path(path, stream):
    try:
        matches = os.path.samestat(path.stat(), os.fstat(stream.fileno()))
    except FileNotFoundError:
        matches = False
    if not matches:
        raise StoreConflictError("Project pathname changed during saving")


def reuse_project_members(destination, current, names: Sequence[str] | None = None):
    """Reference unchanged members of this file without copying their bytes.

    Only use the ``current`` archive supplied to the same write callback.
    Cross-file transfers must stream member bytes instead.
    """
    if current is not getattr(destination, "_nfit_current", None) or current is None:
        raise ValueError("Members can only be reused from this transaction's snapshot")
    if names is None:
        names = current.namelist()
    for name in names:
        if name in destination.NameToInfo:
            raise ValueError(f"Duplicate project member {name!r}")
        item = copy.copy(current.getinfo(name))
        destination.filelist.append(item)
        destination.NameToInfo[name] = item


def _sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)


def _write_slot(stream, offset, page):
    position = 0
    while position < len(page):
        if hasattr(os, "pwrite"):
            written = os.pwrite(stream.fileno(), page[position:], offset + position)
        else:
            stream.seek(offset + position)
            written = stream.write(page[position:])
        if not written:
            raise OSError("Short project commit write")
        position += written


def _append_generation(stream, callback, current, commit, identity, *, before_commit=None):
    end = _PREFIX_BYTES if commit is None else commit.revision.committed_size
    stream.truncate(end)
    stream.seek(end)
    with zipfile.ZipFile(stream, "w", allowZip64=True) as destination:
        destination._nfit_current = current
        callback(destination, current)
        # start_dir points after the last new local member, before its index.
        index_offset = destination.start_dir
        _validate_members(destination, index_offset)
    committed_end = stream.tell()
    index_size = committed_end - index_offset
    if not 22 <= index_size <= _MAX_INDEX_BYTES:
        raise ValueError("Project index exceeds the supported size")
    _checkpoint("after_body_flush")
    os.fsync(stream.fileno())
    _checkpoint("after_body_fsync")
    digest = _digest_range(stream, index_offset, index_size)
    generation = 1 if commit is None else commit.revision.generation + 1
    slot = 0 if commit is None else 1 - commit.slot
    revision = StoreRevision(str(uuid.UUID(bytes=identity)), generation, committed_end)
    payload = _SLOT.pack(_SLOT_MAGIC, identity, generation, committed_end, index_offset, index_size) + digest
    if before_commit is not None:
        before_commit()
    report_operation("Publishing the saved project…")
    _checkpoint("before_slot")
    try:
        _write_slot(stream, (slot + 1) * _PAGE, _page(payload))
        _checkpoint("after_slot_write")
        os.fsync(stream.fileno())
        _checkpoint("after_slot_fsync")
    except BaseException as exc:
        # A reader may already have observed this slot. Never truncate here.
        raise StoreCommitUncertainError("Project commit may have been published; reopen before retrying") from exc
    return revision


def _create_incremental(path, callback):
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "r+b", buffering=0) as stream:
            identity = uuid.uuid4().bytes
            stream.write(_page(_HEADER.pack(_MAGIC, _VERSION, identity)) + bytes(2 * _PAGE))
            with _writer_lock(stream):
                try:
                    revision = _append_generation(stream, callback, None, None, identity)
                except StoreCommitUncertainError as exc:
                    # This private staging file has not been published at all.
                    raise OSError("Staged project could not be committed; destination was not published") from exc
                report_operation("Publishing the saved project…")
                _checkpoint("before_publish")
                try:
                    # Atomic no-overwrite publication also arbitrates racing creators.
                    os.link(temporary, path)
                except FileExistsError as exc:
                    raise StoreConflictError("Project destination was created during saving") from exc
                try:
                    _checkpoint("after_publish")
                    _sync_directory(path.parent)
                except BaseException as exc:
                    raise StoreCommitUncertainError("Project was published but directory durability is uncertain") from exc
                return revision
    finally:
        temporary.unlink(missing_ok=True)


def write_incremental(
    path: str | Path,
    callback: Callable[[zipfile.ZipFile, zipfile.ZipFile | None], None],
    *,
    expected_revision: StoreRevision | None = None,
) -> StoreRevision:
    """Write one generation, retaining unchanged members selected by callback.

    Existing legacy files are rejected: migrate to a new destination. New files
    publish atomically without replacing a racing creator. The writer never
    calls cancellation hooks after commit-slot publication starts.
    """
    path = Path(path)
    try:
        stream = path.open("r+b", buffering=0)
    except FileNotFoundError:
        if expected_revision is not None:
            raise StoreConflictError("Expected project revision no longer exists") from None
        return _create_incremental(path, callback)
    with stream, _writer_lock(stream):
        _check_path(path, stream)
        info, commit = _inspect(stream)
        if commit is None:
            raise ValueError("Migrate a legacy project to a new incremental destination")
        if info.recovered:
            raise StoreRecoveryError("Recovered project is read-only; save it to a new file")
        if expected_revision is not None and expected_revision != info.revision:
            raise StoreConflictError("Project was saved by another writer")
        current, view = _open_snapshot(stream, commit)
        try:
            current.nfit_revision = info.revision
            current.nfit_recovered = False
            current.nfit_file_identity = _file_identity(stream)
            return _append_generation(stream, callback, current, commit,
                                      uuid.UUID(info.revision.file_uuid).bytes,
                                      before_commit=lambda: _check_path(path, stream))
        finally:
            current.close()
            view.close()


def compact_incremental(path, callback, *, expected_revision=None):
    """Atomically replace a store with a new identity and caller-copied closure.

    The callback receives a new empty ZIP and the pinned current ZIP. It must
    copy bytes; local-member offsets cannot be reused across the two files.
    Existing readers retain the old inode. A recovered snapshot is deliberately
    rejected; recovery exports must name a new destination.
    """
    path = Path(path)
    with path.open("r+b", buffering=0) as stream, _writer_lock(stream):
        _check_path(path, stream)
        info, commit = _inspect(stream)
        if commit is None:
            raise ValueError("Only incremental projects can be compacted")
        if info.recovered:
            raise StoreRecoveryError("Recovered project is read-only; save it to a new file")
        if expected_revision is not None and info.revision != expected_revision:
            raise StoreConflictError("Project was saved by another writer")
        current, view = _open_snapshot(stream, commit)
        current.nfit_revision = info.revision
        current.nfit_recovered = False
        current.nfit_file_identity = _file_identity(stream)
        try:
            with tempfile.TemporaryDirectory(prefix=f".{path.name}-compact-", dir=path.parent) as directory:
                candidate = Path(directory) / path.name
                revision = _create_incremental(candidate, lambda destination, _: callback(destination, current))
                saved = candidate.stat()
                shutil.copystat(path, candidate)
                os.utime(candidate, ns=(saved.st_atime_ns, saved.st_mtime_ns))
                with candidate.open("r+b", buffering=0) as replacement, _writer_lock(replacement):
                    os.fsync(replacement.fileno())
                    _check_path(path, stream)
                    report_operation("Publishing the compacted project…")
                    _checkpoint("before_replace")
                    os.replace(candidate, path)
                    try:
                        _checkpoint("after_replace")
                        _sync_directory(path.parent)
                    except BaseException as exc:
                        raise StoreCommitUncertainError("Compacted project was published but durability is uncertain") from exc
                return revision
        finally:
            current.close()
            view.close()
