from __future__ import annotations

import copy
import io
import json
import os
import shutil
import struct
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .operation_control import report_operation
from .project_store import (
    StoreRevision,
    compact_incremental,
    inspect_project_store,
    open_project_zip,
    reuse_project_members,
    write_incremental,
)
from .storage_budget import reserve_disk_space

PROJECT_MANIFEST = "project.json"
ANALYSIS_ASSET_ROOT = PurePosixPath("assets", "analyses")
DATASET_ASSET_ROOT = PurePosixPath("assets", "datasets")
BINNING_ASSET_ROOT = PurePosixPath("assets", "binnings")
REDUCED_EVENT_ASSET_ROOT = PurePosixPath("assets", "reduced_events")
_ARCHIVE_COPY_BUFFER_BYTES = 8 * 1024**2

@dataclass(frozen=True)
class ArchiveMember:
    """A member in another archive that should be streamed into this one."""

    path: Path
    member: str
    identity: tuple[Any, ...] | None = field(default=None, repr=False)

    def __post_init__(self):
        # A deferred copy must not silently read a replacement source project.
        object.__setattr__(self, "path", Path(self.path))
        if self.identity is None:
            with open_project_zip(self.path) as archive:
                object.__setattr__(self, "identity", archive_member_identity(archive, self.member))


ArchiveContent = bytes | Path | ArchiveMember


class _StoredMemberView(io.RawIOBase):
    """Seekable, bounded view of one uncompressed ZIP member."""

    def __init__(self, stream: BinaryIO, offset: int, size: int, *, positional_read=None):
        self._stream = stream
        self._offset = int(offset)
        self._size = int(size)
        self._position = 0
        self._positional_read = positional_read

    def independent_reader(self) -> BinaryIO:
        """Clone the validated range, retaining the original open-file snapshot.

        Positional reads have independent cursors without reopening the path or
        duplicating a file descriptor's shared seek position. The owner keeps
        the underlying descriptor open until its parallel readers have joined.
        """
        return io.BufferedReader(
            _StoredMemberView(
                self._stream, self._offset, self._size, positional_read=os.pread
            ),
            buffer_size=1024 * 1024,
        )

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = int(offset)
        elif whence == os.SEEK_CUR:
            position = self._position + int(offset)
        elif whence == os.SEEK_END:
            position = self._size + int(offset)
        else:
            raise ValueError(f"unsupported seek mode {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = min(position, self._size)
        return self._position

    def readinto(self, buffer: Any) -> int:
        remaining = self._size - self._position
        if remaining <= 0:
            return 0
        view = memoryview(buffer).cast("B")
        count = min(len(view), remaining)
        if self._positional_read is None:
            self._stream.seek(self._offset + self._position)
            data = self._stream.read(count)
        else:
            data = self._positional_read(
                self._stream.fileno(), count, self._offset + self._position
            )
        view[: len(data)] = data
        self._position += len(data)
        return len(data)


def _project_artifact_reader_factory(stream: BinaryIO):
    """Return independent readers only for a validated stored-project view."""
    raw = getattr(stream, "raw", None)
    if isinstance(raw, _StoredMemberView) and callable(getattr(os, "pread", None)):
        return raw.independent_reader
    return None


@contextmanager
def open_project_artifact(path: str | Path, member: str, *, expected_identity=None) -> Iterator[BinaryIO]:
    """Open one project artifact without materializing its bytes in memory.

    Binning artifacts are stored as uncompressed members of the outer project
    ZIP.  In that common case the yielded object is a seekable bounded view of
    the project file, which lets readers such as ``numpy.load`` access the
    nested NPZ directly.  Older projects with a compressed outer member fall
    back to ``ZipExtFile``.
    """

    normalized = _safe_member(member)
    with open_project_zip(path) as archive:
        if expected_identity is not None and expected_identity != archive_member_identity(archive, normalized):
            raise ValueError("Project artifact changed; reload before reading its cached data")
        info = archive.getinfo(normalized)
        if info.compress_type != zipfile.ZIP_STORED:
            with archive.open(info, "r") as stream:
                yield stream
            return
        # Let ZipFile validate the local header, filename, overlap bounds, and
        # encryption flag before bypassing its shared seek lock.  Reading the
        # nested NPZ later validates each inner member's CRC; this bounded raw
        # view intentionally does not scan the whole outer member up front.
        with archive.open(info, "r"):
            pass
        # Reuse the descriptor whose ZIP header was validated. Reopening the
        # path here could select a newly replaced project with different data.
        raw = archive.fp
        raw.seek(info.header_offset)
        header = raw.read(30)
        if len(header) != 30:
            raise ValueError(f"truncated ZIP header for {normalized!r}")
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04":
            raise ValueError(f"invalid ZIP header for {normalized!r}")
        data_offset = info.header_offset + 30 + fields[-2] + fields[-1]
        view = io.BufferedReader(
            _StoredMemberView(raw, data_offset, info.file_size),
            buffer_size=1024 * 1024,
        )
        try:
            yield view
        finally:
            view.close()


def analysis_artifact_member(analysis_id: str, filename: str) -> str:
    """Return the archive member used for one analysis artifact."""

    safe_name = PurePosixPath(filename).name
    if not safe_name or safe_name != filename:
        raise ValueError(f"invalid analysis artifact filename {filename!r}")
    return str(ANALYSIS_ASSET_ROOT / analysis_id / safe_name)


def dataset_artifact_member(dataset_id: str, filename: str = "data.npz") -> str:
    """Return the archive member for a project-owned materialized dataset."""

    safe_id = PurePosixPath(dataset_id).name
    safe_name = PurePosixPath(filename).name
    if not safe_id or safe_id != dataset_id or not safe_name or safe_name != filename:
        raise ValueError("invalid project dataset artifact path")
    return str(DATASET_ASSET_ROOT / safe_id / safe_name)


def binning_artifact_member(cache_id: str, filename: str = "data.npz") -> str:
    """Return the archive member for one optional persisted rebin cache."""

    safe_id = PurePosixPath(cache_id).name
    safe_name = PurePosixPath(filename).name
    if not safe_id or safe_id != cache_id or not safe_name or safe_name != filename:
        raise ValueError("invalid project binning artifact path")
    return str(BINNING_ASSET_ROOT / safe_id / safe_name)


def read_project_manifest(path: str | Path) -> dict[str, Any]:
    """Read and decode the JSON manifest from a single-file nfit project."""

    with open_project_zip(path) as archive:
        try:
            payload = archive.read(PROJECT_MANIFEST)
        except KeyError as exc:
            raise ValueError("nfit project does not contain project.json") from exc
    return json.loads(payload.decode("utf-8"))


def read_project_artifact(path: str | Path, member: str) -> bytes:
    """Read one internal project artifact."""

    normalized = _safe_member(member)
    with open_project_zip(path) as archive:
        return archive.read(normalized)


def project_artifact_exists(path: str | Path, member: str) -> bool:
    """Return whether an internal artifact is present in a project."""

    try:
        normalized = _safe_member(member)
        with open_project_zip(path) as archive:
            archive.getinfo(normalized)
    except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def project_artifact_size(path: str | Path, member: str) -> int | None:
    """Return the uncompressed byte size of an internal artifact."""

    try:
        normalized = _safe_member(member)
        with open_project_zip(path) as archive:
            return int(archive.getinfo(normalized).file_size)
    except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


def project_artifact_compressed_size(path: str | Path, member: str) -> int | None:
    """Return the bytes occupied by an internal artifact in the archive."""

    try:
        normalized = _safe_member(member)
        with open_project_zip(path) as archive:
            return int(archive.getinfo(normalized).compress_size)
    except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


def write_project_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    asset_source: str | Path | None = None,
    preserve_existing: bool = True,
    binning_artifacts: Mapping[str, ArchiveContent] | None = None,
    reduced_event_artifacts: Mapping[str, ArchiveContent] | None = None,
    storage_mode: str | None = None,
    expected_revision=None,
) -> StoreRevision | None:
    """Save a manifest, returning its incremental revision or ``None`` for ZIP."""

    target = Path(path)
    source = (
        Path(asset_source)
        if asset_source is not None
        else (target if preserve_existing else None)
    )
    manifest = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    replacements = {PROJECT_MANIFEST: manifest}
    if binning_artifacts is not None:
        replacements.update(binning_artifacts)
    if reduced_event_artifacts is not None:
        replacements.update(reduced_event_artifacts)
    return _rewrite_archive(
        target,
        source if source is not None and source.exists() else None,
        replacements,
        remove_prefix=(
            str(BINNING_ASSET_ROOT) + "/"
            if binning_artifacts is not None
            else None
        ),
        remove_members_prefixes=(
            (str(REDUCED_EVENT_ASSET_ROOT) + "/",)
            if reduced_event_artifacts is not None else ()
        ),
        storage_mode=storage_mode,
        expected_revision=expected_revision,
    )


def replace_analysis_artifacts(
    path: str | Path,
    analysis_id: str,
    artifacts: Mapping[str, ArchiveContent],
) -> dict[str, str]:
    """Atomically replace every artifact for one analysis in a project."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"save the nfit project before running an analysis: {target}")
    prefix = str(ANALYSIS_ASSET_ROOT / analysis_id) + "/"
    replacements = {
        analysis_artifact_member(analysis_id, filename): content
        for filename, content in artifacts.items()
    }
    _rewrite_archive(target, target, replacements, remove_prefix=prefix)
    return {
        filename: analysis_artifact_member(analysis_id, filename)
        for filename in artifacts
    }


def replace_dataset_artifact(
    path: str | Path,
    dataset_id: str,
    content: ArchiveContent,
) -> str:
    """Atomically store one materialized dataset inside a project archive."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"save the nfit project before materializing data: {target}")
    prefix = str(DATASET_ASSET_ROOT / dataset_id) + "/"
    member = dataset_artifact_member(dataset_id)
    _rewrite_archive(
        target,
        target,
        {member: content},
        remove_prefix=prefix,
    )
    return member


def _rewrite_archive(
    target: Path,
    source: Path | None,
    replacements: Mapping[str, ArchiveContent],
    *,
    remove_prefix: str | None = None,
    remove_members_prefixes: tuple[str, ...] = (),
    storage_mode: str | None = None,
    expected_revision=None,
) -> StoreRevision | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if storage_mode not in {None, "legacy", "incremental"}:
        raise ValueError("storage_mode must be 'legacy', 'incremental', or None")
    existing_mode = inspect_project_store(target).format if target.exists() else "legacy"
    mode = storage_mode or existing_mode
    if mode == "incremental":
        return _write_incremental_archive(
            target, source, replacements, remove_prefix=remove_prefix,
            remove_members_prefixes=remove_members_prefixes,
            expected_revision=expected_revision,
        )
    if existing_mode == "incremental":
        raise ValueError("Use a new Save As destination to convert an incremental project to legacy ZIP")
    required = 0
    with ExitStack() as sources:
        archives = {}
        if source is not None:
            archive = sources.enter_context(open_project_zip(source))
            archives[source.resolve()] = archive
            required += sum(info.file_size + 256 + len(info.filename) * 2 for info in archive.infolist()
                if info.filename not in replacements and not (
                    remove_prefix is not None and info.filename.startswith(remove_prefix)
                ) and not info.filename.startswith(remove_members_prefixes))
        for content in replacements.values():
            if isinstance(content, ArchiveMember):
                key = content.path.resolve()
                archive = archives.get(key)
                if archive is None:
                    archive = sources.enter_context(open_project_zip(content.path))
                    archives[key] = archive
                _validate_reference(content, archive)
                required += archive.getinfo(_safe_member(content.member)).file_size
            elif isinstance(content, Path):
                required += content.stat().st_size
            else:
                required += len(content)
            required += 1024
    with reserve_disk_space(target.parent, required, operation="Saving the project"):
        _rewrite_archive_reserved(target, source, replacements,
            remove_prefix=remove_prefix, remove_members_prefixes=remove_members_prefixes)


def _copy_archive_stream(source, destination):
    while True:
        report_operation("Copying project cache data…")
        block = source.read(_ARCHIVE_COPY_BUFFER_BYTES)
        if not block:
            break
        destination.write(block)


def _rewrite_archive_reserved(
    target, source, replacements, *, remove_prefix=None, remove_members_prefixes=(),
):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as destination:
            if source is not None:
                try:
                    with open_project_zip(source) as existing:
                        seen: set[str] = set()
                        for info in existing.infolist():
                            name = _safe_member(info.filename)
                            if name in seen:
                                raise ValueError(f"duplicate archive member {name!r}")
                            seen.add(name)
                            if name in replacements or (
                                remove_prefix is not None and name.startswith(remove_prefix)
                            ) or name.startswith(remove_members_prefixes):
                                continue
                            with existing.open(info, "r") as source_stream:
                                with destination.open(copy.copy(info), "w") as target_stream:
                                    _copy_archive_stream(source_stream, target_stream)
                except zipfile.BadZipFile as exc:
                    raise ValueError(f"not a single-file nfit project: {source}") from exc
            with ExitStack() as sources:
                _write_replacements(destination, replacements, sources, {})
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        report_operation("Publishing the saved project…")
        if target.exists():
            # Atomic replacement must retain the shared destination's access
            # settings rather than publish NamedTemporaryFile's private mode.
            # copystat also retains supported ACL/xattrs, without changing
            # ownership. Keep the timestamp of this newly completed save.
            saved = temporary.stat()
            shutil.copystat(target, temporary)
            os.utime(temporary, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        temporary.replace(target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def archive_member_identity(archive, member):
    """Identify an immutable artifact independently of unrelated saved objects.

    Legacy copies retain their ZIP CRC/size contract. Incremental snapshots also
    retain a file UUID and immutable local offset, rejecting compaction or a
    replacement generation without reading the scientific payload.
    """
    info = archive.getinfo(_safe_member(member))
    payload = (info.CRC, info.file_size, info.compress_size, info.compress_type)
    revision = archive.nfit_revision
    return (("incremental", revision.file_uuid, info.header_offset, *payload)
            if revision is not None else ("legacy", *payload))


def _validate_reference(content, archive):
    if content.identity != archive_member_identity(archive, content.member):
        raise ValueError(f"Source project changed before copying {content.member!r}: {content.path}")


def _write_replacements(destination, replacements, sources, archives):
    for name, content in replacements.items():
        normalized = _safe_member(name)
        compression = zipfile.ZIP_DEFLATED if normalized == PROJECT_MANIFEST else zipfile.ZIP_STORED
        if isinstance(content, ArchiveMember):
            key = Path(content.path).resolve()
            archive = archives.get(key)
            if archive is None:
                archive = sources.enter_context(open_project_zip(content.path))
                archives[key] = archive
            _validate_reference(content, archive)
            with archive.open(_safe_member(content.member)) as source_stream:
                with destination.open(normalized, "w", force_zip64=True) as target_stream:
                    _copy_archive_stream(source_stream, target_stream)
        elif isinstance(content, Path):
            info = zipfile.ZipInfo(normalized)
            info.compress_type = compression
            with content.open("rb") as source_stream:
                with destination.open(info, "w", force_zip64=True) as target_stream:
                    _copy_archive_stream(source_stream, target_stream)
        else:
            report_operation("Writing project settings…")
            destination.writestr(normalized, content, compress_type=compression)


def _write_incremental_archive(target, source, replacements, *, remove_prefix,
                               remove_members_prefixes, expected_revision):
    """Reuse unchanged opaque artifacts; append only new bytes and the index."""
    same_source = source is not None and source.resolve() == target.resolve()
    with ExitStack() as sources:
        source_archive = sources.enter_context(open_project_zip(source)) if source is not None else None
        pending = dict(replacements)
        reuse = set()
        if same_source and source_archive is not None:
            for name, content in tuple(pending.items()):
                if (isinstance(content, ArchiveMember)
                        and content.path.resolve() == target.resolve()
                        and _safe_member(content.member) == _safe_member(name)):
                    _validate_reference(content, source_archive)
                    source_archive.getinfo(name)
                    reuse.add(name)
                    del pending[name]
        preserved = []
        if source_archive is not None:
            preserved = [info.filename for info in source_archive.infolist()
                         if info.filename not in replacements and not (
                             remove_prefix is not None and info.filename.startswith(remove_prefix)
                         ) and not info.filename.startswith(remove_members_prefixes)]
        required = 16 * 1024 + 512 * (len(preserved) + len(replacements))
        if not same_source and source_archive is not None:
            required += sum(source_archive.getinfo(name).file_size for name in preserved)
        for content in pending.values():
            if isinstance(content, ArchiveMember):
                with open_project_zip(content.path) as archive:
                    _validate_reference(content, archive)
                    required += archive.getinfo(_safe_member(content.member)).file_size
            else:
                required += content.stat().st_size if isinstance(content, Path) else len(content)

        def write(destination, current):
            archives = {}
            if same_source and source_archive is not None:
                # Snapshot opened before staging new bytes; never reopen a name
                # after a concurrent replacement or after this append starts.
                if current.nfit_revision != source_archive.nfit_revision:
                    raise ValueError("Source project changed while preparing the save")
                reuse_project_members(destination, current, [*preserved, *sorted(reuse)])
                archives[target.resolve()] = current
            elif source_archive is not None:
                for name in preserved:
                    info = source_archive.getinfo(name)
                    with source_archive.open(info) as src, destination.open(copy.copy(info), "w") as dst:
                        _copy_archive_stream(src, dst)
                archives[source.resolve()] = source_archive
            _write_replacements(destination, pending, sources, archives)

        with reserve_disk_space(target.parent, required, operation="Saving the project"):
            return write_incremental(target, write, expected_revision=expected_revision)


def project_storage_info(path: str | Path) -> dict[str, Any]:
    """Inspect storage and committed bytes without decoding numerical arrays."""
    with open_project_zip(path) as archive:
        payload_bytes = sum(item.compress_size for item in archive.infolist())
        revision, recovered = archive.nfit_revision, archive.nfit_recovered
    return {"format": "incremental" if revision is not None else "legacy",
            "revision": revision, "recovered": recovered,
            "file_bytes": Path(path).stat().st_size, "live_payload_bytes": payload_bytes}


def compact_project(path: str | Path) -> None:
    """Reclaim replaced/deleted objects in an incremental project explicitly.

    This copies live artifact bytes once and needs space for a second file.
    Ordinary saves never perform compaction.
    """
    target = Path(path)
    with open_project_zip(target) as archive:
        required = sum(info.file_size + 1024 for info in archive.infolist()) + 16 * 1024

    def copy_live(destination, current):
        for info in current.infolist():
            with current.open(info) as src, destination.open(copy.copy(info), "w") as dst:
                _copy_archive_stream(src, dst)

    with reserve_disk_space(target.parent, required, operation="Compacting the project"):
        compact_incremental(target, copy_live)


def _safe_member(member: str) -> str:
    path = PurePosixPath(str(member).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid nfit archive member {member!r}")
    return str(path)
