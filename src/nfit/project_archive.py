from __future__ import annotations

import io
import json
import os
import shutil
import struct
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

PROJECT_MANIFEST = "project.json"
ANALYSIS_ASSET_ROOT = PurePosixPath("assets", "analyses")
DATASET_ASSET_ROOT = PurePosixPath("assets", "datasets")
BINNING_ASSET_ROOT = PurePosixPath("assets", "binnings")
_ARCHIVE_COPY_BUFFER_BYTES = 8 * 1024**2

@dataclass(frozen=True)
class ArchiveMember:
    """A member in another archive that should be streamed into this one."""

    path: Path
    member: str


ArchiveContent = bytes | Path | ArchiveMember


class _StoredMemberView(io.RawIOBase):
    """Seekable, bounded view of one uncompressed ZIP member."""

    def __init__(self, stream: BinaryIO, offset: int, size: int):
        self._stream = stream
        self._offset = int(offset)
        self._size = int(size)
        self._position = 0

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
        self._stream.seek(self._offset + self._position)
        data = self._stream.read(count)
        view[: len(data)] = data
        self._position += len(data)
        return len(data)


@contextmanager
def open_project_artifact(path: str | Path, member: str) -> Iterator[BinaryIO]:
    """Open one project artifact without materializing its bytes in memory.

    Binning artifacts are stored as uncompressed members of the outer project
    ZIP.  In that common case the yielded object is a seekable bounded view of
    the project file, which lets readers such as ``numpy.load`` access the
    nested NPZ directly.  Older projects with a compressed outer member fall
    back to ``ZipExtFile``.
    """

    normalized = _safe_member(member)
    with zipfile.ZipFile(path, "r") as archive:
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
        with Path(path).open("rb") as raw:
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

    with zipfile.ZipFile(path, "r") as archive:
        try:
            payload = archive.read(PROJECT_MANIFEST)
        except KeyError as exc:
            raise ValueError("nfit project does not contain project.json") from exc
    return json.loads(payload.decode("utf-8"))


def read_project_artifact(path: str | Path, member: str) -> bytes:
    """Read one internal project artifact."""

    normalized = _safe_member(member)
    with zipfile.ZipFile(path, "r") as archive:
        return archive.read(normalized)


def project_artifact_exists(path: str | Path, member: str) -> bool:
    """Return whether an internal artifact is present in a project."""

    try:
        normalized = _safe_member(member)
        with zipfile.ZipFile(path, "r") as archive:
            archive.getinfo(normalized)
    except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def project_artifact_size(path: str | Path, member: str) -> int | None:
    """Return the uncompressed byte size of an internal artifact."""

    try:
        normalized = _safe_member(member)
        with zipfile.ZipFile(path, "r") as archive:
            return int(archive.getinfo(normalized).file_size)
    except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


def project_artifact_compressed_size(path: str | Path, member: str) -> int | None:
    """Return the bytes occupied by an internal artifact in the archive."""

    try:
        normalized = _safe_member(member)
        with zipfile.ZipFile(path, "r") as archive:
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
) -> None:
    """Atomically save a manifest while preserving internal project assets."""

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
    _rewrite_archive(
        target,
        source if source is not None and source.exists() else None,
        replacements,
        remove_prefix=(
            str(BINNING_ASSET_ROOT) + "/"
            if binning_artifacts is not None
            else None
        ),
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
) -> None:
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
                    with zipfile.ZipFile(source, "r") as existing:
                        seen: set[str] = set()
                        for info in existing.infolist():
                            name = _safe_member(info.filename)
                            if name in seen:
                                raise ValueError(f"duplicate archive member {name!r}")
                            seen.add(name)
                            if name in replacements or (
                                remove_prefix is not None and name.startswith(remove_prefix)
                            ):
                                continue
                            with existing.open(info, "r") as source_stream:
                                with destination.open(info, "w") as target_stream:
                                    shutil.copyfileobj(
                                        source_stream,
                                        target_stream,
                                        length=_ARCHIVE_COPY_BUFFER_BYTES,
                                    )
                except zipfile.BadZipFile as exc:
                    raise ValueError(f"not a single-file nfit project: {source}") from exc
            with ExitStack() as sources:
                source_archives: dict[Path, zipfile.ZipFile] = {}
                for name, content in replacements.items():
                    normalized = _safe_member(name)
                    compression = (
                        zipfile.ZIP_DEFLATED
                        if normalized == PROJECT_MANIFEST
                        else zipfile.ZIP_STORED
                    )
                    if isinstance(content, ArchiveMember):
                        source_path = Path(content.path)
                        source_archive = source_archives.get(source_path)
                        if source_archive is None:
                            source_archive = sources.enter_context(
                                zipfile.ZipFile(source_path, "r")
                            )
                            source_archives[source_path] = source_archive
                        source_info = source_archive.getinfo(
                            _safe_member(content.member)
                        )
                        with source_archive.open(source_info, "r") as source_stream:
                            with destination.open(
                                normalized, "w", force_zip64=True
                            ) as target_stream:
                                shutil.copyfileobj(
                                    source_stream,
                                    target_stream,
                                    length=_ARCHIVE_COPY_BUFFER_BYTES,
                                )
                    elif isinstance(content, Path):
                        destination.write(
                            content,
                            arcname=normalized,
                            compress_type=compression,
                        )
                    else:
                        destination.writestr(
                            normalized,
                            content,
                            compress_type=compression,
                        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _safe_member(member: str) -> str:
    path = PurePosixPath(str(member).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid nfit archive member {member!r}")
    return str(path)
