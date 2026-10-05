"""Lazy owner-specific staging sessions and conservative orphan cleanup."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import tempfile
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

_SESSION_FILE = "session.json"
_SESSION_VERSION = 1
_POSIX_PERMISSIONS = os.name == "posix"
_LOCK = RLock()
_SESSIONS: dict[tuple[int, str, str, str], _Session] = {}


@dataclass(frozen=True)
class WorkspaceSessionInfo:
    """Ownership metadata; only confirmed dead local sessions are removable."""

    path: Path
    owner_path: str | None
    hostname: str | None
    pid: int | None
    state: str
    reason: str

    @property
    def removable(self) -> bool:
        return self.state == "orphaned"


@dataclass
class _Session:
    path: Path
    key: tuple[int, str, str, str]
    children: set[str] = field(default_factory=set)


def _effective_uid() -> int:
    if hasattr(os, "geteuid"):
        return os.geteuid()
    return os.getuid() if hasattr(os, "getuid") else 0


def _user_identity() -> str:
    if hasattr(os, "geteuid") or hasattr(os, "getuid"):
        return str(_effective_uid())
    # Platforms without numeric user IDs still need separate namespaces for
    # collaborators using the same project path. Home identity is metadata;
    # obtaining it does not create or inspect any directory.
    home = str(Path.home().absolute()).casefold()
    return "home-" + hashlib.sha256(home.encode()).hexdigest()[:24]


def owner_workspace_path(owner: Path, parent: Path) -> Path:
    """Return an owner's hidden workspace without creating any directory."""

    user = _user_identity()
    identity = hashlib.sha256(f"{user}:{owner.absolute()}".encode()).hexdigest()[:24]
    return parent / ".nfit-work" / f"u{user}-{identity}"


def _process_creation_time(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(pid).create_time()
    except (OSError, psutil.Error):
        return None


def _remove_empty_session(session: _Session) -> None:
    if session.children:
        return
    # Never remove unexpected files or another session's directory.
    try:
        if tuple(session.path.iterdir()) != (session.path / _SESSION_FILE,):
            return
        (session.path / _SESSION_FILE).unlink()
        session.path.rmdir()
    except OSError:
        return
    _SESSIONS.pop(session.key, None)
    _remove_empty_containers(session.path.parent)


def _remove_empty_containers(owner_root: Path) -> None:
    for parent in (owner_root, owner_root.parent):
        try:
            parent.rmdir()
        except OSError:
            break


def _session(owner: Path, parent: Path) -> _Session:
    uid = _effective_uid()
    key = (os.getpid(), _user_identity(), str(owner.absolute()), str(parent.absolute()))
    with _LOCK:
        existing = _SESSIONS.get(key)
        if existing is not None:
            return existing
        root = owner_workspace_path(owner, parent)
        if root.parent.is_symlink() or root.is_symlink():
            raise OSError(f"Scientific workspace must not be a symbolic link: {root}")
        try:
            root.parent.mkdir(mode=0o1777 if _POSIX_PERMISSIONS else 0o777)
        except FileExistsError:
            # Existing containers may belong to another user. Never change
            # their permissions; a non-writable choice fails at allocation.
            pass
        else:
            # Umask can remove shared traversal/write permissions. Apply the
            # sticky mode only to our new container, retaining inherited setgid.
            if _POSIX_PERMISSIONS:
                root.parent.chmod(0o1777 | (root.parent.stat().st_mode & 0o2000))
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            if _POSIX_PERMISSIONS and root.stat().st_mode & 0o077:
                raise OSError(f"Scientific owner workspace is not private: {root}") from None
        else:
            if _POSIX_PERMISSIONS:
                root.chmod(0o700)
        path = root / f"session-{os.getpid()}-{uuid.uuid4().hex}"
        path.mkdir(mode=0o700)
        if _POSIX_PERMISSIONS:
            path.chmod(0o700)
        metadata = {
            "version": _SESSION_VERSION, "owner_path": key[2], "uid": uid,
            "user_identity": key[1],
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "process_creation_time": _process_creation_time(os.getpid()),
            "created_time": time.time(),
        }
        try:
            with (path / _SESSION_FILE).open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, sort_keys=True)
            if _POSIX_PERMISSIONS:
                (path / _SESSION_FILE).chmod(0o600)
        except BaseException:
            # A partially written manifest is not a trustworthy live session.
            (path / _SESSION_FILE).unlink(missing_ok=True)
            path.rmdir()
            _remove_empty_containers(root)
            raise
        session = _Session(path, key)
        _SESSIONS[key] = session
        return session


def _cleanup_directory(directory: tempfile.TemporaryDirectory, session: _Session) -> None:
    if session.key[0] != os.getpid():
        # A fork must not retire the parent process's still-live cache files.
        directory._finalizer.detach()
        return
    try:
        directory.cleanup()
    finally:
        if not Path(directory.name).exists():
            with _LOCK:
                session.children.discard(directory.name)
                _remove_empty_session(session)


class SessionTemporaryDirectory:
    """TemporaryDirectory-compatible owner keeping a session alive."""

    def __init__(self, directory: tempfile.TemporaryDirectory, session: _Session):
        self.name = directory.name
        self._directory = directory
        self._session = session
        self._finalizer = weakref.finalize(self, _cleanup_directory, directory, session)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, *_args) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._finalizer.alive:
            self._finalizer()
        elif Path(self.name).exists():
            _cleanup_directory(self._directory, self._session)


def owner_temporary_directory(owner: Path, parent: Path, *, prefix: str) -> SessionTemporaryDirectory:
    """Allocate a private operation directory in the owner's lazy session."""

    with _LOCK:
        session = _session(owner, parent)
        directory = None
        try:
            directory = tempfile.TemporaryDirectory(prefix=prefix, dir=session.path)
            if _POSIX_PERMISSIONS:
                Path(directory.name).chmod(0o700)
        except BaseException:
            if directory is not None:
                directory.cleanup()
            _remove_empty_session(session)
            raise
        session.children.add(directory.name)
        return SessionTemporaryDirectory(directory, session)


def _read_session(path: Path, owner: Path) -> WorkspaceSessionInfo:
    with _LOCK:
        if any(session.path == path and session.children and session.key[:2] == (os.getpid(), _user_identity())
               for session in _SESSIONS.values()):
            return WorkspaceSessionInfo(path, str(owner.absolute()), socket.gethostname(),
                                        os.getpid(), "live", "Current process still owns staging")
    unknown = WorkspaceSessionInfo(path, None, None, None, "unknown", "Missing or invalid ownership metadata")
    if path.is_symlink() or not path.is_dir():
        return unknown
    try:
        manifest = path / _SESSION_FILE
        if manifest.is_symlink() or manifest.stat().st_size > 16 * 1024:
            return unknown
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        hostname, pid = metadata["hostname"], metadata["pid"]
        if (metadata["version"] != _SESSION_VERSION or metadata["owner_path"] != str(owner.absolute())
                or metadata.get("uid") != _effective_uid()
                or metadata.get("user_identity", str(metadata.get("uid"))) != _user_identity()
                or not isinstance(hostname, str) or not hostname or type(pid) is not int or pid < 1):
            return unknown
        creation = metadata.get("process_creation_time")
        if creation is not None and (type(creation) not in (int, float) or not math.isfinite(creation)):
            return unknown
    except (OSError, ValueError, KeyError, TypeError):
        return unknown
    identity = dict(path=path, owner_path=metadata["owner_path"], hostname=hostname, pid=pid)
    if hostname != socket.gethostname():
        return WorkspaceSessionInfo(**identity, state="foreign", reason="Process belongs to another host")
    try:
        import psutil

        current_creation = psutil.Process(pid).create_time()
    except ImportError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return WorkspaceSessionInfo(**identity, state="orphaned", reason="Local owner process no longer exists")
        except OSError:
            return WorkspaceSessionInfo(**identity, state="unknown", reason="Cannot verify the local owner process")
    except psutil.NoSuchProcess:
        return WorkspaceSessionInfo(**identity, state="orphaned", reason="Local owner process no longer exists")
    except (OSError, psutil.Error):
        return WorkspaceSessionInfo(**identity, state="unknown", reason="Cannot verify the local owner process")
    else:
        if creation is not None and abs(current_creation - creation) > 0.01:
            return WorkspaceSessionInfo(**identity, state="orphaned", reason="Local owner PID has been reused")
    return WorkspaceSessionInfo(**identity, state="live", reason="Local owner process is still present")


def inspect_owner_workspaces(owner: Path, parent: Path) -> tuple[WorkspaceSessionInfo, ...]:
    """Read session metadata without loading scientific files or changing disk."""

    root = owner_workspace_path(owner, parent)
    if root.is_symlink() or root.parent.is_symlink() or not root.is_dir():
        return ()
    try:
        paths = sorted(root.iterdir())
    except FileNotFoundError:
        return ()  # The last session may have retired after the directory check.
    return tuple(_read_session(path, owner) for path in paths
                 if path.name.startswith("session-"))


def cleanup_owner_workspaces(owner: Path, parent: Path, paths=None) -> tuple[Path, ...]:
    """Delete only confirmed dead same-host sessions, after rechecking ownership."""

    selected = None if paths is None else {Path(path).absolute() for path in paths}
    removed = []
    with _LOCK:
        for info in inspect_owner_workspaces(owner, parent):
            if selected is not None and info.path.absolute() not in selected:
                continue
            if not _read_session(info.path, owner).removable:
                continue
            shutil.rmtree(info.path)
            removed.append(info.path)
        root = owner_workspace_path(owner, parent)
        for directory in (root, root.parent):
            try:
                directory.rmdir()
            except OSError:
                break
    return tuple(removed)
