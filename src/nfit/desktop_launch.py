"""Launch system desktop applications without inheriting bundled libraries."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _inside_bundle(value: str, bundle: str) -> bool:
    if not value:
        return False
    try:
        root = os.path.normcase(os.path.abspath(bundle))
        candidate = os.path.normcase(os.path.abspath(value))
        return os.path.commonpath((root, candidate)) == root
    except ValueError:
        return False


def system_process_environment() -> dict[str, str]:
    """Copy the environment and undo frozen-app search paths for a child only.

    PyInstaller saves the original Linux library path before prepending its
    bundle. Package hooks can also add bundle paths to desktop/Qt discovery.
    Keep the user's external paths and desktop-session variables intact.
    """
    environment = os.environ.copy()
    bundle = getattr(sys, "_MEIPASS", None)
    for key in ("LD_LIBRARY_PATH", "LIBPATH"):
        original = environment.get(f"{key}_ORIG")
        if original is not None:
            environment[key] = original
        elif bundle is not None or getattr(sys, "frozen", False):
            environment.pop(key, None)
    if bundle is None:
        return environment
    path_lists = (
        "PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
        "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "GIO_EXTRA_MODULES",
        "GI_TYPELIB_PATH", "GTK_PATH",
    )
    for key in path_lists:
        if key not in environment:
            continue
        retained = [part for part in environment[key].split(os.pathsep)
                    if not _inside_bundle(part, str(bundle))]
        if retained:
            environment[key] = os.pathsep.join(retained)
        else:
            environment.pop(key, None)
    for key in ("GIO_MODULE_DIR", "GDK_PIXBUF_MODULE_FILE"):
        if _inside_bundle(environment.get(key, ""), str(bundle)):
            environment.pop(key, None)
    return environment


def open_local_document(path: str | Path) -> subprocess.Popen | None:
    """Ask the platform's default application to open a local document.

    Return the opener process for nonblocking exit monitoring on POSIX, or
    ``None`` after a native Windows association launch. No browser is guessed
    and no shell command is constructed from the filename.
    """
    document = Path(path).expanduser().resolve(strict=True)
    if sys.platform == "win32":
        os.startfile(str(document))
        return None
    if sys.platform == "darwin":
        opener = "open"
    elif sys.platform.startswith("linux"):
        opener = "xdg-open"
    else:
        raise OSError(f"No desktop document opener is configured for {sys.platform}.")
    environment = system_process_environment()
    executable = shutil.which(opener, path=environment.get("PATH", os.defpath))
    if executable is None:
        raise FileNotFoundError(f"The desktop opener {opener!r} is not installed or available on PATH.")
    return subprocess.Popen(
        [executable, document.as_uri()],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        shell=False,
        start_new_session=True,
    )


def show_in_file_manager(path: str | Path) -> subprocess.Popen | None:
    """Open a file's containing directory in the default desktop file manager."""
    return open_local_document(Path(path).expanduser().absolute().parent)
