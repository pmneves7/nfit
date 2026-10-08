"""Independent desktop processes for new and saved nfit projects."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def project_window_command(path: str | Path | None = None) -> list[str]:
    """Return the current installation's command, preserving spaces in paths."""
    if getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        command = [sys.executable, "-c", "from nfit.app_bootstrap import main; raise SystemExit(main())"]
    if path is not None:
        command.extend(["--project", str(Path(path).expanduser().absolute())])
    return command


def launch_project_window(path: str | Path | None = None) -> subprocess.Popen:
    """Launch a project with independent GUI, jobs, and cache state."""
    if path is not None and not Path(path).expanduser().is_file():
        raise FileNotFoundError(f"Project does not exist: {path}")
    environment = os.environ.copy()
    # A frozen child is another application, not a worker borrowing a parent
    # bootloader session. Keep nfit's own bundled library paths intact.
    if getattr(sys, "frozen", False):
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return subprocess.Popen(project_window_command(path), env=environment,
                            stdin=subprocess.DEVNULL, start_new_session=True)
