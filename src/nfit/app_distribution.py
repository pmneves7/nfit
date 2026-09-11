"""Locations and release metadata shared by installed apps and source checkouts."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version as metadata_version
from pathlib import Path


@dataclass(frozen=True)
class UpdateConfiguration:
    """Build-time GitHub release access used by the installed application."""

    repository: str = ""
    token: str = ""
    version: str = ""


def application_version() -> str:
    """Return the authoritative build version, with package metadata as fallback."""
    return update_configuration().version or metadata_version("nfit")


def platform_key() -> str:
    """Identify the installer ABI, independent of the UI's locale."""
    systems = {"darwin": "macos", "win32": "windows", "linux": "linux"}
    machines = {"aarch64": "arm64", "arm64": "arm64", "amd64": "x86_64", "x86_64": "x86_64"}
    return f"{systems.get(sys.platform, sys.platform)}-{machines.get(platform.machine().lower(), platform.machine().lower())}"


def update_configuration() -> UpdateConfiguration:
    """Return repository access embedded in an installer build.

    Beta installers may contain a repository-scoped, read-only token. Source
    checkouts and future public builds work without a token.
    """
    path = Path(__file__).parent / "resources" / "distribution.json"
    if not path.is_file():
        return UpdateConfiguration()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        repository = data.get("github_repository", "")
        token = data.get("github_token", "")
        build_version = data.get("version", "")
        if not all(
            isinstance(value, str)
            for value in (repository, token, build_version)
        ):
            return UpdateConfiguration()
        return UpdateConfiguration(
            repository=repository, token=token, version=build_version
        )
    except (OSError, ValueError, TypeError):
        return UpdateConfiguration()


def local_help_index(*, source_file: str | Path | None = None) -> Path | None:
    """Find packaged offline HTML first, then documentation built in a checkout."""
    package = Path(__file__).parent if source_file is None else Path(source_file).resolve().parent
    candidates = [package / "resources" / "help" / "index.html"]
    if len(package.parents) >= 2:
        candidates.append(package.parents[1] / "docs" / "_build" / "html" / "index.html")
    return next((path for path in candidates if path.is_file()), None)
