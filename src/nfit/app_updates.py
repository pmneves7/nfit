"""GitHub release checks and verified installer downloads without GUI imports."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from packaging.version import InvalidVersion, Version

from .app_distribution import platform_key

RELEASE_LIMIT = 1024 * 1024
INSTALLER_LIMIT = 8 * 1024**3
INSTALLER_SUFFIXES = {"macos": ".pkg", "windows": ".exe", "linux": ".deb"}
GITHUB_API_VERSION = "2026-03-10"
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class UpdateError(Exception):
    """An update could not be checked or downloaded."""


class AuthenticationRequired(UpdateError):
    """The bundled beta-release credential is missing, expired, or revoked."""


class DownloadCancelled(UpdateError):
    """The user cancelled a download; the partial installer was removed."""


@dataclass(frozen=True)
class Release:
    version: str
    notes: str
    installer_url: str
    filename: str
    sha256: str
    size: int


def validate_repository(repository: str) -> str:
    """Validate the owner/name used to construct fixed GitHub API addresses."""
    repository = repository.strip()
    if not _REPOSITORY_PATTERN.fullmatch(repository):
        raise UpdateError("The GitHub update repository is invalid.")
    return repository


def latest_release_url(repository: str) -> str:
    """Return GitHub's latest published release endpoint for a repository."""
    return f"https://api.github.com/repos/{validate_repository(repository)}/releases/latest"


def _validate_api_url(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != "api.github.com"
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise UpdateError("GitHub returned an invalid release asset address.")
    return parts.geturl()


class _GitHubRedirect(HTTPRedirectHandler):
    """Allow HTTPS asset redirects while never forwarding the GitHub token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise UpdateError("GitHub redirected the download to an insecure address.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def _open_github(url: str, token: str, *, binary: bool = False):
    url = _validate_api_url(url)
    headers = {
        "Accept": "application/octet-stream" if binary else "application/vnd.github+json",
        "Accept-Encoding": "identity",
        "User-Agent": "nfit-updater",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return build_opener(_GitHubRedirect()).open(
            Request(url, headers=headers), timeout=30
        )
    except HTTPError as error:
        if error.code in (401, 403, 404):
            raise AuthenticationRequired(
                "This build cannot access the private nfit beta releases. "
                "Ask the maintainer for a current installer."
            ) from error
        raise UpdateError(f"GitHub returned HTTP {error.code}.") from error
    except (URLError, OSError) as error:
        raise UpdateError(
            "GitHub could not be reached. You can continue using nfit offline."
        ) from error


def parse_github_release(
    payload: bytes,
    *,
    repository: str,
    current_version: str,
    target: str | None = None,
) -> Release | None:
    """Select and validate the native installer in a GitHub release response."""
    repository = validate_repository(repository)
    if len(payload) > RELEASE_LIMIT:
        raise UpdateError("GitHub returned release metadata that is too large.")
    try:
        data = json.loads(payload)
        tag = data["tag_name"]
        if not isinstance(tag, str):
            raise ValueError("tag")
        latest = Version(tag.removeprefix("v"))
        current = Version(current_version)
        if latest.is_prerelease or latest.is_devrelease or latest <= current:
            return None
        key = platform_key() if target is None else target
        system = key.split("-", 1)[0]
        suffix = INSTALLER_SUFFIXES.get(system)
        if suffix is None:
            return None
        filename = f"nfit-{latest}-{key}{suffix}"
        assets = data["assets"]
        matches = [asset for asset in assets if asset.get("name") == filename]
        if len(matches) != 1:
            return None
        asset = matches[0]
        if asset.get("state") != "uploaded":
            raise ValueError("state")
        size = asset["size"]
        asset_id = asset["id"]
        digest = asset["digest"]
        if type(size) is not int or not 0 < size <= INSTALLER_LIMIT:
            raise ValueError("size")
        if type(asset_id) is not int or asset_id <= 0:
            raise ValueError("id")
        if not isinstance(digest, str) or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", digest
        ):
            raise ValueError("digest")
        notes = data.get("body", "") or ""
        if not isinstance(notes, str):
            raise ValueError("notes")
        notes = notes[:16000]
        url = f"https://api.github.com/repos/{repository}/releases/assets/{asset_id}"
        return Release(
            str(latest), notes, url, filename, digest.removeprefix("sha256:"), size
        )
    except (KeyError, TypeError, ValueError, AttributeError, InvalidVersion) as error:
        raise UpdateError("GitHub returned invalid release metadata.") from error


def check_for_update(
    repository: str,
    current_version: str,
    *,
    token: str = "",
    target: str | None = None,
) -> Release | None:
    """Check the latest published GitHub release without changing the app."""
    url = latest_release_url(repository)
    with _open_github(url, token) as response:
        payload = response.read(RELEASE_LIMIT + 1)
    return parse_github_release(
        payload,
        repository=repository,
        current_version=current_version,
        target=target,
    )


def download_installer(
    release: Release,
    directory: Path,
    *,
    token: str = "",
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Download an installer in isolation, returning it only after verification."""
    _validate_api_url(release.installer_url)
    if Path(release.filename).name != release.filename or not re.fullmatch(
        r"nfit-[A-Za-z0-9._-]+", release.filename
    ):
        raise UpdateError("Invalid installer filename.")
    if not 0 < release.size <= INSTALLER_LIMIT:
        raise UpdateError("Invalid installer size.")
    if not re.fullmatch(r"[a-f0-9]{64}", release.sha256):
        raise UpdateError("Invalid installer checksum.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < release.size + 64 * 1024**2:
        raise UpdateError("There is not enough free space to download the update.")
    staging = Path(tempfile.mkdtemp(prefix="nfit-update-", dir=directory))
    partial = staging / "download.partial"
    try:
        digest = hashlib.sha256()
        count = 0
        with _open_github(release.installer_url, token, binary=True) as response:
            with partial.open("xb") as output:
                while True:
                    if cancelled is not None and cancelled():
                        raise DownloadCancelled("The update download was cancelled.")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > release.size:
                        raise UpdateError(
                            "The installer is larger than GitHub's release metadata allows."
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    if progress is not None:
                        progress(count, release.size)
        if count != release.size or digest.hexdigest() != release.sha256:
            raise UpdateError(
                "The installer failed its integrity check. Nothing was installed."
            )
        if cancelled is not None and cancelled():
            raise DownloadCancelled("The update download was cancelled.")
        result = staging / release.filename
        partial.replace(result)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
