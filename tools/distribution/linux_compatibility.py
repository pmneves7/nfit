"""Build-time checks for Linux ELF glibc symbol-version requirements."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_ELF_MAGIC = b"\x7fELF"
MAX_GLIBC_VERSION = (2, 34)
_GLIBC_VERSION = re.compile(r"\bName:\s+(GLIBC_[A-Za-z0-9_.]+)")
_VERSION_SECTION = re.compile(r"^Version (?:needs|definition) section", re.MULTILINE)
_MISSING_LIBRARY = re.compile(r"^\s*(\S+)\s+=>\s+not found\s*$", re.MULTILINE)
_MISSING_VERSION = re.compile(r"version [`'\"]([^`'\"]+)[`'\"] not found")


@dataclass(frozen=True)
class GLIBCVersionRequirement:
    """A bundled ELF file requiring a glibc version above the supported limit."""

    path: Path
    required_version: str


class ELFCompatibilityError(RuntimeError):
    """Raised when the ELF inspection tool cannot inspect a bundled file."""


@dataclass(frozen=True)
class MissingSharedLibrary:
    """An ELF file with an unresolved shared-library or symbol-version need."""

    path: Path
    library: str


def _version_tuple(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = value.removeprefix("GLIBC_").split(".")
        return tuple(int(part) for part in parts)
    return tuple(int(part) for part in value)


def _needed_glibc_versions(output: str) -> list[str]:
    """Return GLIBC versions in version-needs sections, excluding definitions."""
    matches: list[str] = []
    sections = list(_VERSION_SECTION.finditer(output))
    for index, section in enumerate(sections):
        if not output.startswith("Version needs section", section.start()):
            continue
        end = sections[index + 1].start() if index + 1 < len(sections) else len(output)
        block = output[section.start():end]
        matches.extend(_GLIBC_VERSION.findall(block))
    return matches


def _requires_newer_glibc(version: str, limit: tuple[int, ...]) -> bool:
    suffix = version.removeprefix("GLIBC_")
    if suffix == "ABI_DT_RELR":
        return (2, 36) > limit
    if re.fullmatch(r"\d+(?:\.\d+)+", suffix):
        return _version_tuple(suffix) > limit
    # Unknown ABI tags, including GLIBC_PRIVATE, have no portable guarantee.
    return True


def _iter_elf_files(base: Path):
    """Yield each regular ELF file once, ignoring symlinks and hardlinks."""
    if not base.is_dir():
        raise ValueError(f"ELF audit path is not a directory: {base}")
    seen: set[tuple[int, int]] = set()
    for path in sorted(base.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.lstat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            with path.open("rb") as stream:
                if stream.read(4) != _ELF_MAGIC:
                    continue
        except OSError as exc:
            raise ELFCompatibilityError(f"cannot inspect {path}: {exc}") from exc
        seen.add(identity)
        yield path


def audit_glibc_compatibility(
    root: str | Path,
    max_version: str | Sequence[int] = MAX_GLIBC_VERSION,
    *,
    readelf: str = "readelf",
    runner: Callable[..., object] = subprocess.run,
) -> list[GLIBCVersionRequirement]:
    """Find ELF files that require glibc newer than ``max_version``.

    Symlinks are ignored (their targets are inspected as regular files), and
    hard-linked files are inspected once. The audit reads version *needs* only;
    version definitions describe symbols provided by a library and are not a
    compatibility requirement of that file.
    """
    limit = _version_tuple(max_version)
    base = Path(root)
    findings: list[GLIBCVersionRequirement] = []
    for path in _iter_elf_files(base):
        try:
            result = runner(
                [readelf, "--version-info", "--wide", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ELFCompatibilityError(f"readelf failed for {path}: {exc}") from exc
        versions = _needed_glibc_versions(getattr(result, "stdout", ""))
        for version in versions:
            if _requires_newer_glibc(version, limit):
                findings.append(GLIBCVersionRequirement(path, version))
    return findings


def audit_shared_library_dependencies(
    root: str | Path,
    *,
    ldd: str = "ldd",
    runner: Callable[..., object] = subprocess.run,
) -> list[MissingSharedLibrary]:
    """Find unresolved shared libraries and symbol versions in bundled ELFs.

    The bundle's ``_internal`` directory is searched first, while the caller's
    existing ``LD_LIBRARY_PATH`` remains available for system libraries.
    """
    base = Path(root)
    search_paths = [str(base / "_internal")]
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        search_paths.append(existing)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(search_paths)

    findings: list[MissingSharedLibrary] = []
    for path in _iter_elf_files(base):
        try:
            result = runner(
                [ldd, str(path)], capture_output=True, text=True, check=False, env=environment
            )
        except OSError as exc:
            raise ELFCompatibilityError(f"ldd failed for {path}: {exc}") from exc
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        output = f"{stdout}\n{stderr}"
        returncode = getattr(result, "returncode", 0)
        libraries = _MISSING_LIBRARY.findall(output)
        libraries.extend(f"version {version}" for version in _MISSING_VERSION.findall(output))
        if returncode and not libraries and not re.search(
            r"not a dynamic executable|statically linked", output, re.I
        ):
            raise ELFCompatibilityError(f"ldd failed for {path}: {output.strip() or f'exit status {returncode}'}")
        findings.extend(MissingSharedLibrary(path, library) for library in libraries)
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing the Linux bundle")
    default_glibc = ".".join(map(str, MAX_GLIBC_VERSION))
    parser.add_argument(
        "--max-glibc",
        default=default_glibc,
        help=f"highest allowed GLIBC version (default: {default_glibc})",
    )
    parser.add_argument("--readelf", default="readelf", help="readelf executable")
    parser.add_argument(
        "--check-libraries",
        action="store_true",
        help="also run ldd to detect unresolved bundled ELF dependencies",
    )
    parser.add_argument("--ldd", default="ldd", help="ldd executable")
    args = parser.parse_args(argv)

    try:
        findings = audit_glibc_compatibility(args.directory, args.max_glibc, readelf=args.readelf)
    except (ValueError, ELFCompatibilityError) as exc:
        print(f"ELF compatibility audit failed: {exc}", file=sys.stderr)
        return 2
    if findings:
        for finding in findings:
            print(f"{finding.path}: requires {finding.required_version} (maximum GLIBC_{args.max_glibc})")
        return 1
    if args.check_libraries:
        try:
            missing = audit_shared_library_dependencies(args.directory, ldd=args.ldd)
        except (ValueError, ELFCompatibilityError) as exc:
            print(f"ELF dependency audit failed: {exc}", file=sys.stderr)
            return 2
        if missing:
            for item in missing:
                print(f"{item.path}: unresolved {item.library}")
            return 1
        print("ELF dependency audit passed: all bundled ELF dependencies resolve")
    print(f"ELF compatibility audit passed: all bundled ELF files require at most GLIBC_{args.max_glibc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
