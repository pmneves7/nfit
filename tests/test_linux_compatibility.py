import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.distribution.linux_compatibility import (
    _needed_glibc_versions,
    audit_glibc_compatibility,
    audit_shared_library_dependencies,
)


def test_version_needs_parser_ignores_definitions_and_sorts_numeric_versions():
    output = """\
Version needs section '.gnu.version_r' contains 2 entries:
  0x0010: Version: 1  File: libc.so.6  Cnt: 2
  0x0020:   Name: GLIBC_2.9  Flags: none  Version: 3
  0x0030:   Name: GLIBC_2.34  Flags: none  Version: 4
Version definition section '.gnu.version_d' contains 1 entry:
  0x0000: Rev: 1  Flags: BASE  Index: 1  Cnt: 1
  0x001c: Name: GLIBC_9.99
"""

    assert _needed_glibc_versions(output) == ["GLIBC_2.9", "GLIBC_2.34"]


def test_audit_scans_elf_only_skips_symlinks_and_hardlink_duplicates(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    elf = bundle / "app"
    elf.write_bytes(b"\x7fELFfake payload")
    hardlink = bundle / "app-copy"
    hardlink.hardlink_to(elf)
    (bundle / "app-link").symlink_to(elf)
    (bundle / "notes.txt").write_text("not an ELF")
    calls = []

    def runner(command, **kwargs):
        calls.append(command[-1])
        return SimpleNamespace(stdout="""\
Version needs section '.gnu.version_r' contains 1 entry:
  0x0010: Name: GLIBC_2.35
""")

    findings = audit_glibc_compatibility(bundle, (2, 34), runner=runner)

    assert calls == [str(elf)]
    assert [(item.path, item.required_version) for item in findings] == [(elf, "GLIBC_2.35")]


def test_audit_accepts_numeric_older_versions_and_newest_supported(tmp_path: Path):
    elf = tmp_path / "library.so"
    elf.write_bytes(b"\x7fELF")
    result = SimpleNamespace(stdout="""\
Version needs section '.gnu.version_r' contains 2 entries:
  Name: GLIBC_2.9
  Name: GLIBC_2.34
""")

    assert audit_glibc_compatibility(tmp_path, "2.34", runner=lambda *a, **k: result) == []


def test_audit_rejects_dt_relr_tag_but_ignores_definition_tags(tmp_path: Path):
    elf = tmp_path / "library.so"
    elf.write_bytes(b"\x7fELF")
    result = SimpleNamespace(stdout="""\
Version needs section '.gnu.version_r' contains 1 entry:
  Name: GLIBC_ABI_DT_RELR
Version definition section '.gnu.version_d' contains 1 entry:
  Name: GLIBC_9.99
""")

    findings = audit_glibc_compatibility(tmp_path, runner=lambda *a, **k: result)

    assert [(item.path, item.required_version) for item in findings] == [
        (elf, "GLIBC_ABI_DT_RELR")
    ]


def test_audit_raises_when_readelf_fails(tmp_path: Path):
    elf = tmp_path / "broken.so"
    elf.write_bytes(b"\x7fELF")

    def runner(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="malformed ELF")

    with pytest.raises(RuntimeError, match="readelf failed"):
        audit_glibc_compatibility(tmp_path, runner=runner)


def test_audit_requires_existing_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="not a directory"):
        audit_glibc_compatibility(tmp_path / "missing")


def test_library_audit_preserves_search_path_and_reports_missing_dependencies(tmp_path: Path, monkeypatch):
    elf = tmp_path / "_internal" / "extension.so"
    elf.parent.mkdir()
    elf.write_bytes(b"\x7fELF")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib:/another/lib")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="""\
libQt6DataVisualization.so.6 => not found
libQt6Core.so.6 => /bundle/_internal/libQt6Core.so.6 (0x1234)
""",
            stderr="",
        )

    findings = audit_shared_library_dependencies(tmp_path, runner=runner)

    assert [(item.path, item.library) for item in findings] == [
        (elf, "libQt6DataVisualization.so.6")
    ]
    assert calls[0][1]["env"]["LD_LIBRARY_PATH"] == f"{tmp_path / '_internal'}:/existing/lib:/another/lib"
    assert calls[0][0] == ["ldd", str(elf)]


def test_library_audit_reports_missing_symbol_versions_and_accepts_static_elf(tmp_path: Path):
    elf = tmp_path / "program"
    elf.write_bytes(b"\x7fELF")

    findings = audit_shared_library_dependencies(
        tmp_path,
        runner=lambda *a, **k: SimpleNamespace(
            returncode=1,
            stdout="version `GLIBCXX_3.4.31' not found (required by program)",
            stderr="",
        ),
    )

    assert [(item.path, item.library) for item in findings] == [(elf, "version GLIBCXX_3.4.31")]


def test_library_audit_raises_for_unexpected_ldd_failure(tmp_path: Path):
    elf = tmp_path / "broken.so"
    elf.write_bytes(b"\x7fELF")

    with pytest.raises(RuntimeError, match="ldd failed"):
        audit_shared_library_dependencies(
            tmp_path,
            runner=lambda *a, **k: SimpleNamespace(returncode=1, stdout="unexpected failure", stderr=""),
        )
