from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mpms_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a compact MPMS export shared by importer and GUI tests."""

    path = tmp_path_factory.mktemp("instrument_data") / "mpms.dat"
    rows = [
        (
            299.499588 - 5.0 * index,
            1000.0 + 100.0 * index,
            0.010 + 0.001 * index,
            0.0001,
            0.002 + 0.0001 * index,
            2.0e-6 + 1.0e-7 * index,
        )
        for index in range(12)
    ]
    header = (
        "[Header]\n"
        "INFO,EXT_Rodriguez synthetic,SAMPLE_MATERIAL\n"
        "INFO,9.73,SAMPLE_MASS\n"
        "INFO,172.8,SAMPLE_MOLECULAR_WEIGHT\n"
        "INFO,2.16,SAMPLE_VOLUME\n"
        "[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Moment (emu),"
        "M. Std. Err. (emu),AC Moment (emu),AC Susceptibility (emu/Oe)\n"
    )
    body = "\n".join(",".join(f"{value:.9g}" for value in row) for row in rows)
    path.write_text(header + body + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def hb2a_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a compact three-column HB2A-style powder dataset."""

    path = tmp_path_factory.mktemp("instrument_data") / "hb2a.dat"
    rows = (
        f"{2.15 + 0.05 * index:.6f} "
        f"{100.0 + index + 20.0 * (index % 7 == 0):.6f} "
        f"{(100.0 + index) ** 0.5:.6f}"
        for index in range(120)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
