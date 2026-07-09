"""Importers that turn instrument text files into metallix data containers.

Importers are intentionally decoupled from the container classes: several
instrument formats may map to the same data type and container, so each importer
is a small function that reads one format and returns a populated
:class:`~metallix.dataset.PointListData`. The :data:`IMPORTERS` registry lets the
GUI offer more than one importer per data type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .dataset import PointListData


# ``Name (unit)`` -> ("Name", "unit"). The unit is the last parenthesized group.
_UNIT_PATTERN = re.compile(r"^(?P<name>.*?)\s*\((?P<unit>[^()]*)\)\s*$")


def split_name_and_unit(label: str) -> tuple[str, str]:
    """Split a column label into a name and unit using trailing parentheses.

    ``"Temperature (K)"`` -> ``("Temperature", "K")``; a label without a
    parenthesized unit returns an empty unit string.
    """

    match = _UNIT_PATTERN.match(label.strip())
    if match is None:
        return label.strip(), ""
    return match.group("name").strip(), match.group("unit").strip()


def _detect_delimiter(line: str) -> str:
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return "whitespace"


def _split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.split()
    return [cell.strip() for cell in line.split(delimiter)]


@dataclass
class DelimitedText:
    """Parsed contents of a delimited text file."""

    columns: dict[str, list[float]]
    units: dict[str, str]
    header_lines: list[str]
    column_labels: list[str]


def read_delimited_text(
    path: str | Path,
    *,
    data_marker: str | None = None,
    has_header_row: bool = True,
    column_names: list[str] | None = None,
    delimiter: str | None = None,
    comment_prefixes: tuple[str, ...] = (";",),
) -> DelimitedText:
    """Read a text file of optional header lines followed by delimited values.

    Parameters
    ----------
    data_marker
        A line whose stripped text equals this marker ends the header block; the
        table starts after it. When ``None``, the header is inferred as the lines
        before the first row that parses as all-numeric (or the column-header
        row when ``has_header_row`` is set).
    has_header_row
        When true, the first line of the table names the columns (with optional
        ``(unit)`` suffixes). When false, ``column_names`` must be given.
    column_names
        Column names to use when there is no header row.
    delimiter
        ``","``, ``"\\t"``, or ``"whitespace"``. Auto-detected from the first
        data line when ``None``.
    comment_prefixes
        Line prefixes to collect as header metadata even inside the table region.
    """

    file_path = Path(path)
    raw_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_lines: list[str] = []
    index = 0
    if data_marker is not None:
        while index < len(raw_lines):
            line = raw_lines[index]
            index += 1
            if line.strip() == data_marker:
                break
            header_lines.append(line)
    else:
        # Header = leading lines that are blank, comments, or non-numeric.
        while index < len(raw_lines):
            line = raw_lines[index]
            stripped = line.strip()
            if not stripped:
                header_lines.append(line)
                index += 1
                continue
            if stripped.startswith(comment_prefixes):
                header_lines.append(line)
                index += 1
                continue
            probe_delim = delimiter or _detect_delimiter(line)
            cells = _split_row(line, probe_delim)
            if has_header_row or not _all_numeric(cells):
                break
            break

    # Find the first non-empty table line to establish the delimiter.
    while index < len(raw_lines) and not raw_lines[index].strip():
        index += 1
    if index >= len(raw_lines):
        raise ValueError(f"no tabular data found in {file_path}")

    resolved_delimiter = delimiter or _detect_delimiter(raw_lines[index])

    if has_header_row:
        labels = _split_row(raw_lines[index], resolved_delimiter)
        index += 1
    elif column_names is not None:
        labels = list(column_names)
    else:
        raise ValueError("column_names is required when has_header_row is False")

    names: list[str] = []
    units: dict[str, str] = {}
    for label in labels:
        name, unit = split_name_and_unit(label)
        names.append(name)
        if unit:
            units[name] = unit

    columns: dict[str, list[float]] = {name: [] for name in names}
    for line in raw_lines[index:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(comment_prefixes):
            header_lines.append(line)
            continue
        cells = _split_row(line, resolved_delimiter)
        for position, name in enumerate(names):
            text = cells[position] if position < len(cells) else ""
            columns[name].append(_to_float(text))

    return DelimitedText(columns=columns, units=units, header_lines=header_lines, column_labels=names)


def _all_numeric(cells: list[str]) -> bool:
    values = [cell for cell in cells if cell.strip()]
    if not values:
        return False
    for cell in values:
        try:
            float(cell)
        except ValueError:
            return False
    return True


def _to_float(text: str) -> float:
    text = text.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _mpms_header_metadata(header_lines: list[str]) -> dict[str, Any]:
    """Extract INFO ``key -> value`` pairs plus the raw header from an MPMS file."""

    info: dict[str, str] = {}
    for line in header_lines:
        parts = [cell.strip() for cell in line.split(",")]
        if len(parts) >= 3 and parts[0] == "INFO":
            # Format: INFO,<value>,<KEY>
            key = parts[-1]
            value = ",".join(parts[1:-1])
            if key:
                info[key] = value
    return {"instrument_header": list(header_lines), "mpms_info": info}


def import_mpms_dat(path: str | Path) -> PointListData:
    """Import a Quantum Design MPMS ``.dat`` magnetization file.

    The ``[Header]`` block is retained as metadata; every data column is kept.
    Default coordinates are Temperature and Magnetic Field, with the DC Moment
    and its standard error as the default channel.
    """

    parsed = read_delimited_text(path, data_marker="[Data]", has_header_row=True, delimiter=",")
    columns = {name: values for name, values in parsed.columns.items()}
    metadata = _mpms_header_metadata(parsed.header_lines)
    metadata["source_file"] = str(Path(path))

    coordinate_names = [name for name in ("Temperature", "Magnetic Field") if name in columns]
    channels = _default_channels(
        columns,
        preferred=[("Moment", "M. Std. Err."), ("DC Moment Fixed Ctr", "DC Moment Err Fixed Ctr")],
    )
    return PointListData(
        columns=columns,
        units=parsed.units,
        coordinate_names=coordinate_names or [next(iter(columns))],
        channels=channels,
        metadata=metadata,
    )


def import_hb2a_powder(path: str | Path) -> PointListData:
    """Import an HB2A-style powder diffraction file: ``2theta, I, dI`` columns.

    The file has no header row, so columns are named explicitly. ``2theta`` is
    the default coordinate; intensity and its error form the default channel.
    """

    parsed = read_delimited_text(
        path,
        has_header_row=False,
        column_names=["2theta", "I", "dI"],
        delimiter="whitespace",
    )
    columns = {name: values for name, values in parsed.columns.items()}
    units = {"2theta": "deg"}
    metadata = {"source_file": str(Path(path)), "instrument_header": list(parsed.header_lines)}
    return PointListData(
        columns=columns,
        units=units,
        coordinate_names=["2theta"],
        channels=[{"label": "I", "value": "I", "error": "dI"}],
        metadata=metadata,
    )


def _default_channels(
    columns: dict[str, Any],
    *,
    preferred: list[tuple[str, str | None]],
) -> list[dict[str, Any]]:
    for value_name, error_name in preferred:
        if value_name in columns:
            return [
                {
                    "label": value_name,
                    "value": value_name,
                    "error": error_name if error_name in columns else None,
                }
            ]
    # Fall back to the first column that is not obviously a coordinate.
    first = next(iter(columns))
    return [{"label": first, "value": first, "error": None}]


@dataclass(frozen=True)
class ImporterSpec:
    """One registered importer for a family of data types."""

    name: str
    label: str
    loader: Callable[[str | Path], PointListData]
    data_types: tuple[str, ...]
    extensions: tuple[str, ...] = ()

    def can_read(self, path: str | Path) -> bool:
        if not self.extensions:
            return True
        return Path(path).suffix.lower() in self.extensions


IMPORTERS: dict[str, ImporterSpec] = {
    "mpms_dat": ImporterSpec(
        name="mpms_dat",
        label="Quantum Design MPMS (.dat)",
        loader=import_mpms_dat,
        data_types=("magnetization",),
        extensions=(".dat",),
    ),
    "hb2a_powder": ImporterSpec(
        name="hb2a_powder",
        label="HB2A powder (2theta, I, dI)",
        loader=import_hb2a_powder,
        data_types=("powder_elastic",),
        extensions=(".dat", ".txt", ""),
    ),
}


def importers_for_data_type(data_type: str) -> list[ImporterSpec]:
    """Return registered importers that produce the given data type."""

    return [spec for spec in IMPORTERS.values() if data_type in spec.data_types]


def import_with(importer_name: str, path: str | Path) -> PointListData:
    """Run a registered importer by name."""

    if importer_name not in IMPORTERS:
        raise KeyError(f"unknown importer {importer_name!r}")
    return IMPORTERS[importer_name].loader(path)
