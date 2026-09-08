"""Importers that turn instrument text files into nfit data containers.

Importers are intentionally decoupled from the container classes: several
instrument formats may map to the same data type and container, so each importer
is a small function that reads one format and returns a populated nfit data
container. The :data:`IMPORTERS` registry lets the GUI probe file contents,
offer more than one importer per data type, and expand multi-stream acquisitions.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from .dataset import PointData4D, PointListData
from .macs import import_macs_nexus, is_macs_nexus_file, macs_nexus_point_count
from .mdhisto import MDHistoAxis, MDHistoData
from .quantities import normalize_unit

ImportResult: TypeAlias = PointData4D | PointListData | MDHistoData
ImporterLoader: TypeAlias = Callable[..., ImportResult]
ImporterProbe: TypeAlias = Callable[[str | Path], bool]
ImporterPointCounter: TypeAlias = Callable[[str | Path, dict[str, Any] | None], int]

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
    # Select the candidate that produces the most fields.  Counting delimiter
    # characters directly is not sufficient because a quoted CSV label may
    # itself contain a comma.
    candidates: list[tuple[int, str]] = []
    for candidate in (",", "\t"):
        try:
            cells = next(
                csv.reader(
                    [line],
                    delimiter=candidate,
                    skipinitialspace=True,
                    strict=True,
                )
            )
        except csv.Error:
            continue
        candidates.append((len(cells), candidate))
    if candidates:
        count, candidate = max(candidates, key=lambda item: item[0])
        if count > 1:
            return candidate
    return "whitespace"


def _split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.split()
    if delimiter not in {",", "\t"}:
        raise ValueError("delimiter must be ',', '\\t', or 'whitespace'")
    try:
        cells = next(
            csv.reader(
                [line],
                delimiter=delimiter,
                skipinitialspace=True,
                strict=True,
            )
        )
    except csv.Error as exc:
        raise ValueError(f"invalid delimited row: {exc}") from exc
    return [cell.strip() for cell in cells]


_MISSING_NUMERIC_VALUES = frozenset({"", "na", "n/a", "nan", "null", "none", "--", "---"})


def _is_comment(line: str, comment_prefixes: tuple[str, ...]) -> bool:
    stripped = line.strip()
    return bool(comment_prefixes) and stripped.startswith(comment_prefixes)


def _row_is_numeric(cells: list[str]) -> bool:
    """Return whether a non-empty row contains numbers or missing sentinels."""

    if not cells or not any(cell.strip() for cell in cells):
        return False
    for cell in cells:
        text = cell.strip()
        if text.lower() in _MISSING_NUMERIC_VALUES:
            continue
        try:
            float(text)
        except ValueError:
            return False
    return True


def _parse_numeric_cell(text: str, *, path: Path, line_number: int, column: str) -> float:
    stripped = text.strip()
    if stripped.lower() in _MISSING_NUMERIC_VALUES:
        return float("nan")
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(
            f"invalid numeric value {text!r} in {path} at line {line_number}, "
            f"column {column!r}"
        ) from exc


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
    if delimiter not in {None, ",", "\t", "whitespace"}:
        raise ValueError("delimiter must be ',', '\\t', or 'whitespace'")
    raw_lines = file_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    header_lines: list[str] = []
    index = 0
    if data_marker is not None:
        found_marker = False
        while index < len(raw_lines):
            line = raw_lines[index]
            index += 1
            if line.strip() == data_marker:
                found_marker = True
                break
            header_lines.append(line)
        if not found_marker:
            raise ValueError(f"data marker {data_marker!r} not found in {file_path}")
    else:
        # Locate the first numeric row.  For a headerless table it begins the
        # data; for a headed table the preceding substantive line is the column
        # header.  This permits arbitrary instrument preambles without treating
        # their first line as either labels or data.
        substantive: list[int] = []
        first_numeric_position: int | None = None
        for line_number, line in enumerate(raw_lines):
            if not line.strip() or _is_comment(line, comment_prefixes):
                continue
            substantive.append(line_number)
            probe_delimiter = delimiter or _detect_delimiter(line)
            try:
                cells = _split_row(line, probe_delimiter)
            except ValueError:
                continue
            if _row_is_numeric(cells):
                first_numeric_position = len(substantive) - 1
                break
        if not substantive:
            raise ValueError(f"no tabular data found in {file_path}")
        if first_numeric_position is None:
            if not has_header_row:
                raise ValueError(f"no numeric tabular data found in {file_path}")
            first_numeric_position = len(substantive) - 1
        if has_header_row:
            if first_numeric_position == 0:
                raise ValueError(f"no column-header row found in {file_path}")
            header_position = first_numeric_position - 1
            # If an early data row contains malformed text, it will not have
            # served as the numeric inference anchor.  Walk back to the first
            # plausible label row whose following rows retain the same shape;
            # parsing can then report the malformed cell precisely.
            for candidate_position in range(first_numeric_position):
                candidate_index = substantive[candidate_position]
                candidate_delimiter = delimiter or _detect_delimiter(
                    raw_lines[candidate_index]
                )
                try:
                    candidate_cells = _split_row(
                        raw_lines[candidate_index], candidate_delimiter
                    )
                    following_cells = [
                        _split_row(raw_lines[row_index], candidate_delimiter)
                        for row_index in substantive[
                            candidate_position + 1 : first_numeric_position + 1
                        ]
                    ]
                except ValueError:
                    continue
                if (
                    len(candidate_cells) >= 2
                    and not _row_is_numeric(candidate_cells)
                    and following_cells
                    and all(
                        len(cells) == len(candidate_cells) for cells in following_cells
                    )
                ):
                    header_position = candidate_position
                    break
            index = substantive[header_position]
        else:
            index = substantive[first_numeric_position]
        header_lines.extend(raw_lines[:index])

    # Find the first substantive table line to establish the delimiter.
    while index < len(raw_lines) and (
        not raw_lines[index].strip() or _is_comment(raw_lines[index], comment_prefixes)
    ):
        header_lines.append(raw_lines[index])
        index += 1
    if index >= len(raw_lines):
        raise ValueError(f"no tabular data found in {file_path}")

    resolved_delimiter = delimiter or _detect_delimiter(raw_lines[index])

    if has_header_row:
        try:
            labels = _split_row(raw_lines[index], resolved_delimiter)
        except ValueError as exc:
            raise ValueError(
                f"invalid column-header row in {file_path} at line {index + 1}: {exc}"
            ) from exc
        index += 1
    elif column_names is not None:
        labels = list(column_names)
    else:
        raise ValueError("column_names is required when has_header_row is False")
    if not labels:
        raise ValueError(f"no column labels found in {file_path}")

    names: list[str] = []
    units: dict[str, str] = {}
    for position, label in enumerate(labels, start=1):
        name, unit = split_name_and_unit(label)
        if not name:
            raise ValueError(
                f"empty column label in {file_path} at position {position}"
            )
        if name in names:
            raise ValueError(f"duplicate column label {name!r} in {file_path}")
        names.append(name)
        if unit:
            units[name] = unit

    columns: dict[str, list[float]] = {name: [] for name in names}
    data_row_count = 0
    for line_number, line in enumerate(raw_lines[index:], start=index + 1):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_comment(line, comment_prefixes):
            header_lines.append(line)
            continue
        try:
            cells = _split_row(line, resolved_delimiter)
        except ValueError as exc:
            raise ValueError(f"{exc} in {file_path} at line {line_number}") from exc
        if len(cells) != len(names):
            raise ValueError(
                f"inconsistent row width in {file_path} at line {line_number}: "
                f"found {len(cells)} columns, expected {len(names)}"
            )
        for position, name in enumerate(names):
            columns[name].append(
                _parse_numeric_cell(
                    cells[position],
                    path=file_path,
                    line_number=line_number,
                    column=name,
                )
            )
        data_row_count += 1

    if data_row_count == 0:
        raise ValueError(f"no data rows found in {file_path}")

    return DelimitedText(columns=columns, units=units, header_lines=header_lines, column_labels=names)


def _all_numeric(cells: list[str]) -> bool:
    """Compatibility wrapper for the historical private helper."""

    return _row_is_numeric(cells)


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
    metadata: dict[str, Any] = {
        "instrument_header": list(header_lines),
        "mpms_info": info,
    }
    numeric_fields = {
        "SAMPLE_MASS": ("sample_mass_mg", "mg"),
        "SAMPLE_MOLECULAR_WEIGHT": ("molar_mass_g_mol", "g/mol"),
        # Quantum Design exports this as a sample property without embedding a
        # unit in the INFO row. Preserve it, but do not use it in conversions.
        "SAMPLE_VOLUME": ("sample_volume_mpms", "MPMS sample-volume unit"),
    }
    for source_name, (target_name, unit) in numeric_fields.items():
        raw = info.get(source_name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        metadata[target_name] = value
        metadata[f"{target_name}_unit"] = unit
    return metadata


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
        quantity_types={
            **{name: "temperature" for name in columns if "temperature" in name.lower()},
            **{name: "magnetic_field" for name in columns if "field" in name.lower()},
            **{name: "magnetic_moment" for name in columns if "moment" in name.lower()},
        },
    )


def _ppms_heat_capacity_header_metadata(header_lines: list[str]) -> dict[str, Any]:
    """Extract sample normalization from a Quantum Design PPMS HC header."""

    metadata: dict[str, Any] = {"instrument_header": list(header_lines)}
    info: dict[str, str] = {}
    for line in header_lines:
        parts = [cell.strip() for cell in line.split(",")]
        if len(parts) < 3 or parts[0] != "INFO":
            continue
        value = parts[1]
        key = parts[2].split(":", 1)[0].strip()
        if key:
            info[key] = value
    metadata["ppms_info"] = info
    for key, target, unit in (
        ("MASS", "sample_mass_mg", "mg"),
        ("MOLWGHT", "molar_mass_g_mol", "g/mol"),
        ("ATOMS", "atoms_per_formula_unit", "1"),
    ):
        try:
            value = float(info[key])
        except (KeyError, TypeError, ValueError):
            continue
        metadata[target] = value
        metadata[f"{target}_unit"] = unit
    return metadata


def import_ppms_heat_capacity_dat(path: str | Path) -> PointListData:
    """Import a Quantum Design PPMS Heat Capacity ``.dat`` file.

    Every exported column is retained.  The canonical coordinate and raw
    channel use ``Sample Temp`` and ``Samp HC``; molar normalization is applied
    later by the dataset transform so edited mass metadata takes effect.
    """

    parsed = read_delimited_text(path, data_marker="[Data]", has_header_row=True, delimiter=",")
    columns = {name: values for name, values in parsed.columns.items()}
    metadata = _ppms_heat_capacity_header_metadata(parsed.header_lines)
    metadata["source_file"] = str(Path(path))
    temperature = next(
        (name for name in ("Sample Temp", "Temperature", "System Temp") if name in columns),
        next((name for name in columns if "temp" in name.lower()), next(iter(columns))),
    )
    value = next(
        (name for name in ("Samp HC", "Sample HC") if name in columns),
        next((name for name in columns if "hc" in name.lower()), next(iter(columns))),
    )
    error = next(
        (name for name in ("Samp HC Err", "Sample HC Err") if name in columns),
        None,
    )
    units = dict(parsed.units)
    units[temperature] = "K"
    source_unit = normalize_unit(parsed.units.get(value, ""))
    units[value] = source_unit or "uJ/K"
    if error is not None:
        units[error] = units[value]
    return PointListData(
        columns=columns,
        units=units,
        coordinate_names=[temperature],
        channels=[{"label": "Sample heat capacity", "value": value, "error": error}],
        metadata=metadata,
        quantity_types={
            **{name: "temperature" for name in columns if "temp" in name.lower()},
            value: "heat_capacity",
            **({error: "heat_capacity"} if error is not None else {}),
        },
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
    units = {"2theta": "°"}
    metadata = {"source_file": str(Path(path)), "instrument_header": list(parsed.header_lines)}
    return PointListData(
        columns=columns,
        units=units,
        coordinate_names=["2theta"],
        channels=[{"label": "I", "value": "I", "error": "dI"}],
        metadata=metadata,
        quantity_types={"2theta": "unknown", "I": "scattering_intensity", "dI": "scattering_intensity"},
    )


def _read_probe_lines(path: str | Path, *, limit: int = 128) -> list[str]:
    """Read a bounded text prefix for inexpensive importer detection."""

    lines: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for _line_number, line in zip(range(limit), handle, strict=False):
            lines.append(line.rstrip("\r\n"))
    return lines


def _labels_after_marker(path: str | Path, marker: str) -> list[str]:
    lines = _read_probe_lines(path)
    try:
        marker_index = next(
            index for index, line in enumerate(lines) if line.strip() == marker
        )
    except StopIteration:
        return []
    for line in lines[marker_index + 1 :]:
        if not line.strip() or line.lstrip().startswith((";", "#")):
            continue
        return [split_name_and_unit(cell)[0] for cell in _split_row(line, ",")]
    return []


def is_mpms_dat_file(path: str | Path) -> bool:
    """Return whether *path* has the characteristic columns of an MPMS export."""

    try:
        labels = {label.casefold() for label in _labels_after_marker(path, "[Data]")}
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        "temperature" in labels
        and "magnetic field" in labels
        and any("moment" in label for label in labels)
    )


def is_ppms_heat_capacity_dat_file(path: str | Path) -> bool:
    """Return whether *path* has PPMS Heat Capacity data columns."""

    try:
        labels = {label.casefold() for label in _labels_after_marker(path, "[Data]")}
    except (OSError, UnicodeError, ValueError):
        return False
    has_temperature = any("temp" in label for label in labels)
    has_heat_capacity = any(
        label in {"samp hc", "sample hc"} or "heat capacity" in label
        for label in labels
    )
    return has_temperature and has_heat_capacity


def is_hb2a_powder_file(path: str | Path) -> bool:
    """Return whether *path* begins with an HB2A-style numeric three-column table."""

    numeric_rows = 0
    data_started = False
    try:
        for line in _read_probe_lines(path):
            stripped = line.strip()
            if not stripped or stripped.startswith((";", "#")):
                continue
            cells = _split_row(line, "whitespace")
            if len(cells) == 3 and _row_is_numeric(cells):
                data_started = True
                numeric_rows += 1
                if numeric_rows >= 2:
                    return True
                continue
            if data_started:
                return False
    except (OSError, UnicodeError, ValueError):
        return False
    return numeric_rows == 1


def _read_csv_rows(path: str | Path) -> list[tuple[int, list[str]]]:
    file_path = Path(path)
    rows: list[tuple[int, list[str]]] = []
    try:
        with file_path.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            reader = csv.reader(handle, strict=True)
            for row in reader:
                if any(cell.strip() for cell in row):
                    rows.append((reader.line_num, [cell.strip() for cell in row]))
    except csv.Error as exc:
        raise ValueError(f"invalid CSV in {file_path}: {exc}") from exc
    return rows


def inspect_powder_ins_csv(path: str | Path) -> dict[str, Any]:
    """Inspect a plot-digitizer CSV without assigning physical conditions.

    Matrix exports begin with ``y\\x`` and contain Q values across the first
    row and energy values down the first column. Headerless three-column files
    are interpreted as a one-dimensional ``x, signal, uncertainty`` cut whose
    x coordinate must be selected during import.
    """

    file_path = Path(path)
    numbered_rows = _read_csv_rows(file_path)
    if not numbered_rows:
        raise ValueError(f"no CSV data found in {file_path}")
    rows = [row for _line_number, row in numbered_rows]
    first = rows[0][0].strip().lower() if rows[0] else ""
    if first in {"y\\x", "y/x", "y"}:
        if len(rows[0]) < 2 or len(rows) < 2:
            raise ValueError("powder INS matrix CSV requires Q columns and energy rows")
        for position, cell in enumerate(rows[0][1:], start=2):
            _float_cell(cell, context=f"matrix Q header column {position}")
        expected_width = len(rows[0])
        for line_number, row in numbered_rows[1:]:
            if len(row) != expected_width:
                raise ValueError(
                    f"matrix row {line_number} has {len(row) - 1} values; "
                    f"expected {expected_width - 1}"
                )
            _float_cell(row[0], context=f"matrix row {line_number} energy")
            for position, cell in enumerate(row[1:], start=2):
                _float_cell(
                    cell,
                    context=f"matrix row {line_number}, column {position}",
                    allow_missing=True,
                )
        return {
            "layout": "matrix_q_energy",
            "point_count": (len(rows) - 1) * (len(rows[0]) - 1),
            "q_count": len(rows[0]) - 1,
            "energy_count": len(rows) - 1,
        }
    column_count = len(rows[0])
    if column_count not in {2, 3}:
        raise ValueError("powder INS cut CSV requires two or three columns")
    for line_number, row in numbered_rows:
        if len(row) != column_count:
            raise ValueError(
                f"cut row {line_number} has {len(row)} columns; expected {column_count}"
            )
        _float_cell(row[0], context=f"cut row {line_number} x")
        _float_cell(
            row[1], context=f"cut row {line_number} signal", allow_missing=True
        )
        if column_count == 3:
            _float_cell(
                row[2], context=f"cut row {line_number} uncertainty", allow_missing=True
            )
    return {
        "layout": "cut",
        "point_count": len(rows),
        "column_count": column_count,
    }


def is_powder_ins_csv_file(path: str | Path) -> bool:
    """Return whether *path* is a valid supported powder-INS CSV layout."""

    try:
        inspect_powder_ins_csv(path)
    except (OSError, UnicodeError, ValueError):
        return False
    return True


def _float_cell(text: str, *, context: str, allow_missing: bool = False) -> float:
    stripped = text.strip()
    if allow_missing and stripped.lower() in _MISSING_NUMERIC_VALUES:
        return float("nan")
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value {text!r} in {context}") from exc


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    """Infer histogram boundaries from ordered digitized bin centers."""

    centers = np.asarray(values, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("a digitized axis requires at least one center")
    if centers.size == 1:
        half_width = max(abs(float(centers[0])) * 1.0e-6, 1.0e-6)
        return np.asarray([centers[0] - half_width, centers[0] + half_width])
    interior = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        (
            [centers[0] - 0.5 * (centers[1] - centers[0])],
            interior,
            [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
        )
    )


def _powder_ins_matrix(path: Path, default_uncertainty: float) -> MDHistoData:
    rows = [row for _line_number, row in _read_csv_rows(path)]
    q = np.asarray(
        [_float_cell(cell, context="matrix Q header") for cell in rows[0][1:]],
        dtype=float,
    )
    energy_values: list[float] = []
    signal_rows: list[list[float]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != q.size + 1:
            raise ValueError(
                f"matrix row {row_number} has {len(row) - 1} values; expected {q.size}"
            )
        energy_values.append(_float_cell(row[0], context=f"matrix row {row_number}"))
        signal_rows.append(
            [
                _float_cell(
                    cell,
                    context=f"matrix row {row_number}",
                    allow_missing=True,
                )
                for cell in row[1:]
            ]
        )
    energy = np.asarray(energy_values, dtype=float)
    # CSV rows are E and columns are Q; nfit's canonical powder order is Q, E.
    signal = np.asarray(signal_rows, dtype=float).T
    q_order = np.argsort(q)
    energy_order = np.argsort(energy)
    q = q[q_order]
    energy = energy[energy_order]
    signal = signal[np.ix_(q_order, energy_order)]
    measured = np.isfinite(signal)
    errors = np.full(signal.shape, float(default_uncertainty), dtype=float)
    errors[~measured] = np.nan
    return MDHistoData(
        axes=(
            MDHistoAxis("Q", _centers_to_edges(q), "1/angstrom", "momentum"),
            MDHistoAxis(
                "DeltaE", _centers_to_edges(energy), "meV", "energy"
            ),
        ),
        signal=signal,
        errors=errors,
        mask=~measured,
        num_events=measured.astype(float),
        metadata={
            "signal_semantics": "density",
            "signal_semantics_source": "digitized_powder_ins_map",
            "zero_event_bins_are_measured": True,
        },
    )


def _powder_ins_cut(
    path: Path,
    *,
    cut_type: str,
    fixed_value: float,
) -> MDHistoData:
    numbered_rows = _read_csv_rows(path)
    values = np.asarray(
        [
            [
                _float_cell(
                    cell,
                    context=f"cut row {line_number}",
                    allow_missing=position > 0,
                )
                for position, cell in enumerate(row[:3])
            ]
            for line_number, row in numbered_rows
        ],
        dtype=float,
    )
    if values.shape[1] < 2:
        raise ValueError("powder INS cut requires x and signal columns")
    x = values[:, 0]
    signal_1d = values[:, 1]
    errors_1d = values[:, 2] if values.shape[1] >= 3 else np.ones_like(signal_1d)
    order = np.argsort(x)
    x = x[order]
    signal_1d = signal_1d[order]
    errors_1d = errors_1d[order]
    measured = np.isfinite(x) & np.isfinite(signal_1d) & np.isfinite(errors_1d)
    if cut_type == "constant_energy":
        q = x
        energy = np.asarray([fixed_value], dtype=float)
        signal = signal_1d[:, None]
        errors = errors_1d[:, None]
        mask = (~measured)[:, None]
    elif cut_type == "constant_q":
        q = np.asarray([fixed_value], dtype=float)
        energy = x
        signal = signal_1d[None, :]
        errors = errors_1d[None, :]
        mask = (~measured)[None, :]
    else:
        raise ValueError("cut_type must be 'constant_energy' or 'constant_q'")
    return MDHistoData(
        axes=(
            MDHistoAxis("Q", _centers_to_edges(q), "1/angstrom", "momentum"),
            MDHistoAxis(
                "DeltaE", _centers_to_edges(energy), "meV", "energy"
            ),
        ),
        signal=signal,
        errors=errors,
        mask=mask,
        num_events=(~mask).astype(float),
        metadata={
            "signal_semantics": "density",
            "signal_semantics_source": "digitized_powder_ins_cut",
            "zero_event_bins_are_measured": True,
        },
    )


def import_powder_ins_csv(
    path: str | Path,
    options: dict[str, Any] | None = None,
) -> MDHistoData:
    """Import a digitized powder INS cut or Q-E matrix CSV.

    Parameters in ``options`` are deliberately explicit because plot digitizer
    files do not carry their physical context. Required for a cut are
    ``cut_type`` (``constant_energy`` or ``constant_q``) and ``fixed_value``.
    Temperature, observable, units, normalization basis, and kinematic state
    are retained as editable dataset parameters.
    """

    source = Path(path)
    settings = {
        "layout": "auto",
        "cut_type": "",
        "fixed_value": None,
        "temperature_K": None,
        "source_representation": "cross_section",
        "source_unit": "arbitrary",
        "fit_representation": "cross_section",
        "normalization_basis": "unknown",
        "normalization_label": "",
        "kf_ki_state": "removed",
        "default_uncertainty": 1.0,
    }
    if isinstance(options, dict):
        settings.update(options)
    detected = inspect_powder_ins_csv(source)
    layout = detected["layout"] if settings["layout"] == "auto" else str(settings["layout"])
    if layout == "matrix_q_energy":
        default_uncertainty = float(settings["default_uncertainty"])
        if not np.isfinite(default_uncertainty) or default_uncertainty <= 0.0:
            raise ValueError("map uncertainty must be finite and positive")
        data = _powder_ins_matrix(source, default_uncertainty)
    elif layout == "cut":
        fixed = settings.get("fixed_value")
        if fixed in (None, ""):
            raise ValueError("a constant-Q or constant-E cut requires its fixed value")
        data = _powder_ins_cut(
            source,
            cut_type=str(settings.get("cut_type", "")),
            fixed_value=float(fixed),
        )
    else:
        raise ValueError(f"unknown powder INS CSV layout {layout!r}")

    from .spectral_channels import default_spectral_channel_config

    spectral = default_spectral_channel_config()
    for key in (
        "source_representation",
        "source_unit",
        "fit_representation",
        "normalization_basis",
        "normalization_label",
        "kf_ki_state",
    ):
        spectral[key] = settings[key]
    temperature = settings.get("temperature_K")
    parameters: dict[str, Any] = {"spectral_channels": spectral}
    if temperature not in (None, ""):
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        parameters["temperature"] = temperature
    if layout == "cut":
        fixed = float(settings["fixed_value"])
        if settings["cut_type"] == "constant_energy":
            parameters["constant_energy_meV"] = fixed
        else:
            parameters["constant_q_inv_angstrom"] = fixed
    data.metadata.update(
        {
            "source_file": str(source),
            "importer": "powder_ins_csv",
            "powder_ins_layout": layout,
            "import_options": dict(settings),
            "dataset_parameters": parameters,
            "signal_unit": str(settings["source_unit"]),
            "signal_quantity_type": (
                "dynamic_susceptibility"
                if settings["source_representation"] == "chi_double_prime"
                else "differential_cross_section"
                if str(settings["source_unit"]).startswith(("mbarn/", "barn/"))
                else "scattering_intensity"
            ),
        }
    )
    return data


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
class ImporterStream:
    """One named scientific stream emitted by a multi-stream source file."""

    name: str
    label: str
    data_type: str


@dataclass(frozen=True)
class ImporterMatch:
    """Evidence that a registered importer recognizes a source file."""

    importer_name: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class ImporterSpec:
    """One registered importer for a family of data types."""

    name: str
    label: str
    loader: ImporterLoader
    data_types: tuple[str, ...]
    extensions: tuple[str, ...] = ()
    options_kind: str | None = None
    probe: ImporterProbe | None = None
    streams: tuple[ImporterStream, ...] = ()
    point_counter: ImporterPointCounter | None = None

    def can_read(self, path: str | Path) -> bool:
        if self.probe is not None:
            try:
                return bool(self.probe(path))
            except (OSError, UnicodeError, ValueError, csv.Error):
                return False
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
        probe=is_mpms_dat_file,
    ),
    "hb2a_powder": ImporterSpec(
        name="hb2a_powder",
        label="HB2A powder (2theta, I, dI)",
        loader=import_hb2a_powder,
        data_types=("powder_elastic",),
        extensions=(".dat", ".txt", ""),
        probe=is_hb2a_powder_file,
    ),
    "ppms_heat_capacity_dat": ImporterSpec(
        name="ppms_heat_capacity_dat",
        label="Quantum Design PPMS Heat Capacity (.dat)",
        loader=import_ppms_heat_capacity_dat,
        data_types=("heat_capacity",),
        extensions=(".dat",),
        probe=is_ppms_heat_capacity_dat_file,
    ),
    "powder_ins_csv": ImporterSpec(
        name="powder_ins_csv",
        label="Powder INS digitizer CSV",
        loader=import_powder_ins_csv,
        data_types=("powder_inelastic",),
        extensions=(".csv",),
        options_kind="powder_ins_csv",
        probe=is_powder_ins_csv_file,
    ),
    "macs_nexus": ImporterSpec(
        name="macs_nexus",
        label="NIST NCNR MACS NeXus (SPEC + DIFF)",
        loader=import_macs_nexus,
        data_types=("single_crystal_inelastic", "single_crystal_energy_integrated"),
        extensions=(".nxs", ".ng0"),
        options_kind="macs_nexus",
        probe=is_macs_nexus_file,
        point_counter=macs_nexus_point_count,
        streams=(
            ImporterStream("spec", "SPEC", "single_crystal_inelastic"),
            ImporterStream("diff", "DIFF", "single_crystal_energy_integrated"),
        ),
    ),
}


def importers_for_data_type(data_type: str) -> list[ImporterSpec]:
    """Return registered importers that produce the given data type."""

    return [spec for spec in IMPORTERS.values() if data_type in spec.data_types]


def probe_importers(
    path: str | Path,
    data_type: str | None = None,
) -> list[ImporterMatch]:
    """Return matching importers ranked by content evidence then extension.

    Instrument-specific probes inspect internal metadata and receive high
    confidence. Extension-only adapters remain available as a compatibility
    fallback, but do not outrank a positive content probe.
    """

    source = Path(path)
    matches: list[ImporterMatch] = []
    for spec in IMPORTERS.values():
        if data_type is not None and data_type not in spec.data_types:
            continue
        if spec.probe is not None:
            if spec.can_read(source):
                matches.append(
                    ImporterMatch(spec.name, 1.0, "recognized source-file contents")
                )
            continue
        if spec.can_read(source):
            reason = "compatible filename extension" if spec.extensions else "generic importer"
            matches.append(ImporterMatch(spec.name, 0.25, reason))
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def import_with(
    importer_name: str,
    path: str | Path,
    options: dict[str, Any] | None = None,
) -> PointData4D | PointListData | MDHistoData:
    """Run a registered importer by name."""

    if importer_name not in IMPORTERS:
        raise KeyError(f"unknown importer {importer_name!r}")
    spec = IMPORTERS[importer_name]
    if spec.options_kind is not None:
        return spec.loader(path, options)
    return spec.loader(path)
