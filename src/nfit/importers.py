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
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .macs import import_macs_nexus, is_macs_nexus_file
from .mdhisto import MDHistoAxis, MDHistoData
from .quantities import normalize_unit

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


def inspect_powder_ins_csv(path: str | Path) -> dict[str, Any]:
    """Inspect a plot-digitizer CSV without assigning physical conditions.

    Matrix exports begin with ``y\\x`` and contain Q values across the first
    row and energy values down the first column. Headerless three-column files
    are interpreted as a one-dimensional ``x, signal, uncertainty`` cut whose
    x coordinate must be selected during import.
    """

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"no CSV data found in {file_path}")
    first = rows[0][0].strip().lower() if rows[0] else ""
    if first in {"y\\x", "y/x", "y"}:
        if len(rows[0]) < 2 or len(rows) < 2:
            raise ValueError("powder INS matrix CSV requires Q columns and energy rows")
        return {
            "layout": "matrix_q_energy",
            "point_count": (len(rows) - 1) * (len(rows[0]) - 1),
            "q_count": len(rows[0]) - 1,
            "energy_count": len(rows) - 1,
        }
    if len(rows[0]) < 2:
        raise ValueError("powder INS cut CSV requires at least x and signal columns")
    return {
        "layout": "cut",
        "point_count": len(rows),
        "column_count": len(rows[0]),
    }


def _float_cell(text: str, *, context: str) -> float:
    try:
        return float(text.strip())
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
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
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
        signal_rows.append([_to_float(cell) for cell in row[1:]])
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
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    values = np.asarray(
        [
            [_float_cell(cell, context=f"cut row {row_number}") for cell in row[:3]]
            for row_number, row in enumerate(rows, start=1)
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
    loader: Callable[..., PointData4D | PointListData | MDHistoData]
    data_types: tuple[str, ...]
    extensions: tuple[str, ...] = ()
    options_kind: str | None = None
    probe: Callable[[str | Path], bool] | None = None
    streams: tuple[ImporterStream, ...] = ()

    def can_read(self, path: str | Path) -> bool:
        if self.probe is not None:
            return bool(self.probe(path))
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
    "ppms_heat_capacity_dat": ImporterSpec(
        name="ppms_heat_capacity_dat",
        label="Quantum Design PPMS Heat Capacity (.dat)",
        loader=import_ppms_heat_capacity_dat,
        data_types=("heat_capacity",),
        extensions=(".dat",),
    ),
    "powder_ins_csv": ImporterSpec(
        name="powder_ins_csv",
        label="Powder INS digitizer CSV",
        loader=import_powder_ins_csv,
        data_types=("powder_inelastic",),
        extensions=(".csv",),
        options_kind="powder_ins_csv",
    ),
    "macs_nexus": ImporterSpec(
        name="macs_nexus",
        label="NIST NCNR MACS NeXus (SPEC + DIFF)",
        loader=import_macs_nexus,
        data_types=("single_crystal_inelastic", "single_crystal_energy_integrated"),
        extensions=(".nxs", ".ng0"),
        options_kind="macs_nexus",
        probe=is_macs_nexus_file,
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
                    ImporterMatch(spec.name, 1.0, "recognized internal instrument metadata")
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
