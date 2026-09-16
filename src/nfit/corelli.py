"""Finite-energy reconstruction for raw CORELLI correlation-chopper events.

CORELLI records a white incident spectrum together with the absolute phase of
its pseudorandom correlation chopper.  A detected neutron therefore does not
have one measured incident energy.  Cross correlation evaluates the chopper
transmission at each requested energy-transfer channel and gives the neutron a
signed weight in every kinematically allowed channel.

The implementation streams raw event banks and samples the reconstruction at
the centres of the requested DeltaE bins.  It follows Mantid's
``CorelliCrossCorrelate`` timing convention, including its open/closed weights,
and extends the elastic incident-time calculation to finite DeltaE.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from . import _parallel

try:
    from . import _corelli_numba as _CORELLI_NUMBA
except Exception:  # Numba remains optional for portable source installations.
    _CORELLI_NUMBA = None

from .mdevent import (
    ENERGY_TO_K2,
    FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER,
    _requested_edges,
    _symmetry_matrices,
    _validated_minimum_samples,
    load_detector_normalization,
)
from .mdhisto import MDHistoAxis, MDHistoData
from .pipeline import DatasetEntry, DatasetGroup
from .raw_dgs import (
    _detector_geometry,
    _DetectorGeometry,
    _good_pulses,
    _he3_tube_efficiency_correction,
    _pulse_charge_values,
    _raw_proton_charge_uah,
    _retained_proton_charge_uah,
    _source_distance,
    _text_scalar,
    _trajectory_worker_count,
    _ub_from_logs,
    _use_ki_kf_correction,
)

ENERGY_FROM_WAVELENGTH_MEV_ANGSTROM_SQ = 81.80421036
CORELLI_TOF_US_PER_M_SQRT_MEV = 2286.271549
DEFAULT_WAVELENGTH_RANGE_ANGSTROM = (0.6, 2.5)
DEFAULT_TIMING_OFFSET_NS = 14_000
CORELLI_NUMBA_MIN_HYPOTHESES = 10_000


@dataclass(frozen=True)
class CorelliRunInfo:
    """Lightweight metadata required for one raw CORELLI run."""

    path: Path
    run_number: str
    event_count: int
    proton_charge: float
    goniometer_angles: tuple[float, float, float]
    l1: float
    source_to_chopper: float
    ub_matrix: np.ndarray


@dataclass(frozen=True)
class _ChopperTiming:
    tdc_ns: np.ndarray
    period_ns: float
    sequence_edges_deg: np.ndarray
    duty_cycle: float


@dataclass
class _CorelliFlux:
    momentum: np.ndarray
    cumulative: np.ndarray
    detector_ids: np.ndarray
    spectrum_rows: np.ndarray

    def row_for_ids(self, detector_ids):
        ids = np.asarray(detector_ids, dtype=np.int64)
        locations = np.searchsorted(self.detector_ids, ids)
        valid = locations < self.detector_ids.size
        valid[valid] &= self.detector_ids[locations[valid]] == ids[valid]
        result = np.full(ids.shape, -1, dtype=np.int64)
        result[valid] = self.spectrum_rows[locations[valid]]
        return result

    def density_for_ids(self, detector_ids, momentum):
        ids = np.asarray(detector_ids, dtype=np.int64)
        wavevector = np.asarray(momentum, dtype=float)
        locations = np.searchsorted(self.detector_ids, ids)
        valid = locations < self.detector_ids.size
        valid[valid] &= self.detector_ids[locations[valid]] == ids[valid]
        result = np.zeros(ids.shape, dtype=float)
        if not np.any(valid):
            return result
        valid &= (wavevector >= self.momentum[0]) & (
            wavevector <= self.momentum[-1]
        )
        if not np.any(valid):
            return result
        rows = self.spectrum_rows[locations[valid]]
        momentum_valid = wavevector[valid]
        right = np.searchsorted(self.momentum, momentum_valid, side="right")
        right = np.clip(right, 1, self.momentum.size - 1)
        left = right - 1
        far_right = np.minimum(self.momentum.size - 1, right + 1)
        spacing = self.momentum[far_right] - self.momentum[left]
        result[valid] = np.maximum(
            0.0,
            (
                self.cumulative[rows, far_right]
                - self.cumulative[rows, left]
            )
            / spacing,
        )
        return result


@dataclass(frozen=True)
class _CorelliDetectorMask:
    masked_ids: np.ndarray

    def value_for_ids(self, detector_ids):
        ids = np.asarray(detector_ids, dtype=np.int64)
        locations = np.searchsorted(self.masked_ids, ids)
        masked = locations < self.masked_ids.size
        masked[masked] &= self.masked_ids[locations[masked]] == ids[masked]
        return (~masked).astype(float)



def _is_corelli_entry(entry) -> bool:
    if "DASlogs" not in entry:
        return False
    logs = entry["DASlogs"]
    if "chopper4_TDC" not in logs or "BL9:Chop:Skf4:MotorSpeed" not in logs:
        return False
    xml = entry["instrument/instrument_xml/data"][()].tobytes().decode()
    return str(ET.fromstring(xml).get("name", "")).upper() == "CORELLI"


def is_corelli_raw_nexus_file(path: str | Path) -> bool:
    """Return whether *path* contains raw CORELLI chopper-tagged events."""

    try:
        import h5py

        with h5py.File(path, "r") as handle:
            entry = handle.get("entry")
            return entry is not None and _is_corelli_entry(entry)
    except (ImportError, KeyError, OSError, ET.ParseError, UnicodeDecodeError):
        return False


def inspect_corelli_run(path: str | Path) -> CorelliRunInfo:
    """Read run, geometry, charge, and sample-angle metadata without events."""

    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        entry = handle["entry"]
        if not _is_corelli_entry(entry):
            raise ValueError(f"{source.name} is not a raw CORELLI correlation-chopper file")
        event_count = sum(
            int(group["event_id"].shape[0])
            for name, group in entry.items()
            if name.startswith("bank") and name.endswith("_events") and "event_id" in group
        )
        angles = tuple(
            _corelli_log_value(entry, f"BL9:Mot:Sample:Axis{index}", 0.0)
            for index in (1, 2, 3)
        )
        return CorelliRunInfo(
            path=source,
            run_number=_text_scalar(entry.get("run_number"), source.stem),
            event_count=event_count,
            proton_charge=_raw_proton_charge_uah(entry),
            goniometer_angles=angles,
            l1=_source_distance(entry),
            source_to_chopper=_source_to_correlation_chopper(entry),
            ub_matrix=_ub_from_logs(entry),
        )


def corelli_dataset_group(
    paths: Iterable[str | Path],
    *,
    solid_angle_path: str | Path | None = None,
    flux_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    name: str | None = None,
    progress_callback: Any | None = None,
    _run_infos: Iterable[CorelliRunInfo] | None = None,
) -> DatasetGroup:
    """Create lightweight entries for native finite-energy CORELLI reduction."""

    resolved_paths = [Path(path) for path in paths]
    infos = list(_run_infos or ())
    if infos and len(infos) != len(resolved_paths):
        raise ValueError("CORELLI paths and pre-read run metadata do not agree")
    if not infos:
        for index, path in enumerate(resolved_paths, start=1):
            infos.append(inspect_corelli_run(path))
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "corelli_import",
                        "iteration": index,
                        "total": len(resolved_paths),
                        "message": (
                            "reading CORELLI run metadata "
                            f"{index:,}/{len(resolved_paths):,}"
                        ),
                    }
                )
    if not infos:
        raise ValueError("select at least one raw CORELLI NeXus file")
    first = infos[0]
    wavelength_min, wavelength_max = DEFAULT_WAVELENGTH_RANGE_ANGSTROM
    ei_max = ENERGY_FROM_WAVELENGTH_MEV_ANGSTROM_SQ / wavelength_min**2
    energy_limit = min(50.0, 0.95 * ei_max)
    q_upper = 2.0 * math.sqrt(ei_max / ENERGY_TO_K2)
    hkl_transform = np.linalg.inv(2.0 * np.pi * first.ub_matrix)
    hkl_limits = q_upper * np.sum(np.abs(hkl_transform), axis=1)
    shared = {
        "format": "corelli-correlation-nexus",
        "source_files": [str(info.path) for info in infos],
        "event_count": sum(info.event_count for info in infos),
        "ub_matrix": first.ub_matrix.tolist(),
        "normalization_file": None if solid_angle_path is None else str(solid_angle_path),
        "flux_file": None if flux_path is None else str(flux_path),
        "mask_file": None if mask_path is None else str(mask_path),
        "timing_offset_ns": DEFAULT_TIMING_OFFSET_NS,
        "wavelength_min_angstrom": wavelength_min,
        "wavelength_max_angstrom": wavelength_max,
        "bad_pulse_threshold": 95.0,
        "ki_kf_normalization": True,
        "he3_detector_efficiency_correction": True,
        "normalization": "cross_correlation_per_retained_proton_charge",
        "dimensions": [
            {"name": axis, "lower": -q_upper, "upper": q_upper}
            for axis in ("Q_lab_x", "Q_lab_y", "Q_lab_z")
        ]
        + [{"name": "DeltaE", "lower": -energy_limit, "upper": energy_limit}],
        "hkl_bounds": [(-float(limit), float(limit)) for limit in hkl_limits]
        + [(-energy_limit, energy_limit)],
        "q_modulus_bounds": [0.0, q_upper],
    }
    datasets = []
    for info in infos:
        datasets.append(
            DatasetEntry(
                name=f"run {info.run_number}",
                data=None,
                kind="raw_dgs_nexus",
                data_type="single_crystal_inelastic",
                metadata={
                    "source_file": str(info.path),
                    "run_number": info.run_number,
                    "event_count": info.event_count,
                    "proton_charge": info.proton_charge,
                    "corelli_goniometer_angles": list(info.goniometer_angles),
                    "l1": info.l1,
                    "source_to_chopper": info.source_to_chopper,
                    "import_status": "pending",
                },
            )
        )
    return DatasetGroup(
        name=name or first.path.stem,
        datasets=datasets,
        metadata={"raw_dgs": shared},
    )


def _run_info_from_dataset(dataset: DatasetEntry, config: dict[str, Any]):
    """Reconstruct run metadata saved at import without reopening the file."""

    metadata = dataset.metadata
    angles = tuple(
        float(value) for value in metadata["corelli_goniometer_angles"]
    )
    if len(angles) != 3:
        raise ValueError(f"{dataset.name} does not have three CORELLI sample angles")
    return CorelliRunInfo(
        path=Path(metadata["source_file"]),
        run_number=str(metadata.get("run_number", dataset.name)),
        event_count=int(metadata.get("event_count", 0)),
        proton_charge=float(metadata.get("proton_charge", 0.0)),
        goniometer_angles=angles,
        l1=float(metadata["l1"]),
        source_to_chopper=float(metadata["source_to_chopper"]),
        ub_matrix=np.asarray(config["ub_matrix"], dtype=float),
    )


def _corelli_fractional_axes(
    fractional_axes: Iterable[bool] | None,
    *,
    dimensions: int,
) -> np.ndarray:
    """Resolve CORELLI assignment modes with discrete reconstructed energy."""

    if fractional_axes is None:
        result = np.ones(dimensions, dtype=bool)
        result[-1] = False
        return result
    values = tuple(fractional_axes)
    if len(values) != dimensions or any(
        not isinstance(value, (bool, np.bool_)) for value in values
    ):
        raise ValueError(
            "fractional_axes must contain one boolean per CORELLI coordinate"
        )
    result = np.asarray(values, dtype=bool)
    if result[-1]:
        raise ValueError(
            "CORELLI energy assignment must be discrete because each output "
            "channel is reconstructed at its requested DeltaE bin centre"
        )
    return result


def _corelli_bin_contributions(
    coordinates: np.ndarray,
    edges: Iterable[np.ndarray],
    shape: tuple[int, ...],
    fractional_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return output indices and weights for mixed CORELLI assignment modes."""

    coords = np.asarray(coordinates, dtype=float)
    axis_edges = tuple(np.asarray(values, dtype=float) for values in edges)
    if coords.ndim != 2 or coords.shape[1] != len(axis_edges):
        raise ValueError("CORELLI coordinates do not match the output grid")
    assignments = np.asarray(fractional_axes, dtype=bool)
    if assignments.shape != (len(axis_edges),):
        raise ValueError("CORELLI assignment modes do not match the output grid")

    valid = np.all(np.isfinite(coords), axis=1)
    lower_indices: list[np.ndarray] = []
    upper_indices: list[np.ndarray] = []
    upper_weights: list[np.ndarray] = []
    for dimension, values in enumerate(axis_edges):
        coordinate = coords[:, dimension]
        valid &= (coordinate >= values[0]) & (coordinate <= values[-1])
        if not assignments[dimension] or values.size == 2:
            index = np.searchsorted(values, coordinate, side="right") - 1
            index[coordinate == values[-1]] = values.size - 2
            index = np.clip(index, 0, values.size - 2)
            lower_indices.append(index)
            upper_indices.append(index)
            upper_weights.append(np.zeros(coordinate.shape, dtype=float))
            continue
        centers = 0.5 * (values[:-1] + values[1:])
        left = np.searchsorted(centers, coordinate, side="right") - 1
        left = np.clip(left, 0, centers.size - 2)
        fraction = (coordinate - centers[left]) / (
            centers[left + 1] - centers[left]
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        lower_indices.append(left)
        upper_indices.append(left + 1)
        upper_weights.append(fraction)

    source_indices = np.flatnonzero(valid)
    flat_blocks: list[np.ndarray] = []
    point_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    fractional_dimensions = np.flatnonzero(assignments)
    for offsets in product((0, 1), repeat=fractional_dimensions.size):
        selected = [indices[source_indices].copy() for indices in lower_indices]
        spatial = np.ones(source_indices.size, dtype=float)
        for dimension, use_upper in zip(
            fractional_dimensions,
            offsets,
            strict=True,
        ):
            fraction = upper_weights[dimension][source_indices]
            if use_upper:
                selected[dimension] = upper_indices[dimension][source_indices]
                spatial *= fraction
            else:
                spatial *= 1.0 - fraction
        keep = spatial > 0.0
        if not np.any(keep):
            continue
        flat_blocks.append(
            np.ravel_multi_index(
                tuple(indices[keep] for indices in selected),
                shape,
            )
        )
        point_blocks.append(source_indices[keep])
        weight_blocks.append(spatial[keep])
    if not flat_blocks:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )
    return (
        np.concatenate(flat_blocks),
        np.concatenate(point_blocks),
        np.concatenate(weight_blocks),
    )


def _event_metadata_exposure_uah(
    pulse_seconds: np.ndarray,
    pulse_charge_uah: np.ndarray,
    event_logs: tuple[tuple[np.ndarray, np.ndarray], ...],
    metadata_edges: tuple[np.ndarray, ...],
    fractional_axes: np.ndarray,
) -> np.ndarray:
    """Accumulate retained beam charge in each event-metadata output bin.

    Event-pulse metadata partition one acquisition into distinct physical
    conditions. Each resulting slice must therefore be divided by its own
    charge exposure, rather than by the total charge for the whole run.
    """

    shape = tuple(edge.size - 1 for edge in metadata_edges)
    if not shape:
        return np.empty(0, dtype=float)
    pulse_seconds = np.asarray(pulse_seconds, dtype=float).reshape(-1)
    charge = np.asarray(pulse_charge_uah, dtype=float).reshape(-1)
    if pulse_seconds.shape != charge.shape:
        raise ValueError("CORELLI pulse times and charge values must be aligned")
    coordinates = np.column_stack(
        [
            np.interp(pulse_seconds, times, values, left=np.nan, right=np.nan)
            for times, values in event_logs
        ]
    )
    flat, pulse_indices, fractions = _corelli_bin_contributions(
        coordinates,
        metadata_edges,
        shape,
        fractional_axes,
    )
    return np.bincount(
        flat,
        weights=charge[pulse_indices] * fractions,
        minlength=int(np.prod(shape, dtype=np.int64)),
    ).reshape(shape)


def _event_pulse_charge_uah(
    entry,
    pulse_seconds: np.ndarray,
    pulse_keep,
    retained_charge_uah: float,
) -> np.ndarray:
    """Return retained charge aligned to a CORELLI event-time-zero series."""

    pulses = np.asarray(pulse_seconds, dtype=float).reshape(-1)
    values = _pulse_charge_values(entry)
    keep = np.ones(pulses.size, dtype=bool)
    if pulse_keep is not None and pulses.size:
        keep &= np.asarray(pulse_keep, dtype=bool)[
            np.minimum(np.arange(pulses.size), len(pulse_keep) - 1)
        ]
    if values is not None and values.size == pulses.size:
        charge = np.asarray(values, dtype=float) / 3.6e9
        charge[~keep] = 0.0
        return charge
    count = int(np.count_nonzero(keep))
    charge = np.zeros(pulses.size, dtype=float)
    if count:
        charge[keep] = float(retained_charge_uah) / count
    return charge


def bin_corelli_group(
    group: DatasetGroup,
    *,
    lower: Iterable[float],
    upper: Iterable[float],
    num_bins: Iterable[int],
    step_size: Iterable[float] | None = None,
    bin_edges: Iterable[Iterable[float] | None] | None = None,
    minimum_samples: float = 0.0,
    datasets: Iterable[DatasetEntry] | None = None,
    vectors: Iterable[Iterable[float]] | None = None,
    axis_names: Iterable[str] | None = None,
    max_batch_bytes: int = 192 * 1024 * 1024,
    progress_callback: Any | None = None,
    symmetry_operations: Iterable[Iterable[Iterable[float]]] | None = None,
    coordinate_mode: str = "hkle",
    fractional_axes: Iterable[bool] | None = None,
    metadata_dimensions: Iterable[Any] | None = None,
) -> MDHistoData:
    """Cross-correlate raw CORELLI events at requested finite DeltaE bins.

    The returned uncertainties are diagonal variances from the signed event
    weights.  Cross correlation makes different DeltaE channels from the same
    measured neutron statistically correlated; that covariance is described in
    metadata but cannot be represented by :class:`MDHistoData`.
    """

    config = group.metadata.get("raw_dgs")
    if not isinstance(config, dict) or config.get("format") != "corelli-correlation-nexus":
        raise ValueError("group is not configured for CORELLI correlation reconstruction")
    if coordinate_mode not in {"hkle", "powder"}:
        raise ValueError("CORELLI coordinate mode must be 'hkle' or 'powder'")
    powder = coordinate_mode == "powder"
    if powder and metadata_dimensions:
        raise ValueError("CORELLI event metadata dimensions require single-crystal HKLE output")
    from .metadata_dimensions import MetadataDimension
    metadata_specs = tuple(
        item if isinstance(item, MetadataDimension) else MetadataDimension(**item)
        for item in (metadata_dimensions or ())
    )
    if any(spec.sampling != "event_pulse_time" for spec in metadata_specs):
        raise ValueError("raw CORELLI metadata dimensions must use event_pulse_time sampling")
    physical_dimensions = 2 if powder else 4
    expected_dimensions = physical_dimensions + len(metadata_specs)
    lo, hi, bins = (
        np.asarray(tuple(values), dtype=dtype)
        for values, dtype in ((lower, float), (upper, float), (num_bins, int))
    )
    if lo.shape != (expected_dimensions,) or hi.shape != (expected_dimensions,) or bins.shape != (expected_dimensions,) or np.any(bins <= 0):
        raise ValueError(f"CORELLI {coordinate_mode} binning requires {expected_dimensions} positive bin counts")
    if powder and lo[0] < 0.0:
        raise ValueError("powder |Q| lower bound must be nonnegative")
    edges = _requested_edges(lo, hi, bins, step_size, bin_edges=bin_edges)
    assignments = _corelli_fractional_axes(
        fractional_axes,
        dimensions=physical_dimensions,
    )
    minimum_samples = _validated_minimum_samples(minimum_samples)
    shape = tuple(edge.size - 1 for edge in edges)
    energy_centres = 0.5 * (edges[3 if not powder else 1][:-1] + edges[3 if not powder else 1][1:])
    basis = None
    basis_inverse = None
    symmetry = (np.eye(3),)
    names = ("|Q|", "DeltaE")
    if not powder:
        basis = np.eye(4) if vectors is None else np.asarray(tuple(tuple(v) for v in vectors), dtype=float)
        if basis.shape != (4, 4) or np.linalg.matrix_rank(basis) != 4:
            raise ValueError("CORELLI coordinate axes must form an invertible 4D basis")
        if np.any(basis[:3, 3]) or np.any(basis[3, :3]) or basis[3, 3] != 1.0:
            raise ValueError("CORELLI momentum axes cannot mix energy")
        basis_inverse = np.linalg.inv(basis)
        symmetry = _symmetry_matrices(symmetry_operations)
        names = (*tuple(axis_names or ("H", "K", "L", "DeltaE")), *(spec.name for spec in metadata_specs))
    if metadata_specs:
        assignments = np.r_[assignments, [bool(spec.binning.fractional) for spec in metadata_specs]]

    data_sum = np.zeros(shape)
    variance_sum = np.zeros(shape)
    hypothesis_count = np.zeros(shape)
    metadata_exposure_uah = (
        np.zeros(shape[physical_dimensions:], dtype=float) if metadata_specs else None
    )
    selected = list(group.datasets if datasets is None else datasets)
    total_events = sum(int(dataset.metadata.get("event_count", 0)) for dataset in selected)
    processed = 0
    retained_charge = 0.0
    normalization_scale = 0.0
    duty_cycles = []
    detector_norm = load_detector_normalization(config["normalization_file"]) if config.get("normalization_file") else None
    incident_flux = _load_corelli_flux(config["flux_file"]) if config.get("flux_file") else None
    detector_mask = _load_corelli_detector_mask(config.get("mask_file"))
    wavelength_min = float(config.get("wavelength_min_angstrom", 0.6))
    wavelength_max = float(config.get("wavelength_max_angstrom", 2.5))
    if not (0.0 < wavelength_min < wavelength_max):
        raise ValueError("CORELLI wavelength limits must be positive and increasing")
    ei_bounds = (
        ENERGY_FROM_WAVELENGTH_MEV_ANGSTROM_SQ / wavelength_max**2,
        ENERGY_FROM_WAVELENGTH_MEV_ANGSTROM_SQ / wavelength_min**2,
    )

    import h5py

    shared_geometry = None
    if selected:
        shared_geometry = _detector_geometry(Path(selected[0].metadata["source_file"]))
        if config.get("normalization_file"):
            shared_geometry = _geometry_from_solid_angle(
                config["normalization_file"], shared_geometry
            )

    for dataset in selected:
        source = Path(dataset.metadata["source_file"])
        info = _run_info_from_dataset(dataset, config)
        geometry = shared_geometry
        if geometry is None:
            raise ValueError("CORELLI reconstruction requires at least one detector geometry")
        gonio = _corelli_goniometer(info.goniometer_angles)
        ub = np.asarray(config["ub_matrix"], dtype=float)
        hkl_transform = np.linalg.inv(2.0 * np.pi * ub)
        rows = max(1, int(max_batch_bytes) // max(128, 8 * max(1, energy_centres.size)))
        with h5py.File(source, "r") as handle:
            entry = handle["entry"]
            timing = _read_chopper_timing_entry(entry, source.name)
            event_logs = _event_metadata_logs(entry, metadata_specs, source.name)
            duty_cycles.append(timing.duty_cycle)
            pulse_keep = _good_pulses(entry, float(config.get("bad_pulse_threshold", 0.0)))
            run_charge = _retained_proton_charge_uah(
                entry, float(config.get("bad_pulse_threshold", 0.0))
            )
            retained_charge += run_charge
            normalization_scale += run_charge * timing.duty_cycle
            if metadata_specs:
                pulse_times = next(
                    (
                        np.asarray(bank["event_time_zero"], dtype=float)
                        for bank_name, bank in entry.items()
                        if bank_name.startswith("bank")
                        and bank_name.endswith("_events")
                        and "event_time_zero" in bank
                    ),
                    np.empty(0, dtype=float),
                )
                metadata_exposure_uah += timing.duty_cycle * _event_metadata_exposure_uah(
                    pulse_times,
                    _event_pulse_charge_uah(entry, pulse_times, pulse_keep, run_charge),
                    event_logs,
                    tuple(edges[physical_dimensions:]),
                    assignments[physical_dimensions:],
                )
            for bank_name, bank in entry.items():
                if not (bank_name.startswith("bank") and bank_name.endswith("_events") and "event_id" in bank):
                    continue
                ids = bank["event_id"]
                if ids.shape[0] == 0:
                    continue
                tofs = bank["event_time_offset"]
                event_index = np.asarray(bank["event_index"], dtype=np.int64)
                pulse_zero = np.asarray(bank["event_time_zero"], dtype=float)
                for start in range(0, ids.shape[0], rows):
                    stop = min(start + rows, ids.shape[0])
                    event_ids = np.asarray(ids[start:stop], dtype=np.int64)
                    total_tof = np.asarray(tofs[start:stop], dtype=float)
                    positions, he3_exponents, valid_detector = geometry.event_geometry_for_ids(event_ids)
                    pulse_index = np.searchsorted(event_index, np.arange(start, stop), side="right") - 1
                    pulse_index = np.clip(pulse_index, 0, pulse_zero.size - 1)
                    base_valid = valid_detector.copy()
                    if pulse_keep is not None:
                        base_valid &= pulse_keep[np.minimum(pulse_index, pulse_keep.size - 1)]
                    if detector_norm is not None:
                        base_valid &= detector_norm.value_for_ids(event_ids) > 0.0
                    if detector_mask is not None:
                        base_valid &= detector_mask.value_for_ids(event_ids) > 0.0
                    if np.any(base_valid):
                        event_ids = event_ids[base_valid]
                        total_tof = total_tof[base_valid]
                        positions = positions[base_valid]
                        he3_exponents = he3_exponents[base_valid]
                        pulse_seconds = pulse_zero[pulse_index[base_valid]]
                        event_metadata = tuple(
                            np.interp(pulse_seconds, times, values, left=np.nan, right=np.nan)
                            for times, values in event_logs
                        )
                        accepted_ids = event_ids
                        solid_angle = (
                            np.ones(accepted_ids.size)
                            if detector_norm is None
                            else detector_norm.value_for_ids(accepted_ids)
                        )
                        l2 = np.linalg.norm(positions, axis=1)
                        direction = positions / l2[:, None]
                        if (
                            _CORELLI_NUMBA is not None
                            and energy_centres.size > 1
                            and not metadata_specs
                            and total_tof.size * energy_centres.size
                            >= CORELLI_NUMBA_MIN_HYPOTHESES
                        ):
                            flux_rows = (
                                np.full(accepted_ids.size, -1, dtype=np.int64)
                                if incident_flux is None
                                else incident_flux.row_for_ids(accepted_ids)
                            )
                            flux_axis = (
                                np.empty(0)
                                if incident_flux is None
                                else incident_flux.momentum
                            )
                            cumulative_flux = (
                                np.empty((0, 0))
                                if incident_flux is None
                                else incident_flux.cumulative
                            )
                            workers = (
                                min(_parallel.num_threads(), energy_centres.size)
                                if energy_centres.size > 1
                                else max(1, _trajectory_worker_count(data_sum.size) // 3)
                            )
                            common = (
                                np.ascontiguousarray(total_tof),
                                np.ascontiguousarray(pulse_seconds * 1.0e9),
                                np.ascontiguousarray(direction),
                                np.ascontiguousarray(l2),
                                np.ascontiguousarray(he3_exponents),
                                np.ascontiguousarray(solid_angle),
                                np.ascontiguousarray(flux_rows),
                                np.ascontiguousarray(energy_centres),
                                float(info.l1),
                                float(info.source_to_chopper),
                                float(ei_bounds[0]),
                                float(ei_bounds[1]),
                                np.ascontiguousarray(timing.tdc_ns),
                                float(timing.period_ns),
                                np.ascontiguousarray(timing.sequence_edges_deg),
                                float(timing.duty_cycle),
                                float(
                                    config.get(
                                        "timing_offset_ns",
                                        DEFAULT_TIMING_OFFSET_NS,
                                    )
                                ),
                                np.ascontiguousarray(flux_axis),
                                np.ascontiguousarray(cumulative_flux),
                            )
                            if powder:
                                partials = _CORELLI_NUMBA.run_powder(
                                    *common,
                                    np.ascontiguousarray(edges[0]),
                                    np.ascontiguousarray(edges[1]),
                                    np.ascontiguousarray(assignments),
                                    bool(
                                        config.get(
                                            "he3_detector_efficiency_correction", True
                                        )
                                    ),
                                    _use_ki_kf_correction(config),
                                    workers=workers,
                                )
                            else:
                                partials = _CORELLI_NUMBA.run_hkle(
                                    *common,
                                    np.ascontiguousarray(
                                        gonio @ hkl_transform.T
                                    ),
                                    np.ascontiguousarray(basis_inverse),
                                    np.ascontiguousarray(symmetry),
                                    np.ascontiguousarray(edges[0]),
                                    np.ascontiguousarray(edges[1]),
                                    np.ascontiguousarray(edges[2]),
                                    np.ascontiguousarray(edges[3]),
                                    np.asarray(shape, dtype=np.int64),
                                    np.ascontiguousarray(assignments),
                                    bool(
                                        config.get(
                                            "he3_detector_efficiency_correction", True
                                        )
                                    ),
                                    _use_ki_kf_correction(config),
                                    workers=workers,
                                )
                            data_sum.ravel()[:] += partials[0]
                            variance_sum.ravel()[:] += partials[1]
                            hypothesis_count.ravel()[:] += partials[2]
                            processed += stop - start
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "stage": "corelli_cross_correlation",
                                        "iteration": processed,
                                        "total": total_events,
                                        "message": f"cross-correlating CORELLI events {processed:,}/{total_events:,}",
                                    }
                                )
                            continue
                        for delta_e in energy_centres:
                            ei = _solve_incident_energy(total_tof, l2, float(delta_e), info.l1, ei_bounds)
                            valid = np.isfinite(ei)
                            if not np.any(valid):
                                continue
                            ef = ei[valid] - float(delta_e)
                            ki = np.sqrt(ei[valid] / ENERGY_TO_K2)
                            kf = np.sqrt(ef / ENERGY_TO_K2)
                            t0 = _corelli_t0(ei[valid])
                            crossing_us = (
                                t0
                                * (
                                    1.0
                                    - info.source_to_chopper
                                    / (info.l1 + l2[valid])
                                )
                                + CORELLI_TOF_US_PER_M_SQRT_MEV
                                * info.source_to_chopper
                                / np.sqrt(ei[valid])
                            )
                            absolute_ns = pulse_seconds[valid] * 1.0e9 + crossing_us * 1.0e3
                            correlation, timed = _correlation_weights(
                                absolute_ns,
                                timing,
                                float(config.get("timing_offset_ns", DEFAULT_TIMING_OFFSET_NS)),
                            )
                            if not np.any(timed):
                                continue
                            chosen = np.flatnonzero(valid)[timed]
                            chosen_direction = direction[chosen]
                            q_lab = np.column_stack(
                                (
                                    -kf[timed] * chosen_direction[:, 0],
                                    -kf[timed] * chosen_direction[:, 1],
                                    ki[timed] - kf[timed] * chosen_direction[:, 2],
                                )
                            )
                            weights = correlation[timed]
                            normalization = solid_angle[chosen]
                            if incident_flux is not None:
                                normalization = normalization * incident_flux.density_for_ids(
                                    accepted_ids[chosen], ki[timed]
                                )
                            normalized = np.isfinite(normalization) & (normalization > 0.0)
                            if not np.any(normalized):
                                continue
                            chosen = chosen[normalized]
                            chosen_direction = chosen_direction[normalized]
                            q_lab = q_lab[normalized]
                            ki = ki[timed][normalized]
                            kf = kf[timed][normalized]
                            weights = weights[normalized] / normalization[normalized]
                            if config.get("he3_detector_efficiency_correction", True):
                                weights = weights * _he3_tube_efficiency_correction(kf, he3_exponents[chosen])
                            if _use_ki_kf_correction(config):
                                weights = weights * ki / kf
                            if powder:
                                coordinate_blocks = (np.column_stack((np.linalg.norm(q_lab, axis=1), np.full(q_lab.shape[0], delta_e))),)
                            else:
                                hkl = (q_lab @ gonio) @ hkl_transform.T
                                coordinate_blocks = tuple(
                                    np.column_stack((hkl @ operation.T, np.full(hkl.shape[0], delta_e))) @ basis_inverse
                                    for operation in symmetry
                                )
                            for coords in coordinate_blocks:
                                if event_metadata:
                                    coords = np.column_stack((coords, *(values[chosen] for values in event_metadata)))
                                flat, point_indices, spatial = (
                                    _corelli_bin_contributions(
                                        coords,
                                        edges,
                                        shape,
                                        assignments,
                                    )
                                )
                                if flat.size:
                                    contributions = weights[point_indices] * spatial
                                    data_sum.ravel()[:] += np.bincount(
                                        flat,
                                        weights=contributions,
                                        minlength=data_sum.size,
                                    )
                                    variance_sum.ravel()[:] += np.bincount(
                                        flat,
                                        weights=contributions**2,
                                        minlength=variance_sum.size,
                                    )
                                    hypothesis_count.ravel()[:] += np.bincount(
                                        flat,
                                        weights=spatial,
                                        minlength=hypothesis_count.size,
                                    )
                    processed += stop - start
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "stage": "corelli_cross_correlation",
                                "iteration": processed,
                                "total": total_events,
                                "message": f"cross-correlating CORELLI events {processed:,}/{total_events:,}",
                            }
                        )

    duty = float(np.mean(duty_cycles)) if duty_cycles else 1.0
    scale = normalization_scale if normalization_scale > 0.0 else (duty if duty > 0.0 else 1.0)
    if metadata_exposure_uah is None:
        output_scale = np.full(shape, scale, dtype=float)
    else:
        output_scale = np.broadcast_to(
            metadata_exposure_uah.reshape((1,) * physical_dimensions + metadata_exposure_uah.shape),
            shape,
        )
    normalized = output_scale > 0.0
    signal = np.zeros_like(data_sum)
    np.divide(data_sum, output_scale, out=signal, where=normalized)
    errors = np.zeros_like(variance_sum)
    np.divide(np.sqrt(variance_sum), output_scale, out=errors, where=normalized)
    nonempty = hypothesis_count > 0.0
    rms = float(np.sqrt(variance_sum.sum() / hypothesis_count.sum())) if np.any(nonempty) else 1.0
    errors[(~nonempty) & normalized] = (
        FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER
        * rms
        / output_scale[(~nonempty) & normalized]
    )
    mask = (~nonempty) | (hypothesis_count < minimum_samples) | ~normalized
    if powder:
        axes = (
            MDHistoAxis("|Q|", edges[0], "1/angstrom", "momentum", frame="Q modulus"),
            MDHistoAxis("DeltaE", edges[1], "meV", "energy", frame="General Frame"),
        )
    else:
        axes = tuple(
            MDHistoAxis(name, edge, "meV" if index == 3 else "r.l.u.", "energy" if index == 3 else "momentum", frame="General Frame" if index == 3 else "HKL")
            for index, (name, edge) in enumerate(zip(names[:4], edges[:4], strict=True))
        )
        axes += tuple(
            MDHistoAxis(spec.name, edge, spec.units, "unknown", metadata={"metadata_dimension": spec.to_dict(), "interpolation": "pulse_time_linear"})
            for spec, edge in zip(metadata_specs, edges[4:], strict=True)
        )
    return MDHistoData(
        axes=axes,
        signal=signal,
        errors=errors,
        mask=mask,
        num_events=hypothesis_count,
        metadata={
            "raw_dgs": config,
            "corelli_reconstruction": {
                "method": "correlation_chopper_finite_energy",
                "energy_sampling": "requested_DeltaE_bin_centres",
                "timing_offset_ns": float(config.get("timing_offset_ns", DEFAULT_TIMING_OFFSET_NS)),
                "wavelength_range_angstrom": [wavelength_min, wavelength_max],
                "duty_cycle": duty,
                "retained_proton_charge_uah": retained_charge,
                "metadata_exposure_uah": (
                    None if metadata_exposure_uah is None else metadata_exposure_uah.tolist()
                ),
                "channel_covariance": "DeltaE channels reconstructed from the same measured events are correlated; MDHisto errors contain diagonal variances only.",
                "fractional_axes": assignments.tolist(),
                "event_metadata_dimensions": [spec.to_dict() for spec in metadata_specs],
                "normalization_limit": "Counts include optional pointwise solid-angle and incident-flux corrections and are normalized by retained proton charge and chopper duty cycle. Event-time metadata slices are normalized by their own retained charge exposure; full four-dimensional MDNorm trajectory normalization is not applied.",
                "solid_angle_file": config.get("normalization_file"),
                "flux_file": config.get("flux_file"),
            },
            "rebin": {
                **({"vectors": basis.tolist()} if basis is not None else {}),
                "bin_edges": [edge.tolist() for edge in edges],
                "minimum_samples": minimum_samples,
                "fractional_axes": assignments.tolist(),
            },
            "signal_semantics": "cross_correlation_intensity",
            "signal_semantics_source": "nfit_corelli_finite_energy_reconstruction",
            "zero_event_bins_are_measured": False,
            "event_weight_rms": rms,
            "ki_kf_normalization": _use_ki_kf_correction(config),
            "he3_detector_efficiency_correction": bool(config.get("he3_detector_efficiency_correction", True)),
            **({"symmetry_operations_hkl": [operation.tolist() for operation in symmetry]} if not powder else {}),
        },
    )


def bin_corelli_powder_group(group: DatasetGroup, **kwargs: Any) -> MDHistoData:
    """Cross-correlate raw CORELLI events directly into ``|Q|, DeltaE``."""

    return bin_corelli_group(group, coordinate_mode="powder", **kwargs)


def _corelli_t0(incident_energy: np.ndarray | float) -> np.ndarray:
    energy = np.asarray(incident_energy, dtype=float)
    return 101.9 * energy**-0.41 * np.exp(-energy / 282.0)


def _solve_incident_energy(total_tof_us, l2, delta_e, l1, energy_bounds):
    """Solve the finite-energy flight equation in Mantid's TOF convention."""

    tof = np.asarray(total_tof_us, dtype=float)
    detector_distance = np.asarray(l2, dtype=float)
    e_min = max(float(energy_bounds[0]), float(delta_e) + 1.0e-6)
    e_max = float(energy_bounds[1])
    result = np.full(tof.shape, np.nan)
    if e_min >= e_max:
        return result
    elastic = (CORELLI_TOF_US_PER_M_SQRT_MEV * (float(l1) + detector_distance) / np.maximum(tof, 1.0)) ** 2
    energy = np.clip(elastic + float(delta_e) * detector_distance / (float(l1) + detector_distance), e_min, e_max)
    for _ in range(7):
        final = np.maximum(energy - float(delta_e), 1.0e-8)
        calculated = CORELLI_TOF_US_PER_M_SQRT_MEV * (float(l1) / np.sqrt(energy) + detector_distance / np.sqrt(final))
        derivative = -0.5 * CORELLI_TOF_US_PER_M_SQRT_MEV * (float(l1) / energy**1.5 + detector_distance / final**1.5)
        step = np.divide(calculated - tof, derivative, out=np.zeros_like(energy), where=np.abs(derivative) > 1.0e-12)
        energy = np.clip(energy - step, e_min, e_max)
    final = energy - float(delta_e)
    calculated = CORELLI_TOF_US_PER_M_SQRT_MEV * (float(l1) / np.sqrt(energy) + detector_distance / np.sqrt(final))
    valid = np.isfinite(energy) & np.isfinite(calculated) & (final > 0.0) & (np.abs(calculated - tof) <= 0.05)
    result[valid] = energy[valid]
    return result


def _read_chopper_timing(path: Path) -> _ChopperTiming:
    import h5py

    with h5py.File(path, "r") as handle:
        return _read_chopper_timing_entry(handle["entry"], path.name)


def _event_metadata_logs(entry, specs, source_name: str):
    """Return timestamp/value pairs for event-aligned NeXus metadata logs."""

    result = []
    for spec in specs:
        path = str(spec.source).replace(">", "/").strip("/")
        if path.startswith("entry/"):
            path = path.removeprefix("entry/")
        if path.endswith("/value"):
            path = path.removesuffix("/value")
        try:
            log = entry[path]
            times = np.asarray(log["time"], dtype=float).reshape(-1)
            values = np.asarray(log["value"], dtype=float).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_name}: event metadata {spec.name!r} requires a NeXus log "
                f"with time and value datasets at entry/{path}"
            ) from exc
        count = min(times.size, values.size)
        times, values = times[:count], values[:count]
        valid = np.isfinite(times) & np.isfinite(values)
        times, values = times[valid], values[valid]
        if times.size < 2 or np.any(np.diff(times) < 0):
            raise ValueError(
                f"{source_name}: event metadata {spec.name!r} needs at least two "
                "finite, increasing timestamps"
            )
        unique = np.r_[True, np.diff(times) > 0]
        result.append((times[unique], values[unique]))
    return tuple(result)


def _read_chopper_timing_entry(entry, source_name: str) -> _ChopperTiming:
    tdc = np.asarray(entry["DASlogs/chopper4_TDC/time"], dtype=float) * 1.0e9
    speed = np.asarray(
        entry["DASlogs/BL9:Chop:Skf4:MotorSpeed/value"], dtype=float
    )
    xml = entry["instrument/instrument_xml/data"][()].tobytes().decode()
    root = ET.fromstring(xml)
    parameter = root.find(
        ".//{*}component-link[@name='correlation-chopper']/"
        "{*}parameter[@name='sequence']/{*}value"
    )
    if parameter is None or not parameter.get("val"):
        raise ValueError(
            f"{source_name} does not define the CORELLI chopper sequence"
        )
    widths = np.asarray([float(item) for item in parameter.get("val").split()], dtype=float)
    edges = np.cumsum(widths)
    total_open = float(np.sum(widths[1::2]))
    duty = total_open / float(edges[-1])
    finite_speed = speed[np.isfinite(speed) & (speed > 0.0)]
    if tdc.size < 2 or not finite_speed.size:
        raise ValueError(f"{source_name} has incomplete CORELLI chopper timing logs")
    return _ChopperTiming(tdc, 1.0e9 / float(np.mean(finite_speed)), edges, duty)


def _load_corelli_flux(path: str | Path) -> _CorelliFlux:
    """Load Mantid's bank-grouped cumulative CORELLI flux workspace."""

    import h5py

    with h5py.File(path, "r") as handle:
        entries = [
            group
            for group in handle.values()
            if "workspace" in group
            and "instrument/detector/detector_list" in group
        ]
        if not entries:
            raise ValueError("flux file does not contain a Mantid MatrixWorkspace")
        entry = entries[0]
        momentum = np.asarray(entry["workspace/axis1"], dtype=float)
        cumulative = np.asarray(entry["workspace/values"], dtype=float)
        detector_ids = np.asarray(
            entry["instrument/detector/detector_list"], dtype=np.int64
        )
        counts = np.asarray(
            entry["instrument/detector/detector_count"], dtype=np.int64
        )
        starts = np.asarray(
            entry["instrument/detector/detector_index"], dtype=np.int64
        )
    if (
        cumulative.ndim != 2
        or cumulative.shape[1] != momentum.size
        or momentum.size < 2
        or np.any(np.diff(momentum) <= 0.0)
    ):
        raise ValueError("CORELLI flux workspace has incompatible axes")
    rows = np.full(detector_ids.size, -1, dtype=np.int64)
    for row, (start, count) in enumerate(zip(starts, counts, strict=True)):
        rows[int(start) : int(start + count)] = row
    valid = rows >= 0
    order = np.argsort(detector_ids[valid])
    return _CorelliFlux(
        momentum,
        cumulative,
        detector_ids[valid][order],
        rows[valid][order],
    )


def _load_corelli_detector_mask(path: str | Path | None):
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() != ".xml":
        return load_detector_normalization(source)
    root = ET.parse(source).getroot()
    masked = []
    for element in root.findall(".//detids") + root.findall(".//{*}detids"):
        for token in (element.text or "").replace(" ", "").split(","):
            if not token:
                continue
            if "-" in token:
                start, stop = (int(value) for value in token.split("-", 1))
                masked.extend(range(start, stop + 1))
            else:
                masked.append(int(token))
    if not masked:
        raise ValueError(f"{source.name} does not contain Mantid detector IDs")
    return _CorelliDetectorMask(np.unique(np.asarray(masked, dtype=np.int64)))


def _geometry_from_solid_angle(path: str | Path, fallback: _DetectorGeometry):
    """Use the calibrated detector positions saved with a Mantid workspace."""

    import h5py

    try:
        with h5py.File(path, "r") as handle:
            entries = [
                group
                for group in handle.values()
                if "instrument/detector/detector_list" in group
                and "instrument/detector/detector_positions" in group
            ]
            if not entries:
                return fallback
            detector = entries[0]["instrument/detector"]
            ids = np.asarray(detector["detector_list"], dtype=np.int64)
            positions = np.asarray(detector["detector_positions"], dtype=float)
        if positions.shape != (ids.size, 3):
            return fallback
        # Mantid saves detector_positions as distance, polar angle, azimuthal
        # angle rather than Cartesian x, y, z for this processed workspace.
        radius = positions[:, 0]
        polar = np.deg2rad(positions[:, 1])
        azimuth = np.deg2rad(positions[:, 2])
        positions = np.column_stack(
            (
                radius * np.sin(polar) * np.cos(azimuth),
                radius * np.sin(polar) * np.sin(azimuth),
                radius * np.cos(polar),
            )
        )
        _, exponents, valid = fallback.event_geometry_for_ids(ids)
        exponents[~valid] = 0.0
        order = np.argsort(ids)
        return _DetectorGeometry(ids[order], positions[order], exponents[order])
    except (KeyError, OSError, ValueError):
        return fallback


def _correlation_weights(absolute_ns, timing, offset_ns):
    shifted_tdc = timing.tdc_ns + float(offset_ns)
    candidate = np.asarray(absolute_ns, dtype=float)
    tdc_index = np.searchsorted(shifted_tdc, candidate, side="right") - 1
    valid = (tdc_index >= 0) & (tdc_index < shifted_tdc.size)
    chosen = np.clip(tdc_index, 0, shifted_tdc.size - 1)
    angle = 360.0 * (candidate - shifted_tdc[chosen]) / timing.period_ns
    valid &= (angle >= 0.0) & (angle <= timing.sequence_edges_deg[-1])
    sequence_index = np.searchsorted(timing.sequence_edges_deg, angle, side="left")
    weights = np.ones(candidate.shape, dtype=float)
    absorbing = sequence_index % 2 == 0
    weights[absorbing] = -timing.duty_cycle / (1.0 - timing.duty_cycle)
    return weights, valid


def _source_to_correlation_chopper(entry) -> float:
    xml = entry["instrument/instrument_xml/data"][()].tobytes().decode()
    root = ET.fromstring(xml)
    source_z = -_source_distance(entry)
    for component in root.findall("{*}component"):
        if component.get("type") == "correlation-chopper":
            location = component.find("{*}location")
            if location is not None:
                return abs(float(location.get("z", 0.0)) - source_z)
    raise ValueError("CORELLI instrument XML does not define the correlation chopper")


def _corelli_log_value(entry, name, default):
    logs = entry.get("DASlogs")
    if logs is None or name not in logs:
        return float(default)
    group = logs[name]
    data = group.get("average_value") or group.get("value")
    if data is None:
        return float(default)
    values = np.asarray(data, dtype=float).reshape(-1)
    return float(np.mean(values)) if values.size else float(default)


def _corelli_goniometer(angles):
    # Garnet/Mantid configures all three CORELLI sample motors about lab +y.
    angle = math.radians(float(sum(angles)))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
