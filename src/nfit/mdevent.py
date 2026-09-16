"""Dependency-free Mantid MDEventWorkspace inspection and reduction helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from . import _parallel
from .dataset import PointData4D
from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from .pipeline import DatasetEntry, DatasetGroup

try:
    from . import _mdevent_numba as _MDEVENT_NUMBA
except Exception:
    _MDEVENT_NUMBA = None


ENERGY_TO_K2 = 2.072124855  # meV Angstrom^2: E = ENERGY_TO_K2 * k^2
# Feldman-Cousins Table II: n = 0, known background b = 0, 68.27% C.L.
FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER = 1.29
EVENT_COLUMNS = {
    "signal": 0, "error_squared": 1, "experiment_index": 2,
    "goniometer_index": 3, "detector_id": 4,
    "qx": 5, "qy": 6, "qz": 7, "energy": 8,
}
MDEVENT_FIXED_MEMORY_BYTES = 1_500_000_000
MDEVENT_BYTES_PER_OUTPUT_BIN = 72


@dataclass(frozen=True)
class MDEventRunInfo:
    experiment_index: int
    run_number: str
    incident_energy: float
    proton_charge: float
    omega: float
    phi: float
    chi: float
    duration: float
    t0: float | None
    goniometer: np.ndarray
    rubw_matrix: np.ndarray


@dataclass(frozen=True)
class MDEventWorkspaceInfo:
    path: Path
    workspace_path: str
    event_count: int
    dimensions: tuple[dict[str, Any], ...]
    runs: tuple[MDEventRunInfo, ...]
    ub_matrix: np.ndarray
    lattice_parameters: dict[str, float]


@dataclass(frozen=True)
class DetectorNormalization:
    path: Path
    detector_ids: np.ndarray
    values: np.ndarray
    errors: np.ndarray

    @property
    def masked_detector_ids(self) -> np.ndarray:
        return self.detector_ids[~np.isfinite(self.values) | (self.values <= 0.0)]

    def value_for_ids(self, detector_ids: np.ndarray) -> np.ndarray:
        order = np.argsort(self.detector_ids)
        sorted_ids = self.detector_ids[order]
        locations = np.searchsorted(sorted_ids, detector_ids)
        valid = (locations < sorted_ids.size) & (sorted_ids[np.minimum(locations, sorted_ids.size - 1)] == detector_ids)
        result = np.zeros(detector_ids.shape, dtype=float)
        result[valid] = self.values[order[locations[valid]]]
        return result


def is_mdevent_file(path: str | Path) -> bool:
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            return "MDEventWorkspace/event_data/event_data" in handle
    except (OSError, ImportError):
        return False


def inspect_mdevent_workspace(
    path: str | Path,
    workspace_path: str = "/MDEventWorkspace",
    *,
    progress_callback: Any | None = None,
) -> MDEventWorkspaceInfo:
    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        workspace = handle[workspace_path]
        dimensions = tuple(_parse_dimension(workspace.attrs[f"dimension{i}"]) for i in range(int(_scalar(workspace["dimensions"]))))
        names = sorted(
            (name for name in workspace if re.fullmatch(r"experiment\d+", name)),
            key=lambda name: int(name.removeprefix("experiment")),
        )
        runs_list = []
        for index, name in enumerate(names, start=1):
            runs_list.append(_read_run(workspace[name], int(name.removeprefix("experiment"))))
            if progress_callback is not None:
                progress_callback({
                    "stage": "mdevent_import", "iteration": index, "total": len(names),
                    "message": f"reading MDEvent run metadata {index:,}/{len(names):,}",
                })
        runs = tuple(runs_list)
        lattice_group = workspace[f"experiment{runs[0].experiment_index}/sample/oriented_lattice"]
        ub = np.asarray(lattice_group["orientation_matrix"][()], dtype=float)
        lattice = {
            name: float(_scalar(lattice_group[f"unit_cell_{name}"]))
            for name in ("a", "b", "c", "alpha", "beta", "gamma")
        }
        return MDEventWorkspaceInfo(
            source, workspace_path, int(workspace["event_data/event_data"].shape[0]),
            dimensions, runs, ub, lattice,
        )


def load_detector_normalization(path: str | Path) -> DetectorNormalization:
    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        entries = [group for group in handle.values() if "workspace" in group and "instrument/detector/detector_list" in group]
        if not entries:
            raise ValueError("normalization file does not contain a Mantid MatrixWorkspace")
        entry = entries[0]
        return DetectorNormalization(
            source,
            np.asarray(entry["instrument/detector/detector_list"][()], dtype=np.int64),
            np.asarray(entry["workspace/values"][:, 0], dtype=float),
            np.asarray(entry["workspace/errors"][:, 0], dtype=float),
        )


def mdevent_dataset_group(
    path: str | Path,
    *,
    normalization_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    name: str | None = None,
    progress_callback: Any | None = None,
) -> DatasetGroup:
    """Create lightweight run entries sharing one MDEvent source and setup."""

    info = inspect_mdevent_workspace(path, progress_callback=progress_callback)
    shared = {
        "format": "mantid-mdevent",
        "source_files": [str(info.path)],
        "workspace_path": info.workspace_path,
        "event_count": info.event_count,
        "dimensions": list(info.dimensions),
        "hkl_bounds": _hkl_bounds(info.dimensions, info.ub_matrix),
        "ub_matrix": info.ub_matrix.tolist(),
        "lattice_parameters": info.lattice_parameters,
        "normalization_file": None if normalization_path is None else str(normalization_path),
        "mask_file": None if mask_path is None else str(mask_path),
        "incident_energy_override": None,
        "t0_override": None,
        "coordinate_frame": "HKL",
        "normalization": "proton_charge_and_detector_trajectory",
    }
    datasets = []
    for run in info.runs:
        datasets.append(DatasetEntry(
            name=f"run {run.run_number}", data=None, kind="mdevent",
            data_type="single_crystal_inelastic",
            metadata={
                "source_file": str(info.path), "import_status": "pending",
                "mdevent_experiment_index": run.experiment_index,
                "run_number": run.run_number, "omega": run.omega,
                "incident_energy": run.incident_energy,
                "proton_charge": run.proton_charge, "duration": run.duration,
            },
        ))
    return DatasetGroup(name=name or info.path.stem, datasets=datasets, metadata={"mdevent": shared})


def append_mdevent_file(group: DatasetGroup, path: str | Path) -> list[DatasetEntry]:
    """Append runs from another MDEvent file while retaining shared setup once."""

    config = group.metadata.get("mdevent")
    if not isinstance(config, dict):
        raise ValueError("target is not an MDEvent dataset group")
    info = inspect_mdevent_workspace(path)
    configured_ub = np.asarray(config["ub_matrix"], dtype=float)
    if not np.allclose(configured_ub, info.ub_matrix, rtol=1e-8, atol=1e-10):
        raise ValueError("MDEvent files must share the configured sample orientation")
    source = str(info.path)
    if source not in config["source_files"]:
        config["source_files"].append(source)
    existing = {dataset.name for dataset in group.datasets}
    added = []
    for run in info.runs:
        base = f"run {run.run_number}"
        name = base
        suffix = 2
        while name in existing:
            name = f"{base} ({suffix})"
            suffix += 1
        existing.add(name)
        entry = DatasetEntry(
            name=name, data=None, kind="mdevent", data_type="single_crystal_inelastic",
            metadata={
                "source_file": source, "import_status": "pending",
                "mdevent_experiment_index": run.experiment_index,
                "run_number": run.run_number, "omega": run.omega,
                "incident_energy": run.incident_energy,
                "proton_charge": run.proton_charge, "duration": run.duration,
            },
        )
        group.datasets.append(entry)
        added.append(entry)
    return added


def load_mdevent_run_points(
    dataset: DatasetEntry,
    *,
    batch_size: int = 1_000_000,
    progress_callback: Any | None = None,
) -> PointData4D:
    """Load one logical run by scanning shared event chunks once."""

    import h5py

    source = Path(dataset.metadata["source_file"])
    experiment_index = int(dataset.metadata["mdevent_experiment_index"])
    info = inspect_mdevent_workspace(source)
    ub = np.asarray(info.ub_matrix, dtype=float)
    transform = np.linalg.inv(2.0 * np.pi * ub)
    pieces: list[np.ndarray] = []
    with h5py.File(source, "r") as handle:
        events = handle[f"{info.workspace_path.strip('/')}/event_data/event_data"]
        total = int(events.shape[0])
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "mdevent_scan",
                    "iteration": 0,
                    "total": total,
                    "message": f"reading MDEvents for run {dataset.metadata.get('run_number', '')}".rstrip(),
                }
            )
        for start in range(0, events.shape[0], batch_size):
            stop = min(start + batch_size, events.shape[0])
            block = np.asarray(events[start:stop, :], dtype=float)
            selected = block[:, EVENT_COLUMNS["experiment_index"]].astype(np.int64) == experiment_index
            if np.any(selected):
                pieces.append(block[selected])
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "mdevent_scan",
                        "iteration": int(stop),
                        "total": total,
                        "message": f"read {int(stop):,}/{total:,} MDEvents",
                    }
                )
    events_array = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 9), dtype=float)
    hkl = events_array[:, 5:8] @ transform.T
    signal = events_array[:, 0]
    sigma = np.sqrt(np.maximum(events_array[:, 1], 0.0))
    return PointData4D(
        H=hkl[:, 0], K=hkl[:, 1], L=hkl[:, 2], E=events_array[:, 8],
        intensity=signal, sigma=sigma,
        metadata={
            "source_file": str(source), "run_number": dataset.metadata.get("run_number"),
            "experiment_index": experiment_index, "coordinate_units": "r.l.u.",
            "energy_units": "meV", "ub_matrix": ub.tolist(),
            "proton_charge": dataset.metadata.get("proton_charge"),
        },
    )


def bin_mdevent_group(
    group: DatasetGroup,
    *,
    lower: Iterable[float], upper: Iterable[float], num_bins: Iterable[int],
    step_size: Iterable[float] | None = None,
    bin_edges: Iterable[Iterable[float] | None] | None = None,
    minimum_samples: float = 0.0,
    run_indices: Iterable[int] | None = None,
    datasets: Iterable[DatasetEntry] | None = None,
    vectors: Iterable[Iterable[float]] | None = None,
    axis_names: Iterable[str] | None = None,
    max_batch_bytes: int = 192 * 1024 * 1024,
    enforce_memory_limit: bool = True,
    progress_callback: Any | None = None,
    symmetry_operations: Iterable[np.ndarray] | None = None,
) -> MDHistoData:
    """Locally bin event data and a detector-trajectory MDNorm denominator."""

    import h5py

    config = group.metadata["mdevent"]
    selected_runs = list(
        datasets if datasets is not None
        else (group.datasets if run_indices is None else (group.datasets[i] for i in run_indices))
    )
    lower_array = np.asarray(tuple(lower), dtype=float)
    upper_array = np.asarray(tuple(upper), dtype=float)
    bins_array = np.asarray(tuple(num_bins), dtype=int)
    if lower_array.shape != (4,) or upper_array.shape != (4,) or bins_array.shape != (4,):
        raise ValueError("MDEvent HKLE binning requires four lower, upper, and bin-count values")
    edges = _requested_edges(
        lower_array, upper_array, bins_array, step_size, bin_edges=bin_edges
    )
    minimum_samples = _validated_minimum_samples(minimum_samples)
    shape = tuple(int(axis_edges.size - 1) for axis_edges in edges)
    if enforce_memory_limit:
        _validate_mdevent_memory(shape, max_batch_bytes=max_batch_bytes)
    data_sum = np.zeros(shape)
    variance_sum = np.zeros(shape)
    event_count = np.zeros(shape)
    ub = np.asarray(config["ub_matrix"], dtype=float)
    basis = np.eye(4) if vectors is None else np.asarray(tuple(tuple(row) for row in vectors), dtype=float)
    if basis.shape != (4, 4) or not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("MDEvent coordinate axes must form a finite, linearly independent 4D basis")
    if np.any(basis[:3, 3] != 0.0) or np.any(basis[3, :3] != 0.0) or basis[3, 3] != 1.0:
        raise ValueError("MDEvent momentum axes cannot mix energy; the energy axis must be [0, 0, 0, 1]")
    basis_inverse = np.linalg.inv(basis)
    symmetry = _symmetry_matrices(symmetry_operations)
    names = (
        tuple(str(name) for name in axis_names)
        if axis_names is not None
        else tuple(_mdevent_axis_name(row, index) for index, row in enumerate(basis))
    )
    if len(names) != 4:
        raise ValueError("MDEvent HKLE binning requires four axis names")
    transform = np.linalg.inv(2.0 * np.pi * ub)
    sources = sorted({str(dataset.metadata["source_file"]) for dataset in selected_runs})
    source_sizes = {}
    for source_text in sources:
        with h5py.File(source_text, "r") as handle:
            source_sizes[source_text] = int(handle[f"{config['workspace_path'].strip('/')}/event_data/event_data"].shape[0])
    scan_total = sum(source_sizes.values())
    contribution_total = scan_total * len(symmetry)
    processed = 0
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_events",
                "iteration": 0,
                "total": contribution_total,
                "message": (
                    f"binning {scan_total:,} MDEvents across {len(symmetry):,} symmetry operations"
                    if len(symmetry) > 1
                    else f"binning {scan_total:,} MDEvents"
                ),
            }
        )
    data_sum_flat = data_sum.ravel()
    variance_sum_flat = variance_sum.ravel()
    event_count_flat = event_count.ravel()
    for source_text in sources:
        source = Path(source_text)
        source_runs = [dataset for dataset in selected_runs if str(dataset.metadata["source_file"]) == source_text]
        wanted = {int(dataset.metadata["mdevent_experiment_index"]): dataset for dataset in source_runs}
        with h5py.File(source, "r") as handle:
            values = handle[f"{config['workspace_path'].strip('/')}/event_data/event_data"]
            mask_norm = load_detector_normalization(config["mask_file"]) if config.get("mask_file") else None
            batch_rows = max(1, min(1_000_000, int(max_batch_bytes) // (9 * 8 * 3)))
            for start in range(0, values.shape[0], batch_rows):
                stop = min(start + batch_rows, values.shape[0])
                block = np.asarray(values[start:stop, :], dtype=float)
                keep = np.isin(block[:, 2].astype(np.int64), list(wanted))
                if np.any(keep):
                    chosen = block[keep]
                    if mask_norm is not None:
                        detector_values = mask_norm.value_for_ids(chosen[:, 4].astype(np.int64))
                        chosen = chosen[detector_values > 0.0]
                    if chosen.size:
                        hkl = chosen[:, 5:8] @ transform.T
                        for operation in symmetry:
                            transformed_hkl = hkl @ operation.T
                            coords = np.column_stack((transformed_hkl, chosen[:, 8])) @ basis_inverse
                            flat = _flat_bin_indices(coords, edges, shape)
                            valid = flat >= 0
                            np.add.at(data_sum_flat, flat[valid], chosen[valid, 0])
                            np.add.at(variance_sum_flat, flat[valid], chosen[valid, 1])
                            np.add.at(event_count_flat, flat[valid], 1.0)
                processed += (stop - start) * len(symmetry)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "mdevent_events",
                            "iteration": processed,
                            "total": contribution_total,
                            "message": (
                                f"binned {processed:,}/{contribution_total:,} "
                                "symmetry-expanded MDEvent contributions"
                                if len(symmetry) > 1
                                else f"binned {processed:,}/{contribution_total:,} MDEvents"
                            ),
                        }
                    )
    normalization = _trajectory_normalization(
        group,
        selected_runs,
        edges,
        shape,
        basis_inverse,
        symmetry,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_finalize",
                "iteration": 1,
                "total": 3,
                "message": "normalizing MDEvent signal and uncertainties",
            }
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = data_sum / normalization
        errors = np.sqrt(variance_sum) / normalization
    covered_zero = (normalization > 0.0) & (event_count == 0.0)
    total_events = float(np.sum(event_count))
    event_weight_rms = (
        float(np.sqrt(np.sum(variance_sum) / total_events)) if total_events > 0.0 else 1.0
    )
    errors[covered_zero] = (
        FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER
        * event_weight_rms
        / normalization[covered_zero]
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_finalize",
                "iteration": 2,
                "total": 3,
                "message": "building MDEvent masks and output channels",
            }
        )
    mask = ~(np.isfinite(signal) & np.isfinite(errors) & (normalization > 0.0))
    mask |= event_count < minimum_samples
    axes = tuple(
        MDHistoAxis(name, edge, units, kind, frame="HKL" if index < 3 else "General Frame")
        for index, (name, edge, units, kind) in enumerate(zip(
            names, edges,
            ("r.l.u.", "r.l.u.", "r.l.u.", "meV"),
            ("momentum", "momentum", "momentum", "energy"), strict=True,
        ))
    )
    result = MDHistoData(
        axes=axes, signal=signal, errors=errors, mask=mask, num_events=event_count,
        metadata={
            "mdevent": config,
            "lattice_parameters": dict(config.get("lattice_parameters", {})),
            "ub_matrix": config.get("ub_matrix"),
            "rebin": {
                "vectors": basis.tolist(),
                "bin_edges": [edge.tolist() for edge in edges],
                "minimum_samples": minimum_samples,
            },
            "signal_semantics": "density",
            "signal_semantics_source": "nfit_mdevent_reduction",
            "normalization_denominator": normalization,
            "zero_event_bins_are_measured": True,
            "zero_count_error_model": "feldman_cousins_68_percent_upper_limit_scaled_by_rms_event_weight",
            "event_weight_rms": event_weight_rms,
            "symmetry_operations_hkl": [operation.tolist() for operation in symmetry],
        },
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(
                normalization,
                label="Detector-trajectory normalization",
                unit="arbitrary normalization units",
            )
        },
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_finalize",
                "iteration": 3,
                "total": 3,
                "message": "MDEvent reduction complete",
            }
        )
    return result


def bin_mdevent_powder_group(
    group: DatasetGroup,
    *,
    lower: Iterable[float],
    upper: Iterable[float],
    num_bins: Iterable[int],
    step_size: Iterable[float] | None = None,
    bin_edges: Iterable[Iterable[float] | None] | None = None,
    minimum_samples: float = 0.0,
    run_indices: Iterable[int] | None = None,
    datasets: Iterable[DatasetEntry] | None = None,
    max_batch_bytes: int = 192 * 1024 * 1024,
    progress_callback: Any | None = None,
) -> MDHistoData:
    """Reduce MDEvents directly onto a powder ``|Q|, DeltaE`` grid.

    The numerator is accumulated from event weights.  The denominator follows
    the same proton-charge and detector-trajectory normalization as the HKLE
    reducer, but radial momentum is independent of the sample goniometer.
    """

    import h5py

    config = group.metadata["mdevent"]
    selected_runs = list(
        datasets
        if datasets is not None
        else (
            group.datasets
            if run_indices is None
            else (group.datasets[index] for index in run_indices)
        )
    )
    lower_array = np.asarray(tuple(lower), dtype=float)
    upper_array = np.asarray(tuple(upper), dtype=float)
    bins_array = np.asarray(tuple(num_bins), dtype=int)
    if lower_array.shape != (2,) or upper_array.shape != (2,) or bins_array.shape != (2,):
        raise ValueError("MDEvent powder binning requires |Q| and energy bounds")
    if lower_array[0] < 0.0:
        raise ValueError("powder |Q| lower bound must be nonnegative")
    edges = _requested_edges(
        lower_array, upper_array, bins_array, step_size, bin_edges=bin_edges
    )
    minimum_samples = _validated_minimum_samples(minimum_samples)
    shape = tuple(int(axis_edges.size - 1) for axis_edges in edges)
    data_sum = np.zeros(shape)
    variance_sum = np.zeros(shape)
    event_count = np.zeros(shape)
    sources = sorted(
        {str(dataset.metadata["source_file"]) for dataset in selected_runs}
    )
    source_sizes: dict[str, int] = {}
    for source_text in sources:
        with h5py.File(source_text, "r") as handle:
            source_sizes[source_text] = int(
                handle[
                    f"{config['workspace_path'].strip('/')}/event_data/event_data"
                ].shape[0]
            )
    scan_total = sum(source_sizes.values())
    processed = 0
    for source_text in sources:
        source_runs = [
            dataset
            for dataset in selected_runs
            if str(dataset.metadata["source_file"]) == source_text
        ]
        wanted = {
            int(dataset.metadata["mdevent_experiment_index"]): dataset
            for dataset in source_runs
        }
        with h5py.File(source_text, "r") as handle:
            values = handle[
                f"{config['workspace_path'].strip('/')}/event_data/event_data"
            ]
            mask_norm = (
                load_detector_normalization(config["mask_file"])
                if config.get("mask_file")
                else None
            )
            batch_rows = max(
                1, min(1_000_000, int(max_batch_bytes) // (9 * 8 * 3))
            )
            for start in range(0, values.shape[0], batch_rows):
                stop = min(start + batch_rows, values.shape[0])
                block = np.asarray(values[start:stop, :], dtype=float)
                keep = np.isin(
                    block[:, EVENT_COLUMNS["experiment_index"]].astype(np.int64),
                    list(wanted),
                )
                if np.any(keep):
                    chosen = block[keep]
                    if mask_norm is not None:
                        detector_values = mask_norm.value_for_ids(
                            chosen[:, EVENT_COLUMNS["detector_id"]].astype(np.int64)
                        )
                        chosen = chosen[detector_values > 0.0]
                    if chosen.size:
                        q_modulus = np.linalg.norm(chosen[:, 5:8], axis=1)
                        coordinates = np.column_stack((q_modulus, chosen[:, 8]))
                        flat = _flat_bin_indices(coordinates, edges, shape)
                        valid = flat >= 0
                        data_sum.ravel()[:] += np.bincount(
                            flat[valid],
                            weights=chosen[valid, EVENT_COLUMNS["signal"]],
                            minlength=data_sum.size,
                        )
                        variance_sum.ravel()[:] += np.bincount(
                            flat[valid],
                            weights=chosen[valid, EVENT_COLUMNS["error_squared"]],
                            minlength=data_sum.size,
                        )
                        event_count.ravel()[:] += np.bincount(
                            flat[valid], minlength=data_sum.size
                        )
                processed += stop - start
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "mdevent_scan",
                            "iteration": processed,
                            "total": scan_total + 1,
                            "message": f"reading MDEvents {processed:,}/{scan_total:,}",
                        }
                    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_normalization",
                "iteration": scan_total,
                "total": scan_total + 1,
                "message": "calculating powder detector normalization",
            }
        )
    normalization = _powder_trajectory_normalization(
        group,
        selected_runs,
        edges,
        shape,
        progress_callback=progress_callback,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = data_sum / normalization
        errors = np.sqrt(variance_sum) / normalization
    covered_zero = (normalization > 0.0) & (event_count == 0.0)
    total_events = float(np.sum(event_count))
    event_weight_rms = (
        float(np.sqrt(np.sum(variance_sum) / total_events))
        if total_events > 0.0
        else 1.0
    )
    errors[covered_zero] = (
        FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER
        * event_weight_rms
        / normalization[covered_zero]
    )
    mask = ~(np.isfinite(signal) & np.isfinite(errors) & (normalization > 0.0))
    mask |= event_count < minimum_samples
    axes = (
        MDHistoAxis("|Q|", edges[0], "1/angstrom", "momentum", frame="Q modulus"),
        MDHistoAxis("DeltaE", edges[1], "meV", "energy", frame="General Frame"),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_normalization",
                "iteration": scan_total + 1,
                "total": scan_total + 1,
                "message": "MDEvent powder reduction complete",
            }
        )
    return MDHistoData(
        axes=axes,
        signal=signal,
        errors=errors,
        mask=mask,
        num_events=event_count,
        metadata={
            "mdevent": config,
            "lattice_parameters": dict(config.get("lattice_parameters", {})),
            "ub_matrix": config.get("ub_matrix"),
            "signal_semantics": "density",
            "signal_semantics_source": "nfit_mdevent_powder_reduction",
            "normalization_denominator": normalization,
            "zero_event_bins_are_measured": True,
            "zero_count_error_model": "feldman_cousins_68_percent_upper_limit_scaled_by_rms_event_weight",
            "event_weight_rms": event_weight_rms,
            "powder_reduction": {
                "coordinates": "|Q|,DeltaE",
                "normalization": "proton_charge_and_detector_trajectory",
            },
            "rebin": {
                "bin_edges": [edge.tolist() for edge in edges],
                "minimum_samples": minimum_samples,
            },
        },
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(
                normalization,
                label="Detector-trajectory normalization",
                unit="arbitrary normalization units",
            )
        },
    )


def _validated_minimum_samples(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("minimum_samples must be finite and nonnegative")
    return result


def _requested_edges(lower, upper, num_bins, step_size=None, *, bin_edges=None):
    from .rebin import _uniform_center_edges

    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper < lower):
        raise ValueError("binning bounds must be finite and increasing")
    if np.any(num_bins < 1):
        raise ValueError("bin counts must be positive")
    if step_size is None:
        uniform_edges = [
            _uniform_center_edges(lo, hi, count=count)
            for lo, hi, count in zip(lower, upper, num_bins, strict=True)
        ]
    else:
        steps = np.asarray(tuple(step_size), dtype=float)
        if steps.shape != lower.shape or np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
            raise ValueError("step sizes must be positive and match the requested dimensions")
        uniform_edges = []
        for lo, hi, step in zip(lower, upper, steps, strict=True):
            uniform_edges.append(_uniform_center_edges(lo, hi, step=step))
    if bin_edges is None:
        return uniform_edges
    explicit = list(bin_edges)
    if len(explicit) != lower.size:
        raise ValueError("bin_edges must contain one entry per requested dimension")
    result = []
    for fallback, values in zip(uniform_edges, explicit, strict=True):
        if values is None:
            result.append(fallback)
            continue
        axis_edges = np.asarray(tuple(values), dtype=float)
        if (
            axis_edges.ndim != 1
            or axis_edges.size < 2
            or np.any(~np.isfinite(axis_edges))
            or np.any(np.diff(axis_edges) <= 0.0)
        ):
            raise ValueError("explicit bin edges must be finite and strictly increasing")
        result.append(axis_edges)
    return result


def estimate_mdevent_peak_memory(num_bins, *, max_batch_bytes=192 * 1024 * 1024):
    """Conservative peak-memory estimate for native event reduction."""

    output_bins = math.prod(int(value) for value in num_bins)
    return MDEVENT_FIXED_MEMORY_BYTES + output_bins * MDEVENT_BYTES_PER_OUTPUT_BIN + max(int(max_batch_bytes), 1)


def assess_mdevent_memory(num_bins, *, max_batch_bytes=192 * 1024 * 1024):
    """Return estimated peak, available memory, and whether confirmation is advised."""

    estimate = estimate_mdevent_peak_memory(num_bins, max_batch_bytes=max_batch_bytes)
    available = _available_memory_bytes()
    return estimate, available, available is not None and estimate > int(available * 0.5)


def _validate_mdevent_memory(shape, *, max_batch_bytes):
    estimate = estimate_mdevent_peak_memory(shape, max_batch_bytes=max_batch_bytes)
    available = _available_memory_bytes()
    if available is not None and estimate > int(available * 0.7):
        bins_text = " x ".join(str(value) for value in shape)
        raise MemoryError(
            f"requested MDEvent grid {bins_text} ({math.prod(shape):,} bins) is estimated to peak at "
            f"{estimate / 1024**3:.1f} GB, exceeding 70% of the currently available "
            f"{available / 1024**3:.1f} GB. Reduce one or more bin counts, narrow the limits, or use a larger-memory machine."
        )


def _available_memory_bytes():
    from .performance import available_memory_bytes

    return available_memory_bytes()


def _trajectory_worker_count(output_size: int) -> int:
    """Resolve the saved CPU ceiling against trajectory-buffer memory."""

    from .performance import transient_rebin_memory_limit_bytes

    output_bytes = max(int(output_size) * 8, 1)
    available_memory = _available_memory_bytes()
    partial_budget = (
        512 * 1024**2
        if available_memory is None
        else max(transient_rebin_memory_limit_bytes(available_memory), output_bytes)
    )
    memory_workers = max(1, partial_budget // output_bytes)
    return min(_parallel.num_threads(), memory_workers)


def _trajectory_normalization(
    group,
    datasets,
    edges,
    shape,
    basis_inverse,
    symmetry_operations=None,
    *,
    progress_callback=None,
):
    import h5py

    config = group.metadata["mdevent"]
    detector_norm = load_detector_normalization(config["normalization_file"]) if config.get("normalization_file") else None
    detector_mask = load_detector_normalization(config["mask_file"]) if config.get("mask_file") else None
    result = np.zeros(shape)
    symmetry = _symmetry_matrices(symmetry_operations)
    run_payloads = []
    detector_payloads = []
    datasets = list(datasets)
    dataset_total = len(datasets)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_normalization_setup",
                "iteration": 0,
                "total": dataset_total,
                "message": f"preparing detector trajectories for {dataset_total:,} runs",
            }
        )
    prepared = 0
    report_stride = max(dataset_total // 100, 1)
    for source_text in sorted({str(dataset.metadata["source_file"]) for dataset in datasets}):
        source_datasets = [dataset for dataset in datasets if str(dataset.metadata["source_file"]) == source_text]
        with h5py.File(source_text, "r") as handle:
            workspace = handle[config["workspace_path"]]
            for dataset in source_datasets:
                index = int(dataset.metadata["mdevent_experiment_index"])
                experiment = workspace[f"experiment{index}"]
                detector_ids = np.asarray(experiment["instrument/physical_detectors/detector_number"][()], dtype=np.int64)
                theta = np.deg2rad(np.asarray(experiment["instrument/physical_detectors/polar_angle"][()], dtype=float))
                phi = np.deg2rad(np.asarray(experiment["instrument/physical_detectors/azimuthal_angle"][()], dtype=float))
                solid = np.ones(detector_ids.size) if detector_norm is None else detector_norm.value_for_ids(detector_ids)
                if detector_mask is not None:
                    solid[detector_mask.value_for_ids(detector_ids) <= 0.0] = 0.0
                charge = float(dataset.metadata["proton_charge"])
                ei = config.get("incident_energy_override") or float(dataset.metadata["incident_energy"])
                original_bounds = np.asarray(experiment["logs/processed_histogram_bins/value"][()], dtype=float)
                gonio = _read_goniometer_matrix(experiment)
                ub = np.asarray(config["ub_matrix"], dtype=float)
                canonical_inverse = np.linalg.inv(gonio @ (2.0 * np.pi * ub))
                inverses = [basis_inverse[:3, :3].T @ operation @ canonical_inverse for operation in symmetry]
                current_detector_payload = (detector_ids, theta, phi, solid)
                geometry_index = next(
                    (
                        payload_index
                        for payload_index, payload in enumerate(detector_payloads)
                        if all(
                            first.shape == current.shape
                            and np.array_equal(first, current)
                            for first, current in zip(
                                payload,
                                current_detector_payload,
                                strict=True,
                            )
                        )
                    ),
                    None,
                )
                if geometry_index is None:
                    geometry_index = len(detector_payloads)
                    detector_payloads.append(current_detector_payload)
                run_payloads.extend(
                    (inverse, ei, original_bounds, charge, geometry_index)
                    for inverse in inverses
                )
                prepared += 1
                if progress_callback is not None and (
                    prepared == dataset_total or prepared % report_stride == 0
                ):
                    progress_callback(
                        {
                            "stage": "mdevent_normalization_setup",
                            "iteration": prepared,
                            "total": dataset_total,
                            "message": (
                                f"prepared detector trajectories for "
                                f"{prepared:,}/{dataset_total:,} runs"
                            ),
                        }
                    )
    if (
        _MDEVENT_NUMBA is not None
        and detector_payloads
        and run_payloads
    ):
        output_size = int(np.prod(shape))
        workers = _trajectory_worker_count(output_size)
        grouped_payloads = [
            [payload for payload in run_payloads if payload[4] == geometry_index]
            for geometry_index in range(len(detector_payloads))
        ]
        task_total = sum(
            len(payloads) * int(detector_payloads[index][1].size)
            for index, payloads in enumerate(grouped_payloads)
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "mdevent_normalization",
                    "iteration": 0,
                    "total": task_total,
                    "message": (
                        f"integrating {task_total:,} detector trajectories "
                        f"with {workers:,} CPU{'s' if workers != 1 else ''}"
                    ),
                    "workers": workers,
                    "output_bins": output_size,
                }
            )
        completed = 0
        for geometry_index, payloads in enumerate(grouped_payloads):
            _, theta, phi, solid = detector_payloads[geometry_index]
            detector_count = int(theta.size)
            payload_batch = max(1, 32_000_000 // max(detector_count, 1))
            for start in range(0, len(payloads), payload_batch):
                batch = payloads[start : start + payload_batch]
                flat = _MDEVENT_NUMBA.run_trajectory_normalization(
                    theta,
                    phi,
                    solid,
                    np.asarray([payload[0] for payload in batch]),
                    np.asarray([payload[1] for payload in batch]),
                    np.asarray([payload[2] for payload in batch]),
                    np.asarray([payload[3] for payload in batch]),
                    *[np.asarray(edge, dtype=float) for edge in edges],
                    np.asarray(shape, dtype=np.int64),
                    workers=workers,
                )
                result += np.asarray(flat, dtype=float).reshape(shape)
                completed += len(batch) * detector_count
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "mdevent_normalization",
                            "iteration": completed,
                            "total": task_total,
                            "message": (
                                f"integrated {completed:,}/{task_total:,} "
                                "detector trajectories"
                            ),
                            "workers": workers,
                            "output_bins": output_size,
                        }
                    )
        return result
    task_total = len(run_payloads)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_normalization",
                "iteration": 0,
                "total": task_total,
                "message": f"integrating detector trajectories for {task_total:,} run transforms",
                "workers": 1,
                "output_bins": int(np.prod(shape)),
            }
        )
    report_stride = max(task_total // 100, 1)
    for payload_index, (inverse, ei, original_bounds, charge, geometry_index) in enumerate(
        run_payloads,
        start=1,
    ):
        _, theta, phi, solid = detector_payloads[geometry_index]
        direction = np.column_stack(
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
        )
        for detector_index in np.flatnonzero(solid > 0.0):
            _accumulate_detector_trajectory(
                result,
                edges,
                inverse,
                direction[detector_index],
                ei,
                original_bounds,
                charge * solid[detector_index],
            )
        if progress_callback is not None and (
            payload_index == task_total or payload_index % report_stride == 0
        ):
            progress_callback(
                {
                    "stage": "mdevent_normalization",
                    "iteration": payload_index,
                    "total": task_total,
                    "message": (
                        f"integrated detector trajectories for "
                        f"{payload_index:,}/{task_total:,} run transforms"
                    ),
                    "workers": 1,
                    "output_bins": int(np.prod(shape)),
                }
            )
    return result


def _powder_trajectory_normalization(
    group,
    datasets,
    edges,
    shape,
    *,
    progress_callback: Any | None = None,
):
    """Accumulate the radial detector-trajectory denominator."""

    import h5py

    config = group.metadata["mdevent"]
    detector_norm = (
        load_detector_normalization(config["normalization_file"])
        if config.get("normalization_file")
        else None
    )
    detector_mask = (
        load_detector_normalization(config["mask_file"])
        if config.get("mask_file")
        else None
    )
    result = np.zeros(shape)
    payloads = []
    for source_text in sorted(
        {str(dataset.metadata["source_file"]) for dataset in datasets}
    ):
        source_datasets = [
            dataset
            for dataset in datasets
            if str(dataset.metadata["source_file"]) == source_text
        ]
        with h5py.File(source_text, "r") as handle:
            workspace = handle[config["workspace_path"]]
            for dataset in source_datasets:
                index = int(dataset.metadata["mdevent_experiment_index"])
                experiment = workspace[f"experiment{index}"]
                detector_ids = np.asarray(
                    experiment["instrument/physical_detectors/detector_number"][()],
                    dtype=np.int64,
                )
                theta = np.deg2rad(
                    np.asarray(
                        experiment["instrument/physical_detectors/polar_angle"][()],
                        dtype=float,
                    )
                )
                solid = (
                    np.ones(detector_ids.size)
                    if detector_norm is None
                    else detector_norm.value_for_ids(detector_ids)
                )
                if detector_mask is not None:
                    solid[detector_mask.value_for_ids(detector_ids) <= 0.0] = 0.0
                charge = float(dataset.metadata["proton_charge"])
                incident_energy = config.get("incident_energy_override") or float(
                    dataset.metadata["incident_energy"]
                )
                energy_bounds = np.asarray(
                    experiment["logs/processed_histogram_bins/value"][()], dtype=float
                )
                payloads.append((theta, solid, incident_energy, energy_bounds, charge))

    trajectory_total = sum(
        int(np.count_nonzero(solid > 0.0))
        for _theta, solid, _incident_energy, _energy_bounds, _charge in payloads
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "mdevent_normalization",
                "iteration": 0,
                "total": trajectory_total,
                "message": (
                    f"integrating {trajectory_total:,} powder detector trajectories"
                ),
                "output_bins": int(np.prod(shape)),
            }
        )
    completed = 0
    report_stride = max(trajectory_total // 100, 1)
    for theta, solid, incident_energy, energy_bounds, charge in payloads:
        for detector_index in np.flatnonzero(solid > 0.0):
            _accumulate_powder_detector_trajectory(
                result,
                edges,
                float(theta[detector_index]),
                float(incident_energy),
                energy_bounds,
                charge * float(solid[detector_index]),
            )
            completed += 1
            if progress_callback is not None and (
                completed == trajectory_total or completed % report_stride == 0
            ):
                progress_callback(
                    {
                        "stage": "mdevent_normalization",
                        "iteration": completed,
                        "total": trajectory_total,
                        "message": (
                            f"integrated {completed:,}/{trajectory_total:,} "
                            "powder detector trajectories"
                        ),
                        "output_bins": int(np.prod(shape)),
                    }
                )
    return result


def _symmetry_matrices(operations) -> tuple[np.ndarray, ...]:
    if operations is None:
        return (np.eye(3),)
    matrices = tuple(np.asarray(operation, dtype=float) for operation in operations)
    if not matrices:
        return (np.eye(3),)
    if any(matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("symmetry operations must be finite 3x3 HKL matrices")
    return matrices


def _mdevent_axis_name(vector, index):
    if index == 3:
        return "DeltaE"
    symbols = ("H", "K", "L")
    terms = []
    for coefficient, symbol in zip(vector[:3], symbols, strict=True):
        if coefficient == 0:
            terms.append("0")
        elif coefficient == 1:
            terms.append(symbol)
        elif coefficient == -1:
            terms.append(f"-{symbol}")
        else:
            terms.append(f"{coefficient:g}{symbol}")
    return f"[{','.join(terms)}]"


def _hkl_bounds(dimensions, ub):
    q_bounds = [(float(dimensions[i]["lower"]), float(dimensions[i]["upper"])) for i in range(3)]
    corners = np.asarray(list(product(*q_bounds)), dtype=float)
    hkl = corners @ np.linalg.inv(2.0 * np.pi * np.asarray(ub, dtype=float)).T
    bounds = [(float(np.min(hkl[:, i])), float(np.max(hkl[:, i]))) for i in range(3)]
    bounds.append((float(dimensions[3]["lower"]), float(dimensions[3]["upper"])))
    return bounds


def _accumulate_detector_trajectory(output, edges, inverse, direction, ei, energy_bounds, weight):
    ki = np.sqrt(max(ei, 0.0) / ENERGY_TO_K2)
    kf_values = np.sqrt(np.maximum(ei - np.asarray(energy_bounds, dtype=float), 0.0) / ENERGY_TO_K2)
    low_kf, high_kf = float(np.min(kf_values)), float(np.max(kf_values))
    qin = inverse @ np.array([0.0, 0.0, ki])
    qout = inverse @ direction
    intersections = [low_kf, high_kf]
    for dim in range(3):
        if abs(qout[dim]) > 1e-14:
            intersections.extend((qin[dim] - boundary) / qout[dim] for boundary in edges[dim])
    intersections.extend(np.sqrt(np.maximum(ei - edges[3], 0.0) / ENERGY_TO_K2))
    points = np.unique(np.clip(np.asarray(intersections), low_kf, high_kf))
    for first, second in zip(points[:-1], points[1:], strict=True):
        if second - first <= 1e-12:
            continue
        middle = 0.5 * (first + second)
        hkl = qin - qout * middle
        energy = ei - ENERGY_TO_K2 * middle * middle
        flat = _flat_bin_indices(np.asarray([[*hkl, energy]]), edges, output.shape)[0]
        if flat >= 0:
            output.ravel()[flat] += weight * ENERGY_TO_K2 * (second * second - first * first)


def _accumulate_powder_detector_trajectory(
    output, edges, scattering_angle, incident_energy, energy_bounds, weight
):
    """Integrate one detector trajectory through radial-Q/energy bins."""

    ki = np.sqrt(max(incident_energy, 0.0) / ENERGY_TO_K2)
    kf_values = np.sqrt(
        np.maximum(incident_energy - np.asarray(energy_bounds, dtype=float), 0.0)
        / ENERGY_TO_K2
    )
    low_kf, high_kf = float(np.min(kf_values)), float(np.max(kf_values))
    cosine = float(np.cos(scattering_angle))
    sine_squared = max(0.0, 1.0 - cosine * cosine)
    intersections = [low_kf, high_kf]
    # q^2 = ki^2 + kf^2 - 2 ki kf cos(theta).  Each radial
    # boundary can cross a detector trajectory twice.
    for q_boundary in edges[0]:
        discriminant = q_boundary * q_boundary - ki * ki * sine_squared
        if discriminant < 0.0:
            continue
        root = np.sqrt(discriminant)
        intersections.extend((ki * cosine - root, ki * cosine + root))
    intersections.extend(
        np.sqrt(np.maximum(incident_energy - edges[1], 0.0) / ENERGY_TO_K2)
    )
    points = np.unique(np.clip(np.asarray(intersections), low_kf, high_kf))
    for first, second in zip(points[:-1], points[1:], strict=True):
        if second - first <= 1.0e-12:
            continue
        middle = 0.5 * (first + second)
        q_modulus = np.sqrt(
            max(
                ki * ki + middle * middle - 2.0 * ki * middle * cosine,
                0.0,
            )
        )
        energy = incident_energy - ENERGY_TO_K2 * middle * middle
        flat = _flat_bin_indices(
            np.asarray([[q_modulus, energy]]), edges, output.shape
        )[0]
        if flat >= 0:
            output.ravel()[flat] += (
                weight * ENERGY_TO_K2 * (second * second - first * first)
            )


def _flat_bin_indices(coords, edges, shape):
    indices = []
    valid = np.ones(coords.shape[0], dtype=bool)
    for dim, edge in enumerate(edges):
        index = np.searchsorted(edge, coords[:, dim], side="right") - 1
        index[coords[:, dim] == edge[-1]] = len(edge) - 2
        valid &= (index >= 0) & (index < len(edge) - 1)
        indices.append(index)
    flat = np.full(coords.shape[0], -1, dtype=np.int64)
    flat[valid] = np.ravel_multi_index(tuple(index[valid] for index in indices), shape)
    return flat


def _read_run(group, index):
    def log(name, default=0.0):
        path = f"logs/{name}/value"
        return _scalar(group[path]) if path in group else default
    t0 = log("CalculatedT0", None)
    return MDEventRunInfo(
        index, str(log("run_number", index)), float(log("Ei")), float(log("gd_prtn_chrg")),
        float(log("omega")), float(log("phi")), float(log("chi")), float(log("duration")),
        None if t0 is None else float(t0),
        _read_goniometer_matrix(group),
        np.asarray(group["logs/RUBW_MATRIX/value"][()], dtype=float).reshape(3, 3),
    )


def _read_goniometer_matrix(experiment):
    """Read Mantid's matrix or reconstruct it from saved goniometer axes."""

    goniometer = experiment["logs/goniometer"]
    if "rotation_matrix" in goniometer:
        return np.asarray(goniometer["rotation_matrix"][()], dtype=float).reshape(3, 3)
    axes = sorted(
        (name for name in goniometer if re.fullmatch(r"axis\d+", name)),
        key=lambda name: int(name.removeprefix("axis")),
    )
    if not axes:
        raise ValueError("MDEvent experiment has no saved goniometer rotation")
    result = np.eye(3)
    for name in axes:
        axis_group = goniometer[name]
        axis = np.asarray(axis_group["rotationaxis"][()], dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        axis /= norm
        angle_dataset = axis_group["angle"]
        angle = float(_scalar(angle_dataset))
        unit = angle_dataset.attrs.get("unit", "deg")
        if isinstance(unit, bytes):
            unit = unit.decode(errors="replace")
        if str(unit).casefold().startswith("deg"):
            angle = np.deg2rad(angle)
        sense = angle_dataset.attrs.get("sense", "CCW")
        if isinstance(sense, bytes):
            sense = sense.decode(errors="replace")
        if str(sense).casefold() == "cw":
            angle = -angle
        cross = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3) * np.cos(angle)
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross
        )
        result = rotation @ result
    return result


def _parse_dimension(value):
    text = value.decode() if isinstance(value, bytes) else str(value)
    def field(name, default=""):
        match = re.search(fr"<{name}>(.*?)</{name}>", text)
        return match.group(1) if match else default
    id_match = re.search(r'<Dimension ID="([^"]+)"', text)
    return {
        "id": id_match.group(1) if id_match else "", "name": field("Name"), "units": field("Units"),
        "frame": field("Frame"), "lower": float(field("LowerBounds", "nan")),
        "upper": float(field("UpperBounds", "nan")), "bins": int(field("NumberOfBins", "1")),
    }


def _scalar(dataset):
    value = np.asarray(dataset[()]).reshape(-1)[0]
    return value.decode(errors="replace") if isinstance(value, (bytes, np.bytes_)) else value.item()
