"""Streaming direct-geometry TOF event reduction for raw NeXus files.

The implementation deliberately reads one bank (and then one event chunk) at a
time.  A raw SNS run can therefore be combined into an HKLE histogram without
materialising its event table, which is important for multi-run SEQUOIA data.
"""

from __future__ import annotations

import ast
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .mdevent import (
    ENERGY_TO_K2, FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER,
    _flat_bin_indices, load_detector_normalization,
)
from .mdhisto import MDHistoAxis, MDHistoData
from .pipeline import DatasetEntry, DatasetGroup

# Neutron velocity in m/s is 437.393 / wavelength(A); combined with
# E(meV)=2.072124855 k^2 this is the usual TOF conversion in us/m.
TOF_US_PER_M_SQRT_MEV = 252.777


@dataclass(frozen=True)
class RawDGSRunInfo:
    path: Path
    run_number: str
    event_count: int
    incident_energy: float
    proton_charge: float
    omega: float
    phi: float
    chi: float
    l1: float
    ub_matrix: np.ndarray


def is_raw_dgs_nexus_file(path: str | Path) -> bool:
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            entry = handle.get("entry")
            return entry is not None and any(
                name.startswith("bank") and name.endswith("_events") for name in entry
            )
    except (ImportError, OSError):
        return False


def inspect_raw_dgs_run(path: str | Path) -> RawDGSRunInfo:
    """Read lightweight run metadata needed to reduce a direct-geometry run."""

    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        entry = handle["entry"]
        event_count = sum(
            int(group["event_id"].shape[0]) for name, group in entry.items()
            if name.startswith("bank") and name.endswith("_events") and "event_id" in group
        )
        return RawDGSRunInfo(
            path=source,
            run_number=_text_scalar(entry.get("run_number"), source.stem),
            event_count=event_count,
            incident_energy=_log_value(entry, ("BL17:Det:TH:BL:Ei", "Ei", "EnergyRequest"), 0.0),
            proton_charge=_log_value(entry, ("proton_charge", "gd_prtn_chrg"), 1.0),
            omega=_log_value(entry, ("omega",), 0.0),
            phi=_log_value(entry, ("phi",), 0.0),
            chi=_log_value(entry, ("chi",), 0.0),
            l1=_source_distance(entry),
            ub_matrix=_ub_from_logs(entry),
        )


def raw_dgs_dataset_group(
    paths: Iterable[str | Path], *, normalization_path: str | Path | None = None,
    mask_path: str | Path | None = None, name: str | None = None,
    progress_callback: Any | None = None,
) -> DatasetGroup:
    """Create lightweight raw-run entries sharing reduction and sample setup."""

    infos = [inspect_raw_dgs_run(path) for path in paths]
    if not infos:
        raise ValueError("select at least one raw direct-geometry NeXus file")
    first = infos[0]
    shared = {
        "format": "raw-direct-geometry-nexus",
        "source_files": [str(info.path) for info in infos],
        "event_count": sum(info.event_count for info in infos),
        "ub_matrix": first.ub_matrix.tolist(),
        "normalization_file": None if normalization_path is None else str(normalization_path),
        "mask_file": None if mask_path is None else str(mask_path),
        "incident_energy_override": None,
        "t0_override": None,
        "bad_pulse_threshold": 0.95,
        "normalization": "proton_charge_and_detector_vanadium",
    }
    datasets = []
    for index, info in enumerate(infos, start=1):
        datasets.append(DatasetEntry(
            name=f"run {info.run_number}", data=None, kind="raw_dgs_nexus",
            data_type="single_crystal_inelastic",
            metadata={"source_file": str(info.path), "run_number": info.run_number,
                      "event_count": info.event_count, "incident_energy": info.incident_energy,
                      "proton_charge": info.proton_charge, "omega": info.omega,
                      "phi": info.phi, "chi": info.chi, "l1": info.l1,
                      "import_status": "pending"},
        ))
        if progress_callback is not None:
            progress_callback({"stage": "raw_dgs_import", "iteration": index, "total": len(infos),
                               "message": f"reading raw run metadata {index}/{len(infos)}"})
    return DatasetGroup(name=name or first.path.stem, datasets=datasets, metadata={"raw_dgs": shared})


def bin_raw_dgs_group(
    group: DatasetGroup, *, lower: Iterable[float], upper: Iterable[float],
    num_bins: Iterable[int], step_size: Iterable[float] | None = None,
    datasets: Iterable[DatasetEntry] | None = None, vectors: Iterable[Iterable[float]] | None = None,
    axis_names: Iterable[str] | None = None, max_batch_bytes: int = 192 * 1024 * 1024,
    progress_callback: Any | None = None,
) -> MDHistoData:
    """Reduce raw direct-geometry event banks into an HKLE histogram.

    Signal is normalized by run proton charge and optional per-detector
    vanadium values.  Invalid/zero vanadium values and mask-file values exclude
    the detector from both numerator and coverage.  This first native pathway
    uses the event-space coverage seen in the requested histogram, rather than
    constructing Mantid's separate MDNorm trajectory denominator.
    """

    config = group.metadata["raw_dgs"]
    selected = list(group.datasets if datasets is None else datasets)
    lo, hi, bins = (np.asarray(tuple(values), dtype=dtype) for values, dtype in
                    ((lower, float), (upper, float), (num_bins, int)))
    if lo.shape != (4,) or hi.shape != (4,) or bins.shape != (4,) or np.any(bins <= 0):
        raise ValueError("raw direct-geometry HKLE binning requires four positive bin counts")
    if step_size is None:
        edges = [np.linspace(a, b, n + 1) for a, b, n in zip(lo, hi, bins, strict=True)]
    else:
        steps = np.asarray(tuple(step_size), dtype=float)
        if steps.shape != (4,) or np.any(steps <= 0.0):
            raise ValueError("raw direct-geometry step sizes must contain four positive values")
        edges = [np.append(np.arange(a, b, step), b) for a, b, step in zip(lo, hi, steps, strict=True)]
    shape = tuple(edge.size - 1 for edge in edges)
    basis = np.eye(4) if vectors is None else np.asarray(tuple(tuple(v) for v in vectors), dtype=float)
    if basis.shape != (4, 4) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("raw direct-geometry coordinate axes must form an invertible 4D basis")
    if np.any(basis[:3, 3]) or np.any(basis[3, :3]) or basis[3, 3] != 1.0:
        raise ValueError("raw direct-geometry momentum axes cannot mix energy")
    basis_inverse = np.linalg.inv(basis)
    names = tuple(axis_names or (_axis_name(row, index) for index, row in enumerate(basis)))
    data_sum = np.zeros(shape); variance_sum = np.zeros(shape); event_count = np.zeros(shape)
    normalization = np.zeros(shape)
    detector_norm = load_detector_normalization(config["normalization_file"]) if config.get("normalization_file") else None
    detector_mask = load_detector_normalization(config["mask_file"]) if config.get("mask_file") else None
    total = sum(int(dataset.metadata.get("event_count", 0)) for dataset in selected)
    processed = 0
    for dataset in selected:
        source = Path(dataset.metadata["source_file"])
        info = inspect_raw_dgs_run(source)
        geometry = _detector_geometry(source)
        ei = float(config.get("incident_energy_override") or info.incident_energy)
        t0 = float(config.get("t0_override") or 0.0)
        if ei <= 0.0 or info.l1 <= 0.0:
            raise ValueError(f"{source.name} has no usable incident energy or source distance")
        ub = np.asarray(config["ub_matrix"], dtype=float)
        hkl_transform = np.linalg.inv(2.0 * np.pi * ub)
        gonio = _goniometer(info.omega, info.phi, info.chi)
        # The raw IDF is x=horizontal, y=vertical, z=beam.  ISAW UB uses
        # x=beam, y=horizontal, z=vertical, hence [z, x, y].
        charge = max(float(info.proton_charge), 1e-30)
        rows = max(1, int(max_batch_bytes) // 96)
        import h5py
        with h5py.File(source, "r") as handle:
            for bank_name, bank in handle["entry"].items():
                if not (bank_name.startswith("bank") and bank_name.endswith("_events") and "event_id" in bank):
                    continue
                ids = bank["event_id"]; tofs = bank["event_time_offset"]
                for start in range(0, ids.shape[0], rows):
                    stop = min(start + rows, ids.shape[0])
                    event_ids = np.asarray(ids[start:stop], dtype=np.int64)
                    event_tof = np.asarray(tofs[start:stop], dtype=float) - t0
                    positions, valid = geometry.positions_for_ids(event_ids)
                    if detector_norm is not None:
                        valid &= detector_norm.value_for_ids(event_ids) > 0.0
                    if detector_mask is not None:
                        valid &= detector_mask.value_for_ids(event_ids) > 0.0
                    if np.any(valid):
                        positions = positions[valid]; ids_valid = event_ids[valid]; tof = event_tof[valid]
                        l2 = np.linalg.norm(positions, axis=1)
                        final_tof = tof - TOF_US_PER_M_SQRT_MEV * info.l1 / math.sqrt(ei)
                        good = final_tof > 0.0
                        positions = positions[good]; ids_valid = ids_valid[good]; l2 = l2[good]; final_tof = final_tof[good]
                        if final_tof.size:
                            ef = (TOF_US_PER_M_SQRT_MEV * l2 / final_tof) ** 2
                            energy = ei - ef
                            kf = np.sqrt(np.maximum(ef, 0.0) / ENERGY_TO_K2)
                            direction = positions / l2[:, None]
                            q_lab = np.column_stack((-kf * direction[:, 0], -kf * direction[:, 1],
                                                     math.sqrt(ei / ENERGY_TO_K2) - kf * direction[:, 2]))
                            q_ipns = q_lab[:, [2, 0, 1]]
                            q_sample = q_ipns @ gonio
                            hkl = q_sample @ hkl_transform.T
                            coords = np.column_stack((hkl, energy)) @ basis_inverse
                            flat = _flat_bin_indices(coords, edges, shape)
                            keep = flat >= 0
                            if np.any(keep):
                                detector_value = (np.ones(ids_valid.size) if detector_norm is None
                                                  else detector_norm.value_for_ids(ids_valid))
                                weights = 1.0 / (charge * detector_value)
                                ravel = data_sum.ravel()
                                ravel += np.bincount(flat[keep], weights=weights[keep], minlength=ravel.size)
                                variance_sum.ravel()[:] += np.bincount(flat[keep], weights=weights[keep] ** 2, minlength=variance_sum.size)
                                event_count.ravel()[:] += np.bincount(flat[keep], minlength=event_count.size)
                                normalization.ravel()[:] += np.bincount(flat[keep], weights=np.full(np.count_nonzero(keep), charge), minlength=normalization.size)
                    processed += stop - start
                    if progress_callback is not None:
                        progress_callback({"stage": "raw_dgs_events", "iteration": processed, "total": total,
                                           "message": f"reducing raw events {processed}/{total}"})
    covered = normalization > 0.0
    errors = np.sqrt(variance_sum)
    signal = data_sum
    total_events = float(event_count.sum())
    rms = float(np.sqrt(variance_sum.sum() / total_events)) if total_events else 1.0
    zeros = covered & (event_count == 0)
    errors[zeros] = FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER * rms
    mask = ~covered
    axes = tuple(MDHistoAxis(name, edge, "meV" if index == 3 else "r.l.u.",
                             "energy" if index == 3 else "momentum", frame="General Frame" if index == 3 else "HKL")
                 for index, (name, edge) in enumerate(zip(names, edges, strict=True)))
    return MDHistoData(axes=axes, signal=signal, errors=errors, mask=mask, num_events=event_count,
                       metadata={"raw_dgs": config, "rebin": {"vectors": basis.tolist()},
                                 "normalization_denominator": normalization,
                                 "zero_event_bins_are_measured": True,
                                 "zero_count_error_model": "feldman_cousins_68_percent_upper_limit_scaled_by_rms_event_weight",
                                 "event_weight_rms": rms})


@dataclass(frozen=True)
class _DetectorGeometry:
    detector_ids: np.ndarray
    positions: np.ndarray

    def positions_for_ids(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.detector_ids); sorted_ids = self.detector_ids[order]
        found = np.searchsorted(sorted_ids, ids)
        valid = found < sorted_ids.size
        valid[valid] &= sorted_ids[found[valid]] == ids[valid]
        result = np.zeros((ids.size, 3), dtype=float)
        result[valid] = self.positions[order[found[valid]]]
        return result, valid


def _detector_geometry(path: Path) -> _DetectorGeometry:
    """Resolve detector pixels from an IDF's component/type hierarchy."""
    import h5py
    with h5py.File(path, "r") as handle:
        xml_data = handle["entry/instrument/instrument_xml/data"][()]
    root = ET.fromstring(xml_data.tobytes().decode())
    namespace = root.tag.split("}")[0] + "}"
    types = {item.get("name"): item for item in root.findall(f"{namespace}type")}
    idlists = {item.get("idname"): item for item in root.findall(f"{namespace}idlist")}
    ids: list[int] = []; positions: list[np.ndarray] = []
    for component in root.findall(f"{namespace}component"):
        idname = component.get("idlist")
        if not idname or idname not in idlists:
            continue
        leaf_positions = _expand_type(component.get("type"), types, np.eye(3), np.zeros(3), namespace)
        detector_ids = _expand_idlist(idlists[idname], namespace)
        if len(leaf_positions) != len(detector_ids):
            continue
        ids.extend(detector_ids); positions.extend(leaf_positions)
    if not ids:
        raise ValueError(f"{path.name} instrument XML did not define detector pixel positions")
    return _DetectorGeometry(np.asarray(ids, dtype=np.int64), np.asarray(positions, dtype=float))


def _expand_type(name, types, rotation, translation, ns):
    item = types.get(name)
    if item is None:
        return []
    leaves = []
    for component in item.findall(f"{ns}component"):
        child = component.get("type")
        for location in component.findall(f"{ns}location") or [None]:
            local_translation, local_rotation = _location_transform(location)
            child_rotation = rotation @ local_rotation
            child_translation = translation + rotation @ local_translation
            if child in types and types[child].get("is") == "detector":
                leaves.append(child_translation)
            else:
                leaves.extend(_expand_type(child, types, child_rotation, child_translation, ns))
    return leaves


def _location_transform(location):
    if location is None:
        return np.zeros(3), np.eye(3)
    translation = np.array([float(location.get(axis, 0.0)) for axis in ("x", "y", "z")])
    rotation = np.eye(3)
    for element in location.findall("{*}rot"):
        axis = np.array([float(element.get(f"axis-{key}", 0.0)) for key in ("x", "y", "z")])
        length = np.linalg.norm(axis)
        if length:
            axis /= length; angle = math.radians(float(element.get("val", 0.0)))
            cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = rotation @ (np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross))
    return translation, rotation


def _expand_idlist(item, ns):
    values = []
    for element in item.findall(f"{ns}id"):
        if element.get("start") is None:
            continue
        start = int(element.get("start")); end = int(element.get("end", start)); step = int(element.get("step", 1))
        values.extend(range(start, end + (1 if step > 0 else -1), step))
    return values


def _log_value(entry, names, default):
    logs = entry.get("DASlogs")
    if logs is None: return default
    for name in names:
        group = logs.get(name)
        if group is not None:
            data = group.get("average_value") or group.get("value")
            if data is not None:
                values = np.asarray(data[()]).reshape(-1)
                if values.size: return float(values[0])
    return default


def _source_distance(entry):
    # Raw SNS files normally encode this in their IDF.  The SEQUOIA moderator is
    # on the beam axis at z=-20.0114 m.
    try:
        xml_data = entry["instrument/instrument_xml/data"][()]
        root = ET.fromstring(xml_data.tobytes().decode())
        for component in root.findall("{*}component"):
            if component.get("type") == "moderator":
                location = component.find("{*}location")
                return float(abs(float(location.get("z", 0.0))))
    except (KeyError, ET.ParseError, TypeError, ValueError):
        pass
    return 0.0


def _ub_from_logs(entry):
    logs = entry.get("DASlogs")
    if logs is not None and "BL17:CS:CrystalAlign:UBMatrix" in logs:
        raw = _text_scalar(logs["BL17:CS:CrystalAlign:UBMatrix"].get("value"), "")
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]
        if len(numbers) >= 9:
            return np.asarray(numbers[:9], dtype=float).reshape(3, 3)
    return np.eye(3)


def _text_scalar(dataset, default):
    if dataset is None: return default
    value = np.asarray(dataset[()]).reshape(-1)
    if not value.size: return default
    first = value[0]
    return first.decode() if isinstance(first, bytes) else str(first)


def _goniometer(omega, phi, chi):
    # Mantid's common single-axis SNS setup is omega about vertical.  The
    # additional rotations are retained for files that supply them.
    def rot(axis, degrees):
        angle = math.radians(degrees); c, s = math.cos(angle), math.sin(angle)
        if axis == 2: return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        if axis == 1: return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
        return np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
    return rot(2, omega) @ rot(1, chi) @ rot(2, phi)


def _axis_name(vector, index):
    if index == 3: return "DeltaE"
    labels = ("H", "K", "L")
    return "[" + ",".join("0" if value == 0 else label if value == 1 else f"{value:g}{label}" for value, label in zip(vector[:3], labels, strict=True)) + "]"
