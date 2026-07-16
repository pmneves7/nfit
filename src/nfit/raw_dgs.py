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
    _MDEVENT_NUMBA, _accumulate_detector_trajectory, _flat_bin_indices, _symmetry_matrices,
    load_detector_normalization,
)
from .mdhisto import MDHistoAxis, MDHistoData
from .pipeline import DatasetEntry, DatasetGroup

# A neutron with energy E(meV) travels one metre in 2286.4/sqrt(E)
# microseconds.  This follows directly from v = 437.393 sqrt(E) m/s.
TOF_US_PER_M_SQRT_MEV = 2286.4
# Mantid He3TubeEfficiency's exponential constant in K / (m Angstrom atm).
HE3_EFFICIENCY_EXPONENTIAL_CONSTANT = 2175.486863864
# Mantid's parameter files select these formula-driven GetEi v2 paths instead
# of fitting two monitor peaks. The formulas are instrument definitions, not
# empirical corrections, and use incident energy in meV.
MANTID_T0_FORMULAS = {
    "CNCS": "288.595706061-110.833514059/sqrt(incidentEnergy)+89.6080314589/incidentEnergy-42.6999684563*sqrt(incidentEnergy)+1.89672170078*incidentEnergy",
    "HYSPEC": "4.0 + (107.0 / (1.0 + (incidentEnergy / 31.0)^3))",
}


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
    t0: float
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
        requested_ei = _log_value(entry, ("EnergyRequest", "Ei", "BL17:Det:TH:BL:Ei"), 0.0)
        calibrated_ei, calibrated_t0 = _monitor_ei_t0(entry, requested_ei)
        return RawDGSRunInfo(
            path=source,
            run_number=_text_scalar(entry.get("run_number"), source.stem),
            event_count=event_count,
            incident_energy=calibrated_ei,
            proton_charge=_raw_proton_charge_uah(entry),
            omega=_log_value(entry, ("omega",), 0.0),
            phi=_log_value(entry, ("phi",), 0.0),
            chi=_log_value(entry, ("chi",), 0.0),
            l1=_source_distance(entry),
            t0=calibrated_t0,
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
        "bad_pulse_threshold": 95.0,
        "ki_kf_normalization": True,
        "he3_detector_efficiency_correction": True,
        "normalization": "proton_charge_and_detector_trajectory",
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
                      "t0": info.t0,
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
    progress_callback: Any | None = None, symmetry_operations: Iterable[Iterable[Iterable[float]]] | None = None,
) -> MDHistoData:
    """Reduce raw direct-geometry event banks into an HKLE histogram.

    Accepted events carry Mantid's direct-geometry He-3 detector-efficiency
    and ki/kf corrections and are divided by an MDNorm-style detector-
    trajectory denominator. The retained proton charge is integrated from the
    raw pulse log in microampere-hours.
    Shiver's ``NormFilename`` convention is preserved here: its non-positive
    detector values are a mask, not a per-detector signal scale.
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
    symmetry = _symmetry_matrices(symmetry_operations)
    names = tuple(axis_names or (_axis_name(row, index) for index, row in enumerate(basis)))
    data_sum = np.zeros(shape); variance_sum = np.zeros(shape); event_count = np.zeros(shape)
    # Shiver's GenerateDGSMDE uses NormFilename only to construct a detector
    # mask. Its MakeSlice call does not pass this workspace to MDNorm as a
    # SolidAngleWorkspace, so matching that path must not weight by vanadium.
    detector_norm = None
    detector_mask = _combined_detector_mask(config)
    total = sum(int(dataset.metadata.get("event_count", 0)) for dataset in selected)
    processed = 0
    for dataset in selected:
        source = Path(dataset.metadata["source_file"])
        info = inspect_raw_dgs_run(source)
        geometry = _detector_geometry(source)
        ei = float(config.get("incident_energy_override") or info.incident_energy)
        t0 = float(config.get("t0_override") if config.get("t0_override") is not None else info.t0)
        if ei <= 0.0 or info.l1 <= 0.0:
            raise ValueError(f"{source.name} has no usable incident energy or source distance")
        ub = np.asarray(config["ub_matrix"], dtype=float)
        hkl_transform = np.linalg.inv(2.0 * np.pi * ub)
        gonio = _goniometer(info.omega, info.phi, info.chi)
        rows = max(1, int(max_batch_bytes) // 96)
        import h5py
        with h5py.File(source, "r") as handle:
            pulse_keep = _good_pulses(handle["entry"], float(config.get("bad_pulse_threshold", 0.0)))
            for bank_name, bank in handle["entry"].items():
                if not (bank_name.startswith("bank") and bank_name.endswith("_events") and "event_id" in bank):
                    continue
                ids = bank["event_id"]; tofs = bank["event_time_offset"]
                event_index = np.asarray(bank["event_index"], dtype=np.int64) if pulse_keep is not None else None
                for start in range(0, ids.shape[0], rows):
                    stop = min(start + rows, ids.shape[0])
                    event_ids = np.asarray(ids[start:stop], dtype=np.int64)
                    event_tof = np.asarray(tofs[start:stop], dtype=float) - t0
                    positions, he3_exponents, valid = geometry.event_geometry_for_ids(event_ids)
                    if pulse_keep is not None:
                        pulse_index = np.searchsorted(event_index, np.arange(start, stop), side="right") - 1
                        valid &= pulse_keep[np.clip(pulse_index, 0, pulse_keep.size - 1)]
                    if detector_norm is not None:
                        valid &= detector_norm.value_for_ids(event_ids) > 0.0
                    if detector_mask is not None:
                        valid &= detector_mask.value_for_ids(event_ids) > 0.0
                    if np.any(valid):
                        positions = positions[valid]; he3_exponents = he3_exponents[valid]
                        ids_valid = event_ids[valid]; tof = event_tof[valid]
                        l2 = np.linalg.norm(positions, axis=1)
                        final_tof = tof - TOF_US_PER_M_SQRT_MEV * info.l1 / math.sqrt(ei)
                        good = final_tof > 0.0
                        positions = positions[good]; ids_valid = ids_valid[good]; l2 = l2[good]
                        final_tof = final_tof[good]; he3_exponents = he3_exponents[good]
                        if final_tof.size:
                            ef = (TOF_US_PER_M_SQRT_MEV * l2 / final_tof) ** 2
                            energy = ei - ef
                            kf = np.sqrt(np.maximum(ef, 0.0) / ENERGY_TO_K2)
                            energy_keep = (energy >= -0.95 * ei) & (energy <= 0.95 * ei)
                            positions = positions[energy_keep]; ids_valid = ids_valid[energy_keep]
                            energy = energy[energy_keep]; kf = kf[energy_keep]; l2 = l2[energy_keep]
                            he3_exponents = he3_exponents[energy_keep]
                            if not energy.size:
                                processed += stop - start
                                continue
                            direction = positions / l2[:, None]
                            q_lab = np.column_stack((-kf * direction[:, 0], -kf * direction[:, 1],
                                                     math.sqrt(ei / ENERGY_TO_K2) - kf * direction[:, 2]))
                            q_sample = q_lab @ gonio
                            hkl = q_sample @ hkl_transform.T
                            for operation in symmetry:
                                coords = np.column_stack((hkl @ operation.T, energy)) @ basis_inverse
                                flat = _flat_bin_indices(coords, edges, shape)
                                keep = flat >= 0
                                if np.any(keep):
                                    weights = np.ones(ids_valid.size)
                                    if config.get("he3_detector_efficiency_correction", True):
                                        weights *= _he3_tube_efficiency_correction(kf, he3_exponents)
                                    if _use_ki_kf_correction(config):
                                        weights *= math.sqrt(ei / ENERGY_TO_K2) / kf
                                    ravel = data_sum.ravel()
                                    ravel += np.bincount(flat[keep], weights=weights[keep], minlength=ravel.size)
                                    variance_sum.ravel()[:] += np.bincount(flat[keep], weights=weights[keep] ** 2, minlength=variance_sum.size)
                                    event_count.ravel()[:] += np.bincount(flat[keep], minlength=event_count.size)
                    processed += stop - start
                    if progress_callback is not None:
                        progress_callback({"stage": "raw_dgs_events", "iteration": processed, "total": total,
                                           "message": f"reducing raw events {processed}/{total}"})
    normalization = _trajectory_normalization(
        group, selected, edges, shape, basis_inverse, detector_norm, detector_mask, symmetry,
    )
    covered = normalization > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = data_sum / normalization
        errors = np.sqrt(variance_sum) / normalization
    total_events = float(event_count.sum())
    rms = float(np.sqrt(variance_sum.sum() / total_events)) if total_events else 1.0
    zeros = covered & (event_count == 0)
    errors[zeros] = (
        FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER
        * rms
        / normalization[zeros]
    )
    mask = ~covered
    axes = tuple(MDHistoAxis(name, edge, "meV" if index == 3 else "r.l.u.",
                             "energy" if index == 3 else "momentum", frame="General Frame" if index == 3 else "HKL")
                 for index, (name, edge) in enumerate(zip(names, edges, strict=True)))
    return MDHistoData(axes=axes, signal=signal, errors=errors, mask=mask, num_events=event_count,
                       metadata={"raw_dgs": config, "rebin": {"vectors": basis.tolist()},
                                 "signal_semantics": "density",
                                 "signal_semantics_source": "nfit_raw_tof_reduction",
                                 "normalization_denominator": normalization,
                                 "zero_event_bins_are_measured": True,
                                 "zero_count_error_model": "feldman_cousins_68_percent_upper_limit_scaled_by_rms_event_weight",
                                 "event_weight_rms": rms,
                                 "symmetry_operations_hkl": [operation.tolist() for operation in symmetry],
                                 "ki_kf_normalization": _use_ki_kf_correction(config),
                                 "he3_detector_efficiency_correction": bool(config.get("he3_detector_efficiency_correction", True)),
                                 "proton_charge_units": "microampere-hour (retained raw pulse charge in picocoulombs divided by 3.6e9)"})


def _trajectory_normalization(
    group, datasets, edges, shape, basis_inverse, detector_norm, detector_mask, symmetry_operations=None,
):
    """Native MDNorm-style detector trajectories for raw direct-geometry runs."""
    config = group.metadata["raw_dgs"]
    result = np.zeros(shape)
    symmetry = _symmetry_matrices(symmetry_operations)
    payloads = []
    detector_payload = None
    for dataset in datasets:
        info = inspect_raw_dgs_run(dataset.metadata["source_file"])
        geometry = _detector_geometry(info.path)
        direction = geometry.positions / np.linalg.norm(geometry.positions, axis=1)[:, None]
        solid = np.ones(geometry.detector_ids.size) if detector_norm is None else detector_norm.value_for_ids(geometry.detector_ids)
        if detector_mask is not None:
            solid[detector_mask.value_for_ids(geometry.detector_ids) <= 0.0] = 0.0
        ub = np.asarray(config["ub_matrix"], dtype=float)
        canonical_inverse = np.linalg.inv(2.0 * np.pi * ub) @ _goniometer(info.omega, info.phi, info.chi).T
        inverses = [basis_inverse[:3, :3].T @ operation @ canonical_inverse for operation in symmetry]
        ei = float(config.get("incident_energy_override") or info.incident_energy)
        import h5py
        with h5py.File(info.path, "r") as handle:
            charge = _retained_proton_charge_uah(
                handle["entry"], float(config.get("bad_pulse_threshold", 95.0)),
            )
        if detector_payload is None:
            theta = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
            phi = np.arctan2(direction[:, 1], direction[:, 0])
            detector_payload = (theta, phi, solid)
        payloads.extend((inverse, ei, (-0.95 * ei, 0.95 * ei), charge) for inverse in inverses)
    if _MDEVENT_NUMBA is not None and detector_payload is not None and len(symmetry) == 1:
        theta, phi, solid = detector_payload
        flat = _MDEVENT_NUMBA.run_trajectory_normalization(
            theta, phi, solid, np.asarray([item[0] for item in payloads]),
            np.asarray([item[1] for item in payloads]), np.asarray([item[2] for item in payloads]),
            np.asarray([item[3] for item in payloads]), *[np.asarray(edge) for edge in edges],
            np.asarray(shape, dtype=np.int64), workers=1,
        )
        return np.asarray(flat).reshape(shape)
    for inverse, ei, energy_bounds, charge in payloads:
        for index in np.flatnonzero(solid > 0.0):
            _accumulate_detector_trajectory(result, edges, inverse, direction[index], ei, energy_bounds, charge * solid[index])
    return result


def _combined_detector_mask(config):
    paths = [config.get("normalization_file"), config.get("mask_file")]
    masks = [load_detector_normalization(path) for path in dict.fromkeys(path for path in paths if path)]
    if not masks:
        return None
    ids = np.unique(np.concatenate([mask.detector_ids for mask in masks]))
    values = np.ones(ids.size)
    for mask in masks:
        values[mask.value_for_ids(ids) <= 0.0] = 0.0
    return type(masks[0])(Path("combined_mask"), ids, values, np.zeros(ids.size))


def _good_pulses(entry, threshold):
    logs = entry.get("DASlogs")
    if threshold <= 0.0 or logs is None or "proton_charge" not in logs:
        return None
    charge_log = logs["proton_charge"]
    if "value" not in charge_log:
        return None
    values = np.asarray(charge_log["value"], dtype=float).reshape(-1)
    if values.size == 0:
        return None
    cutoff = float(threshold) / 100.0 * float(np.mean(values))
    return values >= cutoff


def _pulse_charge_values(entry):
    logs = entry.get("DASlogs")
    if logs is None:
        return None
    charge_log = logs.get("proton_charge")
    if charge_log is None or "value" not in charge_log:
        return None
    values = np.asarray(charge_log["value"], dtype=float).reshape(-1)
    return values if values.size else None


def _raw_proton_charge_uah(entry):
    """Return the unfiltered run charge in Mantid's microampere-hour units."""

    values = _pulse_charge_values(entry)
    if values is not None:
        return float(np.sum(values) / 3.6e9)
    if "proton_charge" in entry:
        raw_charge = np.asarray(entry["proton_charge"], dtype=float).reshape(-1)
        if raw_charge.size:
            return float(raw_charge[0] / 3.6e9)
    return _log_value(entry, ("gd_prtn_chrg",), 1.0)


def _retained_proton_charge_uah(entry, threshold):
    """Integrate the same pulse-charge series used to retain raw events."""

    values = _pulse_charge_values(entry)
    if values is None:
        return _raw_proton_charge_uah(entry)
    keep = _good_pulses(entry, threshold)
    if keep is not None:
        values = values[keep]
    return float(np.sum(values) / 3.6e9)


def _use_ki_kf_correction(config):
    """Read the corrected setting while accepting projects saved by nfit 0.4.x."""

    return bool(config.get("ki_kf_normalization", config.get("kf_ki_normalization", True)))


@dataclass(frozen=True)
class _DetectorGeometry:
    detector_ids: np.ndarray
    positions: np.ndarray
    he3_exponents: np.ndarray

    def event_geometry_for_ids(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(self.detector_ids); sorted_ids = self.detector_ids[order]
        found = np.searchsorted(sorted_ids, ids)
        valid = found < sorted_ids.size
        valid[valid] &= sorted_ids[found[valid]] == ids[valid]
        positions = np.zeros((ids.size, 3), dtype=float)
        exponents = np.zeros(ids.size, dtype=float)
        positions[valid] = self.positions[order[found[valid]]]
        exponents[valid] = self.he3_exponents[order[found[valid]]]
        return positions, exponents, valid


def _detector_geometry(path: Path) -> _DetectorGeometry:
    """Resolve detector pixels from an IDF's component/type hierarchy."""
    import h5py
    with h5py.File(path, "r") as handle:
        xml_data = handle["entry/instrument/instrument_xml/data"][()]
    root = ET.fromstring(xml_data.tobytes().decode())
    namespace = root.tag.split("}")[0] + "}"
    types = {item.get("name"): item for item in root.findall(f"{namespace}type")}
    idlists = {item.get("idname"): item for item in root.findall(f"{namespace}idlist")}
    he3_parameters = _idf_he3_parameters(root, namespace)
    ids: list[int] = []; positions: list[np.ndarray] = []; he3_exponents: list[float] = []
    for component in root.findall(f"{namespace}component"):
        idname = component.get("idlist")
        if not idname or idname not in idlists:
            continue
        component_type = component.get("type")
        leaf_positions = _expand_type(
            component_type, types, np.eye(3), np.zeros(3), namespace, he3_parameters,
            he3_parameters.get(component.get("name") or component_type),
        )
        detector_ids = _expand_idlist(idlists[idname], namespace)
        if len(leaf_positions) != len(detector_ids):
            continue
        ids.extend(detector_ids)
        positions.extend(item[0] for item in leaf_positions)
        he3_exponents.extend(_he3_exponent(*item) for item in leaf_positions)
    if not ids:
        raise ValueError(f"{path.name} instrument XML did not define detector pixel positions")
    return _DetectorGeometry(
        np.asarray(ids, dtype=np.int64), np.asarray(positions, dtype=float),
        np.asarray(he3_exponents, dtype=float),
    )


def _expand_type(name, types, rotation, translation, ns, he3_parameters, inherited_he3):
    item = types.get(name)
    if item is None:
        return []
    leaves = []
    for component in item.findall(f"{ns}component"):
        child = component.get("type")
        child_he3 = he3_parameters.get(component.get("name") or child, inherited_he3)
        for location in component.findall(f"{ns}location") or [None]:
            local_translation, local_rotation = _location_transform(location)
            child_rotation = rotation @ local_rotation
            child_translation = translation + rotation @ local_translation
            if child in types and types[child].get("is") == "detector":
                axis, radius = _idf_detector_cylinder(types[child])
                leaves.append((child_translation, child_rotation @ axis, radius, child_he3))
            else:
                leaves.extend(_expand_type(
                    child, types, child_rotation, child_translation, ns, he3_parameters, child_he3,
                ))
    return leaves


def _idf_he3_parameters(root, ns):
    """Return He-3 tube parameters keyed by IDF component-link name."""

    result = {}
    for link in root.findall(f"{ns}component-link"):
        values = {}
        for parameter in link.findall(f"{ns}parameter"):
            value = parameter.find(f"{ns}value")
            if value is not None and value.get("val") is not None:
                try:
                    values[parameter.get("name")] = float(value.get("val"))
                except ValueError:
                    continue
        required = ("tube_pressure", "tube_thickness", "tube_temperature")
        if link.get("name") and all(key in values for key in required):
            result[link.get("name")] = tuple(values[key] for key in required)
    return result


def _idf_detector_cylinder(detector_type):
    cylinder = detector_type.find("{*}cylinder")
    if cylinder is None:
        return np.array([0.0, 1.0, 0.0]), 0.0
    axis_element = cylinder.find("{*}axis")
    axis = np.array([float(axis_element.get(key, 0.0)) for key in ("x", "y", "z")]) if axis_element is not None else np.array([0.0, 1.0, 0.0])
    length = np.linalg.norm(axis)
    if length <= 0.0:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis /= length
    radius = cylinder.find("{*}radius")
    return axis, float(radius.get("val", 0.0)) if radius is not None else 0.0


def _he3_exponent(position, axis, radius, parameters):
    if parameters is None or radius <= 0.0:
        return 0.0
    pressure, thickness, temperature = parameters
    direction_length = np.linalg.norm(position)
    axis_length = np.linalg.norm(axis)
    straight_path = 2.0 * (radius - thickness)
    if pressure <= 0.0 or temperature <= 0.0 or straight_path <= 0.0 or direction_length <= 0.0 or axis_length <= 0.0:
        return 0.0
    cosine = float(np.dot(axis, position) / (axis_length * direction_length))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    if sine <= 1e-12:
        return 0.0
    return HE3_EFFICIENCY_EXPONENTIAL_CONSTANT * (pressure / temperature) * straight_path / sine


def _he3_tube_efficiency_correction(kf, exponents):
    """Mantid He3TubeEfficiency correction at the final neutron wavelength."""

    correction = np.ones(np.asarray(kf).shape, dtype=float)
    active = np.isfinite(exponents) & (exponents > 0.0) & np.isfinite(kf) & (kf > 0.0)
    if np.any(active):
        wavelength = 2.0 * np.pi / np.asarray(kf)[active]
        correction[active] = 1.0 / (-np.expm1(-np.asarray(exponents)[active] * wavelength))
    return correction


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


def _monitor_ei_t0(entry, energy_guess):
    """Estimate Ei/T0 from an embedded-IDF direct-geometry monitor layout.

    This follows Shiver's Mantid ``GetEi`` route. The two monitor groups are
    discovered from the raw file's IDF, then Mantid's current v2 peak-width,
    rebinning, background, and first-moment calculation is reproduced locally.
    The first two usable monitor spectra in IDF order are used, matching
    Mantid's standard spectrum ordering.
    """
    try:
        if energy_guess <= 0.0:
            return float(energy_guess), 0.0
        xml_data = entry["instrument/instrument_xml/data"][()]
        root = ET.fromstring(xml_data.tobytes().decode())
        instrument_name = str(root.get("name", "")).upper()
        formula = _mantid_t0_formula(root, instrument_name)
        if formula is not None:
            return float(energy_guess), _evaluate_mantid_t0_formula(formula, energy_guess)
        locations = []
        for component in root.findall(".//{*}component[@type='monitor']"):
            for location in component.findall("{*}location"):
                locations.append((location.get("name"), _idf_location(entry, location)))
        source_z = -_source_distance(entry)
        idf_names = {name for name, _ in locations if name}
        monitor_groups = sorted(
            (name for name, group in entry.items()
             if "event_time_offset" in group
             and (getattr(group, "attrs", {}).get("NX_class", b"") in (b"NXmonitor", "NXmonitor")
                  or name in idf_names)),
            key=_natural_sort_key,
        )
        monitor_data = []
        for index, (idf_name, position) in enumerate(locations):
            name = idf_name if idf_name in monitor_groups else (
                monitor_groups[index] if index < len(monitor_groups) else None
            )
            if name is None:
                continue
            values = np.asarray(entry[f"{name}/event_time_offset"], dtype=float)
            if values.size:
                source_to_monitor = np.linalg.norm(position - np.array([0.0, 0.0, source_z]))
                monitor_data.append((name, values, float(source_to_monitor)))
        if len(monitor_data) < 2:
            return float(energy_guess), 0.0

        peaks = []
        for name, values, distance in monitor_data:
            peak_centre = _mantid_getei_v2_peak(values, distance, energy_guess)
            if peak_centre is None:
                continue
            peaks.append((name, distance, peak_centre))
        if len(peaks) < 2:
            return float(energy_guess), 0.0
        _, left_distance, left_time = peaks[0]
        _, right_distance, right_time = peaks[1]
        velocity = (right_distance - left_distance) * 1.0e6 / (right_time - left_time)
        if not np.isfinite(velocity) or velocity <= 0.0:
            return float(energy_guess), 0.0
        energy = (velocity / 437.393) ** 2
        if not np.isfinite(energy) or energy <= 0.0:
            return float(energy_guess), 0.0
        t0 = left_time - left_distance * 1.0e6 / velocity
        return float(energy), float(t0)
    except (KeyError, TypeError, ValueError, ET.ParseError):
        return float(energy_guess), 0.0


def _mantid_t0_formula(root, instrument_name):
    """Read a Mantid ``t0_formula`` from an IDF or supported parameter set."""

    for parameter in root.findall(".//{*}parameter[@name='t0_formula']"):
        value = parameter.find("{*}value")
        if value is not None and value.get("val"):
            return str(value.get("val"))
    return MANTID_T0_FORMULAS.get(instrument_name)


def _evaluate_mantid_t0_formula(formula, incident_energy):
    """Evaluate Mantid's arithmetic t0_formula with only ``sqrt`` enabled."""

    tree = ast.parse(str(formula), mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
               ast.Name, ast.Load, ast.Call)
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(f"unsupported Mantid t0_formula: {formula!r}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in {"incidentEnergy", "sqrt"}:
            raise ValueError(f"unsupported Mantid t0_formula name: {node.id!r}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id != "sqrt"):
            raise ValueError(f"unsupported Mantid t0_formula call: {formula!r}")
    return float(eval(compile(tree, "<mantid-t0-formula>", "eval"), {"__builtins__": {}}, {
        "incidentEnergy": float(incident_energy), "sqrt": math.sqrt,
    }))


def _mantid_getei_v2_peak(values, distance, energy_guess):
    """Reproduce Mantid GetEi v2's monitor peak estimate without Mantid."""

    expected = TOF_US_PER_M_SQRT_MEV * distance / math.sqrt(energy_guess)
    lower, upper = 0.9 * expected, 1.1 * expected
    initial_edges = np.arange(lower, upper + 1.0, 1.0)
    counts, edges = np.histogram(values, bins=initial_edges)
    if not np.any(counts):
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    region = _mantid_getei_peak_region(centres, counts.astype(float), np.sqrt(counts))
    if region is None:
        return None
    _, _, width = region
    if width <= 0.0:
        return None
    width /= 12.0
    rebinned_edges = np.arange(lower, upper + width, width)
    rebinned, rebinned_edges = np.histogram(values, bins=rebinned_edges)
    rebinned_centres = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    region = _mantid_getei_peak_region(
        rebinned_centres, rebinned.astype(float) / width, np.sqrt(rebinned) / width,
    )
    if region is None:
        return None
    x, y, _ = region
    area = np.trapezoid(y, x)
    return None if not np.isfinite(area) or area <= 0.0 else float(np.trapezoid(x * y, x) / area)


def _mantid_getei_peak_region(x, y, errors, prominence=4.0):
    """Port of Mantid GetEi2::calculatePeakWidthAtHalfHeight's peak region."""

    if x.size < 3:
        return None
    peak = int(np.argmax(y)); background_floor = float(np.min(y))
    peak_height = float(y[peak] - background_floor)
    if peak_height <= 0.0:
        return None
    peak_error = float(errors[peak])
    left = peak - 1
    while left >= 0:
        ratio = (y[left] - background_floor) / peak_height
        ratio_error = math.sqrt(errors[left] ** 2 + (ratio * peak_error) ** 2) / peak_height
        if ratio < 1.0 / prominence - 2.0 * ratio_error:
            break
        left -= 1
    right = peak + 1
    while right < x.size:
        ratio = (y[right] - background_floor) / peak_height
        ratio_error = math.sqrt(errors[right] ** 2 + (ratio * peak_error) ** 2) / peak_height
        if ratio < 1.0 / prominence - 2.0 * ratio_error:
            break
        right += 1
    if left < 0 or right >= x.size:
        return None

    # Match GetEi2's derivative extension of the initially prominent region.
    derivative, uncertainty = -1000.0, 0.0
    while right < x.size - 1 and derivative < -uncertainty:
        forward, backward = x[right + 1] - x[right], x[right] - x[right - 1]
        derivative = 0.5 * ((y[right + 1] - y[right]) / forward + (y[right] - y[right - 1]) / backward)
        uncertainty = 0.5 * math.sqrt(
            (errors[right + 1] ** 2 + errors[right] ** 2) / forward ** 2
            + (errors[right] ** 2 + errors[right - 1] ** 2) / backward ** 2
            - 2.0 * errors[right] ** 2 / (forward * backward)
        )
        right += 1
    right -= 1
    if derivative < -uncertainty:
        right = x.size - 1

    derivative, uncertainty = 1000.0, 0.0
    while left > 0 and derivative > uncertainty:
        forward, backward = x[left + 1] - x[left], x[left] - x[left - 1]
        derivative = 0.5 * ((y[left + 1] - y[left]) / forward + (y[left] - y[left - 1]) / backward)
        uncertainty = 0.5 * math.sqrt(
            (errors[left + 1] ** 2 + errors[left] ** 2) / forward ** 2
            + (errors[left] ** 2 + errors[left - 1] ** 2) / backward ** 2
            - 2.0 * errors[left] ** 2 / (forward * backward)
        )
        left -= 1
    left += 1
    if derivative > uncertainty:
        left = 0

    peak_width = x[right] - x[left]
    if peak_width <= 0.0:
        return None
    background_start = max(x[0], x[left] - 0.5 * peak_width)
    background_stop = min(x[-1], x[right] + 0.5 * peak_width)
    background_parts = []
    if left > 0:
        keep = (x >= background_start) & (x <= x[left])
        if keep.sum() > 1:
            background_parts.append((np.trapezoid(y[keep], x[keep]), x[keep][-1] - x[keep][0]))
    if right < x.size - 1:
        keep = (x >= x[right]) & (x <= background_stop)
        if keep.sum() > 1:
            background_parts.append((np.trapezoid(y[keep], x[keep]), x[keep][-1] - x[keep][0]))
    background = (sum(area for area, _ in background_parts) / sum(span for _, span in background_parts)
                  if background_parts else 0.0)
    return x[left:right + 1], y[left:right + 1] - background, peak_width


def _idf_location(entry, location):
    """Resolve a static or log-driven IDF location into lab-frame metres."""

    values = []
    for coordinate in ("x", "y", "z"):
        value = location.get(coordinate)
        parameter = location.find(f"{{*}}parameter[@name='{coordinate}']")
        if parameter is not None:
            logfile = parameter.find("{*}logfile")
            if logfile is not None:
                log_value = _log_value(entry, (str(logfile.get("id", "")),), None)
                if log_value is not None:
                    value = _evaluate_idf_log_expression(logfile.get("eq"), log_value)
        values.append(float(value or 0.0))
    return np.asarray(values, dtype=float)


def _evaluate_idf_log_expression(expression, value):
    """Evaluate Mantid IDF's arithmetic ``value`` expression without calls."""

    if not expression:
        return float(value)
    tree = ast.parse(str(expression), mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load)
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(f"unsupported IDF logfile expression: {expression!r}")
    return float(eval(compile(tree, "<idf-logfile>", "eval"), {"__builtins__": {}}, {"value": float(value)}))


def _natural_sort_key(name):
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(name)))


def _interpolated_half_height_time(histogram, edges, maximum, half_height, direction):
    """Return Mantid GetEi's interpolated half-height crossing for one side."""

    index = maximum
    while 0 < index < histogram.size - 1 and histogram[index] > half_height:
        index += direction
    if index <= 0 or index >= histogram.size - 1:
        return None
    centre = 0.5 * (edges[index] + edges[index + 1])
    previous = index - direction
    gradient = (histogram[index] - histogram[previous]) / (edges[index] - edges[previous])
    if gradient == 0.0:
        return centre
    return float(centre - (histogram[index] - half_height) / gradient)


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
    # SEQUOIA's NeXus/Mantid convention has beam along lab +z and vertical +y.
    # Its recorded omega matrix is therefore a rotation around lab y.
    def rot(axis, degrees):
        angle = math.radians(degrees); c, s = math.cos(angle), math.sin(angle)
        if axis == 1: return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
        if axis == 2: return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        return np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
    return rot(1, omega) @ rot(2, chi) @ rot(1, phi)


def _axis_name(vector, index):
    if index == 3: return "DeltaE"
    labels = ("H", "K", "L")
    return "[" + ",".join("0" if value == 0 else label if value == 1 else f"{value:g}{label}" for value, label in zip(vector[:3], labels, strict=True)) + "]"
