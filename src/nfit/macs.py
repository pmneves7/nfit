"""NIST NCNR MACS NeXus import and coordinate conversion.

MACS records two detector streams for every scan point.  ``SPEC`` is the
energy-analyzed stream and ``DIFF`` records the unanalysed remainder of the
scattered beam.  This module keeps those streams separate while applying the
same sample orientation, monitor normalization, and detector geometry.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .dataset import PointData4D
from .fitting import reciprocal_basis_from_lattice_parameters

MACS_E_KSQ_MEV_ANGSTROM2 = 2.0721246
MACS_DETECTOR_OFFSETS_DEG = np.arange(20, dtype=float) * 8.0
MACS_KIDNEY_ZERO_DEG = -76.0


def _decode(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()
    return str(item).strip()


def _entry_group(handle: h5py.File) -> h5py.Group:
    for item in handle.values():
        if not isinstance(item, h5py.Group):
            continue
        nx_class = _decode(item.attrs.get("NX_class", ""))
        if nx_class == "NXentry" or "DAS_logs" in item:
            return item
    raise ValueError("NeXus file does not contain a MACS NXentry")


def _dataset(entry: h5py.Group, *paths: str, required: bool = True) -> np.ndarray | None:
    for path in paths:
        try:
            obj = entry[path]
        except KeyError:
            continue
        if isinstance(obj, h5py.Dataset):
            return np.asarray(obj[()])
    if required:
        raise ValueError(f"MACS NeXus file is missing {' or '.join(paths)}")
    return None


def _scan_vector(value: np.ndarray | None, count: int, *, name: str) -> np.ndarray:
    if value is None:
        return np.full(count, np.nan, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(count, float(array[0]), dtype=float)
    if array.size != count:
        raise ValueError(f"MACS {name} has {array.size} values; expected {count}")
    return array


def _detector_array(value: np.ndarray, count: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (20, count):
        array = array.T
    if array.shape != (count, 20):
        raise ValueError(f"MACS {name} has shape {array.shape}; expected ({count}, 20)")
    return array


def is_macs_nexus_file(path: str | Path) -> bool:
    """Return whether *path* is an HDF5 NeXus file produced by MACS.

    Detection uses internal instrument metadata and required detector datasets,
    not the filename extension.  Invalid, incomplete, and inaccessible files
    simply return ``False`` so other registered importers can be considered.
    """

    try:
        with h5py.File(Path(path), "r") as handle:
            entry = _entry_group(handle)
            instrument = _dataset(
                entry,
                "DAS_logs/experiment/instrument",
                "instrument/name",
                required=False,
            )
            if _decode(instrument).lower() != "macs":
                return False
            return all(
                candidate in entry
                for candidate in (
                    "DAS_logs/specDetector/counts",
                    "DAS_logs/diffDetector/counts",
                )
            )
    except (OSError, ValueError, KeyError):
        return False


def _first_value(entry: h5py.Group, *paths: str) -> float:
    value = _dataset(entry, *paths)
    array = np.asarray(value, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"MACS metadata {' or '.join(paths)} contains no finite value")
    return float(finite[0])


def _lattice_orientation(entry: h5py.Group) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    lattice = {
        name: _first_value(entry, f"DAS_logs/sampleState/{source}")
        for name, source in (
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
            ("alpha", "Alpha"),
            ("beta", "Beta"),
            ("gamma", "Gamma"),
        )
    }
    u_raw = np.asarray(_dataset(entry, "DAS_logs/sampleState/refPlane1"), dtype=float)
    v_raw = np.asarray(_dataset(entry, "DAS_logs/sampleState/refPlane2"), dtype=float)
    u = u_raw.reshape(-1, 3)[0]
    v = v_raw.reshape(-1, 3)[0]
    return lattice, u, v


def _oriented_ub(lattice: dict[str, float], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    basis = reciprocal_basis_from_lattice_parameters(
        *(lattice[name] for name in ("a", "b", "c", "alpha", "beta", "gamma")),
        include_2pi=False,
    )
    q_u = basis @ u
    q_v = basis @ v
    if np.linalg.norm(q_u) <= 1.0e-14 or np.linalg.norm(np.cross(q_u, q_v)) <= 1.0e-14:
        raise ValueError("MACS orientation vectors must be nonzero and define a plane")
    x_axis = q_u / np.linalg.norm(q_u)
    z_axis = np.cross(q_u, q_v)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack((x_axis, y_axis, z_axis)) @ basis


def _individual_analyzer_angles(entry: h5py.Group, count: int) -> np.ndarray | None:
    columns: list[np.ndarray] = []
    for index in range(1, 21):
        value = _dataset(
            entry,
            f"DAS_logs/anaTwoTheta{index:02d}/softPosition",
            f"DAS_logs/anaTheta{index:02d}/softPosition",
            required=False,
        )
        if value is None:
            return None
        columns.append(_scan_vector(value, count, name=f"analyzer {index} angle"))
    return np.column_stack(columns)


def _parse_channels(values: Any) -> list[int]:
    """Return sorted zero-based channel indices from a 1-based option value."""

    if values is None or (isinstance(values, str) and not values.strip()):
        return []
    if isinstance(values, str):
        raw: Iterable[Any] = values.split(",")
    elif np.isscalar(values):
        raw = [values]
    else:
        raw = values
    result: set[int] = set()
    for value in raw:
        channel = int(str(value).strip())
        if not 1 <= channel <= 20:
            raise ValueError("MACS analyzer channels must be between 1 and 20")
        result.add(channel - 1)
    return sorted(result)


def _spec_analyzer_mask(
    entry: h5py.Group,
    counts: np.ndarray,
    *,
    alignment_tolerance_deg: float,
    detect_dead_channels: bool,
    dead_channel_min_nonzero_fraction: float,
    manual_channels: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a point mask from file masks, DAVE alignment, and channel health."""

    count = counts.shape[0]
    mask = np.ones(counts.shape, dtype=bool)
    reasons: dict[int, set[str]] = {}

    roi = _dataset(entry, "DAS_logs/specDetector/roiMask", required=False)
    if roi is not None:
        roi_mask = _detector_array(roi, count, name="SPEC ROI mask") > 0
        mask &= roi_mask
        for channel in np.where(~np.all(roi_mask, axis=0))[0]:
            reasons.setdefault(int(channel), set()).add("NeXus SPEC ROI mask")

    analyzer = _individual_analyzer_angles(entry, count)
    if analyzer is not None:
        # DAVE compares every blade with the closest blade to the common setting,
        # then excludes blades farther away by the configured tolerance.
        common = _dataset(
            entry,
            "DAS_logs/anaTwoTheta/primaryNode",
            "DAS_logs/anaTheta/primaryNode",
            required=False,
        )
        common_angle = _scan_vector(common, count, name="common analyzer angle")
        distance = np.abs(analyzer - common_angle[:, None])
        # Some scans record no analyzer angle for an entire row. Treat that
        # row as having no alignment evidence instead of emitting an all-NaN
        # reduction warning while loading an otherwise usable file.
        finite_distance = np.where(np.isfinite(distance), distance, np.inf)
        best = np.min(finite_distance, axis=1)
        misset = distance > best[:, None] + float(alignment_tolerance_deg)
        mask &= ~misset
        for channel in np.where(np.any(misset, axis=0))[0]:
            reasons.setdefault(int(channel), set()).add(
                f"analyzer angle exceeds DAVE-style {alignment_tolerance_deg:g}° tolerance"
            )

    if detect_dead_channels and count >= 20:
        finite_rows = np.all(np.isfinite(counts), axis=1)
        if np.any(finite_rows):
            occupancy = np.mean(counts[finite_rows] > 0.0, axis=0)
            typical = float(np.median(occupancy))
            threshold = min(float(dead_channel_min_nonzero_fraction), 0.25 * typical)
            dead = np.where(occupancy < threshold)[0]
            for channel in dead:
                mask[:, channel] = False
                reasons.setdefault(int(channel), set()).add(
                    f"unresponsive SPEC channel ({occupancy[channel]:.1%} nonzero scans)"
                )

    for channel in manual_channels:
        mask[:, channel] = False
        reasons.setdefault(channel, set()).add("user-specified analyzer mask")

    records = [
        {"channel": channel + 1, "reasons": sorted(messages)}
        for channel, messages in sorted(reasons.items())
    ]
    return mask, records


def _temperature(entry: h5py.Group, count: int) -> np.ndarray | None:
    sensor = _decode(
        _dataset(entry, "DAS_logs/temp/primarySensor", required=False)
    )
    candidates = []
    if sensor:
        candidates.append(f"DAS_logs/temp/sensor_{sensor}/average_value")
    candidates.extend(
        ("DAS_logs/temp/primaryNode/average_value", "DAS_logs/temp/primaryNode")
    )
    value = _dataset(entry, *candidates, required=False)
    if value is None:
        return None
    result = _scan_vector(value, count, name="temperature")
    return result if np.any(np.isfinite(result)) else None


def import_macs_nexus(
    path: str | Path,
    options: dict[str, Any] | None = None,
) -> PointData4D:
    """Import one detector stream from a NIST NCNR MACS NeXus file.

    ``options['stream']`` is ``"spec"`` (default) or ``"diff"``. Intensities
    and Poisson errors are normalized to ``monitor_target`` counts.  ``DIFF``
    uses DAVE's elastic-coordinate approximation (``Ef = Ei``, ``DeltaE = 0``)
    because that detector does not analyze the outgoing neutron energy.
    """

    source = Path(path)
    settings: dict[str, Any] = {
        "stream": "spec",
        "monitor_target": 1_000_000.0,
        "a3_offset_deg": None,
        "apply_detector_efficiency": True,
        "mask_misaligned_analyzers": True,
        "analyzer_alignment_tolerance_deg": 1.0,
        "detect_dead_analyzers": True,
        "dead_channel_min_nonzero_fraction": 0.2,
        "masked_analyzer_channels": [],
    }
    if isinstance(options, dict):
        settings.update(options)
    stream = str(settings["stream"]).strip().lower()
    if stream not in {"spec", "diff"}:
        raise ValueError("MACS stream must be 'spec' or 'diff'")
    target = float(settings["monitor_target"])
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("MACS monitor target must be finite and positive")
    tolerance = float(settings["analyzer_alignment_tolerance_deg"])
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("MACS analyzer alignment tolerance must be finite and nonnegative")
    occupancy_limit = float(settings["dead_channel_min_nonzero_fraction"])
    if not np.isfinite(occupancy_limit) or not 0.0 <= occupancy_limit <= 1.0:
        raise ValueError("MACS dead-channel nonzero fraction must be between zero and one")

    with h5py.File(source, "r") as handle:
        entry = _entry_group(handle)
        instrument = _decode(
            _dataset(entry, "DAS_logs/experiment/instrument", "instrument/name")
        )
        if instrument.lower() != "macs":
            raise ValueError(f"expected a MACS NeXus file, found instrument {instrument!r}")
        raw = np.asarray(
            _dataset(entry, f"DAS_logs/{stream}Detector/counts"), dtype=float
        )
        count = raw.shape[0] if raw.ndim == 2 and raw.shape[1] == 20 else raw.shape[-1]
        counts = _detector_array(raw, count, name=f"{stream.upper()} counts")
        monitor = _scan_vector(
            _dataset(entry, "DAS_logs/counter/liveMonitor", "data/monitor"),
            count,
            name="monitor",
        )
        ei = _scan_vector(
            _dataset(entry, "DAS_logs/ei/energy", "DAS_logs/initialEnergy/energy"),
            count,
            name="incident energy",
        )
        ef_measured = _scan_vector(
            _dataset(entry, "DAS_logs/ef/energy", "DAS_logs/finalEnergy/energy"),
            count,
            name="final energy",
        )
        a3 = _scan_vector(
            _dataset(
                entry,
                "DAS_logs/sampleTheta/primaryNode",
                "DAS_logs/sampleTheta/softPosition",
            ),
            count,
            name="sample A3",
        )
        kidney = _scan_vector(
            _dataset(
                entry,
                "DAS_logs/kidneyMotor/softPosition",
                "DAS_logs/kidney/softPosition",
            ),
            count,
            name="kidney angle",
        )
        lattice, u, v = _lattice_orientation(entry)
        ub = _oriented_ub(lattice, u, v)

        file_a3_offset = _dataset(entry, "DAS_logs/sampleState/a3Zero", required=False)
        if settings["a3_offset_deg"] in (None, ""):
            if file_a3_offset is None:
                a3_offset = 0.0
                a3_offset_source = "default"
            else:
                values = np.asarray(file_a3_offset, dtype=float).reshape(-1)
                finite = values[np.isfinite(values)]
                a3_offset = float(finite[0]) if finite.size else 0.0
                a3_offset_source = "NeXus sampleState/a3Zero"
        else:
            a3_offset = float(settings["a3_offset_deg"])
            a3_offset_source = "import option"

        efficiency_value = _dataset(
            entry, f"DAS_logs/{stream}Detector/detectorEfficiency", required=False
        )
        efficiency = np.ones(20, dtype=float)
        if efficiency_value is not None and bool(settings["apply_detector_efficiency"]):
            efficiency = np.asarray(efficiency_value, dtype=float).reshape(-1)
            if efficiency.size != 20 or np.any(~np.isfinite(efficiency)) or np.any(efficiency <= 0.0):
                raise ValueError(
                    f"MACS {stream.upper()} efficiency must contain 20 positive finite values"
                )

        valid_monitor = np.isfinite(monitor) & (monitor > 0.0)
        scale = np.full(count, np.nan, dtype=float)
        scale[valid_monitor] = target / monitor[valid_monitor]
        intensity = counts * efficiency[None, :] * scale[:, None]
        sigma = np.sqrt(np.maximum(counts, 1.0)) * efficiency[None, :] * scale[:, None]

        if stream == "spec":
            if bool(settings["mask_misaligned_analyzers"]):
                applied_tolerance = tolerance
            else:
                applied_tolerance = np.inf
            point_mask, analyzer_reasons = _spec_analyzer_mask(
                entry,
                counts,
                alignment_tolerance_deg=applied_tolerance,
                detect_dead_channels=bool(settings["detect_dead_analyzers"]),
                dead_channel_min_nonzero_fraction=occupancy_limit,
                manual_channels=_parse_channels(settings["masked_analyzer_channels"]),
            )
        else:
            roi = _dataset(entry, "DAS_logs/diffDetector/roiMask", required=False)
            point_mask = (
                np.ones(counts.shape, dtype=bool)
                if roi is None
                else _detector_array(roi, count, name="DIFF ROI mask") > 0
            )
            analyzer_reasons = []
        point_mask &= np.isfinite(counts) & (counts >= 0.0)
        point_mask &= valid_monitor[:, None]

        ef = ef_measured if stream == "spec" else ei
        energy = ei - ef if stream == "spec" else np.zeros(count, dtype=float)
        ki = np.sqrt(ei / MACS_E_KSQ_MEV_ANGSTROM2)
        kf = np.sqrt(ef / MACS_E_KSQ_MEV_ANGSTROM2)
        two_theta = np.deg2rad(
            kidney[:, None] + MACS_KIDNEY_ZERO_DEG + MACS_DETECTOR_OFFSETS_DEG[None, :]
        )
        q_spectrometer = np.stack(
            (
                ki[:, None] - kf[:, None] * np.cos(two_theta),
                kf[:, None] * np.sin(two_theta),
                np.zeros_like(two_theta),
            ),
            axis=-1,
        )
        psi = np.deg2rad(90.0 - (a3 + a3_offset))
        rotation = np.zeros((count, 3, 3), dtype=float)
        rotation[:, 0, 0] = np.cos(psi)
        rotation[:, 0, 1] = -np.sin(psi)
        rotation[:, 1, 0] = np.sin(psi)
        rotation[:, 1, 1] = np.cos(psi)
        rotation[:, 2, 2] = 1.0
        q_sample = np.einsum("nci,nij->ncj", q_spectrometer, rotation)
        rlu_matrix = 2.0 * np.pi * ub
        hkl = np.linalg.solve(rlu_matrix, q_sample.reshape(-1, 3).T).T
        temperature = _temperature(entry, count)
        temp_points = None if temperature is None else np.repeat(temperature, 20)

        selected = sorted({record["channel"] for record in analyzer_reasons})
        parameters: dict[str, Any] = {}
        if temperature is not None:
            parameters["temperature"] = float(np.nanmedian(temperature))
        if stream == "spec":
            from .spectral_channels import default_spectral_channel_config

            spectral = default_spectral_channel_config()
            spectral.update(
                {
                    "source_unit": "arbitrary",
                    "normalization_basis": "unknown",
                    "kf_ki_state": "included",
                    "final_energy_meV": float(np.nanmedian(ef_measured)),
                }
            )
            parameters["spectral_channels"] = spectral

        metadata = {
            "source_file": str(source),
            "importer": "macs_nexus",
            "instrument": "MACS",
            "facility": _decode(_dataset(entry, "facility", required=False)) or "NCNR",
            "detector_stream": stream.upper(),
            "signal_semantics": "monitor_normalized_counts",
            "signal_unit": f"counts/monitor={target:g}",
            "coordinate_units": "RLU",
            "energy_unit": "meV",
            "energy_transfer_convention": "positive neutron energy loss",
            "lattice_parameters": lattice,
            "orientation_u": u.tolist(),
            "orientation_v": v.tolist(),
            "ub_matrix": ub.tolist(),
            "rlu_to_inv_angstrom_matrix": rlu_matrix.tolist(),
            "a3_offset_deg": a3_offset,
            "a3_offset_source": a3_offset_source,
            "monitor_target": target,
            "detector_efficiency_applied": bool(settings["apply_detector_efficiency"]),
            "masked_analyzer_channels": selected,
            "analyzer_mask_reasons": analyzer_reasons,
            "import_options": dict(settings),
            "dataset_parameters": parameters,
        }
        if stream == "spec":
            metadata.update(
                {
                    "fixed_final_energy_meV": float(np.nanmedian(ef_measured)),
                    "energy_analysis": "fixed final energy",
                }
            )
        else:
            metadata.update(
                {
                    "energy_analysis": "none",
                    "coordinate_approximation": (
                        "DAVE-compatible elastic projection: Ef=Ei and DeltaE=0; "
                        "DIFF accepts the unanalysed scattered-energy distribution"
                    ),
                }
            )

    return PointData4D(
        H=hkl[:, 0],
        K=hkl[:, 1],
        L=hkl[:, 2],
        E=np.repeat(energy, 20),
        intensity=intensity.ravel(),
        sigma=sigma.ravel(),
        mask=point_mask.ravel(),
        temperature=temp_points,
        metadata=metadata,
    )
