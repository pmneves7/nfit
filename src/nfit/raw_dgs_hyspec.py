"""HYSPEC preprocessing used by Shiver's raw-event MDE conversion.

TOF bounds apply to recorded microseconds before subtracting time zero. Run
logs here use arithmetic means at full precision; IDF geometry extraction has
its own time averaging and serialization convention. This service reads no
neutron-event arrays and imports neither the reduction facade nor Qt.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .raw_dgs_geometry import numeric_log_values, rotation_matrix
from .reduction_runtime import acquisition_identity

_HYSPEC_CROP_HALF_WIDTH_US = 1.0e6 / 120.0 + 470.0
_HYSPEC_TOF_ENERGY_CONSTANT = 5.227e-6


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _arithmetic_log_mean(entry, name):
    result = float(np.mean(numeric_log_values(entry, name)))
    if not math.isfinite(result):
        raise ValueError(f"HYSPEC {name!r} arithmetic mean must be finite")
    return result


def resolved_hyspec_preprocessing(entry, config, incident_energy, *, instrument_name=None):
    """Return bounded JSON-ready crop/Tank settings, or ``None`` for other instruments.

    ``incident_energy`` is the run's effective incident energy in meV, including
    any override. ``hyspec_tof_crop`` defaults to true. An optional
    ``hyspec_tank_offset_override`` is the additional Tank Y rotation in degrees;
    zero explicitly disables the log-derived offset. Automatic mode uses the
    arithmetic mean ``psda`` in degrees and, when nonzero, ``psr`` in millimetres.
    Crop resolution similarly uses arithmetic mean ``msd`` in millimetres.
    ``instrument_name`` may supply an already inspected source identity.
    """
    name = instrument_name if instrument_name is not None else acquisition_identity(entry)["instrument_name"]
    if str(name).upper() != "HYSPEC":
        return None
    ei = _finite_scalar(incident_energy, "HYSPEC incident energy")
    if ei <= 0.0:
        raise ValueError("HYSPEC incident energy must be positive")
    crop = config.get("hyspec_tof_crop", True)
    if not isinstance(crop, (bool, np.bool_)):
        raise ValueError("hyspec_tof_crop must be true or false")
    bounds = None
    center = None
    msd = None
    if crop:
        msd = _arithmetic_log_mean(entry, "msd")
        distance_mm = 39000.0 + msd + 4500.0
        if distance_mm <= 0.0:
            raise ValueError("HYSPEC TOF crop flight-path distance must be positive")
        center = distance_mm * 1000.0 / math.sqrt(ei / _HYSPEC_TOF_ENERGY_CONSTANT)
        bounds = [center - _HYSPEC_CROP_HALF_WIDTH_US, center + _HYSPEC_CROP_HALF_WIDTH_US]
        if not math.isfinite(center) or not all(math.isfinite(value) for value in bounds):
            raise ValueError("HYSPEC TOF crop bounds must be finite")
    override = config.get("hyspec_tank_offset_override")
    psda = None
    psr = None
    if override is None:
        psda = _arithmetic_log_mean(entry, "psda")
        offset = 0.0
        if psda:
            psr = _arithmetic_log_mean(entry, "psr")
            offset = psda * (1.0 - psr / 4200.0)
        source = "arithmetic_mean_run_logs"
    else:
        offset = _finite_scalar(override, "HYSPEC Tank offset override")
        source = "explicit_override"
    if not math.isfinite(offset):
        raise ValueError("HYSPEC Tank offset must be finite")
    return {
        "version": 1,
        "instrument_name": "HYSPEC",
        "incident_energy_meV": ei,
        "tof_crop_enabled": bool(crop),
        "default_detector_mask_enabled": bool(config.get("hyspec_default_mask", True)),
        "raw_tof_bounds_microseconds": bounds,
        "raw_tof_center_microseconds": center,
        "tof_filter_stage": "raw_event_time_offset_before_t0",
        "msd_arithmetic_mean_millimetres": msd,
        "tank_offset_degrees": offset,
        "tank_offset_source": source,
        "psda_arithmetic_mean_degrees": psda,
        "psr_arithmetic_mean_millimetres": psr,
    }


def raw_hyspec_tof_keep(raw_tof_microseconds, resolved):
    """Select recorded TOFs within resolved bounds before any T0 correction.

    No HYSPEC preprocessing or a disabled crop leaves the selection unchanged.
    A configured crop excludes nonfinite recorded times. The returned boolean
    array is new; the input times remain unchanged.
    """
    tofs = np.asarray(raw_tof_microseconds, dtype=float)
    if resolved is None or not resolved["tof_crop_enabled"]:
        return np.ones(tofs.shape, dtype=bool)
    lower, upper = resolved["raw_tof_bounds_microseconds"]
    return (tofs >= lower) & (tofs <= upper)


def hyspec_default_detector_keep(detector_ids, *, instrument_name, enabled=True):
    """Keep HYSPEC pixels 9–120 of each 128-pixel tube.

    HYSPEC physical detector IDs are zero based, with 128 consecutive IDs per
    tube. This reproduces MaskBTP(Pixel="1-8,121-128") without a mask file.
    Other instrument identities are unaffected, including mixed collections.
    """
    ids = np.asarray(detector_ids, dtype=np.int64)
    if str(instrument_name).upper() != "HYSPEC":
        return np.ones(ids.shape, dtype=bool)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("hyspec_default_mask must be true or false")
    if not enabled:
        return np.ones(ids.shape, dtype=bool)
    pixels = ids % 128
    return (pixels >= 8) & (pixels < 120)


def tank_y_rotation_matrix(offset_degrees):
    """Additional active Tank rotation about Y, in degrees, as a 3×3 matrix.

    Callers compose this relative rotation with the resolved Tank transform and
    choose its component pivot. Applying it to every instrument detector around
    the global sample would assume a Tank position and membership.
    """
    return rotation_matrix(_finite_scalar(offset_degrees, "HYSPEC Tank offset"), [0., 1., 0.])


def rotated_hyspec_idf_xml(xml, offset_degrees):
    """Apply Shiver's additional relative Y rotation to the resolved HYSPEC Tank.

    Only the unique top-level Tank component rotates, about its existing
    position. Its detector positions, tube axes and shape corrections can then
    be reconstructed by the ordinary IDF parser. Relative rotation composes
    ``old_rotation @ Y(offset)``. Other components and all type definitions are
    unchanged. Zero offsets and non-HYSPEC definitions preserve XML bytes.

    The source definition must already have run-log location parameters
    resolved. A missing, ambiguous or nested-only Tank fails explicitly.
    """
    offset = _finite_scalar(offset_degrees, "HYSPEC Tank offset")
    if offset == 0.:
        return xml
    root = ET.fromstring(xml)
    if root.get("name", "").upper() != "HYSPEC":
        return xml
    candidates = []
    for component in root.findall("{*}component"):
        locations = component.findall("{*}location")
        if component.get("name", component.get("type")) == "Tank":
            candidates.extend(locations)
        else:
            candidates.extend(location for location in locations if location.get("name") == "Tank")
    if len(candidates) != 1:
        raise ValueError("HYSPEC offset requires one resolved top-level Tank location")
    location = candidates[0]
    if location.findall("{*}parameter"):
        raise ValueError("HYSPEC Tank location parameters must be resolved before rotation")
    # Mantid's RelativeRotation=True composes old @ delta, including when the
    # original orientation is not about Y. Retain the existing rotation
    # sequence without axis-angle decomposition or precision loss.
    namespace = location.tag[:-len("location")]
    location.append(ET.Element(namespace + "rot", {
        "val": repr(offset), "axis-x": "0", "axis-y": "1", "axis-z": "0",
    }))
    return ET.tostring(root)
