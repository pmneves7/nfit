"""Array-only codec shared by portable and project point-data archives."""

from __future__ import annotations

import json

import numpy as np

from .dataset import PointData4D
from .measurement_dependencies import restore_source_dependencies, source_dependency_archive_payload


def point_data_archive_payload(data: PointData4D):
    """Encode all point observations, context and retained statistics as arrays."""
    payload = {name: getattr(data, name) for name in ("H", "K", "L", "E", "intensity", "sigma", "mask")}
    for name in ("temperature", "magnetic_field", "normalization_denominator"):
        if getattr(data, name) is not None:
            payload[name] = np.asarray(getattr(data, name))
    if data.measurement_payload is not None:
        names = list(data.measurement_payload)
        payload["measurement_payload_names"] = np.asarray(json.dumps(names))
        payload.update({f"measurement_payload_{index}": data.measurement_payload[name] for index, name in enumerate(names)})
    payload.update(source_dependency_archive_payload(data.source_dependencies))
    return payload


def restore_point_data_archive(archive, metadata):
    """Decode optional measurement arrays without placing them in JSON."""
    fields = {name: archive[name] for name in ("H", "K", "L", "E", "intensity", "sigma", "mask")}
    for name in ("temperature", "magnetic_field", "normalization_denominator"):
        if name in archive:
            fields[name] = archive[name]
    payload = None
    if "measurement_payload_names" in archive:
        names = json.loads(str(np.asarray(archive["measurement_payload_names"]).item()))
        if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
            raise ValueError("Invalid point measurement payload names")
        payload = {name: archive[f"measurement_payload_{index}"] for index, name in enumerate(names)}
    return PointData4D(**fields, metadata=metadata, measurement_payload=payload,
        source_dependencies=restore_source_dependencies(archive))
