"""Per-run execution and provenance for the persisted reduction recipe."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import numpy as np

from .dgs_normalization import trajectory_incident_energies


def acquisition_identity(entry):
    """Identify instrument geometry from its full embedded definition, not a name."""
    definition = entry.get("instrument/instrument_xml/data")
    if definition is None:
        return {"instrument_name": "unknown", "geometry_signature": None}
    xml = definition[()].tobytes()
    root = ET.fromstring(xml)
    return {"instrument_name": root.get("name", "unknown"),
            "geometry_signature": hashlib.sha256(xml).hexdigest()}


def effective_trajectory_energies(group, datasets, energies, *, reference_energy=None):
    """Apply a shared trajectory convention and explicit per-run Ei overrides.

    First-run normalization chooses its reference before partitioning runs.
    Explicit per-run overrides apply to that run's trajectory only. Raw-event
    reconstruction always uses the run's own effective incident energy.
    """
    from .reduction_recipes import effective_reduction_config

    datasets = list(datasets)
    configs = [effective_reduction_config(group, dataset) for dataset in datasets]
    values = [config.get("incident_energy_override") if config.get("incident_energy_override") is not None
              else energy for config, energy in zip(configs, energies, strict=True)]
    shared = effective_reduction_config(group)
    # Per-run automatic mode cancels a shared override; the policy then uses
    # the resolved run values. Keeping the override here would conceal that edit.
    shared = {**shared, "incident_energy_override": None}
    result = list(trajectory_incident_energies(shared, values, reference_energy=reference_energy))
    for index, config in enumerate(configs):
        override = config.get("incident_energy_override")
        if override is not None:
            result[index] = float(override)
    return tuple(result)


def record_resolved_reduction(dataset, config, *, automatic=None, provenance=None):
    """Persist bounded resolved settings without changing the source data token."""
    automatic = dict(automatic or {})
    values = dict(automatic)
    for key, value in config.items():
        if key.endswith("_override"):
            target = {"incident_energy_override": "incident_energy_meV", "t0_override": "t0_microseconds"}.get(key, key)
            if value is not None:
                values[target] = value
    dataset.metadata["resolved_reduction"] = {"version": 1, "values": values,
        "automatic_values": automatic, "provenance": dict(provenance or {})}


def run_calibrations(group, dataset, loader, cache):
    """Load each effective calibration once per operation, not once per chunk."""
    from .reduction_recipes import effective_reduction_config

    config = effective_reduction_config(group, dataset)
    key = (config.get("normalization_file"), config.get("mask_file"))
    if key not in cache:
        cache[key] = tuple(loader(path) if path else None for path in key)
    return cache[key]


def mdevent_run_masks(group, wanted, loader):
    """Resolve masks per logical experiment, including runs in one container."""
    from .reduction_recipes import effective_reduction_config

    cache = {}
    result = {}
    for index, dataset in wanted.items():
        path = effective_reduction_config(group, dataset).get("mask_file")
        if path not in cache:
            cache[path] = loader(path) if path else None
        result[index] = cache[path]
    return result


def filter_mdevent_masks(chosen, masks):
    """Apply a run's detector mask without combining masks of other runs."""
    keep = np.ones(len(chosen), dtype=bool)
    for index, mask in masks.items():
        if mask is None:
            continue
        rows = chosen[:, 2].astype(np.int64) == index
        keep[rows] = mask.value_for_ids(chosen[rows, 4].astype(np.int64)) > 0
    return chosen[keep]


def reduction_dependency_signature(group):
    """Include effective per-run settings and calibration file identities."""
    from .raw_dgs_cache import _file_signature, reduction_signature
    from .reduction_recipes import effective_reduction_config, reduction_settings_schema

    shared = effective_reduction_config(group)
    result = []
    for dataset in group.datasets:
        config = effective_reduction_config(group, dataset)
        if shared["format"] == "raw-direct-geometry-nexus":
            signature = reduction_signature(dataset, config)
        else:
            files = {}
            for key in ("normalization_file", "mask_file", "flux_file"):
                try:
                    files[key] = _file_signature(config.get(key))
                except FileNotFoundError:
                    files[key] = [str(config[key]), "missing"]
            settings = {item.key: config.get(item.key, item.default) for item in reduction_settings_schema(group)
                        if item.scope != "storage"}
            if "ki_kf_normalization" not in config and "kf_ki_normalization" in config:
                settings["ki_kf_normalization"] = config["kf_ki_normalization"]
            signature = {"settings": settings, "workspace_path": config.get("workspace_path"), "calibration_files": files}
        result.append([dataset.id, signature])
    return result
