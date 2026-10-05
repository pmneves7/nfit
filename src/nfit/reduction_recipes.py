"""Persistent, editable native-event source and reduction recipes.

The numerical importers retain their existing configuration dictionaries. This
module owns the schema, validation and per-run overlay for those dictionaries;
reading an effective configuration never opens a source or loads a cache.
"""

from __future__ import annotations

import copy
import json
import keyword
import math
import pprint
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .dgs_normalization import DEFAULT_TRAJECTORY_ENERGY_POLICY, TRAJECTORY_ENERGY_POLICIES
from .dgs_reduction_policy import (
    DEFAULT_EVENT_PRECISION_POLICY,
    DEFAULT_MONITOR_VARIANCE_POLICY,
    DEFAULT_SYMMETRY_VARIANCE_POLICY,
    DGS_REDUCTION_POLICY_VERSION,
    EVENT_PRECISION_POLICIES,
    MONITOR_VARIANCE_POLICIES,
    SYMMETRY_VARIANCE_POLICIES,
)

REDUCTION_RECIPE_VERSION = 1
REDUCTION_RECIPE_KEY = "reduction_recipe"
REDUCTION_OVERRIDES_KEY = "reduction_overrides"
_FORMATS = {
    "raw-direct-geometry-nexus": "raw_dgs",
    "corelli-correlation-nexus": "raw_dgs",
    "mantid-mdevent": "mdevent",
}


@dataclass(frozen=True)
class ReductionSetting:
    """One public setting shared by validation, editors and recipe exports.

    ``units`` describes numerical values; ``scope`` names their scientific
    dependency. ``automatic`` allows ``None`` (rather than a numerical sentinel).
    ``per_run`` determines whether an individual source can override the group.
    """

    key: str
    label: str
    kind: str
    default: Any
    tooltip: str
    units: str = ""
    scope: str = "reduction"
    automatic: bool = False
    choices: tuple[tuple[str, str], ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    per_run: bool = True
    reduced_event_dependency: bool | None = None

    @property
    def affects_reduced_events(self):
        """Whether this setting changes laboratory events or cached calibration."""
        if self.reduced_event_dependency is not None:
            return self.reduced_event_dependency
        return self.scope in {"reduction", "normalization"}


@dataclass(frozen=True)
class ReductionEdit:
    """Effective changes made by :func:`set_reduction_settings`."""

    changed_dataset_ids: tuple[str, ...]
    changed_keys: tuple[str, ...]
    scopes: tuple[str, ...]
    settings_changed: bool = False


def _format_config(group_or_format):
    if isinstance(group_or_format, str):
        if group_or_format not in _FORMATS:
            raise ValueError(f"unsupported reduction format {group_or_format!r}")
        return group_or_format, None, _FORMATS[group_or_format]
    for key in ("raw_dgs", "mdevent"):
        config = group_or_format.metadata.get(key)
        if isinstance(config, dict) and config.get("format") in _FORMATS:
            return config["format"], config, key
    raise ValueError("reduction recipes require a native raw DGS, CORELLI or MDEvent group")


def reduction_family(group) -> str | None:
    """Return a native reduction format, or ``None`` for ordinary groups."""
    try:
        return _format_config(group)[0]
    except (ValueError, AttributeError):
        return None


def reduction_settings_schema(group_or_format) -> tuple[ReductionSetting, ...]:
    """Return every supported adjustable setting, without reading source files.

    UB is a shared coordinate transform. It does not belong to raw laboratory
    event reconstruction. MDEvent coordinates are already reconstructed, so its
    incident-energy override affects trajectory normalization only; time zero
    cannot change the stored events and is retained only in import provenance.
    """
    format_name, _, _ = _format_config(group_or_format)
    corelli = format_name == "corelli-correlation-nexus"
    mde = format_name == "mantid-mdevent"
    result = [
        ReductionSetting("normalization_file", "Solid angle / vanadium normalization" if corelli else "Vanadium normalization",
                         "path", None, "Detector normalization calibration file; empty uses no detector calibration.",
                         scope="normalization", automatic=True),
        ReductionSetting("mask_file", "Detector mask", "path", None,
                         "Detector mask file. Masked detectors contribute neither counts nor exposure.",
                         scope="normalization", automatic=True),
    ]
    if corelli:
        from .corelli import DEFAULT_TIMING_OFFSET_NS, DEFAULT_WAVELENGTH_RANGE_ANGSTROM

        result += [
            ReductionSetting("flux_file", "Incident flux", "path", None,
                             "Incident spectrum calibration file used by CORELLI cross-correlation normalization.",
                             scope="normalization", automatic=True),
            ReductionSetting("timing_offset_ns", "Correlation timing offset", "float", DEFAULT_TIMING_OFFSET_NS,
                             "Signed offset of correlation-chopper timing, in nanoseconds.", units="ns"),
            ReductionSetting("wavelength_min_angstrom", "Minimum incident wavelength", "float", DEFAULT_WAVELENGTH_RANGE_ANGSTROM[0],
                             "Lower incident wavelength bound, in angstroms; must be positive.", units="angstrom", minimum=0.0),
            ReductionSetting("wavelength_max_angstrom", "Maximum incident wavelength", "float", DEFAULT_WAVELENGTH_RANGE_ANGSTROM[1],
                             "Upper incident wavelength bound, in angstroms; must exceed the lower bound.", units="angstrom", minimum=0.0),
        ]
    else:
        result += [
            ReductionSetting("incident_energy_override", "Trajectory Ei override" if mde else "Ei override", "float", None,
                             ("Positive incident energy used only for normalization trajectories; stored MDEvent coordinates stay fixed."
                              if mde else "Positive incident energy used in event reconstruction and normalization; automatic uses the calibrated run Ei."),
                             units="meV", automatic=True, minimum=0.0,
                             scope="normalization" if mde else "reduction"),
        ]
        if not mde:
            result += [
                ReductionSetting("t0_override", "T0 override", "float", None,
                                 "Signed time-zero correction in microseconds. Automatic uses the run monitor calibration; negative values are valid.",
                                 units="us", automatic=True),
                ReductionSetting("energy_min_fraction", "Emin / Ei", "float", -0.95,
                                 "Lower reconstructed energy-transfer bound divided by Ei; must be below Emax / Ei."),
                ReductionSetting("energy_max_fraction", "Emax / Ei", "float", 0.95,
                                 "Upper reconstructed energy-transfer bound divided by Ei; must be below one and above Emin / Ei.", maximum=1.0),
                ReductionSetting("monitor_variance_policy", "Monitor variance convention", "choice", DEFAULT_MONITOR_VARIANCE_POLICY,
                                 "Mantid reproduces GetEi peak-tail arithmetic and stopping without calling Mantid. Stable uses an algebraically nonnegative derivative variance; peak selection and resolved Ei/T0 can change. Improved calibration accuracy has not been established.", choices=MONITOR_VARIANCE_POLICIES),
                ReductionSetting("hyspec_default_mask", "Apply default HYSPEC mask", "bool", True,
                                 "Mask pixels 1–8 and 121–128 on every HYSPEC detector tube, matching the Shiver single-crystal default. Applies to events and exposure, in addition to any supplied detector mask. Other instruments are unaffected."),
                ReductionSetting("hyspec_tof_crop", "HYSPEC raw TOF window", "bool", True,
                                 "Apply Shiver's HYSPEC TOF frame window before time-zero correction. Uses the run msd and effective Ei; other instruments are unaffected."),
                ReductionSetting("hyspec_tank_offset_override", "HYSPEC Tank Y offset", "float", None,
                                 "Additional HYSPEC Tank rotation in degrees. Automatic uses mean(psda)*(1-mean(psr)/4200); zero explicitly disables this rotation. Other instruments are unaffected.",
                                 units="deg", automatic=True),
            ]
        result += [
            ReductionSetting("trajectory_energy_policy", "Trajectory incident energy", "choice", DEFAULT_TRAJECTORY_ENERGY_POLICY,
                             "Use the first participating run Ei (Mantid MDNorm) or each run Ei for normalization trajectories.",
                             scope="normalization", choices=TRAJECTORY_ENERGY_POLICIES, per_run=False, reduced_event_dependency=False),
            ReductionSetting("event_precision_policy", "Event precision", "choice", DEFAULT_EVENT_PRECISION_POLICY,
                             "Mantid follows float32 corrected-event storage, projection and histogram boundary arithmetic. High precision keeps float64 corrections and projection; both use the same Mantid He-3 correction. Stored MDE coordinates cannot recover precision lost upstream. This choice applies to the entire grid and can change boundary membership; changing raw precision regenerates reduced events.",
                             scope="histogram" if mde else "reduction", choices=EVENT_PRECISION_POLICIES, per_run=False),
            ReductionSetting("symmetry_variance_policy", "Symmetry uncertainty", "choice", DEFAULT_SYMMETRY_VARIANCE_POLICY,
                             "Independent copies follows Mantid's diagonal event variance. Within-bin covariance includes shared-source terms when copies of one event reach the same final bin. Neither choice retains covariance between different bins through later slices or cuts; replay original events directly onto the final grid for that calculation. Monitor and calibration uncertainty remain separate assumptions.",
                             scope="histogram", choices=SYMMETRY_VARIANCE_POLICIES, per_run=False),
        ]
    if not mde:
        result += [
            ReductionSetting("bad_pulse_threshold", "Minimum pulse charge", "float", 0.0 if corelli else 95.0,
                             "Minimum pulse charge as a percentage of the run's mean. Zero disables charge-based rejection; beam deadtime selection still applies.",
                             units="%", minimum=0.0),
            ReductionSetting("ki_kf_normalization", "Apply ki/kf correction", "bool", True,
                             "Multiply each event by the incident-to-final wavevector ratio and propagate the squared factor to its variance."),
            ReductionSetting("he3_detector_efficiency_correction", "Helium-3 detector efficiency", "bool", True,
                             "Apply the wavelength-dependent helium-3 detector efficiency correction to events and their variance."),
        ]
    if not mde and not corelli:
        result += [ReductionSetting("cache_reduced_events", "Cache reduced events", "bool", True,
                                    "Keep lazy, uncompressed per-run laboratory events for reuse by other binnings.",
                                    scope="storage", per_run=False)]
    result += [ReductionSetting("ub_matrix", "UB matrix", "matrix", None,
                                "Shared reciprocal-space transform in inverse angstroms (without 2 pi). Changes reproject laboratory events without reducing raw TOF again.",
                                units="angstrom^-1", scope="coordinates", per_run=False)]
    return tuple(result)


def _validate_value(setting, value):
    if value is None:
        if setting.automatic:
            return None
        raise ValueError(f"{setting.key} does not support automatic mode")
    if setting.kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{setting.key} must be a boolean")
        return bool(value)
    if setting.kind == "path":
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{setting.key} must be a file path or None")
        return str(value) if str(value).strip() else None
    if setting.kind == "choice":
        if value not in {choice for choice, _ in setting.choices}:
            raise ValueError(f"{setting.key} must be one of {tuple(choice for choice, _ in setting.choices)!r}")
        return value
    if setting.kind == "matrix":
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{setting.key} must be a finite invertible 3 by 3 matrix") from error
        if array.shape != (3, 3) or not np.all(np.isfinite(array)) or np.linalg.matrix_rank(array) != 3:
            raise ValueError(f"{setting.key} must be a finite invertible 3 by 3 matrix")
        return array.tolist()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{setting.key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{setting.key} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{setting.key} must be a finite number")
    if setting.minimum is not None and number < setting.minimum:
        raise ValueError(f"{setting.key} must be at least {setting.minimum}")
    if setting.maximum is not None and number > setting.maximum:
        raise ValueError(f"{setting.key} must be at most {setting.maximum}")
    if setting.key in {"incident_energy_override", "wavelength_min_angstrom", "wavelength_max_angstrom"} and number <= 0:
        raise ValueError(f"{setting.key} must be positive or automatic where supported")
    if setting.key == "energy_max_fraction" and number >= 1:
        raise ValueError("energy_max_fraction must be below one (positive final neutron energy)")
    return number


def _setting_value(config, setting):
    # Keep the historical public ki/kf alias readable without manufacturing a
    # second setting or silently changing old saved projects.
    if setting.key == "ki_kf_normalization" and setting.key not in config:
        return config.get("kf_ki_normalization", setting.default)
    return config.get(setting.key, setting.default)


def validated_reduction_settings(group_or_format, updates, *, base=None, per_run=False) -> dict[str, Any]:
    """Validate an update atomically and return normalized JSON values.

    ``base`` supplies the effective settings needed for bounds validation; it is
    never changed. Unknown or inert settings are rejected rather than persisted
    as though the importer supported them.
    """
    _, current, _ = _format_config(group_or_format)
    schema = {item.key: item for item in reduction_settings_schema(group_or_format)}
    if not isinstance(updates, Mapping):
        raise TypeError("reduction updates must be a mapping")
    unknown = set(updates) - schema.keys()
    if unknown:
        raise ValueError(f"unsupported reduction settings: {', '.join(sorted(unknown))}")
    if per_run:
        shared_only = [key for key in updates if not schema[key].per_run]
        if shared_only:
            raise ValueError(f"settings require shared group ownership: {', '.join(shared_only)}")
    result = {key: _validate_value(schema[key], value) for key, value in updates.items()}
    merged = dict(current if base is None and current is not None else base or {})
    merged.update(result)
    if "energy_min_fraction" in schema:
        lower = float(_setting_value(merged, schema["energy_min_fraction"]))
        upper = float(_setting_value(merged, schema["energy_max_fraction"]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper or upper >= 1:
            raise ValueError("energy fractions require finite Emin / Ei < Emax / Ei < 1")
    if "wavelength_min_angstrom" in schema:
        lower = float(_setting_value(merged, schema["wavelength_min_angstrom"]))
        upper = float(_setting_value(merged, schema["wavelength_max_angstrom"]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0 or lower >= upper:
            raise ValueError("incident wavelength bounds require 0 < minimum < maximum")
    return result


def _group_dataset(group, dataset):
    # Background replay can deliberately construct several transformed views
    # of one physical run with its stable ID. A directly supplied entry owns
    # that view's overrides; an ID alone cannot select between such views.
    if not isinstance(dataset, str) and any(entry is dataset for entry in group.datasets):
        return dataset
    identity = dataset if isinstance(dataset, str) else getattr(dataset, "id", None)
    matches = [entry for entry in group.datasets if entry.id == identity]
    if len(matches) != 1:
        raise ValueError("the run must belong directly to this reduction group")
    return matches[0]


def get_reduction_overrides(group, dataset) -> dict[str, Any]:
    """Read a run's explicit overrides; no sources or caches are loaded."""
    _format_config(group)
    dataset = _group_dataset(group, dataset)
    return copy.deepcopy(dataset.metadata.get(REDUCTION_OVERRIDES_KEY, {}))


def effective_reduction_config(group, dataset=None) -> dict[str, Any]:
    """Return the shared configuration with a validated run overlay.

    Absent keys use current importer defaults when executed. Explicit choices
    remain intact; this pure read does not create or refresh a recipe.
    """
    _, config, _ = _format_config(group)
    result = copy.deepcopy(config)
    if dataset is not None:
        overrides = get_reduction_overrides(group, dataset)
        result.update(validated_reduction_settings(group, overrides, base=result, per_run=True))
    return result


def resolved_reduction_values(group, dataset) -> dict[str, Any]:
    """Read requested and metadata-resolved automatic values, including units."""
    dataset = _group_dataset(group, dataset)
    config = effective_reduction_config(group, dataset)
    resolved = dataset.metadata.get("resolved_reduction", {})
    automatic_values = resolved.get("automatic_values", {})
    values = {}
    for key, metadata_key in (("incident_energy_override", "incident_energy"), ("t0_override", "t0"),
                              ("hyspec_tank_offset_override", "hyspec_tank_offset")):
        if (metadata_key in dataset.metadata or key in config
                or (key == "hyspec_tank_offset_override"
                    and "hyspec_tank_offset_degrees" in automatic_values)):
            automatic = config.get(key) is None
            recorded_key, units = {
                "incident_energy": ("incident_energy_meV", "meV"),
                "t0": ("t0_microseconds", "us"),
                "hyspec_tank_offset": ("hyspec_tank_offset_degrees", "deg"),
            }[metadata_key]
            values[key] = {"automatic": automatic,
                           "value": automatic_values.get(recorded_key, automatic_values.get(metadata_key, automatic_values.get(key, dataset.metadata.get(metadata_key)))) if automatic else config[key],
                           "units": units}
            if resolved.get("stale"):
                values[key]["stale"] = True
    return values


def _file_provenance(path):
    if not path:
        return None
    source = Path(path).expanduser().resolve()
    record = {"path": str(source)}
    try:
        stat = source.stat()
    except OSError:
        record["available"] = False
    else:
        record.update(available=True, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
    return record


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"reduction recipe value is not finite JSON data: {type(value).__name__}")


def _source_paths(group):
    return list(dict.fromkeys(str(dataset.metadata["source_file"]) for dataset in group.datasets))


def _source_selection(paths, selection):
    result = copy.deepcopy(selection or {"mode": "files"})
    if not isinstance(result, dict):
        raise TypeError("source selection must be a JSON mapping")
    result["resolved_files"] = list(paths)
    return _plain(result)


def _run_record(group, dataset):
    metadata = dataset.metadata
    acquisition_keys = ("instrument", "instrument_name", "instrument_id", "geometry_identity", "geometry_signature",
                        "run_number", "event_count", "proton_charge", "duration", "omega", "phi", "chi", "l1",
                        "source_to_chopper", "corelli_goniometer_angles", "goniometer_matrix", "ub_matrix", "incident_energy", "t0")
    return {
        "dataset_id": dataset.id,
        "source_identity": {"path": str(Path(metadata["source_file"]).expanduser().resolve()),
                            "experiment_index": metadata.get("mdevent_experiment_index")},
        "source_file": metadata["source_file"],
        "acquisition": _plain({key: metadata[key] for key in acquisition_keys if key in metadata}),
        "resolved_automatic_values": resolved_reduction_values(group, dataset),
        "calibration_provenance": _plain({key: metadata[key] for key in ("calibration_source", "calibration_warning", "calibration_provenance")
                                         if key in metadata}),
        "resolved_reduction": _plain(metadata.get("resolved_reduction")),
    }


def ensure_reduction_recipe(group, *, source_selection=None) -> dict[str, Any]:
    """Persist a complete metadata-only recipe, retaining the selection expression.

    Source and calibration fingerprints are lightweight file metadata. Geometry
    is saved separately for every source; no common instrument is inferred from
    the first run. Source contents are not opened and reduced caches remain lazy.
    """
    format_name, config, _ = _format_config(group)
    previous = group.metadata.get(REDUCTION_RECIPE_KEY, {})
    selection = previous.get("source_selection") if source_selection is None else source_selection
    schema = reduction_settings_schema(group)
    shared = {item.key: _setting_value(config, item) for item in schema if item.scope != "coordinates"}
    calibrations = {}
    fingerprints = {}

    def fingerprint(path):
        key = str(Path(path).expanduser().resolve()) if path else None
        if key not in fingerprints:
            fingerprints[key] = _file_provenance(path)
        return fingerprints[key]

    for dataset in group.datasets:
        effective = effective_reduction_config(group, dataset)
        calibrations[dataset.id] = {key: fingerprint(effective.get(key))
                                    for key in ("normalization_file", "mask_file", "flux_file") if key in effective}
    recipe = {
        "version": REDUCTION_RECIPE_VERSION,
        "format": format_name,
        "source_selection": _source_selection(_source_paths(group), selection),
        "shared_defaults": _plain(shared),
        "per_run_overrides": {dataset.id: get_reduction_overrides(group, dataset) for dataset in group.datasets},
        "runs": [_run_record(group, dataset) for dataset in group.datasets],
        "coordinate_transform": {"ownership": "group", "ub_matrix": _plain(config.get("ub_matrix"))},
        "provenance": {"algorithm": format_name, "dgs_policy_version": DGS_REDUCTION_POLICY_VERSION,
                       "source_files": [fingerprint(path) for path in _source_paths(group)],
                       "calibrations_by_run": calibrations},
    }
    recipe["source_selection"]["resolved_runs"] = [
        {"dataset_id": dataset.id, "source_file": dataset.metadata["source_file"],
         "run_number": dataset.metadata.get("run_number"),
         "experiment_index": dataset.metadata.get("mdevent_experiment_index")}
        for dataset in group.datasets]
    # MDE conversion parameters are provenance, not adjustable raw TOF controls.
    if format_name == "mantid-mdevent":
        recipe["provenance"]["stored_coordinate_parameters"] = {
            key: _plain(config[key]) for key in ("t0_override", "coordinate_frame", "workspace_path") if key in config}
    group.metadata[REDUCTION_RECIPE_KEY] = recipe
    return copy.deepcopy(recipe)


def _mark_stale(group, changed):
    # Import authoritative ownership constants at the mutation boundary, rather
    # than make the schema depend on preparation services during pure reads.
    from .project_composites import GROUP_COMPOSITE_BINNINGS_KEY
    from .project_imports import GROUP_COMPOSITE_KEY
    from .project_rebinning import DATASET_REBIN_BINNINGS_KEY

    def mark(container, key, registry_key):
        config = container.get(key)
        if isinstance(config, dict):
            config["stale"] = True
        registry = container.get(registry_key)
        if isinstance(registry, dict):
            for item in registry.get("items", ()):
                if isinstance(item, dict) and isinstance(item.get("config"), dict):
                    item["config"]["stale"] = True

    for node in (group, *group.iter_subgroups()):
        mark(node.metadata, GROUP_COMPOSITE_KEY, GROUP_COMPOSITE_BINNINGS_KEY)
    for dataset in changed:
        mark(dataset.parameters, "rebin", DATASET_REBIN_BINNINGS_KEY)


def set_reduction_settings(group, updates, *, dataset_ids=None, inherit=False) -> ReductionEdit:
    """Edit shared settings or explicit per-run overrides and invalidate dependents.

    ``dataset_ids=None`` changes group defaults. For selected runs, ``None`` is
    explicit automatic mode; ``inherit=True`` removes the listed overrides.
    Validation covers every affected effective run before any mutation. Raw
    laboratory-event caches survive coordinate, histogram and storage edits;
    only effective reduction/calibration changes discard the affected run cache.
    """
    _, config, _ = _format_config(group)
    before_shared = copy.deepcopy(config)
    if inherit and dataset_ids is None:
        raise ValueError("inherit applies only to selected per-run overrides")
    schema = {item.key: item for item in reduction_settings_schema(group)}
    targets = group.datasets if dataset_ids is None else [_group_dataset(group, item) for item in dataset_ids]
    before_overrides = {dataset.id: get_reduction_overrides(group, dataset) for dataset in targets}
    if len({item.id for item in targets}) != len(targets):
        raise ValueError("duplicate selected dataset identities")
    if not isinstance(updates, Mapping):
        raise TypeError("reduction updates must be a mapping")
    if inherit:
        if set(updates) - schema.keys() or any(not schema[key].per_run for key in updates if key in schema):
            raise ValueError("inherit requires supported per-run settings")
        normalized = {}
    else:
        normalized = validated_reduction_settings(group, updates, per_run=dataset_ids is not None,
                                                   base=effective_reduction_config(group, targets[0]) if dataset_ids is not None and targets else config)
    new_config = dict(config)
    if dataset_ids is None:
        new_config.update(normalized)
    prepared = []
    for dataset in targets:
        old = effective_reduction_config(group, dataset)
        overrides = get_reduction_overrides(group, dataset)
        if dataset_ids is not None:
            if inherit:
                overrides = {key: value for key, value in overrides.items() if key not in updates}
            else:
                overrides.update(normalized)
        new = dict(new_config)
        new.update(validated_reduction_settings(group, overrides, base=new, per_run=True))
        validated_reduction_settings(group, {}, base=new)
        keys = tuple(key for key, item in schema.items() if _plain(_setting_value(old, item)) != _plain(_setting_value(new, item)))
        prepared.append((dataset, overrides, keys))
    if dataset_ids is None:
        config.update(normalized)
    else:
        for dataset, overrides, _ in prepared:
            if overrides:
                dataset.metadata[REDUCTION_OVERRIDES_KEY] = overrides
            else:
                dataset.metadata.pop(REDUCTION_OVERRIDES_KEY, None)
    changed = [dataset for dataset, _, keys in prepared if any(schema[key].scope != "storage" for key in keys)]
    for dataset, _, keys in prepared:
        if any(schema[key].affects_reduced_events for key in keys):
            from .raw_dgs_cache import clear_reduced_event_cache

            clear_reduced_event_cache(dataset)
            resolved = dataset.metadata.get("resolved_reduction")
            if isinstance(resolved, dict):
                resolved["stale"] = True
    if changed:
        _mark_stale(group, changed)
    ensure_reduction_recipe(group)
    changed_keys = tuple(sorted({key for _, _, keys in prepared for key in keys}))
    return ReductionEdit(tuple(dataset.id for dataset in changed), changed_keys,
                         tuple(sorted({schema[key].scope for key in changed_keys})),
                         _plain(before_shared) != _plain(config) or any(before_overrides[dataset.id] != get_reduction_overrides(group, dataset)
                                                        for dataset in targets))


def _spec_payload(spec):
    return _plain({item.name: getattr(spec, item.name) for item in fields(spec)
                   if item.name not in {"source_entry", "source_group"}})


def export_reduction_recipe(group) -> dict[str, Any]:
    """Return a complete portable source recipe without mutating the project.

    Numerical payloads and cache references are deliberately excluded. Raw
    source files and calibrations remain referenced by their original paths.
    Individual run masks, backgrounds and fitting parameters are retained.
    """
    format_name, config, config_key = _format_config(group)
    if group.subgroups:
        raise ValueError("export each native reduction group separately before assembling its parent")
    temporary = copy.copy(group)
    temporary.metadata = copy.deepcopy(group.metadata)
    recipe = ensure_reduction_recipe(temporary)
    descriptors = []
    for dataset in group.datasets:
        if dataset.transforms:
            raise ValueError("reduction recipe export requires serializable transforms; export transformed sources separately")
        metadata = {key: value for key, value in dataset.metadata.items()
                    if key not in {"raw_dgs_reduction_cache", "_project_path"} and not key.startswith("_nfit")}
        descriptors.append({
            "id": dataset.id, "name": dataset.name, "kind": dataset.kind, "data_type": dataset.data_type,
            "metadata": _plain(metadata), "parameters": _plain(dataset.parameters),
            "enabled": dataset.enabled, "fit_weight": dataset.fit_weight, "scale_factor": dataset.scale_factor,
            "scale_factor_vary": dataset.scale_factor_vary, "scale_factor_group": dataset.scale_factor_group,
            "masks": [_spec_payload(spec) for spec in dataset.masks],
            "backgrounds": [_spec_payload(spec) for spec in dataset.backgrounds],
        })
    structural_config = {**config, "source_files": _source_paths(group)}
    return _plain({**recipe, "group_id": group.id, "group_name": group.name, "group_enabled": group.enabled,
                   "config_key": config_key, "config": structural_config, "datasets": descriptors,
                   "group_masks": [_spec_payload(spec) for spec in group.masks],
                   "group_backgrounds": [_spec_payload(spec) for spec in group.backgrounds],
                   "group_resolution": group.resolution, "format": format_name})


def _replay_config(payload):
    """Apply authoritative editable sections without materializing old defaults."""
    format_name, _, _ = _format_config(payload.get("format"))
    config = copy.deepcopy(payload.get("config", {}))
    shared = payload.get("shared_defaults", {})
    choices = validated_reduction_settings(format_name, shared, base=config)
    schema = {item.key: item for item in reduction_settings_schema(format_name)}
    for key, value in choices.items():
        if _plain(_setting_value(config, schema[key])) != value:
            config[key] = value
    coordinate = payload.get("coordinate_transform", {})
    if coordinate.get("ownership", "group") != "group":
        raise ValueError("UB coordinate transforms require shared group ownership")
    if coordinate.get("ub_matrix") is not None:
        config.update(validated_reduction_settings(format_name, {"ub_matrix": coordinate["ub_matrix"]}, base=config))
    config["source_files"] = list(payload.get("source_selection", {}).get("resolved_files", config.get("source_files", ())))
    return config


def _descriptor_metadata(payload, descriptor):
    metadata = copy.deepcopy(descriptor.get("metadata", {}))
    overrides = payload.get("per_run_overrides", {}).get(descriptor["id"], {})
    if overrides:
        metadata[REDUCTION_OVERRIDES_KEY] = copy.deepcopy(overrides)
    else:
        metadata.pop(REDUCTION_OVERRIDES_KEY, None)
    return metadata


def replay_reduction_recipe(recipe, *, progress_callback=None):
    """Import original sources and replay complete settings without Qt or Mantid.

    Importers freshly inspect source metadata. Stable source/experiment identity
    restores saved dataset IDs and order; cached arrays are never embedded in a
    recipe. The caller can then run any supported public histogram API.
    """
    from .pipeline import BackgroundSpec, MaskSpec

    payload = _plain(recipe)
    if payload.get("version") != REDUCTION_RECIPE_VERSION:
        raise ValueError("unsupported reduction recipe version")
    format_name, _, config_key = _format_config(payload.get("format"))
    config = _replay_config(payload)
    descriptors = payload.get("datasets")
    if not isinstance(config, dict) or not isinstance(descriptors, list) or not descriptors:
        raise ValueError("replay requires a complete exported reduction recipe")
    paths = list(config.get("source_files", ()))
    if not paths:
        raise ValueError("replay requires at least one resolved source file")
    settings = {item.key: config[item.key] for item in reduction_settings_schema(format_name) if item.key in config}
    validated_reduction_settings(format_name, settings, base=config)
    common = dict(normalization_path=config.get("normalization_file"), mask_path=config.get("mask_file"),
                  name=payload.get("group_name"), progress_callback=progress_callback)
    if format_name == "raw-direct-geometry-nexus":
        from .raw_dgs import raw_dgs_dataset_group

        group = raw_dgs_dataset_group(paths, **common)
    elif format_name == "corelli-correlation-nexus":
        from .corelli import corelli_dataset_group

        common.pop("normalization_path")
        group = corelli_dataset_group(paths, solid_angle_path=config.get("normalization_file"),
                                      flux_path=config.get("flux_file"), **common)
    else:
        from .mdevent import append_mdevent_file, mdevent_dataset_group

        group = mdevent_dataset_group(paths[0], **common)
        for path in paths[1:]:
            append_mdevent_file(group, path)
    fresh_by_identity = {}
    for dataset in group.datasets:
        identity = (str(Path(dataset.metadata["source_file"]).expanduser().resolve()),
                    dataset.metadata.get("mdevent_experiment_index"))
        if identity in fresh_by_identity:
            raise ValueError("duplicate source identities in replayed reduction recipe")
        fresh_by_identity[identity] = dataset
    restored = []
    ids = set()
    for descriptor in descriptors:
        metadata = _descriptor_metadata(payload, descriptor)
        identity = (str(Path(metadata["source_file"]).expanduser().resolve()), metadata.get("mdevent_experiment_index"))
        dataset = fresh_by_identity.pop(identity, None)
        if dataset is None or descriptor["id"] in ids:
            raise ValueError("saved run identities do not match replayed source runs")
        ids.add(descriptor["id"])
        fresh_metadata = dataset.metadata
        dataset.metadata = {**metadata, **fresh_metadata}
        for key in ("id", "name", "kind", "data_type", "parameters", "enabled", "fit_weight", "scale_factor", "scale_factor_vary", "scale_factor_group"):
            if key in descriptor:
                setattr(dataset, key, copy.deepcopy(descriptor[key]))
        dataset.masks = [MaskSpec(**item) for item in descriptor.get("masks", ())]
        dataset.backgrounds = [BackgroundSpec(**item) for item in descriptor.get("backgrounds", ())]
        restored.append(dataset)
    # A container may contain unselected experiments. Its saved run descriptors,
    # not the container's complete current catalog, own selection membership.
    group.datasets = restored
    group.id = payload.get("group_id", group.id)
    group.enabled = payload.get("group_enabled", True)
    group.metadata[config_key] = copy.deepcopy(config)
    group.masks = [MaskSpec(**item) for item in payload.get("group_masks", ())]
    group.backgrounds = [BackgroundSpec(**item) for item in payload.get("group_backgrounds", ())]
    group.resolution = copy.deepcopy(payload.get("group_resolution", {}))
    for dataset in group.datasets:
        effective_reduction_config(group, dataset)
    ensure_reduction_recipe(group, source_selection=payload.get("source_selection"))
    return group


def apply_reduction_recipe(group, recipe) -> ReductionEdit:
    """Apply an edited exported recipe to a saved group without loading events.

    Existing source identities retain their IDs, run configuration and reusable
    caches. The recipe owns source membership, shared reduction defaults, run
    overrides and UB; the saved group retains topology, masks, backgrounds,
    resolution, histogram recipes and display settings. Newly added sources are
    inspected with the native importer before the atomic replacement.
    """
    payload = _plain(recipe)
    if payload.get("version") != REDUCTION_RECIPE_VERSION or payload.get("format") != reduction_family(group):
        raise ValueError("recipe format/version does not match the native reduction group")
    config = _replay_config(payload)
    before_config = effective_reduction_config(group)
    before_overrides = {dataset.id: get_reduction_overrides(group, dataset) for dataset in group.datasets}
    before_selection = group.metadata.get(REDUCTION_RECIPE_KEY, {}).get("source_selection", {})
    descriptors = payload.get("datasets", ())
    if not descriptors:
        raise ValueError("a reduction group requires at least one selected run")
    current = {(str(Path(dataset.metadata["source_file"]).expanduser().resolve()),
                dataset.metadata.get("mdevent_experiment_index")): dataset for dataset in group.datasets}
    identities = [(str(Path(item["metadata"]["source_file"]).expanduser().resolve()),
                   item["metadata"].get("mdevent_experiment_index")) for item in descriptors]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate source identities in edited reduction recipe")
    imported = None
    if any(identity not in current for identity in identities):
        imported = replay_reduction_recipe(payload)
    imported_by_identity = {} if imported is None else {
        (str(Path(dataset.metadata["source_file"]).expanduser().resolve()), dataset.metadata.get("mdevent_experiment_index")): dataset
        for dataset in imported.datasets}
    temporary = copy.copy(group)
    temporary.metadata = copy.deepcopy(group.metadata)
    _, _, config_key = _format_config(group)
    temporary.metadata[config_key] = config
    temporary.datasets = []
    old_by_id = {dataset.id: dataset for dataset in group.datasets}
    prepared = []
    schema = {item.key: item for item in reduction_settings_schema(group)}
    changed_keys = set()
    for descriptor, identity in zip(descriptors, identities, strict=True):
        original = current.get(identity)
        dataset = copy.copy(original) if original is not None else imported_by_identity[identity]
        dataset.metadata = copy.deepcopy(dataset.metadata)
        metadata = _descriptor_metadata(payload, descriptor)
        if REDUCTION_OVERRIDES_KEY in metadata:
            dataset.metadata[REDUCTION_OVERRIDES_KEY] = metadata[REDUCTION_OVERRIDES_KEY]
        else:
            dataset.metadata.pop(REDUCTION_OVERRIDES_KEY, None)
        temporary.datasets.append(dataset)
        effective = effective_reduction_config(temporary, dataset)
        before = effective_reduction_config(group, original) if original is not None else {}
        keys = {key for key, setting in schema.items()
                if _plain(_setting_value(before, setting)) != _plain(_setting_value(effective, setting))}
        if original is None:
            keys.update(schema)
        changed_keys.update(keys)
        prepared.append((dataset, keys))
    ids = [dataset.id for dataset in temporary.datasets]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate dataset IDs in edited reduction recipe")
    # Capture before mutating the saved group; this also validates provenance.
    ensure_reduction_recipe(temporary, source_selection=payload.get("source_selection"))
    final_datasets = []
    for dataset in temporary.datasets:
        original = old_by_id.get(dataset.id)
        if original is not None:
            # Runtime background and viewer references point at these objects.
            # Preserve their identity after the entire edit has validated.
            original.metadata = dataset.metadata
            final_datasets.append(original)
        else:
            final_datasets.append(dataset)
    group.datasets = final_datasets
    group.metadata[config_key] = config
    group.metadata[REDUCTION_RECIPE_KEY] = temporary.metadata[REDUCTION_RECIPE_KEY]
    changed = []
    final_by_id = {dataset.id: dataset for dataset in group.datasets}
    for candidate, keys in prepared:
        dataset = final_by_id[candidate.id]
        if any(schema[key].scope != "storage" for key in keys):
            changed.append(dataset)
        if any(schema[key].affects_reduced_events for key in keys):
            from .raw_dgs_cache import clear_reduced_event_cache

            clear_reduced_event_cache(dataset)
            if isinstance(dataset.metadata.get("resolved_reduction"), dict):
                dataset.metadata["resolved_reduction"]["stale"] = True
    membership_changed = set(old_by_id) != set(ids) or [dataset.id for dataset in group.datasets] != list(old_by_id)
    if changed or membership_changed:
        _mark_stale(group, changed)
    ensure_reduction_recipe(group, source_selection=payload.get("source_selection"))
    return ReductionEdit(tuple(dataset.id for dataset in changed), tuple(sorted(changed_keys)),
                         tuple(sorted({schema[key].scope for key in changed_keys} | ({"sources"} if membership_changed else set()))),
                         _plain(before_config) != _plain(config) or membership_changed
                         or before_overrides != {dataset.id: get_reduction_overrides(group, dataset) for dataset in group.datasets}
                         or any(before_selection.get(key) != payload.get("source_selection", {}).get(key)
                                for key in ("mode", "directory", "prefix", "suffix", "expression", "padding")))


def reduction_recipe_script(group, *, group_variable="group") -> str:
    """Export an editable complete source-and-reduction replay script."""
    if not group_variable.isidentifier() or keyword.iskeyword(group_variable):
        raise ValueError("group_variable must be a Python variable name")
    recipe = export_reduction_recipe(group)
    # pprint produces Python literals (including None/bools), unlike JSON text.
    literal = pprint.pformat(json.loads(json.dumps(recipe, allow_nan=False)), sort_dicts=False, width=100)
    return ("from nfit import replay_reduction_recipe\n\n"
            f"REDUCTION_RECIPE = {literal}\n\n"
            f"{group_variable} = replay_reduction_recipe(REDUCTION_RECIPE)\n")
