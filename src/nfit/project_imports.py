"""GUI-independent project dataset import and reload orchestration.

This module bridges registered file importers, lazy neutron data sources, and
project :class:`~nfit.pipeline.DatasetEntry` objects. Keeping that bridge free
of Qt makes the same behavior available to scripts and the project explorer.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .analysis.artifacts import read_project_dataset_artifact
from .dataset import PointData4D, PointListData
from .importers import IMPORTERS, import_with, importers_for_data_type, probe_importers
from .mdevent import (
    is_mdevent_file,
    load_mdevent_run_points,
    mdevent_dataset_group,
)
from .mdhisto import (
    MDHistoData,
    load_mantid_mdhisto_nxs,
)
from .pipeline import DataGroup, DatasetEntry, DatasetGroup
from .project_archive import project_artifact_exists
from .raw_dgs import is_raw_dgs_nexus_file, raw_dgs_dataset_group
from .spectral_channels import (
    SPECTRAL_CHANNEL_CONFIG_KEY,
    default_spectral_channel_config,
)

DATA_TYPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "single_crystal_inelastic": {
        "label": "Single crystal inelastic",
        "container": "mdhisto",
    },
    "powder_inelastic": {
        "label": "Powder inelastic",
        "container": "mdhisto",
    },
    "single_crystal_elastic": {
        "label": "Single crystal elastic",
        "container": "mdhisto",
    },
    "single_crystal_energy_integrated": {
        "label": "Single crystal energy-integrated",
        "container": "point_data_4d",
    },
    "powder_elastic": {
        "label": "Powder elastic",
        "container": "point_list",
        "wavelength": True,
    },
    "powder_elastic_spectrum": {
        "label": "Powder elastic spectrum",
        "container": "mdhisto",
    },
    "magnetization": {
        "label": "Magnetization",
        "container": "point_list",
        "scale": True,
        "susceptibility": True,
    },
    "heat_capacity": {
        "label": "Heat capacity",
        "container": "point_list",
        "heat_capacity": True,
    },
    "bragg_reflections": {
        "label": "Bragg reflections",
        "container": "point_list",
    },
}

DEFAULT_DATA_TYPE = "single_crystal_inelastic"
GROUP_COMPOSITE_KEY = "composite"

DatasetFileLoader = Callable[
    [str | Path], tuple[MDHistoData | PointListData, dict[str, Any]]
]
DatasetDataLoader = Callable[[DatasetEntry], Any]


@dataclass(frozen=True)
class _SourceLoadContext:
    """Injectable loaders used by built-in project source handlers."""

    mdhisto_loader: Callable[..., MDHistoData]
    dataset_file_loader: DatasetFileLoader | None
    progress_callback: Any | None = None


@dataclass(frozen=True)
class _SourceFormatHandler:
    """One declarative source-format strategy used throughout this module."""

    name: str
    load: Callable[[DatasetEntry, _SourceLoadContext], Any]
    can_load: Callable[[DatasetEntry], bool | None]
    can_reload: Callable[[DatasetEntry], bool | None]
    group_import_mode: Literal["raw_dgs", "mdevent"] | None = None
    matches_group_import: Callable[[Path, str], bool] | None = None
    entry_import_priority: int | None = None
    reload_as_pending: Callable[[DatasetEntry], bool] | None = None


_SOURCE_NOT_HANDLED = object()
_SOURCE_FORMAT_HANDLERS: tuple[_SourceFormatHandler, ...]
_MDHISTO_SOURCE_SUFFIXES = frozenset({".nxs", ".h5", ".hdf5"})
_LAZY_SOURCE_SUFFIXES = _MDHISTO_SOURCE_SUFFIXES | {".npz"}


def _mdevent_composite_defaults() -> dict[str, Any]:
    """Return the viewer-ready default for an imported MDEvent collection."""

    return {
        "enabled": True,
        "auto_rebin": False,
        "stale": True,
        "fractional": False,
        "mean_weighting": "uniform",
    }


def available_data_types() -> list[tuple[str, str]]:
    """Return ``(type, label)`` pairs for every registered data type."""

    return [
        (name, definition["label"])
        for name, definition in DATA_TYPE_DEFINITIONS.items()
    ]


def data_type_label(data_type: str) -> str:
    """Return the human-readable label for a data type."""

    definition = DATA_TYPE_DEFINITIONS.get(data_type)
    return definition["label"] if definition else (data_type or "-")


def data_type_container(data_type: str) -> str:
    """Return the storage-container kind used by a data type."""

    definition = DATA_TYPE_DEFINITIONS.get(data_type, {})
    return str(definition.get("container", "mdhisto"))


def default_importer_for_data_type(
    data_type: str,
    path: str | Path | None = None,
) -> str | None:
    """Return the default importer name for a data type, or ``None``."""

    if path is not None:
        matches = probe_importers(path, data_type)
        return matches[0].importer_name if matches else None
    specs = importers_for_data_type(data_type)
    return specs[0].name if specs else None


def _unique_name(base: str, existing: Collection[str]) -> str:
    candidate = base or "Dataset"
    if candidate not in existing:
        return candidate
    index = 1
    while f"{candidate}{index}" in existing:
        index += 1
    return f"{candidate}{index}"


def _unique_dataset_name(base: str, existing: Collection[str]) -> str:
    return _unique_name(base or "Dataset", existing)


def _single_normalization_companion(path: str | Path) -> Path | None:
    companions = [
        candidate
        for candidate in Path(path).parent.glob("van*")
        if candidate.is_file()
    ]
    return companions[0] if len(companions) == 1 else None


def _stream_group_name(importer_name: str, stream_label: str) -> str:
    if importer_name == "macs_nexus":
        return f"MACS {stream_label}"
    return stream_label


def import_mdevent_dataset_group(
    group: DataGroup,
    path: str | Path,
    *,
    normalization_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    into: DatasetGroup | None = None,
    progress_callback: Any | None = None,
) -> DatasetGroup:
    """Import a Mantid MDEvent file as lightweight run datasets sharing setup."""

    if normalization_path is None and mask_path is None:
        companion = _single_normalization_companion(path)
        if companion is not None:
            normalization_path = companion
            mask_path = companion
    subgroup = mdevent_dataset_group(
        path,
        normalization_path=normalization_path,
        mask_path=mask_path,
        progress_callback=progress_callback,
    )
    subgroup.metadata[GROUP_COMPOSITE_KEY] = _mdevent_composite_defaults()
    names = {item.name for item in group.iter_subgroups()}
    subgroup.name = _unique_name(subgroup.name, names)
    (into.subgroups if into is not None else group.subgroups).append(subgroup)
    if group.lattice_parameters is None:
        group.lattice_parameters = dict(
            subgroup.metadata["mdevent"]["lattice_parameters"]
        )
    return subgroup


def import_dataset_paths(
    group: DataGroup,
    paths: list[str | Path],
    *,
    data_type: str | None = None,
    importer_name: str | None = None,
    importer_options: dict[str, dict[str, Any]] | None = None,
    into: DatasetGroup | None = None,
    stream_group_mode: str = "reuse",
    progress_callback: Any | None = None,
    dataset_file_loader: DatasetFileLoader | None = None,
) -> list[DatasetEntry]:
    """Add dataset entries for one or more source files.

    Registered tabular sources are loaded eagerly. MDHisto sources remain lazy,
    while raw DGS and MDEvent sources are represented by import-owned groups.
    Multi-stream importers reuse compatible groups unless ``stream_group_mode``
    is ``"new"``.
    """

    if stream_group_mode not in {"reuse", "new"}:
        raise ValueError("stream_group_mode must be 'reuse' or 'new'")

    entries: list[DatasetEntry] = []
    stream_groups: dict[tuple[str, str], DatasetGroup] = {}
    resolved_type = data_type or DEFAULT_DATA_TYPE
    sources = [Path(path) for path in paths]
    group_import_handlers = {
        source: _group_import_handler(source, resolved_type) for source in sources
    }
    raw_sources = [
        source
        for source in sources
        if (
            group_import_handlers[source] is not None
            and group_import_handlers[source].group_import_mode == "raw_dgs"
        )
    ]
    if raw_sources:
        normalization = _single_normalization_companion(raw_sources[0])
        subgroup = raw_dgs_dataset_group(
            raw_sources,
            normalization_path=normalization,
            mask_path=normalization,
            progress_callback=progress_callback,
        )
        subgroup.name = _unique_name(
            subgroup.name,
            {item.name for item in group.iter_subgroups()},
        )
        (into.subgroups if into is not None else group.subgroups).append(subgroup)
        entries.extend(subgroup.datasets)

    for source in sources:
        if source in raw_sources:
            continue
        handler = group_import_handlers[source]
        if handler is not None and handler.group_import_mode == "mdevent":
            subgroup = import_mdevent_dataset_group(
                group,
                source,
                into=into,
                progress_callback=progress_callback,
            )
            entries.extend(subgroup.datasets)
            continue

        options = None
        if isinstance(importer_options, dict):
            options = importer_options.get(str(source))
        chosen = importer_name or default_importer_for_data_type(
            resolved_type, source
        )
        spec = IMPORTERS.get(chosen) if chosen is not None else None
        if spec is not None and spec.streams and not (
            isinstance(options, dict) and options.get("stream")
        ):
            parent = into if into is not None else group
            for stream in spec.streams:
                stream_options = (
                    copy.deepcopy(options) if isinstance(options, dict) else {}
                )
                stream_options["stream"] = stream.name
                entry = dataset_entry_from_path(
                    source,
                    data_type=stream.data_type,
                    importer_name=spec.name,
                    importer_options=stream_options,
                    dataset_file_loader=dataset_file_loader,
                )
                entry.name = _unique_dataset_name(
                    f"{entry.name} [{stream.label}]", group.dataset_names
                )
                target_group = _stream_target_group(
                    parent,
                    spec.name,
                    stream.name,
                    stream.label,
                    stream_groups,
                    reuse=stream_group_mode == "reuse",
                )
                group.add_dataset(entry, into=target_group)
                entries.append(entry)
                _adopt_imported_crystal(group, entry, target_group)
            continue

        entry = dataset_entry_from_path(
            source,
            data_type=data_type,
            importer_name=chosen,
            importer_options=options,
            dataset_file_loader=dataset_file_loader,
        )
        entry.name = _unique_dataset_name(entry.name, group.dataset_names)
        group.add_dataset(entry, into=into)
        entries.append(entry)
        _adopt_imported_crystal(group, entry, into)

    for entry in entries:
        if entry.data_type in {"single_crystal_inelastic", "powder_inelastic"}:
            entry.parameters.setdefault(
                SPECTRAL_CHANNEL_CONFIG_KEY,
                default_spectral_channel_config(),
            )
    return entries


def _stream_target_group(
    parent: DataGroup | DatasetGroup,
    importer_name: str,
    stream_name: str,
    stream_label: str,
    cached_groups: dict[tuple[str, str], DatasetGroup],
    *,
    reuse: bool,
) -> DatasetGroup:
    key = (importer_name, stream_name)
    target = cached_groups.get(key)
    if target is not None:
        return target
    if reuse:
        target = next(
            (
                candidate
                for candidate in parent.subgroups
                if _dataset_group_import_stream(candidate) == key
            ),
            None,
        )
    if target is None:
        target = DatasetGroup(
            name=_unique_name(
                _stream_group_name(importer_name, stream_label),
                {item.name for item in parent.subgroups},
            ),
            metadata={
                "importer": importer_name,
                "source_stream": stream_name,
                GROUP_COMPOSITE_KEY: {
                    "enabled": True,
                    "auto_rebin": True,
                    "stale": True,
                    "fractional": False,
                    "mean_weighting": "uniform",
                },
            },
        )
        parent.subgroups.append(target)
    else:
        target.metadata.setdefault("importer", importer_name)
        target.metadata.setdefault("source_stream", stream_name)
    cached_groups[key] = target
    return target


def _dataset_group_import_stream(group: DatasetGroup) -> tuple[str, str] | None:
    """Return the common importer/stream identity for a dataset group."""

    importer = str(group.metadata.get("importer", "")).strip()
    stream = str(group.metadata.get("source_stream", "")).strip().lower()
    if importer and stream:
        return importer, stream
    identities = [
        (
            str(dataset.metadata.get("importer", "")).strip(),
            str(
                (
                    dataset.metadata.get("import_options")
                    if isinstance(dataset.metadata.get("import_options"), dict)
                    else {}
                ).get("stream", "")
            )
            .strip()
            .lower(),
        )
        for dataset in group.datasets
    ]
    if not identities or any(
        not identity_importer or not identity_stream
        for identity_importer, identity_stream in identities
    ):
        return None
    unique_identities = set(identities)
    return next(iter(unique_identities)) if len(unique_identities) == 1 else None


def _adopt_imported_lattice(group: DataGroup, entry: DatasetEntry) -> None:
    """Adopt and validate lattice metadata supplied by a registered importer."""

    if entry.data is None or not isinstance(
        getattr(entry.data, "metadata", None), dict
    ):
        return
    imported = entry.data.metadata.get("lattice_parameters")
    if not isinstance(imported, dict):
        return
    values = {
        name: float(imported[name])
        for name in ("a", "b", "c", "alpha", "beta", "gamma")
        if name in imported
    }
    if len(values) != 6:
        return
    if group.lattice_parameters is None:
        group.lattice_parameters = values
        return
    if any(
        not np.isclose(float(group.lattice_parameters.get(name, np.nan)), value)
        for name, value in values.items()
    ):
        raise ValueError(
            f"{entry.name!r} has lattice parameters that differ from data group "
            f"{group.name!r}"
        )


def _adopt_imported_crystal(
    group: DataGroup,
    entry: DatasetEntry,
    target: DatasetGroup | None = None,
) -> None:
    """Promote compatible imported crystal metadata to its composite scope."""

    _adopt_imported_lattice(group, entry)
    metadata = getattr(entry.data, "metadata", None)
    if target is None or not isinstance(metadata, dict):
        return
    keys = (
        "lattice_parameters",
        "ub_matrix",
        "rlu_to_inv_angstrom_matrix",
        "orientation_u",
        "orientation_v",
    )
    for key in keys:
        if key not in metadata:
            continue
        imported = copy.deepcopy(metadata[key])
        existing = target.metadata.get(key)
        if existing is None:
            target.metadata[key] = imported
            continue
        try:
            matches = np.allclose(
                np.asarray(existing, dtype=float),
                np.asarray(imported, dtype=float),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        except (TypeError, ValueError):
            matches = existing == imported
        if not bool(matches):
            raise ValueError(
                f"{entry.name!r} has {key.replace('_', ' ')} that differs from "
                f"dataset group {target.name!r}"
            )
    lattice = target.metadata.get("lattice_parameters")
    ub = _dataset_ub_for_editor(target.metadata)
    if isinstance(lattice, dict) and ub is not None:
        target.metadata["ub_setup"] = {
            "ub_matrix": ub.tolist(),
            "lattice_parameters": copy.deepcopy(lattice),
            "u": copy.deepcopy(
                target.metadata.get("orientation_u", [1.0, 0.0, 0.0])
            ),
            "v": copy.deepcopy(
                target.metadata.get("orientation_v", [0.0, 1.0, 0.0])
            ),
        }


def _dataset_ub_for_editor(metadata: dict[str, Any]) -> np.ndarray | None:
    """Return UB without 2pi, converting full RLU-to-Q matrices when needed."""

    containers = [metadata]
    oriented = metadata.get("oriented_lattice")
    if isinstance(oriented, dict):
        containers.append(oriented)
    for container in containers:
        for key in ("ub_matrix", "orientation_matrix"):
            if key in container:
                matrix = np.asarray(container[key], dtype=float)
                if matrix.shape == (3, 3):
                    return matrix
    for container in containers:
        if "rlu_to_inv_angstrom_matrix" in container:
            matrix = np.asarray(
                container["rlu_to_inv_angstrom_matrix"], dtype=float
            )
            if matrix.shape == (3, 3):
                return matrix / (2.0 * np.pi)
    return None


def parse_dataset_numors(text: str) -> list[int]:
    """Parse comma-separated run numbers and inclusive ``start[:step]:end`` ranges."""

    values: list[int] = []
    seen: set[int] = set()
    for raw_piece in str(text).split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        parts = [part.strip() for part in piece.split(":")]
        try:
            if len(parts) == 1:
                expanded = [int(parts[0])]
            elif len(parts) == 2:
                start, end = (int(part) for part in parts)
                step = 1 if end >= start else -1
                expanded = list(range(start, end + step, step))
            elif len(parts) == 3:
                start, step, end = (int(part) for part in parts)
                if step == 0 or (end - start) * step < 0:
                    raise ValueError
                expanded = list(
                    range(start, end + (1 if step > 0 else -1), step)
                )
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"invalid run range {piece!r}; use 409981:409995, "
                "409981:3:409995, or comma-separated values"
            ) from exc
        for value in expanded:
            if value not in seen:
                seen.add(value)
                values.append(value)
    if not values:
        raise ValueError("enter at least one run number or range")
    return values


def set_dataset_source(dataset: DatasetEntry, path: str | Path) -> None:
    """Point a dataset entry at a new source file and mark loaded data stale."""

    source = Path(path)
    dataset.metadata["source_file"] = str(source)
    dataset.metadata["import_status"] = "pending"
    dataset.metadata.pop("source_point_count", None)
    dataset.kind = source.suffix.lstrip(".").lower()
    dataset.unload_data()
    _load_dataset_source(
        dataset,
        handler_names={"registered_importer", "point_list"},
    )


def set_dataset_data_type(
    dataset: DatasetEntry,
    data_type: str,
    *,
    importer_name: str | None = None,
) -> None:
    """Change a dataset's data type and reload/reset its data accordingly."""

    if data_type not in DATA_TYPE_DEFINITIONS:
        raise ValueError(f"unknown data type {data_type!r}")
    dataset.data_type = data_type
    dataset.metadata.pop("import_error", None)
    dataset.metadata.pop("source_point_count", None)
    if data_type in {"single_crystal_inelastic", "powder_inelastic"}:
        dataset.parameters.setdefault(
            SPECTRAL_CHANNEL_CONFIG_KEY,
            default_spectral_channel_config(),
        )
    chosen = importer_name or default_importer_for_data_type(
        data_type, dataset.metadata.get("source_file")
    )
    if chosen is not None:
        dataset.metadata["importer"] = chosen
        dataset.unload_data()
        if dataset.metadata.get("source_file"):
            try:
                _load_dataset_source(
                    dataset,
                    handler_names={"registered_importer"},
                )
            except Exception as exc:
                dataset.unload_data()
                dataset.metadata["import_status"] = "error"
                dataset.metadata["import_error"] = str(exc)
    elif data_type_container(data_type) == "point_list":
        chosen = default_importer_for_data_type(data_type)
        if chosen is not None:
            dataset.metadata["importer"] = chosen
        dataset.unload_data()
        if dataset.metadata.get("source_file"):
            try:
                _load_dataset_source(
                    dataset,
                    handler_names={"point_list"},
                )
            except Exception as exc:
                dataset.unload_data()
                dataset.metadata["import_status"] = "error"
                dataset.metadata["import_error"] = str(exc)
    elif chosen is None:
        dataset.metadata.pop("importer", None)
        if not isinstance(dataset.data, MDHistoData):
            dataset.unload_data()
            dataset.metadata["import_status"] = "pending"


def _load_registered_importer_dataset(
    dataset: DatasetEntry,
) -> PointData4D | PointListData | MDHistoData | None:
    """Load a dataset through its registered importer and apply import metadata."""

    source = (
        dataset.metadata.get("source_file")
        if isinstance(dataset.metadata, dict)
        else None
    )
    if not source:
        return None
    importer_name = dataset.metadata.get(
        "importer"
    ) or default_importer_for_data_type(dataset.data_type, source)
    if importer_name is None:
        return None
    options = dataset.metadata.get("import_options")
    data = import_with(
        importer_name,
        source,
        options if isinstance(options, dict) else None,
    )
    data = dataset.replace_data(data, source_backed=True)
    dataset.metadata["source_point_count"] = _loaded_data_point_count(data)
    dataset.metadata["importer"] = importer_name
    dataset.metadata["import_status"] = "loaded"
    imported_parameters = data.metadata.get("dataset_parameters")
    if isinstance(imported_parameters, dict) and not bool(
        dataset.metadata.get("import_parameters_applied", False)
    ):
        dataset.parameters.update(copy.deepcopy(imported_parameters))
        dataset.metadata["import_parameters_applied"] = True
    if dataset.data_type in {"magnetization", "heat_capacity"}:
        keys = ["sample_mass_mg", "molar_mass_g_mol"]
        if dataset.data_type == "heat_capacity":
            keys.append("atoms_per_formula_unit")
        for key in keys:
            if key not in dataset.parameters and key in data.metadata:
                dataset.parameters[key] = float(data.metadata[key])
    if dataset.data_type == "heat_capacity" and all(
        float(dataset.parameters.get(key, 0.0) or 0.0) > 0.0
        for key in ("sample_mass_mg", "molar_mass_g_mol")
    ):
        dataset.parameters.setdefault("absolute_units", True)
    if not dataset.kind:
        dataset.kind = Path(source).suffix.lstrip(".").lower()
    return data


def _load_point_list_dataset(dataset: DatasetEntry) -> PointListData | None:
    """Load point-list data from its source with a registered importer."""

    data = _load_registered_importer_dataset(dataset)
    if data is not None and not isinstance(data, PointListData):
        raise TypeError(
            f"importer for {dataset.data_type!r} did not return point-list data"
        )
    return data


def _dataset_source_path(dataset: DatasetEntry) -> Path | None:
    if not isinstance(dataset.metadata, dict):
        return None
    source = dataset.metadata.get("source_file")
    return Path(source) if source else None


def _group_import_handler(
    path: Path,
    data_type: str,
) -> _SourceFormatHandler | None:
    for handler in _SOURCE_FORMAT_HANDLERS:
        if (
            handler.matches_group_import is not None
            and handler.matches_group_import(path, data_type)
        ):
            return handler
    return None


def _load_dataset_source(
    dataset: DatasetEntry,
    *,
    mdhisto_loader: Callable[..., MDHistoData] | None = None,
    dataset_file_loader: DatasetFileLoader | None = None,
    handler_names: Collection[str] | None = None,
    entry_import: bool = False,
    progress_callback: Any | None = None,
) -> Any:
    """Load through the first applicable source handler.

    ``entry_import`` uses the declarative eager-import priorities. This keeps
    nfit ``.npz`` archives ahead of an explicitly stored importer during entry
    construction while preserving the established reload dispatch order.
    """

    context = _SourceLoadContext(
        mdhisto_loader=mdhisto_loader or load_mantid_mdhisto_nxs,
        dataset_file_loader=dataset_file_loader,
        progress_callback=progress_callback,
    )
    handlers = list(_SOURCE_FORMAT_HANDLERS)
    if handler_names is not None:
        handlers = [handler for handler in handlers if handler.name in handler_names]
    if entry_import:
        handlers = sorted(
            (
                handler
                for handler in handlers
                if handler.entry_import_priority is not None
            ),
            key=lambda handler: int(handler.entry_import_priority or 0),
        )
    for handler in handlers:
        loaded = handler.load(dataset, context)
        if loaded is not _SOURCE_NOT_HANDLED:
            return loaded
    return dataset.data


def _source_availability(
    dataset: DatasetEntry,
    *,
    operation: Literal["load", "reload"],
) -> tuple[_SourceFormatHandler | None, bool]:
    attribute = "can_load" if operation == "load" else "can_reload"
    for handler in _SOURCE_FORMAT_HANDLERS:
        verdict = getattr(handler, attribute)(dataset)
        if verdict is not None:
            return handler, bool(verdict)
    return None, False


def _artifact_source_load(
    dataset: DatasetEntry,
    _context: _SourceLoadContext,
) -> Any:
    if dataset.kind == "raw_dgs_nexus":
        return _SOURCE_NOT_HANDLED
    if not (
        dataset.metadata.get("derived_from_analysis")
        or dataset.metadata.get("project_artifact_path")
    ):
        return _SOURCE_NOT_HANDLED
    project_path = dataset.metadata.get("_project_path")
    artifact_path = dataset.metadata.get(
        "project_artifact_path"
    ) or dataset.metadata.get("analysis_artifact_path")
    if not project_path or not artifact_path:
        return _SOURCE_NOT_HANDLED
    loaded = read_project_dataset_artifact(project_path, artifact_path)
    loaded = dataset.replace_data(loaded, source_backed=True)
    dataset.metadata["import_status"] = "loaded"
    dataset.metadata.pop("import_error", None)
    return loaded


def _artifact_can_load(dataset: DatasetEntry) -> bool | None:
    if dataset.kind == "raw_dgs_nexus":
        return None
    if not dataset.metadata.get("derived_from_analysis"):
        return None
    source = _dataset_source_path(dataset)
    project_path = dataset.metadata.get("_project_path")
    artifact_path = dataset.metadata.get("analysis_artifact_path")
    return bool(
        source
        and project_path
        and artifact_path
        and project_artifact_exists(project_path, artifact_path)
    )


def _artifact_can_reload(dataset: DatasetEntry) -> bool | None:
    if not dataset.metadata.get("derived_from_analysis"):
        return None
    project_path = dataset.metadata.get("_project_path")
    artifact_path = dataset.metadata.get(
        "project_artifact_path"
    ) or dataset.metadata.get("analysis_artifact_path")
    return bool(
        project_path
        and artifact_path
        and project_artifact_exists(project_path, artifact_path)
    )


def _raw_dgs_source_load(
    dataset: DatasetEntry,
    _context: _SourceLoadContext,
) -> Any:
    return None if dataset.kind == "raw_dgs_nexus" else _SOURCE_NOT_HANDLED


def _raw_dgs_can_load(dataset: DatasetEntry) -> bool | None:
    return False if dataset.kind == "raw_dgs_nexus" else None


def _raw_dgs_can_reload(dataset: DatasetEntry) -> bool | None:
    if dataset.kind != "raw_dgs_nexus":
        return None
    return _dataset_source_path(dataset) is not None


def _raw_dgs_import_match(path: Path, data_type: str) -> bool:
    return data_type == "single_crystal_inelastic" and is_raw_dgs_nexus_file(path)


def _raw_dgs_reload_as_pending(dataset: DatasetEntry) -> bool:
    return dataset.kind == "raw_dgs_nexus"


def _registered_importer_source_load(
    dataset: DatasetEntry,
    _context: _SourceLoadContext,
) -> Any:
    if _dataset_source_path(dataset) is None or not dataset.metadata.get("importer"):
        return _SOURCE_NOT_HANDLED
    loaded = _load_registered_importer_dataset(dataset)
    return loaded if loaded is not None else _SOURCE_NOT_HANDLED


def _registered_importer_can_load(dataset: DatasetEntry) -> bool | None:
    if not dataset.metadata.get("importer"):
        return None
    source = _dataset_source_path(dataset)
    if source is None:
        return False
    if data_type_container(dataset.data_type) == "point_list":
        return True
    return source.suffix.lower() in _LAZY_SOURCE_SUFFIXES


def _registered_importer_can_reload(dataset: DatasetEntry) -> bool | None:
    if not dataset.metadata.get("importer"):
        return None
    return _dataset_source_path(dataset) is not None


def _point_list_source_load(
    dataset: DatasetEntry,
    _context: _SourceLoadContext,
) -> Any:
    if (
        _dataset_source_path(dataset) is None
        or data_type_container(dataset.data_type) != "point_list"
    ):
        return _SOURCE_NOT_HANDLED
    loaded = _load_point_list_dataset(dataset)
    return loaded if loaded is not None else _SOURCE_NOT_HANDLED


def _point_list_can_load(dataset: DatasetEntry) -> bool | None:
    if data_type_container(dataset.data_type) != "point_list":
        return None
    return _dataset_source_path(dataset) is not None


def _point_list_can_reload(dataset: DatasetEntry) -> bool | None:
    return _point_list_can_load(dataset)


def _npz_source_load(dataset: DatasetEntry, context: _SourceLoadContext) -> Any:
    source = _dataset_source_path(dataset)
    if source is None or source.suffix.lower() != ".npz":
        return _SOURCE_NOT_HANDLED
    if context.dataset_file_loader is None:
        raise ValueError("loading an nfit dataset archive requires a file loader")
    loaded, parameters = context.dataset_file_loader(source)
    dataset.parameters.update(parameters)
    loaded = dataset.replace_data(loaded, source_backed=True)
    dataset.kind = dataset.kind or source.suffix.lstrip(".").lower()
    dataset.metadata["import_status"] = "loaded"
    return loaded


def _extension_can_load(
    dataset: DatasetEntry,
    *,
    suffixes: Collection[str],
) -> bool | None:
    source = _dataset_source_path(dataset)
    if source is None or source.suffix.lower() not in suffixes:
        return None
    return True


def _npz_can_load(dataset: DatasetEntry) -> bool | None:
    return _extension_can_load(dataset, suffixes={".npz"})


def _npz_can_reload(dataset: DatasetEntry) -> bool | None:
    return _npz_can_load(dataset)


def _mdevent_source_load(dataset: DatasetEntry, context: _SourceLoadContext) -> Any:
    source = _dataset_source_path(dataset)
    if (
        dataset.kind != "mdevent"
        or source is None
        or source.suffix.lower() not in _LAZY_SOURCE_SUFFIXES
    ):
        return _SOURCE_NOT_HANDLED
    loaded = load_mdevent_run_points(
        dataset,
        progress_callback=context.progress_callback,
    )
    loaded = dataset.replace_data(loaded, source_backed=True)
    dataset.kind = dataset.kind or source.suffix.lstrip(".").lower()
    dataset.metadata["import_status"] = "loaded"
    return loaded


def _mdevent_can_load(dataset: DatasetEntry) -> bool | None:
    if dataset.kind != "mdevent":
        return None
    return _extension_can_load(dataset, suffixes=_LAZY_SOURCE_SUFFIXES)


def _mdevent_can_reload(dataset: DatasetEntry) -> bool | None:
    return _mdevent_can_load(dataset)


def _mdevent_import_match(path: Path, data_type: str) -> bool:
    return data_type == "single_crystal_inelastic" and is_mdevent_file(path)


def _mdhisto_source_load(dataset: DatasetEntry, context: _SourceLoadContext) -> Any:
    source = _dataset_source_path(dataset)
    if source is None or source.suffix.lower() not in _MDHISTO_SOURCE_SUFFIXES:
        return _SOURCE_NOT_HANDLED
    loaded = context.mdhisto_loader(source, copy_metadata=False)
    loaded = dataset.replace_data(loaded, source_backed=True)
    dataset.kind = dataset.kind or source.suffix.lstrip(".").lower()
    dataset.metadata["import_status"] = "loaded"
    return loaded


def _mdhisto_can_load(dataset: DatasetEntry) -> bool | None:
    return _extension_can_load(dataset, suffixes=_MDHISTO_SOURCE_SUFFIXES)


def _mdhisto_can_reload(dataset: DatasetEntry) -> bool | None:
    return _mdhisto_can_load(dataset)


_SOURCE_FORMAT_HANDLERS = (
    _SourceFormatHandler(
        "project_artifact",
        _artifact_source_load,
        _artifact_can_load,
        _artifact_can_reload,
    ),
    _SourceFormatHandler(
        "raw_dgs",
        _raw_dgs_source_load,
        _raw_dgs_can_load,
        _raw_dgs_can_reload,
        group_import_mode="raw_dgs",
        matches_group_import=_raw_dgs_import_match,
        reload_as_pending=_raw_dgs_reload_as_pending,
    ),
    _SourceFormatHandler(
        "registered_importer",
        _registered_importer_source_load,
        _registered_importer_can_load,
        _registered_importer_can_reload,
        entry_import_priority=10,
    ),
    _SourceFormatHandler(
        "point_list",
        _point_list_source_load,
        _point_list_can_load,
        _point_list_can_reload,
        entry_import_priority=20,
    ),
    _SourceFormatHandler(
        "nfit_npz",
        _npz_source_load,
        _npz_can_load,
        _npz_can_reload,
        entry_import_priority=0,
    ),
    _SourceFormatHandler(
        "mdevent",
        _mdevent_source_load,
        _mdevent_can_load,
        _mdevent_can_reload,
        group_import_mode="mdevent",
        matches_group_import=_mdevent_import_match,
    ),
    _SourceFormatHandler(
        "mdhisto",
        _mdhisto_source_load,
        _mdhisto_can_load,
        _mdhisto_can_reload,
    ),
)


def dataset_entry_from_path(
    path: str | Path,
    *,
    data_type: str | None = None,
    importer_name: str | None = None,
    importer_options: dict[str, Any] | None = None,
    dataset_file_loader: DatasetFileLoader | None = None,
) -> DatasetEntry:
    """Create a project dataset entry for one selected file."""

    source = Path(path)
    resolved_type = data_type or DEFAULT_DATA_TYPE
    entry = DatasetEntry(
        name=_unique_dataset_name(source.stem or source.name, []),
        data=None,
        kind=source.suffix.lstrip(".").lower(),
        data_type=resolved_type,
        metadata={"source_file": str(source), "import_status": "pending"},
    )
    if resolved_type in {"single_crystal_inelastic", "powder_inelastic"}:
        entry.parameters[SPECTRAL_CHANNEL_CONFIG_KEY] = (
            default_spectral_channel_config()
        )
    if importer_name is not None:
        entry.metadata["importer"] = importer_name
        if importer_options is not None:
            entry.metadata["import_options"] = copy.deepcopy(importer_options)
    if source.suffix.lower() != ".npz":
        chosen = importer_name or default_importer_for_data_type(
            resolved_type, source
        )
        if chosen is not None:
            entry.metadata["importer"] = chosen
        elif data_type_container(resolved_type) == "point_list":
            chosen = default_importer_for_data_type(resolved_type)
            if chosen is not None:
                entry.metadata["importer"] = chosen
    _load_dataset_source(
        entry,
        dataset_file_loader=dataset_file_loader,
        entry_import=True,
    )
    return entry


def _ensure_dataset_data_loaded(
    dataset: DatasetEntry,
    *,
    mdhisto_loader: Callable[..., MDHistoData] | None = None,
    dataset_file_loader: DatasetFileLoader | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Return canonical loaded data, using one path for all lazy consumers."""

    if dataset.data is not None:
        return dataset.data
    return _load_dataset_source(
        dataset,
        mdhisto_loader=mdhisto_loader,
        dataset_file_loader=dataset_file_loader,
        progress_callback=progress_callback,
    )


def _dataset_can_load(dataset: DatasetEntry) -> bool:
    if dataset.data is not None:
        return False
    _handler, available = _source_availability(dataset, operation="load")
    return available


def _dataset_can_reload(dataset: DatasetEntry | None) -> bool:
    if dataset is None or not isinstance(dataset.metadata, dict):
        return False
    _handler, available = _source_availability(dataset, operation="reload")
    return available


def _reload_dataset_copy(
    dataset: DatasetEntry,
    *,
    data_loader: DatasetDataLoader | None = None,
) -> DatasetEntry:
    if not _dataset_can_reload(dataset):
        raise ValueError(f"dataset {dataset.name!r} has no reloadable data source")
    working = copy.copy(dataset)
    working.metadata = copy.deepcopy(dataset.metadata)
    working.parameters = copy.deepcopy(dataset.parameters)
    working.unload_data()
    if any(
        handler.reload_as_pending is not None
        and handler.reload_as_pending(working)
        for handler in _SOURCE_FORMAT_HANDLERS
    ):
        working.metadata["import_status"] = "pending"
        working.metadata.pop("import_error", None)
    else:
        loader = data_loader or _ensure_dataset_data_loaded
        loaded = loader(working)
        if loaded is None:
            raise ValueError(
                f"dataset {dataset.name!r} could not be loaded from its source"
            )
    return working


def _install_reloaded_dataset(dataset: DatasetEntry, working: DatasetEntry) -> Any:
    dataset.metadata.clear()
    dataset.metadata.update(working.metadata)
    dataset.parameters.clear()
    dataset.parameters.update(working.parameters)
    dataset.kind = working.kind
    return dataset.replace_data(working.data, source_backed=True)


def reload_dataset_data(
    dataset: DatasetEntry,
    *,
    data_loader: DatasetDataLoader | None = None,
) -> Any:
    """Reload one dataset without mutating it when reading the source fails."""

    working = _reload_dataset_copy(dataset, data_loader=data_loader)
    return _install_reloaded_dataset(dataset, working)


def reload_data_group(
    group: DataGroup | DatasetGroup,
    *,
    data_loader: DatasetDataLoader | None = None,
) -> list[DatasetEntry]:
    """Reload every descendant dataset that has a configured data source."""

    reloaded = []
    for dataset in group.iter_datasets():
        if _dataset_can_reload(dataset):
            reload_dataset_data(dataset, data_loader=data_loader)
            reloaded.append(dataset)
    return reloaded


def _loaded_data_point_count(data: Any) -> int:
    """Return the stored-point count for a loaded nfit data container."""

    if isinstance(data, MDHistoData):
        return int(np.prod(data.shape))
    if isinstance(data, (PointListData, PointData4D)):
        return int(data.size)
    return 0
