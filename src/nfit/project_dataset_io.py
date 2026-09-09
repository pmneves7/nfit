"""Portable dataset archive I/O shared by project and scripting workflows.

The archive codec in this module has no Qt dependency.  The public save helper
resolves a dataset's prepared view through :mod:`nfit.project_data` only when
``use_view`` is requested, keeping the historical API and output unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analysis.coordinates import signal_semantics
from .dataset import PointListData
from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from .pipeline import DatasetEntry
from .spectral_channels import SPECTRAL_CHANNEL_CONFIG_KEY

KINEMATIC_KF_KI_INCLUDED_KEY = "kf_ki_included"


def save_dataset_file(
    dataset: DatasetEntry,
    path: str | Path,
    *,
    use_view: bool = True,
) -> None:
    """Save a supported dataset to a portable, re-importable ``.npz`` file."""

    if use_view:
        # Imported lazily to avoid a module-import cycle: project_data provides
        # the preparation facade, while its source loaders use this codec.
        from .project_data import dataset_for_slice_viewer

        data = dataset_for_slice_viewer(dataset)
    else:
        data = dataset.data
    if isinstance(data, PointListData):
        _save_point_list_file(data, path)
        return
    if not isinstance(data, MDHistoData):
        raise TypeError(
            "dataset saving currently supports MDHistoData or PointListData datasets"
        )
    saved_metadata = dict(data.metadata)
    saved_metadata.setdefault("signal_semantics", signal_semantics(data))
    payload: dict[str, Any] = {
        "nfit_dataset_format": np.asarray("nfit-dataset"),
        "nfit_dataset_version": np.asarray(3, dtype=int),
        "nfit_data_container": np.asarray("mdhisto"),
        "signal": data.signal,
        "errors": data.errors,
        "mask": data.mask,
        "num_events": data.num_events,
        "metadata_json": json.dumps(_json_safe_value(saved_metadata), sort_keys=True),
        "axis_count": np.asarray(len(data.axes), dtype=int),
        "coordinate_system": np.asarray(
            -1 if data.coordinate_system is None else data.coordinate_system
        ),
        "visual_normalization": np.asarray(
            -1 if data.visual_normalization is None else data.visual_normalization
        ),
    }
    context = {
        key: copy.deepcopy(dataset.parameters[key])
        for key in (
            "temperature",
            "magnetic_field",
            KINEMATIC_KF_KI_INCLUDED_KEY,
            SPECTRAL_CHANNEL_CONFIG_KEY,
        )
        if key in dataset.parameters
    }
    observable = data.metadata.get("spectral_observable")
    if isinstance(observable, dict) and SPECTRAL_CHANNEL_CONFIG_KEY in context:
        saved_config = dict(context[SPECTRAL_CHANNEL_CONFIG_KEY])
        saved_config["source_representation"] = str(
            observable.get(
                "fit_representation", saved_config["source_representation"]
            )
        )
        saved_config["source_unit"] = (
            str(data.metadata.get("signal_unit", "arbitrary")) or "arbitrary"
        )
        saved_config["signal_per_mbarn"] = 0.0
        context[SPECTRAL_CHANNEL_CONFIG_KEY] = saved_config
    payload["dataset_context_json"] = np.asarray(
        json.dumps(_json_safe_value(context), sort_keys=True)
    )
    for index, axis in enumerate(data.axes):
        payload[f"axis_{index}_values"] = axis.values
        payload[f"axis_{index}_name"] = np.asarray(axis.name)
        payload[f"axis_{index}_units"] = np.asarray(axis.units)
        payload[f"axis_{index}_kind"] = np.asarray(axis.kind)
        payload[f"axis_{index}_frame"] = np.asarray(axis.frame or "")
        payload[f"axis_{index}_path"] = np.asarray(axis.path or "")
        payload[f"axis_{index}_metadata_json"] = np.asarray(
            json.dumps(_json_safe_value(axis.metadata), sort_keys=True)
        )
    payload["auxiliary_channel_names_json"] = np.asarray(
        json.dumps(list(data.auxiliary_channels))
    )
    for index, channel in enumerate(data.auxiliary_channels.values()):
        payload[f"auxiliary_{index}_values"] = channel.values
        if channel.errors is not None:
            payload[f"auxiliary_{index}_errors"] = channel.errors
        payload[f"auxiliary_{index}_label"] = np.asarray(channel.label)
        payload[f"auxiliary_{index}_unit"] = np.asarray(channel.unit)
        payload[f"auxiliary_{index}_quantity_type"] = np.asarray(
            channel.quantity_type
        )
    np.savez_compressed(path, **payload)


def _save_point_list_file(data: PointListData, path: str | Path) -> None:
    """Save point-list columns and roles to a script-readable ``.npz`` file."""

    payload: dict[str, Any] = {
        "nfit_dataset_format": np.asarray("nfit-dataset"),
        "nfit_dataset_version": np.asarray(1, dtype=int),
        "nfit_data_container": np.asarray("point_list"),
        "column_names_json": json.dumps(list(data.column_names)),
        "coordinate_names_json": json.dumps(list(data.coordinate_names)),
        "channels_json": json.dumps(_json_safe_value(data.channels)),
        "units_json": json.dumps(_json_safe_value(data.units)),
        "quantity_types_json": json.dumps(_json_safe_value(data.quantity_types)),
        "metadata_json": json.dumps(_json_safe_value(data.metadata), sort_keys=True),
    }
    for index, name in enumerate(data.column_names):
        payload[f"column_{index}"] = np.asarray(data.column(name), dtype=float)
    np.savez_compressed(path, **payload)


def _load_nfit_dataset_file(
    path: str | Path,
) -> tuple[MDHistoData | PointListData, dict[str, Any]]:
    """Load an nfit dataset archive written by :func:`save_dataset_file`."""

    source = Path(path)
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read nfit dataset archive {source}") from exc
    with archive:
        files = set(archive.files)
        if {"signal", "errors", "mask", "num_events", "axis_count"} <= files:
            data = _load_nfit_mdhisto_archive(archive, source)
        elif {
            "column_names_json",
            "coordinate_names_json",
            "channels_json",
            "units_json",
        } <= files:
            data = _load_nfit_point_list_archive(archive, source)
        else:
            raise ValueError(f"{source} is not an nfit dataset archive")
        context = _nfit_archive_json_mapping(archive, "dataset_context_json")
    return data, context


def _load_nfit_mdhisto_archive(archive: Any, source: Path) -> MDHistoData:
    axis_count = int(np.asarray(archive["axis_count"]).item())
    axes = []
    for index in range(axis_count):
        prefix = f"axis_{index}_"
        values_key = f"{prefix}values"
        if values_key not in archive:
            raise ValueError(f"{source} is missing {values_key}")
        axes.append(
            MDHistoAxis(
                name=_nfit_archive_text(archive, f"{prefix}name", f"Axis {index}"),
                values=np.asarray(archive[values_key], dtype=float),
                units=_nfit_archive_text(archive, f"{prefix}units"),
                kind=_nfit_archive_text(archive, f"{prefix}kind", "unknown"),
                frame=_nfit_archive_text(archive, f"{prefix}frame") or None,
                path=_nfit_archive_text(archive, f"{prefix}path") or None,
                metadata=_nfit_archive_json_mapping(
                    archive, f"{prefix}metadata_json"
                ),
            )
        )
    metadata = _nfit_archive_json_mapping(archive, "metadata_json")
    metadata["export_file"] = str(source)
    if "signal_semantics" not in metadata:
        raise ValueError(
            f"{source} is missing signal_semantics metadata; "
            "re-export it with a current nfit version"
        )
    channel_names = json.loads(
        _nfit_archive_text(archive, "auxiliary_channel_names_json", "[]")
    )
    auxiliary_channels = {
        str(name): MDHistoChannel(
            values=np.asarray(archive[f"auxiliary_{index}_values"], dtype=float),
            errors=(
                np.asarray(archive[f"auxiliary_{index}_errors"], dtype=float)
                if f"auxiliary_{index}_errors" in archive
                else None
            ),
            label=_nfit_archive_text(
                archive, f"auxiliary_{index}_label", str(name)
            ),
            unit=_nfit_archive_text(archive, f"auxiliary_{index}_unit"),
            quantity_type=_nfit_archive_text(
                archive, f"auxiliary_{index}_quantity_type", "unknown"
            ),
        )
        for index, name in enumerate(channel_names)
    }
    return MDHistoData(
        axes=tuple(axes),
        signal=np.asarray(archive["signal"], dtype=float),
        errors=np.asarray(archive["errors"], dtype=float),
        mask=np.asarray(archive["mask"], dtype=bool),
        num_events=np.asarray(archive["num_events"], dtype=float),
        coordinate_system=_nfit_archive_optional_int(archive, "coordinate_system"),
        visual_normalization=_nfit_archive_optional_int(
            archive, "visual_normalization"
        ),
        metadata=metadata,
        auxiliary_channels=auxiliary_channels,
    )


def _load_nfit_point_list_archive(archive: Any, source: Path) -> PointListData:
    names = json.loads(_nfit_archive_text(archive, "column_names_json"))
    columns = {
        str(name): np.asarray(archive[f"column_{index}"], dtype=float)
        for index, name in enumerate(names)
    }
    metadata = _nfit_archive_json_mapping(archive, "metadata_json")
    metadata["export_file"] = str(source)
    return PointListData(
        columns=columns,
        units=_nfit_archive_json_mapping(archive, "units_json"),
        coordinate_names=json.loads(
            _nfit_archive_text(archive, "coordinate_names_json")
        ),
        channels=json.loads(_nfit_archive_text(archive, "channels_json")),
        metadata=metadata,
        quantity_types=_nfit_archive_json_mapping(archive, "quantity_types_json"),
    )


def _nfit_archive_text(archive: Any, key: str, default: str = "") -> str:
    return str(np.asarray(archive[key]).item()) if key in archive else default


def _nfit_archive_optional_int(archive: Any, key: str) -> int | None:
    """Read an optional archive integer, where ``-1`` represents ``None``."""

    if key not in archive:
        return None
    value = int(np.asarray(archive[key]).item())
    return None if value < 0 else value


def _nfit_archive_json_mapping(archive: Any, key: str) -> dict[str, Any]:
    if key not in archive:
        return {}
    try:
        value = json.loads(_nfit_archive_text(archive, key))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON field {key!r} in nfit dataset archive") from exc
    return dict(value) if isinstance(value, dict) else {}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
