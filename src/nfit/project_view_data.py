"""Immutable transformations that prepare project datasets for viewers and fits."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .mdhisto import MDHistoData
from .pipeline import DatasetEntry
from .spectral_channels import SPECTRAL_CHANNEL_CONFIG_KEY, with_paired_spectral_channels

KINEMATIC_KF_KI_INCLUDED_KEY = "kf_ki_included"


def _apply_spectral_channel_view(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    """Apply paired INS channels, falling back to the legacy kinematic path."""

    config = dataset.parameters.get(SPECTRAL_CHANNEL_CONFIG_KEY)
    if (
        isinstance(data, MDHistoData)
        and dataset.data_type in {"single_crystal_inelastic", "powder_inelastic"}
        and isinstance(config, dict)
    ):
        temperature = dataset.parameters.get("temperature", data.metadata.get("temperature"))
        from .metadata_dimensions import metadata_temperature_grid

        temperatures = metadata_temperature_grid(data)
        return with_paired_spectral_channels(
            data,
            config,
            temperature_K=(
                temperatures
                if temperatures is not None
                else None
                if temperature in (None, "")
                else float(temperature)
            ),
        )
    return _apply_kinematic_normalization_to_view(dataset, data)


def _with_viewer_dataset_metadata(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    metadata = dict(getattr(data, "metadata", {}) or {})
    metadata["nfit_data_type"] = dataset.data_type
    metadata["nfit_dataset_kind"] = dataset.kind
    if isinstance(data, MDHistoData):
        return replace(data, metadata=metadata)
    if isinstance(data, PointListData):
        return PointListData(
            columns={name: np.array(values, dtype=float) for name, values in data.columns.items()},
            units=dict(data.units),
            coordinate_names=list(data.coordinate_names),
            channels=[dict(channel) for channel in data.channels],
            metadata=metadata,
            quantity_types=dict(data.quantity_types),
        )
    if isinstance(data, PointData4D):
        return data.with_updates(metadata=metadata)
    return data


def _kinematic_energy_metadata(
    dataset: DatasetEntry,
    data_metadata: dict[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Find scalar incident/final energies recorded on a dataset or its data."""

    sources = (dataset.parameters, dataset.metadata, data_metadata or {})

    def value_for(names: tuple[str, ...]) -> float | None:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for name in names:
                try:
                    value = float(source[name])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    return value
        return None

    return (
        value_for(("incident_energy", "incident_energy_meV", "Ei", "ei")),
        value_for(("final_energy", "final_energy_meV", "Ef", "ef")),
    )


def _kinematic_kf_ki_factor(
    energy_transfer_meV: Any,
    *,
    incident_energy_meV: float | None,
    final_energy_meV: float | None,
) -> np.ndarray | None:
    """Return ``k_f/k_i`` for ``E = E_i - E_f``, if one energy is known."""

    if incident_energy_meV is None and final_energy_meV is None:
        return None
    energy = np.asarray(energy_transfer_meV, dtype=float)
    if incident_energy_meV is not None:
        ratio_sq = (float(incident_energy_meV) - energy) / float(incident_energy_meV)
    else:
        ratio_sq = float(final_energy_meV) / (float(final_energy_meV) + energy)
    factor = np.full(energy.shape, np.nan, dtype=float)
    np.sqrt(ratio_sq, out=factor, where=np.isfinite(ratio_sq) & (ratio_sq >= 0.0))
    return factor


def _apply_kinematic_normalization_to_view(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    """Normalize binned data to the cross-section ``k_f/k_i`` convention."""

    if bool(dataset.parameters.get(KINEMATIC_KF_KI_INCLUDED_KEY, True)):
        return data
    if isinstance(data, PointData4D):
        return _apply_kinematic_normalization_to_points(dataset, data)
    if not isinstance(data, MDHistoData):
        return data
    energy_dim = next(
        (index for index, axis in enumerate(data.axes) if axis.kind == "energy"),
        None,
    )
    if energy_dim is None:
        return data
    incident, final = _kinematic_energy_metadata(dataset, data.metadata)
    factor_1d = _kinematic_kf_ki_factor(
        data.axes[energy_dim].centers,
        incident_energy_meV=incident,
        final_energy_meV=final,
    )
    if factor_1d is None:
        return data
    shape = [1] * data.signal.ndim
    shape[energy_dim] = factor_1d.size
    factor = factor_1d.reshape(shape)
    metadata = dict(data.metadata)
    metadata["nfit_kinematic_kf_ki_normalized"] = True
    metadata["nfit_kinematic_kf_ki_source"] = "Ei" if incident is not None else "Ef"
    return replace(
        data,
        signal=np.asarray(data.signal, dtype=float) * factor,
        errors=np.asarray(data.errors, dtype=float) * np.abs(factor),
        metadata=metadata,
    )


def _apply_kinematic_normalization_to_points(
    dataset: DatasetEntry,
    points: PointData4D,
) -> PointData4D:
    """Return fit points in the selected kinematic convention."""

    if bool(dataset.parameters.get(KINEMATIC_KF_KI_INCLUDED_KEY, True)):
        return points
    if points.metadata.get("nfit_kinematic_kf_ki_normalized"):
        return points
    incident, final = _kinematic_energy_metadata(dataset, points.metadata)
    factor = _kinematic_kf_ki_factor(
        points.E,
        incident_energy_meV=incident,
        final_energy_meV=final,
    )
    if factor is None:
        return points
    metadata = dict(points.metadata)
    metadata["nfit_kinematic_kf_ki_normalized"] = True
    metadata["nfit_kinematic_kf_ki_source"] = "Ei" if incident is not None else "Ef"
    return points.with_updates(
        intensity=np.asarray(points.intensity, dtype=float) * factor,
        sigma=np.asarray(points.sigma, dtype=float) * np.abs(factor),
        metadata=metadata,
    )


def _mdhisto_without_nfit_masks(data: MDHistoData) -> MDHistoData:
    """Return a cheap file-mask-only view while manual nfit masks are pending."""

    file_mask = np.asarray(data.mask, dtype=bool)
    metadata = dict(data.metadata)
    metadata["file_mask"] = file_mask
    metadata["nfit_mask"] = np.zeros(0, dtype=bool)
    metadata["file_mask_count"] = int(np.count_nonzero(file_mask))
    metadata["nfit_mask_count"] = 0
    metadata["combined_mask_count"] = metadata["file_mask_count"]
    metadata["mask_application_pending"] = True
    return replace(data, mask=file_mask, metadata=metadata)


def _apply_dataset_scale(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    """Multiply a dataset's signal and errors by its scale factor (both channels)."""

    scale = float(getattr(dataset, "scale_factor", 1.0) or 1.0)
    if scale == 1.0:
        return data
    if isinstance(data, MDHistoData):
        return replace(
            data,
            signal=np.asarray(data.signal, dtype=float) * scale,
            errors=np.asarray(data.errors, dtype=float) * abs(scale),
        )
    if isinstance(data, PointListData):
        columns = {name: np.array(values, dtype=float) for name, values in data.columns.items()}
        for channel in data.channels:
            value_name = channel.get("value")
            error_name = channel.get("error")
            if value_name in columns:
                columns[value_name] = columns[value_name] * scale
            if error_name in columns:
                columns[error_name] = columns[error_name] * abs(scale)
        return PointListData(
            columns=columns,
            units=dict(data.units),
            coordinate_names=list(data.coordinate_names),
            channels=[dict(channel) for channel in data.channels],
            metadata=dict(data.metadata),
            quantity_types=dict(data.quantity_types),
        )
    if isinstance(data, PointData4D):
        return data.with_updates(
            intensity=np.asarray(data.intensity, dtype=float) * scale,
            sigma=np.asarray(data.sigma, dtype=float) * abs(scale),
        )
    return data
