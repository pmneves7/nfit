from __future__ import annotations

from typing import Any

import numpy as np

from ..cross_section import (
    KB_MEV_PER_K,
    MILLIBARN_PER_BARN,
    kf_over_ki,
)
from ..dataset import PointListData
from ..mdhisto import MDHistoChannel, MDHistoData
from .coordinates import (
    bin_edges,
    bin_widths,
    measured_mask,
    physical_axis_vectors,
    physical_coordinate_arrays,
    q_bin_volume,
    rlu_to_q_matrix,
)
from .core import DatasetOutput, ScalarOutput
from .corrections import SpectralConvention
from .zones import generate_zone_centers, nearest_zone_indices, reciprocal_basis_hkl


def _kinematic_factor(
    energy_meV: np.ndarray,
    convention: SpectralConvention,
) -> float | np.ndarray:
    if convention.kf_ki_state == "removed":
        return 1.0
    return kf_over_ki(
        energy_meV,
        incident_energy_meV=convention.incident_energy_meV,
        final_energy_meV=convention.final_energy_meV,
    )


def _cross_section_to_barn(
    values: np.ndarray,
    unit: str,
) -> np.ndarray:
    normalized = unit.lower().replace(" ", "")
    if "mbarn" in normalized:
        return np.asarray(values, dtype=float) / MILLIBARN_PER_BARN
    if "barn" in normalized:
        return np.asarray(values, dtype=float)
    raise ValueError(
        "absolute cross-section input units must explicitly contain 'barn' or 'mbarn'"
    )


def convert_spectral_representation(
    data: MDHistoData,
    *,
    convention: SpectralConvention,
    target_representation: str,
    temperature_K: float | np.ndarray,
    scale: float = 1.0,
    background: float | np.ndarray = 0.0,
    form_factor_sq: float | np.ndarray = 1.0,
    polarization: float | np.ndarray = 1.0,
) -> MDHistoData:
    """Convert a complete INS dataset between cross section and ``chi''``.

    ``scale`` is measured signal per barn/(sr meV) for an arbitrary measured
    intensity input.  The conversion is linear, so one-sigma uncertainties are
    propagated with the same absolute factor.  The output convention and every
    correction state are recorded in metadata.
    """

    if target_representation not in {"cross_section", "chi_double_prime"}:
        raise ValueError("target_representation must be cross_section or chi_double_prime")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if convention.normalization_basis == "unknown":
        raise ValueError("absolute INS conversion requires a known normalization basis")
    if convention.representation != "measured_intensity":
        convention.require_absolute("absolute INS conversion")
    energy_dims = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(energy_dims) != 1:
        raise ValueError("spectral conversion requires exactly one energy-transfer axis")
    dim = energy_dims[0]
    centers = data.axes[dim].centers
    shape = [1] * data.signal.ndim
    shape[dim] = centers.size
    energy = centers.reshape(shape)
    from ..spectral_conversion import (
        convert_histogram_spectral_values,
        spectral_conversion_factor,
    )

    source_scale = (
        1.0 / scale if convention.representation == "measured_intensity"
        else float(_cross_section_to_barn(np.asarray(1.0), convention.unit))
        if convention.representation == "cross_section" else 1.0
    )
    factor = spectral_conversion_factor(
        energy, temperature_K,
        source_representation=convention.representation,
        target_representation=target_representation,
        source_scale=source_scale,
        form_factor_sq=form_factor_sq, polarization=polarization,
        kf_ki=_kinematic_factor(energy, convention),
        form_factor_state=convention.form_factor_state,
        polarization_state=convention.polarization_state,
        bose_state=convention.bose_state,
        moment_unit=convention.moment_unit, g_factor=convention.g_factor,
    )
    unit = (
        "barn/(sr meV)" if target_representation == "cross_section"
        else "mu_B^2/meV" if convention.moment_unit == "mu_B_squared"
        else "spin^2/meV"
    )
    target = SpectralConvention(
        representation=target_representation, unit=unit,
        normalization_basis=convention.normalization_basis,
        magnetic_ions_per_basis=convention.magnetic_ions_per_basis,
        moment_unit=convention.moment_unit, g_factor=convention.g_factor,
        form_factor_state="included" if target_representation == "cross_section" else "removed",
        polarization_state="included" if target_representation == "cross_section" else "removed",
        bose_state="included" if target_representation == "cross_section" else "removed",
        kf_ki_state=convention.kf_ki_state, absolute_scale=True,
        incident_energy_meV=convention.incident_energy_meV,
        final_energy_meV=convention.final_energy_meV,
    )
    quantity_type = (
        "differential_cross_section" if target_representation == "cross_section"
        else "dynamic_susceptibility"
    )
    temperature = np.asarray(temperature_K, dtype=float)
    temperature_record = float(temperature) if temperature.ndim == 0 else temperature.tolist()
    pol = np.asarray(polarization, dtype=float)
    pol_record = float(pol) if pol.ndim == 0 else pol.tolist()
    metadata = dict(data.metadata)
    metadata.pop("combined_scientific_identity", None)
    metadata.update(
        signal_quantity_type=quantity_type, signal_unit=unit,
        signal_label=("Scattering cross section" if target_representation == "cross_section" else "Dynamical susceptibility χ″"),
        temperature=temperature_record,
        spectral_convention=target.to_dict(),
        spectral_observable={
            **target.to_dict(),
            "source_representation": target_representation,
            "source_unit": unit,
            "fit_representation": target_representation,
            "polarization_mode": "custom_scalar" if pol.ndim == 0 else "custom_array",
            "polarization_scalar": pol_record,
            "quantity_type": quantity_type,
            "temperature_K": temperature_record,
        },
        spectral_conversion={
            "source_convention": convention.to_dict(),
            "temperature_K": temperature_record,
            "scale_signal_per_barn_sr_meV": float(scale),
        },
    )
    offset = 0.0
    if convention.representation == "measured_intensity":
        background_values = np.asarray(background, dtype=float)
        if np.any(background_values != 0.0):
            with np.errstate(invalid="ignore"):
                offset = np.where(background_values == 0.0, 0.0, -background_values * factor)
    return convert_histogram_spectral_values(
        data, factor, offset=offset, unit=unit,
        quantity_type=quantity_type, metadata=metadata,
    )


def spectral_kernel(name: str, energy_meV: np.ndarray, temperature_K: float, *, power: int = 0) -> np.ndarray:
    energy = np.asarray(energy_meV, dtype=float)
    if name in {"qfi", "total_moment"}:
        temperature = np.asarray(temperature_K, dtype=float)
        if np.any(~np.isfinite(temperature)) or np.any(temperature < 0):
            raise ValueError("temperature_K must be finite and nonnegative")
        energy, temperature = np.broadcast_arrays(energy, temperature)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = energy / (2.0 * KB_MEV_PER_K * temperature)
            if name == "qfi":
                finite_temperature = 4.0 * np.tanh(x) / np.pi
                zero_temperature = np.where(energy > 0, 4.0 / np.pi, 0.0)
            else:
                finite_temperature = 1.0 / (np.pi * np.tanh(x))
                zero_temperature = np.where(energy > 0, 1.0 / np.pi, np.inf)
        return np.where(temperature == 0.0, zero_temperature, finite_temperature)
    if name == "static_susceptibility":
        return 2.0 / (np.pi * energy)
    if name == "energy_moment":
        return energy ** int(power)
    if name == "weighted_integral_arbitrary_units":
        return np.ones_like(energy)
    raise KeyError(f"unknown spectral kernel {name!r}")


def spectral_energy_reduce(
    data: MDHistoData,
    *,
    kernel: str,
    convention: SpectralConvention,
    temperature_K: float,
    energy_min_meV: float = 0.0,
    energy_max_meV: float = np.inf,
    scale: float = 1.0,
    form_factor_sq: float | np.ndarray = 1.0,
    polarization: float | np.ndarray = 1.0,
    minimum_energy_coverage: float = 0.9,
    power: int = 0,
    progress_callback=None,
    cancel_callback=None,
    spin_S: float | None = None,
) -> DatasetOutput | ScalarOutput:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError("Spectral integration cancelled")
    if progress_callback is not None:
        progress_callback({"stage": "spectral_energy_reduction", "completed": 0, "total": 1})
    if kernel != "weighted_integral_arbitrary_units":
        convention.require_absolute(kernel)
    energy_dims = [i for i, axis in enumerate(data.axes) if axis.kind == "energy" or axis.role == "energy_transfer"]
    if len(energy_dims) != 1:
        raise ValueError("spectral integration requires exactly one energy-transfer axis")
    dim = energy_dims[0]
    axis = data.axes[dim]
    edges = bin_edges(axis, data.shape[dim])
    lower = np.maximum(edges[:-1], energy_min_meV)
    upper = np.minimum(edges[1:], energy_max_meV)
    widths = np.maximum(upper - lower, 0.0)
    centers = 0.5 * (lower + upper)
    active = widths > 0
    if not np.any(active):
        raise ValueError("energy window does not overlap the dataset")
    shape = [1] * data.signal.ndim
    shape[dim] = data.shape[dim]
    E = centers.reshape(shape)
    dE = widths.reshape(shape)
    values = np.asarray(data.signal, dtype=float)
    errors = np.asarray(data.errors, dtype=float)
    if kernel == "weighted_integral_arbitrary_units":
        chipp, sigma = values, errors
        correction_valid = np.ones_like(values, dtype=bool)
    else:
        from ..spectral_conversion import spectral_conversion_factor

        source_scale = (
            1.0 / scale if convention.representation == "measured_intensity"
            else float(_cross_section_to_barn(np.asarray(1.0), convention.unit))
            if convention.representation == "cross_section" else 1.0
        )
        if not np.isfinite(source_scale) or source_scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        factor = spectral_conversion_factor(
            E, temperature_K, source_representation=convention.representation,
            target_representation="chi_double_prime", source_scale=source_scale,
            form_factor_sq=form_factor_sq, polarization=polarization,
            kf_ki=_kinematic_factor(E, convention),
            form_factor_state=convention.form_factor_state,
            polarization_state=convention.polarization_state,
            bose_state=convention.bose_state,
            moment_unit=convention.moment_unit, g_factor=convention.g_factor,
        )
        chipp, sigma = values * factor, errors * np.abs(factor)
        correction_valid = np.isfinite(factor)
    weight = spectral_kernel(kernel, E, temperature_K, power=power)
    valid = measured_mask(data) & correction_valid & np.isfinite(chipp) & np.isfinite(sigma) & np.isfinite(weight) & (dE > 0)
    contribution = np.where(valid, weight * chipp * dE, 0.0)
    variance = np.where(valid, (weight * sigma * dE) ** 2, 0.0)
    result = np.sum(contribution, axis=dim)
    uncertainty = np.sqrt(np.sum(variance, axis=dim))
    output_unit = convention.moment_unit
    if kernel == "qfi" and convention.moment_unit == "mu_B_squared":
        if convention.g_factor is None:
            raise ValueError("spin-operator QFI from mu_B^2 data requires an explicit g_factor")
        result = result / convention.g_factor**2
        uncertainty = uncertainty / convention.g_factor**2
        output_unit = "spin_squared"
    normalized = normalized_uncertainty = None
    if kernel == "qfi" and spin_S is not None:
        if spin_S <= 0:
            raise ValueError("spin_S must be positive")
        normalized = result / (12.0 * spin_S**2)
        normalized_uncertainty = uncertainty / (12.0 * spin_S**2)
    requested_width = float(np.sum(widths))
    coverage = np.sum(np.where(valid, dE, 0.0), axis=dim) / requested_width
    absolute_weight = np.where(np.isfinite(weight) & (dE > 0), np.abs(weight) * dE, 0.0)
    weight_denominator = np.sum(absolute_weight, axis=dim)
    weighted_coverage = np.divide(
        np.sum(np.where(valid, absolute_weight, 0.0), axis=dim),
        weight_denominator,
        out=np.zeros_like(coverage, dtype=float),
        where=weight_denominator > 0,
    )
    masked = coverage < minimum_energy_coverage
    remaining_axes = tuple(axis for i, axis in enumerate(data.axes) if i != dim)
    metadata = {**data.metadata, "analysis": {"kernel": kernel, "temperature_K": np.asarray(temperature_K).tolist(), "energy_window_meV": [energy_min_meV, energy_max_meV], "convention": convention.to_dict(), "invalid_correction_bins": int(np.size(correction_valid) - np.count_nonzero(correction_valid)), "normalized_qfi_formula": "f_Q / (12 S^2)" if normalized is not None else None, "spin_S": spin_S}, "energy_coverage": coverage.tolist() if np.ndim(coverage) else float(coverage)}
    label = {"qfi": "Quantum Fisher information", "total_moment": "Total moment"}.get(kernel, kernel.replace("_", " ").title())
    if progress_callback is not None:
        progress_callback({"stage": "spectral_energy_reduction", "completed": 1, "total": 1})
    if not remaining_axes:
        if normalized is not None:
            metadata["normalized_qfi"] = float(normalized)
            metadata["dnormalized_qfi"] = float(normalized_uncertainty)
        return ScalarOutput(float(result), float(uncertainty), output_unit, label, metadata)
    if len(remaining_axes) == 1:
        coordinate = remaining_axes[0].centers
        columns = {remaining_axes[0].name: coordinate, label: result, f"d{label}": uncertainty, "energy_coverage": coverage, "weighted_coverage": weighted_coverage}
        channels = [{"label": label, "value": label, "error": f"d{label}"}]
        if normalized is not None:
            columns.update({"Normalized QFI": normalized, "dNormalized QFI": normalized_uncertainty})
            channels.append({"label": "Normalized QFI", "value": "Normalized QFI", "error": "dNormalized QFI"})
        table = PointListData(columns, units={remaining_axes[0].name: remaining_axes[0].units}, coordinate_names=[remaining_axes[0].name], channels=channels, metadata=metadata)
        return DatasetOutput(table, label, "analysis_curve", metadata)
    auxiliary = {"energy_coverage": MDHistoChannel(coverage, label="Energy coverage", unit="fraction"), "weighted_coverage": MDHistoChannel(weighted_coverage, label="Weighted coverage", unit="fraction")}
    if normalized is not None:
        auxiliary["normalized_qfi"] = MDHistoChannel(normalized, normalized_uncertainty, "Normalized QFI", "dimensionless")
    output = MDHistoData(remaining_axes, result, uncertainty, masked, np.where(masked, 0.0, 1.0), coordinate_system=data.coordinate_system, visual_normalization=data.visual_normalization, metadata=metadata, auxiliary_channels=auxiliary)
    return DatasetOutput(output, label, "analysis_map", metadata)


def integrate_total_moment_by_zone(
    data: MDHistoData,
    *,
    convention: SpectralConvention,
    temperature_K: float,
    spacegroup: str,
    minimum_zone_coverage: float = 0.8,
    zone_basis_hkl: np.ndarray | None = None,
    zone_centers_hkl: np.ndarray | None = None,
    zone_subvoxel_samples: int = 3,
    progress_callback=None,
    cancel_callback=None,
    **reduce_parameters: Any,
) -> tuple[PointListData, ScalarOutput]:
    reduced = spectral_energy_reduce(
        data, kernel="total_moment", convention=convention,
        temperature_K=temperature_K, **reduce_parameters,
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    )
    if not isinstance(reduced, DatasetOutput) or not isinstance(reduced.data, MDHistoData) or reduced.data.signal.ndim != 3:
        raise ValueError("per-zone total moment requires three momentum axes after energy reduction")
    moment = reduced.data
    coords = physical_coordinate_arrays(moment)
    points = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1).reshape(-1, 3)
    qmatrix = rlu_to_q_matrix(moment.metadata)
    basis = np.asarray(zone_basis_hkl, dtype=float) if zone_basis_hkl is not None else reciprocal_basis_hkl(spacegroup, qmatrix)
    centers = (
        np.asarray(zone_centers_hkl, dtype=float).reshape(-1, 3)
        if zone_centers_hkl is not None and np.asarray(zone_centers_hkl).size
        else generate_zone_centers(basis, points.min(axis=0), points.max(axis=0))
    )
    if centers.size == 0:
        raise ValueError("explicit zone centers cannot be empty")
    if zone_subvoxel_samples < 1 or zone_subvoxel_samples % 2 != 1:
        raise ValueError("zone_subvoxel_samples must be a positive odd integer")
    volumes = q_bin_volume(moment)
    ideal_volume = abs(float(np.linalg.det(basis @ qmatrix.T)))
    valid = ~moment.mask & np.isfinite(moment.signal) & np.isfinite(moment.errors)
    count = len(centers)
    measured_by_zone = np.zeros(count)
    accumulated_by_zone = np.zeros(count)
    variance_by_zone = np.zeros(count)
    offsets = (np.arange(zone_subvoxel_samples) + 0.5) / zone_subvoxel_samples - 0.5
    vectors = physical_axis_vectors(moment)[:, :3]
    width_arrays = []
    for dim, axis in enumerate(moment.axes):
        shape = [1] * moment.signal.ndim
        shape[dim] = moment.shape[dim]
        width_arrays.append(bin_widths(axis, moment.shape[dim]).reshape(shape))
    sample_weight = volumes / zone_subvoxel_samples**3
    sample_index = 0
    sample_total = zone_subvoxel_samples**3
    for offset0 in offsets:
        for offset1 in offsets:
            for offset2 in offsets:
                if cancel_callback is not None and cancel_callback():
                    raise RuntimeError("Brillouin-zone integration cancelled")
                if progress_callback is not None:
                    progress_callback({"stage": "brillouin_zone_assignment", "completed": sample_index, "total": sample_total})
                shifted = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1).copy()
                for offset, widths_for_axis, vector in zip((offset0, offset1, offset2), width_arrays, vectors, strict=True):
                    shifted += offset * widths_for_axis[..., None] * vector
                assignments = nearest_zone_indices(shifted.reshape(-1, 3), centers, qmatrix)
                measured_by_zone += np.bincount(assignments, weights=np.where(valid, sample_weight, 0.0).ravel(), minlength=count)
                accumulated_by_zone += np.bincount(assignments, weights=np.where(valid, moment.signal * sample_weight, 0.0).ravel(), minlength=count)
                # Treat a voxel's sub-samples as fully correlated measurements.
                # This is exact away from zone boundaries and conservative for
                # boundary-crossing voxels.
                variance_by_zone += np.bincount(assignments, weights=np.where(valid, (moment.errors * volumes) ** 2 / zone_subvoxel_samples**3, 0.0).ravel(), minlength=count)
                sample_index += 1
    rows = []
    accepted_values = []
    accepted_variances = []
    for index, center in enumerate(centers):
        measured_volume = float(measured_by_zone[index])
        if measured_volume <= 0:
            continue
        coverage = min(measured_volume / ideal_volume, 1.0) if ideal_volume > 0 else 0.0
        accumulated = float(accumulated_by_zone[index])
        variance = float(variance_by_zone[index])
        partial = accumulated / ideal_volume
        dpartial = np.sqrt(variance) / ideal_volume
        corrected = accumulated / measured_volume if measured_volume else np.nan
        dcorrected = np.sqrt(variance) / measured_volume if measured_volume else np.nan
        accepted = coverage >= minimum_zone_coverage
        if accepted and np.isfinite(corrected) and dcorrected > 0:
            accepted_values.append(corrected)
            accepted_variances.append(dcorrected**2)
        rows.append((*center, partial, dpartial, corrected, dcorrected, coverage, measured_volume, ideal_volume, float(accepted), 0.0 if accepted else 1.0))
    names = ("center_H", "center_K", "center_L", "moment_partial", "dmoment_partial", "moment_coverage_corrected", "dmoment_coverage_corrected", "coverage", "measured_Q_volume", "ideal_BZ_volume", "accepted", "status_code")
    columns = dict(zip(names, np.asarray(rows, dtype=float).T, strict=True))
    table = PointListData(columns, coordinate_names=["center_H", "center_K", "center_L"], channels=[{"label": "Partial moment", "value": "moment_partial", "error": "dmoment_partial"}, {"label": "Coverage-corrected moment", "value": "moment_coverage_corrected", "error": "dmoment_coverage_corrected"}], metadata={"zone_basis_hkl": basis.tolist(), "spacegroup": spacegroup})
    if accepted_values:
        weights = 1.0 / np.asarray(accepted_variances)
        value = float(np.sum(weights * accepted_values) / np.sum(weights))
        uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
    else:
        value = uncertainty = np.nan
    scalar = ScalarOutput(value, uncertainty, convention.moment_unit, "Accepted-zone total moment", {"accepted_zones": len(accepted_values)})
    return table, scalar
