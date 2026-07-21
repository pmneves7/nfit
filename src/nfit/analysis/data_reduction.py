from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ..dataset import PointData4D
from ..mdhisto import MDHistoAxis, MDHistoData, mdhisto_measured_bins
from .coordinates import q_modulus_for_spectral
from .core import AnalysisContext

K_B_MEV_PER_K = 0.08617333262145


def separate_bose_elastic(
    first: MDHistoData,
    second: MDHistoData,
    *,
    first_temperature_K: float,
    second_temperature_K: float,
    zero_energy_tolerance_meV: float = 1.0e-12,
) -> tuple[MDHistoData, MDHistoData]:
    """Separate temperature-independent and Bose-scaled components.

    The first output is the inelastic signal at ``first_temperature_K`` and
    the second is the temperature-independent elastic/background component.
    Input datasets are treated as statistically independent.
    """

    _validate_matching_histograms(first, second)
    t1 = _positive_temperature(first_temperature_K, "first")
    t2 = _positive_temperature(second_temperature_K, "second")
    if np.isclose(t1, t2, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError("Bose separation requires two different temperatures")
    energy_dim = _energy_dimension(first)
    energy = np.asarray(first.axes[energy_dim].centers, dtype=float)
    factor1 = _bose_scattering_factor(energy, t1)
    factor2 = _bose_scattering_factor(energy, t2)
    shape = [1] * first.signal.ndim
    shape[energy_dim] = energy.size
    factor1 = factor1.reshape(shape)
    factor2 = factor2.reshape(shape)
    with np.errstate(invalid="ignore"):
        contrast = factor2 - factor1
    zero_energy = np.abs(energy).reshape(shape) <= float(zero_energy_tolerance_meV)
    valid_contrast = np.isfinite(contrast) & (np.abs(contrast) > 1.0e-12) & ~zero_energy

    signal1 = np.asarray(first.signal, dtype=float)
    signal2 = np.asarray(second.signal, dtype=float)
    variance1 = np.square(np.asarray(first.errors, dtype=float))
    variance2 = np.square(np.asarray(second.errors, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coefficient1 = -factor1 / contrast
        coefficient2 = factor1 / contrast
        inelastic_signal = coefficient1 * signal1 + coefficient2 * signal2
        inelastic_variance = coefficient1**2 * variance1 + coefficient2**2 * variance2
        elastic_coefficient1 = factor2 / contrast
        elastic_coefficient2 = -factor1 / contrast
        elastic_signal = elastic_coefficient1 * signal1 + elastic_coefficient2 * signal2
        elastic_variance = (
            elastic_coefficient1**2 * variance1
            + elastic_coefficient2**2 * variance2
        )

    # The Bose factor is singular exactly at zero transfer. The physically
    # conservative decomposition there assigns the bin to the elastic channel.
    inelastic_signal = np.where(zero_energy, 0.0, inelastic_signal)
    inelastic_variance = np.where(zero_energy, 0.0, inelastic_variance)
    elastic_signal = np.where(zero_energy, signal1, elastic_signal)
    elastic_variance = np.where(zero_energy, variance1, elastic_variance)
    shared_valid = (
        mdhisto_measured_bins(first)
        & mdhisto_measured_bins(second)
        & np.isfinite(signal1)
        & np.isfinite(signal2)
        & np.isfinite(variance1)
        & np.isfinite(variance2)
        & (valid_contrast | zero_energy)
    )
    metadata = dict(first.metadata)
    metadata.update(
        {
            "signal_semantics": "density",
            "bose_separation": {
                "first_temperature_K": t1,
                "second_temperature_K": t2,
                "zero_energy_tolerance_meV": float(zero_energy_tolerance_meV),
                "assumption": "temperature_independent_elastic_plus_bose_scaled_inelastic",
            },
        }
    )
    common = {
        "mask": ~shared_valid,
        "num_events": np.minimum(first.num_events, second.num_events),
        "metadata": metadata,
    }
    inelastic = replace(
        first,
        signal=np.where(shared_valid, inelastic_signal, np.nan),
        errors=np.where(shared_valid, np.sqrt(np.maximum(inelastic_variance, 0.0)), np.nan),
        **common,
    )
    elastic = replace(
        first,
        signal=np.where(shared_valid, elastic_signal, np.nan),
        errors=np.where(shared_valid, np.sqrt(np.maximum(elastic_variance, 0.0)), np.nan),
        **common,
    )
    return inelastic, elastic


def spherical_average(
    data: MDHistoData,
    context: AnalysisContext,
    *,
    q_bins: int = 100,
) -> MDHistoData:
    """Return an inverse-variance spherical average on a ``|Q|, E`` grid."""

    if int(q_bins) < 1:
        raise ValueError("q_bins must be positive")
    energy_dim = _energy_dimension(data)
    q = np.broadcast_to(q_modulus_for_spectral(data, context), data.shape)
    energy_shape = [1] * data.signal.ndim
    energy_shape[energy_dim] = data.shape[energy_dim]
    energy = np.broadcast_to(data.axes[energy_dim].centers.reshape(energy_shape), data.shape)
    valid = (
        mdhisto_measured_bins(data)
        & np.isfinite(data.signal)
        & np.isfinite(data.errors)
        & (data.errors > 0.0)
        & np.isfinite(q)
        & np.isfinite(energy)
    )
    if not np.any(valid):
        raise ValueError("spherical averaging found no measured bins with finite uncertainty")
    energy_edges = _axis_edges(data.axes[energy_dim], data.shape[energy_dim])
    q_values = q[valid]
    q_lower = max(0.0, float(np.nanmin(q_values)))
    q_upper = float(np.nanmax(q_values))
    if not q_upper > q_lower:
        raise ValueError("spherical averaging requires a nonzero |Q| range")
    q_edges = np.linspace(q_lower, q_upper, int(q_bins) + 1)
    q_indices = np.searchsorted(q_edges, q, side="right") - 1
    q_indices[q == q_edges[-1]] = int(q_bins) - 1
    energy_index_shape = [1] * data.signal.ndim
    energy_index_shape[energy_dim] = data.shape[energy_dim]
    source_energy_indices = np.broadcast_to(
        np.arange(data.shape[energy_dim]).reshape(energy_index_shape), data.shape
    )
    energy_count = int(energy_edges.size - 1)
    valid &= (q_indices >= 0) & (q_indices < int(q_bins))
    flat_indices = q_indices[valid] * energy_count + source_energy_indices[valid]
    inverse_variance = 1.0 / np.square(data.errors[valid])
    output_size = int(q_bins) * energy_count
    weight_sum = np.bincount(
        flat_indices,
        weights=inverse_variance,
        minlength=output_size,
    ).reshape(int(q_bins), energy_count)
    weighted_signal = np.bincount(
        flat_indices,
        weights=inverse_variance * data.signal[valid],
        minlength=output_size,
    ).reshape(int(q_bins), energy_count)
    counts = np.bincount(flat_indices, minlength=output_size).reshape(
        int(q_bins), energy_count
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = weighted_signal / weight_sum
        errors = np.sqrt(1.0 / weight_sum)
    measured = np.isfinite(signal) & np.isfinite(errors)
    return MDHistoData(
        axes=(
            MDHistoAxis("|Q|", q_edges, "1/angstrom", "momentum", frame="Q modulus"),
            MDHistoAxis(
                data.axes[energy_dim].name,
                energy_edges,
                data.axes[energy_dim].units,
                "energy",
                frame=data.axes[energy_dim].frame,
            ),
        ),
        signal=signal,
        errors=errors,
        mask=~measured,
        num_events=np.asarray(counts, dtype=float),
        metadata={
            **dict(data.metadata),
            "signal_semantics": "density",
            "spherical_average": {
                "q_bins": int(q_bins),
                "weighting": "inverse_variance",
            },
        },
    )


def angle_energy_background(
    runs: Sequence[PointData4D],
    *,
    q_bins: int = 100,
    energy_bins: int = 100,
    lowest_fraction: float = 0.2,
) -> MDHistoData:
    """Estimate a rotation-independent ``|Q|, E`` background.

    Each run is first reduced independently with proton-charge normalization.
    In every output bin, run intensities are sorted and the lowest requested
    fraction is averaged. Statistical variances of the selected runs are
    propagated; uncertainty in the selection/order statistic is not included.
    """

    if len(runs) < 2:
        raise ValueError("angle-energy background requires at least two runs")
    if int(q_bins) < 1 or int(energy_bins) < 1:
        raise ValueError("q_bins and energy_bins must be positive")
    fraction = float(lowest_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("lowest_fraction must be in (0, 1]")
    q_arrays = [_point_q_modulus(run) for run in runs]
    valid_arrays = [
        run.valid_mask() & np.isfinite(q)
        for run, q in zip(runs, q_arrays, strict=True)
    ]
    if not any(np.any(valid) for valid in valid_arrays):
        raise ValueError("angle-energy background found no valid events")
    q_lower = max(0.0, min(float(np.nanmin(q[valid])) for q, valid in zip(q_arrays, valid_arrays, strict=True) if np.any(valid)))
    q_upper = max(float(np.nanmax(q[valid])) for q, valid in zip(q_arrays, valid_arrays, strict=True) if np.any(valid))
    e_lower = min(float(np.nanmin(run.E[valid])) for run, valid in zip(runs, valid_arrays, strict=True) if np.any(valid))
    e_upper = max(float(np.nanmax(run.E[valid])) for run, valid in zip(runs, valid_arrays, strict=True) if np.any(valid))
    if not q_upper > q_lower:
        q_padding = max(abs(q_lower) * 1.0e-6, 1.0e-6)
        q_lower = max(0.0, q_lower - q_padding)
        q_upper += q_padding
    if not e_upper > e_lower:
        e_padding = max(abs(e_lower) * 1.0e-6, 1.0e-6)
        e_lower -= e_padding
        e_upper += e_padding
    q_edges = np.linspace(q_lower, q_upper, int(q_bins) + 1)
    e_edges = np.linspace(e_lower, e_upper, int(energy_bins) + 1)
    signals = []
    variances = []
    measured = []
    for run, q, valid in zip(runs, q_arrays, valid_arrays, strict=True):
        charge = float(run.metadata.get("proton_charge", 1.0) or 1.0)
        if not np.isfinite(charge) or charge <= 0.0:
            raise ValueError("every angle-energy input requires positive proton_charge metadata")
        coordinates = np.column_stack((q[valid], run.E[valid]))
        sums = np.histogramdd(coordinates, bins=(q_edges, e_edges), weights=run.intensity[valid])[0]
        variance = np.histogramdd(coordinates, bins=(q_edges, e_edges), weights=np.square(run.sigma[valid]))[0]
        counts = np.histogramdd(coordinates, bins=(q_edges, e_edges))[0]
        signals.append(sums / charge)
        variances.append(variance / charge**2)
        measured.append(counts > 0.0)
    signal_stack = np.stack(signals)
    variance_stack = np.stack(variances)
    measured_stack = np.stack(measured)
    output_signal = np.full(signal_stack.shape[1:], np.nan)
    output_variance = np.full(signal_stack.shape[1:], np.nan)
    selected_count = np.zeros(signal_stack.shape[1:], dtype=float)
    for index in np.ndindex(output_signal.shape):
        available = np.flatnonzero(measured_stack[(slice(None), *index)])
        if available.size == 0:
            continue
        count = max(1, int(np.ceil(fraction * available.size)))
        order = available[np.argsort(signal_stack[(available, *index)])]
        selected = order[:count]
        output_signal[index] = float(np.mean(signal_stack[(selected, *index)]))
        output_variance[index] = float(np.sum(variance_stack[(selected, *index)]) / count**2)
        selected_count[index] = count
    output_mask = ~(np.isfinite(output_signal) & np.isfinite(output_variance))
    return MDHistoData(
        axes=(
            MDHistoAxis("|Q|", q_edges, "1/angstrom", "momentum", frame="Q modulus"),
            MDHistoAxis("DeltaE", e_edges, "meV", "energy", frame="General Frame"),
        ),
        signal=output_signal,
        errors=np.sqrt(np.maximum(output_variance, 0.0)),
        mask=output_mask,
        num_events=selected_count,
        metadata={
            "signal_semantics": "density",
            "angle_energy_background": {
                "run_count": len(runs),
                "lowest_fraction": fraction,
                "normalization": "proton_charge",
                "uncertainty": "propagated_selected_run_statistics_only",
            },
        },
    )


def _validate_matching_histograms(first: MDHistoData, second: MDHistoData) -> None:
    if first.shape != second.shape or len(first.axes) != len(second.axes):
        raise ValueError("Bose separation requires identically binned datasets")
    for first_axis, second_axis in zip(first.axes, second.axes, strict=True):
        if (
            first_axis.name != second_axis.name
            or first_axis.units != second_axis.units
            or first_axis.values.shape != second_axis.values.shape
            or not np.allclose(first_axis.values, second_axis.values, rtol=1.0e-10, atol=1.0e-12)
        ):
            raise ValueError("Bose separation requires identical axis names, units, and bins")


def _energy_dimension(data: MDHistoData) -> int:
    dimensions = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(dimensions) != 1:
        raise ValueError("analysis requires exactly one energy-transfer axis")
    return dimensions[0]


def _positive_temperature(value: float, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} dataset temperature must be positive and finite")
    return result


def _bose_scattering_factor(energy_meV: np.ndarray, temperature_K: float) -> np.ndarray:
    energy = np.asarray(energy_meV, dtype=float)
    magnitude = np.abs(energy)
    x = magnitude / (K_B_MEV_PER_K * float(temperature_K))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        loss = -1.0 / np.expm1(-x)
        gain = 1.0 / np.expm1(x)
    return np.where(energy >= 0.0, loss, gain)


def _axis_edges(axis: MDHistoAxis, size: int) -> np.ndarray:
    values = np.asarray(axis.values, dtype=float)
    if values.size == size + 1:
        return values.copy()
    if values.size != size:
        raise ValueError(f"axis {axis.name!r} does not match its data dimension")
    if size == 1:
        width = float(axis.metadata.get("bin_width", 1.0))
        return np.array([values[0] - width / 2.0, values[0] + width / 2.0])
    edges = np.empty(size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def _point_q_modulus(data: PointData4D) -> np.ndarray:
    hkl = np.column_stack((data.H, data.K, data.L))
    metadata: dict[str, Any] = data.metadata
    units = str(metadata.get("coordinate_units", "")).lower()
    if "angstrom" in units and "rlu" not in units and "r.l.u" not in units:
        q = hkl
    else:
        matrix = metadata.get("rlu_to_inv_angstrom_matrix")
        if matrix is None:
            matrix = metadata.get("ub_matrix")
            if matrix is not None:
                matrix = 2.0 * np.pi * np.asarray(matrix, dtype=float)
        if matrix is None:
            raise ValueError("angle-energy background requires UB or RLU-to-Q metadata")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("angle-energy background Q matrix must be 3x3")
        q = hkl @ matrix.T
    return np.linalg.norm(q, axis=1)
