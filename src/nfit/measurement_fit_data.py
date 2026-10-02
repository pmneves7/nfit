"""Public fit preparation retaining measurement contracts and statistics."""
from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import PointData4D
from .mdhisto import MDHistoData, mdhisto_measured_bins
from .project_coordinates import _mdhisto_coordinate_grids
from .source_lineage import SOURCE_LINEAGE_KEY


def rebin_declared_fit_points(data, *, lower=None, upper=None, step_size=None,
                             num_bins=None, bin_edges=None):
    """Prepare physical point-bin edges using the authoritative grid planner."""
    from .measurement_point_bins import bin_measurement_points
    from .rebin import NDRebin

    coordinates = np.column_stack(data.coordinates())
    selected = data.mask & np.all(np.isfinite(coordinates), axis=1)
    if not np.any(selected):
        raise ValueError("no finite eligible coordinates remain before rebinning")
    planner = NDRebin(data.intensity[selected], coordinates[selected], lower=lower,
                     upper=upper, step_size=step_size, num_bins=num_bins, bin_edges=bin_edges,
                     fractional=False)
    # These planning methods are shared with NDRebin.run; no accumulation or
    # output-sized arrays are allocated to discover the requested grid.
    planner.Ndims = 4
    planner.coords_flat = coordinates[selected]
    planner._build_limits()
    planner._make_bins()
    histogram = bin_measurement_points(data, planner.bins_list)
    return prepare_histogram_fit_points(histogram)


def prepare_histogram_fit_points(data: MDHistoData) -> PointData4D:
    """Flatten a masked MDHisto view into fit points using axis roles."""

    coords = _mdhisto_coordinate_grids(data)
    reduction = data.metadata.get("measurement_reduction", {})
    sampled_dimension = reduction.get("coordinate_dimension") if reduction.get("assignment") == "piecewise_linear_original_nodes" else None
    sampled_slot = None
    if sampled_dimension is not None and data.axes[int(sampled_dimension)].role not in {"h","k","l","energy"}:
        dim = int(sampled_dimension)
        sampled_slot = ("H","K","L","E")[dim]
        shape = [1]*data.signal.ndim
        shape[dim] = data.shape[dim]
        coords[sampled_slot] = np.broadcast_to(data.axes[dim].centers.reshape(shape),data.shape)
    zeros = np.zeros(data.shape, dtype=float)
    keep = ~np.asarray(data.mask, dtype=bool)
    keep &= mdhisto_measured_bins(data)
    metadata: dict[str, Any] = {
        "fit_coordinates": sorted(
            name for name in ("H", "K", "L", "E", "q_modulus") if name in coords
        ),
    }
    if sampled_slot is not None:
        axis = data.axes[int(sampled_dimension)]
        metadata["fit_coordinate_slots"] = {sampled_slot:{"name":axis.name,"unit":axis.units,"physical_momentum_energy":False}}
    for key in (
        "oriented_lattice",
        "coordinate_units",
        "rlu_to_inv_angstrom_matrix",
        "nfit_kinematic_kf_ki_normalized",
        "nfit_kinematic_kf_ki_source",
        "signal_quantity_type",
        "signal_unit",
        "spectral_observable",
        "measurement_contract", "measurement_statistics", "event_statistics", "num_events_semantics",
        "poisson_count_model", "background_subtractions", "background_projection", "background_channels_version",
        "symmetry_operations_hkl", "rebin", "histogram_arithmetic", "bose_separation", "fit_likelihood",
        "measurement_target_required", "measurement_derivation", "cross_reflection_covariance",
        "measurement_reduction", "measurement_source_id", "coordinate_name",
        SOURCE_LINEAGE_KEY,
    ):
        if key in data.metadata:
            metadata[key] = data.metadata[key]
    if "signal_quantity_type" in metadata:
        metadata["quantity_type"] = metadata["signal_quantity_type"]
    if "signal_unit" in metadata:
        metadata["unit"] = metadata["signal_unit"]
    temperature = data.metadata.get("temperature")
    from .metadata_dimensions import metadata_temperature_grid

    temperatures = metadata_temperature_grid(data)
    powder_q = coords.get("q_modulus")
    powder_only = powder_q is not None and not any(
        axis.role in {"h", "k", "l"} for axis in data.axes
    )
    if powder_only:
        metadata["coordinate_units"] = "1/angstrom"
        metadata["powder_q_modulus_axis"] = True
    from .histogram_statistics import (
        EVENT_STATISTICS_CHANNELS,
        NORMALIZATION_DENOMINATOR,
        selected_event_statistics,
    )
    from .measurement_aggregation import (
        MEASUREMENT_STATISTICS_CHANNELS,
        selected_measurement_statistics,
    )
    payload = {}
    event_stats = selected_event_statistics(data)
    generic_stats = selected_measurement_statistics(data)
    if event_stats is not None:
        payload.update(zip((*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR), (array.ravel() for array in event_stats), strict=True))
    elif generic_stats is not None:
        payload.update(zip(MEASUREMENT_STATISTICS_CHANNELS, (array.ravel() for array in generic_stats), strict=True))
    dependencies = getattr(data, "source_dependencies", None)
    if dependencies is not None:
        from .measurement_dependencies import project_source_dependencies
        flat = np.arange(data.signal.size)
        dependencies = project_source_dependencies(dependencies, flat, flat, np.ones(flat.size), (flat.size,))
    metadata["fit_measurement_provenance"] = {"version": 1, "contract": "explicit" if "measurement_contract" in metadata else "derived_target_required" if metadata.get("measurement_target_required") else "legacy_unspecified",
        "sufficient_statistics": "validated" if payload else "unavailable", "default_likelihood": "gaussian"}
    return PointData4D(
        # PointData4D has a Cartesian-vector momentum slot. For powder data,
        # place |Q| on x and tag the coordinates as inverse angstroms so
        # q_modulus_inv_angstrom recovers the measured scalar without a lattice.
        H=(
            powder_q
            if powder_only
            else coords.get("H", zeros)
        ).ravel(),
        K=coords.get("K", zeros).ravel(),
        L=coords.get("L", zeros).ravel(),
        E=coords.get("E", zeros).ravel(),
        intensity=np.asarray(data.signal, dtype=float).ravel(),
        sigma=np.asarray(data.errors, dtype=float).ravel(),
        mask=keep.ravel(),
        temperature=(
            np.broadcast_to(temperatures, data.shape).ravel() if temperatures is not None
            else float(temperature) if temperature is not None else None
        ),
        metadata=metadata,
        normalization_denominator=payload.get(NORMALIZATION_DENOMINATOR),
        measurement_payload=payload or None,
        source_dependencies=dependencies,
    )
