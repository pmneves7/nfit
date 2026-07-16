from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset import PointListData
from ..form_factors import form_factor_sq as evaluate_form_factor_sq
from ..mdhisto import MDHistoData
from .bragg import generate_bragg_peaks, integrate_bragg_peaks
from .coordinates import q_modulus_for_spectral
from .core import AnalysisExecution, TableOutput
from .corrections import SpectralConvention
from .registry import (
    AnalysisOperationDefinition,
    AnalysisParameterDefinition,
    register_analysis_operation,
)
from .spectral import (
    convert_spectral_representation,
    integrate_total_moment_by_zone,
    spectral_energy_reduce,
)


def _p(name: str, default: Any, description: str, *, kind: str = "value", choices=(), required=False) -> AnalysisParameterDefinition:
    return AnalysisParameterDefinition(name, name.replace("_", " ").title(), kind, default, description, "See tooltip and validation.", repr(default), tuple(choices), required)


def register_builtin_operations() -> None:
    bragg_parameters = (
        _p("peak_source", "crystal", "Generate peaks from crystal symmetry or use an input table.", choices=(("crystal", "Crystal"), ("table", "Table"))),
        _p("peak_table_dataset_id", "", "Dataset ID containing H, K, and L columns."),
        _p("peak_positions_hkl", [], "Script-provided H, K, L peak positions."),
        _p("include_systematic_absences", False, "Include reflections forbidden by the space group.", kind="bool"),
        _p("d_min_angstrom", None, "Minimum d spacing in angstrom."), _p("d_max_angstrom", None, "Maximum d spacing in angstrom."),
        _p("energy_min_meV", None, "Lower elastic energy boundary."), _p("energy_max_meV", None, "Upper elastic energy boundary."),
        _p("method", "box_sum", "Peak integration method.", choices=(("box_sum", "Box"), ("ellipsoid_sum", "Ellipsoid"), ("gaussian_fit", "Gaussian fit"))),
        _p("coordinate_frame", "hkl", "Coordinate frame for region widths."),
        _p("box_half_widths", [0.1, 0.1, 0.1], "Positive box half widths."), _p("ellipsoid_semiaxes", [0.1, 0.1, 0.1], "Positive ellipsoid semiaxes."),
        _p("ellipsoid_rotation", np.eye(3).tolist(), "Orthonormal ellipsoid rotation matrix."), _p("center_mode", "nominal", "Use nominal or centroid-refined centers."),
        _p("centroid_search_radius", 0.1, "Centroid search radius."), _p("background_mode", "none", "Optional local shell background."),
        _p("background_inner_scale", 1.5, "Inner shell scale."), _p("background_outer_scale", 2.0, "Outer shell scale."),
        _p("exclude_neighbor_regions", True, "Exclude neighboring peak regions from background.", kind="bool"),
        _p("minimum_peak_coverage", 0.9, "Minimum measured peak fraction."), _p("minimum_background_coverage", 0.7, "Minimum measured shell fraction."),
        _p("edge_policy", "reject", "Reject or report partially covered peaks."), _p("gaussian_background", "constant", "Gaussian background model."),
        _p("gaussian_max_nfev", 1000, "Maximum Gaussian optimizer evaluations."), _p("gaussian_fallback", "none", "Fallback when Gaussian fitting fails."),
        _p("subvoxel_samples", 3, "Odd samples per voxel dimension."),
    )
    spectral_parameters = (
        _p("kernel", "qfi", "Registered spectral weighting kernel."), _p("input_channel", "signal", "Input signal channel."),
        _p("spectral_convention", {"representation": "measured_intensity", "unit": "counts", "normalization_basis": "unknown", "magnetic_ions_per_basis": None, "moment_unit": "mu_B_squared", "g_factor": None, "form_factor_state": "included", "polarization_state": "included", "bose_state": "included", "kf_ki_state": "removed", "absolute_scale": False}, "Complete quantitative intensity convention.", required=True),
        _p("temperature_source", "dataset", "Read temperature from data or use fixed value."), _p("temperature_K", None, "Fixed temperature in kelvin."),
        _p("energy_min_meV", 0.0, "Lower energy boundary."), _p("energy_max_meV", None, "Upper energy boundary."), _p("elastic_exclusion_meV", 0.0, "Excluded region around zero energy."),
        _p("positive_energy_policy", "positive_only", "Positive-energy handling policy."), _p("form_factor_ion", "", "Magnetic ion form-factor key."),
        _p("custom_form_factor", None, "Optional seven form-factor coefficients."), _p("polarization_mode", "already_corrected", "Polarization correction assumption."),
        _p("polarization_scalar", 1.0, "Custom polarization divisor."), _p("g_factor", None, "Landé g factor."), _p("spin_S", None, "Spin for normalized QFI."),
        _p("zone_mode", "none", "No zones, automatic zones, or explicit centers."), _p("zone_centers_hkl", [], "Explicit zone centers."), _p("zone_basis_hkl", None, "Override reciprocal-zone basis."),
        _p("zone_subvoxel_samples", 3, "Odd zone samples per voxel dimension."), _p("minimum_energy_coverage", 0.9, "Minimum measured energy fraction."),
        _p("minimum_zone_coverage", 0.8, "Minimum measured zone fraction."), _p("partial_zone_policy", "report", "Report or reject partial zones."),
        _p("scale", 1.0, "Absolute intensity scale."), _p("form_factor_sq", 1.0, "Form-factor squared correction."), _p("power", 0, "Energy moment power."),
    )
    conversion_parameters = (
        _p("target_representation", "chi_double_prime", "Output physical representation.", choices=(("chi_double_prime", "Dynamic susceptibility"), ("cross_section", "Absolute cross section"))),
        _p("spectral_convention", {"representation": "measured_intensity", "unit": "counts", "normalization_basis": "unknown", "magnetic_ions_per_basis": None, "moment_unit": "mu_B_squared", "g_factor": None, "form_factor_state": "included", "polarization_state": "included", "bose_state": "included", "kf_ki_state": "removed", "absolute_scale": False}, "Complete input intensity convention.", required=True),
        _p("temperature_source", "dataset", "Read temperature from data or use fixed value.", choices=(("dataset", "Dataset"), ("fixed", "Fixed"))),
        _p("temperature_K", None, "Fixed temperature in kelvin."),
        _p("scale", 1.0, "Measured signal units per barn/(sr meV); may come from a vanadium or nuclear-Bragg normalization."),
        _p("background", 0.0, "Constant background in measured signal units, subtracted before conversion."),
        _p("form_factor_ion", "", "Magnetic ion form-factor key."),
        _p("custom_form_factor", None, "Optional seven form-factor coefficients."),
        _p("polarization_mode", "already_corrected", "Polarization correction assumption.", choices=(("already_corrected", "Already corrected"), ("isotropic_single_component", "Isotropic single component"), ("isotropic_trace", "Isotropic trace"), ("custom_scalar", "Custom scalar"))),
        _p("polarization_scalar", 1.0, "Custom polarization factor."),
    )
    register_analysis_operation(AnalysisOperationDefinition("bragg_integration", "Bragg integration", 1, "Integrate crystallographic peaks.", 1, 2, ("MDHistoData", "PointListData"), bragg_parameters, _validate_bragg, _execute_bragg))
    register_analysis_operation(AnalysisOperationDefinition("spectral_integration", "Spectral integration", 1, "Reduce spectra using physical kernels.", 1, 1, ("MDHistoData",), spectral_parameters, _validate_spectral, _execute_spectral))
    register_analysis_operation(AnalysisOperationDefinition("spectral_conversion", "INS absolute conversion", 1, "Convert measured INS intensity to an absolute cross section or dynamic susceptibility.", 1, 1, ("MDHistoData",), conversion_parameters, _validate_conversion, _execute_conversion))


def _validate_conversion(inputs, parameters):
    convention = SpectralConvention.from_dict(dict(parameters["spectral_convention"]))
    if convention.representation == "measured_intensity" and convention.normalization_basis == "unknown":
        raise ValueError("absolute INS conversion requires a known normalization basis")
    if float(parameters["scale"]) <= 0.0:
        raise ValueError("absolute INS scale must be positive")


def _execute_conversion(inputs, parameters, **callbacks):
    convention = SpectralConvention.from_dict(dict(parameters["spectral_convention"]))
    temperature = parameters["temperature_K"] if parameters["temperature_source"] == "fixed" else inputs[0].context.temperature_K
    if temperature is None:
        raise ValueError("INS conversion requires temperature")
    data = inputs[0].data
    q = q_modulus_for_spectral(data, inputs[0].context)
    form_factor = evaluate_form_factor_sq(
        q,
        ion=parameters["form_factor_ion"] or None,
        coefficients=parameters["custom_form_factor"],
    )
    mode = parameters["polarization_mode"]
    polarization = {
        "already_corrected": 1.0,
        "isotropic_single_component": 2.0,
        "isotropic_trace": 2.0 / 3.0,
        "custom_scalar": parameters["polarization_scalar"],
    }.get(mode)
    if polarization is None:
        raise ValueError(f"unknown polarization mode {mode!r}")
    converted = convert_spectral_representation(
        data,
        convention=convention,
        target_representation=parameters["target_representation"],
        temperature_K=float(temperature),
        scale=float(parameters["scale"]),
        background=parameters["background"],
        form_factor_sq=form_factor,
        polarization=polarization,
    )
    label = "Dynamic susceptibility" if parameters["target_representation"] == "chi_double_prime" else "Absolute cross section"
    from .core import DatasetOutput

    return AnalysisExecution({"converted": DatasetOutput(converted, label, "derived_analysis")})


def _validate_bragg(inputs, parameters):
    if not isinstance(inputs[0].data, MDHistoData):
        raise TypeError("first Bragg input must be MDHistoData")
    if parameters["gaussian_fallback"] != "none":
        raise ValueError("gaussian_fallback is reserved; inspect failed-fit status before rerunning with ellipsoid_sum")


def _execute_bragg(inputs, parameters, **callbacks):
    data = inputs[0].data
    positions = parameters.get("peak_positions_hkl") or []
    if positions:
        peaks = np.asarray(positions, dtype=float)
    elif parameters["peak_source"] == "table":
        if len(inputs) < 2 or not isinstance(inputs[1].data, PointListData):
            raise ValueError("table peak source requires a PointListData input")
        peaks = np.column_stack([inputs[1].data.column(name) for name in ("H", "K", "L")])
    else:
        spacegroup = inputs[0].context.spacegroup
        if not spacegroup:
            raise ValueError("crystal peak generation requires a space group")
        peaks = generate_bragg_peaks(data, spacegroup, include_systematic_absences=parameters["include_systematic_absences"], d_min_angstrom=parameters["d_min_angstrom"], d_max_angstrom=parameters["d_max_angstrom"])
    allowed = {key: parameters[key] for key in ("method", "coordinate_frame", "box_half_widths", "ellipsoid_semiaxes", "ellipsoid_rotation", "subvoxel_samples", "background_mode", "background_inner_scale", "background_outer_scale", "minimum_peak_coverage", "minimum_background_coverage", "edge_policy", "exclude_neighbor_regions", "center_mode", "centroid_search_radius", "gaussian_background", "gaussian_max_nfev", "energy_min_meV", "energy_max_meV")}
    return AnalysisExecution({"peak_table": TableOutput(integrate_bragg_peaks(data, peaks, progress_callback=callbacks.get("progress_callback"), cancel_callback=callbacks.get("cancel_callback"), **allowed), "Integrated Bragg peaks")})


def _validate_spectral(inputs, parameters):
    SpectralConvention.from_dict(dict(parameters["spectral_convention"]))
    if parameters["positive_energy_policy"] != "positive_only":
        raise ValueError("fold_two_sided_S is reserved until detailed-balance folding is implemented")


def _execute_spectral(inputs, parameters, **callbacks):
    convention = SpectralConvention.from_dict(dict(parameters["spectral_convention"]))
    temperature = parameters["temperature_K"] if parameters["temperature_source"] == "fixed" else inputs[0].context.temperature_K
    if temperature is None:
        raise ValueError("spectral integration requires temperature")
    maximum = np.inf if parameters["energy_max_meV"] is None else parameters["energy_max_meV"]
    data = inputs[0].data
    form_factor = parameters["form_factor_sq"]
    if convention.form_factor_state == "included" and parameters["kernel"] != "weighted_integral_arbitrary_units":
        q = q_modulus_for_spectral(data, inputs[0].context)
        form_factor = evaluate_form_factor_sq(q, ion=parameters["form_factor_ion"] or None, coefficients=parameters["custom_form_factor"])
    mode = parameters["polarization_mode"]
    polarization = {"already_corrected": 1.0, "isotropic_single_component": 2.0, "isotropic_trace": 2.0 / 3.0, "custom_scalar": parameters["polarization_scalar"]}.get(mode)
    if polarization is None:
        raise ValueError(f"unknown polarization mode {mode!r}")
    if parameters["kernel"] == "weighted_integral_arbitrary_units":
        polarization = 1.0
    minimum_energy = max(float(parameters["energy_min_meV"]), float(parameters["elastic_exclusion_meV"]))
    common = dict(convention=convention, temperature_K=float(temperature), energy_min_meV=minimum_energy, energy_max_meV=maximum, scale=parameters["scale"], form_factor_sq=form_factor, polarization=polarization, minimum_energy_coverage=parameters["minimum_energy_coverage"], power=parameters["power"], spin_S=parameters["spin_S"])
    if parameters["kernel"] == "total_moment" and parameters["zone_mode"] != "none":
        explicit_centers = parameters["zone_centers_hkl"] if parameters["zone_mode"] == "explicit_centers" else None
        table, scalar = integrate_total_moment_by_zone(inputs[0].data, spacegroup=inputs[0].context.spacegroup or "P1", minimum_zone_coverage=parameters["minimum_zone_coverage"], zone_basis_hkl=parameters["zone_basis_hkl"], zone_centers_hkl=explicit_centers, zone_subvoxel_samples=parameters["zone_subvoxel_samples"], progress_callback=callbacks.get("progress_callback"), cancel_callback=callbacks.get("cancel_callback"), **common)
        if parameters["partial_zone_policy"] == "reject" and np.any(table.column("accepted") < 0.5):
            raise ValueError("one or more Brillouin zones do not meet minimum_zone_coverage")
        return AnalysisExecution({"zones": TableOutput(table, "Total moment by zone"), "total_moment": scalar})
    output = spectral_energy_reduce(inputs[0].data, kernel=parameters["kernel"], progress_callback=callbacks.get("progress_callback"), cancel_callback=callbacks.get("cancel_callback"), **common)
    return AnalysisExecution({parameters["kernel"]: output})
