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
from .curie_weiss import execute_curie_weiss, validate_curie_weiss
from .data_reduction import (
    angle_energy_background,
    combine_aligned_histograms,
    separate_bose_elastic,
    spherical_average,
)
from .heat_capacity import (
    execute_low_temperature_heat_capacity,
    validate_low_temperature_heat_capacity,
)
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
        _p("energy_min_meV", None, "Lower elastic energy boundary; leave both bounds blank to use the bin nearest zero."), _p("energy_max_meV", None, "Upper elastic energy boundary; leave both bounds blank to use the bin nearest zero."),
        _p("method", "gaussian_fit", "Peak integration method.", choices=(("gaussian_fit", "Gaussian fit"), ("box_sum", "Box"), ("ellipsoid_sum", "Ellipsoid"))),
        _p("coordinate_frame", "hkl", "Coordinate frame for region widths.", choices=(("hkl", "HKL (r.l.u.)"), ("q_angstrom_inverse", "Q (1/angstrom)"))),
        _p("box_half_widths", [0.1, 0.1, 0.1], "Positive box half widths."), _p("ellipsoid_semiaxes", [0.1, 0.1, 0.1], "Positive ellipsoid semiaxes."),
        _p("ellipsoid_rotation", np.eye(3).tolist(), "Orthonormal ellipsoid rotation matrix."), _p("center_mode", "nominal", "Use nominal or centroid-refined centers.", choices=(("nominal", "Nominal HKL"), ("centroid", "Refine centroid"))),
        _p("centroid_search_radius", 0.1, "Centroid search radius."), _p("background_mode", "shell", "Optional local shell background.", choices=(("none", "No subtraction"), ("shell", "Local shell"))),
        _p("background_inner_scale", 1.5, "Inner shell scale."), _p("background_outer_scale", 2.0, "Outer shell scale."),
        _p("exclude_neighbor_regions", True, "Exclude neighboring peak regions from background.", kind="bool"),
        _p("minimum_peak_coverage", 0.5, "Minimum measured peak fraction."), _p("minimum_background_coverage", 0.3, "Minimum measured shell fraction."),
        _p("minimum_signal_to_noise", None, "Optional minimum accepted I/dI; leave blank to report all signal-to-noise values."),
        _p("maximum_background", None, "Optional maximum accepted absolute integrated background; leave blank for no background threshold."),
        _p("gaussian_background", "constant", "Gaussian background model.", choices=(("constant", "Constant"), ("linear", "Linear"))),
        _p("gaussian_max_nfev", 1000, "Maximum Gaussian optimizer evaluations."), _p("gaussian_fallback", "none", "Fallback when Gaussian fitting fails.", choices=(("none", "None"),)),
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
    curie_weiss_parameters = (
        AnalysisParameterDefinition(
            "temperature_min_K",
            "Tmin (K)",
            "value",
            2.0,
            "Lowest temperature included in the Curie-Weiss fit.",
            "Finite temperature in kelvin below Tmax.",
            "50.0",
        ),
        AnalysisParameterDefinition(
            "temperature_max_K",
            "Tmax (K)",
            "value",
            300.0,
            "Highest temperature included in the Curie-Weiss fit.",
            "Finite temperature in kelvin above Tmin.",
            "300.0",
        ),
    )
    low_temperature_heat_capacity_parameters = (
        AnalysisParameterDefinition(
            "temperature_min_K", "Tmin (K)", "value", 2.0,
            "Lowest temperature included in the C/T versus T^2 fit.",
            "Finite temperature in kelvin below Tmax.", "2.0",
        ),
        AnalysisParameterDefinition(
            "temperature_max_K", "Tmax (K)", "value", 10.0,
            "Highest temperature included in the C/T versus T^2 fit.",
            "Finite temperature in kelvin above Tmin.", "10.0",
        ),
        AnalysisParameterDefinition(
            "atoms_per_formula_unit", "Atoms / formula unit", "value", 1.0,
            "Atom count used only to convert beta to a Debye temperature.",
            "Positive number.", "4.0",
        ),
    )
    bose_parameters = (
        _p(
            "zero_energy_tolerance_meV",
            1.0e-12,
            "Bins whose center is this close to zero transfer are assigned to the elastic component.",
        ),
    )
    spherical_parameters = (
        _p("q_bins", 100, "Number of bins in the powder |Q| axis."),
        _p("subvoxel_samples", 3, "Positive odd number of samples per source-voxel dimension used to estimate spherical-shell overlap."),
    )
    angle_background_parameters = (
        _p("lowest_fraction", 0.2, "Lowest fraction of run intensities averaged independently in each |Q| and energy bin."),
        _p("q_bins", 100, "Number of bins in the background |Q| axis."),
        _p("energy_bins", 100, "Number of bins in the background energy axis."),
    )
    histogram_arithmetic_parameters = (
        _p(
            "operation",
            "subtract",
            "Subtract the scaled right input from the left input, or add it.",
            choices=(("subtract", "Left - scaled right"), ("add", "Left + scaled right")),
        ),
        _p("right_scale", 1.0, "Multiplier applied to the right input."),
    )
    register_analysis_operation(AnalysisOperationDefinition("bragg_integration", "Bragg integration", 4, "Integrate crystallographic peaks.", 1, 2, ("MDHistoData", "PointListData"), bragg_parameters, _validate_bragg, _execute_bragg))
    register_analysis_operation(AnalysisOperationDefinition("spectral_integration", "Spectral integration", 1, "Reduce spectra using physical kernels.", 1, 1, ("MDHistoData",), spectral_parameters, _validate_spectral, _execute_spectral))
    register_analysis_operation(AnalysisOperationDefinition("spectral_conversion", "INS absolute conversion", 1, "Convert measured INS intensity to an absolute cross section or dynamic susceptibility.", 1, 1, ("MDHistoData",), conversion_parameters, _validate_conversion, _execute_conversion))
    register_analysis_operation(
        AnalysisOperationDefinition(
            "histogram_arithmetic",
            "Histogram arithmetic",
            1,
            "Create an editable sum or difference of two identically binned histograms.",
            2,
            2,
            ("MDHistoData",),
            histogram_arithmetic_parameters,
            _validate_histogram_arithmetic,
            _execute_histogram_arithmetic,
        )
    )
    register_analysis_operation(
        AnalysisOperationDefinition(
            "curie_weiss_fit",
            "Curie-Weiss fit",
            1,
            "Fit absolute molar susceptibility to chi = C / (T - theta_CW).",
            1,
            1,
            ("PointListData",),
            curie_weiss_parameters,
            validate_curie_weiss,
            execute_curie_weiss,
        )
    )
    register_analysis_operation(
        AnalysisOperationDefinition(
            "bose_elastic_separation",
            "Bose-Einstein elastic separation",
            1,
            "Separate a temperature-independent elastic signal from a Bose-scaled inelastic signal measured at two temperatures.",
            2,
            2,
            ("MDHistoData",),
            bose_parameters,
            _validate_bose_separation,
            _execute_bose_separation,
        )
    )
    register_analysis_operation(
        AnalysisOperationDefinition(
            "spherical_average",
            "Spherical average",
            2,
            "Convert single-crystal inelastic data to a powder |Q| and energy dataset.",
            1,
            1,
            ("MDHistoData",),
            spherical_parameters,
            _validate_spherical_average,
            _execute_spherical_average,
        )
    )
    register_analysis_operation(
        AnalysisOperationDefinition(
            "angle_energy_background",
            "Angle-energy background",
            1,
            "Estimate a rotation-independent background from the lowest-intensity fraction of MDEvent runs in each |Q| and energy bin.",
            2,
            None,
            ("PointData4D",),
            angle_background_parameters,
            _validate_angle_background,
            _execute_angle_background,
        )
    )
    register_analysis_operation(
        AnalysisOperationDefinition(
            "low_temperature_heat_capacity_fit",
            "Low-temperature C/T fit",
            1,
            "Fit C/T = gamma + beta T^2 over a selected temperature range.",
            1,
            1,
            ("PointListData",),
            low_temperature_heat_capacity_parameters,
            validate_low_temperature_heat_capacity,
            execute_low_temperature_heat_capacity,
        )
    )


def _validate_conversion(inputs, parameters):
    convention = SpectralConvention.from_dict(dict(parameters["spectral_convention"]))
    if convention.representation == "measured_intensity" and convention.normalization_basis == "unknown":
        raise ValueError("absolute INS conversion requires a known normalization basis")
    if float(parameters["scale"]) <= 0.0:
        raise ValueError("absolute INS scale must be positive")


def _validate_histogram_arithmetic(inputs, parameters):
    operation = str(parameters["operation"])
    if operation not in {"subtract", "add"}:
        raise ValueError("operation must be 'subtract' or 'add'")
    if not np.isfinite(float(parameters["right_scale"])):
        raise ValueError("right_scale must be finite")


def _execute_histogram_arithmetic(inputs, parameters, **callbacks):
    from .core import DatasetOutput

    result = combine_aligned_histograms(
        inputs[0].data,
        inputs[1].data,
        operation=str(parameters["operation"]),
        right_scale=float(parameters["right_scale"]),
    )
    symbol = "-" if parameters["operation"] == "subtract" else "+"
    label = f"{inputs[0].dataset_name} {symbol} {float(parameters['right_scale']):g} x {inputs[1].dataset_name}"
    return AnalysisExecution(
        {
            "histogram": DatasetOutput(
                result,
                label,
                "powder_inelastic" if len(result.axes) == 2 else "single_crystal_inelastic",
            )
        },
        diagnostics={
            "operation": str(parameters["operation"]),
            "right_scale": float(parameters["right_scale"]),
            "uncertainty_model": "independent input variances",
        },
    )


def _validate_bose_separation(inputs, parameters):
    temperatures = [item.context.temperature_K for item in inputs]
    if any(temperature is None for temperature in temperatures):
        raise ValueError(
            "Bose-Einstein elastic separation requires a sample-environment temperature for both datasets"
        )
    if float(parameters["zero_energy_tolerance_meV"]) < 0.0:
        raise ValueError("zero_energy_tolerance_meV must be non-negative")


def _execute_bose_separation(inputs, parameters, **callbacks):
    from .core import DatasetOutput

    first_temperature = inputs[0].context.temperature_K
    second_temperature = inputs[1].context.temperature_K
    assert first_temperature is not None and second_temperature is not None
    inelastic, elastic = separate_bose_elastic(
        inputs[0].data,
        inputs[1].data,
        first_temperature_K=float(first_temperature),
        second_temperature_K=float(second_temperature),
        zero_energy_tolerance_meV=float(parameters["zero_energy_tolerance_meV"]),
    )
    return AnalysisExecution(
        {
            "inelastic": DatasetOutput(
                inelastic,
                f"Inelastic signal at {float(first_temperature):g} K",
                "powder_inelastic" if len(inelastic.axes) == 2 else "single_crystal_inelastic",
            ),
            "elastic": DatasetOutput(
                elastic,
                "Temperature-independent elastic component",
                "powder_elastic_spectrum"
                if len(elastic.axes) == 2
                else "single_crystal_elastic",
            ),
        },
        diagnostics={
            "first_temperature_K": float(first_temperature),
            "second_temperature_K": float(second_temperature),
            "uncertainty_model": "independent input variances propagated through the two-temperature linear solve",
        },
    )


def _validate_spherical_average(inputs, parameters):
    if int(parameters["q_bins"]) < 1:
        raise ValueError("q_bins must be positive")
    samples = int(parameters["subvoxel_samples"])
    if samples < 1 or samples % 2 == 0:
        raise ValueError("subvoxel_samples must be a positive odd integer")


def _execute_spherical_average(inputs, parameters, **callbacks):
    from .core import DatasetOutput

    output = spherical_average(
        inputs[0].data,
        inputs[0].context,
        q_bins=int(parameters["q_bins"]),
        subvoxel_samples=int(parameters["subvoxel_samples"]),
    )
    return AnalysisExecution(
        {"powder": DatasetOutput(output, "Spherical average", "powder_inelastic")},
        diagnostics={
            "weighting": "reciprocal_volume_overlap",
            "uncertainty_model": "independent source-voxel variances propagated through geometric overlap weights",
        },
    )


def _validate_angle_background(inputs, parameters):
    if not 0.0 < float(parameters["lowest_fraction"]) <= 1.0:
        raise ValueError("lowest_fraction must be in (0, 1]")
    if int(parameters["q_bins"]) < 1 or int(parameters["energy_bins"]) < 1:
        raise ValueError("q_bins and energy_bins must be positive")


def _execute_angle_background(inputs, parameters, **callbacks):
    from .core import DatasetOutput

    callback = callbacks.get("progress_callback")
    if callback is not None:
        callback(
            {
                "stage": "angle_energy_background",
                "iteration": 0,
                "total": len(inputs),
                "message": f"estimating background from {len(inputs)} rotation angles",
            }
        )
    output = angle_energy_background(
        [item.data for item in inputs],
        q_bins=int(parameters["q_bins"]),
        energy_bins=int(parameters["energy_bins"]),
        lowest_fraction=float(parameters["lowest_fraction"]),
    )
    if callback is not None:
        callback(
            {
                "stage": "angle_energy_background",
                "iteration": len(inputs),
                "total": len(inputs),
                "message": "angle-energy background complete",
            }
        )
    return AnalysisExecution(
        {"background": DatasetOutput(output, "Angle-energy background", "powder_inelastic")},
        diagnostics={
            "run_count": len(inputs),
            "lowest_fraction": float(parameters["lowest_fraction"]),
            "normalization": "proton_charge_and_q_energy_bin_area",
            "uncertainty_model": "selected-run statistical variances; order-statistic uncertainty excluded",
        },
    )


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
    allowed = {key: parameters[key] for key in ("method", "coordinate_frame", "box_half_widths", "ellipsoid_semiaxes", "ellipsoid_rotation", "subvoxel_samples", "background_mode", "background_inner_scale", "background_outer_scale", "minimum_peak_coverage", "minimum_background_coverage", "minimum_signal_to_noise", "maximum_background", "exclude_neighbor_regions", "center_mode", "centroid_search_radius", "gaussian_background", "gaussian_max_nfev", "energy_min_meV", "energy_max_meV")}
    table = integrate_bragg_peaks(
        data,
        peaks,
        progress_callback=callbacks.get("progress_callback"),
        cancel_callback=callbacks.get("cancel_callback"),
        **allowed,
    )
    peak_count = int(table.metadata["peak_count"])
    accepted_count = int(table.metadata["accepted_count"])
    rejected_count = int(table.metadata["rejected_count"])
    warnings = []
    if peak_count == 0:
        warnings.append("No crystallographic peaks intersect the selected data volume and d-spacing limits.")
    elif accepted_count == 0:
        warnings.append("All generated peaks were rejected by the configured quality thresholds.")
    diagnostics = {
        "generated_peaks": int(len(peaks)),
        "integrated_peaks": peak_count,
        "accepted_peaks": accepted_count,
        "rejected_peaks": rejected_count,
        "acceptance_fraction": accepted_count / peak_count if peak_count else 0.0,
        "status_bits": table.metadata["status_bits"],
    }
    return AnalysisExecution(
        {
            "peak_table": TableOutput(
                table,
                "Integrated Bragg peaks",
                {"data_type": "bragg_reflections", "fit_enabled": False},
            )
        },
        warnings=warnings,
        diagnostics=diagnostics,
    )


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
