"""Tools for reduced magnetic-scattering analysis.

The public API separates import adapters, generic reduced-data containers,
dynamical susceptibility models, measured intensity, and fitting workflows.
Model functions may return chi''(Q,E); cross-section helpers apply Bose,
form-factor, polarization, scale, and background terms.
"""

from .axes import AxisRole, infer_axis_role
from .cross_section import KB_MEV_PER_K, bose_denominator, intensity_from_chipp
from .dataset import PointData4D, from_arrays
from .fitting import (
    FitDataset,
    FitProblem,
    FitResult,
    ModelSpec,
    OptimizationConfig,
    ParameterSpec,
    ResolutionSpec,
    SamplerConfig,
    SamplingResult,
    apply_mask,
    attach_lattice_parameters,
    attach_ub_matrix,
    fit_least_squares,
    fit_problem_least_squares,
    identity_resolution,
    make_box_mask_transform,
    make_ellipsoid_mask_transform,
    make_energy_q_mask_transform,
    make_mask_transform,
    make_phonon_mask_transform,
    make_rebin_transform,
    mask_by_coordinate_range,
    mask_out_box,
    mask_out_ellipsoid,
    mask_out_energy_q_range,
    mask_out_phonon_cone,
    q_modulus_inv_angstrom,
    q_vectors_inv_angstrom,
    rebin_point_data,
    reciprocal_basis_from_lattice_parameters,
    sample_problem_parameters,
)
from .fit_views import (
    FIT_COMPARISON_METADATA_KEY,
    FitComparisonModelView,
    FitComparisonResultView,
    attach_fit_comparisons,
    fit_comparisons_for_data,
    hyspec_hhl_fit_comparison_from_points,
)
from .mdhisto import (
    MDHistoAxis,
    MDHistoData,
    hyspec_hhl_point_indices,
    load_mantid_mdhisto_nxs,
    point_data_from_hyspec_hhl,
)
from .models import (
    compound_additive_model,
    constant_background,
    constant_intensity_model,
    linear_background,
    make_constant_intensity_model,
    multi_q_paramagnon_chipp,
    paramagnon_chipp,
    quadratic_distance_rlu,
    relaxational_chipp,
)
from .pipeline import DataGroup, DatasetEntry, FitHistoryEntry, FitModelSession
from .plotting import (
    MDHistoSliceViewer,
    mdhisto_with_signal_like,
    plot_mdhisto_auto,
    plot_mdhisto_fit_comparison,
    plot_mdhisto_fit_line_comparison,
    plot_mdhisto_line,
    plot_mdhisto_slice,
    residual_mdhisto,
    slice_viewer,
)
from .rebin import NDRebin, rebin_nd
from .resolution import (
    EnergyGaussianResolution,
    constant_fwhm_energy_resolution,
    polynomial_fwhm_energy_resolution,
)


def __getattr__(name: str):
    """Lazily import optional GUI bindings when the Qt viewer is requested."""

    if name == "QtMDHistoSliceViewer":
        from .qt_slice_viewer import QtMDHistoSliceViewer

        return QtMDHistoSliceViewer
    raise AttributeError(f"module 'metallix' has no attribute {name!r}")

__all__ = [
    "FitResult",
    "FitDataset",
    "FitProblem",
    "KB_MEV_PER_K",
    "MDHistoAxis",
    "MDHistoData",
    "MDHistoSliceViewer",
    "ModelSpec",
    "NDRebin",
    "OptimizationConfig",
    "ParameterSpec",
    "PointData4D",
    "QtMDHistoSliceViewer",
    "ResolutionSpec",
    "SamplerConfig",
    "SamplingResult",
    "apply_mask",
    "AxisRole",
    "attach_lattice_parameters",
    "attach_ub_matrix",
    "bose_denominator",
    "compound_additive_model",
    "constant_background",
    "constant_fwhm_energy_resolution",
    "constant_intensity_model",
    "DataGroup",
    "DatasetEntry",
    "EnergyGaussianResolution",
    "FitHistoryEntry",
    "FitModelSession",
    "FIT_COMPARISON_METADATA_KEY",
    "FitComparisonModelView",
    "FitComparisonResultView",
    "attach_fit_comparisons",
    "fit_comparisons_for_data",
    "fit_least_squares",
    "fit_problem_least_squares",
    "from_arrays",
    "identity_resolution",
    "infer_axis_role",
    "intensity_from_chipp",
    "hyspec_hhl_point_indices",
    "hyspec_hhl_fit_comparison_from_points",
    "linear_background",
    "load_mantid_mdhisto_nxs",
    "make_box_mask_transform",
    "make_constant_intensity_model",
    "make_ellipsoid_mask_transform",
    "make_energy_q_mask_transform",
    "make_mask_transform",
    "make_phonon_mask_transform",
    "make_rebin_transform",
    "mask_by_coordinate_range",
    "mask_out_box",
    "mask_out_ellipsoid",
    "mask_out_energy_q_range",
    "mask_out_phonon_cone",
    "multi_q_paramagnon_chipp",
    "paramagnon_chipp",
    "mdhisto_with_signal_like",
    "plot_mdhisto_auto",
    "plot_mdhisto_fit_comparison",
    "plot_mdhisto_fit_line_comparison",
    "plot_mdhisto_line",
    "plot_mdhisto_slice",
    "point_data_from_hyspec_hhl",
    "polynomial_fwhm_energy_resolution",
    "q_modulus_inv_angstrom",
    "q_vectors_inv_angstrom",
    "quadratic_distance_rlu",
    "rebin_point_data",
    "reciprocal_basis_from_lattice_parameters",
    "relaxational_chipp",
    "residual_mdhisto",
    "rebin_nd",
    "sample_problem_parameters",
    "slice_viewer",
]
