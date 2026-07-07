"""Tools for 4D inelastic-neutron susceptibility analysis.

The public API deliberately separates dynamical susceptibility models from
measured neutron intensity. Model functions return chi''(Q,E); cross-section
helpers apply Bose, form-factor, polarization, scale, and background terms.
"""

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
from .mdhisto import (
    MDHistoAxis,
    MDHistoData,
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
from .plotting import MDHistoSliceViewer, plot_mdhisto_slice, slice_viewer
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
    "attach_lattice_parameters",
    "attach_ub_matrix",
    "bose_denominator",
    "compound_additive_model",
    "constant_background",
    "constant_fwhm_energy_resolution",
    "constant_intensity_model",
    "EnergyGaussianResolution",
    "fit_least_squares",
    "fit_problem_least_squares",
    "from_arrays",
    "identity_resolution",
    "intensity_from_chipp",
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
    "plot_mdhisto_slice",
    "point_data_from_hyspec_hhl",
    "polynomial_fwhm_energy_resolution",
    "q_modulus_inv_angstrom",
    "q_vectors_inv_angstrom",
    "quadratic_distance_rlu",
    "rebin_point_data",
    "reciprocal_basis_from_lattice_parameters",
    "relaxational_chipp",
    "rebin_nd",
    "sample_problem_parameters",
    "slice_viewer",
]
