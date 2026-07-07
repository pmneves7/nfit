"""Tools for 4D inelastic-neutron susceptibility analysis.

The public API deliberately separates dynamical susceptibility models from
measured neutron intensity. Model functions return chi''(Q,E); cross-section
helpers apply Bose, form-factor, polarization, scale, and background terms.
"""

from .cross_section import KB_MEV_PER_K, bose_denominator, intensity_from_chipp
from .dataset import PointData4D, from_arrays
from .fitting import FitResult, ParameterSpec, fit_least_squares
from .mdhisto import MDHistoAxis, MDHistoData, load_mantid_mdhisto_nxs
from .models import (
    constant_background,
    linear_background,
    multi_q_paramagnon_chipp,
    paramagnon_chipp,
    quadratic_distance_rlu,
    relaxational_chipp,
)
from .plotting import MDHistoSliceViewer, plot_mdhisto_slice, slice_viewer
from .qt_slice_viewer import QtMDHistoSliceViewer
from .rebin import NDRebin, rebin_nd

__all__ = [
    "FitResult",
    "KB_MEV_PER_K",
    "MDHistoAxis",
    "MDHistoData",
    "MDHistoSliceViewer",
    "NDRebin",
    "ParameterSpec",
    "PointData4D",
    "QtMDHistoSliceViewer",
    "bose_denominator",
    "constant_background",
    "fit_least_squares",
    "from_arrays",
    "intensity_from_chipp",
    "linear_background",
    "load_mantid_mdhisto_nxs",
    "multi_q_paramagnon_chipp",
    "paramagnon_chipp",
    "plot_mdhisto_slice",
    "quadratic_distance_rlu",
    "relaxational_chipp",
    "rebin_nd",
    "slice_viewer",
]
