"""Analytic support, shell geometry, and identifiability regressions."""
import ast
from pathlib import Path

import numpy as np
import pytest

from nfit.analysis import bragg_regions
from nfit.analysis.bragg import (
    _identifiable_fit_covariance,
    bragg_volume,
    integrate_bragg_peaks,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData

LATTICE = {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}


def histogram(edges, signal, *, semantics="density", errors=None, metadata=None):
    if isinstance(edges, np.ndarray):
        edges = (edges,) * 3
    axes = tuple(
        MDHistoAxis(name, values, "rlu", "momentum")
        for name, values in zip(("H", "K", "L"), edges, strict=True)
    )
    return MDHistoData(
        axes, signal, np.ones(signal.shape) if errors is None else errors,
        np.zeros(signal.shape, bool), np.ones(signal.shape),
        metadata={"lattice_parameters": LATTICE, "signal_semantics": semantics, **(metadata or {})},
    )


def test_nonuniform_shell_subtracts_the_same_physical_background_for_both_semantics():
    edges = np.array([-2., -1., -.5, .5, 1., 2.])
    width = np.diff(edges)
    volume = width[:, None, None] * width[None, :, None] * width[None, None, :]
    outputs = []
    for semantics in ("density", "bin_integral"):
        multiplier = 1.0 if semantics == "density" else volume
        data = histogram(edges, np.full(volume.shape, 7.) * multiplier,
                         errors=np.ones(volume.shape) * multiplier, semantics=semantics)
        result = integrate_bragg_peaks(
            data, [[0., 0., 0.]], method="box_sum", box_half_widths=[.5] * 3,
            background_mode="shell", background_inner_scale=1., background_outer_scale=2.,
        )
        assert result.column("Background")[0] == pytest.approx(7.)
        assert result.column("I")[0] == pytest.approx(0., abs=1e-12)
        assert result.column("Coverage")[0] == pytest.approx(1.)
        assert result.column("BackgroundCoverage")[0] == pytest.approx(1.)
        assert result.column("Accepted")[0] == 1
        outputs.append(result.column("dI")[0])
    # The two encodings also describe exactly the same supplied voxel errors.
    assert outputs[0] == pytest.approx(outputs[1])


@pytest.mark.parametrize("semantics", ["density", "bin_integral"])
def test_nonuniform_fully_supported_large_voxel_has_full_coverage(semantics):
    edges = np.array([-1.2, -1.1, -1., -.5, .5, 1., 1.1, 1.2])
    data = histogram(edges, np.ones((7, 7, 7)), semantics=semantics)
    result = integrate_bragg_peaks(data, [[0, 0, 0]], method="box_sum", box_half_widths=[.5] * 3)
    assert result.column("Coverage")[0] == pytest.approx(1.)
    assert result.column("Accepted")[0] == 1


def test_neighbor_exclusion_uses_exact_joint_box_intersections():
    edges = (np.array([0., 1., 2.]), np.array([0., 1.]), np.array([0., 1.]))
    data = histogram(edges, np.array([[[1.]], [[9.]]]))
    result = integrate_bragg_peaks(
        data, [[.5, .5, .5], [1.25, .5, .5]], method="box_sum",
        box_half_widths=[.25] * 3, background_mode="shell",
        background_inner_scale=1., background_outer_scale=3., minimum_background_coverage=0.,
    )
    # Main peak has volume 1/8. Shell support left in the two voxels is
    # 7/8 and 3/16; the latter subtracts the actual neighbor intersection.
    expected = (.875 + 9 * .1875) * .125 / (.875 + .1875)
    assert result.column("Background")[0] == pytest.approx(expected)


def test_exact_box_shell_excludes_union_and_counts_exclusions_beyond_data_edges():
    edges = (np.array([-.5, .5]),) * 3
    center, widths = np.zeros(3), np.full(3, .25)
    neighbor = np.array([.5, 0., 0.])
    fraction, full_measure = bragg_regions.box_shell_fraction(
        edges, (0, 1, 2), center, widths, 1., 3., [neighbor, neighbor],
    )
    # Full outer cube 1.5^3 minus central cube .5^3 and neighbor cube .5^3.
    # Duplicate exclusions remove this neighboring cube only once.
    assert full_measure == pytest.approx(3.375 - .125 - .125)
    # Only half the neighbor lies in the single measured unit cube.
    assert fraction.item() == pytest.approx(1. - .125 - .0625)


@pytest.mark.parametrize("method", ["box_sum", "ellipsoid_sum"])
def test_sampled_shell_and_neighbors_share_the_same_membership_points(method):
    edges = (np.array([-1., 1.]),) * 3
    center, widths = np.zeros(3), np.full(3, .45)
    neighbors = [np.array([.5, 0., 0.]), np.array([.5, 0., 0.])]
    fraction = bragg_regions.sampled_region_fraction(
        edges, np.eye(3), center, widths, np.eye(3), 5, None, method,
        inner_scale=1., outer_scale=2., neighbors=neighbors,
    )
    # Independent point enumeration tests the intersection, not a product
    # of separately averaged overlap fractions.
    coordinates = np.linspace(-.8, .8, 5)
    x = np.array(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")).reshape(3, -1).T
    def inside(origin, scale):
        if method == "box_sum":
            return np.all(np.abs(x - origin) <= widths * scale, axis=1)
        return np.sum(((x - origin) / (widths * scale)) ** 2, axis=1) <= 1
    selected = inside(center, 2.) & ~inside(center, 1.) & ~inside(neighbors[0], 1.)
    assert fraction.item() == pytest.approx(np.mean(selected))


@pytest.mark.parametrize("dimension", [0, 1, 3])
def test_partial_elastic_support_is_reported_and_rejected_without_filling_signal(dimension):
    axes = [MDHistoAxis(name, np.array([-.5, .5]), "rlu", "momentum") for name in ("H", "K", "L")]
    axes.insert(dimension, MDHistoAxis("DeltaE", np.array([-1., 0., 1.]), "meV", "energy"))
    shape = [1] * 4
    shape[dimension] = 2
    signal = np.full(shape, 4.)
    mask = np.zeros(shape, bool)
    selection = [slice(None)] * 4
    selection[dimension] = 0
    mask[tuple(selection)] = True
    data = MDHistoData(tuple(axes), signal, np.ones(shape), mask, np.ones(shape),
                       metadata={"lattice_parameters": LATTICE, "signal_semantics": "density"})
    volume = bragg_volume(data, -1., 1.)
    assert np.asarray(volume.metadata["energy_coverage"]).item() == pytest.approx(.5)
    result = integrate_bragg_peaks(
        data, [[0, 0, 0]], method="box_sum", box_half_widths=[.5] * 3,
        energy_min_meV=-1., energy_max_meV=1.,
    )
    assert result.column("I")[0] == pytest.approx(4.)
    assert result.column("dI")[0] == pytest.approx(1.)
    assert result.column("EnergyCoverage")[0] == pytest.approx(.5)
    assert result.column("Coverage")[0] == pytest.approx(.5)
    assert result.column("Accepted")[0] == 0
    assert int(result.column("Status")[0]) & 1
    assert result.metadata["elastic_energy_window_meV"] == [-1., 1.]
    np.testing.assert_array_equal(data.signal, signal)
    np.testing.assert_array_equal(data.mask, mask)


def test_requested_energy_window_outside_data_is_incomplete_even_with_no_masked_bins():
    q_axes = tuple(MDHistoAxis(name, np.array([-.5, .5]), "rlu", "momentum") for name in ("H", "K", "L"))
    energy = MDHistoAxis("DeltaE", np.array([-1., 1.]), "meV", "energy")
    data = MDHistoData((*q_axes, energy), np.full((1, 1, 1, 1), 4.),
        np.ones((1, 1, 1, 1)), np.zeros((1, 1, 1, 1), bool), np.ones((1, 1, 1, 1)),
        metadata={"lattice_parameters": LATTICE, "signal_semantics": "density"})
    result = integrate_bragg_peaks(data, [[0, 0, 0]], method="box_sum", box_half_widths=[.5] * 3,
                                  energy_min_meV=-2., energy_max_meV=2.)
    assert result.column("I")[0] == pytest.approx(8.)
    assert result.column("EnergyCoverage")[0] == pytest.approx(.5)
    assert result.column("Accepted")[0] == 0


def test_shell_coverage_includes_retained_elastic_energy_support():
    edges = np.arange(6.)
    energy_support = np.full((5, 5, 5), .5)
    energy_support[2, 2, 2] = 1.
    data = histogram(edges, np.ones((5, 5, 5)), metadata={"energy_coverage": energy_support.tolist()})
    result = integrate_bragg_peaks(data, [[2.5] * 3], method="box_sum", box_half_widths=[.5] * 3,
        background_mode="shell", background_inner_scale=1., background_outer_scale=2.)
    assert result.column("Coverage")[0] == pytest.approx(1.)
    assert result.column("BackgroundCoverage")[0] == pytest.approx(.5)
    assert result.column("Accepted")[0] == 0
    assert int(result.column("Status")[0]) & 2


def test_symmetric_eight_point_gaussian_cannot_measure_integral_and_is_rejected():
    edges = np.array([-.5, 0., .5])
    centers = .5 * (edges[:-1] + edges[1:])
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    sigma = 1 / 6
    signal = 2 + 5 * np.exp(-.5 * (x*x + y*y + z*z) / sigma**2)
    result = integrate_bragg_peaks(histogram(edges, signal, errors=np.full(signal.shape, .05)),
                                  [[0, 0, 0]], ellipsoid_semiaxes=[sigma] * 3)
    assert result.column("Accepted")[0] == 0
    assert int(result.column("Status")[0]) & 16
    assert np.isnan(result.column("dI")[0])


def test_gaussian_with_many_points_but_unobserved_width_is_rejected():
    edges = (np.linspace(-.25, .25, 6), np.linspace(-.25, .25, 6), np.array([-.05, .05]))
    x, y, z = np.meshgrid(*(0.5 * (edge[:-1] + edge[1:]) for edge in edges), indexing="ij")
    sigma = np.array([.08, .08, 1/60])
    signal = 2 + 5 * np.exp(-.5 * ((x/sigma[0])**2 + (y/sigma[1])**2 + (z/sigma[2])**2))
    result = integrate_bragg_peaks(histogram(edges, signal, errors=np.full(signal.shape, .05)),
                                  [[0, 0, 0]], ellipsoid_semiaxes=sigma)
    assert signal.size > 8
    assert result.column("Accepted")[0] == 0
    assert int(result.column("Status")[0]) & 16
    assert np.isnan(result.column("dI")[0])


def test_absolute_error_covariance_is_invariant_to_parameter_units():
    # An orthogonal full-rank Jacobian has covariance diag(1/a_i^2), regardless
    # of how differently its parameter columns are scaled.
    jacobian = np.vstack((np.eye(3), np.eye(3))) * np.array([1e-9, 1., 1e9])
    covariance = _identifiable_fit_covariance(jacobian)
    np.testing.assert_allclose(covariance, np.diag([.5e18, .5, .5e-18]))
    assert _identifiable_fit_covariance(np.column_stack((np.ones(10), np.ones(10)))) is None


def test_region_geometry_service_is_independent_of_gui_and_bragg_facade():
    tree = ast.parse(Path(bragg_regions.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("qt" in name.lower() or "gui" in name.lower() or name.endswith("bragg") for name in imported)


def test_recentering_does_not_exclude_the_peaks_own_nominal_region_from_shell():
    edges = np.arange(-1.5, 3.5)
    signal = np.ones((4, 4, 4))
    signal[2, 1, 1] = 10.
    data = histogram(edges, signal)
    options = dict(method="box_sum", box_half_widths=[.4] * 3, background_mode="shell",
        background_inner_scale=1., background_outer_scale=2., center_mode="centroid",
        centroid_search_radius=1.1, minimum_background_coverage=0.)
    included = integrate_bragg_peaks(data, [[0, 0, 0]], exclude_neighbor_regions=False, **options)
    excluded = integrate_bragg_peaks(data, [[0, 0, 0]], exclude_neighbor_regions=True, **options)
    assert included.column("H")[0] == pytest.approx(1.)
    # A single requested reflection has no neighbors, even after recentering.
    for column in ("I", "dI", "Background", "BackgroundCoverage"):
        np.testing.assert_allclose(excluded.column(column), included.column(column))


def test_gaussian_rejects_partial_energy_support_without_extrapolating_missing_energy():
    edges = np.linspace(-1.5, 1.5, 16)
    centers = .5 * (edges[:-1] + edges[1:])
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    sigma = .3
    signal = 2 + 5 * np.exp(-.5 * (x*x + y*y + z*z) / sigma**2)
    data = histogram(edges, signal, errors=np.full(signal.shape, .05),
                     metadata={"energy_coverage": np.full(signal.shape, .5).tolist()})
    result = integrate_bragg_peaks(data, [[0, 0, 0]], ellipsoid_semiaxes=[sigma] * 3)
    assert result.column("EnergyCoverage")[0] == pytest.approx(.5)
    assert result.column("Coverage")[0] == pytest.approx(.5)
    assert result.column("Accepted")[0] == 0
    assert int(result.column("Status")[0]) & 1
    assert int(result.column("Status")[0]) & 16
    assert np.isnan(result.column("I")[0])
    assert np.isnan(result.column("dI")[0])


@pytest.mark.parametrize("background", ["constant", "linear"])
def test_gaussian_absolute_uncertainty_scales_with_supplied_errors_even_for_exact_model(background):
    edges = np.linspace(-1.5, 1.5, 16)
    centers = .5 * (edges[:-1] + edges[1:])
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    sigma = .3
    signal = 2 + .1 * x - .2 * y + 5 * np.exp(-.5 * (x*x + y*y + z*z) / sigma**2)
    if background == "constant":
        signal -= .1 * x - .2 * y
    uncertainties = []
    for error in (.05, .1):
        result = integrate_bragg_peaks(histogram(edges, signal, errors=np.full(signal.shape, error)),
            [[0, 0, 0]], ellipsoid_semiaxes=[sigma] * 3, gaussian_background=background)
        assert result.column("Accepted")[0] == 1
        assert result.column("ReducedChi2")[0] < 1e-10
        uncertainties.append(result.column("dI")[0])
    assert uncertainties[0] > 0
    assert uncertainties[1] == pytest.approx(2 * uncertainties[0], rel=1e-6)


@pytest.mark.parametrize("semantics", ["density", "bin_integral"])
@pytest.mark.parametrize("shared", [False, True])
def test_shell_scales_to_measured_joint_momentum_energy_support(semantics, shared):
    from nfit.measurement_dependencies import SourceDependencies

    axes = tuple(MDHistoAxis(name, np.arange(6.), "rlu", "momentum") for name in ("H", "K", "L")) + (
        MDHistoAxis("DeltaE", np.array([-1., 0., 1.]), "meV", "energy"),
    )
    shape = (5, 5, 5, 2)
    mask = np.zeros(shape, bool)
    mask[2, 2, 2, 0] = True
    dependencies = None
    if shared:
        dependencies = SourceDependencies(shape=shape, source_ids=("common background",),
            source_variances=np.array([1.]), observation_indices=np.arange(np.prod(shape)),
            source_indices=np.zeros(np.prod(shape), dtype=int), coefficients=np.ones(np.prod(shape)))
    data = MDHistoData(axes, np.full(shape, 7.), np.ones(shape), mask, np.ones(shape),
        metadata={"lattice_parameters": LATTICE, "signal_semantics": semantics},
        source_dependencies=dependencies)
    result = integrate_bragg_peaks(data, [[2.5] * 3], method="box_sum", box_half_widths=[.5] * 3,
        background_mode="shell", background_inner_scale=1., background_outer_scale=2.,
        energy_min_meV=-1., energy_max_meV=1., minimum_peak_coverage=.4)
    # The peak has one measured 1-meV energy slice and the shell has two.
    # Flat Q×E background density predicts 7 only over the measured peak slice.
    assert result.column("RawI")[0] == pytest.approx(7.)
    assert result.column("Background")[0] == pytest.approx(7.)
    assert result.column("I")[0] == pytest.approx(0., abs=1e-12)
    assert result.column("Coverage")[0] == pytest.approx(.5)
    assert result.column("BackgroundCoverage")[0] == pytest.approx(1.)
    assert result.column("Accepted")[0] == 1
    if shared:
        # One common additive primitive cancels with matched joint support.
        assert result.column("dI")[0] == pytest.approx(0., abs=1e-12)
    else:
        # Shell voxel overlap squares sum to (1+1/4+1/4)^3-1=19/8.
        # Two independent energy slices and ratio 1/14 give 19/784 variance.
        assert result.column("dI")[0] == pytest.approx(np.sqrt(1. + 19./784.))


def test_partial_energy_gaussian_is_rejected_even_when_coverage_threshold_allows_it():
    edges = np.linspace(-1.5, 1.5, 16)
    centers = .5 * (edges[:-1] + edges[1:])
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    signal = 2 + 5 * np.exp(-.5 * (x*x + y*y + z*z) / .3**2)
    support = np.ones(signal.shape)
    support[7, 7, 7] = .5
    signal[7, 7, 7] *= .5
    result = integrate_bragg_peaks(histogram(edges, signal, errors=np.full(signal.shape, .05),
        metadata={"energy_coverage": support.tolist()}), [[0, 0, 0]], ellipsoid_semiaxes=[.3] * 3,
        minimum_peak_coverage=0.)
    assert result.column("Coverage")[0] > .9
    assert result.column("Accepted")[0] == 0
    assert int(result.column("Status")[0]) & 16
    assert np.isnan(result.column("I")[0])
    assert np.isnan(result.column("dI")[0])
