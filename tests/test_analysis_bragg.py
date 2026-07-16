import numpy as np

from nfit.analysis.bragg import _elastic_reduce, generate_bragg_peaks, integrate_bragg_peaks
from nfit.analysis.coordinates import physical_axis_vectors
from nfit.mdhisto import MDHistoAxis, MDHistoData


def test_box_integration_uses_exact_partial_bin_overlap():
    axes = tuple(MDHistoAxis(name, np.array([0.0, 1.0, 2.0]), "rlu", "momentum") for name in ("H", "K", "L"))
    shape = (2, 2, 2)
    data = MDHistoData(axes, np.full(shape, 10.0), np.ones(shape), np.zeros(shape, bool), np.ones(shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[1, 1, 1]], box_half_widths=[0.5, 0.5, 0.5])
    assert result.column("I")[0] == 10.0
    np.testing.assert_allclose(result.column("dI")[0], np.sqrt(8 * 0.125**2))
    assert result.column("Coverage")[0] == 1.0


def test_shell_background_subtracts_constant_density():
    edges = np.arange(0.0, 6.0)
    axes = tuple(MDHistoAxis(name, edges, "rlu", "momentum") for name in ("H", "K", "L"))
    shape = (5, 5, 5)
    data = MDHistoData(axes, np.full(shape, 7.0), np.ones(shape), np.zeros(shape, bool), np.ones(shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[2.5, 2.5, 2.5]], box_half_widths=[0.5] * 3, background_mode="shell", background_inner_scale=1.0, background_outer_scale=2.0)
    np.testing.assert_allclose(result.column("I"), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.column("Background"), 7.0)


def test_ellipsoid_subvoxel_integration_approaches_known_volume():
    edges = np.linspace(-1.5, 1.5, 13)
    axes = tuple(MDHistoAxis(name, edges, "rlu", "momentum") for name in ("H", "K", "L"))
    shape = (12, 12, 12)
    data = MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[0, 0, 0]], method="ellipsoid_sum", ellipsoid_semiaxes=[1, 1, 1], subvoxel_samples=3)
    np.testing.assert_allclose(result.column("I"), 4 * np.pi / 3, rtol=0.03)


def test_gaussian_fit_recovers_integrated_intensity():
    edges = np.linspace(-1.5, 1.5, 16)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    sigma = 0.3
    amplitude = 5.0
    signal = 2.0 + amplitude * np.exp(-0.5 * (x*x + y*y + z*z) / sigma**2)
    axes = tuple(MDHistoAxis(name, edges, "rlu", "momentum") for name in ("H", "K", "L"))
    data = MDHistoData(axes, signal, np.full(signal.shape, 0.05), np.zeros(signal.shape, bool), np.ones(signal.shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[0, 0, 0]], method="gaussian_fit", ellipsoid_semiaxes=[sigma] * 3)
    expected = amplitude * (2 * np.pi) ** 1.5 * sigma**3
    np.testing.assert_allclose(result.column("I"), expected, rtol=1e-3)


def test_gaussian_fit_supports_linear_background():
    edges = np.linspace(-1.0, 1.0, 12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    sigma, amplitude = 0.25, 3.0
    signal = 2.0 + 0.4 * x - 0.2 * y + amplitude * np.exp(-0.5 * (x*x + y*y + z*z) / sigma**2)
    axes = tuple(MDHistoAxis(name, edges, "rlu", "momentum") for name in ("H", "K", "L"))
    data = MDHistoData(axes, signal, np.full(signal.shape, 0.05), np.zeros(signal.shape, bool), np.ones(signal.shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[0, 0, 0]], method="gaussian_fit", ellipsoid_semiaxes=[sigma] * 3, gaussian_background="linear")
    np.testing.assert_allclose(result.column("I"), amplitude * (2 * np.pi) ** 1.5 * sigma**3, rtol=2e-3)


def test_peak_generation_respects_centering_absences():
    edges = np.array([-1.5, -0.5, 0.5, 1.5])
    axes = tuple(MDHistoAxis(name, edges, "rlu", "momentum") for name in ("H", "K", "L"))
    shape = (3, 3, 3)
    data = MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape), metadata={"lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    peaks = generate_bragg_peaks(data, "I m -3 m")
    assert all(int(h + k + l) % 2 == 0 for h, k, l in peaks)


def test_four_dimensional_input_uses_fractional_energy_overlap():
    q_axes = tuple(MDHistoAxis(name, np.array([0.0, 1.0]), "rlu", "momentum") for name in ("H", "K", "L"))
    energy = MDHistoAxis("DeltaE", np.array([-1.0, 1.0]), "meV", "energy")
    shape = (1, 1, 1, 1)
    data = MDHistoData((*q_axes, energy), np.full(shape, 4.0), np.ones(shape), np.zeros(shape, bool), np.ones(shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[0.5, 0.5, 0.5]], box_half_widths=[0.5] * 3, energy_min_meV=-0.5, energy_max_meV=0.5)
    assert result.column("I")[0] == 4.0


def test_four_dimensional_input_defaults_to_energy_bin_nearest_zero():
    q_axes = tuple(MDHistoAxis(name, np.array([0.0, 1.0]), "rlu", "momentum") for name in ("H", "K", "L"))
    energy = MDHistoAxis("DeltaE", np.array([-2.0, -1.0, 1.0]), "meV", "energy")
    data = MDHistoData(
        (*q_axes, energy),
        np.array([[[[2.0, 4.0]]]]),
        np.ones((1, 1, 1, 2)),
        np.zeros((1, 1, 1, 2), bool),
        np.ones((1, 1, 1, 2)),
        metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}},
    )
    result = integrate_bragg_peaks(data, [[0.5, 0.5, 0.5]], box_half_widths=[0.5] * 3)
    assert result.column("I")[0] == 8.0


def test_four_dimensional_rebin_vectors_keep_independent_hkl_directions():
    momentum_axes = (
        MDHistoAxis("[H,H,H]", np.array([0.0, 1.0]), "r.l.u.", "momentum"),
        MDHistoAxis("[L,L,-2L]", np.array([0.0, 1.0]), "r.l.u.", "momentum"),
        MDHistoAxis("[H,-H,0]", np.array([0.0, 1.0]), "r.l.u.", "momentum"),
    )
    energy = MDHistoAxis("DeltaE", np.array([-1.0, 1.0]), "meV", "energy")
    data = MDHistoData(
        (energy, *momentum_axes),
        np.ones((1, 1, 1, 1)),
        np.ones((1, 1, 1, 1)),
        np.zeros((1, 1, 1, 1), bool),
        np.ones((1, 1, 1, 1)),
        metadata={
            "signal_semantics": "density",
            "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi},
            "rebin": {
                "vectors": [
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, -2.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0],
                ]
            },
        },
    )
    reduced = _elastic_reduce(data, -1.0, 1.0)
    np.testing.assert_allclose(
        physical_axis_vectors(reduced)[:, :3],
        [[1.0, 1.0, 1.0], [1.0, 1.0, -2.0], [1.0, -1.0, 0.0]],
    )


def test_projected_hkl_axes_integrate_supplied_peak():
    edges = np.linspace(-1.0, 1.0, 9)
    axes = (
        MDHistoAxis("[H,H,0]", edges, "rlu", "momentum"),
        MDHistoAxis("[H,-H,0]", edges, "rlu", "momentum"),
        MDHistoAxis("L", edges, "rlu", "momentum"),
    )
    shape = (8, 8, 8)
    data = MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    result = integrate_bragg_peaks(data, [[0, 0, 0]], box_half_widths=[0.5, 0.5, 0.5], subvoxel_samples=5)
    np.testing.assert_allclose(result.column("I"), 1.0, rtol=0.12)
