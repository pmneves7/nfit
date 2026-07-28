import numpy as np

from nfit.analysis.core import AnalysisContext
from nfit.analysis.data_reduction import (
    angle_energy_background,
    separate_bose_elastic,
    spherical_average,
)
from nfit.dataset import PointData4D
from nfit.mdhisto import MDHistoAxis, MDHistoData


def _histogram(signal, errors=None):
    signal = np.asarray(signal, dtype=float)
    if errors is None:
        errors = np.ones_like(signal)
    return MDHistoData(
        axes=(
            MDHistoAxis("H", np.array([-0.5, 0.5, 1.5]), "r.l.u.", "momentum", frame="HKL"),
            MDHistoAxis("K", np.array([-0.5, 0.5]), "r.l.u.", "momentum", frame="HKL"),
            MDHistoAxis("L", np.array([-0.5, 0.5]), "r.l.u.", "momentum", frame="HKL"),
            MDHistoAxis("DeltaE", np.array([-1.5, -0.5, 0.5, 1.5]), "meV", "energy"),
        ),
        signal=signal,
        errors=np.asarray(errors, dtype=float),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={"signal_semantics": "density"},
    )


def test_bose_separation_recovers_components_and_propagates_variance():
    from nfit.analysis.data_reduction import _bose_scattering_factor

    energy = np.array([-1.0, 0.0, 1.0])
    f1 = _bose_scattering_factor(energy, 10.0)
    f2 = _bose_scattering_factor(energy, 50.0)
    amplitude = np.array([2.0, 0.0, 3.0])
    elastic = np.array([4.0, 7.0, 5.0])
    with np.errstate(invalid="ignore"):
        first_inelastic = f1 * amplitude
        second_inelastic = f2 * amplitude
    first_inelastic[1] = 0.0
    second_inelastic[1] = 0.0
    first_signal = (elastic + first_inelastic).reshape(1, 1, 1, 3)
    second_signal = (elastic + second_inelastic).reshape(1, 1, 1, 3)
    first = _histogram(np.repeat(first_signal, 2, axis=0), np.full((2, 1, 1, 3), 2.0))
    second = _histogram(np.repeat(second_signal, 2, axis=0), np.full((2, 1, 1, 3), 3.0))

    inelastic, recovered_elastic = separate_bose_elastic(
        first,
        second,
        first_temperature_K=10.0,
        second_temperature_K=50.0,
    )

    np.testing.assert_allclose(inelastic.signal[0, 0, 0, [0, 2]], first_inelastic[[0, 2]])
    np.testing.assert_allclose(recovered_elastic.signal[0, 0, 0], elastic)
    assert inelastic.signal[0, 0, 0, 1] == 0.0
    assert recovered_elastic.errors[0, 0, 0, 1] == 2.0
    assert np.all(inelastic.errors >= 0.0)


def test_bose_separation_rejects_different_binning():
    first = _histogram(np.ones((2, 1, 1, 3)))
    second = _histogram(np.ones((2, 1, 1, 3)))
    second.axes = (*second.axes[:-1], MDHistoAxis("DeltaE", np.array([-2.0, -1.0, 0.0, 1.0]), "meV", "energy"))
    with np.testing.assert_raises_regex(ValueError, "identical"):
        separate_bose_elastic(first, second, first_temperature_K=10.0, second_temperature_K=50.0)


def test_spherical_average_returns_powder_axes_and_weighted_errors():
    data = _histogram(np.arange(6.0).reshape(2, 1, 1, 3) + 1.0)
    context = AnalysisContext(
        "group",
        {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi, "alpha": 90, "beta": 90, "gamma": 90},
        None,
        None,
        None,
    )
    powder = spherical_average(data, context, q_bins=2)
    assert powder.shape == (2, 3)
    assert powder.axes[0].role == "q_modulus"
    assert powder.metadata["signal_semantics"] == "density"
    assert powder.metadata["spherical_average"]["weighting"] == "reciprocal_volume_overlap"
    assert "powder_coverage" in powder.auxiliary_channels
    assert np.count_nonzero(~powder.mask) == 6


def test_spherical_average_uses_voxel_volume_not_inverse_variance():
    axes = (
        MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "r.l.u.", "momentum", frame="HKL"),
        MDHistoAxis("K", np.array([-0.5, 0.5]), "r.l.u.", "momentum", frame="HKL"),
        MDHistoAxis("L", np.array([-0.5, 0.5]), "r.l.u.", "momentum", frame="HKL"),
        MDHistoAxis("DeltaE", np.array([0.0, 1.0]), "meV", "energy"),
    )
    data = MDHistoData(
        axes,
        np.array([[[[1.0]]], [[[3.0]]]]),
        np.array([[[[0.1]]], [[[100.0]]]]),
        np.zeros((2, 1, 1, 1), dtype=bool),
        np.ones((2, 1, 1, 1)),
        metadata={"signal_semantics": "density"},
    )
    context = AnalysisContext(
        "group",
        {
            "a": 2 * np.pi,
            "b": 2 * np.pi,
            "c": 2 * np.pi,
            "alpha": 90,
            "beta": 90,
            "gamma": 90,
        },
        None,
        None,
        None,
    )

    powder = spherical_average(data, context, q_bins=1, subvoxel_samples=3)

    np.testing.assert_allclose(powder.signal, [[2.0]])
    np.testing.assert_allclose(
        powder.errors,
        [[np.sqrt(0.1**2 + 100.0**2) / 2.0]],
    )


def test_spherical_average_converts_bin_integrals_to_density():
    density = _histogram(np.arange(6.0).reshape(2, 1, 1, 3) + 1.0)
    density.axes = (
        MDHistoAxis("H", np.array([-0.5, 0.5, 2.5]), "r.l.u.", "momentum", frame="HKL"),
        *density.axes[1:],
    )
    context = AnalysisContext(
        "group",
        {
            "a": 2 * np.pi,
            "b": 2 * np.pi,
            "c": 2 * np.pi,
            "alpha": 90,
            "beta": 90,
            "gamma": 90,
        },
        None,
        None,
        None,
    )
    q_volumes = np.array([1.0, 2.0])[:, None, None, None]
    integrated = MDHistoData(
        density.axes,
        density.signal * q_volumes,
        density.errors * q_volumes,
        density.mask,
        density.num_events,
        metadata={"signal_semantics": "bin_integral"},
    )

    expected = spherical_average(density, context, q_bins=2)
    actual = spherical_average(integrated, context, q_bins=2)

    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)


def _run(intensity):
    return PointData4D(
        H=[1.0, 2.0],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[-1.0, 1.0],
        intensity=[intensity, intensity],
        sigma=[1.0, 1.0],
        metadata={
            "coordinate_units": "1/angstrom",
            "proton_charge": 1.0,
        },
    )


def test_angle_energy_background_averages_lowest_fraction_per_bin():
    result = angle_energy_background(
        [_run(10.0), _run(2.0), _run(6.0), _run(4.0)],
        q_bins=1,
        energy_bins=2,
        lowest_fraction=0.5,
    )
    np.testing.assert_allclose(result.signal, [[3.0, 3.0]])
    np.testing.assert_allclose(result.errors, [[np.sqrt(2) / 2, np.sqrt(2) / 2]])
    np.testing.assert_allclose(result.num_events, [[2.0, 2.0]])
    assert (
        result.metadata["angle_energy_background"]["normalization"]
        == "proton_charge_and_q_energy_bin_area"
    )


def test_angle_energy_background_divides_signal_and_variance_by_bin_area():
    result = angle_energy_background(
        [_run(2.0), _run(4.0)],
        q_bins=1,
        energy_bins=1,
        lowest_fraction=1.0,
    )

    # Each run contributes two events to a bin of area
    # (2 - 1) inverse angstrom * (1 - (-1)) meV = 2.
    np.testing.assert_allclose(result.signal, [[3.0]])
    np.testing.assert_allclose(result.errors, [[0.5]])
