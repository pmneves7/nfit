import numpy as np
import pytest

from nfit.analysis import SpectralConvention
from nfit.analysis.spectral import (
    convert_spectral_representation,
    integrate_total_moment_by_zone,
    spectral_energy_reduce,
    spectral_kernel,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.plotting import MDHistoSliceViewer
from nfit.spectral_channels import (
    CHIPP_CHANNEL,
    CROSS_SECTION_CHANNEL,
    default_spectral_channel_config,
    with_paired_spectral_channels,
)


def _convention():
    return SpectralConvention("chi_double_prime", "mu_B^2/meV", "per_magnetic_ion", 1.0, "mu_B_squared", 2.0, "removed", "removed", "removed", "removed", True)


def test_qfi_reduction_returns_viewer_ready_point_list_with_uncertainty():
    axes = (
        MDHistoAxis("Q", np.array([0.0, 1.0, 2.0]), "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 3.0]), "meV", "energy"),
    )
    data = MDHistoData(axes, np.ones((2, 2)), np.full((2, 2), 0.1), np.zeros((2, 2), bool), np.ones((2, 2)), metadata={"signal_semantics": "density"})
    output = spectral_energy_reduce(data, kernel="qfi", convention=_convention(), temperature_K=10.0, energy_max_meV=3.0, spin_S=0.5)
    table = output.data
    expected = np.sum(spectral_kernel("qfi", np.array([0.5, 2.0]), 10.0) * np.array([1.0, 2.0])) / 4.0
    np.testing.assert_allclose(table.column("Quantum Fisher information"), expected)
    np.testing.assert_allclose(table.column("Normalized QFI"), expected / 3.0)
    assert table.coordinate_names == ["Q"]


def test_masked_energy_lowers_coverage_and_masks_map():
    axes = (
        MDHistoAxis("H", np.array([0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("K", np.array([0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0]), "meV", "energy"),
    )
    mask = np.zeros((1, 1, 2), bool)
    mask[..., 1] = True
    data = MDHistoData(axes, np.ones((1, 1, 2)), np.ones((1, 1, 2)), mask, np.ones((1, 1, 2)))
    output = spectral_energy_reduce(data, kernel="total_moment", convention=_convention(), temperature_K=10.0, energy_max_meV=2.0, minimum_energy_coverage=0.9)
    assert output.data.mask[0, 0]
    viewer = MDHistoSliceViewer(output.data)
    assert "energy_coverage" in viewer.CHANNELS
    assert viewer.CHANNEL_LABELS["energy_coverage"] == "Energy coverage (fraction)"


def test_constant_full_zone_returns_coverage_corrected_moment():
    axes = tuple(MDHistoAxis(name, np.array([-0.5, 0.5]), "rlu", "momentum") for name in ("H", "K", "L")) + (MDHistoAxis("DeltaE", np.array([0.5, 1.5]), "meV", "energy"),)
    shape = (1, 1, 1, 1)
    data = MDHistoData(axes, np.ones(shape), np.full(shape, 0.1), np.zeros(shape, bool), np.ones(shape), metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi}})
    table, scalar = integrate_total_moment_by_zone(data, convention=_convention(), temperature_K=10.0, spacegroup="P1", energy_min_meV=0.5, energy_max_meV=1.5)
    expected = spectral_kernel("total_moment", np.array([1.0]), 10.0)[0]
    np.testing.assert_allclose(table.column("moment_coverage_corrected"), expected)
    assert scalar.value == pytest.approx(expected)


def test_arbitrary_integral_accepts_uncalibrated_counts_without_corrections():
    axes = (MDHistoAxis("Q", np.array([0.0, 1.0]), "1/angstrom", "momentum"), MDHistoAxis("DeltaE", np.array([0.0, 2.0]), "meV", "energy"))
    data = MDHistoData(axes, np.array([[3.0]]), np.array([[0.5]]), np.zeros((1, 1), bool), np.ones((1, 1)))
    convention = SpectralConvention("measured_intensity", "counts", "unknown", None, "mu_B_squared", None, "included", "included", "included", "removed", False)
    output = spectral_energy_reduce(data, kernel="weighted_integral_arbitrary_units", convention=convention, temperature_K=10.0, energy_max_meV=2.0)
    assert output.data.column("Weighted Integral Arbitrary Units")[0] == 6.0


def test_absolute_ins_conversion_returns_chipp_with_explicit_metadata():
    axes = (
        MDHistoAxis("Q", np.array([0.0, 1.0]), "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", np.array([1.0, 3.0]), "meV", "energy"),
    )
    original = np.array([[1.7]])
    from nfit.cross_section import cross_section_from_chipp

    cross = cross_section_from_chipp(original, np.array([[2.0]]), 25.0, form_factor_sq=0.8, polarization=2.0 / 3.0)
    counts_per_cross_section = 2500.0
    data = MDHistoData(
        axes,
        cross * counts_per_cross_section,
        cross * counts_per_cross_section * 0.1,
        np.zeros((1, 1), bool),
        np.ones((1, 1)),
    )
    input_convention = SpectralConvention(
        "measured_intensity", "counts", "per_formula_unit", 2.0,
        "mu_B_squared", 2.0, "included", "included", "included", "removed", False,
    )
    converted = convert_spectral_representation(
        data,
        convention=input_convention,
        target_representation="chi_double_prime",
        temperature_K=25.0,
        scale=counts_per_cross_section,
        form_factor_sq=0.8,
        polarization=2.0 / 3.0,
    )
    np.testing.assert_allclose(converted.signal, original)
    assert converted.metadata["signal_quantity_type"] == "dynamic_susceptibility"
    assert converted.metadata["signal_unit"] == "mu_B^2/meV"
    assert converted.metadata["spectral_convention"]["absolute_scale"] is True


def test_arbitrary_ins_view_exposes_cross_section_and_chipp_channels():
    axes = (
        MDHistoAxis("Q", np.array([0.0, 1.0]), "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", np.array([1.0, 3.0]), "meV", "energy"),
    )
    data = MDHistoData(
        axes,
        np.array([[12.0]]),
        np.array([[1.2]]),
        np.zeros((1, 1), bool),
        np.ones((1, 1)),
    )
    config = default_spectral_channel_config()
    config["fit_representation"] = "chi_double_prime"
    converted = with_paired_spectral_channels(data, config, temperature_K=25.0)
    expected = (
        data.signal
        * (1.0 - np.exp(-2.0 / (0.08617333262 * 25.0)))
        / (2.0 / 3.0)
    )
    np.testing.assert_allclose(converted.signal, expected)
    assert CROSS_SECTION_CHANNEL in converted.auxiliary_channels
    assert CHIPP_CHANNEL in converted.auxiliary_channels
    assert converted.channel_quantity_type() == "dynamic_susceptibility"
    assert (
        converted.channel_quantity_type(CROSS_SECTION_CHANNEL)
        == "differential_cross_section"
    )
    assert converted.channel_unit(CHIPP_CHANNEL) == "arb. units"


def test_absolute_mbarn_ins_channels_round_trip_and_keep_formula_unit_basis():
    axes = (
        MDHistoAxis("Q", np.array([0.0, 1.0]), "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", np.array([1.0, 3.0]), "meV", "energy"),
    )
    original_chipp = np.array([[1.7]])
    from nfit.cross_section import cross_section_from_chipp

    cross_mbarn = (
        cross_section_from_chipp(
            original_chipp,
            np.array([[2.0]]),
            25.0,
            polarization=2.0 / 3.0,
        )
        * 1000.0
    )
    data = MDHistoData(
        axes,
        cross_mbarn,
        cross_mbarn * 0.1,
        np.zeros((1, 1), bool),
        np.ones((1, 1)),
    )
    config = default_spectral_channel_config()
    config.update(
        {
            "source_unit": "mbarn/sr/meV/f.u.",
            "fit_representation": "chi_double_prime",
        }
    )
    converted = with_paired_spectral_channels(data, config, temperature_K=25.0)
    np.testing.assert_allclose(converted.signal, original_chipp)
    assert converted.channel_unit() == "mu_B^2/meV/f.u."
    assert (
        converted.channel_unit(CROSS_SECTION_CHANNEL)
        == "mbarn/sr/meV/f.u."
    )


def test_spin_susceptibility_applies_g_squared_exactly_once():
    axes = (
        MDHistoAxis("Q", np.array([0.0, 1.0]), "1/angstrom", "momentum"),
        MDHistoAxis("DeltaE", np.array([1.0, 3.0]), "meV", "energy"),
    )
    data = MDHistoData(
        axes,
        np.array([[2.0]]),
        np.array([[0.2]]),
        np.zeros((1, 1), bool),
        np.ones((1, 1)),
    )
    config = default_spectral_channel_config()
    config.update(
        {
            "source_representation": "chi_double_prime",
            "source_unit": "spin^2/meV/f.u.",
            "moment_unit": "spin_squared",
            "g_factor": 2.5,
        }
    )
    converted = with_paired_spectral_channels(data, config, temperature_K=20.0)
    from nfit.cross_section import cross_section_from_chipp

    expected = (
        cross_section_from_chipp(
            data.signal,
            np.array([[2.0]]),
            20.0,
            polarization=2.0 / 3.0,
            moment_unit="spin_squared",
            g_factor=2.5,
        )
        * 1000.0
    )
    np.testing.assert_allclose(
        converted.auxiliary_channels[CROSS_SECTION_CHANNEL].values, expected
    )
