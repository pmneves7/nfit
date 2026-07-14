import numpy as np
import pytest

from nfit.cross_section import (
    KB_MEV_PER_K,
    MAGNETIC_GAMMA0_PER_MU_B,
    bose_denominator,
    intensity_from_chipp,
)


def test_bose_denominator_small_energy_uses_linear_limit():
    E = np.array([1e-14])
    T = 100.0
    denom = bose_denominator(E, T)
    np.testing.assert_allclose(denom, E / (KB_MEV_PER_K * T), rtol=1e-12)


def test_bose_denominator_zero_temperature_positive_energy():
    np.testing.assert_allclose(bose_denominator([1.0, 2.0], 0.0), [1.0, 1.0])


def test_intensity_from_chipp_separates_terms():
    intensity = intensity_from_chipp(
        chipp=[2.0],
        E_meV=[5.0],
        temperature_K=0.0,
        scale=3.0,
        form_factor_sq=0.5,
        polarization=0.25,
        background=1.0,
    )
    np.testing.assert_allclose(
        intensity,
        [1.0 + 3.0 * MAGNETIC_GAMMA0_PER_MU_B**2 / np.pi * 0.5 * 0.25 * 2.0],
    )


def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        bose_denominator([1.0], -1.0)


def test_bose_denominator_array_temperature_matches_scalar_loop():
    E = np.array([0.5, 1.0, -2.0, 1e-14])
    temperatures = np.array([1.8, 50.0, 300.0, 100.0])
    denom = bose_denominator(E, temperatures)
    expected = np.concatenate(
        [bose_denominator(E[i : i + 1], float(temperatures[i])) for i in range(E.size)]
    )
    np.testing.assert_allclose(denom, expected, rtol=1e-12)


def test_bose_denominator_mixed_zero_and_finite_temperatures():
    E = np.array([1.0, -1.0, 1.0])
    temperatures = np.array([0.0, 0.0, 10.0])
    denom = bose_denominator(E, temperatures)
    assert denom[0] == 1.0
    assert np.isnan(denom[1])
    np.testing.assert_allclose(denom[2], -np.expm1(-1.0 / (KB_MEV_PER_K * 10.0)))


def test_bose_denominator_array_temperature_rejects_negative_entry():
    with pytest.raises(ValueError):
        bose_denominator([1.0, 1.0], [10.0, -1.0])


def test_intensity_from_chipp_accepts_array_temperature():
    E = np.array([1.0, 2.0])
    temperatures = np.array([5.0, 200.0])
    intensity = intensity_from_chipp([1.0, 1.0], E, temperatures)
    np.testing.assert_allclose(
        intensity,
        MAGNETIC_GAMMA0_PER_MU_B**2 / np.pi / bose_denominator(E, temperatures),
    )
