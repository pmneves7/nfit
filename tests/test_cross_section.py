import numpy as np
import pytest

from metallix.cross_section import KB_MEV_PER_K, bose_denominator, intensity_from_chipp


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
    np.testing.assert_allclose(intensity, [1.0 + 3.0 * 0.5 * 0.25 * 2.0])


def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        bose_denominator([1.0], -1.0)

