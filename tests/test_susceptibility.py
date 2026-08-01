import numpy as np
import pytest

from nfit.susceptibility import (
    ScalarSusceptibilityResponse,
    coupled_scalar_susceptibility,
)


def test_coupled_susceptibility_matches_two_by_two_dyson_inverse():
    chi_a = np.array([0.4 + 0.2j, 0.2 + 0.3j])
    chi_b = np.array([0.3 + 0.1j, 0.5 + 0.4j])
    coupling = 0.7
    amplitude_a = np.array([1.0, 0.8])
    amplitude_b = np.array([0.6, -0.2])

    actual = coupled_scalar_susceptibility(
        chi_a,
        chi_b,
        coupling=coupling,
        amplitude_a=amplitude_a,
        amplitude_b=amplitude_b,
    )
    expected = []
    for a, b, fa, fb in zip(
        chi_a, chi_b, amplitude_a, amplitude_b, strict=True
    ):
        inverse = np.array([[1.0 / a, -coupling], [-coupling, 1.0 / b]])
        visible = np.array([fa, fb])
        expected.append(visible @ np.linalg.inv(inverse) @ visible)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_zero_coupling_is_form_factor_weighted_source_sum():
    chi_a = np.array([0.4 + 0.2j, 0.2 + 0.3j])
    chi_b = np.array([0.3 + 0.1j, 0.5 + 0.4j])
    fa = np.array([0.9, 0.8])
    fb = np.array([0.6, 0.2])
    actual = coupled_scalar_susceptibility(
        chi_a, chi_b, coupling=0.0, amplitude_a=fa, amplitude_b=fb
    )
    np.testing.assert_array_equal(actual, fa * fa * chi_a + fb * fb * chi_b)


def test_coupled_susceptibility_rejects_sampled_pole():
    with pytest.raises(np.linalg.LinAlgError, match="sampled pole"):
        coupled_scalar_susceptibility(1.0, 1.0, coupling=1.0)


def test_scalar_response_broadcasts_form_factor_and_validates_shape():
    response = ScalarSusceptibilityResponse(
        np.array([1.0 + 1.0j, 2.0 + 0.5j]), 0.75, "test"
    )
    np.testing.assert_array_equal(response.form_factor, [0.75, 0.75])
    with pytest.raises(ValueError, match="match chi"):
        ScalarSusceptibilityResponse(np.ones(2, dtype=complex), np.ones(3))
