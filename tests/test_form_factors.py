import numpy as np
import pytest

from metallix.form_factors import (
    J0_COEFFICIENTS,
    available_ions,
    form_factor_sq,
    magnetic_form_factor_j0,
)


def test_all_tabulated_ions_are_normalized_at_zero_q():
    for ion, (A, a, B, b, C, c, D) in J0_COEFFICIENTS.items():
        assert abs((A + B + C + D) - 1.0) < 0.01, ion
        np.testing.assert_allclose(
            magnetic_form_factor_j0(0.0, ion=ion), A + B + C + D, rtol=1e-12
        )


def test_mn2_matches_published_coefficients():
    # Mn2+ <j0>: A=0.4220 a=17.684 B=0.5948 b=6.005 C=0.0043 c=-0.609 D=-0.0219
    # (P. J. Brown, Int. Tables Cryst. C, Section 4.4.5 / ILL tables)
    q = 2.0
    s_sq = (q / (4.0 * np.pi)) ** 2
    expected = (
        0.4220 * np.exp(-17.684 * s_sq)
        + 0.5948 * np.exp(-6.005 * s_sq)
        + 0.0043 * np.exp(0.609 * s_sq)
        - 0.0219
    )
    np.testing.assert_allclose(magnetic_form_factor_j0(q, ion="Mn2"), expected, rtol=1e-6)


def test_fe2_decreases_monotonically_at_moderate_q():
    q = np.linspace(0.0, 6.0, 200)
    f = magnetic_form_factor_j0(q, ion="Fe2")
    assert f[0] == pytest.approx(1.0, abs=0.01)
    assert np.all(np.diff(f) < 0.0)


def test_custom_coefficients_override_ion():
    coeffs = (0.5, 10.0, 0.5, 5.0, 0.0, 1.0, 0.0)
    q = np.array([0.0, 1.5, 3.0])
    expected = magnetic_form_factor_j0(q, coefficients=coeffs)
    with_ion = magnetic_form_factor_j0(q, ion="Mn2", coefficients=coeffs)
    np.testing.assert_allclose(with_ion, expected)


def test_blank_custom_coefficients_fall_back_to_ion():
    q = np.array([0.0, 1.5, 3.0])
    np.testing.assert_allclose(
        magnetic_form_factor_j0(q, ion="V2", coefficients=""),
        magnetic_form_factor_j0(q, ion="V2"),
    )


def test_comma_separated_custom_coefficients_parse_from_gui_text():
    coeffs = "0.5, 10.0, 0.5, 5.0, 0.0, 1.0, 0.0"
    q = np.array([0.0, 1.5, 3.0])
    expected = magnetic_form_factor_j0(q, coefficients=(0.5, 10.0, 0.5, 5.0, 0.0, 1.0, 0.0))
    np.testing.assert_allclose(magnetic_form_factor_j0(q, coefficients=coeffs), expected)


def test_malformed_custom_coefficients_raise_field_specific_error():
    with pytest.raises(ValueError, match="form factor coefficients"):
        magnetic_form_factor_j0(1.0, ion="V2", coefficients="not, numbers")


def test_form_factor_sq_squares_j0():
    q = np.array([0.5, 2.5])
    f = magnetic_form_factor_j0(q, ion="Co2")
    np.testing.assert_allclose(form_factor_sq(q, ion="Co2"), f**2)


def test_unknown_ion_raises_with_suggestions():
    with pytest.raises(KeyError, match="Mn2"):
        magnetic_form_factor_j0(1.0, ion="Unobtainium9")


def test_missing_ion_and_coefficients_raises():
    with pytest.raises(ValueError):
        magnetic_form_factor_j0(1.0)


def test_wrong_coefficient_count_raises():
    with pytest.raises(ValueError):
        magnetic_form_factor_j0(1.0, coefficients=(1.0, 2.0, 3.0))


def test_available_ions_sorted_and_nonempty():
    ions = available_ions()
    assert "Mn2" in ions and "Fe2" in ions and "U4" in ions
    assert ions == sorted(ions)
