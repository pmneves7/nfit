import numpy as np
import pytest

from nfit.form_factors import (
    J0_COEFFICIENTS,
    J2_COEFFICIENTS,
    available_ions,
    dipole_j2_weight,
    form_factor_profile_sq,
    form_factor_sq,
    magnetic_form_factor,
    magnetic_form_factor_j0,
    magnetic_form_factor_j2,
    magnetic_form_factor_profile,
)


def test_all_tabulated_ions_are_normalized_at_zero_q():
    for ion, (A, _a, B, _b, C, _c, D) in J0_COEFFICIENTS.items():
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


def test_effective_form_factor_mixture_sums_amplitudes_before_squaring():
    q = np.asarray([0.0, 2.0, 5.0])
    config = {
        "form_factor_mode": "mixture",
        "form_factor_mixture": [
            {"ion": "V3", "weight": 0.35},
            {"ion": "V4", "weight": 0.65},
        ],
    }
    expected = 0.35 * magnetic_form_factor(q, ion="V3") + 0.65 * magnetic_form_factor(
        q, ion="V4"
    )
    np.testing.assert_allclose(magnetic_form_factor_profile(q, config), expected)
    np.testing.assert_allclose(form_factor_profile_sq(q, config), expected**2)


def test_effective_form_factor_mixture_requires_normalized_amplitude_weights():
    with pytest.raises(ValueError, match="weights must sum to one"):
        magnetic_form_factor_profile(
            [1.0],
            {
                "form_factor_mode": "mixture",
                "form_factor_mixture": [
                    {"ion": "V3", "weight": 0.4},
                    {"ion": "V4", "weight": 0.4},
                ],
            },
        )


def test_form_factor_profile_infers_legacy_single_ion_and_custom_modes():
    q = np.asarray([0.0, 3.0])
    np.testing.assert_allclose(
        magnetic_form_factor_profile(q, {"ion": "V4"}),
        magnetic_form_factor(q, ion="V4"),
    )
    coefficients = J0_COEFFICIENTS["V3"]
    np.testing.assert_allclose(
        magnetic_form_factor_profile(
            q,
            {"ion": "__custom__", "form_factor_coefficients": coefficients},
        ),
        magnetic_form_factor(q, coefficients=coefficients),
    )


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


def _periodictable_tables():
    """Return the ``<j0>`` and ``<j2>`` tables as nfit keys them."""

    periodictable = pytest.importorskip("periodictable")
    j0: dict[str, tuple[float, ...]] = {}
    j2: dict[str, tuple[float, ...]] = {}
    for element in periodictable.elements:
        magnetic = getattr(element, "magnetic_ff", None)
        if not magnetic:
            continue
        for charge, data in magnetic.items():
            key = f"{element.symbol}{charge}"
            if getattr(data, "j0", None) is not None:
                j0[key] = tuple(float(value) for value in data.j0)
            if getattr(data, "j2", None) is not None:
                j2[key] = tuple(float(value) for value in data.j2)
    return j0, j2


def test_vendored_tables_match_their_periodictable_source():
    """The vendored coefficients must stay identical to the cited source.

    nfit vendors the tables so a published result cannot change because a
    dependency was updated, which makes an upstream revision or a transcription
    slip invisible without this check.
    """

    source_j0, source_j2 = _periodictable_tables()
    assert set(J0_COEFFICIENTS) == set(source_j0)
    assert set(J2_COEFFICIENTS) == set(source_j2)
    for key, values in J0_COEFFICIENTS.items():
        np.testing.assert_allclose(
            values, source_j0[key], rtol=0.0, atol=1.0e-6, err_msg=f"<j0> {key}"
        )
    for key, values in J2_COEFFICIENTS.items():
        np.testing.assert_allclose(
            values, source_j2[key], rtol=0.0, atol=1.0e-6, err_msg=f"<j2> {key}"
        )


def test_radial_integrals_match_periodictable_evaluation():
    """Evaluating the tables must reproduce the source implementation."""

    _source_j0, _source_j2 = _periodictable_tables()
    periodictable = pytest.importorskip("periodictable")
    q = np.array([0.0, 0.7, 1.9, 4.4, 8.0])
    for symbol, charge in (("Yb", 3), ("Nd", 3), ("Fe", 2), ("Mn", 2)):
        data = getattr(periodictable, symbol).magnetic_ff[charge]
        ion = f"{symbol}{charge}"
        np.testing.assert_allclose(
            magnetic_form_factor_j0(q, ion=ion), data.j0_Q(q), atol=1.0e-6
        )
        np.testing.assert_allclose(
            magnetic_form_factor_j2(q, ion=ion), data.j2_Q(q), atol=1.0e-6
        )


def test_j2_vanishes_at_zero_momentum_transfer():
    """<j2> carries a leading s^2, so it must vanish at Q = 0."""

    for ion in ("Yb3", "Nd3", "Fe2"):
        assert magnetic_form_factor_j2(0.0, ion=ion) == pytest.approx(0.0)


def test_dipole_approximation_reduces_to_j0_for_a_spin_only_moment():
    """g_J = 2 must return <j0> exactly and never consult the <j2> table."""

    q = np.array([0.0, 1.3, 5.5])
    assert dipole_j2_weight(2.0) == 0.0
    np.testing.assert_array_equal(
        magnetic_form_factor(q, ion="Yb3"), magnetic_form_factor_j0(q, ion="Yb3")
    )
    # Pr3 has <j0> but no <j2>; the spin-only default must still work.
    np.testing.assert_array_equal(
        magnetic_form_factor(q, ion="Pr3"), magnetic_form_factor_j0(q, ion="Pr3")
    )
    with pytest.raises(KeyError, match="no tabulated <j2>"):
        magnetic_form_factor(q, ion="Pr3", g_J=1.5)


def test_dipole_approximation_adds_the_orbital_term():
    """f = <j0> + (2/g_J - 1) <j2>, a large correction for a rare earth."""

    q = np.array([0.0, 2.0, 6.0])
    g_J = 8.0 / 7.0  # Yb(3+)
    assert dipole_j2_weight(g_J) == pytest.approx(0.75)
    expected = magnetic_form_factor_j0(q, ion="Yb3") + 0.75 * magnetic_form_factor_j2(
        q, ion="Yb3"
    )
    np.testing.assert_allclose(
        magnetic_form_factor(q, ion="Yb3", g_J=g_J), expected, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        form_factor_sq(q, ion="Yb3", g_J=g_J), expected**2, rtol=1.0e-12
    )
    # The orbital term matters: at 6 inverse Angstrom it is a third of f.
    spin_only = magnetic_form_factor_j0(q, ion="Yb3")[-1]
    assert magnetic_form_factor(q, ion="Yb3", g_J=g_J)[-1] / spin_only > 1.3
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="g_J"):
            dipole_j2_weight(bad)
