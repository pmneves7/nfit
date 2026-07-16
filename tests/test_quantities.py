import numpy as np
import pytest

from nfit.quantities import convert_quantity, infer_quantity_type, normalize_unit


def test_cgs_si_bulk_conversions_include_four_pi():
    value = convert_quantity([1.0], "bulk_susceptibility", "cm^3/mol", "m^3/mol")
    np.testing.assert_allclose(value, [4.0 * np.pi * 1.0e-6])
    recovered = convert_quantity(value, "bulk_susceptibility", "m^3/mol", "cm^3/mol")
    np.testing.assert_allclose(recovered, [1.0])


def test_field_and_moment_conversions():
    np.testing.assert_allclose(convert_quantity([10000.0], "magnetic_field", "Oe", "T"), [1.0])
    np.testing.assert_allclose(convert_quantity([1.0], "magnetic_moment", "emu", "A m^2"), [1.0e-3])


def test_inference_and_unknown_conversion_are_conservative():
    assert normalize_unit("emu/mol/Oe") == "cm^3/mol"
    assert infer_quantity_type("Magnetic Field", "Oe") == "magnetic_field"
    assert infer_quantity_type("AC Susceptibility", "emu/Oe") == "bulk_susceptibility"
    assert infer_quantity_type("Inverse susceptibility") == "inverse_bulk_susceptibility"
    with pytest.raises(ValueError, match="cannot convert"):
        convert_quantity([1.0], "magnetic_moment", "counts", "emu")
