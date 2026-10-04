"""Golden source-arithmetic probes without a Mantid runtime dependency."""

import math

import numpy as np
import pytest

from nfit import raw_dgs
from nfit.raw_dgs_geometry_precision import (
    HE3_EFFICIENCY_EXPONENTIAL_CONSTANT,
    mantid_cylinder_radius,
    mantid_he3_exponent,
)


@pytest.mark.parametrize("axis", ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -2, 0]))
def test_source_radius_includes_half_height_shape_center(axis):
    # Mantid v6.16 ShapeFactory, Line::intersect, solveQuadratic, and He3
    # distToSurface produce this radius; it is below the nominal IDF radius.
    radius = mantid_cylinder_radius(.0127, axis, [0, 0, 0], .009375)
    assert radius == .012699997477625402
    assert radius < .0127


def test_source_radius_uses_supplied_bottom_center_and_height():
    # Centering the cylinder about the origin instead changes the cancellation
    # sign, demonstrating why discarding bottom/height loses Mantid parity.
    centered = mantid_cylinder_radius(.0127, [0, 1, 0], [0, -.0046875, 0], .009375)
    assert centered == .012700002060910265
    assert centered > .0127
    assert mantid_cylinder_radius(.0127, [0, 1, 0], [0, 0, 0], .0020240625) is not None


@pytest.mark.parametrize("radius,axis,bottom,height", (
    (.0127, [0, 1, 0], [0, 0, 0], None),
    (.0127, [0, 0, 0], [0, 0, 0], .009375),
    (.0127, [0, 1, 1], [0, 0, 0], .009375),
    (.0127, [1e-20, 1, 0], [0, 0, 0], .009375),
    (.0127, [0, 1, 0], [1e-20, 0, 0], .009375),
    (.0127, [0, 1, 0], [0, .1, 0], .009375),
    (.0127, [0, 1, 0], [0, 0, 0], 0),
    (-.0127, [0, 1, 0], [0, 0, 0], .009375),
    (math.nan, [0, 1, 0], [0, 0, 0], .009375),
    (.0127, [0, math.inf, 0], [0, 0, 0], .009375),
))
def test_unsupported_geometry_is_explicit(radius, axis, bottom, height):
    assert mantid_cylinder_radius(radius, axis, bottom, height) is None


def test_mantid_radius_recovers_float32_he3_stage_weights():
    radius = mantid_cylinder_radius(.0127, [0, 1, 0], [0, 0, 0], .009375)
    # Synthetic unit direction at the same angle as a Mantid reference probe.
    position = [.9667238238100632, .2558222986333482, 0]
    exponent = mantid_he3_exponent(position, [0, 1, 0], radius, (10, .0008, 290))
    assert exponent == 1.84685547384065
    wavelengths = np.array([1.164273692918552, 1.1674777928926148,
                            1.1762355270882145, 1.1777308432761968])
    expected = np.array([1.1318050622940063, 1.1309255361557007,
                         1.1285549402236938, 1.128154993057251], dtype=np.float32)
    weights = np.asarray(1 / (1 - np.exp(-exponent * wavelengths)), dtype=np.float32)
    np.testing.assert_array_equal(weights, expected)
    nominal_exponent = mantid_he3_exponent(position, [0, 1, 0], .0127, (10, .0008, 290))
    nominal_weights = np.asarray(1 / (1 - np.exp(-nominal_exponent * wavelengths)), dtype=np.float32)
    assert np.count_nonzero(nominal_weights != expected) == 3


def test_exponent_has_source_association_and_world_axis_normalization():
    result = mantid_he3_exponent([2, 0, 0], [0, 3, 0], .0127, (10, .0008, 290))
    pathlength = (2 * .0127 - 2 * .0008) / 1.0
    assert result == HE3_EFFICIENCY_EXPONENTIAL_CONSTANT * (10 / 290) * pathlength
    assert mantid_he3_exponent([0, 1, 0], [0, 1, 0], .0127, (10, .0008, 290)) == 0


@pytest.mark.parametrize("parameters", (None, (0, .0008, 290), (10, .02, 290),
                                        (10, .0008, 0), (math.nan, .0008, 290)))
def test_inactive_exponent_contract(parameters):
    assert mantid_he3_exponent([1, 0, 0], [0, 1, 0], .0127, parameters) == 0


def _idf(*, height=".009375", axis='x="0" y="1" z="0"',
         bottom='x="0" y="0" z="0"', cylinder_extra="", type_extra=""):
    height_element = "" if height is None else f'<height val="{height}"/>'
    return f'''<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="panel" idlist="panel"><location/></component>
      <component-link name="panel">
        <parameter name="tube_pressure"><value val="10"/></parameter>
        <parameter name="tube_thickness"><value val=".0008"/></parameter>
        <parameter name="tube_temperature"><value val="290"/></parameter>
      </component-link>
      <type name="panel"><component type="pixel"><location x="1" z="2"/></component></type>
      <type name="pixel" is="detector"><cylinder id="tube">
        <centre-of-bottom-base {bottom}/><axis {axis}/><radius val=".0127"/>
        {height_element}{cylinder_extra}
      </cylinder>{type_extra}</type>
      <idlist idname="panel"><id start="42" end="42"/></idlist>
    </instrument>'''.encode()


def test_full_xml_cache_key_distinguishes_height_with_unchanged_positions():
    # Distinct valid heights alter the ray's cancellation while nominal
    # efficiency and detector positions remain identical.
    first_xml, second_xml = _idf(), _idf(height=".001")
    first = raw_dgs._detector_geometry_from_xml(first_xml)
    second = raw_dgs._detector_geometry_from_xml(second_xml)
    assert raw_dgs._detector_geometry_from_xml(first_xml) is first
    assert second is not first
    np.testing.assert_array_equal(second.positions, first.positions)
    np.testing.assert_array_equal(second.he3_exponents, first.he3_exponents)
    assert second.mantid_he3_exponents[0] != first.mantid_he3_exponents[0]
    assert first.mantid_he3_exponents[0] < first.he3_exponents[0]
    assert second.mantid_he3_exponents[0] > second.he3_exponents[0]
    assert not first.mantid_he3_exponents.flags.writeable


def test_lookup_explicitly_selects_precision_and_retains_nominal_default():
    geometry = raw_dgs._DetectorGeometry(
        np.array([99, 42]), np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([.99, .42]), np.array([.991, .421]),
    )
    ids = np.array([42, 99, 900, 42])
    positions, nominal, valid = geometry.event_geometry_for_ids(ids)
    mantid_positions, precise, mantid_valid = geometry.event_geometry_for_ids(ids, mantid_precision=True)
    np.testing.assert_array_equal(nominal, [.42, .99, 0, .42])
    np.testing.assert_array_equal(precise, [.421, .991, 0, .421])
    np.testing.assert_array_equal(positions, mantid_positions)
    np.testing.assert_array_equal(valid, mantid_valid)
    np.testing.assert_array_equal(valid, [True, True, False, True])


def test_mask_preserves_both_exponents_and_excludes_unsupported_detectors():
    from nfit.mdevent import DetectorNormalization

    geometry = raw_dgs._DetectorGeometry(
        np.array([42, 43]), np.array([[1, 0, 2], [1, 0, 3]]),
        np.array([.42, .43]), np.array([.421, math.nan]),
    )
    with pytest.raises(ValueError, match="detector cylinder shape"):
        geometry.event_geometry_for_ids(np.array([42, 43]), mantid_precision=True)
    mask = DetectorNormalization(None, np.array([42, 43]), np.array([1, 0]), np.zeros(2))
    masked = raw_dgs._masked_detector_geometry(geometry, None, mask)
    np.testing.assert_array_equal(masked.he3_exponents, [.42])
    np.testing.assert_array_equal(masked.mantid_he3_exponents, [.421])
    _, exponents, valid = masked.event_geometry_for_ids(np.array([42, 43]), mantid_precision=True)
    np.testing.assert_array_equal(exponents, [.421, 0])
    np.testing.assert_array_equal(valid, [True, False])
    assert not masked.mantid_he3_exponents.flags.writeable


@pytest.mark.parametrize("settings", (
    {"axis": 'x="0" y="1" z="1"'},
    {"bottom": 'x=".001" y="0" z="0"'},
    {"bottom": 'r=".001" t="90" p="0"'},
    {"cylinder_extra": '<rotate x="0" y="0" z="45"/>'},
    {"type_extra": '<rotate-all x="0" y="0" z="45"/>'},
))
def test_xml_unsupported_shape_is_rejected_only_for_selected_mantid_events(settings):
    geometry = raw_dgs._detector_geometry_from_xml(_idf(**settings))
    _, nominal, valid = geometry.event_geometry_for_ids(np.array([42]))
    assert valid[0] and np.isfinite(nominal[0]) and nominal[0] > 0
    with pytest.raises(ValueError, match="detector cylinder shape"):
        geometry.event_geometry_for_ids(np.array([42]), mantid_precision=True)
    # Unknown IDs do not turn an unused unsupported shape into an error.
    _, exponent, valid = geometry.event_geometry_for_ids(np.array([900]), mantid_precision=True)
    assert not valid[0] and exponent[0] == 0


def test_incomplete_historical_idf_keeps_nominal_radius_compatibility():
    geometry = raw_dgs._detector_geometry_from_xml(_idf(height=None))
    np.testing.assert_allclose(geometry.mantid_he3_exponents, geometry.he3_exponents, rtol=2e-15)
    _, exponents, valid = geometry.event_geometry_for_ids(np.array([42]), mantid_precision=True)
    assert valid[0] and np.isfinite(exponents[0])


def test_spherical_origin_and_cartesian_origin_are_equivalent():
    cartesian = raw_dgs._detector_geometry_from_xml(_idf())
    spherical = raw_dgs._detector_geometry_from_xml(_idf(bottom='r="0" t="0" p="0"'))
    np.testing.assert_array_equal(cartesian.mantid_he3_exponents, spherical.mantid_he3_exponents)


def test_inactive_he3_parameters_do_not_reject_unused_shape_precision():
    xml = _idf(axis='x="0" y="1" z="1"').replace(b'val="10"', b'val="0"')
    geometry = raw_dgs._detector_geometry_from_xml(xml)
    _, exponents, valid = geometry.event_geometry_for_ids(np.array([42]), mantid_precision=True)
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(exponents, [0])


def test_corelli_calibrated_geometry_uses_nominal_default_lookup(tmp_path):
    from nfit.corelli import _geometry_from_solid_angle

    h5py = pytest.importorskip("h5py")
    # A non-cardinal shape is unsupported by the DGS Mantid branch, but CORELLI
    # keeps its existing nominal correction even after calibrated positions.
    fallback = raw_dgs._detector_geometry_from_xml(_idf(axis='x="0" y="1" z="1"'))
    assert np.isnan(fallback.mantid_he3_exponents[0])
    source = tmp_path / "corelli_solid_angle.nxs"
    with h5py.File(source, "w") as handle:
        detector = handle.create_group("mantid_workspace_1/instrument/detector")
        detector.create_dataset("detector_list", data=[42])
        detector.create_dataset("detector_positions", data=[[2.0, 90.0, 0.0]])
    calibrated = _geometry_from_solid_angle(source, fallback)
    _, exponents, valid = calibrated.event_geometry_for_ids(np.array([42]))
    assert valid[0]
    np.testing.assert_array_equal(exponents, fallback.he3_exponents)
