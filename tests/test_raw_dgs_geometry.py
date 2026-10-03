"""Run-dependent detector geometry has a complete identity and explicit bounds."""

import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from nfit import inspect_raw_dgs_run, raw_dgs
from nfit.raw_dgs_geometry import (
    evaluate_log_expression,
    location_transform,
    resolved_geometry_signature,
    resolved_idf_xml,
    source_distance,
)


def _idf(instrument="HYSPEC", *, tank_location=None):
    if tank_location is None:
        tank_location = '''<location>
          <parameter name="r-position"><value val="0"/></parameter>
          <parameter name="t-position"><logfile id="s2" eq="0.0+value"/></parameter>
          <parameter name="p-position"><value val="0"/></parameter>
          <parameter name="rotx"><value val="0"/></parameter>
          <parameter name="roty"><logfile id="s2" eq="0.0+value"/></parameter>
          <parameter name="rotz"><value val="0"/></parameter>
        </location>'''
    return f'''<instrument xmlns="http://www.mantidproject.org/IDF/1.0" name="{instrument}">
      <component type="moderator"><location>
        <parameter name="z"><logfile id="msd" eq="-0.001*value-38.980"/></parameter>
      </location></component>
      <component type="Tank" idlist="Tank">{tank_location}</component>
      <type name="moderator" is="Source"/>
      <type name="Tank"><component type="pixel"><location x="1" z="2"/></component></type>
      <type name="pixel" is="detector"/>
      <idlist idname="Tank"><id start="42" end="42"/></idlist>
    </instrument>'''.encode()


def _write(path, *, xml=None, s2=-70., msd=1803.):
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("run_number", data=[b"42"])
        entry.create_dataset("instrument/instrument_xml/data", data=np.frombuffer(xml or _idf(), dtype="u1"))
        entry.create_dataset("bank1_events/event_id", data=[42])
        entry.create_dataset("bank1_events/event_time_offset", data=[15000.])
        for name, value in (("s2", s2), ("msd", msd), ("EnergyRequest", 15.),
                            ("omega", 12.), ("phi", 0.), ("chi", 0.)):
            entry.create_dataset(f"DASlogs/{name}/value", data=[value])


@pytest.fixture(autouse=True)
def _cache():
    raw_dgs._detector_geometry_from_xml.cache_clear()
    yield
    raw_dgs._detector_geometry_from_xml.cache_clear()


def test_dynamic_moderator_and_top_level_tank_match_analytic_geometry(tmp_path):
    source = tmp_path / "HYS_42.nxs.h5"
    _write(source)
    with h5py.File(source, "r") as handle:
        entry = handle["entry"]
        assert source_distance(entry) == pytest.approx(40.783)
        assert raw_dgs._source_distance(entry) == source_distance(entry)
        moderator = ET.fromstring(_idf()).find("{*}component[@type='moderator']/{*}location")
        np.testing.assert_allclose(raw_dgs._idf_location(entry, moderator), [0, 0, -40.783])
    geometry = raw_dgs._detector_geometry(source)
    angle = np.deg2rad(-70.)
    expected = [np.cos(angle) + 2 * np.sin(angle), 0., -np.sin(angle) + 2 * np.cos(angle)]
    np.testing.assert_allclose(geometry.positions[0], expected, atol=1e-15)
    info = inspect_raw_dgs_run(source)
    assert info.l1 == pytest.approx(40.783)
    assert info.instrument_name == "HYSPEC"
    assert info.omega == 12.  # Sample goniometer is not the detector-tank log.
    with h5py.File(source, "r") as handle:
        assert info.geometry_signature == resolved_geometry_signature(handle["entry"])


def test_same_xml_changed_referenced_logs_do_not_reuse_geometry_or_identity(tmp_path):
    a, b, same = (tmp_path / name for name in ("a.h5", "b.h5", "same.h5"))
    _write(a, s2=-70.)
    _write(b, s2=-34.)
    _write(same, s2=-70.)
    first = raw_dgs._detector_geometry(a)
    assert raw_dgs._detector_geometry(same) is first
    second = raw_dgs._detector_geometry(b)
    assert second is not first
    np.testing.assert_array_equal(second.detector_ids, first.detector_ids)
    assert not np.allclose(second.positions, first.positions)
    assert inspect_raw_dgs_run(a).geometry_signature != inspect_raw_dgs_run(b).geometry_signature
    with h5py.File(a, "r+") as handle:
        handle["entry/DASlogs/s2/value"][0] = -34.
    assert raw_dgs._detector_geometry(a) is second


def test_same_xml_different_moderator_distance_updates_complete_identity(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write(a, msd=1803.)
    _write(b, msd=2200.)
    assert inspect_raw_dgs_run(a).geometry_signature != inspect_raw_dgs_run(b).geometry_signature
    assert inspect_raw_dgs_run(b).l1 == pytest.approx(41.18)
    # Full definition identity includes source geometry even when detector
    # pixel positions alone happen to be identical.
    assert raw_dgs._detector_geometry(a) is not raw_dgs._detector_geometry(b)
    np.testing.assert_array_equal(raw_dgs._detector_geometry(a).positions, raw_dgs._detector_geometry(b).positions)


def test_mixed_instruments_and_irrelevant_log_edits_are_distinguished_correctly(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write(a)
    _write(b, xml=_idf("SEQUOIA"))
    assert raw_dgs._detector_geometry(a) is not raw_dgs._detector_geometry(b)
    assert inspect_raw_dgs_run(a).geometry_signature != inspect_raw_dgs_run(b).geometry_signature
    first = raw_dgs._detector_geometry(a)
    with h5py.File(a, "r+") as handle:
        handle["entry/DASlogs/omega/value"][0] = 20.
    assert raw_dgs._detector_geometry(a) is first
    assert inspect_raw_dgs_run(a).omega == 20.


def test_static_definition_retains_original_cache_bytes_and_position(tmp_path):
    xml = _idf(tank_location='<location/>').replace(
        b'<location>\n        <parameter name="z"><logfile id="msd" eq="-0.001*value-38.980"/></parameter>\n      </location>',
        b'<location z="-40.783"/>',
    )
    assert b"logfile" not in xml
    source = tmp_path / "static.h5"
    _write(source, xml=xml)
    with h5py.File(source, "r") as handle:
        assert resolved_idf_xml(handle["entry"]) == xml
    geometry = raw_dgs._detector_geometry(source)
    assert geometry is raw_dgs._detector_geometry_from_xml(xml)
    np.testing.assert_array_equal(geometry.positions, [[1., 0., 2.]])


def test_spherical_parent_translation_and_nested_rotations_are_composed():
    xml = _idf(tank_location='<location r="3" t="90" p="0"><rot val="90" axis-y="1"/></location>')
    geometry = raw_dgs._detector_geometry_from_xml(xml)
    np.testing.assert_allclose(geometry.positions, [[5., 0., -1.]], atol=1e-15)
    position, rotation = location_transform(ET.fromstring(
        '<location><rot val="90" axis-x="1"><rot val="90" axis-y="1"/></rot></location>',
    ))
    np.testing.assert_array_equal(position, np.zeros(3))
    np.testing.assert_allclose(rotation @ [0., 0., 1.], [1., 0., 0.], atol=1e-15)


@pytest.mark.parametrize("edit,message", [
    (lambda xml: xml.replace(b'id="s2"', b'id="missing"'), "missing"),
    (lambda xml: xml.replace(b'0.0+value', b'sqrt(value)'), "Unsupported"),
    (lambda xml: xml.replace(b'name="roty"', b'name="unmodeled-motion"'), "Unsupported IDF location"),
    (lambda xml: xml.replace(b'id="s2"', b'id="s2" extract-single-value-as="sum"'), "extraction"),
    (lambda xml: xml.replace(b'<logfile id="s2"', b'<logfile interpolation="unknown" id="s2"'), "attributes"),
])
def test_unsupported_dynamic_definitions_fail_before_geometry_reduction(tmp_path, edit, message):
    source = tmp_path / "bad.h5"
    _write(source, xml=edit(_idf()))
    with pytest.raises(ValueError, match=message):
        raw_dgs._detector_geometry(source)


def test_nonfinite_dynamic_log_and_unresolved_dynamic_xml_fail_explicitly(tmp_path):
    source = tmp_path / "bad.h5"
    _write(source, s2=np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        raw_dgs._detector_geometry(source)
    with pytest.raises(ValueError, match="requires referenced run logs"):
        raw_dgs._detector_geometry_from_xml(_idf())


def test_referenced_log_uses_time_mean_and_explicit_first_extraction(tmp_path):
    source = tmp_path / "mean.h5"
    _write(source)
    with h5py.File(source, "r+") as handle:
        log = handle["entry/DASlogs/s2"]
        del log["value"]
        log.create_dataset("value", data=[0., 90., 0.])
        log.create_dataset("time", data=[0., 10., 100.])
        log.create_dataset("average_value", data=[30.])
        location = ET.fromstring('<location><parameter name="roty"><logfile id="s2"/></parameter></location>')
        _, rotation = location_transform(location, entry=handle["entry"])
        angle = np.rad2deg(np.arctan2(rotation[0, 2], rotation[0, 0]))
        assert angle == pytest.approx(42.6316, rel=1e-13)
        location.find("parameter/logfile").set("extract-single-value-as", "first")
        _, rotation = location_transform(location, entry=handle["entry"])
        assert np.rad2deg(np.arctan2(rotation[0, 2], rotation[0, 0])) == 0


def test_dynamic_euler_parameters_replace_static_rotation_in_xyz_order(tmp_path):
    source = tmp_path / "euler.h5"
    location = '''<location><rot val="45" axis-z="1"/>
      <parameter name="rotx"><value val="20"/></parameter>
      <parameter name="roty"><value val="30"/></parameter>
      <parameter name="rotz"><value val="40"/></parameter></location>'''
    _write(source, xml=_idf(tank_location=location))
    x, y, z = np.deg2rad([20, 30, 40])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    expected = (rx @ ry @ rz) @ [1, 0, 2]
    np.testing.assert_allclose(raw_dgs._detector_geometry(source).positions[0], expected, atol=1e-15)


def test_varying_log_without_timestamps_is_explicitly_unsupported(tmp_path):
    source = tmp_path / "no-times.h5"
    _write(source)
    with h5py.File(source, "r+") as handle:
        group = handle["entry/DASlogs/s2"]
        del group["value"]
        group.create_dataset("value", data=[0., 1.])
        group.create_dataset("average_value", data=[.5])
    with pytest.raises(ValueError, match="timestamps"):
        raw_dgs._detector_geometry(source)


def test_expression_arithmetic_is_compatible_bounded_and_fail_explicit():
    assert evaluate_log_expression("-0.001*value-38.980", 1803.) == pytest.approx(-40.783)
    assert raw_dgs._evaluate_idf_log_expression("value^2", 3.) == 9.
    for expression in ("sqrt(value)", "other + value", "'text'", "2**100000000"):
        with pytest.raises(ValueError):
            evaluate_log_expression(expression, 1.)


def test_mantid_log_scalar_rounding_precedes_equation_evaluation(tmp_path):
    source = tmp_path / "rounding.h5"
    _write(source, s2=123.456789)
    with h5py.File(source, "r") as handle:
        location = ET.fromstring('<location><parameter name="x"><logfile id="s2" eq="value/3"/></parameter></location>')
        position, _ = location_transform(location, entry=handle["entry"])
        assert position[0] == pytest.approx(123.457 / 3, rel=1e-14)
        assert position[0] != pytest.approx(41.1523, rel=1e-8)
        location.find("parameter/logfile").set("eq", "value-123.456")
        position, _ = location_transform(location, entry=handle["entry"])
        assert position[0] == pytest.approx(.001, abs=2e-14)


@pytest.mark.parametrize("mode,expected", [
    ("first", 12.3457), ("last", 23.4568), ("median", 23.4568),
    ("min", 12.3457), ("max", 78.9012), ("position2", 78.9012), ("mean", 49.1352),
])
def test_supported_log_extraction_matches_mantid_reference(mode, expected, tmp_path):
    source = tmp_path / "extraction.h5"
    _write(source)
    with h5py.File(source, "r+") as handle:
        log = handle["entry/DASlogs/s2"]
        del log["value"]
        log.create_dataset("value", data=[12.3456789, 78.9012345, 23.456789])
        log.create_dataset("time", data=[0., 10., 100.])
        location = ET.fromstring(
            f'<location><parameter name="x"><logfile id="s2" extract-single-value-as="{mode}"/></parameter></location>',
        )
        position, _ = location_transform(location, entry=handle["entry"])
        assert position[0] == pytest.approx(expected, rel=1e-13)
