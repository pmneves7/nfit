"""Raw HYSPEC preparation has explicit TOF and detector-component conventions."""

import json
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import h5py
import numpy as np
import pytest

from nfit.raw_dgs_geometry import location_transform, rotation_matrix
from nfit.raw_dgs_hyspec import (
    raw_hyspec_tof_keep,
    resolved_hyspec_preprocessing,
    rotated_hyspec_idf_xml,
    tank_y_rotation_matrix,
)


@contextmanager
def _entry(tmp_path, *, instrument="HYSPEC", logs=None):
    with h5py.File(tmp_path / "source.h5", "w") as handle:
        entry = handle.create_group("entry")
        xml = f'<instrument name="{instrument}"/>'.encode()
        entry.create_dataset("instrument/instrument_xml/data", data=np.frombuffer(xml, dtype="u1"))
        for name, values in (logs if logs is not None else {"msd": [1803.], "psda": [0.]}).items():
            group = entry.create_group("DASlogs/" + name)
            group.create_dataset("value", data=values)
            group.create_dataset("time", data=np.arange(len(values), dtype=float)**2)
        yield entry


def test_default_crop_matches_independent_shiver_formula_and_effective_energy(tmp_path):
    with _entry(tmp_path) as entry:
        resolved = resolved_hyspec_preprocessing(entry, {}, 15.)
        # Independent numerical reference to the installed Shiver HYSPEC branch.
        np.testing.assert_allclose(resolved["raw_tof_bounds_microseconds"],
            [17939.510164997206, 35546.17683166387], rtol=0, atol=1e-11)
        assert resolved["raw_tof_center_microseconds"] == pytest.approx(26742.843498330538)
        assert resolved["tof_filter_stage"] == "raw_event_time_offset_before_t0"
        assert resolved["tank_offset_degrees"] == 0.
        higher = resolved_hyspec_preprocessing(entry, {}, 60.)
        assert higher["raw_tof_center_microseconds"] == resolved["raw_tof_center_microseconds"] / 2
        assert np.diff(higher["raw_tof_bounds_microseconds"])[0] == pytest.approx(2*(1e6/120+470))
        assert len(resolved) < 20
        json.dumps(resolved, allow_nan=False)


def test_crop_uses_recorded_times_before_t0_and_preserves_input(tmp_path):
    with _entry(tmp_path) as entry:
        resolved = resolved_hyspec_preprocessing(entry, {}, 15.)
    lower, upper = resolved["raw_tof_bounds_microseconds"]
    recorded = np.array([lower+10, upper-10])
    before = recorded.copy()
    np.testing.assert_array_equal(raw_hyspec_tof_keep(recorded, resolved), [True, True])
    # Applying a 100-us T0 shift before this crop would wrongly reject an event.
    np.testing.assert_array_equal(raw_hyspec_tof_keep(recorded-100., resolved), [False, True])
    np.testing.assert_array_equal(recorded, before)


def test_crop_boundary_and_nonfinite_recorded_times(tmp_path):
    with _entry(tmp_path) as entry:
        resolved = resolved_hyspec_preprocessing(entry, {}, 15.)
    lower, upper = resolved["raw_tof_bounds_microseconds"]
    times = [np.nextafter(lower, -np.inf), lower, np.nextafter(lower, np.inf),
             np.nextafter(upper, -np.inf), upper, np.nextafter(upper, np.inf), np.nan, np.inf]
    np.testing.assert_array_equal(raw_hyspec_tof_keep(times, resolved),
        [False, True, True, True, True, False, False, False])


def test_arithmetic_means_ignore_time_spacing_and_idf_six_digit_rounding(tmp_path):
    logs = {"msd": [1803.123456789, 1900.234567891, 2100.345678912],
            "psda": [4.123456789, 7.234567891, 1.345678912],
            "psr": [1001.123456789, 1500.234567891, 3000.345678912]}
    with _entry(tmp_path, logs=logs) as entry:
        # Artificially make the final duration dominate: this must not change
        # Shiver's getStatistics().mean into IDF time-averaged extraction.
        for name in logs:
            entry[f"DASlogs/{name}/time"][...] = [0., 1., 100.]
        resolved = resolved_hyspec_preprocessing(entry, {}, 15.)
    mean_msd, mean_psda, mean_psr = (float(np.mean(logs[name])) for name in ("msd", "psda", "psr"))
    assert resolved["msd_arithmetic_mean_millimetres"] == mean_msd
    assert resolved["psda_arithmetic_mean_degrees"] == mean_psda
    assert resolved["psr_arithmetic_mean_millimetres"] == mean_psr
    assert mean_msd != float(format(mean_msd, ".6g"))
    expected = mean_psda*(1-mean_psr/4200.)
    assert resolved["tank_offset_degrees"] == expected
    assert resolved["tank_offset_source"] == "arithmetic_mean_run_logs"
    center = (39000+mean_msd+4500)*1000/np.sqrt(15./5.227e-6)
    assert resolved["raw_tof_center_microseconds"] == center


@pytest.mark.parametrize("offset", [0., -2.123456789, 3.5])
def test_explicit_tank_override_including_zero_does_not_read_polarization_logs(tmp_path, offset):
    with _entry(tmp_path, logs={"msd": [1803.]}) as entry:
        resolved = resolved_hyspec_preprocessing(entry, {"hyspec_tank_offset_override": offset}, 15.)
    assert resolved["tank_offset_degrees"] == offset
    assert resolved["tank_offset_source"] == "explicit_override"
    assert resolved["psda_arithmetic_mean_degrees"] is None
    assert resolved["psr_arithmetic_mean_millimetres"] is None


def test_disabled_crop_does_not_require_msd_or_filter_any_recorded_times(tmp_path):
    with _entry(tmp_path, logs={"psda": [0.]}) as entry:
        resolved = resolved_hyspec_preprocessing(entry, {"hyspec_tof_crop": False}, 15.)
    assert resolved["raw_tof_bounds_microseconds"] is None
    assert resolved["raw_tof_center_microseconds"] is None
    assert resolved["msd_arithmetic_mean_millimetres"] is None
    np.testing.assert_array_equal(raw_hyspec_tof_keep([-10., np.nan, np.inf], resolved), [True]*3)


@pytest.mark.parametrize("instrument", ["SEQUOIA", "HYS", "CORELLI", "unknown"])
def test_other_instruments_are_unchanged_and_do_not_require_hyspec_logs(tmp_path, instrument):
    with _entry(tmp_path, instrument=instrument, logs={}) as entry:
        resolved = resolved_hyspec_preprocessing(entry, {"hyspec_tof_crop": "invalid"}, -1.)
    assert resolved is None
    np.testing.assert_array_equal(raw_hyspec_tof_keep([-100., 0., np.nan], resolved), [True]*3)


def test_already_inspected_instrument_identity_is_supported(tmp_path):
    with _entry(tmp_path, instrument="unknown") as entry:
        resolved = resolved_hyspec_preprocessing(entry, {}, 15., instrument_name="HYSPEC")
    assert resolved["instrument_name"] == "HYSPEC"


@pytest.mark.parametrize("log", ["msd", "psda", "psr"])
def test_missing_required_log_fails_explicitly(tmp_path, log):
    logs = {"msd": [1803.], "psda": [2.], "psr": [1200.]}
    logs.pop(log)
    with _entry(tmp_path, logs=logs) as entry:
        with pytest.raises(ValueError, match=log):
            resolved_hyspec_preprocessing(entry, {}, 15.)


@pytest.mark.parametrize("settings,energy", [({}, 0.), ({}, -1.), ({}, np.nan),
    ({"hyspec_tof_crop": "false"}, 15.), ({"hyspec_tank_offset_override": np.inf}, 15.),
    ({"hyspec_tank_offset_override": False}, 15.)])
def test_invalid_preprocessing_settings_fail(tmp_path, settings, energy):
    with _entry(tmp_path) as entry:
        with pytest.raises(ValueError, match="HYSPEC|hyspec_tof_crop"):
            resolved_hyspec_preprocessing(entry, settings, energy)


def test_tank_matrix_is_active_right_handed_y_rotation_and_composes_relatively():
    rotation = tank_y_rotation_matrix(90.)
    np.testing.assert_allclose(rotation @ np.array([1., 0., 0.]), [0., 0., -1.], atol=1e-15)
    np.testing.assert_allclose(rotation @ np.array([0., 0., 1.]), [1., 0., 0.], atol=1e-15)
    np.testing.assert_allclose(rotation @ np.array([0., 1., 0.]), [0., 1., 0.], atol=1e-15)
    np.testing.assert_allclose(tank_y_rotation_matrix(2.) @ tank_y_rotation_matrix(-70.),
        tank_y_rotation_matrix(-68.), atol=1e-15)
    np.testing.assert_array_equal(tank_y_rotation_matrix(0.), np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        tank_y_rotation_matrix(np.nan)


@pytest.mark.parametrize("namespace", ["", ' xmlns="http://www.mantidproject.org/IDF/1.0"'])
def test_tank_xml_rotates_about_component_position_with_noncommuting_old_rotation(namespace):
    xml = f'''<instrument name="HYSPEC"{namespace}>
      <component type="Tank" idlist="detectors">
        <location x="1" y="2" z="3" rot="31.123456789" axis-x="1" axis-y="0" axis-z="0">
          <rot val="-70" axis-y="1"><rot val="12" axis-z="1"/></rot>
        </location>
      </component>
      <component type="monitor"><location z="4"/></component>
      <type name="Tank"><component type="tube"><location x="2"/></component></type>
      <idlist idname="detectors"><id start="1" end="128"/></idlist>
    </instrument>'''.encode()
    before = ET.fromstring(xml)
    result = rotated_hyspec_idf_xml(xml, 7.3456789123)
    after = ET.fromstring(result)
    components_before = before.findall("{*}component")
    components_after = after.findall("{*}component")
    position, old_rotation = location_transform(components_before[0].find("{*}location"))
    new_position, new_rotation = location_transform(components_after[0].find("{*}location"))
    np.testing.assert_array_equal(new_position, position)
    delta = rotation_matrix(7.3456789123, [0, 1, 0])
    np.testing.assert_allclose(new_rotation, old_rotation @ delta, atol=1e-15)
    assert not np.allclose(new_rotation, delta @ old_rotation)
    local_pixel = np.array([2., 0., 0.])
    old_pixel = position + old_rotation @ local_pixel
    new_pixel = new_position + new_rotation @ local_pixel
    relative_global = old_rotation @ delta @ old_rotation.T
    np.testing.assert_allclose(new_pixel, position + relative_global @ (old_pixel-position), atol=1e-15)
    # Tube directions and all other detector shape metadata remain owned by
    # the existing parser; the changed parent transform rotates their axes.
    tube_axis = np.array([0., 1., 0.])
    np.testing.assert_allclose(new_rotation @ tube_axis, old_rotation @ delta @ tube_axis, atol=1e-15)
    assert components_after[0].attrib == components_before[0].attrib
    assert ET.tostring(components_after[1]) == ET.tostring(components_before[1])
    assert ET.tostring(after.find("{*}type")) == ET.tostring(before.find("{*}type"))
    assert ET.tostring(after.find("{*}idlist")) == ET.tostring(before.find("{*}idlist"))
    assert result != xml
    assert b'rot="31.123456789"' in xml


def test_named_tank_location_and_spherical_position_are_preserved():
    xml = b'''<instrument name="HYSPEC"><component type="assembly">
      <location name="Tank" r="2" t="40" p="15"><rot val="-70" axis-y="1"/></location>
      </component></instrument>'''
    before = location_transform(ET.fromstring(xml).find(".//{*}location"))
    after = location_transform(ET.fromstring(rotated_hyspec_idf_xml(xml, 2.)).find(".//{*}location"))
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_allclose(after[1], tank_y_rotation_matrix(-68.), atol=1e-15)


def test_relative_rotation_matches_independent_mantid_noncommuting_probe():
    # External Mantid RelativeRotation=True reference, retained as numerical
    # constants. Unit tests neither import nor execute Mantid.
    xml = b'''<instrument name="HYSPEC"><component type="Tank"><location>
      <rot val="20" axis-x="1"/><rot val="30" axis-y="1"/><rot val="40" axis-z="1"/>
      </location></component></instrument>'''
    _, rotation = location_transform(ET.fromstring(rotated_hyspec_idf_xml(xml, 17.)).find(".//{*}location"))
    expected = [[.4882400614, -.5566703992, .6721158449],
                [.7895069847, .6099231552, -.0683554372],
                [-.3718875683, .5640140170, .7372840872]]
    np.testing.assert_allclose(rotation, expected, rtol=0, atol=5e-11)


def test_zero_offset_and_other_instrument_preserve_exact_xml_bytes():
    xml = b'<instrument name="HYSPEC"> <!-- retained --> </instrument>'
    assert rotated_hyspec_idf_xml(xml, 0.) is xml
    other = b'<instrument name="SEQUOIA"> <!-- retained --> </instrument>'
    assert rotated_hyspec_idf_xml(other, 3.) is other


def test_existing_detector_parser_rebuilds_tank_positions_and_he3_shape_corrections():
    from nfit.raw_dgs import _detector_geometry_from_xml
    from nfit.raw_dgs_geometry_precision import mantid_cylinder_radius, mantid_he3_exponent

    xml = b'''<instrument name="HYSPEC" xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="Tank" idlist="Tank"><location x="1" y="1" z="3"/></component>
      <component type="other" idlist="other"><location z="5"/></component>
      <component-link name="Tank">
        <parameter name="tube_pressure"><value val="10"/></parameter>
        <parameter name="tube_thickness"><value val=".0008"/></parameter>
        <parameter name="tube_temperature"><value val="290"/></parameter>
      </component-link>
      <type name="Tank"><component type="pixel"><location x="2"/></component></type>
      <type name="other"><component type="pixel"><location x="-2"/></component></type>
      <type name="pixel" is="detector"><cylinder id="tube">
        <centre-of-bottom-base x="0" y="0" z="0"/><axis x="0" y="1" z="0"/>
        <radius val=".0127"/><height val=".009375"/>
      </cylinder></type>
      <idlist idname="Tank"><id start="42" end="42"/></idlist>
      <idlist idname="other"><id start="43" end="43"/></idlist>
    </instrument>'''
    old = _detector_geometry_from_xml(xml)
    new = _detector_geometry_from_xml(rotated_hyspec_idf_xml(xml, 90.))
    assert new is not old
    np.testing.assert_array_equal(new.detector_ids, old.detector_ids)
    np.testing.assert_allclose(new.positions[0], [1., 1., 1.], atol=1e-15)
    np.testing.assert_array_equal(new.positions[1], old.positions[1])
    ray_radius = mantid_cylinder_radius(.0127, [0, 1, 0], [0, 0, 0], .009375)
    expected = mantid_he3_exponent([1, 1, 1], [0, 1, 0], ray_radius, (10, .0008, 290))
    assert new.mantid_he3_exponents[0] == pytest.approx(expected, rel=1e-15)
    assert new.mantid_he3_exponents[0] != old.mantid_he3_exponents[0]
    assert new.mantid_he3_exponents[1] == old.mantid_he3_exponents[1]


@pytest.mark.parametrize("components", ["",
    '<component type="Tank"><location/><location/></component>',
    '<component type="Tank"><location/></component><component type="Tank"><location/></component>',
    '<type name="assembly"><component type="Tank"><location/></component></type>',
    '<component type="Tank"><location><parameter name="roty"><value val="1"/></parameter></location></component>',
])
def test_invalid_or_unresolved_tank_location_fails_explicitly(components):
    xml = ('<instrument name="HYSPEC">' + components + '</instrument>').encode()
    with pytest.raises(ValueError, match="Tank"):
        rotated_hyspec_idf_xml(xml, 1.)
