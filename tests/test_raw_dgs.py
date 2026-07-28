import numpy as np
import pytest

from nfit import bin_raw_dgs_group, inspect_raw_dgs_run, raw_dgs, raw_dgs_dataset_group
from nfit.raw_dgs import (
    TOF_US_PER_M_SQRT_MEV,
    _evaluate_mantid_t0_formula,
    _monitor_ei_t0,
    _retained_proton_charge_uah,
)


def _write_raw_dgs(path, *, with_he3=False):
    h5py = pytest.importorskip("h5py")
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-10"/></component>
      <component type="panel" idlist="panel"><location/></component>
      <type name="moderator" is="Source"/><type name="pixel" is="detector"/>
      <type name="panel"><component type="pixel"><location x="1" y="0" z="2"/></component></type>
      <idlist idname="panel"><id start="42" end="42"/></idlist>
    </instrument>"""
    if with_he3:
        xml = xml.replace(
            '<type name="moderator" is="Source"/><type name="pixel" is="detector"/>',
            """<type name="moderator" is="Source"/>
      <type name="pixel" is="detector"><cylinder id="tube"><axis x="0" y="1" z="0"/><radius val="0.01"/></cylinder></type>
      <component-link name="panel"><parameter name="tube_pressure"><value val="10.0"/></parameter><parameter name="tube_thickness"><value val="0.0008"/></parameter><parameter name="tube_temperature"><value val="290.0"/></parameter></component-link>""",
        )
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("run_number", data=[b"42"])
        instrument = entry.create_group("instrument")
        instrument.create_group("instrument_xml").create_dataset(
            "data", data=np.frombuffer(xml.encode(), dtype="u1")
        )
        bank = entry.create_group("bank1_events")
        bank.create_dataset("event_id", data=[42, 42])
        # One useful event and one prompt event, which must be discarded.
        bank.create_dataset("event_time_offset", data=[9000.0, 1.0])
        logs = entry.create_group("DASlogs")
        for name, value in (
            ("BL17:Det:TH:BL:Ei", 20.0),
            ("proton_charge", 2.0),
            ("omega", 0.0),
            ("phi", 0.0),
            ("chi", 0.0),
        ):
            logs.create_group(name).create_dataset("average_value", data=[value])


def test_raw_dgs_metadata_and_streamed_hkle_binning(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    info = inspect_raw_dgs_run(source)
    group = raw_dgs_dataset_group([source])

    assert info.run_number == "42"
    assert info.event_count == 2
    assert info.incident_energy == 20.0
    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
        max_batch_bytes=128,
    )

    # The prompt event is rejected; the remaining event carries ki/kf and is
    # divided by the detector-trajectory normalization.
    assert result.num_events.item() == 1.0
    assert np.isfinite(result.signal.item()) and result.signal.item() > 0.0
    assert np.isfinite(result.errors.item()) and result.errors.item() > 0.0
    assert result.metadata["ki_kf_normalization"] is True
    assert not result.mask.item()


def test_raw_dgs_trajectory_normalization_keeps_each_runs_detector_geometry(monkeypatch, tmp_path):
    first_source = tmp_path / "SEQ_42.nxs.h5"
    second_source = tmp_path / "SEQ_43.nxs.h5"
    _write_raw_dgs(first_source)
    _write_raw_dgs(second_source)
    group = raw_dgs_dataset_group([first_source, second_source])
    geometries = {
        first_source: raw_dgs._DetectorGeometry(
            detector_ids=np.array([42]),
            positions=np.array([[1.0, 0.0, 0.0]]),
            he3_exponents=np.zeros(1),
        ),
        second_source: raw_dgs._DetectorGeometry(
            detector_ids=np.array([42]),
            positions=np.array([[0.0, 1.0, 0.0]]),
            he3_exponents=np.zeros(1),
        ),
    }
    directions = []

    monkeypatch.setattr(
        raw_dgs,
        "_detector_geometry",
        lambda path: geometries[path],
    )
    monkeypatch.setattr(raw_dgs, "_MDEVENT_NUMBA", None)
    monkeypatch.setattr(
        raw_dgs,
        "_accumulate_detector_trajectory",
        lambda result, edges, inverse, direction, ei, bounds, weight: directions.append(
            direction.copy()
        ),
    )

    raw_dgs._trajectory_normalization(
        group,
        group.datasets,
        [np.array([-10.0, 10.0])] * 4,
        (1, 1, 1, 1),
        np.eye(4),
        None,
        None,
    )

    np.testing.assert_allclose(
        directions,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )


def test_raw_dgs_detector_mask_removes_events(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    mask = tmp_path / "mask.nxs"
    _write_raw_dgs(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(mask, "w") as handle:
        entry = handle.create_group("mantid_workspace_1")
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("detector_list", data=[42])
        workspace = entry.create_group("workspace")
        workspace.create_dataset("values", data=[[0.0]])
        workspace.create_dataset("errors", data=[[0.0]])
    group = raw_dgs_dataset_group([source], mask_path=mask)
    result = bin_raw_dgs_group(
        group, lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1, 1, 1, 1]
    )
    assert result.mask.item()
    assert result.num_events.item() == 0.0


def test_raw_dgs_uses_mantid_ki_over_kf_event_weight(monkeypatch, tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", lambda *args: np.ones(args[3]))

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    incident_tof = TOF_US_PER_M_SQRT_MEV * 10.0 / np.sqrt(20.0)
    final_tof = 9000.0 - incident_tof
    final_energy = (TOF_US_PER_M_SQRT_MEV * np.sqrt(5.0) / final_tof) ** 2
    expected = np.sqrt(20.0 / final_energy)
    assert result.signal.item() == pytest.approx(expected)
    assert result.errors.item() == pytest.approx(expected)


def test_raw_dgs_applies_mantid_he3_tube_efficiency(monkeypatch, tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source, with_he3=True)
    group = raw_dgs_dataset_group([source])
    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", lambda *args: np.ones(args[3]))

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    incident_tof = TOF_US_PER_M_SQRT_MEV * 10.0 / np.sqrt(20.0)
    final_tof = 9000.0 - incident_tof
    final_energy = (TOF_US_PER_M_SQRT_MEV * np.sqrt(5.0) / final_tof) ** 2
    kf = np.sqrt(final_energy / raw_dgs.ENERGY_TO_K2)
    exponent = (
        raw_dgs.HE3_EFFICIENCY_EXPONENTIAL_CONSTANT * (10.0 / 290.0) * (2.0 * (0.01 - 0.0008))
    )
    expected = np.sqrt(20.0 / final_energy) / (-np.expm1(-exponent * 2.0 * np.pi / kf))
    assert result.signal.item() == pytest.approx(expected)
    assert result.errors.item() == pytest.approx(expected)
    assert result.metadata["he3_detector_efficiency_correction"] is True


def test_raw_dgs_integrates_retained_pulse_charge_in_microampere_hours(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "a") as handle:
        log = handle["entry/DASlogs/proton_charge"]
        log.create_dataset("value", data=[100.0, 100.0, 10.0])

    with h5py.File(source, "r") as handle:
        charge = _retained_proton_charge_uah(handle["entry"], 95.0)
    assert charge == pytest.approx(200.0 / 3.6e9)


def test_raw_dgs_feldman_cousins_zero_error_is_normalized(monkeypatch, tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", lambda *args: np.full(args[3], 2.0))

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, 19],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    assert result.num_events.item() == 0.0
    assert result.signal.item() == 0.0
    assert result.errors.item() == pytest.approx(1.29 / 2.0)


def test_monitor_fit_discovers_non_sequoia_monitor_names(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "CNCS_1.nxs.h5"
    energy, t0 = 20.0, 7.0
    distances = (8.0, 22.0)
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-15"/></component>
      <component type="monitor"><location name="upstream_monitor" z="-7"/></component>
      <component type="monitor"><location name="downstream_monitor" z="7"/></component>
    </instrument>"""
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("instrument").create_group("instrument_xml").create_dataset(
            "data",
            data=np.frombuffer(xml.encode(), dtype="u1"),
        )
        for name, distance in zip(
            ("upstream_monitor", "downstream_monitor"), distances, strict=True
        ):
            centre = t0 + TOF_US_PER_M_SQRT_MEV * distance / np.sqrt(energy)
            entry.create_group(name).create_dataset(
                "event_time_offset", data=np.repeat(centre, 200)
            )
    with h5py.File(source, "r") as handle:
        fitted_energy, fitted_t0 = _monitor_ei_t0(handle["entry"], energy)
    assert fitted_energy == pytest.approx(energy, rel=2e-3)
    assert fitted_t0 == pytest.approx(t0, abs=2.0)


def test_monitor_fit_uses_idf_order_and_log_driven_positions(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "HYSPEC_1.nxs.h5"
    energy, t0, msd = 12.0, 5.0, 6000.0
    # The locations intentionally have no names, like some Mantid IDFs. Their
    # z positions are resolved from the supplied run log.
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-15"/></component>
      <component type="monitor"><location><parameter name="z"><logfile id="msd" eq="-0.001*value-3.0"/></parameter></location>
      <location><parameter name="z"><logfile id="msd" eq="-0.001*value+2.0"/></parameter></location></component>
    </instrument>"""
    distances = (6.0, 11.0)
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("instrument").create_group("instrument_xml").create_dataset(
            "data",
            data=np.frombuffer(xml.encode(), dtype="u1"),
        )
        entry.create_group("DASlogs").create_group("msd").create_dataset(
            "average_value", data=[msd]
        )
        for name, distance in zip(
            ("monitor1", "monitor2"), distances, strict=True
        ):
            centre = t0 + TOF_US_PER_M_SQRT_MEV * distance / np.sqrt(energy)
            monitor = entry.create_group(name)
            monitor.attrs["NX_class"] = "NXmonitor"
            monitor.create_dataset("event_time_offset", data=np.repeat(centre, 200))
    with h5py.File(source, "r") as handle:
        fitted_energy, fitted_t0 = _monitor_ei_t0(handle["entry"], energy)
    assert fitted_energy == pytest.approx(energy, rel=2e-3)
    assert fitted_t0 == pytest.approx(t0, abs=2.0)


def test_mantid_cncs_t0_formula_uses_requested_incident_energy():
    energy = 12.0
    formula = "288.595706061-110.833514059/sqrt(incidentEnergy)+89.6080314589/incidentEnergy-42.6999684563*sqrt(incidentEnergy)+1.89672170078*incidentEnergy"
    expected = (
        288.595706061
        - 110.833514059 / np.sqrt(energy)
        + 89.6080314589 / energy
        - 42.6999684563 * np.sqrt(energy)
        + 1.89672170078 * energy
    )
    assert _evaluate_mantid_t0_formula(formula, energy) == pytest.approx(expected)
