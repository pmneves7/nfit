import numpy as np
import pytest

from nfit import bin_raw_dgs_group, inspect_raw_dgs_run, raw_dgs_dataset_group


def _write_raw_dgs(path):
    h5py = pytest.importorskip("h5py")
    xml = '''<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-10"/></component>
      <component type="panel" idlist="panel"><location/></component>
      <type name="moderator" is="Source"/><type name="pixel" is="detector"/>
      <type name="panel"><component type="pixel"><location x="1" y="0" z="2"/></component></type>
      <idlist idname="panel"><id start="42" end="42"/></idlist>
    </instrument>'''
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("run_number", data=[b"42"])
        instrument = entry.create_group("instrument")
        instrument.create_group("instrument_xml").create_dataset("data", data=np.frombuffer(xml.encode(), dtype="u1"))
        bank = entry.create_group("bank1_events")
        bank.create_dataset("event_id", data=[42, 42])
        # One useful event and one prompt event, which must be discarded.
        bank.create_dataset("event_time_offset", data=[9000.0, 1.0])
        logs = entry.create_group("DASlogs")
        for name, value in (("BL17:Det:TH:BL:Ei", 20.0), ("proton_charge", 2.0), ("omega", 0.0), ("phi", 0.0), ("chi", 0.0)):
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
        group, lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1, 1, 1, 1],
        max_batch_bytes=128,
    )

    # The prompt event is rejected; the remaining unit event is normalized by
    # the run proton charge while retaining Poisson variance.
    assert result.num_events.item() == 1.0
    assert result.signal.item() == pytest.approx(0.5)
    assert result.errors.item() == pytest.approx(0.5)
    assert not result.mask.item()


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
    result = bin_raw_dgs_group(group, lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1, 1, 1, 1])
    assert result.mask.item()
    assert result.num_events.item() == 0.0
