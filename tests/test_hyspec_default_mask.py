"""The HYSPEC tip selection masks observed events and trajectory exposure."""
import json

import numpy as np
import pytest

from nfit import bin_raw_dgs_group, raw_dgs_dataset_group
from nfit.raw_dgs_hyspec import hyspec_default_detector_keep
from nfit.reduction_recipes import set_reduction_settings
from tests.test_raw_dgs import _rewrite_instrument_xml, _write_raw_dgs


def test_default_tip_boundaries_and_other_instruments():
    ids = np.array([0, 7, 8, 119, 120, 127, 128, 135, 136, 247, 248])
    original = ids.copy()
    np.testing.assert_array_equal(hyspec_default_detector_keep(ids, instrument_name="HYSPEC"),
        [False, False, True, True, False, False, False, False, True, True, False])
    for name, enabled in (("SEQUOIA", True), ("HYSPEC", False)):
        assert np.all(hyspec_default_detector_keep(ids, instrument_name=name, enabled=enabled))
    np.testing.assert_array_equal(ids, original)


def test_default_matches_explicit_mask_and_rebuilds_events_and_exposure(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "HYS_42.nxs.h5"
    _write_raw_dgs(path)
    _rewrite_instrument_xml(path, '<instrument xmlns=', '<instrument name="HYSPEC" xmlns=')
    _rewrite_instrument_xml(path, '<location x="1" y="0" z="2"/>',
        '<location x="1" y="0" z="2"/><location x="2" y="0" z="2"/>')
    _rewrite_instrument_xml(path, '<id start="42" end="42"/>', '<id start="7" end="8"/>')
    with h5py.File(path, "r+") as f:
        f["entry/bank1_events/event_id"][:] = [7, 8]
        f["entry/bank1_events/event_time_offset"][:] = [9000., 9000.]
    group = raw_dgs_dataset_group([path])
    set_reduction_settings(group, {"t0_override": 0., "bad_pulse_threshold": 0.,
        "ki_kf_normalization": False, "he3_detector_efficiency_correction": False,
        "hyspec_tof_crop": False, "hyspec_tank_offset_override": 0.})
    options = dict(lower=[-10., -10., -10., -19.], upper=[10., 10., 10., 19.], num_bins=[1]*4)
    default = bin_raw_dgs_group(group, **options)
    assert default.num_events.sum() == 1
    with group.datasets[0]._raw_dgs_reduction_cache.open() as archive:
        np.testing.assert_array_equal(archive['solid'], [0., 1.])
        assert json.loads(str(archive['header_json'].item()))['hyspec_preprocessing']['default_detector_mask_enabled']
    set_reduction_settings(group, {"hyspec_default_mask": False})
    unmasked = bin_raw_dgs_group(group, **options)
    assert unmasked.num_events.sum() == 2
    assert unmasked.metadata['reduced_event_cache']['misses'] == 1
    assert unmasked.metadata['normalization_denominator'].item() > default.metadata['normalization_denominator'].item()
    mask = tmp_path / 'mask.nxs'
    with h5py.File(mask, 'w') as f:
        f.create_dataset('mantid_workspace_1/instrument/detector/detector_list', data=[7,8])
        f.create_dataset('mantid_workspace_1/workspace/values', data=[[0.], [1.]])
        f.create_dataset('mantid_workspace_1/workspace/errors', data=[[0.], [0.]])
    set_reduction_settings(group, {"mask_file": str(mask)})
    explicit = bin_raw_dgs_group(group, **options)
    np.testing.assert_array_equal(explicit.signal, default.signal)
    np.testing.assert_array_equal(explicit.errors, default.errors)
    np.testing.assert_array_equal(explicit.num_events, default.num_events)
    np.testing.assert_array_equal(explicit.metadata['normalization_denominator'], default.metadata['normalization_denominator'])
    set_reduction_settings(group, {"hyspec_default_mask": True})
    combined = bin_raw_dgs_group(group, **options)
    np.testing.assert_array_equal(combined.signal, explicit.signal)
