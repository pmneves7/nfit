"""Saved detector masks constrain MDEvent exposure without a Mantid runtime."""

import ast
import inspect

import h5py
import numpy as np
import pytest

from nfit import (
    bin_mdevent_group,
    bin_mdevent_powder_group,
    mdevent_dataset_group,
    mdevent_detector_masks,
)
from nfit.mdevent import _read_run, _trajectory_payloads
from nfit.mdevent_detector_masks import apply_saved_detector_mask, saved_detector_mask_ids
from tests.test_mdevent import _write_mdevent


def _parameter_map(experiment, text):
    return experiment.create_dataset("instrument/instrument_parameter_map/data", data=[text.encode()])


def test_real_serialization_golden_masks_only_detector_boolean_flags(tmp_path):
    # Same record layout as SaveMD's HYSPEC reference instrument parameter map.
    text = ("HYSPEC/Tank;Quat;rot;[0.819203,0,-0.573504,0];visible:true|"
            "HYSPEC/Tank;double;tube_pressure;10;visible:true|"
            "detID:4350;bool;masked;1;visible:true|"
            "detID:12925;bool;masked;1;visible:true|"
            "detID:7551;bool;masked;0;visible:true|")
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        _parameter_map(handle, text)
        ids = saved_detector_mask_ids(handle)
    np.testing.assert_array_equal(ids, [4350, 12925])
    assert ids.dtype == np.int64 and not ids.flags.writeable


def test_repeated_mask_records_use_final_boolean_and_ignore_unrelated_records(tmp_path):
    text = ("detID:10;bool;masked;1|detID:10;bool;masked;false|"
            "detID:11;bool;masked;0|detID:11;bool;masked;true|"
            "detID:-2;bool;masked;1|detID:12;double;tube_pressure;10|"
            "HYSPEC;string;label;masked|")
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        _parameter_map(handle, text)
        np.testing.assert_array_equal(saved_detector_mask_ids(handle), [-2, 11])


@pytest.mark.parametrize("record", ["detID:x;bool;masked;1", "detID:1.0;bool;masked;1",
    "detID:9223372036854775808;bool;masked;1", "detID:1;bool;masked;2",
    "detID:1;bool;masked", "detID:1;double;masked;1", "Tank;bool;masked;1"])
def test_malformed_detector_masks_fail_explicitly(tmp_path, record):
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        _parameter_map(handle, record)
        with pytest.raises(ValueError, match="mask"):
            saved_detector_mask_ids(handle)


def test_missing_parameter_map_is_unmasked_and_uint8_text_is_supported(tmp_path):
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        assert saved_detector_mask_ids(handle).size == 0
        handle.create_dataset("instrument/instrument_parameter_map/data",
            data=np.frombuffer(b"detID:10;bool;masked;1|", dtype="u1"))
        np.testing.assert_array_equal(saved_detector_mask_ids(handle), [10])


def test_mask_application_uses_detector_identity_and_preserves_calibration(tmp_path):
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        _parameter_map(handle, "detID:10;bool;masked;1|detID:11;bool;masked;0|")
        weights = np.array([2., 3., 4.])
        weights.setflags(write=False)
        result = apply_saved_detector_mask(handle, [12, 10, 11], weights)
        np.testing.assert_array_equal(result, [2., 0., 4.])
        np.testing.assert_array_equal(weights, [2., 3., 4.])
        with pytest.raises(ValueError, match="equal shapes"):
            apply_saved_detector_mask(handle, [10], weights)


def test_saved_mask_changes_geometry_identity_and_all_trajectory_consumers(tmp_path):
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    with h5py.File(path, "r+") as handle:
        workspace = handle["MDEventWorkspace"]
        before = _read_run(workspace["experiment0"], 0).geometry_signature
        _parameter_map(workspace["experiment0"], "detID:10;bool;masked;1|")
        after = _read_run(workspace["experiment0"], 0).geometry_signature
        assert before != after
    group = mdevent_dataset_group(path)
    detectors, runs = _trajectory_payloads(group, list(group.datasets), np.eye(4))
    assert len(detectors) == 2
    np.testing.assert_array_equal(detectors[0][3], [0.])
    np.testing.assert_array_equal(detectors[1][3], [1.])
    assert [run[4] for run in runs] == [0, 1]
    histogram = bin_mdevent_group(group, lower=[-1]*4, upper=[1]*4, num_bins=[1]*4)
    powder = bin_mdevent_powder_group(group, lower=[0, -1], upper=[5, 1], num_bins=[1, 1])
    np.testing.assert_allclose(histogram.metadata["normalization_denominator"], 4.)
    np.testing.assert_allclose(powder.metadata["normalization_denominator"], 4.)


def test_saved_mask_service_is_gui_independent_and_does_not_import_facade():
    tree = ast.parse(inspect.getsource(mdevent_detector_masks))
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module in ("mdevent", "raw_dgs") or "gui" in module or "Qt" in module for module in modules)


def test_exact_text_cache_reuses_parse_but_source_edits_and_instrument_differences_miss(tmp_path, monkeypatch):
    mdevent_detector_masks._cached_mask_ids.cache_clear()
    original = mdevent_detector_masks._parse_mask_ids
    parsed = []
    def record(text):
        parsed.append(text)
        return original(text)
    monkeypatch.setattr(mdevent_detector_masks, "_parse_mask_ids", record)
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        text = "HYSPEC;string;name;HYSPEC|detID:10;bool;masked;1|"
        source = _parameter_map(handle, text)
        np.testing.assert_array_equal(saved_detector_mask_ids(handle), [10])
        np.testing.assert_array_equal(saved_detector_mask_ids(handle), [10])
        assert len(parsed) == 1
        source[...] = [text.replace("masked;1", "masked;0").encode()]
        assert saved_detector_mask_ids(handle).size == 0
        assert len(parsed) == 2
        source[...] = [text.replace("HYSPEC", "ARCS__").encode()]
        np.testing.assert_array_equal(saved_detector_mask_ids(handle), [10])
        assert len(parsed) == 3


def test_cached_parse_value_is_immutable_even_when_caller_resets_array_write_flag(tmp_path):
    mdevent_detector_masks._cached_mask_ids.cache_clear()
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        text = "detID:10;bool;masked;1|detID:10;bool;masked;0|detID:11;bool;masked;1|"
        _parameter_map(handle, text)
        first = saved_detector_mask_ids(handle)
        assert isinstance(mdevent_detector_masks._cached_mask_ids(text), tuple)
        first.setflags(write=True)
        first[0] = 999
        second = saved_detector_mask_ids(handle)
        np.testing.assert_array_equal(second, [11])
        assert not second.flags.writeable and second is not first


def test_oversized_parameter_maps_skip_cache_without_changing_mask_interpretation(tmp_path, monkeypatch):
    mdevent_detector_masks._cached_mask_ids.cache_clear()
    monkeypatch.setattr(mdevent_detector_masks, "_MASK_CACHE_MAX_TEXT_BYTES", 50)
    text = "HYSPEC;string;label;" + "x"*60 + "|detID:10;bool;masked;1|"
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        _parameter_map(handle, text)
        for _ in range(2):
            np.testing.assert_array_equal(saved_detector_mask_ids(handle), [10])
    assert mdevent_detector_masks._cached_mask_ids.cache_info().currsize == 0


def test_saved_mask_parse_cache_has_bounded_entry_count():
    mdevent_detector_masks._cached_mask_ids.cache_clear()
    for index in range(25):
        mdevent_detector_masks._cached_mask_ids(f"detID:{index};bool;masked;1|")
    assert mdevent_detector_masks._cached_mask_ids.cache_info().currsize == 16
