"""Successful source calibration is reused only for the same file identity."""

import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from nfit import raw_dgs
from tests.test_raw_dgs import _rewrite_instrument_xml, _write_raw_dgs


@pytest.fixture(autouse=True)
def isolated_run_info_cache():
    with raw_dgs._RUN_INFO_CACHE_LOCK:
        raw_dgs._RUN_INFO_CACHE.clear()
    yield
    with raw_dgs._RUN_INFO_CACHE_LOCK:
        raw_dgs._RUN_INFO_CACHE.clear()


@pytest.fixture
def calibration_calls(monkeypatch):
    original = raw_dgs._monitor_ei_t0
    calls = []

    def calibrate(entry, requested_energy, **kwargs):
        calls.append((entry.file.filename, kwargs["variance_policy"]))
        return original(entry, requested_energy, **kwargs)

    monkeypatch.setattr(raw_dgs, "_monitor_ei_t0", calibrate)
    return calls


def test_import_then_reduction_reuses_calibration_without_changing_metadata(tmp_path, calibration_calls):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs.raw_dgs_dataset_group([source])
    before = raw_dgs.inspect_raw_dgs_run(source)
    result = raw_dgs.bin_raw_dgs_group(
        group, lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[2] * 4,
    )
    after = raw_dgs.inspect_raw_dgs_run(source)
    assert len(calibration_calls) == 1
    assert result.num_events.sum() == 1
    for field in fields(before):
        a, b = getattr(before, field.name), getattr(after, field.name)
        if isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes()
        else:
            assert a == b
    assert before.ub_matrix is not after.ub_matrix


def test_alias_paths_keep_callers_path_and_private_writable_ub(tmp_path, monkeypatch, calibration_calls):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    alias = tmp_path / "alias.nxs.h5"
    alias.symlink_to(source)
    monkeypatch.chdir(tmp_path)
    relative = Path(source.name)
    first = raw_dgs.inspect_raw_dgs_run(relative)
    first.ub_matrix[:] = 37.
    second = raw_dgs.inspect_raw_dgs_run(source)
    third = raw_dgs.inspect_raw_dgs_run(alias)
    assert first.path == relative
    assert second.path == source
    assert third.path == alias
    assert len(calibration_calls) == 1
    assert first.ub_matrix.flags.writeable
    assert second.ub_matrix.flags.writeable
    assert third.ub_matrix.flags.writeable
    np.testing.assert_array_equal(second.ub_matrix, np.eye(3))
    np.testing.assert_array_equal(third.ub_matrix, np.eye(3))
    assert not np.shares_memory(second.ub_matrix, third.ub_matrix)
    retained, _ = next(iter(raw_dgs._RUN_INFO_CACHE.values()))
    assert not retained.ub_matrix.flags.writeable
    assert not np.shares_memory(retained.ub_matrix, second.ub_matrix)


def test_source_changes_with_preserved_mtime_recompute_energy_and_geometry(tmp_path, calibration_calls):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    before = raw_dgs.inspect_raw_dgs_run(source)
    stat = source.stat()
    with h5py.File(source, "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][0] = 40.
        handle["entry/DASlogs/omega/average_value"][0] = 15.
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source.stat().st_mtime_ns == stat.st_mtime_ns
    assert source.stat().st_ctime_ns != stat.st_ctime_ns
    changed = raw_dgs.inspect_raw_dgs_run(source)
    assert before.incident_energy == 20.
    assert changed.incident_energy == 40.
    assert changed.omega == 15.
    _rewrite_instrument_xml(source, 'x="1" y="0" z="2"', 'x="2" y="0" z="2"')
    changed_geometry = raw_dgs.inspect_raw_dgs_run(source)
    assert changed_geometry.geometry_signature != changed.geometry_signature
    assert len(calibration_calls) == 3


def test_replaced_file_with_same_size_and_mtime_has_new_identity(tmp_path, calibration_calls):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "SEQ_42.nxs.h5"
    replacement = tmp_path / "replacement.nxs.h5"
    _write_raw_dgs(source)
    before = raw_dgs.inspect_raw_dgs_run(source)
    old_stat = source.stat()
    _write_raw_dgs(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][0] = 30.
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert replacement.stat().st_size == old_stat.st_size
    os.replace(replacement, source)
    assert source.stat().st_ino != old_stat.st_ino
    assert source.stat().st_mtime_ns == old_stat.st_mtime_ns
    after = raw_dgs.inspect_raw_dgs_run(source)
    assert before.incident_energy == 20.
    assert after.incident_energy == 30.
    assert len(calibration_calls) == 2


def test_variance_policies_have_independent_cache_entries(tmp_path, calibration_calls):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    raw_dgs.inspect_raw_dgs_run(source)
    raw_dgs.inspect_raw_dgs_run(source, monitor_variance_policy="stable")
    raw_dgs.inspect_raw_dgs_run(source)
    raw_dgs.inspect_raw_dgs_run(source, monitor_variance_policy="stable")
    assert [policy for _, policy in calibration_calls] == ["mantid", "stable"]
    assert len(raw_dgs._RUN_INFO_CACHE) == 2
    with pytest.raises(ValueError, match="monitor_variance_policy"):
        raw_dgs.inspect_raw_dgs_run(source, monitor_variance_policy="invalid")


def test_missing_source_cannot_be_served_from_cached_inspection(tmp_path, calibration_calls):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    raw_dgs.inspect_raw_dgs_run(source)
    source.unlink()
    with pytest.raises(FileNotFoundError):
        raw_dgs.inspect_raw_dgs_run(source)
    assert len(calibration_calls) == 1


def test_failed_calibration_retries_and_warns_on_every_call(tmp_path, calibration_calls):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    _rewrite_instrument_xml(source, '<component type="moderator">',
        '<parameter name="t0_formula"><value val="sin(incidentEnergy)"/></parameter>'
        '<component type="moderator">')
    for _ in range(2):
        with pytest.warns(RuntimeWarning, match="Set explicit Ei/T0 overrides") as warnings:
            info = raw_dgs.inspect_raw_dgs_run(source)
        assert len(warnings) == 1
        assert info.calibration_warning is not None
        assert info.calibration_source == "requested_energy_failed_t0_formula"
        assert info.incident_energy == 20.
        assert info.t0 == 0.
    assert len(calibration_calls) == 2
    assert not raw_dgs._RUN_INFO_CACHE


def test_inspection_cache_is_bounded_and_evicts_least_recent_source(tmp_path, monkeypatch, calibration_calls):
    monkeypatch.setattr(raw_dgs, "_RUN_INFO_CACHE_MAXSIZE", 2)
    paths = [tmp_path / f"SEQ_{run}.nxs.h5" for run in range(3)]
    for source in paths:
        _write_raw_dgs(source)
        raw_dgs.inspect_raw_dgs_run(source)
    assert len(raw_dgs._RUN_INFO_CACHE) == 2
    raw_dgs.inspect_raw_dgs_run(paths[1])
    assert len(calibration_calls) == 3
    raw_dgs.inspect_raw_dgs_run(paths[0])
    assert len(calibration_calls) == 4
    assert len(raw_dgs._RUN_INFO_CACHE) == 2


@pytest.mark.parametrize("stored_run_number", [None, "SEQ_42"])
def test_filename_run_number_fallback_is_rendered_per_caller_alias(tmp_path, calibration_calls, stored_run_number):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    with h5py.File(source, "r+") as handle:
        del handle["entry/run_number"]
        if stored_run_number is not None:
            handle["entry"]["run_number"] = np.array([stored_run_number.encode()])
    alias = tmp_path / "alias.nxs.h5"
    alias.symlink_to(source)
    first = raw_dgs.inspect_raw_dgs_run(source)
    second = raw_dgs.inspect_raw_dgs_run(alias)
    assert first.run_number == (source.stem if stored_run_number is None else stored_run_number)
    assert second.run_number == (alias.stem if stored_run_number is None else stored_run_number)
    assert len(calibration_calls) == 1
