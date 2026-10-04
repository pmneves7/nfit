"""Native-only sample-angle references, including pulse-filtered time means."""

import numpy as np
import pytest

from nfit.raw_dgs_goniometer import sample_rotation, step_log_mean

NS = 10**9
ORIGIN = "2026-01-01T00:00:00.123456789Z"


def _log(entry, name, times, values, *, origin=ORIGIN):
    log = entry.require_group("DASlogs").create_group(name)
    log["value"] = values
    if times is not None:
        time = log.create_dataset("time", data=times)
        time.attrs["start"] = origin
    return log


@pytest.fixture
def entry(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "logs.nxs.h5", "w") as handle:
        yield handle.create_group("entry")


@pytest.mark.parametrize("times,values,intervals,expected", [
    ([0, 2, 10], [0, 10, 20], [(1, 9)], 8.75),
    ([0, 2, 10], [0, 10, 20], [(0, 2), (7, 9)], 5.),
    ([0, 5, 5, 10], [0, 10, 20, 30], [(5, 8)], 20.),
    ([5, 0, 10, 5], [10, 0, 30, 20], [(0, 10)], 10.),
    ([5, 5, 10], [0, 20, 30], [(0, 8)], 7.5),
    ([2, 4], [10, 20], [(0, 6)], 40 / 3),
    ([0, 2, 10], [0, 10, 20], [(0, 2)], 0.),
])
def test_step_means_hold_values_and_respect_gaps_duplicates_and_endpoints(times, values, intervals, expected):
    assert step_log_mean(np.array(times) * NS, values, np.array(intervals) * NS) == pytest.approx(expected)


@pytest.mark.parametrize("intervals", [[], [(2, 2)]])
def test_varying_angles_require_positive_accepted_duration(intervals):
    with pytest.raises(ValueError, match="accepted time"):
        step_log_mean([0, NS], [0, 10], np.array(intervals) * NS)


def test_actual_hyspec_two_record_log_matches_full_precision_filtered_mean(entry):
    _log(entry, "omega", [0., 2.668696542], [48.498771596000005, 48.504482709])
    _log(entry, "proton_charge", [0., 168.886083], [1., 1.])
    _log(entry, "pause", [0.], [0])
    angles = sample_rotation(entry)
    assert angles.omega == 48.504392463390396
    assert angles.phi == angles.chi == 0.
    assert angles.varying_logs


def test_pause_gaps_and_charge_threshold_use_accepted_times(entry):
    _log(entry, "omega", [0., 2., 10.], [0., 10., 20.])
    _log(entry, "proton_charge", [0., 2., 7., 9.], [10., 1., 10., 10.])
    _log(entry, "pause", [0., 2., 7., 9.], [0, 1, 0, 1])
    assert sample_rotation(entry).omega == 5.
    assert sample_rotation(entry, bad_pulse_threshold=95.).omega == 10.


def test_pause_roi_expands_to_charge_bounds_without_clipping_log_support(entry):
    _log(entry, "omega", [0., 5.], [0., 10.])
    _log(entry, "proton_charge", [2., 8.], [1., 1.])
    _log(entry, "pause", [0., 10.], [0, 1])
    assert sample_rotation(entry).omega == 5.


@pytest.mark.parametrize("run_end,expected", [(None, 13.333333333333332), (12., 11.428571428571429)])
def test_unrestricted_log_appends_run_end_then_extends_final_interval(entry, run_end, expected):
    _log(entry, "omega", [0., 2., 10.], [0., 10., 20.])
    if run_end is not None:
        entry["end_time"] = [b"2026-01-01T00:00:12.123456789Z"]
    assert sample_rotation(entry).omega == expected


def test_log_and_pulse_timestamp_origins_are_absolute_and_preserve_nanoseconds(entry):
    angle = _log(entry, "omega", [0., 2.], [0., 10.], origin="2025-12-31T19:00:00.123456789-05:00")
    angle["time"].attrs["units"] = "s"
    charge = _log(entry, "proton_charge", [0., 4.], [1., 1.])
    del charge["time"].attrs["start"]
    charge["time"].attrs["offset_seconds"] = 1136073600  # Since 1990, not Unix.
    charge["time"].attrs["offset_nanoseconds"] = 123456789
    _log(entry, "pause", [0.], [0])
    assert sample_rotation(entry).omega == 5.


def test_timezone_less_nexus_timestamp_is_utc_on_every_host():
    from nfit.raw_dgs_pulses import iso_timestamp_ns

    assert iso_timestamp_ns("2026-01-01T00:00:00.123456789") == 1767225600123456789
    assert iso_timestamp_ns("2026-01-01T00:00:00.123456789Z") == 1767225600123456789


def test_constant_and_legacy_scalar_angles_do_not_read_time_or_pulse_payloads():
    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("constant-angle inspection must not load timestamps or charge")

    angles = sample_rotation({
        "DASlogs/omega": {"value": np.array([17., 17.]), "time": Unreadable()},
        "DASlogs/phi": {"average_value": np.array([2.])},
        "DASlogs/proton_charge": Unreadable(),
    })
    assert (angles.omega, angles.phi, angles.chi, angles.varying_logs) == (17., 2., 0., False)


@pytest.mark.parametrize("times,values,message", [
    (None, [0., 1.], "timestamps"),
    ([0., 0.], [0., 1.], "positive timestamp"),
    ([0., 1.], [0., np.nan], "finite values"),
    ([0.], [0., 1.], "timestamps"),
    ([0., 1.], [], "finite values"),
    ([np.nan, 1.], [0., 1.], "timestamps must be finite"),
])
def test_incomplete_varying_logs_fail_instead_of_using_arithmetic_means(entry, times, values, message):
    _log(entry, "omega", times, values)
    with pytest.raises(ValueError, match=message):
        sample_rotation(entry)


def test_fully_paused_varying_rotation_has_no_accepted_mean(entry):
    _log(entry, "omega", [0., 2.], [0., 10.])
    _log(entry, "proton_charge", [0., 4.], [1., 1.])
    _log(entry, "pause", [0.], [1])
    with pytest.raises(ValueError, match="accepted time"):
        sample_rotation(entry)
