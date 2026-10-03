"""Reduced-event parity for immutable detector geometry precomputation."""

import math

import numpy as np
import pytest

from nfit import raw_dgs


def _synthetic_run(tmp_path, geometry_variant=0):
    h5py = pytest.importorskip("h5py")
    detector_ids = np.array([7, 2, 5, 99])
    positions = np.array([
        [-1., .3, 2.7], [.5, 2.4, 3.4], [2., -.2, 1.3], [.1, 0., 4.4],
    ])
    if geometry_variant:
        positions *= np.array([1.01, .99, 1.02])
    geometry = raw_dgs._DetectorGeometry(
        detector_ids, positions, np.array([.21, .35, .27, .41]),
        np.array([.22, .33, .28, .43]),
    )
    rng = np.random.default_rng(7319)
    detector_rows = rng.integers(0, 4, 64)
    energies = rng.uniform(-65, 59, 64)
    energies[:6] = [-57., 57., -57., 57., -57., 57.]
    detector_rows[:6] = [0, 1, 0, 1, 0, 1]
    tofs = (
        2.5 + raw_dgs.TOF_US_PER_M_SQRT_MEV * 10 / math.sqrt(60)
        + raw_dgs.TOF_US_PER_M_SQRT_MEV
        * np.linalg.norm(positions[detector_rows], axis=1) / np.sqrt(60 - energies)
    )
    tofs[2:4] = np.nextafter(tofs[2:4], np.inf)
    tofs[4:6] = np.nextafter(tofs[4:6], -np.inf)
    tofs[6:10] = [
        0., 2.5 + raw_dgs.TOF_US_PER_M_SQRT_MEV * 10 / math.sqrt(60),
        np.nan, np.inf,
    ]
    event_ids = detector_ids[detector_rows]
    event_ids[10:13] = [3, 100, -1]
    path = tmp_path / f"events-{geometry_variant}.h5"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        bank = entry.create_group("bank1_events")
        bank["event_id"] = event_ids
        bank["event_time_offset"] = tofs
        bank["event_index"] = np.array([0, 32])
        entry.create_group("DASlogs/proton_charge")["value"] = np.array([2., 1.])
    info = raw_dgs.RawDGSRunInfo(path, "synthetic", 64, 60, 1, 0, 0, 0, 10, 2.5, np.eye(3))
    return info, geometry, event_ids, tofs


def _position_based_reference(geometry, event_ids, tofs, precision, he3, ki_kf, crop):
    """Precompute-free reference: gather positions, then derive event geometry.

    Only the geometry path is under comparison. Unchanged efficiency physics
    uses its authoritative correction helper rather than a second formula.
    """
    row_by_id = {int(detector_id): row for row, detector_id in enumerate(geometry.detector_ids)}
    selected = np.array([
        index < 32 and int(detector_id) in row_by_id
        and (not crop or 6900 <= tofs[index] <= 13000)
        for index, detector_id in enumerate(event_ids)
    ])
    rows = np.array([row_by_id[int(detector_id)] for detector_id in event_ids[selected]], dtype=int)
    positions = geometry.positions[rows]
    exponents = (
        geometry.mantid_he3_exponents if precision == "mantid" and he3 else geometry.he3_exponents
    )[rows]
    distances = np.linalg.norm(positions, axis=1)
    final_tof = tofs[selected] - 2.5 - raw_dgs.TOF_US_PER_M_SQRT_MEV * 10 / math.sqrt(60)
    good = final_tof > 0
    positions, exponents = positions[good], exponents[good]
    distances, final_tof = distances[good], final_tof[good]
    final_energy = (raw_dgs.TOF_US_PER_M_SQRT_MEV * distances / final_tof) ** 2
    energy = 60 - final_energy
    kf_energy = 60 - energy if precision == "mantid" else final_energy
    kf = np.sqrt(np.maximum(kf_energy, 0.) / raw_dgs.ENERGY_TO_K2)
    keep = (energy >= -57) & (energy <= 57)
    directions = positions[keep] / distances[keep, None]
    kf, energy, exponents = kf[keep], energy[keep], exponents[keep]
    q_lab = np.column_stack((
        -kf * directions[:, 0], -kf * directions[:, 1],
        math.sqrt(60 / raw_dgs.ENERGY_TO_K2) - kf * directions[:, 2],
    ))
    mantid = precision == "mantid"
    weights = np.ones(energy.size, dtype=np.float32 if mantid else float)
    variance = np.ones_like(weights)
    if he3:
        correction = raw_dgs._he3_tube_efficiency_correction(kf, exponents, mantid_precision=mantid)
        if mantid:
            correction = correction.astype(np.float32)
        weights *= correction
        variance *= correction * correction
    if ki_kf:
        correction = np.sqrt(60 / (60 - energy))
        if mantid:
            correction = correction.astype(np.float32)
        weights *= correction
        variance *= correction * correction
    return np.column_stack((q_lab, energy, weights, variance))


@pytest.mark.parametrize("geometry_variant", [0, 1])
@pytest.mark.parametrize("precision", ["mantid", "high_precision"])
@pytest.mark.parametrize("he3", [False, True])
@pytest.mark.parametrize("ki_kf", [False, True])
@pytest.mark.parametrize("crop", [False, True])
def test_reduced_event_bytes_match_position_based_geometry(
    tmp_path, geometry_variant, precision, he3, ki_kf, crop,
):
    info, geometry, event_ids, tofs = _synthetic_run(tmp_path, geometry_variant)
    config = {
        "event_precision_policy": precision,
        "he3_detector_efficiency_correction": he3,
        "ki_kf_normalization": ki_kf,
        "bad_pulse_threshold": 95,
    }
    hyspec = {
        "tof_crop_enabled": True, "raw_tof_bounds_microseconds": [6900., 13000.],
    } if crop else None
    with np.errstate(invalid="ignore"):
        chunks = list(raw_dgs._iter_reduced_event_chunks(
            info, config, geometry, 60, (-57, 57), 96 * 7, hyspec_preprocessing=hyspec,
        ))
        expected = _position_based_reference(geometry, event_ids, tofs, precision, he3, ki_kf, crop)
    actual = np.concatenate([values for values, _ in chunks])
    assert actual.tobytes() == expected.tobytes()
    assert [count for _, count in chunks] == [7] * 9 + [1]


def test_geometry_distances_and_directions_are_readonly_snapshot_values(tmp_path):
    _, geometry, event_ids, _ = _synthetic_run(tmp_path)
    _, changed, _, _ = _synthetic_run(tmp_path, 1)
    selected = np.array([True, False, True, False])
    masked = raw_dgs._DetectorGeometry(
        geometry.detector_ids[selected], geometry.positions[selected],
        geometry.he3_exponents[selected], geometry.mantid_he3_exponents[selected],
    )
    for snapshot in (geometry, changed, masked):
        assert snapshot.distances is snapshot.distances
        assert snapshot.directions is snapshot.directions
        assert not snapshot.distances.flags.writeable
        assert not snapshot.directions.flags.writeable
        indices, exponents, valid = snapshot.event_indices_for_ids(event_ids, mantid_precision=True)
        positions, old_exponents, old_valid = snapshot.event_geometry_for_ids(event_ids, mantid_precision=True)
        assert old_valid.tobytes() == valid.tobytes()
        assert old_exponents.tobytes() == exponents.tobytes()
        assert positions[valid].tobytes() == snapshot.positions[indices[valid]].tobytes()
        distances = np.linalg.norm(positions[valid], axis=1)
        assert snapshot.distances[indices[valid]].tobytes() == distances.tobytes()
        assert snapshot.directions[indices[valid]].tobytes() == (positions[valid] / distances[:, None]).tobytes()
    assert not np.array_equal(geometry.distances, changed.distances)
    assert masked.distances.tobytes() == geometry.distances[selected].tobytes()
    assert masked.directions.tobytes() == geometry.directions[selected].tobytes()


def test_unsupported_shape_is_rejected_before_pulse_and_crop_filtering(tmp_path):
    info, geometry, _, _ = _synthetic_run(tmp_path)
    h5py = pytest.importorskip("h5py")
    with h5py.File(info.path, "r+") as handle:
        event_ids = handle["entry/bank1_events/event_id"][:]
        event_ids[event_ids == 5] = 7
        event_ids[40] = 5  # The only unsupported detector is on the rejected pulse.
        handle["entry/bank1_events/event_id"][:] = event_ids
    exponents = np.array(geometry.mantid_he3_exponents)
    exponents[2] = np.nan
    geometry = raw_dgs._DetectorGeometry(
        geometry.detector_ids, geometry.positions, geometry.he3_exponents, exponents,
    )
    crop = {"tof_crop_enabled": True, "raw_tof_bounds_microseconds": [0., 0.]}
    with pytest.raises(ValueError, match="detector cylinder shape"):
        list(raw_dgs._iter_reduced_event_chunks(
            info, {"bad_pulse_threshold": 95}, geometry, 60, (-57, 57), 96 * 64,
            hyspec_preprocessing=crop,
        ))
    for config in ({"he3_detector_efficiency_correction": False}, {"event_precision_policy": "high_precision"}):
        with np.errstate(invalid="ignore"):
            list(raw_dgs._iter_reduced_event_chunks(info, config, geometry, 60, (-57, 57), 96 * 64))


def test_empty_geometry_discards_events_without_invalid_row_access(tmp_path):
    info, _, _, _ = _synthetic_run(tmp_path)
    geometry = raw_dgs._DetectorGeometry(np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))
    chunks = list(raw_dgs._iter_reduced_event_chunks(info, {}, geometry, 60, (-57, 57), 96 * 7))
    assert sum(count for _, count in chunks) == 64
    assert all(values.shape == (0, 6) for values, _ in chunks)


@pytest.mark.parametrize("unused_position", [[0., 0., 0.], [np.nan, 0., 1.], [np.inf, 0., 1.], [1e308, 0., 1.], [1e-310, 0., 3.]])
def test_unused_exceptional_pixels_preserve_selected_event_error_behavior(tmp_path, unused_position):
    info, geometry, event_ids, tofs = _synthetic_run(tmp_path)
    geometry = raw_dgs._DetectorGeometry(
        np.append(geometry.detector_ids, 1000), np.vstack([geometry.positions, unused_position]),
        np.append(geometry.he3_exponents, .2), np.append(geometry.mantid_he3_exponents, .2),
    )
    with np.errstate(all="raise"):
        chunks = list(raw_dgs._iter_reduced_event_chunks(info, {}, geometry, 60, (-57, 57), 96 * 7))
        expected = _position_based_reference(geometry, event_ids, tofs, "mantid", True, True, False)
    assert np.concatenate([values for values, _ in chunks]).tobytes() == expected.tobytes()
    assert geometry.has_exceptional_positions


@pytest.mark.parametrize("position, bounds, failure", [
    ([0., 0., 0.], (-57, 60), "invalid value"),
    ([1e308, 0., 1.], (-57, 57), "overflow"),
    ([1e-310, 0., 3.], (-57, 57), "underflow"),
])
def test_used_exceptional_pixels_keep_numpy_error_handling(tmp_path, position, bounds, failure):
    info, geometry, _, _ = _synthetic_run(tmp_path)
    positions = np.array(geometry.positions)
    positions[0] = position
    geometry = raw_dgs._DetectorGeometry(
        geometry.detector_ids, positions, geometry.he3_exponents, geometry.mantid_he3_exponents,
    )
    with np.errstate(all="raise"), pytest.raises(FloatingPointError, match=failure):
        list(raw_dgs._iter_reduced_event_chunks(info, {}, geometry, 60, bounds, 96 * 64))
