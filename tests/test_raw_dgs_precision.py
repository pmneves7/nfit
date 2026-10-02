"""Six-column reduced events preserve independent weight/variance rounding."""

import copy

import numpy as np
import pytest

from nfit import (
    DataGroup,
    NfitProject,
    bin_raw_dgs_group,
    load_project,
    raw_dgs,
    raw_dgs_dataset_group,
    save_project,
)
from nfit.raw_dgs_cache import iter_cached_event_chunks
from tests.test_raw_dgs import _write_raw_dgs

OPTIONS = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[2]*4)


@pytest.mark.parametrize("policy", ["mantid", "high_precision"])
def test_raw_corrections_store_weight_and_variance_separately(tmp_path, monkeypatch, policy):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source], event_precision_policy=policy)
    config = group.metadata["raw_dgs"]
    info = raw_dgs.inspect_raw_dgs_run(source)
    geometry = raw_dgs._detector_geometry(source)
    efficiency = 1.23456789
    monkeypatch.setattr(raw_dgs, "_he3_tube_efficiency_correction", lambda kf, _exp, **kwargs: np.full_like(kf, efficiency))
    chunks = list(raw_dgs._iter_reduced_event_chunks(info, config, geometry,
        info.incident_energy, (-19., 19.), 192))
    rows = np.concatenate([events for events, _ in chunks])
    assert rows.shape == (1, 6)
    assert rows.dtype == np.float64
    assert sum(raw_count for _, raw_count in chunks) == 2
    # Independently reconstruct final flight energy from the known synthetic
    # detector distance, source distance and 9000-us event; prompt event excluded.
    distance = np.sqrt(5.)
    final_time = 9000. - raw_dgs.TOF_US_PER_M_SQRT_MEV * 10. / np.sqrt(20.)
    ef = (raw_dgs.TOF_US_PER_M_SQRT_MEV * distance / final_time)**2
    correction = np.sqrt(20. / ef)
    assert rows[0, 3] == pytest.approx(20. - ef)
    if policy == "mantid":
        detector_factor, kinematic_factor = np.float32(efficiency), np.float32(correction)
        expected_weight = detector_factor * kinematic_factor
        expected_variance = (detector_factor * detector_factor) * (kinematic_factor * kinematic_factor)
        assert rows[0, 4] == float(expected_weight)
        assert rows[0, 5] == float(expected_variance)
        assert rows[0, 5] != rows[0, 4]**2
        np.testing.assert_array_equal(rows[:, 4:], rows[:, 4:].astype(np.float32).astype(np.float64))
    else:
        expected_weight = efficiency * correction
        expected_variance = efficiency**2 * correction**2
        assert rows[0, 4] == pytest.approx(expected_weight, rel=1e-15)
        assert rows[0, 5] == pytest.approx(expected_variance, rel=1e-15)
        assert rows[0, 5] == pytest.approx(rows[0, 4]**2, rel=1e-15)
        assert rows[0, 4] != float(np.float32(rows[0, 4]))


@pytest.mark.parametrize("policy", ["mantid", "high_precision"])
def test_six_column_cache_matches_fresh_and_remains_lazy_in_project(tmp_path, monkeypatch, policy):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source, with_he3=True)
    group = raw_dgs_dataset_group([source], event_precision_policy=policy)
    fresh_group = copy.deepcopy(group)
    fresh_group.metadata["raw_dgs"]["cache_reduced_events"] = False
    fresh = bin_raw_dgs_group(fresh_group, **OPTIONS)
    cached = bin_raw_dgs_group(group, **OPTIONS)
    cache = group.datasets[0]._raw_dgs_reduction_cache
    with cache.open() as archive:
        rows = np.concatenate([events for events, _ in iter_cached_event_chunks(archive, 48)])
    assert rows.shape == (1, 6)
    signal = cached.auxiliary_channels["event_signal_numerator"].values
    variance = cached.auxiliary_channels["event_variance_numerator"].values
    assert signal.sum() == rows[:, 4].sum()
    assert variance.sum() == rows[:, 5].sum()
    if policy == "mantid":
        assert variance.sum() != np.square(rows[:, 4]).sum()
    project = NfitProject([DataGroup("Data", subgroups=[group])])
    destination = tmp_path / "cached.nfit"
    save_project(project, destination)

    def must_remain_lazy(*args, **kwargs):
        pytest.fail("opening the project read a numerical event payload")

    with monkeypatch.context() as context:
        context.setattr(np, "load", must_remain_lazy)
        reopened = load_project(destination)
        save_project(reopened, tmp_path / "lazy-copy.nfit")
    restored = reopened.data_groups[0].subgroups[0]
    with restored.datasets[0]._raw_dgs_reduction_cache.open() as archive:
        decoded = np.concatenate([events for events, _ in iter_cached_event_chunks(archive, 48)])
    np.testing.assert_array_equal(decoded.view(np.uint64), rows.view(np.uint64))
    source.unlink()
    monkeypatch.setattr(raw_dgs, "inspect_raw_dgs_run", must_remain_lazy)
    result = bin_raw_dgs_group(restored, max_batch_bytes=48, **OPTIONS)
    assert result.metadata["reduced_event_cache"] == {"hits": 1, "misses": 0}
    for field in ("signal", "errors", "mask", "num_events"):
        np.testing.assert_allclose(getattr(result, field), getattr(fresh, field), equal_nan=True)
    for key in ("event_signal_numerator", "event_variance_numerator", "normalization_denominator"):
        np.testing.assert_array_equal(result.auxiliary_channels[key].values,
            cached.auxiliary_channels[key].values)
