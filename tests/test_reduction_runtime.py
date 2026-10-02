"""Edited per-run recipes execute and preserve unrelated cached reductions."""

import copy

import numpy as np
import pytest

from nfit import bin_raw_dgs_group, raw_dgs_dataset_group, set_reduction_settings
from nfit.reduction_runtime import effective_trajectory_energies
from tests.test_raw_dgs import _rewrite_instrument_xml, _write_raw_dgs
from tests.test_raw_dgs_cache import OPTIONS, assert_equal


def test_edit_one_run_preserves_other_reduction_and_ub_reuses_both(tmp_path):
    sources = [tmp_path / f'SEQ_{index}.nxs.h5' for index in (42, 43)]
    for source in sources:
        _write_raw_dgs(source)
    group = raw_dgs_dataset_group(sources)
    bin_raw_dgs_group(group, **OPTIONS)
    unchanged = group.datasets[1]._raw_dgs_reduction_cache
    set_reduction_settings(group, {'t0_override': -15.0}, dataset_ids=[group.datasets[0].id])
    assert group.datasets[1]._raw_dgs_reduction_cache is unchanged
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert result.metadata['reduced_event_cache'] == {'hits': 1, 'misses': 1}
    assert group.datasets[0].metadata['resolved_reduction']['values']['t0_microseconds'] == -15
    reference = copy.deepcopy(group)
    set_reduction_settings(reference, {'cache_reduced_events': False})
    assert_equal(result, bin_raw_dgs_group(reference, **OPTIONS))
    set_reduction_settings(group, {'ub_matrix': (np.eye(3)*.5).tolist()})
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert result.metadata['reduced_event_cache'] == {'hits': 2, 'misses': 0}


def test_per_run_incident_energy_and_automatic_override(tmp_path):
    sources = [tmp_path / f'SEQ_{index}.nxs.h5' for index in (42, 43)]
    for source in sources:
        _write_raw_dgs(source)
    group = raw_dgs_dataset_group(sources)
    set_reduction_settings(group, {'incident_energy_override': 60.0})
    set_reduction_settings(group, {'incident_energy_override': None}, dataset_ids=[group.datasets[1].id])
    assert effective_trajectory_energies(group, group.datasets, [10., 20.]) == (60., 60.)
    set_reduction_settings(group, {'trajectory_energy_policy': 'per_run'})
    assert effective_trajectory_energies(group, group.datasets, [10., 20.]) == (60., 20.)


@pytest.mark.parametrize('powder', [False, True])
def test_mdevent_calibration_and_mask_override_one_experiment(tmp_path, powder):
    from nfit import bin_mdevent_group, bin_mdevent_powder_group, mdevent_dataset_group
    from tests.test_mdevent import _write_mdevent, _write_normalization

    source = tmp_path / 'events.nxs'
    mask = tmp_path / 'mask.nxs'
    norm = tmp_path / 'van.nxs'
    _write_mdevent(source)
    _write_normalization(mask, 0.)
    _write_normalization(norm, 2.)
    group = mdevent_dataset_group(source)
    run = group.datasets[1]
    reducer = bin_mdevent_powder_group if powder else bin_mdevent_group
    bounds = {'lower': [0., -1.], 'upper': [20., 1.], 'num_bins': [1, 1]} if powder else {
        'lower': [-2.]*3+[-1.], 'upper': [2.]*3+[1.], 'num_bins': [1]*4}
    set_reduction_settings(group, {'normalization_file': str(norm)}, dataset_ids=[run.id])
    calibrated = reducer(group, **bounds)
    assert calibrated.num_events.sum() == 2
    set_reduction_settings(group, {'mask_file': str(mask)}, dataset_ids=[run.id])
    masked = reducer(group, **bounds)
    only_first = reducer(group, datasets=[group.datasets[0]], **bounds)
    np.testing.assert_allclose(masked.signal, only_first.signal)
    np.testing.assert_allclose(masked.errors, only_first.errors)
    np.testing.assert_allclose(masked.num_events, only_first.num_events)


def test_corelli_mixed_geometry_matches_individual_runs(tmp_path, monkeypatch):
    from nfit import corelli, corelli_dataset_group
    from tests.test_corelli import _write_corelli

    sources = [tmp_path / f'CORELLI_{index}.nxs.h5' for index in (7, 8)]
    for source in sources:
        _write_corelli(source)
    _rewrite_instrument_xml(sources[1], 'x="1" y="0" z="2"', 'x="2" y="0" z="2"')
    monkeypatch.setattr(corelli, '_CORELLI_NUMBA', None)
    group = corelli_dataset_group(sources)
    options = dict(lower=[0., -1.], upper=[10., 1.], num_bins=[5, 1], coordinate_mode='powder')
    together = corelli.bin_corelli_group(group, **options)
    individually = [corelli.bin_corelli_group(corelli_dataset_group([source]), **options) for source in sources]
    np.testing.assert_allclose(together.signal, (individually[0].signal+individually[1].signal)/2)
    # Empty CORELLI cells carry a legacy confidence-bound placeholder rather
    # than observed variance. Only measured hypotheses enter this comparison.
    observed = sum(np.where(item.num_events > 0, item.errors**2, 0.) for item in individually)/4
    keep = together.num_events > 0
    np.testing.assert_allclose(together.errors[keep]**2, observed[keep])
    assert group.datasets[0].metadata['geometry_signature'] != group.datasets[1].metadata['geometry_signature']
