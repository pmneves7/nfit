"""Raw reductions are reusable per run and remain lazy in project archives."""

import copy
import zipfile

import numpy as np
import pytest

from nfit import (
    NfitProject,
    bin_raw_dgs_group,
    bin_raw_dgs_powder_group,
    clear_reduced_event_cache,
    load_project,
    raw_dgs,
    raw_dgs_dataset_group,
    reduced_event_cache_info,
    save_project,
)
from nfit.pipeline import DataGroup
from nfit.raw_dgs_cache import iter_cached_event_chunks
from tests.test_raw_dgs import _rewrite_instrument_xml, _write_raw_dgs

OPTIONS = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[2]*4)


def assert_equal(left, right):
    for name in ('signal', 'errors', 'mask', 'num_events'):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        left.metadata['normalization_denominator'], right.metadata['normalization_denominator'],
        rtol=1e-13, atol=1e-13,
    )


def test_cache_retains_events_outside_first_grid_and_reuses_after_ub_symmetry_changes(tmp_path, monkeypatch):
    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source, with_he3=True)
    group = raw_dgs_dataset_group([source])
    first = bin_raw_dgs_group(group, **{**OPTIONS, 'lower': [99, 99, 99, -100], 'upper': [100, 100, 100, 20]})
    assert first.num_events.sum() == 0
    group.metadata['raw_dgs']['ub_matrix'] = (np.eye(3)*.5).tolist()
    symmetry = [np.eye(3), -np.eye(3)]
    reference_group = copy.deepcopy(group)
    reference_group.metadata['raw_dgs']['cache_reduced_events'] = False
    reference = bin_raw_dgs_group(reference_group, symmetry_operations=symmetry, **OPTIONS)

    def fail(*args, **kwargs):
        pytest.fail('cached binning read raw/calibration data')

    monkeypatch.setattr(raw_dgs, 'inspect_raw_dgs_run', fail)
    monkeypatch.setattr(raw_dgs, '_detector_geometry', fail)
    monkeypatch.setattr(raw_dgs, 'load_detector_normalization', fail)
    result = bin_raw_dgs_group(group, symmetry_operations=symmetry, max_batch_bytes=40, **OPTIONS)
    assert_equal(result, reference)
    assert result.num_events.sum() == 2
    assert result.metadata['reduced_event_cache'] == {'hits': 1, 'misses': 0}
    assert reduced_event_cache_info(group.datasets[0]) is not None


def test_cache_is_per_run_and_detects_source_and_reduction_changes(tmp_path, monkeypatch):
    sources = [tmp_path / f'SEQ_{run}.nxs.h5' for run in (42, 43, 44)]
    for source in sources:
        _write_raw_dgs(source)
    group = raw_dgs_dataset_group(sources[:2])
    bin_raw_dgs_group(group, **OPTIONS)
    original = raw_dgs.inspect_raw_dgs_run
    calls = []

    def inspect(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(raw_dgs, 'inspect_raw_dgs_run', inspect)
    new = raw_dgs_dataset_group([sources[2]]).datasets[0]
    calls.clear()
    group.datasets.append(new)
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert calls == [sources[2]]
    assert result.metadata['reduced_event_cache'] == {'hits': 2, 'misses': 1}
    _rewrite_instrument_xml(sources[0], 'x="1" y="0" z="2"', 'x="2" y="0" z="2"')
    calls.clear()
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert calls == [sources[0]]
    assert result.metadata['reduced_event_cache'] == {'hits': 2, 'misses': 1}
    group.metadata['raw_dgs']['t0_override'] = 100.0
    calls.clear()
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert calls == sources
    assert result.metadata['reduced_event_cache'] == {'hits': 0, 'misses': 3}
    group.datasets.remove(new)
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert result.num_events.sum() == 2


def test_cache_mask_file_changes_invalidate_events_and_normalization(tmp_path):
    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source)
    mask = tmp_path / 'mask.nxs'
    h5py = pytest.importorskip('h5py')
    with h5py.File(mask, 'w') as handle:
        handle.create_dataset('mantid_workspace_1/instrument/detector/detector_list', data=[42])
        handle.create_dataset('mantid_workspace_1/workspace/values', data=[[1.]])
        handle.create_dataset('mantid_workspace_1/workspace/errors', data=[[0.]])
    group = raw_dgs_dataset_group([source], mask_path=mask)
    first = bin_raw_dgs_group(group, **OPTIONS)
    assert first.num_events.sum() == 1
    with h5py.File(mask, 'r+') as handle:
        handle['mantid_workspace_1/workspace/values'][0, 0] = 0
    second = bin_raw_dgs_group(group, **OPTIONS)
    assert second.metadata['reduced_event_cache']['misses'] == 1
    assert second.num_events.sum() == 0
    assert not second.metadata['normalization_denominator'].any()


def test_cache_roundtrip_is_lazy_streamed_and_prunes_removed_runs(tmp_path, monkeypatch):
    sources = [tmp_path / f'SEQ_{run}.nxs.h5' for run in (42, 43)]
    for source in sources:
        _write_raw_dgs(source)
    group = raw_dgs_dataset_group(sources)
    expected = bin_raw_dgs_group(group, **OPTIONS)
    project = NfitProject([DataGroup('test', subgroups=[group])])
    path = tmp_path / 'events.nfit'
    save_project(project, path)
    assert len([n for n in zipfile.ZipFile(path).namelist() if n.startswith('assets/reduced_events/')]) == 2
    original = np.load

    def fail(*args, **kwargs):
        pytest.fail('project open/save loaded event arrays')

    with monkeypatch.context() as context:
        context.setattr(np, 'load', fail)
        reopened = load_project(path)
        save_project(reopened, tmp_path / 'copy.nfit')
        duplicated = copy.deepcopy(reopened)
        save_project(duplicated, tmp_path / 'deepcopy.nfit')
    assert np.load is original
    group = reopened.data_groups[0].subgroups[0]
    # A saved reduction also remains useful when the original raw file is absent.
    for source in sources:
        source.unlink()
    actual = bin_raw_dgs_group(group, max_batch_bytes=40, **OPTIONS)
    assert_equal(actual, expected)
    assert actual.metadata['reduced_event_cache']['hits'] == 2
    removed = group.datasets.pop()
    save_project(reopened, path)
    with zipfile.ZipFile(path) as archive:
        assert not any(removed.id in name for name in archive.namelist())
    cache = group.datasets[0]._raw_dgs_reduction_cache
    with cache.open() as archive:
        assert max(len(events) for events, _ in iter_cached_event_chunks(archive, 40)) <= 1
    clear_reduced_event_cache(group.datasets[0])
    assert reduced_event_cache_info(group.datasets[0]) is None
    save_project(reopened, path)
    with zipfile.ZipFile(path) as archive:
        assert not any(n.startswith('assets/reduced_events/') for n in archive.namelist())


def test_powder_and_hkle_share_cached_laboratory_events(tmp_path):
    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    bin_raw_dgs_group(group, **OPTIONS)
    options = dict(lower=[0, -100], upper=[20, 20], num_bins=[2, 2])
    cached = bin_raw_dgs_powder_group(group, **options)
    group.metadata['raw_dgs']['cache_reduced_events'] = False
    uncached = bin_raw_dgs_powder_group(group, **options)
    assert_equal(cached, uncached)
    assert cached.metadata['reduced_event_cache']['hits'] == 1


def test_incomplete_reduction_never_publishes_a_cache(tmp_path):
    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])

    def cancel(progress):
        if progress['stage'] == 'raw_dgs_events':
            raise RuntimeError('cancelled')

    with pytest.raises(RuntimeError, match='cancelled'):
        bin_raw_dgs_group(group, progress_callback=cancel, max_batch_bytes=96, **OPTIONS)
    assert reduced_event_cache_info(group.datasets[0]) is None


def test_copied_run_has_an_independent_project_cache_reference(tmp_path):
    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    bin_raw_dgs_group(group, **OPTIONS)
    original = group.datasets[0]
    copied = original.copy(name='copy')
    group.datasets.append(copied)
    project = NfitProject([DataGroup('test', subgroups=[group])])
    path = tmp_path / 'copies.nfit'
    save_project(project, path)
    assert original.id in reduced_event_cache_info(original)['member']
    assert copied.id in reduced_event_cache_info(copied)['member']
    assert original.metadata is not copied.metadata
    group.datasets.remove(copied)
    save_project(project, path)
    result = bin_raw_dgs_group(load_project(path).data_groups[0].subgroups[0], **OPTIONS)
    assert result.metadata['reduced_event_cache']['hits'] == 1
