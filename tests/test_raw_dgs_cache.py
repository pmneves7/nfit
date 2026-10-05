"""Raw reductions are reusable per run and remain lazy in project archives."""

import copy
import json
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
from nfit.raw_dgs_cache import cache_event_chunks, iter_cached_event_chunks
from tests.test_raw_dgs import _rewrite_instrument_xml, _write_raw_dgs

OPTIONS = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[2]*4)


def test_abandoned_cache_generator_does_not_restore_finished_binning_reservation(tmp_path, monkeypatch):
    from nfit import raw_dgs_cache
    from nfit.pipeline import DatasetEntry
    from nfit.resource_budget import reserve_memory, snapshot_memory

    monkeypatch.setattr(raw_dgs_cache, "_EVENT_BLOCK_ROWS", 7)
    dataset = DatasetEntry(name="run", data=None,
                           metadata={"source_file": str(tmp_path / "raw.nxs")})
    events = np.arange(114.).reshape(19, 6)
    generator = cache_event_chunks(dataset, "signature", {}, {}, iter([(events, 19)]))
    before = snapshot_memory().reserved_bytes
    try:
        with reserve_memory(1024**2):
            assert next(generator)[0] is events
            assert snapshot_memory().reserved_bytes == before + 1024**2
        assert snapshot_memory().reserved_bytes == before
    finally:
        generator.close()
    assert snapshot_memory().reserved_bytes == before
    assert dataset._raw_dgs_reduction_cache is None
    assert not list(tmp_path.iterdir())


def test_reduced_event_staging_follows_project_instead_of_source_or_tmp(tmp_path):
    sources = tmp_path / "raw"
    outputs = tmp_path / "projects"
    sources.mkdir()
    outputs.mkdir()
    source = sources / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    project = NfitProject(data_groups=[DataGroup(name="runs", subgroups=[group])])
    path = outputs / "project.nfit"
    save_project(project, path)
    loaded = load_project(path)
    loaded_group = loaded.data_groups[0].subgroups[0]
    bin_raw_dgs_group(loaded_group, **OPTIONS)
    staged = loaded_group.datasets[0]._raw_dgs_reduction_cache.content
    assert staged.parent.parent == outputs
    assert not list(sources.glob("nfit-reduced-events-*"))
    save_project(loaded, path)
    assert not staged.exists()


def test_new_imported_runs_inherit_saved_project_workspace(tmp_path):
    from nfit import create_data_group, import_dataset_paths

    raw = tmp_path / "raw"
    outputs = tmp_path / "outputs"
    raw.mkdir()
    outputs.mkdir()
    source = raw / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    project = NfitProject()
    path = outputs / "project.nfit"
    save_project(project, path)
    group = create_data_group(project, "new runs")
    entries = import_dataset_paths(group, [source], data_type="single_crystal_inelastic")
    assert entries[0].metadata["_project_path"] == str(path)
    bin_raw_dgs_group(group.subgroups[0], **OPTIONS)
    assert entries[0]._raw_dgs_reduction_cache.content.parent.parent == outputs


@pytest.mark.parametrize('empty', [False, True])
def test_cache_stores_bounded_blocks_and_preserves_event_bits(tmp_path, monkeypatch, empty):
    from nfit import raw_dgs_cache
    from nfit.pipeline import DatasetEntry

    monkeypatch.setattr(raw_dgs_cache, '_EVENT_BLOCK_ROWS', 7)
    dataset = DatasetEntry(name='run', data=None)
    # Include empty banks, a chunk straddling several blocks, signed zero, NaN,
    # and more raw events than retained events. Storage must preserve exact bits.
    events = np.arange(114, dtype=np.float64).reshape(19, 6)
    events[0, :3] = [0., -0., np.nan]
    chunks = [(events[:0], 11), (events[:3], 6), (events[3:], 32), (events[:0], 5)]
    if empty:
        chunks = [(events[:0], 11), (events[:0], 5)]
    yielded = list(cache_event_chunks(dataset, 'signature', {}, {}, iter(chunks)))
    assert all(left[0] is right[0] for left, right in zip(yielded, chunks, strict=True))
    with zipfile.ZipFile(dataset._raw_dgs_reduction_cache.content) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert archive.testzip() is None
    with dataset._raw_dgs_reduction_cache.open() as archive:
        header = json.loads(str(archive['header_json'].item()))
        assert header['chunk_count'] == (1 if empty else 3)
        assert archive['events_0'].flags.f_contiguous
        decoded = list(iter_cached_event_chunks(archive, 80))
    assert max(len(chunk) for chunk, _ in decoded) <= 2
    actual = np.concatenate([chunk for chunk, _ in decoded])
    expected = events[:0] if empty else events
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert sum(count for _, count in decoded) == sum(count for _, count in chunks)


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

    def inspect(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

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


@pytest.mark.parametrize('previously_compressed', [False, True])
def test_cache_roundtrip_is_lazy_streamed_and_prunes_removed_runs(tmp_path, monkeypatch, previously_compressed):
    sources = [tmp_path / f'SEQ_{run}.nxs.h5' for run in (42, 43)]
    for source in sources:
        _write_raw_dgs(source)
    group = raw_dgs_dataset_group(sources)
    expected = bin_raw_dgs_group(group, **OPTIONS)
    if previously_compressed:
        for dataset in group.datasets:
            cache = dataset._raw_dgs_reduction_cache
            with cache.open() as archive:
                payload = {key: archive[key] for key in archive.files}
            np.savez_compressed(cache.content, **payload)
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



def test_reduction_version_invalidates_event_and_histogram_caches(tmp_path, monkeypatch):
    from nfit import project_composites, raw_dgs_cache

    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    bin_raw_dgs_group(group, **OPTIONS)
    signature = project_composites._composite_cache_signature(group)
    monkeypatch.setattr(raw_dgs_cache, 'RAW_DGS_REDUCTION_VERSION', raw_dgs_cache.RAW_DGS_REDUCTION_VERSION + 1)
    monkeypatch.setattr(project_composites, 'RAW_DGS_REDUCTION_VERSION', raw_dgs_cache.RAW_DGS_REDUCTION_VERSION)
    assert project_composites._composite_cache_signature(group) != signature
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert result.metadata['reduced_event_cache']['misses'] == 1


@pytest.mark.parametrize("per_run_override", [False, True])
@pytest.mark.parametrize("policy, enabled, changed", [
    ("mantid", True, False), ("high_precision", True, True), ("high_precision", False, False),
])
def test_mantid_he3_convention_invalidates_only_affected_reductions_and_histograms(
    tmp_path, monkeypatch, policy, enabled, changed, per_run_override,
):
    from nfit import project_composites, raw_dgs_cache

    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source, with_he3=True)
    group = raw_dgs_dataset_group([source], event_precision_policy=policy)
    group.metadata["raw_dgs"]["he3_detector_efficiency_correction"] = enabled
    if per_run_override:
        from nfit.reduction_recipes import set_reduction_settings

        group.metadata["raw_dgs"]["he3_detector_efficiency_correction"] = False
        set_reduction_settings(group, {
            "he3_detector_efficiency_correction": enabled,
        }, dataset_ids=[group.datasets[0].id])
    with monkeypatch.context() as legacy:
        legacy.setattr(raw_dgs_cache, "reduction_convention_signature", lambda config: {})
        legacy.setattr(project_composites, "reduction_convention_signature", lambda config: {})
        bin_raw_dgs_group(group, **OPTIONS)
        old_signature = group.datasets[0]._raw_dgs_reduction_cache.signature
        old_histogram_signature = project_composites._composite_cache_signature(group)
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert (group.datasets[0]._raw_dgs_reduction_cache.signature != old_signature) == changed
    assert (project_composites._composite_cache_signature(group) != old_histogram_signature) == changed
    assert result.metadata["reduced_event_cache"] == {
        "hits": int(not changed), "misses": int(changed),
    }


@pytest.mark.parametrize("explicit_alternatives", [False, True])
def test_old_saved_dgs_reduction_rebuilds_lazily_with_current_defaults(
    tmp_path, monkeypatch, explicit_alternatives,
):
    from nfit import project_composites, raw_dgs_cache
    from nfit.dgs_reduction_policy import resolved_dgs_reduction_policies

    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    config = group.metadata["raw_dgs"]
    keys = ("monitor_variance_policy", "event_precision_policy", "symmetry_variance_policy")
    for key in keys:
        config.pop(key, None)
    if explicit_alternatives:
        config.update(monitor_variance_policy="stable", event_precision_policy="high_precision",
                      symmetry_variance_policy="within_bin_covariance")
    with monkeypatch.context() as old:
        old.setattr(raw_dgs_cache, "RAW_DGS_REDUCTION_VERSION", 7)
        old.setattr(project_composites, "RAW_DGS_REDUCTION_VERSION", 7)
        old.setattr(project_composites, "DGS_REDUCTION_POLICY_VERSION", 3)
        bin_raw_dgs_group(group, **OPTIONS)
        previous = project_composites._composite_cache_signature(group)
        path = tmp_path / "old.nfit"
        save_project(NfitProject([DataGroup("test", subgroups=[group])]), path)
    with monkeypatch.context() as lazy:
        def fail(*args, **kwargs):
            pytest.fail("opening an old project loaded scientific arrays")
        lazy.setattr(np, "load", fail)
        reopened = load_project(path)
    group = reopened.data_groups[0].subgroups[0]
    assert project_composites._composite_cache_signature(group) != previous
    effective = resolved_dgs_reduction_policies(group.metadata["raw_dgs"])
    assert effective == ({
        "monitor_variance_policy": "stable", "event_precision_policy": "high_precision",
        "symmetry_variance_policy": "within_bin_covariance",
    } if explicit_alternatives else {
        "monitor_variance_policy": "mantid", "event_precision_policy": "mantid",
        "symmetry_variance_policy": "independent_copies",
    })
    result = bin_raw_dgs_group(group, **OPTIONS)
    assert result.metadata["reduced_event_cache"] == {"hits": 0, "misses": 1}
    repeated = bin_raw_dgs_group(group, **OPTIONS)
    assert repeated.metadata["reduced_event_cache"] == {"hits": 1, "misses": 0}
    assert_equal(repeated, result)
