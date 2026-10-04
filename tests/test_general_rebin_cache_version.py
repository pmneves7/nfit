"""General rebin fixes invalidate old results without invalidating native DGS."""

import json

import numpy as np
import pytest

from nfit import (
    DataGroup,
    DatasetEntry,
    corelli_dataset_group,
    mdevent_dataset_group,
    project_data,
    raw_dgs_dataset_group,
    refresh_composite_dataset,
)
from nfit import project_composites as composites
from nfit.project_cache_compat import composite_cache_signatures_match
from nfit.rebin import REBIN_NUMERICAL_VERSION
from tests.project_gui_test_support import _tiny_mdhisto_data
from tests.test_corelli import _write_corelli
from tests.test_mdevent import _write_mdevent
from tests.test_raw_dgs import _write_raw_dgs


def _unversioned(signature, index):
    payload = json.loads(signature)
    numerical = json.loads(payload[index])
    assert numerical.pop("general_rebin_numerical_version") == REBIN_NUMERICAL_VERSION
    payload[index] = json.dumps(numerical, sort_keys=True, default=str)
    return json.dumps(payload)


@pytest.fixture
def general_group():
    composites._COMPOSITE_DATA_CACHE.clear()
    project_data._VIEWER_VIEW_CACHE.clear()
    entry = DatasetEntry("source", _tiny_mdhisto_data(2.), kind="mdhisto")
    group = DataGroup("sample", datasets=[entry])
    yield group, entry
    composites._COMPOSITE_DATA_CACHE.clear()
    project_data._VIEWER_VIEW_CACHE.clear()


@pytest.mark.parametrize("fractional, normalize", [(True, True), (False, False)])
def test_old_general_composite_cache_replays_instead_of_returning_old_numerics(
    general_group, monkeypatch, fractional, normalize
):
    group, entry = general_group
    config = composites.data_group_composite_config(group)
    config.update(enabled=True, minimum_coverage=0., normalize=normalize)
    for axis in config["axes"]:
        axis["fractional"] = fractional
    expected = refresh_composite_dataset(group)
    key = composites._composite_cache_key(group)
    signature, _ = composites._COMPOSITE_DATA_CACHE.get(key)
    previous = _unversioned(signature, 4)
    assert not composite_cache_signatures_match(previous, signature)
    composites._COMPOSITE_DATA_CACHE[key] = (previous, expected.with_updates(signal=np.full(expected.shape, 99.)))
    calls = []
    original = composites.composite_dataset_data
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(composites, "composite_dataset_data", record)
    actual = refresh_composite_dataset(group)
    assert calls
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_array_equal(actual.num_events, expected.num_events)


@pytest.mark.parametrize("fractional, normalize", [(True, True), (False, False)])
def test_old_general_view_cache_replays_instead_of_returning_old_numerics(
    general_group, monkeypatch, fractional, normalize
):
    group, entry = general_group
    config = project_data.dataset_rebin_config(entry)
    config.update(enabled=True, minimum_coverage=0., normalize=normalize)
    for axis in config["axes"]:
        axis["fractional"] = fractional
    expected = project_data.dataset_for_slice_viewer(entry)
    signature, _ = project_data._VIEWER_VIEW_CACHE.get(entry.id)
    previous = _unversioned(signature, 3)
    project_data._VIEWER_VIEW_CACHE[entry.id] = (previous, expected.with_updates(signal=np.full(expected.shape, 99.)))
    calls = []
    original = project_data._viewer_data_before_scale_uncached
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(project_data, "_viewer_data_before_scale_uncached", record)
    actual = project_data.dataset_for_slice_viewer(entry)
    assert calls
    np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
    np.testing.assert_array_equal(actual.num_events, expected.num_events)


@pytest.mark.parametrize("native", ["dgs", "mdevent", "corelli"])
def test_general_rebin_version_does_not_change_native_group_cache_signatures(tmp_path, monkeypatch, native):
    source = tmp_path / f"{native}.nxs.h5"
    if native == "dgs":
        _write_raw_dgs(source)
        group = raw_dgs_dataset_group([source])
    elif native == "mdevent":
        _write_mdevent(source)
        group = mdevent_dataset_group(source)
    else:
        _write_corelli(source)
        group = corelli_dataset_group([source])
    config = composites.data_group_composite_config(group)
    config["enabled"] = True
    before = composites._composite_cache_signature(group)
    assert "general_rebin_numerical_version" not in before
    monkeypatch.setattr(composites, "REBIN_NUMERICAL_VERSION", REBIN_NUMERICAL_VERSION + 1)
    assert composites._composite_cache_signature(group) == before
