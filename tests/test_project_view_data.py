from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import nfit.project_data as project_data
import nfit.project_view_data as project_view_data
from nfit.dataset import PointListData
from nfit.pipeline import DatasetEntry
from tests.project_gui_test_support import _tiny_mdhisto_data

VIEW_DATA_EXPORTS = (
    "KINEMATIC_KF_KI_INCLUDED_KEY",
    "_apply_dataset_scale",
    "_apply_kinematic_normalization_to_points",
    "_apply_kinematic_normalization_to_view",
    "_apply_spectral_channel_view",
    "_kinematic_energy_metadata",
    "_kinematic_kf_ki_factor",
    "_mdhisto_without_nfit_masks",
    "_with_viewer_dataset_metadata",
)


@pytest.mark.parametrize("name", VIEW_DATA_EXPORTS)
def test_project_data_reexports_view_data_objects(name: str) -> None:
    assert getattr(project_data, name) is getattr(project_view_data, name)


def test_view_data_service_does_not_import_project_facade_or_qt() -> None:
    path = Path(project_view_data.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert "project_data" not in imported_modules
    assert not any("PySide" in module or module.startswith("qt") for module in imported_modules)


def test_final_scaled_view_is_canonical_and_scale_change_reuses_source(monkeypatch):
    dataset = DatasetEntry("sample", _tiny_mdhisto_data(3.0), kind="mdhisto", scale_factor=2.0)
    project_data.dataset_rebin_config(dataset).update(enabled=False, stale=False)
    calls = []
    original = project_data._apply_dataset_scale

    def scale(entry, data):
        calls.append((entry.scale_factor, data))
        return original(entry, data)

    monkeypatch.setattr(project_data, "_apply_dataset_scale", scale)
    try:
        first = project_data.dataset_for_slice_viewer(dataset)
        second = project_data.dataset_for_slice_viewer(dataset)
        assert first is second
        assert len(calls) == 1
        assert first.signal is second.signal
        np.testing.assert_allclose(first.signal, 6.0)

        dataset.scale_factor = 4.0
        changed = project_data.dataset_for_slice_viewer(dataset)
        assert changed is not first
        assert len(calls) == 2
        assert calls[0][1] is calls[1][1]
        np.testing.assert_allclose(changed.signal, 12.0)
        np.testing.assert_allclose(first.signal, 6.0)
    finally:
        project_data._VIEWER_VIEW_CACHE.discard_matching(lambda key: isinstance(key, str) and key.startswith(dataset.id))


def test_replacing_source_invalidates_final_prepared_view():
    dataset = DatasetEntry("sample", _tiny_mdhisto_data(3.0), kind="mdhisto", scale_factor=2.0)
    project_data.dataset_rebin_config(dataset).update(enabled=False, stale=False)
    try:
        first = project_data.dataset_for_slice_viewer(dataset)
        dataset.replace_data(_tiny_mdhisto_data(5.0))
        second = project_data.dataset_for_slice_viewer(dataset)
        assert first is not second
        np.testing.assert_allclose(second.signal, 10.0)
    finally:
        project_data._VIEWER_VIEW_CACHE.discard_matching(lambda key: isinstance(key, str) and key.startswith(dataset.id))


def test_point_list_viewer_metadata_keeps_immutable_columns_shared():
    source = PointListData(
        columns={"T": [1, 2], "M": [3, 4], "dM": [.1, .2]},
        coordinate_names=["T"], channels=[{"value": "M", "error": "dM"}],
    )
    dataset = DatasetEntry("moment", source, data_type="magnetization")
    prepared = project_view_data._with_viewer_dataset_metadata(dataset, source)
    assert prepared is not source
    assert all(prepared.columns[name] is source.columns[name] for name in source.columns)
    assert not prepared.columns["M"].flags.writeable
    assert "nfit_data_type" not in source.metadata


def test_final_view_preflight_rejects_transform_before_allocating(monkeypatch):
    from nfit import resource_budget

    dataset = DatasetEntry("sample", _tiny_mdhisto_data(3.0), kind="mdhisto", scale_factor=2.0)
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 0)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: 1)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 1)
    cache = {}
    with pytest.raises(resource_budget.ResourceLimitError, match="Preparing viewer data"):
        project_view_data.cached_viewer_preparation(
            dataset, dataset.data, cache=cache, key="sample",
            prepare=lambda: pytest.fail("allocation occurred before preflight"),
        )
    assert cache == {}
    assert resource_budget.snapshot_memory().reserved_bytes == 0
