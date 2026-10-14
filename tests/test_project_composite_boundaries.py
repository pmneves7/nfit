from __future__ import annotations

import ast
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from nfit import (
    project_composites,
    project_data,
    project_masks,
    project_rebinning,
    project_view_data,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry


def _histogram():
    return MDHistoData(
        axes=(MDHistoAxis("Q", np.array([0., 1., 2.]), "rlu", "h"),),
        signal=np.array([2., 4.]), errors=np.ones(2),
        mask=np.zeros(2, dtype=bool), num_events=np.ones(2),
    )


def test_composite_service_imports_pure_helpers_from_their_owners():
    tree = ast.parse(Path(project_composites.__file__).read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    owners = {
        "project_rebinning": project_rebinning,
        "project_masks": project_masks,
        "project_view_data": project_view_data,
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in owners:
            for alias in node.names:
                assert getattr(project_composites, alias.asname or alias.name) is getattr(
                    owners[node.module], alias.name
                )
    modules = {node.module or "" for node in imports}
    modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names)
    assert not any("project_data" in name or "project_gui" in name or "PySide" in name
                   or "PyQt" in name for name in modules)
    wiring = {"configure_composite_backend", "_composite_operations", "_composite_policy"}
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in wiring:
            assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id in {"globals", "locals", "eval", "exec", "getattr"}
                           for node in ast.walk(function))


@pytest.mark.parametrize("name", ("effective_dataset_masks", "_mask_signature"))
def test_facade_and_composite_mask_helpers_share_the_service_owner(name):
    assert getattr(project_data, name) is getattr(project_masks, name)
    assert getattr(project_composites, name) is getattr(project_masks, name)


def test_composite_backend_accepts_only_the_bounded_operations_contract():
    assert {field.name for field in fields(project_composites.CompositeOperations)} == {
        "load_dataset", "apply_dataset_backgrounds", "dataset_point_count",
        "dataset_rebin_source_points", "viewer_signature", "composite_builder", "policy",
    }
    before = project_composites._composite_operations()
    with pytest.raises(TypeError, match="requires CompositeOperations"):
        project_composites.configure_composite_backend(vars(project_data))
    assert project_composites._composite_operations() is before


def test_explicit_operations_prepare_background_sources_without_a_facade(monkeypatch):
    histogram = _histogram()
    dataset = DatasetEntry("lazy", None, kind="mdhisto", backgrounds=[BackgroundSpec("b")])
    group = DataGroup("sample", datasets=[dataset])
    calls = []

    def load(candidate, **kwargs):
        assert candidate is dataset
        calls.append("load")
        return histogram

    def background(candidate, data):
        assert candidate is dataset
        calls.append("background")
        return data.with_updates(signal=data.signal - 1)

    operations = replace(project_composites._composite_operations(),
                         load_dataset=load, apply_dataset_backgrounds=background)
    monkeypatch.setattr(project_composites, "_COMPOSITE_OPERATIONS", operations)
    result = project_composites._source_data_for_group_composite(group, dataset)

    assert calls == ["load", "background"]
    np.testing.assert_array_equal(result.signal, [1., 3.])
    np.testing.assert_array_equal(histogram.signal, [2., 4.])
    assert dataset.data is None


def test_facade_loading_background_and_view_signature_callbacks_remain_live(monkeypatch):
    histogram = _histogram()
    dataset = DatasetEntry("sample", histogram, kind="mdhisto",
                           backgrounds=[BackgroundSpec("background")])
    group = DataGroup("combined", datasets=[dataset])
    calls = []
    monkeypatch.setattr(project_data, "_ensure_dataset_data_loaded",
                        lambda candidate: calls.append(("load", candidate)) or histogram)
    monkeypatch.setattr(project_data, "_apply_dataset_backgrounds",
                        lambda candidate, data: calls.append(("background", candidate)) or data)
    monkeypatch.setattr(project_data, "_viewer_view_signature", lambda *args: "before")
    config = project_composites.data_group_composite_config(group)
    before = project_composites._composite_cache_signature(group, config_override=config)
    monkeypatch.setattr(project_data, "_viewer_view_signature", lambda *args: "after")
    after = project_composites._composite_cache_signature(group, config_override=config)
    project_composites._source_data_for_group_composite(group, dataset)

    assert before != after
    assert calls[-2:] == [("load", dataset), ("background", dataset)]


def test_composite_policy_reads_all_live_facade_preferences(monkeypatch):
    monkeypatch.setattr(project_data, "GROUP_COMPOSITE_NAME", "Union")
    monkeypatch.setattr(project_data, "DEFAULT_REBIN_MAX_BATCH_MB", 17)
    monkeypatch.setattr(project_data, "REBIN_AUTO_MAX_CONTRIBUTIONS", 31)
    monkeypatch.setattr(project_data, "REBIN_AUTO_MAX_OUTPUT_BINS", 43)
    monkeypatch.setattr(project_data, "REBIN_RESOLUTION_MODE_KEY", "legacy_resolution")

    assert project_composites._composite_policy() == project_composites.CompositePolicy(
        name="Union", max_batch_mb=17, max_contributions=31, max_output_bins=43,
        resolution_mode_key="legacy_resolution",
    )
    assert project_composites._composite_dataset_name(DataGroup("sample")) == "sample Union"
