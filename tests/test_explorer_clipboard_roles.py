"""GUI round trips for every copyable explorer object role."""

from __future__ import annotations

import pytest

from nfit.analysis.core import AnalysisEntry
from nfit.pipeline import BackgroundSpec, DatasetGroup, FitTimelineEntry, PlotEntry
from nfit.project_gui import ensure_fit_history
from tests.project_gui_test_support import (
    DataGroup,
    DatasetEntry,
    NfitProject,
    NfitProjectExplorer,
    create_mask,
    create_model_component,
)


def _find_item(explorer, role, predicate=lambda _item: True):
    def visit(item):
        if explorer._objects_for_item(item)[4] == role and predicate(item):
            return item
        for index in range(item.childCount()):
            found = visit(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(explorer.tree.topLevelItemCount()):
        found = visit(explorer.tree.topLevelItem(index))
        if found is not None:
            return found
    raise AssertionError(f"No {role!r} item matched")


def _fixture():
    source_dataset = DatasetEntry("source dataset", None)
    target_dataset = DatasetEntry("target dataset", None)
    source_mask = create_mask(source_dataset, "source mask")
    create_mask(target_dataset, "target mask")
    source_background = BackgroundSpec("source background", "")
    source_dataset.backgrounds.append(source_background)
    target_dataset.backgrounds.append(BackgroundSpec("target background", ""))
    source_node = DatasetGroup("source collection", datasets=[source_dataset])
    target_node = DatasetGroup("target collection", datasets=[target_dataset])
    source_group_mask = create_mask(DatasetEntry("template", None), "source shared mask")
    source_node.masks.append(source_group_mask)
    target_node.masks.append(create_mask(DatasetEntry("target template", None), "target shared mask"))
    source_group_background = BackgroundSpec("source shared background", "")
    source_node.backgrounds.append(source_group_background)
    target_node.backgrounds.append(BackgroundSpec("target shared background", ""))
    group = DataGroup("Workspace", subgroups=[source_node, target_node])
    model = create_model_component(group, "source model")
    ensure_fit_history(group)
    group.fits.append(FitTimelineEntry("saved result", kind="result"))
    analysis = AnalysisEntry("source analysis", "identity", [], {})
    group.analyses.append(analysis)
    plot = PlotEntry("source plot")
    group.plots.append(plot)
    explorer = NfitProjectExplorer(NfitProject([group]))
    return {
        "explorer": explorer,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source_node": source_node,
        "target_node": target_node,
        "source_mask": source_mask,
        "source_background": source_background,
        "source_group_mask": source_group_mask,
        "source_group_background": source_group_background,
        "model": model,
        "analysis": analysis,
        "plot": plot,
        "group": group,
    }


def _collection(state, kind):
    if kind == "mask":
        return state["target_dataset"].masks
    if kind == "group_mask":
        return state["target_node"].masks
    if kind == "background":
        return state["target_dataset"].backgrounds
    if kind == "group_background":
        return state["target_node"].backgrounds
    if kind == "model":
        return list(state["group"].models.values())
    if kind == "fit":
        return state["group"].fits
    if kind == "analysis":
        return state["group"].analyses
    if kind == "plot":
        return state["group"].plots
    raise AssertionError(kind)


@pytest.mark.parametrize("owner_kind", ["dataset", "collection", "workspace"])
@pytest.mark.parametrize("source_kind", ["dataset", "collection"])
def test_paste_on_background_creates_independent_siblings(monkeypatch, owner_kind, source_kind):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    source = DatasetEntry("powder source", None)
    source_node = DatasetGroup("powder collection", datasets=[source])
    sample = DatasetEntry("sample run", None)
    sample_node = DatasetGroup("sample collection", datasets=[sample])
    workspace = DataGroup("workspace", subgroups=[sample_node, source_node])
    owner = {"dataset": sample, "collection": sample_node, "workspace": workspace}[owner_kind]
    original = BackgroundSpec(
        "powder subtraction",
        source.id if source_kind == "dataset" else "",
        source_group_id=source_node.id if source_kind == "collection" else None,
        scale=0.75,
        metadata={"custom": {"value": 1}},
    )
    owner.backgrounds.append(original)
    explorer = NfitProjectExplorer(NfitProject([workspace]))
    role = "background" if owner_kind == "dataset" else "group_background"
    item = _find_item(explorer, role, lambda item: explorer._background_for_item(item) is original)
    explorer.tree.setCurrentItem(item)
    explorer.copy_selected()

    for expected_count in (2, 3):
        current = explorer.tree.currentItem()
        assert dict(explorer._context_menu_action_specs(current))["Paste"] is True
        explorer.paste_into_selection()
        assert len(owner.backgrounds) == expected_count
        copied = owner.backgrounds[-1]
        assert explorer._background_for_item(explorer.tree.currentItem()) is copied
        assert copied is not original
        assert copied.scale == original.scale
        assert copied.projection == original.projection
        assert copied.interpolation == original.interpolation
        assert copied.enabled == original.enabled
        assert copied.source_dataset_id == original.source_dataset_id
        assert copied.source_group_id == original.source_group_id
        if source_kind == "dataset":
            assert copied.source_entry is source
        else:
            assert copied.source_group is source_node
        assert copied.metadata == original.metadata

    assert len({background.name for background in owner.backgrounds}) == 3
    owner.backgrounds[-1].metadata["custom"]["value"] = 2
    owner.backgrounds[-1].scale = 2.0
    assert original.metadata["custom"]["value"] == 1
    assert original.scale == 0.75
    assert explorer.has_unsaved_changes


@pytest.mark.parametrize("role", ["mask", "group_mask"])
def test_paste_on_mask_creates_independent_siblings(monkeypatch, role):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    state = _fixture()
    explorer = state["explorer"]
    original = state["source_mask" if role == "mask" else "source_group_mask"]
    owner = state["source_dataset" if role == "mask" else "source_node"]
    item = _find_item(explorer, role, lambda item: explorer._objects_for_item(item)[2] is original)
    explorer.tree.setCurrentItem(item)
    explorer.copy_selected()
    for expected_count in (2, 3):
        assert dict(explorer._context_menu_action_specs(explorer.tree.currentItem()))["Paste"]
        explorer.paste_into_selection()
        assert len(owner.masks) == expected_count
        assert explorer._objects_for_item(explorer.tree.currentItem())[2] is owner.masks[-1]
    assert len({mask.name for mask in owner.masks}) == 3
    owner.masks[-1].enabled = not original.enabled
    assert owner.masks[1].enabled == original.enabled


def test_background_can_be_pasted_into_root_datasets_collection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    state = _fixture()
    explorer = state["explorer"]
    item = _find_item(explorer, "background")
    explorer.tree.setCurrentItem(item)
    explorer.copy_selected()
    target = _find_item(explorer, "datasets")
    explorer.tree.setCurrentItem(target)
    assert dict(explorer._context_menu_action_specs(target))["Paste"]
    explorer.paste_into_selection()
    assert len(state["group"].backgrounds) == 1
    assert explorer._background_for_item(explorer.tree.currentItem()) is state["group"].backgrounds[0]
    assert len(state["source_dataset"].backgrounds) == 1


@pytest.mark.parametrize(
    ("kind", "header"),
    [
        ("mask", False), ("mask", True),
        ("group_mask", False), ("group_mask", True),
        ("background", False), ("background", True),
        ("group_background", False), ("group_background", True),
        ("model", False), ("model", True),
        ("fit", False), ("fit", True),
        ("analysis", False), ("analysis", True),
        ("plot", False), ("plot", True),
    ],
)
@pytest.mark.parametrize("target_leaf", [False, True])
def test_copy_paste_roles_round_trip_through_gui(monkeypatch, kind, header, target_leaf):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    state = _fixture()
    explorer = state["explorer"]

    if kind == "mask":
        source = _find_item(explorer, "masks" if header else "mask", lambda item: header or explorer._objects_for_item(item)[2] is state["source_mask"])
        target = _find_item(explorer, "masks", lambda item: explorer._objects_for_item(item)[1] is state["target_dataset"])
    elif kind == "group_mask":
        source = _find_item(explorer, "group_masks" if header else "group_mask", lambda item: header or explorer._objects_for_item(item)[2] is state["source_group_mask"])
        target = _find_item(explorer, "group_masks", lambda item: explorer._dataset_group_for_item(item) is state["target_node"])
    elif kind == "background":
        source = _find_item(explorer, "backgrounds" if header else "background", lambda item: header or explorer._background_for_item(item) is state["source_background"])
        target = _find_item(explorer, "backgrounds", lambda item: explorer._objects_for_item(item)[1] is state["target_dataset"])
    elif kind == "group_background":
        source = _find_item(explorer, "group_backgrounds" if header else "group_background", lambda item: header or explorer._background_for_item(item) is state["source_group_background"])
        target = _find_item(explorer, "group_backgrounds", lambda item: explorer._dataset_group_for_item(item) is state["target_node"])
    elif kind == "model":
        source = _find_item(explorer, "models" if header else "model")
        target = _find_item(explorer, "models")
    elif kind == "fit":
        source = _find_item(explorer, "fits" if header else "fit", lambda item: header or explorer._fit_entry_for_item(item).kind != "initial")
        target = _find_item(explorer, "fits")
    elif kind == "analysis":
        source = _find_item(explorer, "analyses" if header else "analysis")
        target = _find_item(explorer, "analyses")
    else:
        source = _find_item(explorer, "plots" if header else "plot")
        target = _find_item(explorer, "plots")

    if target_leaf:
        def matches_target(item):
            if kind in {"mask", "group_mask"}:
                return explorer._objects_for_item(item)[2] is _collection(state, kind)[0]
            if kind in {"background", "group_background"}:
                return explorer._background_for_item(item) is _collection(state, kind)[0]
            if kind == "fit":
                return explorer._fit_entry_for_item(item).kind != "initial"
            return True

        target = _find_item(explorer, kind, matches_target)

    before = _collection(state, kind)
    before_count = len(before)
    before_ids = {id(item) for item in before}
    explorer.tree.setCurrentItem(source)
    source = explorer.tree.currentItem()
    if kind == "fit":
        # Selecting a saved fit restores its snapshot and deliberately
        # rebuilds the tree, invalidating the old item handles.
        target = _find_item(explorer, "fit" if target_leaf else "fits",
                            lambda item: not target_leaf or explorer._fit_entry_for_item(item).kind != "initial")
    assert "Copy" in explorer.context_menu_action_names(source)
    explorer.copy_selected()
    copied_count = len(explorer._clipboard.items)
    explorer.tree.setCurrentItem(target)
    target = explorer.tree.currentItem()
    assert dict(explorer._context_menu_action_specs(target))["Paste"] is True
    explorer.paste_into_selection()

    after = _collection(state, kind)
    assert len(after) == before_count + copied_count
    added = [item for item in after if id(item) not in before_ids]
    assert added
    copied = added[-1]
    original = {
        "mask": state["source_mask"],
        "group_mask": state["source_group_mask"],
        "background": state["source_background"],
        "group_background": state["source_group_background"],
        "model": state["model"],
        "analysis": state["analysis"],
        "plot": state["plot"],
    }.get(kind)
    if original is not None:
        original_name = original.name
        copied.name = "changed copied name"
        assert original.name == original_name
