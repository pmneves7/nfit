from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nfit import project_data, project_rebinning
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from nfit.project_binning_policy import (
    background_rebin_explanation,
    background_source_explanations,
    rebin_presentation_policy,
)
from nfit.project_imports import GROUP_COMPOSITE_KEY


def _enabled(name: str, **kwargs) -> DatasetGroup:
    group = DatasetGroup(name, **kwargs)
    group.metadata[GROUP_COMPOSITE_KEY] = {"enabled": True}
    return group


def test_standalone_dataset_owns_editable_recipe_without_mutation() -> None:
    dataset = DatasetEntry("scan", None, parameters={"rebin": {"enabled": True}})
    root = DataGroup("root", datasets=[dataset])
    before = dict(dataset.parameters["rebin"])

    policy = rebin_presentation_policy(root, dataset)

    assert policy.owner is dataset
    assert policy.owner_kind == "dataset"
    assert policy.show_editor and not policy.linked
    assert dataset.parameters["rebin"] == before


def test_outermost_enabled_nested_composite_owns_member_grid() -> None:
    dataset = DatasetEntry("scan", None)
    child = _enabled("child", datasets=[dataset])
    parent = _enabled("parent", subgroups=[child])
    root = DataGroup("root", subgroups=[parent])

    policy = rebin_presentation_policy(root, dataset)

    assert policy.owner is parent
    assert policy.linked and not policy.show_editor
    assert not policy.allow_independent_recipe
    child_policy = rebin_presentation_policy(root, child)
    assert child_policy.owner is parent
    assert not child_policy.show_editor
    parent.metadata[GROUP_COMPOSITE_KEY]["enabled"] = False
    assert rebin_presentation_policy(root, dataset).owner is child
    assert rebin_presentation_policy(root, child).show_editor


def test_disabled_background_branch_keeps_its_own_grid() -> None:
    source = DatasetEntry("background run", None)
    background = _enabled("background", datasets=[source], enabled=False)
    parent = _enabled("sample", subgroups=[background])
    root = DataGroup("root", subgroups=[parent])
    assert rebin_presentation_policy(root, background).owner is background
    assert rebin_presentation_policy(root, background).show_editor
    assert rebin_presentation_policy(root, source).owner is background


def test_parent_grid_suppresses_child_rebins_and_save_targets(tmp_path, monkeypatch):
    import numpy as np

    from nfit import MDHistoAxis, MDHistoData, NfitProject
    from nfit import project_composites as composites
    from nfit import project_gui as gui

    data = MDHistoData(
        (MDHistoAxis("H", np.array([-.5, .5, 1.5]), "rlu", "momentum"),),
        np.ones(2), np.ones(2), np.zeros(2, dtype=bool), np.ones(2),
    )
    children = [DatasetGroup(f"child{i}", datasets=[DatasetEntry(f"run{i}", data, kind="mdhisto")]) for i in range(2)]
    for index, child in enumerate(children):
        entry = child.datasets[0]
        source = tmp_path / f"run{index}.npz"
        gui.save_dataset_file(entry, source, use_view=False)
        entry.metadata["source_file"] = str(source)
        entry.replace_data(data, source_backed=True)
    parent = DatasetGroup("parent", subgroups=children)
    root = DataGroup("root", subgroups=[parent])
    project = NfitProject([root])
    project.settings["cache_binnings"] = True
    for node, step in [(child, 1.) for child in children] + [(parent, .5)]:
        config = gui.data_group_composite_config(gui._composite_scope(root, node))
        config.update(enabled=True, minimum_coverage=0.)
        config["axes"][0].update(lower=0., upper=1., step_size=step, mode="step", auto_lower=False, auto_upper=False, auto_step_size=False)
    reduce = composites.composite_dataset_data
    calls = []

    def record(scope, **kwargs):
        config = kwargs.get("config_override")
        if config is not None:
            calls.append((scope.name, config["axes"][0]["step_size"]))
        return reduce(scope, **kwargs)

    monkeypatch.setattr(composites, "composite_dataset_data", record)
    scope = gui._composite_scope(root, parent)
    gui._cached_composite_dataset_data(scope)
    assert calls and all(step == .5 for _, step in calls)
    assert {name for name, _ in calls} == {"parent", "child0", "child1"}
    assert [target[1] for target in gui._project_binning_targets(project)] == ["parent · Default"]
    assert not gui.project_binnings_need_refresh(project)
    children[0].metadata[GROUP_COMPOSITE_KEY]["axes"][0]["step_size"] = 99.
    assert not gui.project_binnings_need_refresh(project)
    count = len(calls)
    gui.save_project(project, tmp_path / "parent.nfit")
    assert len(calls) == count
    assert {entry.get("node_id") for entry in project.settings[gui.PROJECT_BINNING_CACHE_ENTRIES_KEY]} == {parent.id}
    parent.metadata[GROUP_COMPOSITE_KEY]["enabled"] = False
    assert {target[1] for target in gui._project_binning_targets(project)} == {"child0 · Default", "child1 · Default"}


def test_disabled_organizational_collection_has_no_recipe_editor() -> None:
    child = DatasetGroup("folder")
    root = DataGroup("root", subgroups=[child])

    policy = rebin_presentation_policy(root, child)

    assert policy.owner is None
    assert policy.owner_kind == "none"
    assert not policy.show_editor
    assert not policy.show_binning_selector


def test_disabled_collection_still_exposes_enabled_named_output_selector() -> None:
    child = DatasetGroup(
        "folder",
        metadata={
            "composite_binnings": {
                "items": [{"id": "view", "config": {"enabled": True}}]
            }
        },
    )
    root = DataGroup("root", subgroups=[child])

    policy = rebin_presentation_policy(root, child)

    assert policy.owner is child
    assert policy.owner_kind == "composite"
    assert not policy.show_editor
    assert policy.show_binning_selector


@pytest.mark.parametrize(
    ("kind", "metadata_key"),
    [("mdevent", "mdevent"), ("raw_dgs_nexus", "raw_dgs")],
)
def test_native_event_leaf_is_controlled_by_collection_before_combine_enabled(
    kind: str, metadata_key: str
) -> None:
    dataset = DatasetEntry("run", None, kind=kind, parameters={"rebin": {"legacy": True}})
    collection = DatasetGroup(
        "events", datasets=[dataset], metadata={metadata_key: {"source": "runs"}}
    )
    root = DataGroup("root", subgroups=[collection])

    policy = rebin_presentation_policy(root, dataset)

    assert policy.owner is collection
    assert policy.owner_kind == "composite"
    assert policy.linked and not policy.show_editor
    assert not policy.allow_independent_recipe
    assert dataset.parameters["rebin"] == {"legacy": True}


def test_policy_module_stays_gui_independent_and_avoids_data_facade() -> None:
    source_path = Path(__file__).parents[1] / "src/nfit/project_binning_policy.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    assert not any(
        name == forbidden or name.startswith(f"{forbidden}.")
        for name in imports
        for forbidden in ("PySide6", "project_gui", "project_data")
    )


def test_dataset_binning_registry_key_is_reexported_from_authoritative_service() -> None:
    assert project_data.DATASET_REBIN_BINNINGS_KEY is (
        project_rebinning.DATASET_REBIN_BINNINGS_KEY
    )
    assert project_rebinning.DATASET_REBIN_BINNINGS_KEY == "rebin_binnings"


def test_root_data_group_can_own_enabled_composite() -> None:
    root = DataGroup("root", metadata={GROUP_COMPOSITE_KEY: {"enabled": True}})

    policy = rebin_presentation_policy(root, root)

    assert policy.owner is root
    assert policy.show_editor


def test_foreign_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="not contained"):
        rebin_presentation_policy(DataGroup("root"), DatasetEntry("foreign", None))


def test_background_explanations_distinguish_recipe_consumption() -> None:
    source_group = DatasetGroup("backgrounds")
    grouped = BackgroundSpec("group", source_group=source_group)
    direct = BackgroundSpec("direct", source_entry=DatasetEntry("background", None))
    measured = BackgroundSpec(
        "events", source_group=source_group, projection="measured_events"
    )

    assert "composite rebin recipe" in background_rebin_explanation(grouped)
    assert "does not control subtraction" in background_rebin_explanation(direct)
    assert "private view recipe is not used" in background_rebin_explanation(measured)


def test_background_explanation_uses_stable_ids_without_runtime_links() -> None:
    grouped = BackgroundSpec("group", source_group_id="group-id")
    direct = BackgroundSpec("direct", source_dataset_id="dataset-id")
    disabled = BackgroundSpec("off", source_group_id="group-id", enabled=False)

    assert "composite rebin recipe" in background_rebin_explanation(grouped)
    assert "does not control subtraction" in background_rebin_explanation(direct)
    assert "disabled" in background_rebin_explanation(disabled)


def test_dataset_owned_background_consumes_source_private_recipe() -> None:
    source = DatasetEntry("source", None)
    background = BackgroundSpec("background", source_dataset_id=source.id)
    target = DatasetEntry("target", None, backgrounds=[background])
    root = DataGroup("root", datasets=[source, target])

    detail = background_rebin_explanation(background, owner=target)
    uses = background_source_explanations(root, source)

    assert "enabled private rebin recipe" in detail
    assert len(uses) == 1
    assert "target" in uses[0]


def test_source_explanations_resolve_group_id_without_runtime_link() -> None:
    source = DatasetGroup("source")
    background = BackgroundSpec("background", source_group_id=source.id)
    target = DatasetGroup("target", backgrounds=[background])
    root = DataGroup("root", subgroups=[source, target])

    uses = background_source_explanations(root, source)

    assert len(uses) == 1
    assert "composite rebin recipe" in uses[0]
