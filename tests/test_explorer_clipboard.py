from __future__ import annotations

import pytest

from nfit.analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
from nfit.pipeline import (
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    ModelComponentSpec,
    PlotEntry,
    PlotSourceRef,
)
from nfit.project_clipboard import (
    clone_workspace,
    make_payload,
    paste_capability,
    paste_payload,
)
from nfit.project_history import restore_data_group_state, snapshot_data_group_state
from nfit.project_io import NfitProject, _project_from_dict, _project_to_dict


def test_workspace_clone_remaps_ids_references_and_live_derived_recipe():
    source_data = None
    raw = DatasetEntry("raw", source_data)
    subgroup = DatasetGroup("scope", datasets=[raw])
    analysis = AnalysisEntry(
        "clone", "dataset_clone", [raw.id, f"group-composite:{subgroup.id}"], {}
    )
    derived = DatasetEntry(
        "derived",
        None,
        kind="derived_recipe",
        metadata={
            "derived_from_analysis": {"analysis_id": analysis.id},
            "derived_recipe": {
                "analysis_id": analysis.id,
                "input_source_ids": [raw.id, f"group-composite:{subgroup.id}"],
            },
        },
    )
    subgroup.datasets.append(derived)
    analysis.result = AnalysisResultRecord(
        "recipe", {raw.id: "fingerprint"},
        [AnalysisOutputRef("clone", "derived", "dataset", dataset_id=derived.id)],
        "success", "now",
    )
    fit = FitTimelineEntry("fit")
    plot = PlotEntry(
        "plot",
        sources=[PlotSourceRef(raw.id, fit.id)],
        settings={
            "source_composite": {"dataset_group_id": subgroup.id},
            "source_rebin_configs": {raw.id: {"enabled": True}},
        },
    )
    subgroup.backgrounds.append(
        BackgroundSpec("background", raw.id, source_group_id=subgroup.id)
    )
    source = DataGroup(
        "Workspace", subgroups=[subgroup], fits=[fit], analyses=[analysis], plots=[plot]
    )
    fit.snapshot = snapshot_data_group_state(source)

    cloned = clone_workspace(source, ["Workspace"])
    cloned_scope = cloned.subgroups[0]
    cloned_raw, cloned_derived = cloned_scope.datasets
    cloned_analysis = cloned.analyses[0]

    assert cloned.name == "Workspace1"
    assert cloned_raw.data is source_data
    assert cloned_raw.id != raw.id
    assert cloned_analysis.id != analysis.id
    assert cloned_analysis.input_dataset_ids == [
        cloned_raw.id, f"group-composite:{cloned_scope.id}"
    ]
    assert cloned_analysis.result.outputs[0].dataset_id == cloned_derived.id
    assert cloned_derived.metadata["derived_from_analysis"]["analysis_id"] == cloned_analysis.id
    assert cloned_derived.metadata["derived_recipe"]["input_source_ids"] == [
        cloned_raw.id, f"group-composite:{cloned_scope.id}"
    ]
    assert cloned_derived._derived_owner_group is cloned
    assert cloned_scope.backgrounds[0].source_dataset_id == cloned_raw.id
    assert cloned_scope.backgrounds[0].source_group_id == cloned_scope.id
    assert cloned.plots[0].sources[0].dataset_id == cloned_raw.id
    assert cloned.plots[0].sources[0].fit_id == cloned.fits[0].id
    assert cloned.plots[0].settings["source_composite"]["dataset_group_id"] == cloned_scope.id
    assert set(cloned.plots[0].settings["source_rebin_configs"]) == {cloned_raw.id}

    cloned_scope.backgrounds.clear()
    restore_data_group_state(cloned, cloned.fits[0].snapshot)
    restored_background = cloned_scope.backgrounds[0]
    assert restored_background.source_entry is cloned_raw
    assert restored_background.source_group is cloned_scope

    reopened = _project_from_dict(_project_to_dict(NfitProject([cloned]))).data_groups[0]
    reopened_scope = reopened.subgroups[0]
    reopened_raw, reopened_derived = reopened_scope.datasets
    assert reopened.analyses[0].input_dataset_ids == [
        reopened_raw.id, f"group-composite:{reopened_scope.id}"
    ]
    assert reopened_derived.metadata["derived_recipe"]["input_source_ids"] == [
        reopened_raw.id, f"group-composite:{reopened_scope.id}"
    ]
    assert reopened_derived._derived_owner_group is reopened
    assert reopened.plots[0].settings["source_composite"]["dataset_group_id"] == reopened_scope.id
    assert set(reopened.plots[0].settings["source_rebin_configs"]) == {reopened_raw.id}


def test_paste_dataset_batch_shares_data_and_assigns_unique_names():
    numerical_payload = object()
    source = DatasetEntry("scan", numerical_payload, metadata={"nested": {"value": 1}})
    destination = DataGroup("Target", datasets=[DatasetEntry("scan", None)])
    payload = make_payload("dataset_page", [source, source])

    result = paste_payload(
        payload, target_role="datasets", data_group=destination, dataset_node=destination
    )

    assert [item.name for item in destination.datasets] == ["scan", "scan1", "scan2"]
    assert all(item.data is numerical_payload for item in result.items)
    assert len({item.id for item in result.items}) == 2
    result.items[0].metadata["nested"]["value"] = 2
    assert source.metadata["nested"]["value"] == 1

    source.metadata["nested"]["value"] = 99
    second_destination = DataGroup("Second")
    second = paste_payload(
        payload, target_role="datasets", data_group=second_destination
    ).items[0]
    assert second.metadata["nested"]["value"] == 1
    second.metadata["nested"]["value"] = 3
    third_destination = DataGroup("Third")
    third = paste_payload(
        payload, target_role="datasets", data_group=third_destination
    ).items[0]
    assert third.metadata["nested"]["value"] == 1


def test_repeated_subtree_paste_uniquifies_every_nested_name():
    existing = DatasetGroup(
        "Runs", datasets=[DatasetEntry("scan", None)],
        subgroups=[DatasetGroup("Nested", datasets=[DatasetEntry("child", None)])],
    )
    group = DataGroup("Target", subgroups=[existing])
    payload = make_payload("dataset_group", [existing])

    paste_payload(payload, target_role="datasets", data_group=group)
    paste_payload(payload, target_role="datasets", data_group=group)

    group_names = [item.name for item in group.iter_subgroups()]
    dataset_names = [item.name for item in group.iter_datasets()]
    assert len(group_names) == len(set(group_names))
    assert len(dataset_names) == len(set(dataset_names))
    assert group_names == ["Runs", "Nested", "Runs1", "Nested1", "Runs2", "Nested2"]
    assert dataset_names == ["scan", "child", "scan1", "child1", "scan2", "child2"]


def test_model_batch_rename_remaps_constraints_once_without_touching_metadata():
    target = DataGroup(
        "Target",
        models={
            "A": ModelComponentSpec("A"),
            "A1": ModelComponentSpec("A1"),
        },
    )
    first = ModelComponentSpec(
        "A",
        constraints=[{"parameter": "A.x", "expression": "`A1.y` + A.x"}],
        metadata={"path": "notes/A.reference.txt"},
    )
    second = ModelComponentSpec(
        "A1", constraints=[{"parameter": "A1.y", "reference": "A.x"}]
    )
    pasted = paste_payload(
        make_payload("models", [first, second]), target_role="models", data_group=target
    ).items

    assert [item.name for item in pasted] == ["A2", "A11"]
    assert pasted[0].constraints == [
        {"parameter": "A2.x", "expression": "`A11.y` + A2.x"}
    ]
    assert pasted[1].constraints == [{"parameter": "A11.y", "reference": "A2.x"}]
    assert pasted[0].metadata["path"] == "notes/A.reference.txt"


def test_cross_workspace_reference_paste_is_rejected_before_mutation():
    source_dataset = DatasetEntry("derived", None, metadata={
        "derived_from_analysis": {"analysis_id": "missing-analysis"}
    })
    destination = DataGroup("Target")
    payload = make_payload("dataset", [source_dataset])

    with pytest.raises(ValueError, match="does not contain the analysis"):
        paste_payload(payload, target_role="datasets", data_group=destination)
    assert destination.datasets == []

    analysis = AnalysisEntry("analysis", "dataset_clone", ["missing-dataset"], {})
    analysis_payload = make_payload("analysis", [analysis])
    with pytest.raises(ValueError, match="does not contain every dataset"):
        paste_payload(analysis_payload, target_role="analyses", data_group=destination)
    assert destination.analyses == []


def test_pasted_analysis_is_a_fresh_recipe_and_current_fit_becomes_history():
    dataset = DatasetEntry("scan", None)
    group = DataGroup("Target", datasets=[dataset])
    analysis = AnalysisEntry("recipe", "dataset_clone", [dataset.id], {})
    analysis.result = AnalysisResultRecord(
        "old", {}, [AnalysisOutputRef("clone", "old", "dataset", dataset_id=dataset.id)],
        "success", "then",
    )
    pasted_analysis = paste_payload(
        make_payload("analysis", [analysis]), target_role="analyses", data_group=group
    ).items[0]
    assert pasted_analysis.id != analysis.id
    assert pasted_analysis.result is None

    current = FitTimelineEntry(
        "Current state", kind="current", snapshot={"datasets": [{"name": "scan"}]}
    )
    pasted_fit = paste_payload(
        make_payload("fit", [current]), target_role="fits", data_group=group
    ).items[0]
    assert pasted_fit.kind == "result"
    assert pasted_fit.snapshot == current.snapshot


def test_plot_and_fit_preflight_validate_all_stable_references():
    dataset = DatasetEntry("scan", None)
    source_group = DatasetGroup("scope")
    fit = FitTimelineEntry(
        "saved",
        snapshot={
            "datasets": [{
                "name": "scan",
                "backgrounds": [{"source_dataset_id": "other", "source_group_id": source_group.id}],
            }],
            "models": [],
        },
    )
    target = DataGroup("Target", datasets=[dataset], subgroups=[source_group])
    fit_payload = make_payload("fit", [fit])
    assert not paste_capability(fit_payload, "fits", data_group=target).allowed

    plot = PlotEntry(
        "plot",
        settings={
            "source_composite": {"dataset_group_id": "missing-group"},
            "source_rebin_configs": {dataset.id: {}},
        },
    )
    plot_payload = make_payload("plot", [plot])
    assert not paste_capability(plot_payload, "plots", data_group=target).allowed

    plot.settings["source_composite"]["dataset_group_id"] = source_group.id
    plot.settings["source_rebin_configs"] = {"missing-dataset": {}}
    plot_payload = make_payload("plot", [plot])
    assert not paste_capability(plot_payload, "plots", data_group=target).allowed


def test_capability_matrix_includes_collections_and_protects_outputs():
    dataset = DatasetEntry("scan", None)
    page = make_payload("dataset_page", [dataset])
    assert paste_capability(page, "dataset_group").allowed
    assert paste_capability(page, "models").allowed is False
    with pytest.raises(ValueError, match="references owned"):
        make_payload("analysis_output", [object()])
    assert make_payload("group_mask", [project_mask("mask")]).kind == "mask"
    assert make_payload("group_background", [BackgroundSpec("bg")]).kind == "background"
    with pytest.raises(ValueError, match="fitting-session"):
        make_payload("models", [object()])


def project_mask(name):
    from nfit.pipeline import MaskSpec

    return MaskSpec(name)


def test_gui_copies_unexpanded_lazy_page_and_structural_collection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit import project_gui
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import NfitProjectExplorer

    monkeypatch.setattr(project_gui, "TREE_DATASET_COMPACT_THRESHOLD", 2)
    monkeypatch.setattr(project_gui, "TREE_DATASET_PAGE_SIZE", 2)
    source = DataGroup(
        "Source",
        datasets=[DatasetEntry(f"scan {index}", None) for index in range(4)],
        models={"constant": ModelComponentSpec("constant")},
    )
    target = DataGroup("Target")
    explorer = NfitProjectExplorer(NfitProject([source, target]))

    first_page = explorer.tree.topLevelItem(0).child(0).child(0)
    assert explorer._objects_for_item(first_page)[4] == "dataset_page"
    assert first_page.child(0).isDisabled()  # still contains only the lazy placeholder
    explorer.tree.setCurrentItem(first_page)
    explorer.copy_selected()
    target_datasets = explorer.tree.topLevelItem(1).child(0)
    explorer.tree.setCurrentItem(target_datasets)
    explorer.paste_into_selection()
    assert [item.name for item in target.datasets] == ["scan 0", "scan 1"]

    source_models = explorer.tree.topLevelItem(0).child(1)
    explorer.tree.setCurrentItem(source_models)
    explorer.copy_selected()
    target_models = explorer.tree.topLevelItem(1).child(1)
    explorer.tree.setCurrentItem(target_models)
    assert explorer._can_paste_into_role("models", None)
    explorer.paste_into_selection()
    assert list(target.models) == ["constant"]
