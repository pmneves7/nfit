from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from nfit import (
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    MaskSpec,
    NfitProject,
    PointData4D,
    composite_dataset_data,
    dataset_entry_from_path,
    load_project,
    mdevent_dataset_group,
    save_dataset_file,
    save_project,
)
from nfit.composite_workflow import export_composite_recipe, replay_composite_recipe
from nfit.project_composites import _composite_scope, data_group_composite_config
from nfit.project_history import _link_project_backgrounds
from nfit.workflow import WorkflowValidationError, composite_workflow_script


def _grid(root, node=None):
    config = data_group_composite_config(_composite_scope(root, node))
    config.update(enabled=True, auto_rebin=False, minimum_coverage=0.0,
                  resolution_mode="bins", fractional=False)
    for axis in config["axes"]:
        axis.update(lower=-1.0, upper=1.0, auto_lower=False, auto_upper=False,
                    num_bins=2, mode="bins", auto_step_size=False)
    return config


def _point_source(tmp_path, name, values=(2.0, 4.0)):
    path = tmp_path / f"{name}.npz"
    points = PointData4D(H=np.array([-0.5, 0.5]), K=np.zeros(2), L=np.zeros(2),
                         E=np.zeros(2), intensity=np.array(values), sigma=np.ones(2))
    save_dataset_file(DatasetEntry(name, points), path, use_view=False)
    return dataset_entry_from_path(path)


def _equivalent(left, right):
    np.testing.assert_allclose(left.signal, right.signal, equal_nan=True)
    np.testing.assert_allclose(left.errors, right.errors, equal_nan=True)
    np.testing.assert_array_equal(left.mask, right.mask)
    np.testing.assert_array_equal(left.num_events, right.num_events)


def test_portable_nested_closure_preserves_ancestor_masks_and_omits_unrelated_sources(tmp_path):
    source = _point_source(tmp_path, "sample")
    source.scale_factor = 3.0
    source.fit_weight = 2.0
    target = DatasetGroup("target", datasets=[source])
    ancestor = DatasetGroup("ancestor", subgroups=[target],
        masks=[MaskSpec("exclude left", parameters={"H": [-0.7, -0.2]})])
    unrelated = DatasetGroup("unrelated", datasets=[DatasetEntry("source-less", None)])
    root = DataGroup("workspace", subgroups=[ancestor, unrelated],
                     lattice_parameters={"a": 4.0, "b": 4.0, "c": 4.0,
                                         "alpha": 90.0, "beta": 90.0, "gamma": 90.0})
    config = _grid(root, target)
    before = copy.deepcopy(root.metadata)
    recipe = export_composite_recipe(NfitProject([root]), root.name, node_id=target.id)
    assert root.metadata == before
    assert len(recipe["workspaces"][0]["subgroups"]) == 1
    json.dumps(recipe, allow_nan=False)
    replay_root, replay_target = replay_composite_recipe(recipe)
    assert replay_root.subgroups[0].masks[0].name == "exclude left"
    assert replay_root.lattice_parameters == root.lattice_parameters
    assert replay_target.datasets[0].id == source.id
    assert replay_target.datasets[0].fit_weight == 2.0
    _equivalent(composite_dataset_data(root, node=target, config_override=config),
                composite_dataset_data(replay_root, node=replay_target, config_override=config))
    script = composite_workflow_script(NfitProject([root]), root.name, node_id=target.id)
    assert "load_project" not in script
    namespace = {"__name__": "standalone"}
    exec(compile(script, "portable.py", "exec"), namespace)
    _equivalent(namespace["run"](), composite_dataset_data(root, node=target, config_override=config))
    namespace["SCALING"]["result_scale"] = 2.0
    edited = namespace["run"]()
    baseline = composite_dataset_data(root, node=target, config_override=config)
    np.testing.assert_allclose(edited.signal, baseline.signal * 2, equal_nan=True)


def test_portable_parent_composite_keeps_child_scientific_recipes(tmp_path):
    low = DatasetGroup("low", datasets=[_point_source(tmp_path, "low")])
    high = DatasetGroup("high", datasets=[_point_source(tmp_path, "high", (6.0, 8.0))])
    root = DataGroup("workspace", subgroups=[low, high])
    for child in (low, high):
        _grid(root, child)
    config = _grid(root)
    replay_root, replay_node = replay_composite_recipe(export_composite_recipe(NfitProject([root]), root.name))
    assert replay_node is None
    assert [node.id for node in replay_root.subgroups] == [low.id, high.id]
    _equivalent(composite_dataset_data(root, config_override=config),
                composite_dataset_data(replay_root, config_override=config))


def test_external_native_background_owner_context_survives_archive_and_portable_replay(tmp_path):
    from tests.test_mdevent import _write_mdevent

    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path, name="sample")
    reference = mdevent_dataset_group(path, name="reference")
    reference.datasets[1].enabled = False
    reference_parent = DatasetGroup("reference parent", subgroups=[reference],
        masks=[MaskSpec("reference mask", parameters={"H": [-0.1, 0.1]})])
    source_root = DataGroup("references", subgroups=[reference_parent],
        lattice_parameters={"a": 3.0, "b": 3.0, "c": 3.0,
                            "alpha": 90.0, "beta": 90.0, "gamma": 90.0})
    root = DataGroup("sample workspace", subgroups=[sample], backgrounds=[BackgroundSpec(
        "reference", source_group_id=reference.id, projection="measured_events", scale=0.25)])
    config = _grid(root, sample)
    project = NfitProject([root, source_root])
    _link_project_backgrounds(project)
    expected = composite_dataset_data(root, node=sample, config_override=config)
    recipe = export_composite_recipe(project, root.name, node_id=sample.id)
    replay_root, replay_target = replay_composite_recipe(recipe)
    replay_background = replay_root.backgrounds[0]
    assert replay_background._source_root.name == "references"
    assert replay_background._source_root.lattice_parameters == source_root.lattice_parameters
    assert replay_background._source_root.subgroups[0].masks[0].name == "reference mask"
    _equivalent(expected, composite_dataset_data(replay_root, node=replay_target, config_override=config))
    saved = tmp_path / "external.nfit"
    save_project(project, saved)
    restored = load_project(saved)
    restored_root = restored.data_groups[0]
    assert restored_root.backgrounds[0].source_group.id == reference.id
    assert restored_root.backgrounds[0]._source_root is restored.data_groups[1]
    _equivalent(expected, composite_dataset_data(restored_root, node=restored_root.subgroups[0], config_override=config))


def test_root_background_is_not_recursively_inherited_by_its_reference_and_cache_is_distinct(tmp_path):
    from nfit.project_composites import (
        _background_group_scope,
        _composite_cache_key,
        _composite_cache_signature,
    )

    sample = DatasetGroup("sample", datasets=[_point_source(tmp_path, "sample")])
    reference = DatasetGroup("reference", datasets=[_point_source(tmp_path, "reference", (1.0, 1.0))])
    root = DataGroup("workspace", subgroups=[sample, reference], backgrounds=[BackgroundSpec(
        "reference", source_group_id=reference.id, scale=0.25)])
    config = _grid(root, sample)
    _grid(root, reference)
    _link_project_backgrounds(NfitProject([root]))
    reference_scope = _background_group_scope(root, root.backgrounds[0])
    assert not reference_scope.backgrounds
    assert _composite_cache_key(reference_scope) != _composite_cache_key(_composite_scope(root, reference))
    assert _composite_cache_signature(reference_scope)
    signature = _composite_cache_signature(_composite_scope(root, sample))
    assert "background_application_version" in signature
    assert "background_application_version" not in _composite_cache_signature(
        _composite_scope(root, sample), include_backgrounds=False)
    expected = composite_dataset_data(root, node=sample, config_override=config)
    root.backgrounds[0].scale = 0.5
    actual = composite_dataset_data(root, node=sample, config_override=config)
    assert not np.allclose(expected.signal, actual.signal, equal_nan=True)
    # A reference's own self link is a genuine dependency cycle.
    reference.backgrounds = [BackgroundSpec("self", source_group_id=reference.id)]
    _link_project_backgrounds(NfitProject([root]))
    with pytest.raises(ValueError, match="dependency cycle"):
        composite_dataset_data(root, node=sample, config_override=config)


def test_portable_export_rejects_missing_replaced_and_derived_sources(tmp_path):
    entry = DatasetEntry("in memory", None)
    root = DataGroup("workspace", datasets=[entry])
    with pytest.raises(WorkflowValidationError, match="source_file"):
        composite_workflow_script(NfitProject([root]), root.name)
    source = _point_source(tmp_path, "sample")
    root.datasets = [source]
    source.metadata["derived_recipe"] = {"operation": "clone"}
    with pytest.raises(ValueError, match="live derived"):
        export_composite_recipe(NfitProject([root]), root.name)
    source.metadata.pop("derived_recipe")
    source.replace_data(source.data, source_backed=False)
    with pytest.raises(TypeError, match="replacement data"):
        export_composite_recipe(NfitProject([root]), root.name)
    with pytest.raises(ValueError, match="version"):
        replay_composite_recipe({"version": 99})


def test_hierarchical_root_background_is_applied_once(tmp_path):
    children = [DatasetGroup(name, datasets=[_point_source(tmp_path, name, values)])
                for name, values in (("low", (4.0, 6.0)), ("high", (8.0, 10.0)))]
    root = DataGroup("workspace", subgroups=children)
    for child in children:
        _grid(root, child)
    config = _grid(root)
    reference = DatasetGroup("reference", datasets=[_point_source(tmp_path, "reference", (1.0, 2.0))])
    source_root = DataGroup("references", subgroups=[reference])
    _grid(source_root, reference)
    baseline = composite_dataset_data(root, config_override=config)
    reference_data = composite_dataset_data(source_root, node=reference, config_override=config)
    root.backgrounds = [BackgroundSpec("reference", source_group_id=reference.id)]
    _link_project_backgrounds(NfitProject([root, source_root]))
    actual = composite_dataset_data(root, config_override=config)
    np.testing.assert_allclose(actual.signal, baseline.signal - reference_data.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, np.hypot(baseline.errors, reference_data.errors), equal_nan=True)
    replay_root, _ = replay_composite_recipe(export_composite_recipe(NfitProject([root, source_root]), root.name))
    _equivalent(actual, composite_dataset_data(replay_root, config_override=config))


def test_center_background_result_scale_is_applied_once(tmp_path):
    from nfit import configure_composite_scaling

    sample = DatasetGroup("sample", datasets=[_point_source(tmp_path, "sample", (8.0, 10.0))])
    reference = DatasetGroup("reference", datasets=[_point_source(tmp_path, "reference", (1.0, 2.0))])
    root = DataGroup("workspace", subgroups=[sample, reference])
    config = _grid(root, sample)
    _grid(root, reference)
    configure_composite_scaling(root, node=reference, result_scale=2.0)
    baseline = composite_dataset_data(root, node=sample, config_override=config)
    reference_data = composite_dataset_data(root, node=reference, config_override=config)
    sample.backgrounds = [BackgroundSpec("reference", source_group_id=reference.id)]
    _link_project_backgrounds(NfitProject([root]))
    actual = composite_dataset_data(root, node=sample, config_override=config)
    np.testing.assert_allclose(actual.signal, baseline.signal - reference_data.signal, equal_nan=True)
    np.testing.assert_allclose(actual.errors, np.hypot(baseline.errors, reference_data.errors), equal_nan=True)


def test_portable_composite_service_is_gui_independent():
    path = Path(__file__).parents[1] / "src/nfit/composite_workflow.py"
    tree = ast.parse(path.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ("gui" in name or name in {"workflow", "project_data"}) for name in imports)
