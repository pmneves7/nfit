"""Shared project orientation ownership, immutable views, and public replay."""

import copy

import numpy as np
import pytest

from nfit.cache_utils import dataset_content_signature
from nfit.composite_workflow import export_composite_recipe, replay_composite_recipe
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from nfit.project_data import dataset_for_slice_viewer
from nfit.project_io import NfitProject
from nfit.project_ub_setup import get_ub_setup, set_ub_setup
from nfit.reduction_recipes import (
    effective_reduction_config,
    export_reduction_recipe,
    reduction_recipe_script,
    replay_reduction_recipe,
    set_reduction_settings,
)
from nfit.ub_orientation import (
    ipns_ub_from_mantid,
    lattice_orientation_from_ub,
    mantid_ub_from_ipns,
    ub_from_lattice_orientation,
)
from tests.project_gui_test_support import _tiny_mdhisto_data
from tests.test_reduction_recipes import _group


def test_matrix_is_authoritative_and_get_is_pure():
    data = _tiny_mdhisto_data(4)
    dataset = DatasetEntry("histogram", data)
    root = DataGroup("workspace", datasets=[dataset])
    before = copy.deepcopy(root.metadata)
    assert get_ub_setup(root, dataset) is None
    assert root.metadata == before
    matrix = np.diag([0.2, 0.25, 0.1])
    canonical = set_ub_setup(root, dataset, {"ub_matrix": matrix, "u": [99, 1, 2]})
    lattice, u, v = lattice_orientation_from_ub(matrix)
    assert canonical["lattice_parameters"] == lattice
    np.testing.assert_allclose(canonical["u"], u)
    np.testing.assert_allclose(canonical["v"], v)
    canonical["ub_matrix"][0][0] = 999
    assert get_ub_setup(root, dataset)["ub_matrix"][0][0] == 0.2
    assert dataset.data is data and data.metadata == {}
    with pytest.raises(ValueError, match="belong"):
        set_ub_setup(root, DatasetEntry("foreign", None), ub=matrix)
    saved = copy.deepcopy(dataset.metadata)
    with pytest.raises(ValueError):
        set_ub_setup(root, dataset, ub=np.zeros((3, 3)))
    assert dataset.metadata == saved


@pytest.mark.parametrize("key,value", [
    ("ub_matrix", np.eye(3).tolist()),
    ("rlu_to_inv_angstrom_matrix", np.eye(3).tolist()),
    ("oriented_lattice", {"ub_matrix": np.eye(3).tolist()}),
    ("lattice_parameters", {"a": 4, "b": 5, "c": 6}),
    ("ub_matrix_includes_2pi", True),
])
def test_direct_orientation_metadata_participates_in_content_signature(key, value):
    dataset = DatasetEntry("histogram", _tiny_mdhisto_data(4))
    before = dataset_content_signature(dataset)
    dataset.metadata[key] = value
    assert dataset_content_signature(dataset) != before


@pytest.mark.parametrize("nested,full_q", [(False, False), (False, True), (True, False), (True, True)])
def test_adopted_generic_orientation_uses_shared_metadata_scale_and_precedence(nested, full_q):
    from nfit.project_imports import _adopt_imported_crystal, _dataset_ub_for_editor

    canonical = np.diag([0.2, 0.3, 0.4])
    values = {"ub_matrix": (2 * np.pi * canonical).tolist(), "ub_matrix_includes_2pi": True}
    if full_q:
        values["rlu_to_inv_angstrom_matrix"] = (2 * np.pi * canonical * 2).tolist()
        canonical *= 2
    metadata = {"oriented_lattice": values} if nested else values
    dataset = DatasetEntry("imported", _tiny_mdhisto_data(4).with_updates(metadata=metadata))
    group = DatasetGroup("ordinary", datasets=[dataset])
    root = DataGroup("workspace", subgroups=[group])
    _adopt_imported_crystal(root, dataset, group)
    np.testing.assert_allclose(_dataset_ub_for_editor(metadata), canonical)
    np.testing.assert_allclose(get_ub_setup(root, group)["ub_matrix"], canonical)
    assert group.metadata["ub_setup"]["lattice_parameters"] == lattice_orientation_from_ub(canonical)[0]


@pytest.mark.parametrize("format_name", ["raw-direct-geometry-nexus", "corelli-correlation-nexus", "mantid-mdevent"])
def test_native_child_updates_shared_transform_and_preserves_lab_cache(tmp_path, format_name):
    group = _group(tmp_path, format_name)
    root = DataGroup("workspace", subgroups=[group])
    dataset = group.datasets[0]
    cache = {"identity": "lab-events"}
    dataset.metadata["raw_dgs_reduction_cache"] = cache
    old_lattice = {"a": 7, "b": 7, "c": 7, "alpha": 90, "beta": 90, "gamma": 90}
    root.lattice_parameters = old_lattice.copy()
    setup = set_ub_setup(root, dataset, ub=np.eye(3) * 0.25)
    assert effective_reduction_config(group)["ub_matrix"] == setup["ub_matrix"]
    assert get_ub_setup(root, dataset) == setup
    assert "ub_setup" not in dataset.metadata
    assert dataset.metadata["raw_dgs_reduction_cache"] is cache
    assert root.lattice_parameters == old_lattice
    assert group.metadata["composite"]["stale"]
    # A later matrix-only script edit must precede an old saved dialog setup.
    set_reduction_settings(group, {"ub_matrix": (0.5 * np.eye(3)).tolist()})
    np.testing.assert_allclose(get_ub_setup(root, dataset)["ub_matrix"], np.eye(3) * 0.5)


def test_explicit_dataset_metadata_reaches_masks_and_viewer_without_changing_hkl():
    data = _tiny_mdhisto_data(4)
    original = {"ub_matrix": (np.eye(3) * 0.1).tolist(),
                "rlu_to_inv_angstrom_matrix": (np.eye(3) * 0.2 * np.pi).tolist(),
                "ub_matrix_includes_2pi": True,
                "oriented_lattice": {"ub_matrix": np.eye(3).tolist()}}
    data = data.with_updates(metadata=original)
    dataset = DatasetEntry("histogram", data, data_type="single_crystal_inelastic")
    root = DataGroup("workspace", datasets=[dataset])
    # The H center is 0.5. Original |Q| is 0.1*pi, edited |Q| is pi.
    dataset.masks.append(MaskSpec("low Q", type="energy_q_range", parameters={"q_modulus": [0, 1]}))
    first = dataset_for_slice_viewer(dataset)
    signature = dataset_content_signature(dataset)
    setup = set_ub_setup(root, dataset, ub=np.eye(3))
    second = dataset_for_slice_viewer(dataset)
    assert dataset_content_signature(dataset) != signature
    np.testing.assert_allclose(second.metadata["ub_matrix"], setup["ub_matrix"])
    np.testing.assert_allclose(second.metadata["rlu_to_inv_angstrom_matrix"], 2 * np.pi * np.eye(3))
    assert not second.metadata["ub_matrix_includes_2pi"]
    assert dataset.data is data
    assert dataset.data.metadata is data.metadata
    np.testing.assert_array_equal(second.signal, data.signal)
    np.testing.assert_array_equal(second.axes[0].values, data.axes[0].values)
    assert first.mask[0, 0] and not second.mask[0, 0]


def test_macs_shared_editor_reimports_coordinates_and_supports_automatic(tmp_path):
    from nfit.project_imports import import_dataset_paths
    from tests.test_macs import _write_macs_nexus

    path = _write_macs_nexus(tmp_path / "macs.nxs.ng0")
    root = DataGroup("workspace")
    import_dataset_paths(root, [path], data_type="single_crystal_inelastic")
    group = root.subgroups[0]
    dataset = group.datasets[0]
    original = dataset.data.H.copy()
    original_native = np.asarray(dataset.data.metadata["ub_matrix"])
    source_lattice = dataset.data.metadata["lattice_parameters"]
    source_u = dataset.data.metadata["orientation_u"]
    source_v = dataset.data.metadata["orientation_v"]
    source_calculated = ub_from_lattice_orientation(source_lattice, source_u, source_v)
    np.testing.assert_allclose(get_ub_setup(root, dataset)["ub_matrix"], source_calculated, atol=1e-12)
    # Calculating the original source vectors and applying in the editor is
    # scientifically identical to the source-derived MACS orientation.
    set_ub_setup(root, dataset, ub=source_calculated)
    unchanged = dataset_for_slice_viewer(dataset)
    np.testing.assert_allclose(unchanged.H, original, atol=1e-12)
    np.testing.assert_allclose(unchanged.metadata["ub_matrix"], original_native, atol=1e-12)
    matrix = np.asarray(get_ub_setup(root, dataset)["ub_matrix"]) * 2
    setup = set_ub_setup(root, dataset, ub=matrix)
    assert dataset.data is None
    np.testing.assert_allclose(dataset.metadata["import_options"]["ub_matrix"], ipns_ub_from_mantid(matrix))
    view = dataset_for_slice_viewer(dataset)
    np.testing.assert_allclose(view.H, original / 2, atol=1e-12)
    np.testing.assert_allclose(view.metadata["ub_matrix"], original_native * 2, atol=1e-12)
    assert view.metadata["lattice_parameters"]["a"] == pytest.approx(source_lattice["a"] / 2)
    np.testing.assert_allclose(mantid_ub_from_ipns(view.metadata["ub_matrix"]), setup["ub_matrix"])
    set_ub_setup(root, group, ub=None)
    assert dataset.metadata["import_options"]["ub_matrix"] is None
    assert "ub_setup" not in group.metadata
    automatic = dataset_for_slice_viewer(dataset)
    np.testing.assert_allclose(automatic.H, original)


def test_native_matrix_only_reduction_recipe_reconstructs_setup(tmp_path):
    from nfit.mdevent import mdevent_dataset_group
    from tests.test_mdevent import _write_mdevent

    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    group = mdevent_dataset_group(path)
    root = DataGroup("workspace", subgroups=[group])
    expected = set_ub_setup(root, group, ub=np.diag([0.2, 0.3, 0.4]))
    recipe = export_reduction_recipe(group)
    restored = replay_reduction_recipe(recipe)
    restored_root = DataGroup("replayed", subgroups=[restored])
    assert get_ub_setup(restored_root, restored) == expected
    namespace = {}
    exec(reduction_recipe_script(group, group_variable="replayed"), namespace)
    scripted = namespace["replayed"]
    assert get_ub_setup(DataGroup("scripted", subgroups=[scripted]), scripted) == expected


def test_generic_metadata_and_scope_overrides_survive_complete_workflow(tmp_path):
    from nfit.project_composites import composite_dataset_data, refresh_composite_dataset
    from tests.test_composite_workflow import _grid, _point_source

    dataset = _point_source(tmp_path, "source")
    group = DatasetGroup("ordinary", datasets=[dataset])
    root = DataGroup("workspace", subgroups=[group])
    _grid(root, group)
    dataset_setup = set_ub_setup(root, dataset, ub=np.eye(3) * 0.5)
    scope_setup = set_ub_setup(root, group, ub=np.eye(3) * 0.25)
    recipe = export_composite_recipe(NfitProject([root]), root.name, node_id=group.id)
    replay_root, replay_group = replay_composite_recipe(recipe)
    assert get_ub_setup(replay_root, replay_group.datasets[0]) == dataset_setup
    assert get_ub_setup(replay_root, replay_group) == scope_setup
    actual = composite_dataset_data(replay_root, node=replay_group)
    np.testing.assert_allclose(actual.metadata["ub_matrix"], scope_setup["ub_matrix"])
    cached = refresh_composite_dataset(replay_root, node=replay_group)
    np.testing.assert_allclose(cached.metadata["ub_matrix"], scope_setup["ub_matrix"])
    changed = set_ub_setup(replay_root, replay_group, ub=np.eye(3) * 0.125)
    updated = refresh_composite_dataset(replay_root, node=replay_group)
    np.testing.assert_allclose(updated.metadata["ub_matrix"], changed["ub_matrix"])


def test_workspace_default_does_not_replace_native_collection_orientation(tmp_path):
    group = _group(tmp_path)
    root = DataGroup("workspace", subgroups=[group])
    expected = get_ub_setup(root, group)
    set_ub_setup(root, root, ub=np.eye(3) * 0.01)
    assert get_ub_setup(root, group) == expected
    ordinary = DatasetEntry("unknown", None)
    root.datasets.append(ordinary)
    assert get_ub_setup(root, ordinary) is None


def test_unknown_automatic_native_orientation_does_not_invent_workspace_default(tmp_path):
    from nfit.macs_reduction import MACS_REDUCTION_FORMAT

    dataset = DatasetEntry("lazy", None, metadata={"importer": "macs_nexus", "source_file": str(tmp_path / "missing.nxs.ng0")})
    group = DatasetGroup("MACS", datasets=[dataset], metadata={
        "macs": {"format": MACS_REDUCTION_FORMAT, "stream": "spec", "ub_matrix": None}})
    root = DataGroup("workspace", subgroups=[group])
    set_ub_setup(root, root, ub=np.eye(3) * 0.1)
    assert get_ub_setup(root, dataset) is None
    set_ub_setup(root, group, ub=np.eye(3) * 0.2)
    set_reduction_settings(group, {"ub_matrix": None})
    assert get_ub_setup(root, dataset) is None
    assert set_ub_setup(root, group, ub=None) is None
    assert get_ub_setup(root, dataset) is None


def test_required_native_ub_cannot_reset_to_automatic(tmp_path):
    group = _group(tmp_path)
    root = DataGroup("workspace", subgroups=[group])
    before = copy.deepcopy(group.metadata)
    with pytest.raises(ValueError, match="Only MACS"):
        set_ub_setup(root, group, ub=None)
    assert group.metadata == before


def test_generic_override_survives_project_save_and_lazy_source_reload(tmp_path):
    from nfit import load_project, save_project
    from tests.test_composite_workflow import _point_source

    dataset = _point_source(tmp_path, "source")
    root = DataGroup("workspace", datasets=[dataset])
    expected = set_ub_setup(root, dataset, ub=np.eye(3) * 0.25)
    path = tmp_path / "project.nfit"
    save_project(NfitProject([root]), path)
    restored_root = load_project(path).data_groups[0]
    restored = restored_root.datasets[0]
    assert get_ub_setup(restored_root, restored) == expected
    view = dataset_for_slice_viewer(restored)
    np.testing.assert_allclose(view.metadata["ub_matrix"], expected["ub_matrix"])


@pytest.mark.parametrize("family", ["raw", "mdevent"])
def test_native_symmetry_uses_own_cell_and_workspace_edit_preserves_cache(tmp_path, monkeypatch, family):
    from nfit import project_composites as composites
    from nfit.mdevent import mdevent_dataset_group
    from nfit.raw_dgs import raw_dgs_dataset_group
    from nfit.symmetry import SymmetrySpec, symmetry_config
    from tests.test_composite_workflow import _grid
    from tests.test_mdevent import _write_mdevent
    from tests.test_raw_dgs import _write_raw_dgs

    path = tmp_path / "events.nxs"
    if family == "raw":
        _write_raw_dgs(path)
        native = raw_dgs_dataset_group([path])
    else:
        _write_mdevent(path)
        native = mdevent_dataset_group(path)
    root = DataGroup("workspace", subgroups=[native])
    set_ub_setup(root, native, ub=np.eye(3) * 0.25)
    scope = composites._composite_scope(root, native)
    config = _grid(root, native)
    config["symmetry"] = symmetry_config(SymmetrySpec("generators", "rotate(order=2, axis=[1,1,0])"))
    captured = []
    name = "bin_raw_dgs_group" if family == "raw" else "bin_mdevent_group"
    original = getattr(composites, name)

    def record(*args, **kwargs):
        captured.append(np.asarray(kwargs["symmetry_operations"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(composites, name, record)
    composites.composite_dataset_data(root, node=native)
    first = captured[-1]
    expected = composites._rebin_symmetry_matrices(config, lattice_orientation_from_ub(np.eye(3) * 0.25)[0])
    np.testing.assert_allclose(first, expected, atol=1e-12)
    before = composites._composite_cache_signature(scope)
    set_ub_setup(root, root, ub=np.diag([0.1, 0.2, 0.3]))
    assert composites._composite_cache_signature(scope) == before
    composites.composite_dataset_data(root, node=native)
    np.testing.assert_allclose(captured[-1], first, atol=1e-12)
    # Matrix-only script edits must change the actual symmetry operation,
    # even while a prior dialog setup and workspace cell remain saved.
    matrix = np.diag([0.2, 0.3, 0.4])
    set_reduction_settings(native, {"ub_matrix": matrix.tolist()})
    composites.composite_dataset_data(root, node=native)
    expected_changed = composites._rebin_symmetry_matrices(config, lattice_orientation_from_ub(matrix)[0])
    np.testing.assert_allclose(captured[-1], expected_changed, atol=1e-12)
    assert not np.allclose(first, captured[-1])
    assert composites._composite_cache_signature(scope) != before
