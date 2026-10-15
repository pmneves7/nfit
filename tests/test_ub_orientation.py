"""Numerical and metadata-only import contracts for sample UB editing."""

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

import nfit
from nfit.ub_orientation import (
    ipns_ub_from_mantid,
    lattice_orientation_from_ub,
    mantid_ub_from_ipns,
    read_isaw_ub,
    read_nexus_ub,
    ub_from_lattice_orientation,
    ub_matrix_from_metadata,
    ub_setup_from_matrix,
    ub_setup_metadata,
    validate_lattice_parameters,
    validate_ub_matrix,
    with_ub_setup_metadata,
    write_isaw_ub,
)


def _cell(**changes):
    return {"a": 4.44, "b": 4.44, "c": 6.1, "alpha": 90., "beta": 90., "gamma": 90., **changes}


def test_mantid_shiver_vectors_keep_inverse_matrix_scale():
    # With [110] perpendicular to the beam, Mantid/Shiver display
    # v=[a/sqrt(2), a/sqrt(2), 0], rather than unit HKL components.
    matrix = ub_from_lattice_orientation(_cell(), [0, 0, 1], [1, 1, 0])
    lattice, u, v = lattice_orientation_from_ub(matrix)
    np.testing.assert_allclose(u, [0, 0, 6.1], atol=1e-14)
    np.testing.assert_allclose(v, [4.44 / np.sqrt(2), 4.44 / np.sqrt(2), 0], atol=1e-14)
    np.testing.assert_allclose(list(lattice.values()), list(_cell().values()))
    _, unit_u, unit_v = lattice_orientation_from_ub(matrix, normalize_vectors=True)
    np.testing.assert_allclose(unit_u, [0, 0, 1], atol=1e-14)
    np.testing.assert_allclose(unit_v, [1 / np.sqrt(2), 1 / np.sqrt(2), 0], atol=1e-14)


def test_oblique_cell_arbitrary_rotation_round_trip():
    # Construct an independent direct basis (columns in sample coordinates),
    # with nonorthogonal axes and all orientation components nontrivial.
    direct = np.array([[3., 1., 2.], [1., 4., -.5], [.5, .8, 5.]])
    original = np.linalg.inv(direct).T
    lattice, u, v = lattice_orientation_from_ub(original)
    np.testing.assert_allclose([lattice[name] for name in ("a", "b", "c")], np.linalg.norm(direct, axis=0))
    np.testing.assert_allclose(original @ u, [0, 0, 1], atol=1e-14)
    np.testing.assert_allclose(original @ v, [1, 0, 0], atol=1e-14)
    rebuilt = ub_from_lattice_orientation(lattice, u, v)
    np.testing.assert_allclose(rebuilt, original, atol=1e-14)
    np.testing.assert_allclose(ub_from_lattice_orientation(lattice, 7 * u, 2 * v), original, atol=1e-14)
    # Units are Q/(2pi); the direct and reciprocal bases are dual without 2pi.
    np.testing.assert_allclose(original.T @ direct, np.eye(3), atol=1e-14)


def test_named_frame_conversion_preserves_native_macs_orientation():
    from nfit.macs import _oriented_ub

    cell = _cell(beta=103, gamma=83)
    u, v = np.array([1, 1, 0]), np.array([0, 0, 1])
    native = _oriented_ub(cell, u, v)
    canonical = mantid_ub_from_ipns(native)
    np.testing.assert_allclose(canonical, ub_from_lattice_orientation(cell, u, v), atol=1e-14)
    np.testing.assert_array_equal(ipns_ub_from_mantid(canonical), native)
    np.testing.assert_allclose(np.cross(native @ u, native @ v)[2], np.cross(canonical @ u, canonical @ v)[1])
    np.testing.assert_array_equal(ipns_ub_from_mantid(-canonical), -native)


def test_pure_metadata_helpers_preserve_macs_native_frame_and_source_arrays():
    from nfit.dataset import PointData4D

    canonical = ub_from_lattice_orientation(_cell(), [1, 1, 0], [0, 0, 1])
    setup = ub_setup_from_matrix(canonical)
    metadata = ub_setup_metadata(setup)
    source = PointData4D(H=[1.], K=[1.], L=[0.], E=[2.], intensity=[4.], sigma=[1.],
                         metadata={"importer": "macs_nexus"})
    view = with_ub_setup_metadata(source, metadata)
    assert view is not source
    assert source.metadata == {"importer": "macs_nexus"}
    np.testing.assert_array_equal(view.H, source.H)
    np.testing.assert_allclose(view.metadata["ub_matrix"], ipns_ub_from_mantid(canonical))
    np.testing.assert_allclose(view.metadata["ub_setup"]["ub_matrix"], canonical)
    np.testing.assert_allclose(view.metadata["rlu_to_inv_angstrom_matrix"], 2 * np.pi * ipns_ub_from_mantid(canonical))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("key", ["ub_matrix", "orientation_matrix", "rlu_to_inv_angstrom_matrix"])
def test_metadata_extraction_removes_declared_2pi_without_changing_frame(nested, key):
    matrix = np.array([[.1, .2, 0], [0, -.3, .1], [.4, 0, .2]])
    payload = {key: (2 * np.pi * matrix).tolist(), key + "_includes_2pi": True}
    metadata = {"oriented_lattice": payload} if nested else payload
    np.testing.assert_allclose(ub_matrix_from_metadata(metadata), matrix)


def test_mantid_published_isaw_example_and_explanatory_appendix(tmp_path):
    # Mantid LoadIsawUB-v1 example (2,4,5 angstrom orthorhombic cell).
    path = tmp_path / "mantid.mat"
    path.write_text("0 0.5 0\n0 0 0.25\n0.2 0 0\n2 4 5 90 90 90 40\n0 0 0 0 0 0 0\n\nThe above matrix is in IPNS convention.\n")
    matrix, lattice = read_isaw_ub(path)
    np.testing.assert_array_equal(matrix, np.diag([.5, .25, .2]))
    np.testing.assert_allclose(list(lattice.values()), [2, 4, 5, 90, 90, 90])
    _, u, v = lattice_orientation_from_ub(matrix)
    np.testing.assert_allclose(u, [0, 0, 5])
    np.testing.assert_allclose(v, [2, 0, 0])


def test_isaw_oblique_round_trip_uses_matrix_metric(tmp_path):
    original = ub_from_lattice_orientation(_cell(beta=112, gamma=76), [1, 2, 1], [-1, 1, 0])
    path = tmp_path / "oblique.mat"
    write_isaw_ub(path, original)
    actual, lattice = read_isaw_ub(path)
    np.testing.assert_allclose(actual, original, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(list(lattice.values()), list(_cell(beta=112, gamma=76).values()))
    with pytest.raises(ValueError, match="do not agree"):
        write_isaw_ub(path, original, _cell())


@pytest.mark.parametrize("root", ["mantid_workspace_1", "MDEventWorkspace/experiment0", "MDHistoWorkspace/experiment0"])
def test_processed_nexus_reads_orientation_without_signal(tmp_path, root):
    path = tmp_path / "processed.nxs"
    expected = ub_from_lattice_orientation(_cell(beta=103), [1, 0, 1], [0, 1, 0])
    with h5py.File(path, "w") as handle:
        group = handle.create_group(root + "/sample/oriented_lattice")
        group["orientation_matrix"] = expected
        # Saved scalar metadata can be rounded or stale; UB is authoritative.
        group["unit_cell_a"] = 99
        # Deliberately missing external signal: full workspace loaders would fail.
        handle["data"] = h5py.ExternalLink("absent-signal.h5", "/events")
    actual, lattice = read_nexus_ub(path)
    np.testing.assert_array_equal(actual, expected)
    assert lattice["a"] == pytest.approx(4.44)


@pytest.mark.parametrize("log_name", ["BL17:CS:CrystalAlign:UBMatrix", "BL12:Mot:UBMatrix"])
def test_raw_nexus_string_logs_use_mantid_frame_without_transpose(tmp_path, log_name):
    path = tmp_path / "raw.nxs.h5"
    expected = ub_from_lattice_orientation(_cell(alpha=83), [1, 0, 2], [0, 1, 0])
    with h5py.File(path, "w") as handle:
        handle[f"entry/DASlogs/{log_name}/value"] = np.array([f"array({expected.tolist()})"], dtype="S")
    actual, lattice = read_nexus_ub(path)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(list(lattice.values()), list(_cell(alpha=83).values()))


def test_processed_matrix_has_priority_over_stale_raw_log(tmp_path):
    path = tmp_path / "processed_with_logs.nxs"
    with h5py.File(path, "w") as handle:
        handle["entry/sample/oriented_lattice/orientation_matrix"] = np.diag([.2, .3, .4])
        handle["entry/DASlogs/UBMatrix/value"] = np.eye(3)
    matrix, _ = read_nexus_ub(path)
    np.testing.assert_array_equal(matrix, np.diag([.2, .3, .4]))


def test_nexus_ambiguity_requires_selection_and_equal_experiments_are_allowed(tmp_path):
    path = tmp_path / "multi.nxs"
    first = "/MDHistoWorkspace/experiment0/sample/oriented_lattice/orientation_matrix"
    second = "/MDHistoWorkspace/experiment1/sample/oriented_lattice/orientation_matrix"
    with h5py.File(path, "w") as handle:
        handle[first] = np.eye(3)
        handle[second] = np.eye(3)
    np.testing.assert_array_equal(read_nexus_ub(path)[0], np.eye(3))
    with h5py.File(path, "a") as handle:
        handle[second][...] = np.eye(3) * .5
    with pytest.raises(ValueError, match="Ambiguous NeXus.*matrix_path"):
        read_nexus_ub(path)
    np.testing.assert_array_equal(read_nexus_ub(path, matrix_path=second)[0], np.eye(3) * .5)


def test_raw_nexus_rejects_time_varying_orientation(tmp_path):
    path = tmp_path / "changing.nxs"
    with h5py.File(path, "w") as handle:
        handle["entry/DASlogs/UBMatrix/value"] = np.array([np.eye(3), np.eye(3) * 2])
    with pytest.raises(ValueError, match="changing UB"):
        read_nexus_ub(path)


@pytest.mark.parametrize("value", ["array([[1, 0], [0, 1]])", "array([[1,0,0],[0,1,0],[0,0,float('nan')]])", "__import__('os').system('false')", "array([[0,0,0],[0,0,0],[0,0,0]])"])
def test_raw_nexus_invalid_log_never_substitutes_identity(tmp_path, value):
    path = tmp_path / "invalid_log.nxs"
    with h5py.File(path, "w") as handle:
        handle["entry/DASlogs/UBMatrix/value"] = np.bytes_(value)
    with pytest.raises(ValueError):
        read_nexus_ub(path)


def test_nexus_rejects_large_orientation_metadata_before_reading(tmp_path):
    path = tmp_path / "large_ub.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/DASlogs/UBMatrix/value", shape=(1000000,), dtype="f8")
    with pytest.raises(ValueError, match="too large"):
        read_nexus_ub(path)


def test_nexus_missing_ub_never_substitutes_identity(tmp_path):
    path = tmp_path / "no_ub.nxs"
    with h5py.File(path, "w") as handle:
        handle["entry/data"] = np.eye(3)
        # Generic NXsample orientation_matrix can mean U rather than UB.
        handle["entry/sample/orientation_matrix"] = np.eye(3)
    with pytest.raises(ValueError, match="does not contain"):
        read_nexus_ub(path)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.ones((3, 3)), np.eye(4), np.diag([1, np.inf, 1]), np.diag([1, np.nan, 1])])
def test_invalid_matrices_rejected(matrix):
    with pytest.raises(ValueError):
        validate_ub_matrix(matrix)


def test_matrix_validation_preserves_scale_sign_and_storage_independence():
    matrix = np.diag([-.2, .3, .4])
    validated = validate_ub_matrix(matrix)
    np.testing.assert_array_equal(validated, matrix)
    assert not np.shares_memory(validated, matrix)


@pytest.mark.parametrize("scale", [1e-200, 1e-309])
def test_decomposition_rejects_unrepresentable_derived_lattice(scale):
    with pytest.raises(ValueError, match="finite|volume"):
        lattice_orientation_from_ub(np.eye(3) * scale)


@pytest.mark.parametrize("changes", [{"a": 0}, {"b": -1}, {"c": np.inf}, {"alpha": np.nan}, {"beta": 180}, {"alpha": 10, "beta": 10, "gamma": 170}, {"a": 1e308}])
def test_invalid_lattice_rejected(changes):
    with pytest.raises(ValueError):
        validate_lattice_parameters(_cell(**changes))


@pytest.mark.parametrize("u,v", [([0, 0, 0], [1, 0, 0]), ([1, 0, 0], [2, 0, 0]), ([1, 0], [0, 1, 0]), ([np.nan, 0, 1], [1, 0, 0])])
def test_invalid_orientation_rejected(u, v):
    with pytest.raises(ValueError):
        ub_from_lattice_orientation(_cell(), u, v)


def test_public_service_exports_and_gui_independent_import_direction():
    import nfit.project_ub_setup as project_setup
    import nfit.ub_orientation as service

    assert project_setup.ub_setup_metadata is service.ub_setup_metadata
    assert project_setup.with_ub_setup_metadata is service.with_ub_setup_metadata
    for name in ("lattice_orientation_from_ub", "ub_from_lattice_orientation", "read_isaw_ub", "write_isaw_ub", "read_nexus_ub", "mantid_ub_from_ipns", "ipns_ub_from_mantid"):
        assert getattr(nfit, name) is getattr(service, name)
    tree = ast.parse(Path(service.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any("gui" in name or "PySide" in name or "qt_" in name for name in imports)
