"""Mantid arithmetic contracts without requiring Mantid at runtime."""

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit import dgs_reduction_policy as policy


def test_effective_defaults_and_invalid_choices():
    assert policy.resolved_dgs_reduction_policies({}) == {
        "monitor_variance_policy": "mantid",
        "event_precision_policy": "mantid",
        "symmetry_variance_policy": "independent_copies",
    }
    for key in policy.resolved_dgs_reduction_policies({}):
        with pytest.raises(ValueError, match=key):
            policy.resolved_dgs_reduction_policies({key: "unknown"})
    assert "monitor_variance_policy" not in policy.resolved_dgs_reduction_policies({}, include_monitor=False)


def test_numerical_policy_version_invalidates_saved_mde_histogram_signature(tmp_path, monkeypatch):
    from nfit import mdevent_dataset_group, project_composites
    from tests.test_mdevent import _write_mdevent

    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    before = project_composites._composite_cache_signature(group)
    monkeypatch.setattr(project_composites, "DGS_REDUCTION_POLICY_VERSION", policy.DGS_REDUCTION_POLICY_VERSION + 1)
    assert project_composites._composite_cache_signature(group) != before


def test_mantid_energy_coefficients_preserve_reference_evaluation_order():
    assert policy.ENERGY_TO_K2 == 2.0721246560534285
    assert policy.ENERGY_TO_K == 0.48259644856724077
    assert 1 / policy.ENERGY_TO_K != policy.ENERGY_TO_K2


def test_policy_service_has_no_importer_or_gui_dependency():
    tree = ast.parse(Path(policy.__file__).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imports.intersection({"raw_dgs", "mdevent", "project_gui", "PySide6"})


def test_mantid_dimension_limits_round_but_nonuniform_boundaries_survive():
    regular = np.array([.1, .2, .3])
    irregular = np.array([.1, .2, .35])
    rounded, retained = policy.dgs_histogram_edges((regular, irregular))
    width = (np.float32(.3) - np.float32(.1)) / np.float32(2)
    np.testing.assert_equal(rounded, (np.arange(3, dtype=np.float32) * width + np.float32(.1)).astype(float))
    np.testing.assert_equal(retained, irregular)
    np.testing.assert_equal(policy.dgs_histogram_edges((regular,), "high_precision")[0], regular)
    tiny = np.array([100., 100.0000001, 100.0000002])
    with pytest.raises(ValueError, match="high_precision"):
        policy.dgs_histogram_edges((tiny,))
    np.testing.assert_equal(policy.dgs_histogram_edges((tiny,), "high_precision")[0], tiny)


def test_prepared_projector_reuses_affine_without_global_geometry_cache(monkeypatch):
    original = policy._mantid_affine
    calls = []
    def tracked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(policy, "_mantid_affine", tracked)
    projector = policy.prepare_dgs_event_projector(np.eye(3), np.eye(4), np.eye(3), ([0, 1],) * 4)
    for _ in range(3):
        projector(np.zeros((2, 3)), np.zeros(2))
    assert len(calls) == 1


def test_mantid_fractional_bin_transform_and_exclusive_upper_boundary():
    # 2*pi*UB=identity makes the supplied QSample coordinates already HKL.
    ub = np.eye(3) / (2 * np.pi)
    edges = (np.array([0., .5, 1.]),) * 4
    q = np.array([[0., 0., 0.], [.5, .5, .5], [1., 1., 1.]])
    energy = q[:, 0]
    coords, integer_edges = policy.dgs_event_coordinates(q, energy, ub, np.eye(4), np.eye(3), edges)
    np.testing.assert_equal(coords[:2], [[0.] * 4, [1.] * 4])
    assert np.isnan(coords[2]).all()
    for edge in integer_edges:
        np.testing.assert_equal(edge, [0, 1, 2])
    physical, unchanged = policy.dgs_event_coordinates(q, energy, ub, np.eye(4), np.eye(3), edges, "high_precision")
    np.testing.assert_allclose(physical, np.column_stack((q, energy)))
    np.testing.assert_equal(unchanged, edges)


def test_md_norm_six_digit_basis_serialization_is_observable():
    scale = 1.23456789
    ub = np.eye(3) * scale / (2 * np.pi)
    edges = (np.array([0., 1., 2.]),) * 4
    # This event lies above the exact boundary but below the six-digit basis
    # boundary. It reproduces a compatibility difference, not a tolerance.
    q = np.array([[1.234569, .2, .2]])
    coords, _ = policy.dgs_event_coordinates(q, [0.2], ub, np.eye(4), np.eye(3), edges)
    physical, _ = policy.dgs_event_coordinates(q, [.2], ub, np.eye(4), np.eye(3), edges, "high_precision")
    assert coords[0, 0] < 1
    assert physical[0, 0] > 1


def test_md_norm_six_digit_extent_serialization_changes_boundary_acceptance():
    edges = (np.linspace(.123456789, .323456789, 3),) + (np.array([0., 1.]),) * 3
    ub = np.eye(3) / (2 * np.pi)
    q = [[.1234569, .2, .2]]
    mantid, _ = policy.dgs_event_coordinates(q, [.2], ub, np.eye(4), np.eye(3), edges)
    high_precision, _ = policy.dgs_event_coordinates(q, [.2], ub, np.eye(4), np.eye(3), edges, "high_precision")
    assert mantid[0, 0] < 0  # Outside the serialized MDNorm lower extent.
    assert high_precision[0, 0] > edges[0][0]
    effective = policy.dgs_histogram_edges(edges)
    assert effective[0][0] == float(np.float32(.123457))
    assert effective[0][-1] == float(np.float32(.323457))
    # Stepped MDNorm recipes first cast dimMin to coord_t; integrated bounds
    # stream the original doubles. This distinction matters at decimal ties.
    assert policy._mantid_limits(np.linspace(.1234565, .3234565, 3))[0] == .123457
    assert policy._mantid_limits(np.array([.1234565, .3234565]))[0] == .123456


def test_nonorthogonal_transform_and_symmetry_have_expected_orientation():
    ub = np.array([[.2, .01, 0], [0, .22, .03], [.01, 0, .24]])
    basis = np.eye(4)
    basis[:3, :3] = [[1, 1, 1], [1, -1, 0], [1, 1, -2]]
    operation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    hkl = np.array([[.12, .4, .6], [-.2, .3, .1]])
    q = hkl @ (2 * np.pi * ub).T
    edges = (np.linspace(-1, 1, 11),) * 3 + (np.linspace(0, 10, 11),)
    coordinates, _ = policy.dgs_event_coordinates(q, [1., 2.], ub, basis, operation, edges)
    physical, _ = policy.dgs_event_coordinates(q, [1., 2.], ub, basis, operation, edges, "high_precision")
    np.testing.assert_allclose(coordinates[:, :3], (physical[:, :3] + 1) * 5, rtol=2e-6)
    np.testing.assert_equal(coordinates[:, 3], [1, 2])


def test_powder_policy_preserves_nonuniform_edges_and_excludes_maximum():
    edges = (np.array([0., .25, 1.]), np.array([0., 1., 2.]))
    coordinates, actual_edges = policy.dgs_powder_coordinates([.5, 1.], [.5, 2.], edges)
    np.testing.assert_equal(actual_edges, edges)
    np.testing.assert_equal(coordinates[0], [.5, .5])
    assert np.isnan(coordinates[1]).all()


def test_powder_aligned_transform_subtracts_before_multiplying():
    edges = (np.linspace(.1, .9, 18), np.linspace(-.2, .8, 9))
    values = np.array([[.3, .1], [.7, .4]], dtype=np.float32)
    coordinates, _ = policy.dgs_powder_coordinates(values[:, 0], values[:, 1], edges)
    for index, edge in enumerate(edges):
        width = (np.float32(edge[-1]) - np.float32(edge[0])) / np.float32(len(edge) - 1)
        expected = (values[:, index] - np.float32(edge[0])) * (np.float32(1.) / width)
        np.testing.assert_equal(coordinates[:, index], expected)


def test_mantid_normalization_index_uses_float32_first_bin_width():
    from nfit import mdevent

    edges = policy.dgs_histogram_edges((np.linspace(-.5, .5, 101),))
    edge = edges[0]
    points = np.array([
        edge[7], np.nextafter(np.float32(edge[7]), np.float32(-np.inf)),
        edge[99], edge[-1], 0., np.nan,
    ])
    expected = [7, 7, 99, -1, 50, -1]
    np.testing.assert_equal(policy.dgs_trajectory_bin_indices(points[:, None], edges, (100,)), expected)
    # One ulp below edge7 still lands in bin7 with Mantid's boxLength32.
    assert np.searchsorted(edge, points[1], side="right") - 1 == 6
    if mdevent._MDEVENT_NUMBA is not None:
        actual = [mdevent._MDEVENT_NUMBA._trajectory_bin_index(value, edge, True) for value in points]
        np.testing.assert_equal(actual, expected)
    assert policy.dgs_uses_mantid_trajectory_grid((np.linspace(-.5, .5, 101),))
    assert not policy.dgs_uses_mantid_trajectory_grid((np.array([0., .3, 1.]),))
    assert not policy.dgs_uses_mantid_trajectory_grid((np.linspace(-.5, .5, 101),), "high_precision")
