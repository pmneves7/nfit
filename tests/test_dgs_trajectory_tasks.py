"""Exact exposure-work pooling, without an external reduction-engine oracle."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit.dgs_trajectory_tasks import pool_trajectory_tasks


def _row(*, matrix=None, energy=60.0, bounds=(-57.0, 57.0), charge=1.0, tail="geometry"):
    return (np.eye(3) if matrix is None else matrix, energy, np.asarray(bounds), charge, tail)


def test_literal_affine_and_clipped_energy_window_pool_without_changing_sources():
    first = _row(charge=2.0)
    second = _row(bounds=(-58.0, 58.0), charge=3.0)
    different = _row(matrix=np.eye(3) + 1e-14, charge=7.0)
    original = [first, different, second]
    edges = np.array([-0.25, 0.0, 25.0, 50.25])
    result = pool_trajectory_tasks(original, edges, checked_shared_geometry=True)
    assert len(result) == 2
    assert result[0][0] is first[0]
    assert result[0][2] is first[2]
    assert result[0][3:] == (5.0, "geometry")
    assert result[1] is different
    assert original == [first, different, second]
    assert first[3] == 2.0 and second[3] == 3.0


def test_unchecked_or_mixed_geometry_collection_is_not_pooled():
    rows = [_row(charge=2.0, tail="sample"), _row(charge=3.0, tail="other_instrument")]
    result = pool_trajectory_tasks(rows, [-0.25, 50.25])
    assert all(actual is expected for actual, expected in zip(result, rows, strict=True))


def test_per_run_energy_disjoint_windows_and_sub_ulp_affine_identity_stay_distinct():
    matrix = np.eye(3)
    changed = matrix.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    rows = [
        _row(),
        _row(energy=np.nextafter(60.0, np.inf)),
        _row(bounds=(0.0, 5.0)),
        _row(bounds=(10.0, 15.0)),
        _row(matrix=changed),
        _row(matrix=matrix.astype(np.float32)),
    ]
    result = pool_trajectory_tasks(rows, [-0.25, 50.25], checked_shared_geometry=True)
    assert len(result) == len(rows)
    assert all(actual is expected for actual, expected in zip(result, rows, strict=True))


@pytest.mark.parametrize("charge", [0.0, -1.0, np.inf, np.nan])
def test_nonpositive_or_nonfinite_charges_do_not_pool(charge):
    rows = [_row(charge=charge), _row(charge=charge)]
    result = pool_trajectory_tasks(rows, [-0.25, 50.25], checked_shared_geometry=True)
    assert len(result) == 2 and result[0] is rows[0] and result[1] is rows[1]


@pytest.mark.parametrize("bounds", [(np.nan, 57.0), (-57.0, np.inf), (-57.0,)])
def test_nonfinite_or_malformed_source_metadata_stays_on_existing_reducer_path(bounds):
    rows = [_row(bounds=bounds), _row(bounds=bounds)]
    result = pool_trajectory_tasks(rows, [-0.25, 50.25], checked_shared_geometry=True)
    assert len(result) == 2 and result[0] is rows[0]


def test_charge_overflow_retains_original_tasks():
    rows = [_row(charge=1e308), _row(charge=1e308)]
    assert len(pool_trajectory_tasks(rows, [-0.25, 50.25], checked_shared_geometry=True)) == 2


@pytest.mark.parametrize("edges", [[50.25, -0.25], [-0.25, np.nan, 50.25], [-0.25]])
def test_invalid_grid_metadata_does_not_alter_tasks(edges):
    rows = [_row(), _row()]
    result = pool_trajectory_tasks(rows, edges, checked_shared_geometry=True)
    assert len(result) == 2 and result[0] is rows[0]


@pytest.mark.parametrize("mantid_precision", [False, True])
def test_actual_trajectory_kernel_preserves_nonzero_normalization_and_support(mantid_precision):
    pytest.importorskip("numba")
    from nfit._mdevent_numba import trajectory_normalization_accumulator

    edges = (
        np.linspace(-2.0, 2.0, 9), np.linspace(-2.0, 2.0, 11),
        np.linspace(-2.0, 2.0, 7), np.linspace(-0.25, 50.25, 12),
    )
    shape = np.asarray([len(edge) - 1 for edge in edges], dtype=np.int64)
    rows = [
        _row(matrix=np.eye(3) * 0.3, charge=0.7),
        _row(matrix=np.eye(3) * 0.3, bounds=(-58.0, 58.0), charge=1.3),
        _row(matrix=np.eye(3) * 0.3, energy=61.0, charge=0.9),
        _row(matrix=np.eye(3) * 0.3, bounds=(0.0, 5.0), charge=2.1),
        _row(matrix=np.eye(3) * 0.3, bounds=(10.0, 15.0), charge=0.2),
    ]
    pooled = pool_trajectory_tasks(rows, edges[-1], checked_shared_geometry=True)
    assert len(pooled) == 4
    theta = np.array([0.25, 0.5, 0.9, 1.4, 2.0])
    phi = np.array([-1.0, 0.1, 0.5, 1.5, 2.8])
    solid = np.array([0.0, 0.7, 1.0, 1.2, 2.0])

    def reduce(tasks):
        accumulator = trajectory_normalization_accumulator(*edges, shape, workers=1)
        accumulator.accumulate(
            theta, phi, solid, np.asarray([row[0] for row in tasks]),
            np.asarray([row[1] for row in tasks]), np.asarray([row[2] for row in tasks]),
            np.asarray([row[3] for row in tasks]), *edges, shape, mantid_precision,
        )
        return accumulator.result()

    reference, actual = reduce(rows), reduce(pooled)
    assert np.any(reference > 0)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(actual > 0, reference > 0)


def test_task_pooling_service_has_no_gui_or_reducer_dependencies():
    path = Path(__file__).parents[1] / "src/nfit/dgs_trajectory_tasks.py"
    tree = ast.parse(path.read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules.extend(alias.name for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for alias in node.names)
    forbidden = ("PySide", "PyQt", "qt_", "project_", "raw_dgs", "mdevent", "mantid", "shiver")
    assert not any(module.startswith(forbidden) for module in modules)
