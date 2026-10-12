"""Source-independent ownership and compatibility for shared DGS primitives."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nfit import corelli, dgs_binning, dgs_trajectory_accumulation, mdevent, raw_dgs


@pytest.mark.parametrize("name", (
    "_requested_edges", "_symmetry_matrices", "_validated_minimum_samples",
))
def test_reducers_share_one_grid_implementation(name):
    owner = getattr(dgs_binning, name)
    assert getattr(mdevent, name) is owner
    assert getattr(raw_dgs, name) is owner
    assert getattr(corelli, name) is owner


@pytest.mark.parametrize("name", (
    "_accumulate_detector_trajectory", "_accumulate_powder_detector_trajectory",
))
def test_reducers_share_one_fallback_trajectory_implementation(name):
    owner = getattr(dgs_trajectory_accumulation, name)
    assert getattr(mdevent, name) is owner
    assert getattr(raw_dgs, name) is owner


@pytest.mark.parametrize("module", (dgs_binning, dgs_trajectory_accumulation))
def test_shared_primitives_do_not_import_source_reducers_or_gui(module):
    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    imported = {node.module or "" for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)}
    imported.update(alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names)
    imported.update(alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module in {None, "nfit"}
                    for alias in node.names)
    imported = {name.removeprefix("nfit.") for name in imported}
    assert not any(name.split(".")[0] in {"mdevent", "raw_dgs", "corelli"}
                   or name.startswith(("project_", "PySide", "PyQt", "qt_"))
                   for name in imported)


@pytest.mark.parametrize("module", (raw_dgs, corelli))
def test_raw_reducers_import_shared_primitives_from_neutral_owners(module):
    moved_names = {"_requested_edges", "_symmetry_matrices", "_validated_minimum_samples",
                   "_accumulate_detector_trajectory", "_accumulate_powder_detector_trajectory"}
    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    reducer_imports = {alias.name for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) and node.module in {"mdevent", "nfit.mdevent"}
                       for alias in node.names}
    assert not reducer_imports.intersection(moved_names)


def test_memory_planner_preserves_live_reducer_estimator_and_iterables(monkeypatch):
    received = []
    def estimator(counts, *, max_batch_bytes):
        received.append((counts, max_batch_bytes))
        return 1234
    monkeypatch.setattr(mdevent, "estimate_mdevent_peak_memory", estimator)
    bound = SimpleNamespace(arguments={
        "lower": iter([0., 0.]), "upper": iter([2., 4.]),
        "num_bins": iter([3, 5]), "step_size": iter([1., 1.]),
        "bin_edges": iter([None, iter([0., 2., 4.])]), "max_batch_bytes": 128,
    })
    assert mdevent._dgs_grid_memory(bound) == 1234
    assert received == [([3, 2], 128)]
    assert bound.arguments["lower"] == (0., 0.)
    assert bound.arguments["bin_edges"] == (None, (0., 2., 4.))


@pytest.mark.parametrize("mantid_precision", (False, True))
def test_fallback_trajectories_integrate_complete_energy_interval(mantid_precision):
    output = np.zeros((1, 1, 1, 1))
    edges = (np.array([-100., 100.]),) * 3 + (np.array([0., 4.]),)
    dgs_trajectory_accumulation._accumulate_detector_trajectory(
        output, edges, np.eye(3), np.array([0., 0., 1.]), 10., [0., 4.], 3.,
        mantid_precision=mantid_precision,
    )
    np.testing.assert_allclose(output.item(), 12., rtol=1e-14)
    powder = np.zeros((1, 1))
    dgs_trajectory_accumulation._accumulate_powder_detector_trajectory(
        powder, (np.array([0., 100.]), np.array([0., 4.])), .3,
        10., [0., 4.], 3., mantid_precision=mantid_precision,
    )
    np.testing.assert_allclose(powder.item(), 12., rtol=1e-14)
