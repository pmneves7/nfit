"""Fused DGS projection retains exact event order, boundaries and copy variance."""

import ast
import importlib
import inspect

import numpy as np
import pytest

from nfit import dgs_event_accumulation
from nfit.dgs_event_accumulation import accumulate_projected_dgs_events
from nfit.dgs_reduction_policy import prepare_dgs_event_projector
from nfit.event_covariance import accumulate_copy_covariance
from nfit.mdevent import _accumulate_discrete_event_coordinates
from nfit.symmetry import SymmetrySpec, resolve_symmetry


def _case():
    ub = np.array([[.0841234567, .0231, -.00122], [.02407, -.0577321, .006], [.003, .0052, .06789]])
    basis = np.array([[1, 1, 1, 0], [1, -1, 0, 0], [1, 1, -2, 0], [0, 0, 0, 1]])
    edges = tuple(np.linspace(low, high, bins+1) for low, high, bins in
                  ((-1, 1, 17), (-2, 2, 23), (-1, 1, 13), (-2, 22, 24)))
    return ub, basis, edges


def _arrays(rows=3000, layout="C"):
    rng = np.random.default_rng(712)
    packed = rng.normal(size=(rows, 9))
    packed[:, 3] = rng.uniform(-5, 30, rows)
    packed[:, 4] *= 3
    packed[:, 5] = rng.uniform(.01, 20, rows)
    if layout == "F":
        packed = np.asfortranarray(packed)
    elif layout == "reversed":
        packed = packed[::-1]
    elif layout == "sliced":
        packed = packed[::2]
    return packed[:, :3], packed[:, 3], packed[:, 4], packed[:, 5]


def _compare(projector, inputs, shape, *, enabled=None, variances=True, chunks=None):
    q, energy, weights, variance = inputs
    variance = variance if variances else None
    reference = [np.zeros(shape) for _ in range(3)]
    candidate = [np.zeros(shape) for _ in range(3)]
    bins = [np.full(len(energy), -1, dtype=np.int64) for _ in range(2)]
    chunks = chunks or [(0, len(energy))]
    for lo, hi in chunks:
        selection = slice(lo, hi)
        coords, edges = projector(q[selection], energy[selection])
        _accumulate_discrete_event_coordinates(coords, weights[selection],
            None if variance is None else variance[selection], edges, shape, *reference,
            enabled=None if enabled is None else enabled[selection], bin_indices=bins[0][selection])
        accumulate_projected_dgs_events(projector, q[selection], energy[selection], weights[selection],
            None if variance is None else variance[selection], shape, *candidate,
            fallback_accumulator=_accumulate_discrete_event_coordinates,
            enabled=None if enabled is None else enabled[selection], bin_indices=bins[1][selection])
    for expected, actual in zip(reference, candidate, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(bins[1], bins[0])
    return bins, reference, candidate


@pytest.mark.parametrize("precision", ["mantid", "high_precision"])
@pytest.mark.parametrize("layout", ["C", "F", "reversed", "sliced"])
@pytest.mark.parametrize("variances", [False, True])
def test_unequal_signed_weights_layouts_masks_and_streamed_chunks_are_literal(layout, variances, precision):
    ub, basis, edges = _case()
    projector = prepare_dgs_event_projector(ub, basis, np.eye(3), edges, precision)
    inputs = _arrays(layout=layout)
    before = [array.copy() for array in inputs]
    enabled = np.arange(len(inputs[1])) % 3 != 0
    rows = len(enabled)
    _compare(projector, inputs, tuple(len(edge)-1 for edge in edges), enabled=enabled,
             variances=variances, chunks=[(0, 1), (1, 113), (113, rows)])
    for array, original in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("precision", ["mantid", "high_precision"])
def test_exact_and_nextafter_edges_invalid_coordinates_and_coverage_fringe_bins(precision):
    ub = np.eye(3)/(2*np.pi)
    edges = tuple(np.linspace(-1, 1, 9) for _ in range(4))
    projector = prepare_dgs_event_projector(ub, np.eye(4), np.eye(3), edges, precision)
    levels = np.concatenate((edges[0], np.nextafter(edges[0], -np.inf),
                             np.nextafter(edges[0], np.inf), [np.nan, np.inf, -np.inf, -0.]))
    points = np.zeros((4*len(levels), 4))
    for dim in range(4):
        points[dim*len(levels):(dim+1)*len(levels), dim] = levels
    weights = np.linspace(-2, 3, len(points))
    variance = np.linspace(.2, 4, len(points))
    _compare(projector, (points[:, :3], points[:, 3], weights, variance), (8,)*4)


@pytest.mark.parametrize("precision", ["mantid", "high_precision"])
@pytest.mark.parametrize("copies", [1, 6, 12])
@pytest.mark.parametrize("covariance", [False, True])
def test_symmetry_order_and_selected_copy_variance_policy_are_literal(copies, covariance, precision):
    ub, basis, edges = _case()
    shape = tuple(len(edge)-1 for edge in edges)
    expression = "x,y,z;y,z,x;z,x,y;y,x,z;x,z,y;z,y,x;-x,-y,-z;-y,-z,-x;-z,-x,-y;-y,-x,-z;-x,-z,-y;-z,-y,-x"
    operations = resolve_symmetry(SymmetrySpec("operations", expression))[:copies]
    q, energy, weights, variances = _arrays(rows=5000)
    # Coincident zero-Q symmetry copies exercise nonzero covariance corrections.
    q[:50] = 0
    outputs = [[np.zeros(shape) for _ in range(3)] for _ in range(2)]
    bins = [np.full((copies, len(energy)), -1, dtype=np.int64) for _ in range(2)]
    for copy, operation in enumerate(operations):
        projector = prepare_dgs_event_projector(ub, basis, operation.matrix_hkl, edges, precision)
        coordinates, accumulation_edges = projector(q, energy)
        _accumulate_discrete_event_coordinates(coordinates, weights, variances, accumulation_edges,
            shape, *outputs[0], bin_indices=bins[0][copy])
        accumulate_projected_dgs_events(projector, q, energy, weights, variances, shape, *outputs[1],
            fallback_accumulator=_accumulate_discrete_event_coordinates, bin_indices=bins[1][copy])
    if covariance:
        summaries = [accumulate_copy_covariance(indices, variances, output[1])
                     for indices, output in zip(bins, outputs, strict=True)]
        assert summaries[0] == summaries[1]
        if copies > 1:
            assert summaries[1]["within_bin_pairs_corrected"] > 0
    np.testing.assert_array_equal(bins[0], bins[1])
    for expected, actual in zip(outputs[0], outputs[1], strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["nonuniform", "numba_unavailable"])
def test_established_fallback_remains_exact(monkeypatch, mode):
    ub, basis, edges = _case()
    if mode == "nonuniform":
        edges = (np.array([-1, -.7, -.55, .2, 1]), *edges[1:])
    policy = "mantid"
    projector = prepare_dgs_event_projector(ub, basis, np.eye(3), edges, policy)
    if mode == "numba_unavailable":
        monkeypatch.setattr(dgs_event_accumulation, "_compiled", None)
    calls = []
    original = _accumulate_discrete_event_coordinates
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    q, energy, weights, variances = _arrays(rows=300)
    sums = [np.zeros(tuple(len(edge)-1 for edge in edges)) for _ in range(3)]
    accumulate_projected_dgs_events(projector, q, energy, weights, variances, sums[0].shape,
        *sums, fallback_accumulator=record)
    assert calls == [True]
    _compare(projector, (q, energy, weights, variances), sums[0].shape)


def test_fused_service_has_no_reverse_import_or_gui_dependency():
    tree = ast.parse(inspect.getsource(dgs_event_accumulation))
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module in ("raw_dgs", "mdevent") or "gui" in module or "Qt" in module for module in modules)


def test_disabled_jit_retains_established_projection_fallback(monkeypatch):
    from numba import config

    original_disabled = config.DISABLE_JIT
    with monkeypatch.context() as patched:
        patched.setattr(config, "DISABLE_JIT", True)
        importlib.reload(dgs_event_accumulation)
        assert dgs_event_accumulation._compiled is None
        ub, basis, edges = _case()
        projector = prepare_dgs_event_projector(ub, basis, np.eye(3), edges)
        _compare(projector, _arrays(rows=40), tuple(len(edge)-1 for edge in edges))
    importlib.reload(dgs_event_accumulation)
    assert (dgs_event_accumulation._compiled is None) == bool(original_disabled)


def test_fused_shape_mismatch_fails_before_unchecked_native_indexing():
    ub, basis, edges = _case()
    projector = prepare_dgs_event_projector(ub, basis, np.eye(3), edges)
    q, energy, weights, variances = _arrays(rows=3)
    sums = [np.zeros((17, 23, 13, 24)) for _ in range(3)]
    with pytest.raises(ValueError, match="matching"):
        accumulate_projected_dgs_events(projector, q[:2], energy, weights, variances,
            sums[0].shape, *sums, fallback_accumulator=_accumulate_discrete_event_coordinates)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_high_precision_fuses_physical_edges_without_calling_numpy_projector(nonuniform):
    ub, basis, edges = _case()
    if nonuniform:
        edges = (np.array([-1, -.7, -.55, .2, 1]), *edges[1:])
    projector = prepare_dgs_event_projector(ub, basis, np.eye(3), edges, "high_precision")
    class FusedOnly:
        _dgs_float64_projection = projector._dgs_float64_projection
        def __call__(self, *args):
            pytest.fail("compiled high precision constructed event-sized coordinates")
    q, energy, weights, variances = _arrays(rows=300)
    outputs = [np.zeros(tuple(len(edge)-1 for edge in edges)) for _ in range(3)]
    reference = [np.zeros_like(array) for array in outputs]
    coordinates, accumulation_edges = projector(q, energy)
    _accumulate_discrete_event_coordinates(coordinates, weights, variances, accumulation_edges,
        outputs[0].shape, *reference)
    accumulate_projected_dgs_events(FusedOnly(), q, energy, weights, variances,
        outputs[0].shape, *outputs, fallback_accumulator=_accumulate_discrete_event_coordinates)
    for expected, actual in zip(reference, outputs, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_high_precision_retains_fallback_when_compiler_is_unavailable(monkeypatch):
    monkeypatch.setattr(dgs_event_accumulation, "_compiled_high_precision", None)
    ub, basis, edges = _case()
    _compare(prepare_dgs_event_projector(ub, basis, np.eye(3), edges, "high_precision"),
             _arrays(rows=30), tuple(len(edge)-1 for edge in edges))
