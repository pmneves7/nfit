import numpy as np
import pytest

numba = pytest.importorskip("numba")

from nfit import _mdevent_background_numba as kernels  # noqa: E402


def _run(lab, energies, detector_indices, signal, source_variance, inverses, weights, edges, shape, *, accepted=None, exclusions=None, workers=1):
    transforms = len(weights)
    accepted = accepted or [np.ones(1, dtype=bool) for _ in range(transforms)]
    exclusions = exclusions or [np.zeros(0, dtype=bool) for _ in range(transforms)]
    flags = kernels.prepare_replay_flags(
        accepted, exclusions, output_size=int(np.prod(shape))
    )
    outputs = [np.zeros(int(np.prod(shape))) for _ in range(3)]
    used = kernels.accumulate_replayed_events(
        lab,
        energies,
        detector_indices,
        signal,
        source_variance,
        inverses,
        np.tile([edges[3][0], edges[3][-1]], (transforms, 1)),
        weights,
        *flags,
        edges,
        np.asarray(shape, dtype=np.int64),
        *outputs,
        workers=workers,
    )
    return used, outputs


def test_replay_kernel_combines_collisions_before_variance_and_counts_zero_weight():
    edges = tuple(np.array([0.0, 0.5, 1.0]) for _ in range(4))
    lab = np.array([[0.25, 0.25, 0.25], [1.0, 1.0, 1.0], [0.25, 0.25, 0.25]])
    energies = np.array([0.25, 1.0, 0.25])
    detectors = np.array([0, 0, -1])
    inverses = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    excluded = np.ones(16, dtype=bool)
    _, (numerator, variance, events) = _run(
        lab,
        energies,
        detectors,
        np.array([2.0, 3.0, 4.0]),
        np.array([5.0, 7.0, 11.0]),
        inverses,
        np.array([1.0, -1.0, 4.0]),
        edges,
        (2, 2, 2, 2),
        exclusions=[np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), excluded],
    )
    assert numerator[0] == variance[0] == 0.0
    assert events[0] == 1.0
    # The inclusive final edge was mapped, then rejected by the per-transform mask.
    assert events[15] == 1.0
    assert events.sum() == 2.0


def test_replay_kernel_hash_matches_across_threads_for_many_distinct_transforms():
    transforms = 722
    edges = (
        np.arange(1025, dtype=float),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    inverses = np.repeat(np.eye(3)[None, :, :], transforms, axis=0)
    inverses[:, 0, 0] = np.arange(transforms) + 0.5
    weights = np.linspace(-1.0, 2.0, transforms)
    args = (
        np.array([[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]),
        np.array([0.5, 0.5]),
        np.array([0, 0]),
        np.array([2.0, 3.0]),
        np.array([5.0, 7.0]),
        inverses,
        weights,
        edges,
        (1024, 1, 1, 1),
    )
    _, serial = _run(*args, workers=1)
    used, parallel = _run(*args, workers=4)
    assert used == min(2, kernels.effective_workers(4))
    for actual, expected in zip(parallel, serial, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(serial[0][:transforms], 5.0 * weights)
    np.testing.assert_allclose(serial[1][:transforms], 12.0 * weights**2)
    np.testing.assert_array_equal(serial[2][:transforms], 2.0)
    assert kernels.replay_scratch_bytes(2, transforms) == 2 * transforms * 56


def test_replay_kernel_hash_avalanches_strided_flat_indices():
    transforms = 722
    mask = 2047
    slots = {kernels._hash_slot(index * 384, mask) for index in range(transforms)}
    assert len(slots) > 550

    edges = (
        np.arange(1025, dtype=float),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.arange(33, dtype=float),
    )
    inverses = np.repeat(np.eye(3)[None, :, :], transforms, axis=0)
    inverses[:, 0, 0] = np.arange(transforms) + 0.5
    weights = np.linspace(0.5, 1.5, transforms)
    _, (numerator, variance, events) = _run(
        np.array([[1.0, 0.5, 0.5]]),
        np.array([0.5]),
        np.array([0]),
        np.array([2.0]),
        np.array([3.0]),
        inverses,
        weights,
        edges,
        (1024, 1, 1, 32),
        workers=4,
    )
    flat = np.arange(transforms) * 32
    np.testing.assert_allclose(numerator[flat], 2.0 * weights)
    np.testing.assert_allclose(variance[flat], 3.0 * weights**2)
    np.testing.assert_array_equal(events[flat], 1.0)


def test_prepare_replay_flags_accepts_empty_detector_table_and_validates_masks():
    accepted, excluded = kernels.prepare_replay_flags(
        [np.zeros(0, dtype=bool)], [np.zeros(4, dtype=bool)], output_size=4
    )
    assert accepted[0].size == 0
    assert excluded[0].size == 4
    with pytest.raises(ValueError, match="flattened output size"):
        kernels.prepare_replay_flags(
            [np.ones(1, dtype=bool)], [np.zeros(3, dtype=bool)], output_size=4
        )

    readonly_accepted = np.ones(1, dtype=bool)
    readonly_excluded = np.zeros(4, dtype=bool)
    readonly_accepted.setflags(write=False)
    readonly_excluded.setflags(write=False)
    prepared = kernels.prepare_replay_flags(
        [readonly_accepted], [readonly_excluded], output_size=4
    )
    assert prepared[0][0].ctypes.data == readonly_accepted.ctypes.data
    assert prepared[1][0].ctypes.data == readonly_excluded.ctypes.data


def test_replay_kernel_restores_numba_threads_on_success_and_failure(monkeypatch):
    previous = numba.get_num_threads()
    edges = tuple(np.array([0.0, 1.0]) for _ in range(4))
    args = (
        np.array([[0.5, 0.5, 0.5]]),
        np.array([0.5]),
        np.array([0]),
        np.array([1.0]),
        np.array([1.0]),
        np.eye(3)[None, :, :],
        np.array([1.0]),
        edges,
        (1, 1, 1, 1),
    )
    _run(*args, workers=1)
    assert numba.get_num_threads() == previous

    def fail(*_args):
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(kernels, "_map_and_combine", fail)
    with pytest.raises(RuntimeError, match="injected kernel failure"):
        _run(*args, workers=1)
    assert numba.get_num_threads() == previous
