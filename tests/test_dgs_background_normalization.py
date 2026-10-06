"""Exact geometry guards and bounded exposure recipe batching."""

import numpy as np
import pytest

import nfit.dgs_background_normalization as service
from nfit.dgs_background_normalization import ReplayExposureBatch


def _detector():
    return (np.array([10, 20]), np.array([.3, .4]),
            np.array([.1, .2]), np.array([1., 2.]))


def _payload(charge=1., index=0):
    return (np.eye(3), 10., np.array([-.5, .5]), charge, index)


def _integrate(batch, normalize, output=None, progress=None):
    if output is None:
        output = np.zeros(2)
    batch.integrate(output, normalize, [np.array([0., 1., 2.])], (2,),
                    sources=2, angles=1, progress_callback=progress)
    return output


@pytest.mark.parametrize("component", range(4))
@pytest.mark.parametrize("difference", ["value", "dtype"])
def test_geometry_requires_complete_identity_even_with_hash_collisions(
    monkeypatch, component, difference,
):
    # Hashes are only candidate selectors, never proof of numerical equality.
    monkeypatch.setattr(service, "_array_identity", lambda array: "collision")
    original = _detector()
    changed = list(a.copy() for a in original)
    if difference == "value":
        changed[component][0] += 1
    else:
        changed[component] = changed[component].astype(np.float32)
    batch = ReplayExposureBatch(192*1024**2)
    batch.add([original], [_payload()], None)
    batch.add([tuple(a.copy() for a in original), tuple(changed)],
              [_payload(2.), _payload(3., 1)], None)
    calls = []

    def normalize(detectors, payloads, *args, **kwargs):
        calls.append((detectors, payloads))
        return np.array([4., 5.])

    np.testing.assert_array_equal(_integrate(batch, normalize), [4., 5.])
    assert len(calls) == 1
    assert len(calls[0][0]) == 2
    assert [p[4] for p in calls[0][1]] == [0, 0, 1]
    assert [p[3] for p in calls[0][1]] == [1., 2., 3.]
    assert not batch.groups
    assert batch.pending_bytes == 0


def test_mask_partition_and_readonly_normalization(monkeypatch):
    monkeypatch.setattr(service, "_array_identity", lambda array: "collision")
    batch = ReplayExposureBatch(1024**2)
    batch.add([_detector()], [_payload()], np.array([True, False]))
    batch.add([_detector()], [_payload()], np.array([False, True]))
    calls = []

    def normalize(*args, **kwargs):
        result = np.array([3., 4.])
        result.setflags(write=False)
        calls.append(result)
        return result

    np.testing.assert_array_equal(_integrate(batch, normalize), [3., 4.])
    assert len(calls) == 2
    for array in calls:
        np.testing.assert_array_equal(array, [3., 4.])


def test_flush_quota_releases_recipes_and_preserves_global_progress():
    batch = ReplayExposureBatch(1)
    progress = []
    output = np.zeros(2)

    def normalize(*args, progress_callback, **kwargs):
        for step in (0, 1):
            progress_callback(dict(iteration=step, total=1))
        return np.ones(2)

    for _ in range(2):
        batch.add([_detector()], [_payload()], None)
        assert batch.full
        _integrate(batch, normalize, output, progress.append)
        assert not batch.full
        assert not batch.groups
    np.testing.assert_array_equal(output, [2., 2.])
    assert [p["iteration"] for p in progress] == [0, 1000, 1000, 2000]
    assert progress[-1]["total"] == 2000
    assert ReplayExposureBatch(10**12).limit_bytes == 32*1024**2


def test_cancel_propagates_before_denominator_changes():
    batch = ReplayExposureBatch(1024**2)
    batch.add([_detector()], [_payload()], None)
    output = np.zeros(2)

    def cancel(event):
        raise RuntimeError("cancelled")

    def normalize(*args, progress_callback, **kwargs):
        progress_callback(dict(iteration=0, total=1))
        pytest.fail("Reduction continued after cancellation")

    with pytest.raises(RuntimeError, match="cancelled"):
        _integrate(batch, normalize, output, cancel)
    np.testing.assert_array_equal(output, 0.)


def test_compatibility_constant_has_one_owner_and_no_reverse_facade_import():
    import ast
    from pathlib import Path

    from nfit import mdevent_background

    assert mdevent_background.MAX_REPLAY_NORMALIZATION_BATCH_TASKS is (
        service.MAX_REPLAY_NORMALIZATION_BATCH_TASKS
    )
    tree = ast.parse(Path(service.__file__).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imports & {"mdevent", "mdevent_background", "project_data", "project_gui"}
