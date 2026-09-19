"""Resource ceilings and cancellation for the public replay path."""

import numpy as np
import pytest

from nfit import bin_mdevent_group, mdevent_dataset_group
from nfit import mdevent_background as replay
from tests.test_mdevent import _write_mdevent


def test_compiled_replay_checks_cancellation_between_bounded_batches(tmp_path, monkeypatch):
    if replay._REPLAY_NUMBA is None:
        pytest.skip("Numba is unavailable")
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample = mdevent_dataset_group(path)
    source = mdevent_dataset_group(path)
    target = bin_mdevent_group(sample, lower=[-1] * 4, upper=[1] * 4, num_bins=[1] * 4)
    original = target.signal.copy()
    monkeypatch.setattr(replay._parallel, "num_threads", lambda: 1)
    monkeypatch.setattr(replay, "MAX_REPLAY_TRANSFORM_TASKS", 1)
    calls = []
    accumulate = replay._REPLAY_NUMBA.accumulate_replayed_events

    def record(*args, **kwargs):
        calls.append((len(args[0]), kwargs["workers"]))
        return accumulate(*args, **kwargs)

    monkeypatch.setattr(replay._REPLAY_NUMBA, "accumulate_replayed_events", record)

    def cancel_after_batch(event):
        if event["stage"] == "mdevent_background_replay" and event["iteration"] > 0:
            assert event["backend"] == "numba"
            assert event["workers"] == 1
            raise RuntimeError("stop after completed replay batch")

    with pytest.raises(RuntimeError, match="stop after completed replay batch"):
        replay.project_measured_background_mdevent(
            sample, source, target, progress_callback=cancel_after_batch,
        )
    assert calls == [(1, 1)]
    np.testing.assert_array_equal(target.signal, original)
