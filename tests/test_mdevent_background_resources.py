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


@pytest.mark.parametrize("byte_limit", [1, 4096, 192*1024**2])
def test_replay_batch_respects_memory_and_work_limits(tmp_path, monkeypatch, byte_limit):
    from contextlib import contextmanager

    if replay._REPLAY_NUMBA is None:
        pytest.skip("Numba is unavailable")
    path = tmp_path / "events.nxs"
    _write_mdevent(path)
    sample, source = mdevent_dataset_group(path), mdevent_dataset_group(path)
    target = bin_mdevent_group(sample, lower=[-1]*4, upper=[1]*4, num_bins=[1]*4)
    requests, transforms = [], []
    stream = replay.open_event_stream
    accumulate = replay._REPLAY_NUMBA.accumulate_replayed_events

    @contextmanager
    def record_stream(*args, rows):
        requests.append(rows)
        with stream(*args, rows=rows) as value:
            yield value

    def record_accumulation(*args, **kwargs):
        transforms.append(len(args[5]))
        return accumulate(*args, **kwargs)

    monkeypatch.setattr(replay, "open_event_stream", record_stream)
    monkeypatch.setattr(replay._REPLAY_NUMBA, "accumulate_replayed_events", record_accumulation)
    replay.project_measured_background_mdevent(sample, source, target, max_batch_bytes=byte_limit)
    count = transforms[0]
    bytes_per_row = 128 + replay._REPLAY_NUMBA.replay_scratch_bytes(1, count)
    assert all(rows*bytes_per_row <= max(byte_limit, bytes_per_row) for rows in requests)
    assert all(rows*count <= max(replay.MAX_REPLAY_TRANSFORM_TASKS, count) for rows in requests)
    assert all(rows <= 100_000 for rows in requests)
