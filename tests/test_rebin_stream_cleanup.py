"""Streaming rebin resources are released when cooperative cancellation raises."""

from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

import numpy as np
import pytest

import nfit.rebin as rebin_module
from nfit.rebin import ArrayRebinSource, rebin_nd_stream


def test_stream_executor_shuts_down_when_progress_callback_raises(monkeypatch):
    pytest.importorskip("numba")
    shutdown_calls = []

    class RecordingExecutor(RealThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdown_calls.append((wait, cancel_futures))
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(rebin_module, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(rebin_module, "_numba_min_points", lambda: 0)
    source = ArrayRebinSource(
        np.ones(128),
        np.linspace(0.0, 1.0, 128)[:, None],
        data_errs=np.ones(128),
        batch_size=64,
    )

    def cancel(event):
        if event.get("stage") == "rebin":
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        rebin_nd_stream(
            source,
            lower=[0.0],
            upper=[1.0],
            num_bins=[8],
            backend="numba",
            workers=2,
            parallel_strategy="dense",
            progress_callback=cancel,
        )

    assert shutdown_calls == [(True, True)]
