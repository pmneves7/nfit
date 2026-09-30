"""Keep the timing comparison's scientific modes and output grid distinct."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_rebin_modes_benchmark_assigns_axes_and_conserves_events():
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "benchmark_rebin_modes.py"),
         "--shape", "8,8,8,8", "--workers", "1", "--repeats", "1"],
        check=True, capture_output=True, text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    assert rows[0]["shape"] == [8, 8, 8, 8]
    assert [row["fractional_axes"] for row in rows[1:]] == [
        [False] * 4, [True, True, True, False], [True] * 4,
    ]
    assert all(row["output_shape"] == [4] * 4 for row in rows[1:])
    events = [row["stats"]["events_sum"] for row in rows[1:]]
    assert events[0] > 0
    np.testing.assert_allclose(events, events[0], rtol=1e-12)
