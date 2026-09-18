"""Smoke test for the isolated dense/streaming crossover benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_rebin_crossover_benchmark_reports_forced_stream_path():
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable, str(root / "benchmarks" / "benchmark_rebin_crossover.py"),
            "--shape", "4,4,4,4", "--path", "stream", "--operation", "regular",
            "--occupancy", "sparse", "--workers", "1",
        ],
        check=True, capture_output=True, text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
    )
    report = json.loads(completed.stdout)
    assert report["path_actual"] == "stream"
    assert report["stream_threshold_forced"] == 0
    assert report["input_payload_bytes"] == 4**4 * 41
    assert len(report["output_shape"]) == 4
    assert all(size > 0 for size in report["output_shape"])
    assert report["result_stats"]["finite_signal_bins"] > 0
