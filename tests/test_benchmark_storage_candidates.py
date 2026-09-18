"""Smoke test for the isolated storage-candidate benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_storage_candidate_benchmark_reports_exactly_checked_operations(tmp_path: Path):
    root = Path(__file__).parents[1]
    environment = os.environ | {"PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "benchmark_storage_candidates.py"),
         "--shape", "4,4,4,4", "--repeats", "1", "--results", str(tmp_path / "results.json")],
        check=True, capture_output=True, text=True, env=environment,
    )
    report = json.loads(completed.stdout)
    assert report["input"]["shape"] == [4, 4, 4, 4]
    assert {row["candidate"] for row in report["results"]} >= {"resident_numpy", "npy_memmap"}
    assert all(row["compression_ratio_vs_raw"] > 0 for row in report["results"])
