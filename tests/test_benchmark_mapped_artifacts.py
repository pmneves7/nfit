"""Smoke test for the isolated resident/mapped artifact benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_mapped_artifact_benchmark_compares_exact_fresh_process_reads(tmp_path: Path):
    root = Path(__file__).parents[1]
    script = root / "benchmarks" / "benchmark_mapped_artifacts.py"
    artifact = tmp_path / "fixture.npz"
    environment = os.environ | {"PYTHONPATH": str(root / "src")}
    prepared = subprocess.run(
        [sys.executable, str(script), "--prepare", str(artifact),
         "--shape", "4,5,6,7", "--chunk-points", "31"],
        check=True, capture_output=True, text=True, env=environment,
    )
    preparation = json.loads(prepared.stdout)
    before = artifact.read_bytes()
    assert preparation["raw_bytes"] == 4 * 5 * 6 * 7 * 41

    completed = subprocess.run(
        [sys.executable, str(script), "--artifact", str(artifact),
         "--mapped-min-bytes", "1", "--temp-dir", str(tmp_path),
         "--results", str(tmp_path / "results.json")],
        check=True, capture_output=True, text=True, env=environment,
    )
    report = json.loads(completed.stdout)
    assert [row["reader"] for row in report["results"]] == ["resident", "mapped"]
    resident, mapped = report["results"]
    assert resident["exact_array_hashes"] == mapped["exact_array_hashes"]
    assert resident["storage_bytes"]["heap"] >= preparation["raw_bytes"]
    assert mapped["storage_bytes"]["mapped"] >= preparation["raw_bytes"]
    assert all(row["source_stat_unchanged"] for row in report["results"])
    assert all(set(row["operations"]) == {
        "first_plane_sum", "warm_plane_sum", "bounded_roi_sum", "first_full_sum",
        "warm_full_sum_median",
    } for row in report["results"])
    assert artifact.read_bytes() == before
