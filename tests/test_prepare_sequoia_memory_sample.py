"""Tests for the standalone SEQUOIA sample-preparation benchmark utility."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


def test_preparer_crops_payloads_and_axis_edges_without_loading_project_code(tmp_path: Path):
    shape = (4, 6, 8, 10)
    payload = tmp_path / "data.npz"
    signal = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    np.savez_compressed(
        payload, signal=signal, errors=signal + 1, num_events=signal + 2,
        axis_0_values=np.arange(5), axis_1_values=np.arange(7),
        axis_2_values=np.arange(9), axis_3_values=np.arange(11),
        metadata_json=np.asarray('{"unchanged": true}'),
    )
    project = tmp_path / "project.nfit"
    member = "assets/binnings/example/data.npz"
    with zipfile.ZipFile(project, "w") as archive:
        archive.write(payload, member)
    output = tmp_path / "sample.npz"
    script = Path(__file__).parents[1] / "benchmarks" / "prepare_sequoia_memory_sample.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--project", str(project), "--member", member,
         "--output", str(output), "--shape", "2,4,6,8"],
        check=True, text=True, capture_output=True,
    )
    manifest = json.loads(completed.stdout)
    assert manifest["slice_starts"] == [1, 1, 1, 1]
    assert manifest["target_shape"] == [2, 4, 6, 8]
    with np.load(output) as sample:
        np.testing.assert_array_equal(sample["signal"], signal[1:3, 1:5, 1:7, 1:9])
        np.testing.assert_array_equal(sample["axis_2_values"], np.arange(1, 8))
        assert str(sample["metadata_json"].item()) == '{"unchanged": true}'


def test_preparer_refuses_to_overwrite_an_existing_output(tmp_path: Path):
    project = tmp_path / "project.nfit"
    project.write_bytes(b"not a project because output refusal happens first")
    output = tmp_path / "existing.npz"
    output.write_bytes(b"keep this output")
    script = Path(__file__).parents[1] / "benchmarks" / "prepare_sequoia_memory_sample.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--project", str(project), "--member", "unused",
         "--output", str(output)], text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert output.read_bytes() == b"keep this output"
