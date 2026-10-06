"""Background performance trials require literal whole-grid numerical parity."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


def harness():
    path = Path(__file__).parents[1] / "benchmarks" / "trial_dgs_background_workflow.py"
    spec = importlib.util.spec_from_file_location("background_workflow_trial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_background_trial_checks_nan_support_and_small_boundary_discrepancies(tmp_path):
    h5py = pytest.importorskip("h5py")
    reference, actual = tmp_path / "a.h5", tmp_path / "b.h5"
    values = np.arange(24., dtype=float).reshape(2, 3, 4)
    values[0, 0, 0] = np.nan
    with h5py.File(reference, "w") as handle:
        handle["signal"] = values
        handle["mask"] = ~np.isfinite(values)
        handle["edge0"] = [0., 1., 2.]
    shutil.copyfile(reference, actual)
    output = tmp_path / "parity.json"
    module = harness()
    module.compare_outputs(reference, actual, output)
    assert json.loads(output.read_text())["literal_parity"]
    with h5py.File(actual, "r+") as handle:
        handle["signal"][1, 2, 3] += 1e-12
    with pytest.raises(AssertionError, match="numerical cells"):
        module.compare_outputs(reference, actual, output)
    assert json.loads(output.read_text())["channels"]["signal"]["different_cells"] == 1
    shutil.copyfile(reference, actual)
    with h5py.File(actual, "r+") as handle:
        handle["signal"][0, 0, 0] = 0.
    with pytest.raises(AssertionError, match="numerical cells"):
        module.compare_outputs(reference, actual, output)
