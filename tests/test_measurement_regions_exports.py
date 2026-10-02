"""Linear targets and exported declarations retain source information."""
import json

import numpy as np
import pytest

from nfit import estimate_measurement_region, save_measurement_grid_csv
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import (
    independent_source_dependencies,
    project_source_dependencies,
    restore_source_dependencies,
)
from nfit.measurement_profiles import MeasurementProfile
from nfit.plotting_core import WaterfallTrace
from nfit.viewer_export import save_measurement_profile_csv, save_waterfall_csv


def shared_sources():
    primitive = independent_source_dependencies(np.array([4.]), "calibration")
    return project_source_dependencies(primitive, [0, 0], [0, 1], [1, 1], (2,))


def test_linear_target_retains_shared_covariance_and_signed_cancellation():
    dependencies = shared_sources()
    contract = MeasurementContract(kind="linear_reconstruction", estimator="linear_sum", quantity="integral", value_units="U K", dependence="shared_sources")
    result = estimate_measurement_region([3, 3], [2, 2], weights=[1, 1], source_dependencies=dependencies, contract=contract)
    assert result.value == 6 and result.variance == 16
    cancelled = estimate_measurement_region([3, 3], [2, 2], weights=[1, -1], source_dependencies=dependencies, contract=contract)
    assert cancelled.value == cancelled.variance == 0
    with pytest.raises(ValueError, match="source"):
        estimate_measurement_region([3, 3], [2, 2], contract=contract)


def test_region_masks_remove_coefficients_and_missing_results_are_not_zero():
    result = estimate_measurement_region([3, 3], [2, 2], mask=[False, True], source_dependencies=shared_sources())
    assert result.value == 3 and result.variance == 4 and result.included == 1
    missing = estimate_measurement_region([np.nan], [np.nan])
    assert np.isnan(missing.value) and np.isnan(missing.standard_error)


def profile():
    contract = MeasurementContract(kind="continuous", estimator="uniform_mean", quantity="value", value_units="U", dependence="shared_sources")
    data = MDHistoData((MDHistoAxis("x", [0, 1, 2], "K", "unknown"),), np.array([3., 3.]), np.array([2., 2.]),
        np.zeros(2, bool), np.zeros(2), metadata={"measurement_contract": contract.to_dict(), "numpy_example": np.int64(3)},
        auxiliary_channels={"coverage_fraction": MDHistoChannel(np.ones(2))}, source_dependencies=shared_sources())
    return MeasurementProfile(data, contract)


def test_profile_and_waterfall_export_source_sidecars_roundtrip(tmp_path):
    prepared = profile()
    path = save_measurement_profile_csv(tmp_path / "profile.csv", prepared, coordinate_name="x")
    declaration = json.loads(path.with_suffix(".csv.json").read_text())
    assert declaration["metadata"]["numpy_example"] == 3
    source_spec = declaration["source_dependencies"]
    with np.load(tmp_path / source_spec["file"]) as archive:
        prefix = source_spec["payloads"]["profile"]["prefix"]
        restored = restore_source_dependencies({key.removeprefix(prefix): archive[key] for key in archive.files if key.startswith(prefix)})
    np.testing.assert_array_equal(restored.variance(), prepared.data.source_dependencies.variance())
    trace = WaterfallTrace(prepared.arrays[0], prepared.arrays[1], errors=prepared.arrays[2], model_values=None, label="test", measurement=prepared)
    output = save_waterfall_csv(tmp_path / "waterfall.csv", [trace])
    assert "mask" in output.read_text().splitlines()[0]
    assert json.loads(output.with_suffix(".csv.json").read_text())["traces"]["0"]["contract"] == prepared.contract.to_dict()


def test_grid_export_distinguishes_diagnostics_from_observed_statistics(tmp_path):
    view = {"x_centers": [.5], "y_centers": [.5, 1.5], "x_edges": [0, 1], "y_edges": [0, 1, 2],
        "signal": np.array([[3.], [3.]]), "errors": np.array([[2.], [2.]]), "mask": np.zeros((2, 1), bool),
        "coverage_fraction": np.ones((2, 1)), "num_events": np.zeros((2, 1)), "measurement_contract": profile().contract.to_dict(),
        "source_dependencies": project_source_dependencies(shared_sources(), [0, 1], [0, 1], [1, 1], (2, 1))}
    path = save_measurement_grid_csv(tmp_path / "grid.csv", view, coordinate_units=("K", "meV"))
    declaration = json.loads(path.with_suffix(".csv.json").read_text())
    assert declaration["coordinate_units"] == ["K", "meV"]
    assert "source_dependencies" in declaration
    diagnostic = save_measurement_grid_csv(tmp_path / "error.csv", view, channel="errors")
    declaration = json.loads(diagnostic.with_suffix(".csv.json").read_text())
    assert "measurement_contract" not in declaration["metadata"]
    assert "source_dependencies" not in declaration
