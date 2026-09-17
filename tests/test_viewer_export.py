from __future__ import annotations

import numpy as np
import pytest

from nfit.plotting import WaterfallTrace
from nfit.viewer_export import (
    save_grid_csv,
    save_profile_csv,
    save_waterfall_csv,
)


def test_profile_csv_uses_requested_coordinate_and_uncertainty_columns(tmp_path):
    path = tmp_path / "x_cut.csv"

    save_profile_csv(
        path,
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([0.1, 0.2]),
        coordinate_name="x",
    )

    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "x,intensity,uncertainty"
    )
    np.testing.assert_allclose(
        np.loadtxt(path, delimiter=",", skiprows=1),
        [[1.0, 3.0, 0.1], [2.0, 4.0, 0.2]],
    )


def test_grid_csv_writes_tidy_x_y_intensity_and_uncertainty(tmp_path):
    path = tmp_path / "slice.csv"

    save_grid_csv(
        path,
        np.array([1.0, 2.0]),
        np.array([10.0, 20.0]),
        np.array([[3.0, 4.0], [5.0, 6.0]]),
        np.full((2, 2), 0.25),
    )

    assert path.read_text(encoding="utf-8").splitlines()[0] == "x,y,I,dI"
    np.testing.assert_allclose(
        np.loadtxt(path, delimiter=",", skiprows=1),
        [
            [1.0, 10.0, 3.0, 0.25],
            [2.0, 10.0, 4.0, 0.25],
            [1.0, 20.0, 5.0, 0.25],
            [2.0, 20.0, 6.0, 0.25],
        ],
    )


def test_waterfall_csv_uses_physical_coordinates_and_unshifted_model(tmp_path):
    traces = [
        WaterfallTrace(
            x=np.array([0.0, 1.0]),
            values=np.array([2.0, 3.0]),
            errors=np.array([0.2, 0.3]),
            model_values=np.array([1.5, 2.5]),
            label="4 K",
            waterfall_coordinate=4.0,
        ),
        WaterfallTrace(
            x=np.array([0.0, 1.0]),
            values=np.array([5.0, 6.0]),
            errors=np.array([0.5, 0.6]),
            model_values=np.array([4.5, 5.5]),
            label="8 K",
            waterfall_coordinate=8.0,
        ),
    ]
    data_path = tmp_path / "waterfall.csv"
    model_path = tmp_path / "waterfall_model.csv"

    save_waterfall_csv(data_path, traces)
    save_waterfall_csv(model_path, traces, model=True)

    assert data_path.read_text(encoding="utf-8").splitlines()[0] == "x,y,I,dI"
    assert model_path.read_text(encoding="utf-8").splitlines()[0] == "x,y,I"
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    model = np.loadtxt(model_path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 1], [4.0, 4.0, 8.0, 8.0])
    np.testing.assert_allclose(data[:, 2], [2.0, 3.0, 5.0, 6.0])
    np.testing.assert_allclose(model[:, 2], [1.5, 2.5, 4.5, 5.5])


def test_grid_csv_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        save_grid_csv(
            tmp_path / "bad.csv",
            np.array([1.0, 2.0]),
            np.array([3.0]),
            np.ones((2, 2)),
        )
