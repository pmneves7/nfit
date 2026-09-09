from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import nfit.project_data as project_data
import nfit.project_point_lists as project_point_lists
from nfit.dataset import PointListData
from nfit.pipeline import DatasetEntry

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"


def _magnetization_dataset() -> DatasetEntry:
    data = PointListData(
        {
            "Temperature": [2.0, 3.0],
            "Magnetic Field": [100.0, 200.0],
            "Moment": [1.0, 4.0],
            "Moment Error": [0.1, 0.4],
        },
        units={
            "Temperature": "K",
            "Magnetic Field": "Oe",
            "Moment": "emu",
            "Moment Error": "emu",
        },
        coordinate_names=["Temperature", "Magnetic Field"],
        channels=[
            {
                "label": "Moment",
                "value": "Moment",
                "error": "Moment Error",
                "quantity_type": "magnetic_moment",
                "unit": "emu",
            }
        ],
    )
    return DatasetEntry("magnetization", data, data_type="magnetization")


def test_project_data_reexports_point_list_service_objects() -> None:
    assert project_data.point_list_config is project_point_lists.point_list_config
    assert project_data.prepared_point_list_data is project_point_lists.prepared_point_list_data
    assert (
        project_data._prepared_point_list_signature
        is project_point_lists._prepared_point_list_signature
    )
    assert project_data._PREPARED_POINT_LIST_CACHE is project_point_lists._PREPARED_POINT_LIST_CACHE


def test_facade_and_focused_module_prepare_equivalent_scientific_data() -> None:
    dataset = _magnetization_dataset()
    config = project_data.point_list_config(dataset)
    config["scale"].update({"factor": 2.0, "channel": "Moment", "units": "emu"})
    config["susceptibility"].update(
        {
            "enabled": True,
            "field": "Magnetic Field",
            "moment": "Moment",
            "field_unit": "Oe",
            "moment_unit": "emu",
        }
    )

    via_facade = project_data.prepared_point_list_data(dataset)
    via_service = project_point_lists.prepared_point_list_data(dataset)

    assert via_facade is via_service
    np.testing.assert_allclose(via_facade.channel_values("Moment"), [2.0, 8.0])
    np.testing.assert_allclose(via_facade.channel_values("Susceptibility"), [0.02, 0.04])


def test_prepared_point_list_cache_tracks_configuration_and_data_revision() -> None:
    project_point_lists._PREPARED_POINT_LIST_CACHE.clear()
    dataset = _magnetization_dataset()
    config = project_point_lists.point_list_config(dataset)

    first = project_point_lists.prepared_point_list_data(dataset)
    assert project_point_lists.prepared_point_list_data(dataset) is first

    config["scale"]["factor"] = 3.0
    configured = project_point_lists.prepared_point_list_data(dataset)
    assert configured is not first
    np.testing.assert_allclose(configured.channel_values("Moment"), [3.0, 12.0])

    replacement = dataset.data.with_updates(
        columns={**dataset.data.columns, "Moment": np.asarray([2.0, 5.0])}
    )
    dataset.replace_data(replacement)
    replaced = project_point_lists.prepared_point_list_data(dataset)
    assert replaced is not configured
    np.testing.assert_allclose(replaced.channel_values("Moment"), [6.0, 15.0])


def test_point_list_service_has_no_gui_or_project_data_dependency() -> None:
    tree = ast.parse((PACKAGE_ROOT / "project_point_lists.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any("PySide" in module or module.startswith("qt_") for module in imported_modules)
    assert "project_gui" not in imported_modules
    assert "project_data" not in imported_modules
