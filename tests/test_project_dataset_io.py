from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import nfit.project_data as project_data
import nfit.project_dataset_io as project_dataset_io
from nfit.dataset import PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DatasetEntry

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"


def test_project_data_reexports_dataset_io_service_objects() -> None:
    names = (
        "save_dataset_file",
        "_save_point_list_file",
        "_load_nfit_dataset_file",
        "_load_nfit_mdhisto_archive",
        "_load_nfit_point_list_archive",
        "_nfit_archive_text",
        "_nfit_archive_optional_int",
        "_nfit_archive_json_mapping",
        "_json_safe_value",
    )
    for name in names:
        assert getattr(project_data, name) is getattr(project_dataset_io, name)


def test_dataset_io_service_round_trips_mdhisto_data_and_context(tmp_path) -> None:
    data = MDHistoData(
        axes=(
            MDHistoAxis(
                "Energy", np.asarray([0.0, 1.0, 2.0]), "meV", "energy"
            ),
        ),
        signal=np.asarray([2.0, 4.0]),
        errors=np.asarray([0.2, 0.4]),
        mask=np.asarray([False, True]),
        num_events=np.asarray([3.0, 0.0]),
        metadata={"signal_semantics": "density", "array": np.asarray([1, 2])},
    )
    dataset = DatasetEntry(
        "scan", data, parameters={"temperature": 5.0, "magnetic_field": None}
    )
    path = tmp_path / "scan.npz"

    project_dataset_io.save_dataset_file(dataset, path, use_view=False)
    restored, context = project_dataset_io._load_nfit_dataset_file(path)

    assert isinstance(restored, MDHistoData)
    np.testing.assert_allclose(restored.signal, data.signal)
    np.testing.assert_array_equal(restored.mask, data.mask)
    assert restored.axes[0].name == "Energy"
    assert restored.metadata["array"] == [1, 2]
    assert context == {"temperature": 5.0, "magnetic_field": None}


def test_dataset_io_service_round_trips_point_list_data(tmp_path) -> None:
    data = PointListData(
        {"x": [1.0, 2.0], "signal": [3.0, 4.0]},
        units={"x": "K", "signal": "emu"},
        coordinate_names=["x"],
        channels=[{"label": "signal", "value": "signal"}],
        metadata={"nested": {"array": np.asarray([1.0])}},
    )
    path = tmp_path / "points.npz"

    project_dataset_io.save_dataset_file(
        DatasetEntry("points", data), path, use_view=False
    )
    restored, context = project_dataset_io._load_nfit_dataset_file(path)

    assert isinstance(restored, PointListData)
    np.testing.assert_allclose(restored.column("signal"), [3.0, 4.0])
    assert restored.coordinate_names == ["x"]
    assert restored.metadata["nested"] == {"array": [1.0]}
    assert context == {}


def test_dataset_io_service_has_no_gui_dependency() -> None:
    tree = ast.parse((PACKAGE_ROOT / "project_dataset_io.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        "PySide" in module or module.startswith("qt_") for module in imported_modules
    )
    assert "project_gui" not in imported_modules
