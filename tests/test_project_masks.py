from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import nfit.project_coordinates as project_coordinates
import nfit.project_data as project_data
import nfit.project_masks as project_masks
from nfit.dataset import PointData4D
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DatasetEntry, MaskSpec

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"


def _point_data() -> PointData4D:
    return PointData4D(
        H=[0.0, 1.0, 2.0],
        K=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0],
        E=[-1.0, 0.0, 1.0],
        intensity=[10.0, 20.0, 30.0],
        sigma=[1.0, 2.0, 3.0],
        mask=[True, True, False],
        metadata={"coordinate_units": "angstrom^-1"},
    )


def _mdhisto_data() -> MDHistoData:
    return MDHistoData(
        axes=(
            MDHistoAxis("H", np.asarray([-0.5, 0.5, 1.5]), "rlu", "momentum"),
            MDHistoAxis(
                "DeltaE",
                np.asarray([-1.5, -0.5, 0.5, 1.5]),
                "meV",
                "energy_transfer",
            ),
        ),
        signal=np.arange(6.0).reshape(2, 3),
        errors=np.ones((2, 3)),
        mask=np.asarray(
            [[False, False, False], [False, False, True]], dtype=bool
        ),
        num_events=np.ones((2, 3)),
    )


def test_project_data_reexports_mask_and_coordinate_service_objects() -> None:
    mask_names = (
        "_point_data_with_nfit_masks",
        "_evaluate_point_data_mask",
        "_mdhisto_with_nfit_masks",
        "_evaluate_mdhisto_mask",
        "_mdhisto_coordinate_range_mask",
        "_mdhisto_q_matrix",
    )
    coordinate_names = (
        "_coordinate_range_axis_specs",
        "_mdhisto_coordinate_grids",
        "_mdhisto_rebin_axis_vector",
        "_axis_projection_vector",
        "_project_hkle_vector",
    )
    for name in mask_names:
        assert getattr(project_data, name) is getattr(project_masks, name)
    for name in coordinate_names:
        assert getattr(project_data, name) is getattr(project_coordinates, name)


def test_point_mask_service_preserves_source_payload_and_facade_result() -> None:
    data = _point_data()
    dataset = DatasetEntry(
        "points",
        data,
        masks=[
            MaskSpec(
                "middle", "coordinate_range", parameters={"H": [0.5, 1.5]}
            )
        ],
    )
    source_mask = data.mask.copy()
    source_intensity = data.intensity.copy()

    via_service = project_masks._point_data_with_nfit_masks(dataset, data)
    via_facade = project_data._point_data_with_nfit_masks(dataset, data)

    np.testing.assert_array_equal(via_service.mask, [True, False, False])
    np.testing.assert_array_equal(via_facade.mask, via_service.mask)
    np.testing.assert_array_equal(data.mask, source_mask)
    np.testing.assert_allclose(data.intensity, source_intensity)
    assert via_service.metadata["nfit_mask_count"] == 1
    assert via_service.metadata["combined_mask_count"] == 2


def test_mdhisto_mask_service_combines_file_and_coordinate_masks() -> None:
    data = _mdhisto_data()
    dataset = DatasetEntry(
        "histogram",
        data,
        masks=[
            MaskSpec(
                "elastic",
                "coordinate_range",
                parameters={"E": [-0.1, 0.1]},
            )
        ],
    )
    source_mask = data.mask.copy()

    masked = project_masks._mdhisto_with_nfit_masks(dataset)

    expected = source_mask.copy()
    expected[:, 1] = True
    np.testing.assert_array_equal(masked.mask, expected)
    np.testing.assert_array_equal(data.mask, source_mask)
    assert masked.metadata["file_mask_count"] == 1
    assert masked.metadata["nfit_mask_count"] == 2
    assert masked.metadata["combined_mask_count"] == 3


def test_projected_coordinate_resolution_rejects_incomplete_hkle_basis() -> None:
    coords = {
        "H": np.asarray([1.0, 2.0]),
        "K": np.asarray([3.0, 4.0]),
        "L": np.asarray([5.0, 6.0]),
    }

    projected = project_coordinates._resolve_projected_axis_grid(
        [1.0, -1.0, 0.5], coords
    )

    np.testing.assert_allclose(projected, [0.5, 1.0])
    assert (
        project_coordinates._resolve_projected_axis_grid(
            [1.0, 0.0, 0.0, 1.0], coords
        )
        is None
    )


def test_mask_and_coordinate_services_have_no_facade_or_gui_dependency() -> None:
    for filename in ("project_masks.py", "project_coordinates.py"):
        tree = ast.parse((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )

        assert not any(
            "PySide" in module or module.startswith("qt_")
            for module in imported_modules
        )
        assert "project_gui" not in imported_modules
        assert "project_data" not in imported_modules
        assert "rebin" not in imported_modules
