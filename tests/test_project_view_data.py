from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nfit.project_data as project_data
import nfit.project_view_data as project_view_data

VIEW_DATA_EXPORTS = (
    "KINEMATIC_KF_KI_INCLUDED_KEY",
    "_apply_dataset_scale",
    "_apply_kinematic_normalization_to_points",
    "_apply_kinematic_normalization_to_view",
    "_apply_spectral_channel_view",
    "_kinematic_energy_metadata",
    "_kinematic_kf_ki_factor",
    "_mdhisto_without_nfit_masks",
    "_with_viewer_dataset_metadata",
)


@pytest.mark.parametrize("name", VIEW_DATA_EXPORTS)
def test_project_data_reexports_view_data_objects(name: str) -> None:
    assert getattr(project_data, name) is getattr(project_view_data, name)


def test_view_data_service_does_not_import_project_facade_or_qt() -> None:
    path = Path(project_view_data.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert "project_data" not in imported_modules
    assert not any("PySide" in module or module.startswith("qt") for module in imported_modules)
