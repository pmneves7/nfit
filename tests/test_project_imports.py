from __future__ import annotations

import ast
from pathlib import Path

from nfit import project_gui, project_imports


def test_project_import_service_has_no_gui_dependency():
    source = Path(project_imports.__file__).read_text()
    tree = ast.parse(source)

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "project_gui" not in imported_modules
    assert ".project_gui" not in imported_modules


def test_project_gui_reexports_import_service_contract():
    assert project_gui.DATA_TYPE_DEFINITIONS is project_imports.DATA_TYPE_DEFINITIONS
    assert project_gui.available_data_types is project_imports.available_data_types
    assert project_gui.data_type_label is project_imports.data_type_label
    assert project_gui.data_type_container is project_imports.data_type_container
    assert project_gui.default_importer_for_data_type is (
        project_imports.default_importer_for_data_type
    )
    assert project_gui.parse_dataset_numors is project_imports.parse_dataset_numors
    assert project_gui.set_dataset_source is project_imports.set_dataset_source
    assert project_gui.set_dataset_data_type is project_imports.set_dataset_data_type
