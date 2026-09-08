from __future__ import annotations

import ast
from pathlib import Path

from nfit import project_gui, project_imports
from nfit.dataset import PointData4D
from nfit.pipeline import DatasetEntry


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


def test_source_handler_registry_declares_dispatch_contract():
    handlers = project_imports._SOURCE_FORMAT_HANDLERS

    assert [handler.name for handler in handlers] == [
        "project_artifact",
        "raw_dgs",
        "registered_importer",
        "point_list",
        "nfit_npz",
        "mdevent",
        "mdhisto",
    ]
    assert {
        handler.name: handler.group_import_mode
        for handler in handlers
        if handler.group_import_mode is not None
    } == {"raw_dgs": "raw_dgs", "mdevent": "mdevent"}
    assert {
        handler.name: handler.entry_import_priority
        for handler in handlers
        if handler.entry_import_priority is not None
    } == {"registered_importer": 10, "point_list": 20, "nfit_npz": 0}


def test_group_import_dispatch_uses_registered_source_strategies(monkeypatch):
    monkeypatch.setattr(
        project_imports,
        "is_raw_dgs_nexus_file",
        lambda path: Path(path).name == "raw.nxs",
    )
    monkeypatch.setattr(
        project_imports,
        "is_mdevent_file",
        lambda path: Path(path).name == "events.nxs",
    )

    raw = project_imports._group_import_handler(
        Path("raw.nxs"), "single_crystal_inelastic"
    )
    events = project_imports._group_import_handler(
        Path("events.nxs"), "single_crystal_inelastic"
    )

    assert raw is not None and raw.name == "raw_dgs"
    assert events is not None and events.name == "mdevent"
    assert (
        project_imports._group_import_handler(Path("raw.nxs"), "powder_inelastic")
        is None
    )


def test_entry_and_lazy_dispatch_preserve_injected_loader_seams(tmp_path):
    archive_path = tmp_path / "archived.npz"
    nexus_path = tmp_path / "reduced.nxs"
    archive_data = PointData4D([0.0], [0.0], [0.0], [1.0], [2.0], [0.1])
    nexus_data = PointData4D([0.0], [0.0], [0.0], [2.0], [3.0], [0.2])
    archive_calls = []
    nexus_calls = []

    imported = project_imports.dataset_entry_from_path(
        archive_path,
        importer_name="explicit_but_archive_wins",
        dataset_file_loader=lambda path: (
            archive_calls.append(path) or archive_data,
            {"temperature": 5.0},
        ),
    )
    lazy = DatasetEntry(
        "reduced",
        None,
        kind="nxs",
        data_type="single_crystal_inelastic",
        metadata={"source_file": str(nexus_path), "import_status": "pending"},
    )
    loaded = project_imports._ensure_dataset_data_loaded(
        lazy,
        mdhisto_loader=lambda path, copy_metadata=False: (
            nexus_calls.append((path, copy_metadata)) or nexus_data
        ),
    )

    assert imported.data is archive_data
    assert imported.parameters["temperature"] == 5.0
    assert imported.metadata["importer"] == "explicit_but_archive_wins"
    assert archive_calls == [archive_path]
    assert loaded is nexus_data
    assert nexus_calls == [(nexus_path, False)]


def test_source_availability_contract_is_strategy_driven(tmp_path):
    raw = DatasetEntry(
        "raw",
        None,
        kind="raw_dgs_nexus",
        metadata={"source_file": str(tmp_path / "raw.nxs")},
    )
    registered = DatasetEntry(
        "table",
        None,
        kind="csv",
        data_type="single_crystal_inelastic",
        metadata={
            "source_file": str(tmp_path / "table.csv"),
            "importer": "custom",
        },
    )
    reduced = DatasetEntry(
        "reduced",
        None,
        kind="nxs",
        data_type="single_crystal_inelastic",
        metadata={"source_file": str(tmp_path / "reduced.nxs")},
    )

    assert not project_imports._dataset_can_load(raw)
    assert project_imports._dataset_can_reload(raw)
    assert not project_imports._dataset_can_load(registered)
    assert project_imports._dataset_can_reload(registered)
    assert project_imports._dataset_can_load(reduced)
    assert project_imports._dataset_can_reload(reduced)
