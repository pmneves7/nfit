"""Keep event-mask reduction independent of GUI facade modules."""

import ast
from pathlib import Path


def test_event_masks_has_no_gui_or_facade_imports():
    source_path = Path(__file__).parents[1] / "src" / "nfit" / "event_masks.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden = ("PySide", "PyQt", "nfit.project_gui", "nfit.project_data", "project_gui", "project_data")
    assert not any(name.startswith(forbidden) for name in imported)


def test_native_mdevent_reducers_use_gui_independent_mask_helper():
    source = (Path(__file__).parents[1] / "src" / "nfit" / "mdevent.py").read_text(
        encoding="utf-8"
    )
    assert "from .event_masks import reduce_masked_event_runs" in source
    assert "reduce_masked_event_runs(" in source
