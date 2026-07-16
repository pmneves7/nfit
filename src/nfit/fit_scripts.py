"""Readable backend-only script generation for saved fit states."""

from __future__ import annotations

from pathlib import Path


def fit_state_script(*, project_path: str | Path, group_name: str, fit_id: str) -> str:
    """Return a script that restores, and optionally reruns, one saved fit state."""

    return "\n".join(
        [
            "from pathlib import Path",
            "",
            "from nfit import load_project, restore_data_group_state, run_group_fit",
            "",
            f"PROJECT_PATH = Path({str(project_path)!r})",
            f"# Generated from workspace {group_name!r}.",
            f"FIT_ID = {fit_id!r}",
            "RUN_FIT = False  # Set True to run the optimizer and append a new result.",
            "",
            "def walk_fit_entries(entries):",
            "    for entry in entries:",
            "        yield entry",
            "        yield from walk_fit_entries(entry.children)",
            "",
            "def main():",
            "    project = load_project(PROJECT_PATH)",
            "    group = next(",
            "        item for item in project.data_groups",
            "        if any(entry.id == FIT_ID for entry in walk_fit_entries(item.fits))",
            "    )",
            "    fit_state = next(entry for entry in walk_fit_entries(group.fits) if entry.id == FIT_ID)",
            "    restore_data_group_state(group, fit_state.snapshot)",
            "    if RUN_FIT:",
            "        result = run_group_fit(group, fit_state)",
            "        print(result.goodness)",
            "    else:",
            "        print('Restored:', fit_state.name)",
            "        print(fit_state.goodness)",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )
