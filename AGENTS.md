# Agent Instructions

This repository should be developed and tested with the local `nfit` conda
environment.

Use this interpreter explicitly unless the user tells you otherwise:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python
```

Preferred test commands:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pytest -q
/Users/pmneves/anaconda3/envs/nfit/bin/python -m compileall -q src/nfit tests examples
/Users/pmneves/anaconda3/envs/nfit/bin/python -m ruff check
/Users/pmneves/anaconda3/envs/nfit/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Do not rely on the shell's default `python`; on this machine it may point to a
different Anaconda environment and can give misleading failures, especially for
Qt/PySide and NumPy-version behavior.

If the environment is missing, tell the user rather than silently switching to a
different Python.

## Branches and commits

- Work directly on `main` so the current application state is immediately
  available for interactive testing.
- Do not create or switch to a feature branch or worktree unless the user
  explicitly requests it.
- Commit each completed, coherent change before handing it back to the user
  unless the user explicitly asks for uncommitted work.
- A shared `main` worktree may already contain user or other-agent changes.
  Preserve them and stage only the exact files that belong to the completed
  change. Never sweep unrelated dirty files into a commit.
- Do not commit when relevant tests or validation are failing. Report the
  failure instead.

## Versioning and authorship

- Every commit must update the version consistently in `pyproject.toml`,
  `docs/conf.py`, `docs/index.md`, and `tests/test_packaging.py`.
- Use a patch increment for fixes, documentation, refactoring, and small
  backward-compatible behavior changes.
- Use a minor increment for substantial new backward-compatible features.
- Only a human may change the major version. Agents must never increment it.
- Keep the authorship statement intact: this project was authored by Paul M.
  Neves (Johns Hopkins University, pneves1@jhu.edu) with use of LLM coding
  tools.

## Project map

- `src/nfit/project_gui.py`: project explorer and GUI workflows.
- `src/nfit/qt_slice_viewer.py`: interactive data viewer.
- `src/nfit/fit_config.py`: model-component compilation and parameter mapping.
- `src/nfit/fitting.py`: optimizer-facing fitting framework.
- `src/nfit/cross_section.py`: neutron cross-section and susceptibility
  conversions.
- `src/nfit/analysis/`: non-destructive analysis operations.
- `docs/physics_conventions.md`: authoritative physics and unit conventions.
- `docs/gui_workflows.md`: GUI documentation entry point.
- `docs/data_import.md`, `docs/gui_fitting.md`, and `docs/data_viewer.md`:
  detailed GUI workflows.
- `docs/data_philosophy.md`: extension, serialization, and data contracts.

Keep this map structural rather than exhaustive. Update it only when subsystem
ownership or documentation entry points move.

## Data-container changes

- Treat the numerical payloads of `PointData4D`, `PointListData`, and
  `MDHistoData` as immutable.
- Use `with_updates(...)` for a direct replacement or `mutable_copy()` for
  isolated working arrays.
- Install changed project data through `DatasetEntry.replace_data(...)`; never
  assign `dataset.data` directly. Set `source_backed=True` only in import or
  lazy-load paths that exactly reproduce the referenced source file.
- Any cache derived from dataset contents must include
  `dataset.data_cache_token` or an equivalently complete content signature.

## Documentation maintenance

- When changing user-facing behavior, update the relevant source documentation in
  `docs/` during the same change.
- When adding or removing a user-facing capability, update the concise feature
  summary in `README.md` and the appropriate Sphinx page or navigation entry.
- Write for a user or developer trying to understand the current package.
  Prefer direct, task-oriented prose; remove development history, planning
  narration, repeated caveats, and step-by-step widget descriptions that do not
  help someone use or extend the feature.
- Keep overview pages short and route detail to focused pages. Preserve
  information needed to use, validate, or extend nfit even when tightening the
  prose.
- Define every physics quantity at first use and state its units or
  dimensionless convention. Keep equations dimensionally consistent and use
  `docs/physics_conventions.md` as the canonical source for shared notation,
  normalization, and unit conversions.
- Treat `docs/gui_workflows.md` as the GUI entry point. Put detailed import,
  fitting, viewer, physics, and API behavior in their focused pages rather than
  growing that overview into a second manual.
- Keep documentation, tests, and tooltips consistent. New user-interactable GUI
  controls should include useful hover text and test coverage that prevents
  missing tooltip regressions.
- Build Sphinx with warnings treated as errors after changing documentation or
  public behavior.

`AGENTS.md` should contain stable workflow rules and pointers, not detailed
feature descriptions or scientific equations. The Sphinx pages remain the
authoritative source for behavior and conventions.

## Before committing

- Run tests proportional to the change; run the full suite for broad or risky
  changes.
- Run Ruff, byte-compilation, and `git diff --check`.
- Rebuild Sphinx when documentation or public behavior changes.
- Confirm that the four version declarations agree.
- Review the staged diff and verify that it contains no unrelated user changes.
