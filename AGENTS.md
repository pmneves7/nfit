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
```

Do not rely on the shell's default `python`; on this machine it may point to a
different Anaconda environment and can give misleading failures, especially for
Qt/PySide and NumPy-version behavior.

If the environment is missing, tell the user rather than silently switching to a
different Python.

Versioning and authorship:

- Update the project version in `pyproject.toml` and `docs/conf.py` as part of
  every commit. Use a reasonable semantic-version bump for the behavioral scope
  of the commit.
- Keep the authorship statement intact: this project was authored by Paul M.
  Neves (Johns Hopkins University, pneves1@jhu.edu) with use of LLM coding
  tools.

Documentation and wiki maintenance:

- When changing user-facing behavior, update the relevant source documentation in
  `docs/` during the same change.
- Treat `docs/gui_workflows.md` as the in-repository wiki for the current GUI
  and data-viewer workflow. Update it whenever GUI controls, fit-history
  behavior, data-viewer behavior, or project-file workflow changes.
- Keep documentation, tests, and tooltips consistent. New user-interactable GUI
  controls should include useful hover text and test coverage that prevents
  missing tooltip regressions.
