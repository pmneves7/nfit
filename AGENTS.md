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
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pytest --cov=nfit --cov-branch --cov-report=term-missing
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

- `src/nfit/project_gui.py`: project explorer and GUI workflow coordination.
- `src/nfit/project_group_panels.py`, `project_dataset_panels.py`, and
  `project_data_panels.py`: focused dataset/composite detail builders and their
  compatibility dispatch.
- `src/nfit/project_window_builder.py`, `project_model_editor.py`,
  `project_lindhard_editor.py`, and `project_tight_binding_editor.py`: project
  window construction and focused model-editor presentation helpers.
- `src/nfit/qt_widget_state.py`: shared Qt editor focus, tab, and scroll
  preservation during panel rebuilds.
- `src/nfit/qt_operation_guard.py`: shared operation input and popup-lifetime
  guards for Qt workflows.
- `src/nfit/project_imports.py`: GUI-independent import, source, and reload orchestration.
- `src/nfit/source_selection.py`, `source_selection_imports.py`, and
  `source_selection_gui.py`: bounded run expressions, resolved previews, atomic
  imports into existing dataset groups, and focused Qt selection controls.
- `src/nfit/source_lineage.py`: bounded acquisition identity metadata and shared-
  source guards for combination and separate fit blocks.
- `src/nfit/project_data.py`: GUI-independent dataset preparation facade,
  derived-data coordination, and viewer-ready orchestration.
- `src/nfit/project_composites.py`: hierarchical composite planning,
  materialization, caching, and background alignment.
- `src/nfit/project_background_cache.py`: persistence of background histograms
  owned by sample binnings.
- `src/nfit/project_dataset_io.py`, `project_point_lists.py`, `project_masks.py`,
  and `project_coordinates.py`: dataset archives, point-list preparation,
  masks, and coordinate projection services.
- `src/nfit/project_rebinning.py`: numerical grid configuration, symmetry-aware
  rebinning, and histogram construction.
- `src/nfit/raw_dgs.py`, `raw_dgs_geometry.py`, `raw_dgs_geometry_precision.py`,
  `raw_dgs_hyspec.py`, and `raw_dgs_cache.py`: direct-geometry reduction, resolved
  instrument geometry, HYSPEC preprocessing, and dataset-owned reduced-event caches.
- `src/nfit/mdevent.py` and `mdevent_detector_masks.py`: reduced-event import,
  trajectory normalization, and retained detector-mask metadata.
- `src/nfit/event_bin_indices.py`: shared arbitrary-edge event membership,
  explicitly re-exported by the MDE importer for compatibility.
- `src/nfit/cached_background_replay.py` and `background_profile_queries.py`:
  bounded measured-background recipes, aligned composite propagation, and
  original-grid profile selection and covariance replay.
- `src/nfit/qt_background_profiles.py`: asynchronous background box-profile
  replay, stale-result guards, and viewer export readiness.
- `src/nfit/dgs_reduction_policy.py` and `dgs_reduction_settings.py`: shared DGS
  numerical conventions and public policy configuration; the focused
  `dgs_reduction_policy_gui.py` presents those choices.
- `src/nfit/dgs_event_accumulation.py` and `_dgs_event_numba.py`: shared ordered
  event projection/accumulation using the policy-owned affine transform.
- `src/nfit/reduction_recipes.py`, `reduction_runtime.py`, and
  `reduction_recipe_gui.py`: source/reduction schemas, execution provenance,
  targeted edits, public replay, and focused Qt controls.
- `src/nfit/corelli.py`, `_corelli_numba.py`, and `corelli_constants.py`: CORELLI
  reconstruction, its compiled kernels, and their shared numerical conventions.
- `src/nfit/project_derived_grid.py`: shared output-grid planning for live
  derived dataset arithmetic.
- `src/nfit/project_rebin_panels.py`: shared Qt controls for dataset and
  composite rebin settings, metadata dimensions, and bin summaries.
- `src/nfit/project_binning_policy.py`: read-only rebin ownership and background
  recipe-consumption presentation policy.
- `src/nfit/project_view_data.py`: immutable scale, kinematic, spectral-channel,
  and metadata transformations for viewer-ready datasets.
- `src/nfit/project_summary.py`: bounded, reusable numerical counts for
  dataset summaries without retaining their array payloads.
- `src/nfit/project_viewer_loading.py` and `viewer_data.py`: metadata-only
  viewer catalogs and on-demand dataset preparation.
- `src/nfit/project_io.py`, `project_history.py`, and `project_models.py`: project
  serialization, fit-history snapshots, and model-state reconciliation.
- `src/nfit/mapped_archive.py`: bounded temporary-disk array loading and
  accounting for heap and mapped numerical storage.
- `src/nfit/data_workspace.py`: operation-owned temporary scientific storage
  and save-destination ownership without startup filesystem dependencies.
- `src/nfit/project_clipboard.py`: GUI-independent project clipboard validation,
  copying, and reference remapping.
- `src/nfit/project_background_panels.py`: background link and shared source-symmetry controls.
- `src/nfit/project_import_dialogs.py`, `fit_diagnostics_gui.py`, and
  `analysis_window_builder.py`: focused Qt presentation helpers.
- `src/nfit/qt_slice_viewer.py`, `qt_slice_modes.py`, and `plotting_core.py`:
  interactive viewer coordination, slice-mode controllers, and plot primitives.
- `src/nfit/box_cuts.py`, `qt_rotated_box.py`, and `qt_box_cut_viewers.py`:
  box-profile arithmetic, rotated selection interaction, and live cut viewers.
- `src/nfit/viewer_export.py`: GUI-independent profile, map, and waterfall CSV export.
- `src/nfit/figure_export.py` and `qt_figure_export.py`: GUI-independent figure
  file export and its Qt destination chooser.
- `src/nfit/project_caches.py`: project cache eviction and model-overlay cache ownership.
- `src/nfit/app_distribution.py`, `app_updates.py`, `app_updates_gui.py`, and
  `tools/distribution/`: desktop packaging metadata, verified updates, installers,
  and startup presentation.
- `src/nfit/fit_config.py`, `fit_config_electronic.py`, and
  `fit_config_heisenberg_rpa.py`: model-component compilation and parameter mapping.
- `src/nfit/fitting.py`: optimizer-facing fitting framework.
- `src/nfit/cross_section.py`: neutron cross-section and susceptibility
  conversions.
- `src/nfit/analysis/`: non-destructive analysis operations.
- `src/nfit/workflow.py`: dependency graphs and human-readable script export.
- `src/nfit/composite_workflow.py`: portable source closure and composite recipe replay.
- `docs/physics_conventions.md`: authoritative physics and unit conventions.
- `docs/gui_workflows.md`: GUI documentation entry point.
- `docs/data_import.md`, `docs/gui_fitting.md`, and `docs/data_viewer.md`:
  detailed GUI workflows.
- `docs/data_philosophy.md`: extension, serialization, and data contracts.
- `src/nfit/measurement_contracts.py` and `measurement_statistics.py`: explicit
  statistical targets, serializable assumptions, and one-bin reference estimators.
- `src/nfit/measurement_profiles.py` and `box_cuts.py`: shared prepared histogram
  profiles, retained measurement payloads, and box-coordinate selection.
- `src/nfit/measurement_aggregation.py`, `measurement_rebinning.py`, and
  `measurement_point_bins.py`: retained estimator statistics and declared binning.
- `src/nfit/measurement_dependencies.py`, `measurement_scaling.py`, and
  `measurement_replay.py`: bounded primitive sensitivities, calibration, and
  original-source replay.
- `src/nfit/measurement_regions.py`, `measurement_waterfalls.py`,
  `measurement_fit_data.py`, and `measurement_likelihoods.py`: statistical
  region/trace preparation, fit payloads, and declared objectives.
- `src/nfit/measurement_diagnostics.py`, `measurement_diagnostics_gui.py`, and
  `measurement_smoothing.py`: statistical provenance, model-specific final-bin
  intervals, and optional source-aware count smoothing.
- `src/nfit/collection_workflow_gui.py`: focused dataset-group workflow sections.
- `docs/measurement_statistics.md`: measurement contract and estimator scripting guide.

Keep this map structural rather than exhaustive. Update it only when subsystem
ownership or documentation entry points move.

## Maintainability boundaries

- Put new behavior in the focused module that owns the relevant domain. Keep
  `project_gui.py`, `project_data.py`, `qt_slice_viewer.py`, `fit_config.py`,
  and `model_registry.py` as coordinators or compatibility facades rather than
  growing new self-contained subsystems inside them.
- Keep dependencies one-way: GUI modules may call GUI-independent services, but
  services must not import GUI coordinators or Qt. Focused services must not
  reverse-import their facade; pass narrow callbacks, protocols, or context
  objects when orchestration is required.
- Maintain one authoritative definition for constants, registries, caches, and
  domain rules. Compatibility modules may re-export those objects explicitly,
  but must not duplicate their values or implementations.
- Treat public import paths, saved-project schemas, scripting APIs, subclass
  hooks, and documented extension points as compatibility contracts. Preserve
  them during extraction or provide an explicit migration/deprecation path.
- Extract reusable builders, editors, panels, or numerical services when a new
  feature would add a second responsibility to an existing module. Prefer a
  small cohesive module over adding another large conditional branch to a
  coordinator.
- Add or extend architecture tests whenever module ownership changes. Tests
  should enforce GUI independence, import direction, facade identity where
  compatibility matters, and behavior equivalence across GUI and scripting
  entry points.

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

## GUI and scripting parity

- Every scientific GUI action must use a public, human-readable scripting API.
  GUI-only state is acceptable for presentation details, not for imports,
  transforms, fits, plots, or analyses.
- Batchable GUI workflows must be exportable as editable scripts. New GUI
  actions must extend the appropriate script generator and add a round-trip or
  behavior-equivalence test.
- Do not claim scripting parity for a workflow until its complete scientific
  configuration can be saved and rerun without constructing Qt widgets. Track
  remaining gaps in `docs/planned_features.md`.

## Before committing

- Run tests proportional to the change; run the full suite for broad or risky
  changes.
- Run Ruff, byte-compilation, and `git diff --check`.
- Rebuild Sphinx when documentation or public behavior changes.
- Confirm that the four version declarations agree.
- Review the staged diff and verify that it contains no unrelated user changes.
