# nfit documentation

```{warning}
**nfit is experimental beta software.** It is under active development and may
contain incomplete or incorrect behavior. Validate scientific results independently.
Paul M. Neves developed nfit with assistance from AI coding tools.
```

`nfit` analyzes magnetic neutron-scattering and related bulk measurements in
physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy transfer. It
supports reduced MDHisto data, file-backed MDEvent, compatible direct-geometry
spectrometer events, raw finite-energy CORELLI correlation-chopper reconstruction,
NIST NCNR MACS NeXus data, powder cuts, magnetization, heat capacity, and general
point tables.

nfit was authored and is maintained by Paul M. Neves (Johns Hopkins
University, pneves1@jhu.edu). AI and large-language-model coding tools have
assisted with portions of the code, tests, and documentation. Authorship and
responsibility for the project remain with Paul M. Neves.

Current package version: 0.93.0.

Start with [Getting started](getting_started.md), then use
[GUI workflows](gui_workflows.md) for interactive work or the
[API reference](api.md) for scripts. Scientific definitions are collected in
[Physics conventions](physics_conventions.md).

```{toctree}
:maxdepth: 2
:caption: Start here

getting_started
desktop_installers
features
gui_workflows
workflow_scripts
data_import
gui_fitting
data_viewer
data_playground
plotting
```

```{toctree}
:maxdepth: 2
:caption: Physics and models

physics_conventions
electronic_structure_models
spin_fluctuation_models
powder_convergence
electronic_response_contract
curie_weiss
heat_capacity
theory_notes
```

```{toctree}
:maxdepth: 2
:caption: Fitting and development

modeling_pipeline
fit_constraints
fit_reports
performance
data_philosophy
api
references
planned_features
```
