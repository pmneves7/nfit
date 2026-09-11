# nfit

<img src="docs/_static/nfit-logo.svg" alt="nfit logo" width="240">

`nfit` analyzes magnetic neutron-scattering and related bulk measurements.
It works with physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy
transfer, independently of the originating instrument or file layout.

## Install nfit

Download the current application from [GitHub Releases](https://github.com/pmneves7/nfit/releases/latest),
then follow the [macOS, Windows, or Linux installation instructions](docs/desktop_installers.md).
The same guide covers system requirements, updates, and uninstalling nfit.

On macOS, a [local app launcher](docs/getting_started.md#macos-local-app-launcher)
provides double-click startup and a Dock icon using your existing environment.

The package provides:

- native MDHisto, single-crystal and direct-powder MDEvent, and compatible
  direct-geometry spectrometer workflows, with project-owned composite
  materialization and scaled powder or aligned-histogram background subtraction;
- NIST NCNR MACS NeXus import with separate energy-analyzed SPEC and
  energy-integrated DIFF streams, DAVE-compatible analyzer-energy recovery,
  monitor normalization, detector masks, and viewer-ready
  batch composites that can join existing compatible collections, with promoted
  crystal orientation;
- powder-cut, MPMS magnetization, PPMS heat-capacity, and tabular imports;
- masking, coverage- and sample-aware symmetry/rebinning with per-axis
  Discrete, Step, Bins, Edges, or tolerance-clustered modes, editable
  fractional/discrete point assignment, expandable bin-grid information,
  multiple named fit or visualization binnings per dataset and composite,
  unified timed, cancellable dataset- and point-level rebin progress with stage-resolved
  native MDEvent reduction,
  momentum-coordinate matrices, automatic
  zero-aligned uniform grids specified by bin centers, composites, backgrounds
  with automatic grid matching for unrebinned neutron references, and
  non-destructive analyses, including live hierarchical composites and editable
  source-linked histogram sums, differences, or comparison clones with
  independent raw-data binning, plus lazy navigation for run-heavy projects;
- metadata dimensions for temperature or other sample conditions, accumulated with physical coordinates in the central N-dimensional rebinner,
  with selectable channels, nominal coordinates, display and integration axes,
  exact tiled slices, visible zero-valued background references, temperature-aware
  fitting, and per-axis discrete or fractional rebinning;
- source reload actions for individual datasets and complete dataset collections,
  with automatic refresh of dependent composites and open viewers;
- folder-level enable controls for masks and backgrounds, plus a shared scale
  editor for all backgrounds in a folder;
- paired neutron cross-section and dynamical-susceptibility channels for datasets
  and composites, with temperature-coordinate conversion for temperature series;
- local, MMP, generalized-paramagnon/damped-mode, conserved-ferromagnetic,
  relaxational or inertial Heisenberg-RPA, and coupled-susceptibility models
  for inelastic, quasistatic elastic, and optional bulk-response comparisons,
  with a tabbed crystal/exchange workflow and explicit powder and
  self-consistency convergence controls;
- a linked bare Lindhard response with complex multiband susceptibility,
  Cartesian spin and neutron projections, fixed-filling support, automatic
  formula-unit and represented-magnetic-center normalization, per-orbital
  tabulated, custom, or coherent effective magnetic form factors, workflow-focused
  controls, fitting, model-owned plots, and editable scripts;
- modular scalar Stoner, user-matrix, and local multiorbital Hubbard--Hund
  RPA interaction dressings with eV input, canonical meV vertices, shared
  electronic-response dependencies, pole diagnostics, fitting, and reports;
- production electronic-response controls with bounded transition batches,
  fit-shared response caches, reusable parameter-resolved momentum
  Hamiltonians, factorized orbital-pair responses, resource-aware
  independent-Q scheduling, workload-gated fused CPU and end-to-end optional
  GPU contractions, model-digest-aware eigensystem and bare-response reuse,
  exact commensurate-Q permutation,
  digest-bearing tolerance-certified periodic Q interpolation with nonlinear
  RPA revalidation and reusable fit geometry, certified little-group
  reduction, observable-specific automatic DOS certificates and dataset-domain
  complete TB--Lindhard--RPA mesh certificates,
  backend-equivalence probes, separate
  mesh/broadening convergence plots, and scheduler-neutral response chunks
  with editable Slurm launchers;
- CIF/manual crystal geometry and arbitrary manual or Wannier90 tight-binding
  models with explicit local orbital frames, site-point-group harmonic
  subspaces, symmetry-allowed onsite terms, selectable orbital-explicit
  Slater--Koster or general-matrix hopping candidates, lazy primitive-cell
  resolution, Hinuma/HPKOT and Setyawan--Curtarolo paths, eV/meV conversion,
  bands, orbital projections, Gaussian and linear-tetrahedron density of
  states, Fermi surfaces, bounded CPU and optional
  GPU eigensystem backends, certified total-DOS symmetry reduction,
  implicit/collinear/spinor bases,
  manifold-resolved onsite spin-orbit coupling, Hamiltonian and orbital-block
  matrix inspection, shared 3D model-geometry inspection, and
  primitive-cell first-Brillouin-zone views; atoms, orbitals, and bonds in the
  model-geometry viewer are clickable and identified in its side panel, while
  all electronic viewers use a common plot-plus-settings layout and the band,
  DOS, and Fermi viewers own their plot-specific calculation controls; DOS
  production meshes are selected and certified in the model editor with
  per-iteration convergence progress, while Fermi grids may use a physical
  reciprocal-space spacing or explicit axis sizes. Band and DOS figures
  also expose scriptable curve, marker, typography, frame, legend, Fermi-level,
  projection-visibility, and high-symmetry-guide styling, while DOS energy
  limits can be derived automatically from the sampled bands,
  band, DOS, and Fermi-surface viewers can copy or save their rendered figures,
  three-dimensional Fermi surfaces use GPU-accelerated interaction with
  scriptable opacity, text, grid, legend, and smooth/flat shading controls, and the
  Brillouin-zone viewer provides scriptable styling, visibility, projection,
  and image-export controls; named onsite, hopping, and SOC coefficients share
  the standard bounds, fit-selection, dataset-sharing, report, and script
  machinery;
- an extensible model registry shared by fitting, GUI metadata, diagnostics,
  reports, plots, project files, and workflow scripts, with a tiered
  category/model selector for primitive, spin-fluctuation, electronic,
  heat-capacity, and magnetization components;
- simultaneous least-squares fitting with resource-aware parallel numerical
  derivatives, explicit uncertainty conventions, optional
  differential-evolution initialization, and `emcee` posterior sampling;
- single-file `.nfit` projects with optional persisted rebin caches, temporary
  disk overflow for binnings that exceed the memory cache, generated
  analysis artifacts, fit timelines,
  provenance, safe transactional script editing, and GUI detection of external
  file changes, installation-local remembered file-dialog locations, plus
  editable script export for dataset preparation, analyses, fits, and saved
  plots;
- interactive slice, waterfall, tiled 2D-slice, and volumetric visualization
  with large-volume safeguards, optional visualization-only smoothing across
  empty bins, per-panel tiled cursor inspection, optional metric- and
  centering-aware Brillouin-zone boundaries with scriptable line styling,
  dedicated experimental momentum-path maps with automatic or absolute-HKL
  higher-zone paths, propagated tube-average errors, symmetry guides, and 3D
  inspection,
  customizable tiled-slice labels,
  independent, same-state viewer duplication and stored plot recipes that
  retain independent source/composite-rebin configurations, separate save/update
  actions for stored plots, and provide named,
  previewed Matplotlib, cmcrameri, Colorcet, cmocean, MyCarta, and CartoColors
  color maps, plus drop-in custom RGB palette files accessible through
  **File → Preferences**, installation-local continuous and waterfall colormap
  defaults, and display-only Gaussian interpolation into adjacent empty bins,
  with toolbar **Help** opening the local documentation;
- machine-local rebin performance defaults, cancellable machine calibration and
  configuration-specific benchmarks with timing/memory reports and script export;
- multi-selection dataset operations and recursive copy/paste for nested
  dataset groups.

See the [documentation](docs/index.md) for workflows, physics conventions,
model equations, and the Python API.

## Authorship

nfit was authored and is maintained by Paul M. Neves (Johns Hopkins
University, pneves1@jhu.edu). AI and large-language-model coding tools have
assisted with portions of the code, tests, and documentation. Authorship and
responsibility for the project remain with Paul M. Neves.
See [References and software influences](docs/references.md) for the scientific
and software projects that informed nfit.

## Installation

nfit requires Python 3.12 or newer. The supplied conda environment uses
Python 3.14 and a current NumPy/SciPy/Numba and Qt/VTK stack.

Native macOS, 64-bit Windows 10/11, and 64-bit Ubuntu 22.04 or newer beta
installers provide a self-contained application with the nfit icon, a branded
loading screen, offline help, and consent-based verified updates. Testers do
not need Python, Conda, Git, or the source repository. See
[Desktop installers and updates](docs/desktop_installers.md) for download,
installation, system requirements, and update instructions.

If you are new to GitHub, Python, and Conda, follow the step-by-step
[first-time setup guide](docs/getting_started.md#first-time-collaborator-setup).
It covers accepting a GitHub invitation (or downloading a ZIP), installing Git
and Miniforge, creating the `nfit` environment, and launching the application.

For an existing checkout:

```bash
conda env update -f environment.yml --prune
conda activate nfit
```

Launch the project explorer with:

```bash
nfit
```

## Development

```bash
python -m pytest -q
python -m pytest --cov=nfit --cov-branch --cov-report=term-missing
python -m ruff check
```

## Example

Run the small script-only analysis example:

```bash
python examples/data_playground.py
```

Build the documentation with warnings treated as errors:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

Validate release artifacts locally with:

```bash
python -m build
python -m twine check dist/*
python -m pip install dist/nfit-*.whl
nfit
```

## Documentation

Documentation is built with Sphinx, MyST Markdown, MyST-NB/Jupyter notebooks,
and the Furo theme. User-facing behavior, tests, tooltips, and the relevant
source page should change together.
