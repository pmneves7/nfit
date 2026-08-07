# nfit

`nfit` analyzes magnetic neutron-scattering and related bulk measurements.
It works with physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy
transfer, independently of the originating instrument or file layout.

The package provides:

- native MDHisto, MDEvent, and compatible direct-geometry spectrometer workflows;
- powder-cut, MPMS magnetization, PPMS heat-capacity, and tabular imports;
- masking, coverage-aware symmetry/rebinning and composites, backgrounds, and
  non-destructive analyses;
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
- single-file `.nfit` projects with generated analysis artifacts, fit timelines,
  provenance, safe transactional script editing, and GUI detection of external
  file changes, plus editable script export for dataset preparation, analyses,
  fits, and saved plots;
- interactive slice, waterfall, and volumetric visualization with independent,
  same-state viewer duplication.

See the [documentation](docs/index.md) for workflows, physics conventions,
model equations, and the Python API.

## Authorship

nfit was authored by Paul M. Neves (Johns Hopkins University,
pneves1@jhu.edu) with use of LLM coding tools.
See [References and software influences](docs/references.md) for the scientific
and software projects that informed nfit.

## Installation

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
