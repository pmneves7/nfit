# nfit

`nfit` analyzes magnetic neutron-scattering and related bulk measurements.
It works with physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy
transfer, independently of the originating instrument or file layout.

The package provides:

- native MDHisto, MDEvent, and compatible direct-geometry spectrometer workflows;
- powder-cut, MPMS magnetization, PPMS heat-capacity, and tabular imports;
- masking, symmetry-aware rebinning, backgrounds, and non-destructive analyses;
- local, MMP, generalized-paramagnon/damped-mode, and Heisenberg-RPA
  spin-fluctuation models for inelastic, quasistatic elastic, and optional
  bulk-response comparisons;
- a linked bare Lindhard response with complex multiband susceptibility,
  Cartesian spin and neutron projections, fixed-filling support, full-mesh
  reference evaluation, fitting, model-owned plots, and editable scripts;
- modular scalar Stoner, user-matrix, and local multiorbital Hubbard--Hund
  RPA interaction dressings with eV input, canonical meV vertices, shared
  electronic-response dependencies, pole diagnostics, fitting, and reports;
- production electronic-response controls with bounded transition batches,
  model-digest-aware eigensystem reuse, certified little-group reduction,
  backend-equivalence probes, separate mesh/broadening convergence plots, and
  scheduler-neutral response chunks with editable Slurm launchers;
- CIF/manual crystal geometry and arbitrary manual or Wannier90 tight-binding
  models with explicit local orbital frames, site-point-group harmonic
  subspaces, symmetry-allowed onsite terms, selectable orbital-explicit
  Slater--Koster or general-matrix hopping candidates, lazy primitive-cell
  resolution, Hinuma/HPKOT paths, eV/meV conversion, bands, orbital
  projections, density of states, Fermi surfaces, bounded CPU and optional
  GPU eigensystem backends, certified total-DOS symmetry reduction,
  implicit/collinear/spinor bases,
  manifold-resolved onsite spin-orbit coupling, Hamiltonian and orbital-block
  matrix inspection, shared 3D model-geometry inspection, and
  primitive-cell first-Brillouin-zone views; atoms, orbitals, and bonds in the
  model-geometry viewer are clickable and identified in its side panel, while
  all electronic viewers use a common plot-plus-settings layout,
  three-dimensional Fermi surfaces use GPU-accelerated interaction, and the
  Brillouin-zone viewer provides scriptable styling, visibility, projection,
  and image-export controls; named onsite, hopping, and SOC coefficients share
  the standard bounds, fit-selection, dataset-sharing, report, and script
  machinery;
- an extensible model registry shared by fitting, GUI metadata, diagnostics,
  reports, plots, project files, and workflow scripts;
- simultaneous least-squares fitting with explicit uncertainty conventions,
  optional differential-evolution initialization, and `emcee` posterior
  sampling;
- project files with fit timelines and provenance, plus editable script export
  for dataset preparation, analyses, fits, and saved plots;
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
