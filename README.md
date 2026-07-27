# nfit

`nfit` analyzes magnetic neutron-scattering and related bulk measurements.
It works with physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy
transfer, independently of the originating instrument or file layout.

The package provides:

- native MDHisto, MDEvent, and raw direct-geometry NeXus workflows;
- powder-cut, MPMS magnetization, PPMS heat-capacity, and tabular imports;
- masking, symmetry-aware rebinning, backgrounds, and non-destructive analyses;
- relaxational, MMP, and Heisenberg-RPA spin-fluctuation models;
- simultaneous least-squares fitting, optional differential-evolution
  initialization, and `emcee` posterior sampling;
- project files with fit timelines, provenance, saved plots, and script export;
- interactive slice, waterfall, and volumetric visualization.

See the [documentation](docs/index.md) for workflows, physics conventions,
model equations, and the Python API.

## Authorship

Nfit was authored by Paul M. Neves (Johns Hopkins University,
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
python -m ruff check
```

The repository’s `AGENTS.md` specifies the exact local interpreter used by
automation on the primary development machine. Interactive users should
activate the `nfit` environment and use its `python`.

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
