# nfit

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pmneves7/nfit/main/docs/_static/nfit-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pmneves7/nfit/main/docs/_static/nfit-logo.svg">
  <img src="https://raw.githubusercontent.com/pmneves7/nfit/main/docs/_static/nfit-logo.svg" alt="nfit logo" width="240">
</picture>

[![Documentation Status](https://readthedocs.org/projects/nfit/badge/?version=stable)](https://nfit.readthedocs.io/en/stable/)

**Documentation:** [nfit.readthedocs.io](https://nfit.readthedocs.io/en/stable/)

> **Experimental beta software:** nfit is under active development and may
> contain incomplete or incorrect behavior. Validate scientific results independently.
> Paul M. Neves developed nfit with assistance from AI coding tools.

`nfit` analyzes magnetic neutron-scattering and related bulk measurements.
It works with physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy
transfer, independently of the originating instrument or file layout.

## Install nfit

Download the current application from [GitHub Releases](https://github.com/pmneves7/nfit/releases/latest),
then follow the [macOS, Windows, or Linux installation instructions](https://nfit.readthedocs.io/en/stable/desktop_installers.html).
The same guide covers system requirements, updates, and uninstalling nfit.

To install the Python package in an active Python 3.12 or newer environment:

```bash
python -m pip install nfit
nfit
```

On macOS, a [local app launcher](https://nfit.readthedocs.io/en/stable/getting_started.html#macos-local-app-launcher)
provides double-click startup and a Dock icon using your existing environment.

A few of its main capabilities are:

- import, combine, mask, rebin, and visualize multidimensional neutron data,
  with parent-owned hierarchical binning and explicit background recipe context,
  including raw ARCS, CNCS, HYSPEC, and SEQUOIA single-crystal or powder data,
  raw CORELLI finite-energy correlation-chopper reconstruction with fractional
  momentum and discrete reconstructed-energy assignment, and reduced
  CORELLI and WAND² single-crystal data, with conventional major-tick or
  Brillouin-zone gridline overlays, trajectory-projected powder-background
  subtraction with masked correction windows, directional measured-background
  replay at sample angles with bounded compiled parallel acceleration, reversible displayed-axis
  coarsening, held cross-dataset viewer settings, symmetric color limits, and
  CSV export of maps and cuts;
- fit magnetic response models alongside bulk magnetization and heat-capacity data;
- open viewers on demand, with an optional preference to preload all datasets
  and binnings for faster switching;
- construct and inspect crystal, tight-binding, Lindhard, and RPA models;
- save complete `.nfit` projects, load cached binnings on demand, reuse unchanged
  compressed artifacts when saving, and manage cached results through shared
  CPU/RAM preferences, memory estimates, automatic disk-backed loading of large
  saved histograms under memory pressure, and optional session disk caching;
- organize project items with drag and drop, batch copy, paste, and delete,
  including lazy run pages and nested collections;
- guard editor and viewer interactions during rebinning, saves, and background
  tasks while keeping progress and cancellation controls available;
- export editable analysis scripts; and
- use native desktop installers or the Python package on macOS, Windows, and Linux.

See the [detailed feature overview](https://nfit.readthedocs.io/en/stable/features.html)
for the complete capability list.

See the [documentation](https://nfit.readthedocs.io/en/stable/) for workflows, physics conventions,
model equations, and the Python API.

## Authorship

nfit was authored and is maintained by Paul M. Neves (Johns Hopkins
University, pneves1@jhu.edu). AI and large-language-model coding tools have
assisted with portions of the code, tests, and documentation. Authorship and
responsibility for the project remain with Paul M. Neves.
See [References and software influences](https://nfit.readthedocs.io/en/stable/references.html) for the scientific
and software projects that informed nfit.

## Installation

nfit requires Python 3.12 or newer. The supplied conda environment uses
Python 3.14 and a current NumPy/SciPy/Numba and Qt/VTK stack.

Native macOS, 64-bit Windows 10/11, and 64-bit Ubuntu 22.04 or newer beta
installers provide a self-contained application with the nfit icon, a branded
loading screen, offline help, and consent-based verified updates. Testers do
not need Python, Conda, Git, or the source repository. See
[Desktop installers and updates](https://nfit.readthedocs.io/en/stable/desktop_installers.html) for download,
installation, system requirements, and update instructions.

If you are new to Git, Python, and Conda, follow the step-by-step
[first-time source setup guide](https://nfit.readthedocs.io/en/stable/getting_started.html#first-time-source-setup).
It covers downloading the public source, installing Git and Miniforge, creating
the `nfit` environment, and launching the application.

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

Clone the public repository and create the tested development environment:

```bash
git clone https://github.com/pmneves7/nfit.git
cd nfit
conda env create -f environment.yml
conda activate nfit
python -m pip install -e ".[dev,docs,distribution]"
nfit
```

For an existing checkout, use `conda env update -f environment.yml --prune`
before refreshing the editable installation. See the
[developer setup guide](https://nfit.readthedocs.io/en/stable/getting_started.html#developer-setup) for the full
workflow and platform notes.

Run the development checks with:

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
