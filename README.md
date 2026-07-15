# nfit

`nfit` is an early-stage Python package for analyzing reduced magnetic
scattering and related experimental data from nearly magnetic metals. The
name stands for neutron fitting and N-dimensional fitting. The
package is meant to work in physical coordinates such as `H`, `K`, `L`, `|Q|`,
and energy transfer after data reduction, regardless of whether the data came
from HYSPEC, another spectrometer, a triple-axis experiment, a powder
diffractometer, or a text export. The first implementation focuses on clean,
testable building blocks:

- flattened 4D point data containers for H, K, L, and energy transfer E.
- generic axis-role inference from file labels and units.
- overdamped paramagnon susceptibility models returning chi''(Q,E).
- neutron cross-section helpers that convert chi'' into measured intensity.
- deterministic least-squares fitting for synthetic-data recovery.
- a simultaneous-fitting framework with dataset weights, preprocessing
  transforms, model hooks, resolution hooks, and uncertainty-sampling
  placeholders.
- data groups and model sessions for related datasets, flexible metadata,
  per-dataset parameter bindings, optimizer settings, and fit history.
- constant and additive compound measured-intensity models, plus dataset-level
  parameter bindings for shared or grouped constraints.
- Gaussian energy-resolution broadening with constant or polynomial FWHM and
  configurable energy oversampling.
- composable masks for energy windows, `|Q|` ranges, projected boxes,
  ellipsoids, and phonon-like cones.
- Mantid `SaveMD` / `MDHistoWorkspace` NeXus import as one reduced-data adapter.
- simple plotting helpers for 1D cuts, 2D maps, and MDHisto slices.
- static data/fit/residual comparison plots for matching MDHisto outputs.
- a PySide6 MDHisto slice viewer with axis integration, color controls,
  cursor readout, histogram box cuts, channel selection, fit-comparison panels,
  and script export.
- a PySide6 project explorer GUI launched with `nfit` for organizing
  workspaces, datasets, masks, models, rebinned views, and fit timelines.

The package is intentionally small at this stage. MCMC has framework entry
points but not a concrete backend yet; Dask, Numba, and JAX are not included.

## Authorship

Nfit was authored by Paul M. Neves (Johns Hopkins University,
pneves1@jhu.edu) with use of LLM coding tools.

## Recommended development environment

Use a project-specific conda environment from conda-forge. This keeps compiled
scientific dependencies such as NumPy, SciPy, BLAS, and Fortran runtimes isolated
from your base Python installation.

```bash
cd ~/code/nfit
conda env create -f environment.yml
conda activate nfit
```

If the environment already exists and `environment.yml` changes, update it with:

```bash
conda env update -f environment.yml --prune
conda activate nfit
```

The environment installs the package in editable mode with development and
documentation extras.

## Tests

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pytest -q
```

Lint the Data Playground implementation:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m ruff check
```

Agents and local automation should use the explicit `nfit` conda
interpreter above rather than the shell's default `python`.

Run the first reference example:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/synthetic_single_q_fit.py
```

Import a Mantid MDHisto NeXus file and inspect the axes/channels. This is the
current example adapter; the fitting layer is not specific to HYSPEC or MDHisto:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/import_hyspec_mdhisto_nxs.py
```

Open the PySide6 slice viewer for the bundled MDHisto example data:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/view_hyspec_mdhisto_slice.py
```

Launch the project explorer GUI:

```bash
nfit
```

The project explorer can create workspaces, import datasets, inspect structured
metadata, add masks and models, configure rebinning, run fits, and open the data
viewer. The docs page `docs/gui_workflows.md` is the in-repository wiki for the
current GUI behavior. Rebinning honors masks up front: enabled file, inherited
group, and dataset masks are applied before point or MDHisto data contribute to
rebinned bins, while disabled masks are ignored.

Render the local HYSPEC test datasets (`1D_test.nxs`, `2D_test.nxs`, and
`4D_test.nxs`) with automatic 1D line plotting and 2D+ slice plotting:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/plot_hyspec_test_datasets.py
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/plot_hyspec_test_datasets.py --channel multiplicity --save-dir /tmp/hyspec-plots
```

The slice viewer can display `signal`, propagated `errors`, `num_events`
(`multiplicity`), `combined_mask`, `file_mask`, or `nfit_mask`; choose x/y
axes, integrate hidden axes, tune color normalization, switch between loaded
datasets, copy the figure to the clipboard,
compare attached fit results as linked data/fit/residual panels, or export a
static `plot_mdhisto_slice(...)` script for notebooks and batch figure
generation. `load_mantid_mdhisto_nxs` reads only the core MDHisto arrays by
default; pass `copy_metadata=True` only when you need a shallow FAIR-style copy
of ancillary NeXus metadata such as experiment groups.

Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

The eventual goal is a normal release install:

```bash
pip install nfit
```

Once published, that command should install the package and expose the GUI
launcher:

```bash
nfit
```

Until the project is ready for public release, validate packaging locally with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m build
/Users/pmneves/anaconda3/envs/nfit/bin/python -m twine check dist/*
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pip install dist/nfit-*.whl
nfit
```

## Documentation

Documentation is built with Sphinx, MyST Markdown, MyST-NB/Jupyter notebooks,
and Read the Docs-compatible source pages in `docs/`. When GUI or workflow
behavior changes, update both the relevant docs page and the wiki-style
workflow page in `docs/gui_workflows.md`.
