# metallix

`metallix` is an early-stage Python package for analyzing four-dimensional
inelastic neutron scattering data from nearly magnetic metals. The first
implementation focuses on clean, testable building blocks:

- flattened 4D point data containers for H, K, L, and energy transfer E.
- overdamped paramagnon susceptibility models returning chi''(Q,E).
- neutron cross-section helpers that convert chi'' into measured intensity.
- deterministic least-squares fitting for synthetic-data recovery.
- Mantid `SaveMD` / `MDHistoWorkspace` NeXus import for binned 4D data.
- simple plotting helpers for 1D cuts, 2D maps, and MDHisto slices.
- a PySide6 MDHisto slice viewer with axis integration, color controls,
  cursor readout, histogram box cuts, channel selection, and script export.

The package is intentionally small at this stage. It does not yet include
resolution convolution, MCMC, Dask, Numba, or JAX.

## Recommended development environment

Use a project-specific conda environment from conda-forge. This keeps compiled
scientific dependencies such as NumPy, SciPy, BLAS, and Fortran runtimes isolated
from your base Python installation.

```bash
cd ~/code/metallix
conda env create -f environment.yml
conda activate metallix
```

If the environment already exists and `environment.yml` changes, update it with:

```bash
conda env update -f environment.yml --prune
conda activate metallix
```

The environment installs the package in editable mode with development and
documentation extras.

## Tests

```bash
pytest
```

Run the first reference example:

```bash
python examples/synthetic_single_q_fit.py
```

Import a Mantid MDHisto NeXus file and inspect the axes/channels:

```bash
python examples/import_hyspec_mdhisto_nxs.py
```

Open the PySide6 slice viewer for the HYSPEC example data:

```bash
python examples/view_hyspec_mdhisto_slice.py
```

The slice viewer can display `signal`, propagated `errors`, `num_events`
(`multiplicity`), or `mask`; choose x/y axes, integrate hidden axes, tune color
normalization, copy the figure to the clipboard, or export a static
`plot_mdhisto_slice(...)` script for notebooks and batch figure generation.

Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

The eventual goal is a normal release install:

```bash
pip install metallix
```

## Documentation

Documentation is planned around Sphinx, MyST Markdown, MyST-NB/Jupyter notebooks,
and Read the Docs. Initial source pages live in `docs/`.
