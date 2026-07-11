# Getting started

## Recommended conda setup

Use a dedicated conda environment for `nfit`. This avoids mixing compiled
scientific packages from the project with your base Python installation.

```bash
cd ~/code/nfit
conda env create -f environment.yml
conda activate nfit
```

If the environment already exists, update it after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
conda activate nfit
```

The environment uses conda-forge for NumPy, SciPy, Matplotlib, pytest, Sphinx,
MyST Markdown, MyST-NB, and the documentation theme. It also installs `nfit`
in editable mode with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pip install -e ".[dev,docs]"
```

## Validate the install

Run the test suite:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pytest -q
```

Launch the graphical project explorer:

```bash
nfit
```

The GUI opens the project explorer, where users can create workspaces, import
datasets, add masks and models, inspect metadata, run fits, and open the data
viewer. See [GUI workflows](gui_workflows.md) for the current user-facing
workflow and tooltip/documentation expectations.

Run the first reference example:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/synthetic_single_q_fit.py
```

Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

The long-term distribution goal is:

```bash
pip install nfit
```

After the package is published, `pip install nfit` should install the runtime
dependencies and expose the GUI launcher:

```bash
nfit
```

Before publishing, check release artifacts locally:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m build
/Users/pmneves/anaconda3/envs/nfit/bin/python -m twine check dist/*
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pip install dist/nfit-*.whl
nfit
```
