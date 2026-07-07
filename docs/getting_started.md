# Getting started

## Recommended conda setup

Use a dedicated conda environment for `metallix`. This avoids mixing compiled
scientific packages from the project with your base Python installation.

```bash
cd ~/code/metallix
conda env create -f environment.yml
conda activate metallix
```

If the environment already exists, update it after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
conda activate metallix
```

The environment uses conda-forge for NumPy, SciPy, Matplotlib, pytest, Sphinx,
MyST Markdown, MyST-NB, and the documentation theme. It also installs `metallix`
in editable mode with:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python -m pip install -e ".[dev,docs]"
```

## Validate the install

Run the test suite:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python -m pytest -q
```

Run the first reference example:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python examples/synthetic_single_q_fit.py
```

Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

The long-term distribution goal is:

```bash
pip install metallix
```
