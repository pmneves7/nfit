# metallix

`metallix` is an early-stage Python package for analyzing four-dimensional
inelastic neutron scattering data from nearly magnetic metals. The first
implementation focuses on clean, testable building blocks:

- flattened 4D point data containers for H, K, L, and energy transfer E.
- overdamped paramagnon susceptibility models returning chi''(Q,E).
- neutron cross-section helpers that convert chi'' into measured intensity.
- deterministic least-squares fitting for synthetic-data recovery.
- simple plotting helpers for 1D cuts and 2D maps.

The package is intentionally small at this stage. It does not yet include GUI
tools, Mantid interoperability, resolution convolution, MCMC, Dask, Numba, or
JAX.

## Development install

```bash
pip install -e ".[dev]"
```

Run tests with:

```bash
pytest
```

The eventual goal is a normal release install:

```bash
pip install metallix
```

## Documentation

Documentation is planned around Sphinx, MyST Markdown, MyST-NB/Jupyter notebooks,
and Read the Docs. Initial source pages live in `docs/`.

