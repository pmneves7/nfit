# Local Agent Task List

Use this as a concrete checklist for an agentic coding system.

## Environment for all tasks

Use the local `metallix` conda environment explicitly:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python
```

Run tests with:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python -m pytest -q
```

Do not use the shell's default `python`; it may point to a different Anaconda
environment and produce misleading Qt/PySide or NumPy-version failures.

## Task 1: Create repository skeleton

Create a Python package with `src`, `tests`, and `examples`. Add a modern
`pyproject.toml` with packaging metadata compatible with an eventual distribution
named `metallix`. Add README with a short project description.

Run:

```bash
/Users/pmneves/.conda/envs/metallix/bin/python -m pip install -e .[dev]
/Users/pmneves/.conda/envs/metallix/bin/python -m pytest -q
```

Do not proceed until import and empty tests work.

## Task 2: Implement PointData4D

Implement the `PointData4D` dataclass in `dataset.py`. Add tests for:

- shape validation
- valid-mask generation
- sigma <= 0 handling
- NaN handling

## Task 3: Implement basic paramagnon model

Implement:

- `quadratic_distance_rlu`
- `paramagnon_chipp`
- `multi_q_paramagnon_chipp`
- `relaxational_chipp`

Add tests for vectorization and limiting behavior.

## Task 4: Implement cross-section wrapper

Implement:

- `bose_denominator`
- `scattering_from_chipp`

Add tests near E=0 and at positive E.

## Task 5: Implement deterministic fitting

Implement:

- `ParameterSpec`
- parameter packing/unpacking
- `fit_least_squares`
- `FitResult`

Then generate synthetic data from the paramagnon model and recover parameters.

## Task 6: Add example notebook or script

Create `examples/synthetic_single_q_fit.py` or notebook that:

1. Generates synthetic H,K,L,E points.
2. Adds Gaussian noise.
3. Fits the model.
4. Prints best-fit parameters and uncertainties.
5. Plots data/model/residual for an energy cut and a Q cut.

## Task 7: Implement lattice harmonics

Implement user-defined neighbor-shell harmonic functions:

`gamma_shell(H,K,L,shell_vectors)`

where shell_vectors are fractional lattice coordinates. Use:


gamma(Q) = sum_d cos(2 pi [H d_x + K d_y + L d_z]).

Then implement:

`lambda_q(H,K,L, shells, coefficients)`

and a harmonic relaxational model.

## Task 8: Add global fit support

Support fitting multiple datasets with shared parameters. Start simple: concatenate residuals from several `PointData4D` objects and allow a model function to know dataset index or metadata.

## Task 9: Add uncertainty tools

Implement bootstrap and profile likelihood. Do not start MCMC until deterministic fits and synthetic tests are reliable.

## Task 10: Add QFI and moment integral tools

Implement finite-window integrals with clear assumptions. Return structured results that include:

- value
- integration bounds
- temperature
- normalization convention
- whether result is lower-bound or model-extrapolated

## Task 11: Add documentation website

Create a `docs/` tree and a static documentation build using Sphinx, MyST
Markdown, MyST-NB/Jupyter notebooks, and Read the Docs configuration. Start with
source pages even if some content is initially concise.

Required pages:

- home page with package scope and current limitations.
- getting-started guide from install to first synthetic fit.
- physics conventions page for chi'', S(Q,E), Bose factor, form factor, polarization factor, RLU coordinates, energy-transfer sign, units, and normalization assumptions.
- model reference with equations, parameters, units, validity regimes, and limiting behavior.
- fitting and uncertainty guide.
- derived quantities guide for static susceptibility, local susceptibility, moments, and QFI-related integrals.
- approximations and caveats page.
- annotated scientific references page.
- examples gallery linked to scripts or notebooks.
- distribution page explaining local editable installs and the eventual `pip install metallix` installation path.

Acceptance checks:

- documentation builds locally.
- Sphinx configuration is committed.
- MyST Markdown pages and MyST-NB/Jupyter notebook examples render.
- Read the Docs configuration is present when public hosting is enabled.
- links between pages work.
- equations and code examples render clearly.
- the quickstart can be run by a new user after installation.
- public functions have docstrings that agree with the website conventions.

## Task 12: Add robust reference examples

Create examples that are both user tutorials and validation cases for human and
automated checking.

Initial reference examples:

- synthetic single-Q paramagnon fit with known parameters.
- noisy single-Q fit with uncertainty and residual diagnostics.
- lattice-harmonic fit with user-defined shells.
- two-temperature global fit with shared and temperature-dependent parameters.
- finite-window moment and QFI-related integral example.
- caveat example showing a deliberately imperfect background or finite-window model.

Each example should include:

- script or notebook source.
- deterministic input data or fixed random seed.
- expected fitted parameters and derived quantities with tolerances.
- compact JSON or YAML summary output for automated comparison.
- plots suitable for human review and documentation.
- short notes explaining conventions, assumptions, and interpretation limits.

Acceptance checks:

- examples run from a clean install.
- automated validation compares numerical summaries to stored references.
- generated plots can be inspected manually.
- examples are linked from the documentation website.
- reference output updates are intentional and explained.

## Far-future backlog: polarized neutron scattering

Do not start this until the unpolarized data model, fitting workflow, uncertainty
tools, normalization handling, and resolution treatment are stable.

Eventually add support for polarized neutron scattering measurements:

- represent spin-flip and non-spin-flip channels explicitly.
- store incident/analyzer polarization directions and guide-field or XYZ coordinate conventions.
- support flipping-ratio and polarization-efficiency corrections.
- model leakage between polarization channels with propagated uncertainties.
- keep magnetic, nuclear, spin-incoherent, and background terms distinguishable when the experiment supports it.
- add synthetic tests before using real polarized datasets.

## Coding constraints for the agent

- Keep code simple and readable.
- Do not introduce a GUI.
- Do not introduce JAX, Numba, Dask, PyMC, or Mantid until the base implementation passes tests.
- Do not implement polarized neutron scattering support until it is explicitly promoted from the far-future backlog.
- Use modern `pyproject.toml` packaging and plan for an eventual `pip install metallix` release.
- Write tests before or alongside each feature.
- Do not hard-code a specific material.
- Do not silently assume absolute normalization.
- Do not conflate chi'' with measured intensity S(Q,E).
- Keep documentation, docstrings, tests, and examples consistent about physics conventions and units.
- Keep reference examples deterministic and use documented tolerances for validation.
