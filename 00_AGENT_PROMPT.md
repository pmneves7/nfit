# Agent Prompt: Build a 4D Inelastic-Neutron Susceptibility Analysis Package

You are helping build a Python package for analyzing four-dimensional inelastic neutron scattering data from nearly magnetic frustrated metals. The core task is to model and fit the dynamical magnetic susceptibility, especially the imaginary part chi''(Q,E), across H, K, L, and energy-transfer coordinates.

The user is an experimental condensed matter physicist studying itinerant or nearly itinerant magnetic fluctuations, including systems such as YMn2, beta-Mn-like metals, LiV2O4-like metals, rare-earth and transition-metal intermetallics, and other frustrated metallic magnets. The package should be scientifically transparent, well commented, modular, and robust enough for exploratory analysis and later publication-quality fitting.

## Primary goals

1. Load, organize, mask, visualize, and fit 4D neutron data: H, K, L, E, intensity, uncertainty, and metadata.
2. Model magnetic chi''(Q,E,T) using several physically motivated forms:
   - Overdamped paramagnon / relaxational susceptibility.
   - Moriya/SCR-inspired inverse-susceptibility forms.
   - Symmetry-constrained momentum-space expansions using Heisenberg-like exchange harmonics J(Q), interpreted phenomenologically.
   - Broad-shell, ridge, or multi-Q frustrated fluctuation models.
3. Fit model parameters to data with uncertainties.
4. Support global fits across multiple temperatures, incident energies, or datasets.
5. Compute useful derived quantities:
   - Static chi(Q,T).
   - Relaxation rate Gamma_Q(T).
   - Correlation lengths or inverse widths.
   - Local susceptibility chi''_local(E).
   - Fluctuating moment estimates from sum rules.
   - QFI-related integrals and candidate normalizations.
6. Keep the code modular so that model functions are independent of any single optimizer.
7. Prefer correctness, clarity, and testability before performance. Add JAX, Numba, or Dask only after the NumPy/SciPy version is validated.

## Recommended language and stack

Use Python as the primary language.

Core dependencies:
- numpy
- scipy
- matplotlib
- pandas, optional
- xarray for labeled multidimensional data
- h5py, zarr, or netCDF4 for storage
- lmfit for convenient named-parameter nonlinear fits
- scipy.optimize.least_squares for lower-level deterministic fitting

Optional later dependencies:
- dask for larger-than-memory arrays and parallel operations
- numba or jax for accelerated model evaluation
- emcee or pymc for posterior sampling
- mantid Python bindings for neutron data interoperability, when available
- ase or spglib only if crystal symmetry support becomes useful

Do not start with a monolithic GUI. Start with a clean Python package plus notebooks/scripts.

## Architectural principle

Separate the project into layers:

- data_io.py: import/export, HDF5/Zarr/NeXus/Mantid/Horace interoperability
- dataset.py: common data structures, masks, metadata, coordinate conventions
- coordinates.py: reciprocal lattice, BZ folding, coordinate transforms, symmetry operations
- models.py: chi'' models and static susceptibility models
- lattice_harmonics.py: symmetry-allowed J(Q) or Lambda(Q) expansions
- cross_section.py: neutron magnetic cross section, Bose factor, form factor, polarization factor
- resolution.py: resolution convolution or Monte Carlo resolution sampling, initially optional
- fitting.py: residual functions, parameter packing, deterministic fitting
- uncertainty.py: covariance, bootstrap, profile likelihood, MCMC wrappers
- qfi.py: QFI integrals, moment integrals, normalization options
- visualization.py: cuts, slices, residual maps, fit reports
- tests/: unit tests for formulas, shapes, limiting cases, synthetic data recovery

Each physical model should be a pure function that maps coordinates and parameters to chi'' or intensity. Optimizers should call these pure functions; models should not depend on lmfit, emcee, plotting, or file formats.

## Scientific conventions

Use energy transfer E = hbar omega. Store E in meV by default. Let the code support hbar = 1 style internally, but document units explicitly.

Use reciprocal lattice units (H,K,L) as default coordinates, but provide conversion to absolute Q in inverse Angstroms for magnetic form factors and moment sum rules.

Model neutron intensity as something like:

I_model(Q,E) = scale * magnetic_cross_section(Q,E) + background(Q,E)

where magnetic_cross_section may include:
- magnetic form factor |f(Q)|^2
- polarization factor, if needed
- Bose population factor: S(Q,E) = chi''(Q,E) / [1 - exp(-E/kBT)] up to constants
- absolute-unit prefactors, eventually
- resolution convolution, optionally

Keep chi'' and measured S(Q,E) conceptually separate.

## Near-term deliverables

1. Minimal package skeleton with modern pyproject.toml packaging metadata and tests.
2. Data class for flattened 4D points: H, K, L, E, intensity, sigma, mask, temperature, metadata.
3. Basic model: relaxational overdamped susceptibility with one or more Q centers.
4. Lattice-harmonic model for chi^{-1}(Q,E): r(T) + Lambda(Q) - i E/Gamma or r(T)+Lambda(Q)-i C E.
5. Fitting interface using scipy.optimize.least_squares and lmfit.
6. Synthetic-data generator with known parameters.
7. Unit test that synthetic parameters are recovered within tolerance.
8. Plotting tools for observed slices, model slices, and residuals.
9. Documentation website built with Sphinx, MyST Markdown, MyST-NB/Jupyter notebooks, and Read the Docs; it should explain installation, examples, physics conventions, scientific references, approximations, limitations, and interpretation rules.
10. Robust reference examples that double as user tutorials, human-checkable validation cases, and regression checks for known scientific workflows.
11. Eventual package distribution so users can install with a command like `pip install metallix`.

## Documentation goals

The documentation should make the package usable by other researchers, not only
by the original author. Treat the website as part of the scientific interface.

Include:

- a short README that points to full documentation.
- getting-started workflow from installation to first fit.
- Sphinx documentation configuration, MyST Markdown pages, MyST-NB/Jupyter notebook examples, and Read the Docs hosting configuration.
- packaging notes for editable development installs and eventual `pip install metallix` distribution.
- physics convention pages for chi'', S(Q,E), energy transfer, RLU coordinates, Bose factor, magnetic form factor, polarization factor, and absolute normalization.
- model-reference pages with equations, parameter meanings, units, limits, and references.
- examples that can be run against the current API.
- reference examples with known inputs, expected outputs, tolerance checks, and plots for human inspection.
- approximation and caveat notes for finite measurement windows, backgrounds, resolution, phenomenological lattice harmonics, uncertainty estimates, QFI-related quantities, and moment integrals.
- annotated references explaining which equations or assumptions each source supports.

## Coding style

- Write typed, readable Python.
- Use dataclasses or pydantic-like structures only where they simplify code; avoid overengineering.
- Use docstrings with equations and units.
- Include references with links to papers where relevant equations are developed as functions.
- Include nice human and agent readable documentation for all code.
- All arrays should be vectorized over N points.
- Validate shapes and units early.
- Make functions deterministic and testable.
- Avoid hidden global state.
- Keep notebooks as examples, not core logic.

## Important warning

Do not treat fitted J_n coefficients as literal microscopic exchange constants unless the user explicitly decides to. In itinerant metals they are usually phenomenological symmetry-allowed coefficients in an inverse susceptibility, capturing low-resolution momentum dependence induced by band structure, interactions, matrix elements, and frustration.
