# Planned features

This page lists possible extensions, not compatibility promises or scheduled
releases. Current behavior is documented in the workflow and API pages.

## Scientific analysis

- Expose polarized-neutron SF, NSF, and chiral channels. The tensor RPA kernel
  already retains the full dissipative susceptibility tensor, but only the
  unpolarized contraction is currently available to users.
- Add general multidimensional instrument-resolution convolution. The current
  resolution API supports identity and Gaussian energy broadening.
- Add reusable bootstrap and profile-likelihood uncertainty workflows alongside
  covariance estimates and MCMC.
- Add aligned single-crystal dataset subtraction with uncertainty propagation.
  Powder-background subtraction and analysis recipes already provide parts of
  the required infrastructure.
- Evaluate additional temperature-dependent closures only when their assumptions
  and parameter identifiability can be documented and tested.

## Distribution and interoperability

- Publish versioned releases to a package index when release automation and
  binary-dependency testing are ready.
- Expand import/export adapters where they improve interoperability without
  coupling the fitting layer to an instrument-specific representation.
- Support optional external model engines such as Sunny, SpinW, SpinInteract,
  PyCrystalField, and command-line DFT or DMRG workflows. The first real
  integration should determine the interface. It must preserve a standalone
  `pip install nfit`, record the engine and version in projects and scripts,
  and make parameter units and returned physical quantities explicit.

## Reproducible scripting

- Extend editable script export beyond fits and saved plots to data imports,
  transformations, masks, rebinning, analyses, and complete batch workflows.
- Add behavior-equivalence tests showing that exported scripts reproduce the
  scientific project state without creating Qt widgets.

## Tutorials

- Create a polished first-fit tutorial using a small synthetic dataset. Cover
  import, units and conventions, masking, model construction, parameter links,
  fitting, diagnostics, saved plots, reports, and project reopening.
- Add focused tutorials for powder INS, magnetization, heat capacity, and the
  Analysis Window. Show both the GUI workflow and the corresponding public API
  where that comparison is useful.
- Add a complete direct-geometry tutorial when a redistributable dataset is
  available. Include raw-event reduction, MDHisto export, and versioned expected
  outputs that can be checked in automated tests.
- Keep tutorial inputs small, outputs reproducible, and commands exercised in
  continuous integration so examples do not silently become obsolete.
