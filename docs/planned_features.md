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
