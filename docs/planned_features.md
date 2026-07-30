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

## Electronic-response models

Current tight-binding, Lindhard, and RPA capabilities are documented under
[Electronic-structure and electronic-response models](electronic_structure_models.md).
The [electronic-response design contract](electronic_response_contract.md)
defines the conventions that future extensions must preserve.

Planned electronic-structure extensions include:

- nonorthogonal bases with an explicit overlap matrix $S(\mathbf k)$;
- user-supplied symmetry representations for general numerical or Wannier
  bases;
- additional external interfaces when they preserve basis, gauge, unit, spin,
  and provenance information;
- spin-dependent generated hopping and intentionally time-reversal-breaking
  electronic Hamiltonians; and
- richer orbital-resolved controls for electronic plots.

Planned response models include:

- BCS/Nambu quasiparticles, gap functions, coherence factors, and the
  superconducting particle--hole response;
- frequency- and momentum-dependent self-energies that separate static onsite
  coefficients from quasiparticle renormalization and incoherent weight;
- auxiliary-boson or slave-particle responses for systems where separate
  itinerant and local degrees of freedom are a useful controlled
  approximation;
- additional interaction channels and a spinor-aware Hubbard--Hund vertex;
  and
- interfaces to advanced solvers such as DMFT or Bethe--Salpeter workflows
  without embedding those solvers in nfit.

Performance work may add wider operator-aware symmetry reduction,
workload-specific compiled or sparse kernels, and more complete accelerator
support. Such paths must be checked against the serial float64/complex128
reference and retain deterministic provenance.

Automatic Brillouin-zone density selection should use observable-specific
certificates: an energy-resolved DOS norm, topology and geometric distance for
Fermi surfaces, and complex matrix errors for Lindhard response. A selected
mesh should remain fixed during a fit so adaptive refinement does not make the
objective discontinuous. Electronic-RPA stability scans should similarly
converge the static feedback over a declared q mesh; current diagnostics report
evaluated zero-energy points and do not claim a global stability proof.

A capability is considered implemented only when its public calculation,
fitting and plotting behavior, scripts, reports, documentation, and validation
agree.

Instrument resolution, finite-bin integration, absorption, and related
measurement effects will form an optional dataset-owned systematics layer
before comparison with observations. This layer is separate from the
electronic-response implementation.

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

- Extend the workflow graph to grouped raw direct-geometry and MDEvent
  reductions, derived-analysis dependencies, workspace composites, group
  backgrounds, plots, and complete active project state.
- Add target-specific export from plots, workspaces, and whole
  projects. Each export should contain the dependency closure needed for that
  target, while fit history remains an interactive convenience.
- Expand behavior-equivalence tests as each workflow node becomes exportable.

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
