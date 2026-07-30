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

Electronic-response models are being added in independently testable stages.
The tight-binding, bare Lindhard, and RPA interaction-dressing stages are
available; correlated-electron responses beyond RPA remain planned. Their conventions, initial
scope, provenance requirements, dependency policy, validation matrix, and
stage gates are fixed in the
[Electronic-response design contract](electronic_response_contract.md).
The CIF-to-orbital manual construction workflow is specified separately in the
[Tight-binding model-builder plan](tight_binding_builder_plan.md).

The shared model registry, generalized paramagnon/damped-mode response, and
arbitrary tight-binding electronic-structure layer are implemented. The
CIF/manual builder now includes orbital manifolds, local frames, static onsite
invariants, symmetry-constrained hoppings, hopping-path inspection, a shared
tight-binding/Heisenberg 3D geometry viewer, and a labelled 3D
Brillouin-zone viewer. Named electronic Hamiltonian terms now share the common
parameter, bounds, fit-selection, dataset-sharing, report, and script
machinery. Optional explicit collinear and spinor representations,
manifold-resolved onsite spin-orbit coupling, time-reversal validation, and a
matrix/subspace inspector are implemented. GUI-built models now provide
Slater--Koster and general symmetry-matrix hopping conventions, lazy canonical
Hamiltonian resolution, primitive-cell folding, and Hinuma/HPKOT standard
paths. Full-precision electronic eigensystems now support bounded serial or
threaded CPU batches, explicit optional CuPy execution, immutable-model
caches, and certified symmetry policies for total-DOS meshes. The linked
Lindhard response now provides a complex bare susceptibility, Cartesian spin
and neutron projections, source-chemical-potential or filling control,
neutron and bulk dataset comparison, plotting, fitting, and script export.
Scalar Stoner, user-matrix, and local multiorbital Hubbard--Hund RPA
interaction dressings are implemented. Later response stages add production
convergence and acceleration, BCS superconductivity, and advanced correlated extensions.
Later performance work includes scheduler adapters, wider certified symmetry
support, convergence management, and workload-specific compiled or sparse
kernels. A capability remains planned until its model page documents an
implemented public API.

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
