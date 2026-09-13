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
- Evaluate additional temperature-dependent closures only when their assumptions
  and parameter identifiability can be documented and tested.
- Extend the magnetic form-factor tables to $5d$ ions (Re, Os, Ir, Pt, W, Ta)
  and to Ru$^{2+}$/Ru$^{3+}$, Rh$^{3+}$, and Ce$^{3+}$, so iridates, osmates,
  and ruthenates do not need explicit custom coefficients. These are absent from
  International Tables C 4.4.5 itself, so they need a separately cited source
  rather than a regeneration from the existing one.
- Add $\langle j_4\rangle$ and $\langle j_6\rangle$ if a beyond-dipole
  (multipolar) form factor is ever needed; the dipole approximation using
  $\langle j_0\rangle$ and $\langle j_2\rangle$ is implemented.
- Fold a conventional-cell model onto the primitive cell even when symmetry
  expansion gave translation-equivalent sites different local orbital frames
  (the $Fd\bar3m$ $16c$ pyrochlore case). This needs the orbital rotation
  between the two frames, so it belongs with the manifold representation
  machinery rather than the current label-and-position matching. Until then
  those models keep the correct but larger conventional cell.

## Electronic-response models

Current tight-binding, Lindhard, and RPA capabilities are documented under
[Electronic-structure and electronic-response models](electronic_structure_models.md).
The [electronic-response design contract](electronic_response_contract.md)
defines the conventions that future extensions must preserve.

Planned electronic-structure extensions include:

- generalized spinor probe and intrinsic operator bases so explicit-spin
  electronic RPA can use orbital-resolved form factors; implicit-spin bare,
  Stoner, matrix, and Hubbard--Hund responses and explicit-spin bare responses
  already support them;
- nonorthogonal bases with an explicit overlap matrix $S(\mathbf k)$;
- user-supplied symmetry representations for general numerical or Wannier
  bases;
- additional external interfaces when they preserve basis, gauge, unit, spin,
  and provenance information;
- spin-dependent generated hopping and intentionally time-reversal-breaking
  electronic Hamiltonians; and
- richer orbital-resolved controls for electronic plots.

Planned Fermi-surface visualization extensions include:

- display in the primitive or conventional reciprocal parallelepiped, or in
  the first Wigner--Seitz Brillouin zone with its outline;
- independent integer tiling along the three reciprocal basis directions; and
- an arbitrary-plane slicer that reports and plots the two-dimensional
  intersections of every band sheet.

The cell choices and tiling workflow may take design inspiration from IFermi,
while retaining nfit's own reciprocal-coordinate, scripting, and provenance
conventions.

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

Current fits already share bounded response contexts across datasets, reuse
parameter-resolved momentum Hamiltonians, factorize ordered orbital-pair
responses, schedule independent wavevectors within one CPU allocation, and
can keep the full Lindhard contraction on one explicitly selected CuPy
device. Larger future speedups may come from:

- operator-aware Brillouin-zone symmetry for general orbital, Wannier, spin,
  and spin--orbit bases, with an exact full-zone fallback whenever basis
  transformations or gauge information are incomplete;
- matrix-free projected Hubbard--Hund RPA, so neutron projections can be
  solved without materializing and inverting the complete orbital-pair
  response tensor;
- fused, memory-aware GPU kernels and q batching, followed by multi-GPU
  distribution when a single device is insufficient;
- analytic or tangent electronic derivatives with a documented treatment of
  degeneracies, reducing repeated finite-difference band calculations during
  fitting; and
- persistent distributed workers that retain Fourier components,
  eigensystems, and response workspaces across fit evaluations.

More speculative work includes all-q FFT formulations and certified
reduced-order surrogates over parameter space. Any accelerated path must be
checked against the serial float64/complex128 reference, preserve the stated
scientific tolerance, and retain deterministic provenance.

The shared convergence-certificate framework now selects fixed DOS and
Lindhard production meshes. Future extensions should add topology and
geometric-distance checks for Fermi surfaces and could govern energy-window
pruning of inactive particle--hole transitions. Electronic-RPA stability scans
should similarly converge the static feedback over a declared q mesh; current
diagnostics report evaluated zero-energy points and do not claim a global
stability proof.

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

- Extend metadata dimensions to raw event streams with explicit
  timestamp alignment, time-dependent normalization/coverage, and hierarchical
  stacking. Loaded points and histograms already support aligned pointwise
  coordinates; asynchronous logs are rejected until an adapter aligns them.

- Extend the workflow graph to grouped raw direct-geometry, CORELLI, and
  MDEvent reductions, portable standalone expansion of live project-composite
  analysis dependencies, plots, and complete active project state. Composite-backed
  analysis scripts currently rerun the saved project recipe directly.
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
