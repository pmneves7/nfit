# Electronic-response design contract

This page fixes the architecture and conventions for nfit's tight-binding and
itinerant spin-response models. Stages 0--3 are implemented; later response and
interaction stages remain an implementation contract. Available models remain
listed on [Spin-fluctuation models](spin_fluctuation_models.md).

The goal is a material-independent path from an electronic Hamiltonian to a
complex magnetic susceptibility and then to the neutron observable. No core
class may assume a particular compound, orbital count, lattice type, or
magnetic ordering vector.

## Initial scope

The first electronic-response implementation will support:

- periodic one-, two-, and three-dimensional models;
- any finite number of sites and orbitals;
- real or complex hoppings and onsite matrices;
- collinear, noncollinear, and spinor bases, including spin-orbit coupling;
- manual models and orthonormal Wannier Hamiltonians;
- finite temperature, chemical potential or filling, and lifetime broadening;
- complex bare and interaction-dressed susceptibilities; and
- public calculation, fitting, plotting, and export APIs used by the GUI.

An arbitrary basis state is identified by metadata rather than by a fixed
`n_orbitals` or `n_spin` layout. The metadata may associate it with a site,
species, orbital label, correlated shell, and spin character. A spinor model
must also provide the spin-operator matrices needed to project its generalized
response onto physical spin components.

The first concrete tight-binding model will use an orthonormal basis. The
electronic-model interface will leave room for a nonorthogonal overlap matrix
$S(\mathbf k)$, but generalized eigenproblems are not part of the first
implementation. Superconducting Nambu Hamiltonians, self-consistent electronic
mean fields, and full DMFT or Bethe--Salpeter solvers are also deferred. These
extensions must not require changing the conventions below. A later BCS
response model will add Nambu bands, superconducting gaps, coherence factors,
and the resulting continuum or spin-resonance response as an explicit
extension of the normal-state electronic model.

## Layered response

The calculation is divided into composable layers:

| layer | input | output |
| --- | --- | --- |
| electronic structure | lattice, basis, and $H(\mathbf k)$ | eigenvalues and eigenvectors |
| electronic renormalization | an electronic model and renormalization parameters | another electronic model or Green function |
| bare response | eigensystem or Green function and an operator basis | complex $\boldsymbol\chi^0(\mathbf q,E)$ |
| interaction dressing | $\boldsymbol\chi^0$ and an interaction vertex | complex $\boldsymbol\chi(\mathbf q,E)$ |
| neutron projection | generalized response and magnetic operators | $\chi_{s,\alpha\beta}(\mathbf Q,E)$ |
| experimental systematics | projected response and dataset/instrument configuration | response after resolution, bin integration, absorption, and other selected corrections |
| dataset comparison | projected response and dataset metadata | the selected measured quantity |

Electronic renormalization is optional. It is the insertion point for
quasiparticle weights, fitted band shifts, slave-boson mean fields, or an
imported self-energy. It is distinct from the interaction dressing that binds
or enhances particle-hole excitations.

Phenomenological models may construct the projected susceptibility directly,
but they use the same complex-response, neutron-projection, dataset, report,
and export contracts. The bare-response implementation must not depend on a
particular interaction dressing.

The experimental-systematics layer is optional and is owned by the dataset,
not the electronic model. It is the insertion point for instrument resolution,
finite-bin integration, absorption or self-shielding corrections, and related
measurement effects. Its detailed implementation is outside the
electronic-response stages, but electronic kernels must permit these operations
immediately before or during dataset comparison.

## Coordinates, units, and Fourier phases

All numerical kernels use the conventions on
[Physics conventions](physics_conventions.md):

- neutron energy transfer is $E=E_i-E_f$ in meV, with $E>0$ for neutron energy
  loss;
- electronic eigenvalues, chemical potentials, broadenings, hoppings, and
  interaction energies are converted to meV on import;
- direct-space vectors are in Å and physical reciprocal vectors are in
  Å$^{-1}$;
- reduced reciprocal coordinates $\mathbf k$, $\mathbf q$, and
  $\mathbf Q=(H,K,L)$ are dimensionless; and
- temperature is in K.

Basis vectors are stored as columns. If $A$ is the direct-lattice matrix, then

$$
B=2\pi A^{-\mathsf T},\qquad
\mathbf r_{\rm cart}=A\mathbf r,\qquad
\mathbf k_{\rm cart}=B\mathbf k ,
$$

where $B$ is the reciprocal-lattice matrix and unadorned coordinates on the
right are fractional or reduced column vectors. This agrees with
`rlu_to_inv_angstrom_matrix`, which includes $2\pi$.

The canonical real-space Hamiltonian uses the Wannier gauge

$$
H_{ab}(\mathbf k)=
\sum_{\mathbf R}
w_{\mathbf R}H_{ab}(\mathbf R)
\exp(2\pi i\,\mathbf k\mathbin{\cdot}\mathbf R).
$$

$a$ and $b$ label basis states, $\mathbf R$ is an integer cell translation,
and $w_{\mathbf R}$ is an explicit interpolation weight. For a Wannier90
Wigner--Seitz grid it includes the reciprocal of the reported degeneracy.
Orbital centers are not inserted into this phase. They are stored separately
and enter position-sensitive operators and the neutron projection. An adapter
for another tight-binding gauge must convert to this convention and record the
conversion.

The representation must satisfy

$$
H(-\mathbf R)=H(\mathbf R)^\dagger,
\qquad
H(\mathbf k+\mathbf G)=H(\mathbf k)
$$

for any reduced reciprocal-lattice vector $\mathbf G$. Importers validate
Hermiticity rather than silently repairing inconsistent input.

## Susceptibility convention

nfit writes the causal response as

$$
\boldsymbol\chi=\boldsymbol\chi'+i\boldsymbol\chi'',
$$

with positive diagonal $\chi''$ for a positive-energy absorption process. This
matches the current relaxational form
$\chi(E)=\chi(0)/(1-iE/\Gamma)$ and the fluctuation--dissipation convention in
[Physics conventions](physics_conventions.md).

For a band Hamiltonian, the canonical bare response is

$$
\chi^0_{AB}(\mathbf q,E)=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\frac{
f(\epsilon_{n\mathbf k})-f(\epsilon_{m,\mathbf k+\mathbf q})
}{
E+\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}+i\eta
}
M^A_{nm}M^B_{mn}.
$$

$N_k$ is the number of sampled wavevectors, $f$ is the Fermi function,
$\eta>0$ is the broadening in meV, and $A,B$ label an ordered operator basis.
The matrix elements include the wavevector and band labels suppressed in
$M^A_{nm}$ and $M^B_{mn}$. This sign produces nonnegative diagonal
$\chi^{0\prime\prime}$ at positive energy for Hermitian operators.

The initial multiorbital implementation uses ordered orbital-pair operators,
with a recorded conjugate map. Physical density and spin operators are
projections of that basis. This permits Hubbard--Hund vertices without
restricting later implementations to a spin-rotation-invariant orbital model.

An interaction dressing implements a map

$$
\mathcal D:
\boldsymbol\chi^0\longmapsto\boldsymbol\chi .
$$

The matrix RPA dressing is

$$
\boldsymbol\chi(\mathbf q,E)=
\left[
\mathbb 1-\boldsymbol\chi^0(\mathbf q,E)
\boldsymbol\Gamma(\mathbf q,E)
\right]^{-1}
\boldsymbol\chi^0(\mathbf q,E).
$$

The ordered operator basis, multiplication order, vertex sign, and channel
must be stored with every result. A scalar Stoner model is the one-operator
limit. Hubbard--Hund parameters belong to the interaction object, not to
$H_0$, unless a separately named electronic-renormalization step applies a
mean field or correlated-band correction.

The physical spin response is

$$
\chi_{s,\alpha\beta}(\mathbf Q,E)
=
\sum_{AB}
P^\alpha_A(\mathbf Q)
\chi_{AB}(\mathbf q,E)
P^{\beta *}_B(\mathbf Q),
$$

where $\mathbf q$ is $\mathbf Q$ reduced into the electronic Brillouin zone and
$P^\alpha_A$ contains the basis-position phases, spin matrix elements, and any
declared orbital form-factor approximation. Keeping $\mathbf Q$ and
$\mathbf q$ distinct preserves extended-zone neutron intensity.

The projected result states its normalization explicitly, for example per
unit cell or per correlated site. Conversion to a dataset's normalization
basis is never inferred from orbital count. The existing dataset layer remains
responsible for form factor, polarization, Bose population, resolution, and
experimental scale unless a model explicitly supplies a documented
orbital- or site-resolved magnetic operator. Such an operator must state its
position phases, spin or magnetization matrix elements, and form-factor
approximation. Exactly one layer applies each form factor.

## Geometry and wavevector sampling

Manual tight-binding construction may reuse the Heisenberg RPA crystal, site
expansion, symmetry, and bond-orbit machinery for orbital locations and
symmetry-related hoppings. The shared part is geometry and orbit generation:
a hopping may be a directed complex matrix between orbital subspaces, so it is
not represented as a scalar Heisenberg exchange constant. The planned
structure-first GUI, orbital representations, onsite invariants, hopping
covariance, fitting boundary, and SOC stages are specified in the
[Tight-binding model-builder plan](tight_binding_builder_plan.md).

Band paths and Brillouin-zone meshes use one serializable wavevector-sampling
interface with different configurations. An ordered path carries segment
labels and physical path distance for visualization. A periodic mesh carries
integration weights, dimensions, shifts, and symmetry reduction information
for density-of-states and response integrals. Code may share coordinate
conversion, provenance, and evaluation without pretending that a path is an
integration mesh.

## Canonical models and results

Electronic models and calculated results are immutable value objects. Large
arrays may live in a portable artifact referenced by project JSON, but their
metadata and content fingerprint remain inspectable.

An electronic model records:

- direct lattice and periodic directions;
- ordered basis-state metadata and orbital centers;
- $H(\mathbf R)$, interpolation weights, and the Fourier gauge;
- optional named parameters controlling onsite or hopping terms;
- spin-operator matrices;
- internal energy and coordinate units; and
- source and conversion provenance.

A band result records reduced wavevectors, physical path distance in Å$^{-1}$,
labels, band energies in meV, Fermi level, band indices, optional eigenvectors
or orbital/site/spin projected weights, and the complete calculation
provenance. A density-of-states result records the energy grid, total density
of states, optional projected contributions, normalization, broadening or
integration method, and mesh provenance.

A Fermi-surface result records the target energy, contributing bands,
contours or surface vertices in both reduced and physical coordinates,
connectivity where applicable, projected weights, and provenance. A saved
Fermi-surface plot must reference this scientific configuration rather than
only rendered geometry.

A susceptibility result records:

- reduced $\mathbf q$ and, when projected, physical or extended-zone
  $\mathbf Q$;
- energy transfer in meV;
- complex values with the operator or Cartesian basis;
- whether the response is bare, dressed, or neutron-projected;
- units and normalization basis;
- temperature, chemical potential or filling, and broadening;
- mesh, symmetry, backend, and convergence settings;
- interaction and projection definitions; and
- diagnostics and provenance.

The complex response is authoritative. $\chi'$ and $\chi''$ are views or
exports of its real and imaginary parts, not independently evaluated models.
Fit results retain the fitted parameter state and sufficient provenance to
recompute derived bands and responses. Large sampled arrays are exported as
separate artifacts rather than copied repeatedly into fit history.

## Provenance and scripting

An imported electronic model records:

- every source file's role, path, byte size, and SHA-256 digest;
- source format and, when available, producer and version;
- importer name and nfit version;
- source units and all unit conversions;
- source and canonical Fourier gauges;
- treatment of degeneracies, replica vectors, centers, and spin;
- warnings or deliberately omitted source information; and
- the content digest of the canonical model.

A calculation additionally records its resolved path or mesh, symmetry
provider and tolerances, temperature, chemical potential or filling solver,
broadening, operator basis, interaction, projection, numerical backend,
floating-point precision, convergence settings, and random seed when relevant.

GUI actions call public functions that accept these serializable
configurations. Workflow export includes the dependency closure: model sources
or a portable canonical artifact, construction settings, calculation settings,
plot recipe, and fit configuration. External Python objects and pickles are
not project formats. Optional adapters convert at the boundary and record the
package version used.

## Scalable execution

Electronic calculations use chunkable map/reduce kernels with explicit memory
budgets, deterministic aggregation, and backend-neutral scientific results.
The same public calculation should be able to select a serial NumPy path on a
laptop, local CPU workers, an accelerated CPU or GPU backend, or a distributed
scheduler without changing the model configuration. Backend, worker count,
precision, chunking, timing, and convergence settings are provenance.

Symmetry reduction should eventually evaluate only the irreducible part of a
Brillouin-zone mesh when the Hamiltonian, operators, and requested observable
permit it. Full-zone fallback is required, and symmetry expansion must be
testable independently. As in nfit's rebinner, temporary memory is bounded by
user-configurable budgets rather than by constructing all intermediate
transitions at once.

## Interoperability decisions

| package or format | planned role | dependency policy |
| --- | --- | --- |
| Wannier90 | native import of `*_tb.dat` or `*_hr.dat` with associated lattice, center, and replica files | no Wannier90 runtime dependency |
| PythTB | optional model adapter and comparison oracle | optional; not a core dependency |
| ASE | optional structure and band-path adapter | optional |
| pymatgen | optional structure, band-result, and path adapter | optional |
| SeeK-path | automatic crystallographic high-symmetry paths | candidate for a focused optional extra |
| TBmodels or sisl | comparison of Wannier interpolation and edge cases | development-time oracle unless a later use case justifies an adapter |

Manual high-symmetry paths remain available without an optional package. An
automatically generated path is resolved once and serialized with its provider,
convention, version, and symmetry tolerances.

Wannier90 text outputs are the interchange boundary; nfit does not perform the
DFT calculation or wannierization. `*_hr.dat` alone does not contain all
lattice and orbital-center information, and replica corrections such as
`*_wsvec.dat` must not be discarded when they are required for faithful
interpolation.

PythTB 2 is useful as a reference because it supports parameterized arbitrary
tight-binding models, Wannier90 input, meshes, and band plotting. It is not a
core dependency because its current release requires Python 3.12 and NumPy 2,
while nfit supports older Python versions, and its public API changed
substantially at version 2; see the
[PythTB 2 release notes](https://pythtb.readthedocs.io/en/stable/release/2.0.0-notes.html).

## Validation matrix

Each later stage is complete only when its applicable checks pass.

| capability | reference problem | required checks |
| --- | --- | --- |
| Fourier convention | one-dimensional chain and square lattice | analytic dispersion, periodicity, Hermiticity, origin shift |
| arbitrary basis | two-sublattice and spinor models | basis permutation, unit-cell relabeling, degenerate-subspace invariance |
| Wannier import | small checked-in Wannier90 fixtures | source bands, replica handling, comparison with an independent reader |
| band paths | cubic and lower-symmetry cells | manual path and optional-provider coordinates in the original cell |
| Fermi surfaces | square-lattice metal and simple three-dimensional metal | known topology, coordinate conversion, mesh convergence |
| bare response | free or tight-binding electron gas | sign, causality, frequency symmetry, static limit, $2k_F$ or nesting feature |
| multiorbital response | analytically reducible two-orbital model | operator-basis permutation and projection invariance |
| scalar RPA | one-channel Stoner model | closed-form denominator and pole location |
| Hubbard--Hund RPA | one- and two-shell limits | one-orbital reduction, rotational constraints, spin/charge channel signs |
| BCS response | one-band isotropic-gap model | normal-state limit, pair-breaking threshold, coherence factors, causality |
| neutron projection | local isotropic and multisite models | polarization limit, extended-zone phases, origin and cell invariance |
| serialization | manual and imported models | project reopen, digest verification, editable script equivalence |
| fitting | small synthetic spectra | parameter recovery, cache invalidation, report and exported-result parity |
| performance | representative multiorbital meshes | bounded memory, deterministic chunking, recorded backend |

Kramers--Kronig checks use a stated finite energy window and tail treatment.
Mesh and broadening convergence are reported separately; changing $\eta$ is
not treated as a substitute for increasing the wavevector mesh.

## Literature and software benchmarks

| subject | primary reference | use in nfit |
| --- | --- | --- |
| bare electron response | J. Lindhard, *Kgl. Danske Videnskab. Selskab, Mat.-Fys. Medd.* **28**, no. 8 (1954) | analytic limits and sign checks |
| itinerant spin fluctuations | T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism* (1985), [doi:10.1007/978-3-642-82499-9](https://doi.org/10.1007/978-3-642-82499-9) | paramagnon and SCR conventions |
| multiorbital RPA | S. Graser *et al.*, *New J. Phys.* **11**, 025016 (2009), [doi:10.1088/1367-2630/11/2/025016](https://doi.org/10.1088/1367-2630/11/2/025016) | orbital-pair susceptibility and interaction matrices |
| Hubbard--Hund interactions | A. Georges *et al.*, *Annu. Rev. Condens. Matter Phys.* **4**, 137 (2013), [doi:10.1146/annurev-conmatphys-020911-125045](https://doi.org/10.1146/annurev-conmatphys-020911-125045) | correlated-shell parameters and rotational constraints |
| Wannier interpolation | A. A. Mostofi *et al.*, *Comput. Phys. Commun.* **178**, 685 (2008), [doi:10.1016/j.cpc.2007.11.016](https://doi.org/10.1016/j.cpc.2007.11.016); G. Pizzi *et al.*, *J. Phys.: Condens. Matter* **32**, 165902 (2020), [doi:10.1088/1361-648X/ab51ff](https://doi.org/10.1088/1361-648X/ab51ff) | file interpretation and comparison bands |
| high-symmetry paths | Y. Hinuma *et al.*, *Comput. Mater. Sci.* **128**, 140 (2017), [doi:10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015) | path convention and SeeK-path comparisons |
| slave-boson spin excitons | W. T. Fuhrman *et al.*, *Phys. Rev. Lett.* **114**, 036401 (2015), [doi:10.1103/PhysRevLett.114.036401](https://doi.org/10.1103/PhysRevLett.114.036401); W. T. Fuhrman and P. Nikolić, *Phys. Rev. B* **90**, 195144 (2014), [doi:10.1103/PhysRevB.90.195144](https://doi.org/10.1103/PhysRevB.90.195144) | advanced Anderson-lattice benchmark |

Software used as an implementation oracle must be cited at the tested version.
The relevant official documentation includes
[Wannier90 file formats](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/),
[PythTB](https://pythtb.readthedocs.io/en/stable/),
[ASE band paths](https://docs.ase-lib.org/ase/geometry.html),
[pymatgen band structures](https://pymatgen.org/pymatgen.electronic_structure.html),
and [SeeK-path](https://seekpath.readthedocs.io/en/latest/maindoc.html).

## Stage gates

The implementation proceeds in independently testable stages:

| stage | deliverable |
| --- | --- |
| 0 | this design contract, documentation hierarchy, scope, provenance, and validation references |
| 1 | one shared model framework for evaluation, GUI metadata, diagnostics, reports, plots, and scripts |
| 2 | generalized paramagnons and damped propagating modes using the complex-response contract |
| 3 | arbitrary electronic models, Wannier90 import, bands, orbital-projected bands, density of states, and Fermi surfaces |
| 4 | generalized complex Lindhard response and its projections |
| 5 | scalar Stoner, user-defined, and Hubbard--Hund RPA interaction dressings |
| 6 | production convergence, acceleration, uncertainty, reports, and multi-dataset fitting |
| 7 | BCS superconductivity, slave-boson, imported self-energy, and other explicitly documented beyond-RPA extensions |

Every scientific model added in these stages receives its own daughter page
under [Spin-fluctuation models](spin_fluctuation_models.md). A stage is ready
for user testing only after its public API, GUI behavior where applicable,
workflow script, fit-result export, documentation, and validation tests agree.

Stages 0--3 are implemented. The generalized-paramagnon model supplies the
Stage 2 relaxational and damped-propagating limits through one causal complex
response. Stage 3 supplies arbitrary orthonormal electronic models, native
Wannier90 import, shared path and mesh sampling, bands and orbital projections,
density of states, Fermi surfaces, model-owned GUI plots, and editable scripts.
