# Electronic-response design contract

This page defines the architectural invariants for nfit's electronic models,
response components, and future external adapters. User-facing construction
and physics are documented under
[Electronic-structure and electronic-response models](electronic_structure_models.md).

The goal is a material-independent path from a periodic electronic
Hamiltonian to a complex magnetic susceptibility and then to a dataset
observable. Core interfaces must not assume a particular compound, lattice,
orbital count, or ordering wavevector.

## Layer boundaries

The implemented dependency chain is:

| Layer | Public object or component | Responsibility |
| --- | --- | --- |
| electronic structure | `ElectronicModel`, `tight_binding` | lattice, basis, $H(\mathbf R)$, $H(\mathbf k)$, and named one-electron parameters |
| bare response | `SusceptibilityResult`, `lindhard` | operator matrix elements and the particle--hole bubble |
| interaction dressing | `InteractionVertex`, RPA components | map $\boldsymbol\chi^0$ to $\boldsymbol\chi$ without changing the source Hamiltonian |
| experimental projection | fitting and cross-section services | momentum conversion, spin projection, form factor, fluctuation--dissipation convention, bulk conversion, and dataset normalization |
| systematics | dataset-owned transforms | optional resolution, binning, absorption, and related measurement effects |

An electronic-structure component is a parameter-providing dependency, not an
additive dataset observable. A dressing consumes the referenced bare
observable on its dataset scope. This prevents accidental sums of bare and
dressed intensities while preserving every upstream parameter in fitting,
uncertainty propagation, reports, and scripts.

## Canonical electronic model

`ElectronicModel` is immutable and represents a finite orthonormal basis. It
stores:

- a direct-lattice matrix with vectors as columns;
- ordered `BasisState` metadata and fractional orbital centers;
- integer translations and complex $H(\mathbf R)$ blocks;
- positive interpolation weights;
- one to three periodic axes;
- named linear parameter values and matrix blocks;
- optional $S_x,S_y,S_z$ matrices; and
- source, unit, gauge, and symmetry provenance.

The Hamiltonian convention is

$$
H_{ab}(\mathbf R)
=\langle\mathbf0,a|\hat H|\mathbf R,b\rangle,
\qquad
H(-\mathbf R)=H(\mathbf R)^\dagger ,
$$

and

$$
H(\mathbf k)=
\sum_{\mathbf R}w_{\mathbf R}H(\mathbf R)
e^{2\pi i\mathbf k\cdot\mathbf R}.
$$

This is the Wannier gauge. Orbital centers are stored separately and enter
position-sensitive operators; an adapter using another gauge must transform
to this convention and record the transformation.

Named parameters are linear:

$$
H(\mathbf R;\boldsymbol\theta)
=H_0(\mathbf R)+\sum_p\theta_pP_p(\mathbf R).
$$

The resolved-model digest includes scientific values and conventions, so
cache entries cannot cross parameter states.

The current canonical model does not include an overlap matrix
$S(\mathbf k)$, a frequency-dependent self-energy, anomalous Nambu blocks, or
self-consistent electronic mean fields. Those are separate model types rather
than hidden options on `ElectronicModel`.

## Basis metadata and operators

A basis state is identified by a unique label and optional site, species,
orbital, correlated-shell, spin, and extension metadata. No core numerical
kernel may infer array layout from an orbital name.

An explicit spinor model supplies Hermitian $S_x,S_y,S_z$ in its actual basis.
An implicit-spin model records a two-fold degeneracy and uses the documented
analytic spin trace. A response operator basis has ordered labels,
one-particle matrices, and a conjugate-index involution.

The full experimental transfer $\mathbf Q$ and the Brillouin-zone-reduced
$\mathbf q$ remain distinct. Band denominators use $\mathbf q$; orbital-center
phases, the neutron projector, and form factors use $\mathbf Q$. An operator
adapter must state:

- its basis ordering;
- position-phase convention;
- spin or magnetization matrices;
- normalization; and
- whether it contains a form factor.

Exactly one layer applies each form factor.

## Units and coordinates

The canonical numerical conventions are:

| Quantity | Convention |
| --- | --- |
| Hamiltonian, band, chemical-potential, broadening, and interaction energies | meV |
| normal electronic input and plot unit | declared explicitly, normally eV |
| direct-space vectors | Å |
| physical reciprocal vectors | Å$^{-1}$ |
| reduced $\mathbf k$, $\mathbf q$, and crystallographic $(H,K,L)$ | dimensionless |
| temperature | K |
| susceptibility | stated per primitive cell, formula unit, or selected subspace |

If $A$ contains direct-lattice columns,

$$
B=2\pi A^{-\mathsf T},\qquad
\mathbf r_{\rm cart}=A\mathbf r,\qquad
\mathbf k_{\rm cart}=B\mathbf k .
$$

Electronic inputs are converted to meV once at construction or import. A
unit-neutral adapter must require an explicit energy unit. Rendering may
convert an energy-resolved ordinate as well as its axis; for example, a DOS
shown in states/eV must preserve its integrated state count.

Shared neutron, elastic, bulk, and dataset-scale conventions remain
authoritative on [Physics conventions](physics_conventions.md).

## Response result

`SusceptibilityResult` carries paired $\mathbf q,E$ points, the original
extended-zone transfer, a complex ordered operator tensor, conjugate indices,
model digest, thermodynamic settings, broadening, normalization, and
provenance.

The causal sign convention is

$$
\boldsymbol\chi=\boldsymbol\chi'+i\boldsymbol\chi'',
$$

with positive diagonal $\chi''$ for positive-energy absorption. The bare
response is

$$
\chi^0_{AB}(\mathbf q,E)=
-\sum_{\mathbf k,n,m}w_{\mathbf k}
\frac{f_{n\mathbf k}-f_{m,\mathbf k+\mathbf q}}
{E+\varepsilon_{n\mathbf k}-\varepsilon_{m,\mathbf k+\mathbf q}+i\eta}
M^A_{nm}M^{B*}_{nm}.
$$

An `InteractionVertex` declares its ordered basis, complex matrix, canonical
meV unit, physical channel, multiplication convention, and provenance. The
implemented RPA convention is

$$
\boldsymbol\chi
=\left[\mathbb1-\boldsymbol\chi^0\boldsymbol\Gamma\right]^{-1}
\boldsymbol\chi^0.
$$

The vertex belongs to the two-particle response. $U$, $J_H$, or a Stoner
parameter do not modify $H(\mathbf k)$ unless a separately named electronic
renormalization layer is introduced.

## Sampling and execution

Paths and meshes share `WavevectorSampling`. A path stores ordered physical
distance and labels. A mesh stores normalized weights, shape, shift, and
provenance.

Observable-specific automatic sampling resolves to a concrete mesh before
fitting. A `SamplingCertificate` records the declared domain, input digest,
accuracy policy, attempted meshes, errors, and either a certified production
mesh or explicit budget exhaustion. Accuracy tolerances and computational
budgets remain independent. An optimizer must never refine the mesh as a
function of its trial parameters; changed scientific inputs make the saved
certificate stale and require a separate recertification.

Symmetry reduction is fail closed:

- total electronic DOS may use the full reciprocal group only for a
  certified nfit-built representation;
- a response may use only the little group that fixes every requested
  transfer and preserves the operator; and
- `auto` records a full-mesh fallback, while `reduced` raises when proof is
  unavailable.

Wavevector acceleration is also fail closed. A transfer commensurate with a
complete uniform integration mesh may reuse the base eigensystem by exact
periodic permutation. Off-mesh interpolation is permitted only under an
explicit absolute or relative tolerance, direct validation, deterministic
refinement, and recorded exact fallback. Interaction dressing and
extended-zone magnetic projection occur after interpolation.

Backend and memory policies must not change scientific sampling. Results
record requested and resolved backend, precision, batch size, worker count,
and approximation state. A non-reference response backend can be checked on a
deterministic probe against serial NumPy before use.

Caches are bounded, owned by one compiled observable component, and shared by
all compatible datasets evaluated by that component. Keys include the complete
parameter-point model digest, sampling, operators, thermodynamic state,
broadening, and execution inputs. A separate structural digest permits exact
reuse of Fourier-transformed \(H_0(\mathbf k)\) and \(H_a(\mathbf k)\) across
fitted Hamiltonian coefficients. Host or accelerator caches may retain those
components, eigensystems, or completed bare responses. They are performance
state and are not serialized as part of the scientific model.

Distributed response work is partitioned by deterministic contiguous point
ranges. Merging verifies the model digest, operator ordering, thermodynamic
state, broadening, normalization, and contiguous nonoverlapping point order.
The provided sequence must start at global point zero; the caller remains
responsible for supplying the final expected chunk. Scheduler submission
remains outside the numerical API.

## Serialization and scripting

Every scientific GUI action must call a public GUI-free API. A project or
copied script must preserve:

- source files and digests, or a portable canonical model;
- lattice, basis order, orbital centers, local frames, and Fourier gauge;
- all named parameter values, bounds, fit selections, and sharing;
- response mesh, thermodynamic settings, operator basis, and interactions;
- backend request and numerical convergence controls; and
- dataset bindings and normalization settings.

Presentation-only window state need not enter the scientific component.

An imported source is re-read and verified against its canonical digest.
Manual and structure-built models serialize their editable specification and
resolved digest. Adapters must report their own package name, version, source
unit, conversion, and any discarded information.

## External adapter requirements

A new electronic adapter is acceptable only if it defines:

1. direct and reciprocal lattice conventions;
2. basis ordering, orthogonality, and orbital centers;
3. energy unit and reference;
4. Fourier gauge and real-space translation convention;
5. spin representation and physical spin matrices;
6. interpolation or overlap semantics;
7. source provenance and deterministic digest; and
8. validation against the source package on representative bands.

Wannier90 is the implemented external Hamiltonian interface. ASE supplies
Setyawan--Curtarolo paths and linear-tetrahedron DOS, but nfit does not import
ASE calculator or `Atoms` electronic models. Pymatgen, PythTB, sisl, and
external many-body engines are not implied compatibility targets merely
because their outputs can be adapted in a user script.

## Validation requirements

Changes to the electronic stack should retain tests for:

- Hermiticity and periodicity of $H(\mathbf k)$;
- unit conversion and Fourier/gauge round trips;
- parameter-block linearity and immutable digest changes;
- primitive-cell equivalence when folding is accepted;
- orbital and spin projection sum rules;
- time-reversal symmetry of nonmagnetic spinor builders;
- path, mesh, DOS, and Fermi-surface limiting cases;
- Lindhard causality, static limits, and full-versus-reduced equivalence;
- RPA scalar and matrix limiting cases and pole diagnostics;
- backend agreement within declared tolerances;
- deterministic batching, caching, chunking, and merging;
- project and editable-script round trips; and
- simultaneous multi-dataset fitting and uncertainty export.

Scientific extensions that cannot meet the relevant validation and
provenance requirements remain experimental rather than becoming a registered
model component.

## Deferred capabilities

Nonorthogonal bases, general numerical symmetry representations, self-energy
models, superconducting responses, advanced correlated solvers, and wider
operator-aware symmetry reduction are not implemented. They are described
without stage numbering in [Planned features](planned_features.md).
