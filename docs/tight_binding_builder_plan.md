# Tight-binding model-builder plan

This page tracks the staged structure-first builder for manual tight-binding
models. The detailed current behavior is documented on
[Tight-binding electronic structure](tight_binding.md).

Stages 3.1 through 3.5 are implemented. The GUI and public API share CIF import,
editable crystal geometry, site expansion, orbital manifolds and local frames,
site-point-group identification, calculated harmonic subspaces,
site-symmetry representations, static onsite invariants, selectable
symmetry-generated hopping candidates, canonical model resolution, project and builder-script
serialization, and an eV-default input unit backed by canonical meV storage.
The shared 3D model viewer displays the cell, smooth element-colored atom
spheres, active or ghost sites, orbital tokens, local frames, and selected
hopping or exchange pathways using batched geometry. A separate 3D
Brillouin-zone view shows the configured labelled path and reciprocal basis
vectors. Named Hamiltonian coefficients use the common value, bounds,
fit-selection, sharing, reporting, and scripting machinery. Their optimizer
evaluation begins when an electronic-response model supplies a dataset
observable. Optional collinear and spinor representations, onsite
$\lambda\mathbf L\cdot\mathbf S$, time-reversal validation, and matrix/subspace
inspection are available. Compact parameterizations remain planned.

## Goal

A user should be able to start from a CIF or a manually entered crystal,
choose arbitrary orbital content on selected sites, generate
symmetry-constrained onsite and hopping terms, and opt individual terms into a
fit. Wannier90 import remains an independent route to the same canonical
`ElectronicModel`.

The builder must not assume a compound, orbital count, lattice type, or
correlation model. It should support, for example:

- one effective orbital on each selected site;
- a complete degenerate atomic shell;
- site-symmetry submanifolds calculated from any supported crystallographic
  point group and selected local harmonic shell;
- a localized orbital hybridized with broader orbitals on the same or other
  species; and
- spinor bases with spin-orbit coupling.

## Crystal and orbital model

The existing Heisenberg RPA crystal machinery should become a shared geometry
service rather than being copied into the tight-binding model. CIF import
provides the direct lattice, space group, representative atomic sites, and
fractional positions. A CIF represents a three-dimensional periodic crystal,
so the initial periodic axes are `(0, 1, 2)`; a user may explicitly choose a
reduced-dimensional model afterward.

The CIF does not determine the electronic basis. The user selects site
representatives and adds one or more **orbital manifolds** to each. A proposed
manifold record contains:

| Field | Meaning | Example |
| --- | --- | --- |
| `site_label` | crystallographic site receiving the manifold | `"M1"` |
| `label` | unique manifold label | `"M1_d_subspace"` |
| `basis_kind` | effective scalar, real spherical harmonics, complex spherical harmonics, or custom | `"real_harmonic"` |
| `l` | angular-momentum quantum number when applicable | `2` for a $d$ shell |
| `orbitals` | ordered orbital labels | `["d_yz", "d_zx"]` |
| `irrep` | calculated subspace description or established irrep label | `"4/mmm subspace 2 (dimension 2; d_yz, d_zx)"` |
| `degeneracy_groups` | orbitals constrained to one onsite energy | `[["d_yz", "d_zx"], ["d_xy"]]` |
| `local_frame` | orthonormal local axes expressed in crystal Cartesian coordinates | `[[1,0,0], [0,1,0], [0,0,1]]` |
| `spin_basis` | spinless, collinear, or spinor | `"spinless"` |
| `correlated_shell` | grouping used by later $U$ and $J_H$ interactions | `"M1_3d"` |
| `site_point_group` | point group used for an opted-in subspace | `"4/mmm"` |
| `submanifold_id` | stable calculated-subspace identifier | `"d:4/mmm:8705c5723841"` |

The always-available choices are an effective orbital, complete `s`, `p`, `d`,
and `f` shells, and a custom basis. They are not separate model classes.
For a harmonic shell, the user may opt into the identified site point group
and select a symmetry-closed subspace calculated from its representation.
Arbitrary labels and custom transformation matrices remain available.

Real spherical harmonics are the primary built-in convention. The crystal
Cartesian frame is the default. A user may replace it with any right-handed
orthonormal local frame, including a frame chosen to describe a particular
crystal-field environment. Subspaces are derived after applying that frame;
they are not a hard-coded list of cubic or trigonal names.

Automatic symmetry is permitted only when the selected subspace is closed
under every operation in the site's stabilizer. nfit reports a non-closed
operation rather than projecting it away. Custom numerical and Wannier bases
remain usable with automatic symmetry disabled; a later adapter may attach
explicit representation matrices. Imported Wannier90 models remain an
independent construction route and are not converted automatically into
symmetry-aware builder models.

The local frame is essential for directional orbitals, crystal-field terms,
spin-orbit coupling, and symmetry-related hopping matrices. nfit must not
infer it silently when the crystallographic environment does not define it
uniquely.

## Onsite terms and degeneracy

The builder separates a static onsite Hamiltonian from a dynamical many-body
self-energy. The first implementation concerns onsite energies and
symmetry-allowed onsite hybridization:

$$
H_{\mathrm{onsite}}
=\sum_p \epsilon_p P_p.
$$

$P_p$ is a symmetry-allowed Hermitian matrix and $\epsilon_p$ is a named
energy. It is entered in the model's electronic energy unit, which defaults to
eV, and converted once to canonical meV. One parameter is created for each
independent onsite invariant. An explicitly declared degeneracy group shares
one parameter, but the builder must reject a requested degeneracy that
conflicts with the selected site-symmetry representation.

A frequency-dependent self-energy
$\Sigma_{ab}(\mathbf k,E)$ belongs to a later correlated-electron layer. It
must use a separate interface because it changes the Green function rather
than the static tight-binding Hamiltonian.

Every generated onsite parameter stores a value, unit, bounds, fit checkbox,
sharing rule, generating symmetry basis, and stable identifier. Parameters are
fixed unless the user opts them into fitting.

## Symmetry-constrained hopping

The existing bond-orbit generator can provide site pairs, translations,
distances, and the space-group operation relating each bond to its orbit
representative. Tight binding requires an additional orbital transformation
layer.

Let $T_{ij}(\mathbf R)$ be a hopping matrix from the orbital subspace on site
$j$ to that on site $i$. If a space-group operation $g$ maps this bond to
another bond, the corresponding matrix is

$$
T_{g(i)g(j)}(\mathbf R_g)
=D_i(g)\,T_{ij}(\mathbf R)\,D_j(g)^\dagger,
$$

where $D_i(g)$ and $D_j(g)$ act on the selected orbital manifolds in their
local frames. If canonical bond storage reverses the directed bond, the
matrix is additionally Hermitian conjugated. Translational offsets and
nonsymmorphic operations must remain explicit.

For each spatial bond orbit up to a user cutoff, nfit should solve for a basis
of symmetry-allowed hopping matrices,

$$
T_{\mathrm{rep}}=\sum_p t_p B_p.
$$

Each coefficient $t_p$ is a named energy parameter in the selected electronic
input unit. It is fixed by default, stored canonically in meV, and may be opted
into fitting. This gives:

- one scalar nearest-neighbor parameter for a one-orbital manifold;
- independent intra- and inter-manifold hopping or hybridization when
  symmetry permits them;
- the correct reduction of a multiorbital hopping matrix without assuming all
  entries are independent; and
- exact regeneration of every symmetry-related $H(\mathbf R)$ block.

The initial parameterization should use symmetry-allowed matrix bases because
it is general. Slater--Koster two-center integrals can later be an optional
compact parameterization for recognized angular-momentum bases, not the only
route.

## Shared 3D model viewer

The renderer-independent model-geometry scene is shared by tight binding and
Heisenberg RPA. Its Stage 3.2 foundation contains:

- the crystallographic unit cell;
- all atoms as element-colored, smooth-shaded sphere glyphs, or active sites
  only, with inactive atoms optionally shown as translucent ghosts;
- every orbital center and an abstract colored token for each simultaneous
  basis orbital; and
- the local Cartesian frame of each orbital manifold.

The tokens identify basis states and do not claim to be real-space
wavefunction isosurfaces. Analytic orbital lobes and imported volumetric
Wannier functions may be added later without changing the scientific scene
API.

Cell edges, atom glyphs, orbital tokens, local-frame axes, and pathways are
batched by visual role. This keeps scene construction responsive while
retaining smooth camera interaction.
The shared renderer maps each batched actor back to its scene records.
Left-clicking an atom, orbital token, or pathway reports its name and relevant
site, manifold, element, or orbit context in the side panel.

Stage 3.3 adds hopping-path and hopping-term selection to the same scene. The
pathway layer accepts Heisenberg exchange or tight-binding bond orbits and
distinguishes one representative bond from all symmetry-equivalent bonds. It
does not encode hopping magnitude or phase: geometry must not imply that a
multiorbital hopping matrix is a single scalar.

The renderer-independent Brillouin-zone scene is separate from the crystal
scene. It constructs the Wigner--Seitz cell of the primitive reciprocal
translation lattice, including the centering translations of a conventional
crystal cell, and overlays the configured labelled band path plus
$\mathbf b_1$, $\mathbf b_2$, and $\mathbf b_3$. The three-dimensional view
and band plot use the same path configuration.

## Spin-orbit coupling

Spin-orbit coupling is an optional onsite term

$$
H_{\mathrm{SOC}}=\lambda\,\mathbf L\cdot\mathbf S.
$$

Enabling it expands a spinless orbital manifold into a spinor basis, constructs
$\mathbf L$ in the declared orbital convention and local frame, and adds a
named $\lambda$ parameter in the electronic input unit, stored canonically in
meV. The generated Hamiltonian and spin operators must use the same basis
ordering. Custom orbital manifolds require explicit angular-momentum matrices
before SOC can be enabled.

The default `auto` representation keeps spin implicit until an SOC term
requires a spinor basis. Collinear spin labels without SOC remain a simpler
explicit option. Spin-independent spatial terms are lifted with
$I_2$ rather than regenerated as separate up/down coefficients.

## GUI workflow

The model editor should present five focused sections:

1. **Structure** — import CIF, edit lattice and space group, inspect
   crystallographic sites, and choose periodic axes.
2. **Orbitals** — select sites, inspect the identified point group, add a
   complete shell, an opted-in symmetry subspace, or a custom manifold, define
   local frames, and inspect the expanded ordered basis.
3. **Onsite terms** — generate symmetry-allowed onsite invariants, set
   crystal-field energies and hybridizations, and choose fitted terms.
4. **Hoppings** — choose a distance cutoff, generate suggested spatial-orbit
   matrix terms, inspect their source and destination orbitals, select terms
   to add to the Hamiltonian, set values, and choose fitted terms.
5. **Calculations** — configure projections, bands, DOS, and Fermi surfaces
   using the current model-owned plot actions.

Generated tables must show the representative sites or bonds, distance,
multiplicity, participating manifolds, matrix-basis label, current value,
bounds, and fit state. Regeneration should preserve values and fit settings
for stable identifiers that still exist, as the Heisenberg editor currently
does for exchange labels.

The GUI should offer three equally visible construction routes:

- **Import CIF / build orbitals and hoppings**;
- **Build a lattice and basis manually**; and
- **Import Wannier90**.

No scientific object should exist only in Qt state.

## Public API and serialization

The GUI must call public, JSON-serializable operations. The implementation
should introduce records equivalent to:

- crystal geometry;
- site-attached orbital manifolds;
- onsite invariant bases;
- hopping-orbit invariant bases;
- spin and SOC configuration; and
- resolved named Hamiltonian parameters.

The exact class names are an implementation detail, but the serialized form
must retain the user's high-level construction choices as well as the resolved
canonical `ElectronicModel`. Saving only the final dense matrices would make
the model impossible to edit or regenerate.

Script export must reproduce CIF import or manual crystal construction,
orbital assignment, symmetry generation, parameter values and fit flags, and
all plot settings without constructing GUI widgets. The exported script should
verify the canonical model digest.

## Fitting boundary

Onsite and hopping coefficients can enter the existing parameter-sharing and
bounds machinery once the electronic model feeds a fit observable. Before the
Lindhard response is available, the editor may expose values and fit flags but
must state that no optimizer consumes them yet.

When fitting is enabled:

- changing a named coefficient must rebuild only the affected Hamiltonian
  blocks;
- derived band and response caches must include the resolved parameter state;
- fit reports must export the high-level orbital and hopping specification,
  parameter covariance, canonical model digest, and calculation provenance;
  and
- symmetry-constrained degeneracies must be represented by shared parameters,
  not by independent parameters plus equality penalties.

## Implementation stages

### 3.1 — Shared crystal geometry (implemented)

- Extract CIF import, lattice conversion, site expansion, and spatial
  bond-orbit generation behind model-independent APIs.
- Preserve the Heisenberg RPA behavior while moving both models onto the
  shared service.
- Add project and script round trips for a structure-only tight-binding
  component.

### 3.2 — Orbital and onsite builder (implemented)

- Add orbital-manifold records, presets, local frames, basis expansion, and
  symmetry representations.
- Generate onsite invariants and symmetry-required degeneracies.
- Add the Structure, Orbitals, and Onsite GUI sections.
- Resolve the generated static terms into an editable canonical
  `ElectronicModel`, preserving values and parameter metadata.
- Add builder-script round trips and the shared 3D viewer foundation.

### 3.3 — Hopping generator (implemented)

- Reuse spatial bond orbits and add orbital covariance matrices.
- Generate symmetry-allowed hopping candidates up to a cutoff and activate
  only user-selected terms.
- Resolve named parameters into canonical $H(\mathbf R)$ blocks and add the
  Hoppings GUI section.
- Extend the shared viewer with representative and symmetry-equivalent hopping
  paths and matrix-term selection.
- Add a labelled first-Brillouin-zone viewer using the primitive translation
  lattice, configured band path, and reciprocal basis vectors.

### 3.4 — Parameter and fit integration (implemented)

- Connect named onsite and hopping terms to shared bounds, fit selection,
  grouping, reports, and scripts.
- Enable optimizer use when a compatible electronic-response observable is
  implemented.

### 3.5 — Spin, SOC, and matrix inspection (implemented)

- Keep SU(2)-symmetric models in an implicit $N$-orbital representation unless
  a calculation requires explicit spin.
- Add collinear and spinor $2N$-state expansion without duplicating
  spin-independent Hamiltonian coefficients.
- Add spin operators and manifold-resolved
  $\lambda\mathbf L\cdot\mathbf S$ for complete analytic shells, projected
  analytic subspaces, and effective bases with explicit orbital operators.
- Validate nonmagnetic spinor Hamiltonians under time reversal and expose
  orbital-plus-spin double-group representations.
- Inspect $H(\mathbf k)$, named parameter matrices, spin operators,
  representative onsite and hopping bases, exact elements, and
  orbital-subspace blocks in a separate viewer.

### 3.6 — Compact and accelerated parameterizations

- Add optional Slater--Koster parameter sets and external structure/path
  adapters where they reduce user work.
- Add symmetry-reduced evaluation, sparse or compiled kernels, GPU and
  distributed backends without changing the scientific model.

## Validation

Tests should use small synthetic cubic, trigonal, low-symmetry, and
mixed-manifold crystals rather than encoding a package-specific material
model. They should verify:

- CIF and manual crystals produce the same expanded sites;
- site-symmetry operations preserve generated onsite Hamiltonians;
- hopping matrices obey the covariance equation for every orbit member;
- generated $H(-\mathbf R)=H(\mathbf R)^\dagger$;
- declared degeneracies are exact;
- origin shifts, equivalent-cell choices, and basis permutations preserve
  band energies;
- projection weights transform consistently with the basis;
- SOC gives a Hermitian, time-reversal-consistent spinor Hamiltonian where
  applicable; and
- GUI actions and exported scripts produce identical canonical digests.

## Fixed design decisions

- Real spherical harmonics are the primary built-in convention; complex
  harmonics, crystal-field subspaces, and custom numerical bases remain
  explicit alternatives.
- The crystal Cartesian frame is the default, with an arbitrary local frame
  available per manifold.
- General symmetry-allowed hopping matrices are the default. Slater--Koster
  integrals will be an optional later parameterization.
- Tight-binding “orbital self energies” are named static **onsite energies**.
  Self-energy is reserved for a later frequency-dependent
  $\Sigma(\mathbf k,E)$ interface.
- SCDM-generated Wannier functions enter through the normal Wannier90 import
  route. nfit does not infer atomic symmetry character from their labels or
  centers.

## References

- J. C. Slater and G. F. Koster, *Phys. Rev.* **94**, 1498 (1954),
  [doi:10.1103/PhysRev.94.1498](https://doi.org/10.1103/PhysRev.94.1498).
- V. Vitale *et al.*, *npj Comput. Mater.* **6**, 66 (2020),
  [doi:10.1038/s41524-020-0312-y](https://doi.org/10.1038/s41524-020-0312-y).
- [Gemmi crystallography documentation](https://gemmi.readthedocs.io/).
- [Electronic-response design contract](electronic_response_contract.md).
