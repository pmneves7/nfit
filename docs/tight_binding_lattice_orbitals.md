# Lattice, sites, and orbitals

The structure-first builder starts from a periodic crystal, attaches orbital
manifolds to representative sites, and generates symmetry-compatible onsite
matrices. This route is intended for compact physical models whose basis can
be stated explicitly.

## Crystal definition

Importing a CIF populates the lattice parameters, space group, element labels,
and representative fractional positions. The same data can be entered
manually. The serialized `crystal` record contains:

The lattice lengths $a,b,c$ are $|\mathbf a_1|,|\mathbf a_2|,|\mathbf a_3|$;
$\alpha,\beta,\gamma$ are the angles between $(\mathbf a_2,\mathbf a_3)$,
$(\mathbf a_1,\mathbf a_3)$, and $(\mathbf a_1,\mathbf a_2)$, respectively.

| Field | Meaning | Example |
| --- | --- | --- |
| `lattice` | $a,b,c$ in Å and $\alpha,\beta,\gamma$ in degrees | `{"a":4,"b":4,"c":6,"alpha":90,"beta":90,"gamma":90}` |
| `spacegroup` | Hermann--Mauguin symbol used for site and bond expansion | `"P 4/mmm"` |
| `sites` | representative site labels, fractional `position`, and elements | `[{"label":"M1","position":[0,0,0],"element":"Fe"}]` |

The lattice matrix uses vectors as columns. nfit expands each representative
site with the configured space group. The same crystal machinery is shared
with the Heisenberg model, but an electronic site is active only after at
least one orbital manifold is attached to it.

A CIF establishes geometry, not an electronic basis. Oxidation states, orbital
occupancies, crystal-field levels, and hopping amplitudes are not inferred
from element names.

## Orbital manifolds

`OrbitalManifold` is an ordered local subspace attached to one representative
site.

| Field | Meaning | Example |
| --- | --- | --- |
| `site_label` | representative crystallographic site | `"M1"` |
| `label` | unique manifold name | `"M1_d"` |
| `basis_kind` | basis transformation convention | `"real_harmonic"`, `"complex_harmonic"`, `"effective_scalar"`, `"custom"`, or `"wannier"` |
| `orbitals` | ordered labels within the manifold | `["d_xy","d_yz","d_zx"]` |
| `l` | angular-momentum quantum number for a harmonic basis | `2` |
| `irrep` | verified or descriptive site-symmetry label | `"E_g (copy 1)"` |
| `degeneracy_groups` | orbital groups constrained to share a diagonal onsite energy | `[["d_yz","d_zx"]]` |
| `local_frame` | right-handed orthonormal local axes as columns in crystal Cartesian coordinates | `[[1,0,0],[0,1,0],[0,0,1]]` |
| `spin_basis` | spatial-manifold spin convention before later spin expansion | `"spinless"` |
| `correlated_shell` | label used to select local Hubbard--Hund subspaces | `"M1_3d"` |
| `magnetic_form_factors` | mapping from orbital label to a tabulated ion or full radial-profile mapping | `{"d_xy":"V3","d_yz":"V4"}` |
| `symmetry_mode` | use an analytic symmetry representation or none | `"automatic"` or `"none"` |
| `harmonic_transform` | orthonormal columns mapping the chosen subspace into the complete $l$ shell | a $5\times2$ matrix for two selected $d$ states |
| `preset` | convenience recipe that created the record | `"d"` or `"site_symmetry"` |
| `site_point_group` | point group found from the site stabilizer | `"4/mmm"` |
| `submanifold_id` | stable identifier of a calculated subspace | `"d:4/mmm:8705c5723841"` |

$l=0,1,2,3$ labels the orbital angular-momentum quantum number of the
$s,p,d,f$ shells. A complex harmonic $Y_l^m(\theta,\varphi)$ is a normalized
angular eigenfunction of dimensionless $L^2,L_z$, with eigenvalues $l(l+1),m$;
$\theta,\varphi$ are polar and azimuthal angles in the local frame and
$\int|Y_l^m|^2d\Omega=1$. nfit uses Condon--Shortley phases and, for $m>0$,

$$
Y_{lm}^{\cos}=\frac{Y_l^{-m}+(-1)^mY_l^m}{\sqrt2},\qquad
Y_{lm}^{\sin}=\frac{i[Y_l^{-m}-(-1)^mY_l^m]}{\sqrt2}.
$$

The $m=0$ function is unchanged. These definitions specify orbital phases,
not a radial wavefunction or radial integral. Higher-$l$ real bases use order
$(m0,\cos1,\sin1,\cos2,\sin2,\ldots)$; the named $p,d$ orders follow below.

Always-available presets are a scalar effective orbital and complete $s$,
$p$, $d$, and $f$ shells. The real-harmonic orders are
$(p_x,p_y,p_z)$ and
$(d_{xy},d_{yz},d_{zx},d_{x^2-y^2},d_{z^2})$. Complex harmonics use
$m=-l,\ldots,l$.

The local frame defaults to the crystal Cartesian frame. Choose another frame
when the orbital definition follows a local ligand environment or an imported
numerical convention. Changing the frame changes the symmetry and hopping
matrices; it is not a display-only rotation.

## Site-symmetry subspaces

For a complete analytic shell, nfit can identify symmetry-closed subspaces of
the site's point-group representation. If $D(g)$ represents stabilizer
operation $g$ and the columns of $C$ span the selected subspace, nfit requires

$$
(I-CC^\dagger)D(g)C=0,\qquad C^\dagger C=I
$$

$D(g)$ is the dimensionless unitary matrix of a site symmetry that leaves
the site fixed modulo translation. $C$ has orthonormal columns in the complete
shell; $CC^\dagger$ projects onto their span and the left identity acts on
the full shell. The right identity acts on the selected subspace. This states
that symmetry never takes a selected vector out of the subspace,
for every $g$. A non-closed subspace is rejected rather than silently
projected.

Conventional irrep names are shown only when nfit has verified the
representation characters. Repeated copies of one irrep receive deterministic
copy numbers and orbital-weight descriptions. The copy number is a display
identifier, not an additional quantum number; symmetry permits equivalent
copies to mix.

`degeneracy_groups` are explicit additional constraints. A complete shell does
not otherwise imply spherical degeneracy: the site point group determines the
allowed splitting.

Custom and Wannier bases use `symmetry_mode="none"` unless explicit
representation matrices are available. nfit does not infer transformation
properties from an arbitrary orbital name.

### Orbital magnetic form factors

Magnetic radial profiles are properties of the basis orbitals, not of a
Lindhard response. In the GUI, enter a JSON mapping in the manifold's
**Magnetic form factors** cell. A tabulated spin-only profile can be abbreviated
as `{"d_xy":"V3"}`. A full value may instead use `form_factor_mode` with
`single_ion`, `custom`, or `mixture` and the coefficient, $g_J$, or mixture
fields documented under [physics conventions](physics_conventions.md). An
omitted orbital has unit probe amplitude.

Profiles are copied to every symmetry-expanded copy of the orbital and to both
spin states when a spinor basis is constructed. They affect the neutron probe
operator but do not modify the Hamiltonian, electron filling, density of
states, formula-unit count, or magnetic-center normalization.
Bare responses support implicit and explicit spin bases. Stoner, matrix, and
Hubbard--Hund RPA with orbital-specific profiles currently require an
implicit-spin normal-state basis; nfit rejects an explicit-spin RPA combination
rather than putting a probe form factor into its interaction denominator.

## Onsite Hamiltonian

For the ordered basis on a site, the builder finds Hermitian matrices $P_p$
that satisfy

$$
D(g)P_pD(g)^\dagger=P_p
$$

for every stabilizer operation and respect declared degeneracy constraints.
The onsite block is

$$
H_{\rm onsite}=\sum_p\epsilon_pP_p .
$$

$p$ labels independent allowed Hermitian matrices, $P_p$ is dimensionless
and normalized by $\|P_p\|_F=\sqrt{\operatorname{Tr}(P_p^\dagger P_p)}=1$,
and $\epsilon_p$ is its fitted energy in meV. Such an invariant is not
necessarily an idempotent projector. $H_{\rm onsite}$ has units meV.

This includes diagonal crystal-field energies and symmetry-allowed
same-site hybridization. An onsite energy is a static one-electron coefficient;
it is not a frequency-dependent many-body self-energy.

Each generated onsite term stores:

| Field | Meaning | Example |
| --- | --- | --- |
| `identifier` | stable hash of site, ordered basis, and matrix | `"M1:onsite:6054a1a6e2ad"` |
| `label` | readable coefficient name | `"M1 epsilon_1"` |
| `site_label` | representative site | `"M1"` |
| `basis_labels` | ordered basis addressed by the matrix | `["M1_d:d_xy","M1_d:d_yz"]` |
| `matrix` | unit-Frobenius Hermitian invariant | a $2\times2$ selector |
| `kind` | diagonal energy or allowed hybridization | `"onsite_energy"` or `"onsite_hybridization"` |
| `value_meV` | canonical coefficient | `25.0` |
| `bounds_meV` | optional canonical fit bounds | `[-100.0,100.0]` |
| `fit` | whether the optimizer may vary the coefficient | `false` |
| `source` | origin of the constraint | `"site_symmetry"` or `"declared_degeneracy"` |

Values entered in eV are converted immediately to canonical meV. Editing a
crystal, site, orbital manifold, local frame, or degeneracy declaration
automatically regenerates the allowed onsite matrix basis. Coefficients,
bounds, fit selections, and sharing survive when their stable identifiers
remain valid. nfit also checks the generated basis when loading and before
resolving the Hamiltonian, so no manual regeneration step is required.
Scientific builder edits mark the resolved model stale; the next electronic or
response calculation rebuilds it once. With onsite terms but no active
hoppings, the bands are flat.

## Primitive-cell resolution

Three-dimensional GUI construction may begin in a centered conventional cell
because that is the natural crystallographic representation. With
`use_primitive_cell=true`, nfit folds a complete, translation-equivalent
construction onto the primitive cell before spin expansion and
diagonalization.

Orbitals are matched by primitive center, manifold, orbital, species,
correlated shell, and spin label. The fold is accepted only when every class
has the crystallographic multiplicity and all Hamiltonian terms are
translation compatible. It is an exact change of representation, not a
symmetry approximation. If nfit cannot certify those conditions, it retains
the conventional-cell Hamiltonian and records the reason in
`provenance["primitive_reduction"]`. Disable the attempt only for diagnostic
comparison.

Matching by label and position does not see a site's local orbital frame, and
symmetry expansion can give translation-equivalent copies different frames —
the $Fd\bar3m$ $16c$ pyrochlore site is the standard case. Merging those would
average matrix elements written in different orientations and silently change
the band structure for any $l>0$ manifold, so nfit additionally certifies the
fold numerically: each block family must reproduce the conventional spectrum at
probe wavevectors. A model that fails keeps the conventional cell, which is
correct but larger and therefore slower to diagonalize. Folding such a model
exactly would require rotating the merged orbitals onto a common frame; that is
tracked in [Planned features](planned_features.md).

## Minimal scripted construction

```python
from nfit import (
    add_tight_binding_orbital_manifold,
    create_model_component,
    orbital_manifold_from_site_symmetry,
    set_model_crystal,
    set_tight_binding_onsite_term,
    site_symmetry_harmonic_submanifolds,
)

component = create_model_component(group, "electrons", type="tight_binding")
set_model_crystal(component, crystal, group=group)
choices = site_symmetry_harmonic_submanifolds(crystal, "M1", "d")
manifold = orbital_manifold_from_site_symmetry(
    crystal,
    "M1",
    "d",
    choices[0].identifier,
    correlated_shell="M1_3d",
)
add_tight_binding_orbital_manifold(component, manifold)

onsite_id = component.config["onsite_terms"][0]["identifier"]
set_tight_binding_onsite_term(
    component,
    onsite_id,
    value=0.025,
    energy_unit="eV",
)
```

The GUI uses these same public operations. Project files and copied builder
scripts retain the crystal, selected manifolds, generated matrices, parameter
state, and resolved-model digest.
