# Hoppings

A hopping term couples orbital subspaces on two sites. nfit first groups
geometrically equivalent bonds into spatial orbits, then generates candidate
matrix coefficients for each orbit. Candidates are suggestions; only terms
that the user adds to `hopping_terms` enter the Hamiltonian.

This separation prevents a short bond cutoff from silently creating a large
electronic model.

## Directed-matrix convention

For a representative bond from site $j$ in translated cell $\mathbf R$ to
site $i$ in the home cell, the matrix

$$
T_{ij}(\mathbf R)
$$

has destination orbitals on its rows and source orbitals on its columns. The
GUI uses an arrow such as

`M1_d[d_xy,d_yz] ← X1_p[p_x,p_y]`

to show the same convention.

If space-group operation $g$ maps the representative bond to another member
of the spatial orbit, nfit constructs

$$
T_{g(i)g(j)}(\mathbf R_g)
=D_i(g)T_{ij}(\mathbf R)D_j(g)^\dagger ,
$$

$\mathbf R_g$ is the integer source-cell offset of the transformed bond
relative to the transformed destination cell; $g(i),g(j)$ are its endpoints.
The hopping matrix $T_{ij}$ is a block of $H(\mathbf R)$ in meV.
The symmetry matrices are dimensionless, and a dagger means conjugate transpose.
The matrices $D_i$ and $D_j$ act in the endpoint orbital subspaces. The reverse
directed bond supplies the Hermitian partner. The resolved model therefore
satisfies $H(-\mathbf R)=H(\mathbf R)^\dagger$.

## Spatial bond orbits

`hopping_cutoff_angstrom` sets the largest center-to-center distance examined.
The crystal space group expands active sites and groups equivalent bonds into
orbits `B1`, `B2`, and so on. Inequivalent orbits at the same distance receive
suffixes.

The cutoff controls which pathways are offered, not which coefficients are
automatically active. A physically minimal workflow is:

1. choose a cutoff large enough to include plausible pathways;
2. inspect the generated spatial orbits in the 3D viewer;
3. add only the required matrix candidates;
4. compare bands and Fermi surfaces; and
5. enlarge the range only when the data require it.

## Slater--Koster parameterization

The default `slater_koster` convention generates compact two-centre integrals
for analytic $s$, $p$, $d$, and $f$ manifolds, including selected
symmetry-closed subspaces. In a bond-axis frame,

$$
T_{\rm rep}
=\sum_{\mu=\sigma,\pi,\delta,\phi}
V_{l_i l_j\mu}\,
C_i^\dagger D_i^\dagger P_\mu D_j C_j .
$$

$l_i,l_j$ are the endpoint orbital angular-momentum quantum numbers,
$V_{l_il_j\mu}$ an energy in meV, and $\mu$ here labels a bond channel
$\sigma,\pi,\delta,\phi$, not chemical potential. In complete bond-axis
complex-harmonic bases the rectangular selector has entries
$(P_\mu)_{m_i m_j}=\delta_{m_i m_j}$ when $|m_i|$ is the channel's value,
and zero otherwise. All rotation, projection, and selector matrices are
dimensionless. The [orbital page](tight_binding_lattice_orbitals.md) fixes
harmonic phases and normalization.

$P_\mu$ selects equal bond-axis magnetic quantum numbers with
$|m|=0,1,2,3$, respectively. $D_i$ and $D_j$ rotate the endpoint local frames
to the bond frame, while $C_i$ and $C_j$ project complete harmonic shells into
the selected subspaces.

The fitted coefficient $V_{pd\pi}$ or $V_{dd\sigma}$ is the corresponding
axial two-centre matrix element. It is not a normalized generic matrix
coefficient.

Slater--Koster terms are not the only symmetry-allowed hoppings. They add the
physical assumption that the hopping is described by two-centre spherical-
harmonic integrals. This can be a useful compact model even at a low-symmetry
site, but lower symmetry generally permits additional matrix structure through
the environment or through non-atomic Wannier functions.

The current generator applies this convention only when nfit knows the
analytic harmonic transformations. Effective scalar orbitals act as $l=0$.
Complex-harmonic, custom, and Wannier bases use the general convention because
nfit does not guess their phases or angular character.

## General symmetry-matrix parameterization

The `general` convention finds a real matrix basis $B_p$ allowed by the
representative bond stabilizer:

$$
T_{\rm rep}=\sum_p t_pB_p .
$$

$p$ labels independent allowed basis matrices, $t_p$ is the fitted meV
coefficient, and $B_p$ is dimensionless. The matrices obey
$\|B_p\|_F=\sqrt{\sum_{ab}|(B_p)_{ab}|^2}=1$ (unit Frobenius norm), so $t_p$ is an energy scale rather
than a named axial integral. This basis is complete within the implemented
spin-independent, real, time-reversal-symmetric orbital representation. It is
more flexible than Slater--Koster but can produce many correlated parameters.

Use it when:

- the two-centre approximation is too restrictive;
- ligand or crystal-field effects generate additional mixing;
- a compact symmetry-complete model is needed for a small basis; or
- testing whether a Slater--Koster restriction is responsible for a poor fit.

For a custom numerical basis, automatic generation is available only when the
required site mappings are identities. Otherwise nfit would need explicit
representation matrices that it does not currently infer.

## Hopping-term fields

Each candidate or active `HoppingInvariant` stores:

| Field | Meaning | Example |
| --- | --- | --- |
| `identifier` | stable hash of orbit, endpoint bases, and matrix | `"B1:hopping:2a9374d7d0f1"` |
| `label` | readable automatic parameter name | `"B1 V_pdπ: M1_d ← X1_p"` |
| `orbit_label` | spatial pathway family | `"B1"` |
| `distance_angstrom` | representative center-to-center distance | `3.9` |
| `representative_bond` | endpoint indices and integer cell offset | `{"site_i":0,"site_j":1,"offset":[0,0,0]}` |
| `basis_i` | ordered destination basis labels | `["M1_d:d_xy","M1_d:d_yz"]` |
| `basis_j` | ordered source basis labels | `["X1_p:p_x","X1_p:p_y"]` |
| `matrix` | dimensionless hopping selector | a real $2\times2$ matrix |
| `value_meV` | canonical coefficient | `-80.0` |
| `bounds_meV` | optional canonical fit bounds | `[-200.0,20.0]` |
| `fit` | optimizer selection | `false` |
| `source` | parameterization used to generate the term | `"slater_koster"` or `"spinless_time_reversal_space_group"` |

Changing `hopping_parameterization` regenerates the candidate list. Existing
active state is preserved only for identifiers that remain exactly the same.
This avoids transferring a numerical coefficient between physically different
matrix bases.

Generated collinear and spinor models lift each spatial hopping as
$B_p\otimes I_2$ without creating separate spin coefficients. Spin-dependent
hopping is not generated automatically. A manual or Wannier90 Hamiltonian may
nevertheless contain general complex hopping matrices.

## Scripted workflow

```python
from nfit import (
    add_tight_binding_hopping_term,
    regenerate_tight_binding_hopping_terms,
    set_tight_binding_hopping_parameterization,
    set_tight_binding_hopping_term,
)

set_tight_binding_hopping_parameterization(component, "slater_koster")
generation = regenerate_tight_binding_hopping_terms(
    component,
    cutoff_angstrom=4.2,
)

hopping_id = generation.terms[0].identifier
add_tight_binding_hopping_term(component, hopping_id)
set_tight_binding_hopping_term(
    component,
    hopping_id,
    value=-0.080,
    energy_unit="eV",
    lower=-0.20,
    upper=0.02,
)
```

The selected coefficient becomes a named linear Hamiltonian parameter. It can
be fixed, fitted globally, fitted independently per dataset, or shared across
named dataset groups.

## Identifiability

Symmetry reduces parameter count but does not guarantee that a dataset
identifies each term. Several hopping combinations can reproduce the same
small energy window, while orbital projections may change strongly. Prefer the
smallest range and matrix basis consistent with available band, Fermi-surface,
filling, and neutron data. Inspect parameter covariance and compare alternative
parameterizations rather than interpreting one fitted hopping in isolation.
