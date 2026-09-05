# Hubbard--Hund RPA

`hubbard_hund_rpa` constructs a local multiorbital spin-channel interaction on
the correlated shells of a referenced tight-binding model, dresses its
orbital-pair Lindhard response, and then projects the result onto physical
spin.

## Local-interaction ansatz

A local normal-state multiorbital interaction motivates the spin vertex.
For orbitals $l$ of one site and shell, define
$n_{l\sigma}=c^\dagger_{l\sigma}c_{l\sigma}$,
$n_l=\sum_\sigma n_{l\sigma}$,
$\mathbf S_l=\tfrac12\sum_{\sigma\sigma'}c^\dagger_{l\sigma}
\boldsymbol\sigma_{\sigma\sigma'}c_{l\sigma'}$, and
$P_l=c_{l\downarrow}c_{l\uparrow}$, the local pair-annihilation operator.
The $c,c^\dagger$ are fermion operators, $\sigma,\sigma'=\uparrow,\downarrow$
are spin labels, and $\boldsymbol\sigma$ are the Pauli matrices defined in
[Physics conventions](physics_conventions.md#spin-operators-and-equilibrium-averages).
One conventional Kanamori form is

$$
H_{\rm int}=U\sum_l n_{l\uparrow}n_{l\downarrow}
+\sum_{l<m}\left[(U'-J_H/2)n_ln_m-2J_H\mathbf S_l\cdot\mathbf S_m
+J_{\rm pair}(P_l^\dagger P_m+P_m^\dagger P_l)\right].
$$

$l<m$ counts each distinct orbital pair once. $U$ penalizes double occupation
of one orbital, $U'$ is direct interorbital repulsion, positive $J_H$ favors
parallel spins, and $J_{\rm pair}$ transfers a pair between orbitals.
All are energies (input eV, canonical meV). This Hamiltonian motivates the
local spin-channel RPA ansatz; nfit implements the ordered vertex below,
not a many-body solution or a self-consistent band Hamiltonian.

## Orbital-pair susceptibility and interaction

For basis states $l$ selected by site-attached `correlated_shell` labels, nfit
first calculates the ordered $|l_1\rangle\langle l_2|$ response
$\chi^0_{l_1l_2,l_3l_4}$. The local spin-channel vertex is

$$
\Gamma_{l_1l_2,l_3l_4} =
\begin{cases}
U & l_1=l_2=l_3=l_4,\\
U' & l_1=l_3\ne l_2=l_4,\\
J_H & l_1=l_2\ne l_3=l_4,\\
J_{\rm pair} & l_1=l_4\ne l_2=l_3,\\
0 & \text{otherwise}.
\end{cases}
$$

Each matrix index is an ordered pair of spatial-orbital labels:
$A=(l_1,l_2)$ means $O_A=|l_1\rangle\langle l_2|$, and similarly for
$B=(l_3,l_4)$. The response uses $O_A,O_B^\dagger$, so
$M^A_{nm}=u_{l_1n}^*(\mathbf k)u_{l_2m}(\mathbf k+\mathbf q)$;
all eigenvector, mesh, and occupation definitions are those of
[the bare bubble](lindhard.md#complex-susceptibility). With $N_o$ selected
orbitals the matrices have shape $N_o^2\times N_o^2$, the identity acts on
that pair space, $\chi^0$ has units meV$^{-1}$ per model cell for one spin
species, and $\Gamma$ has units meV.

All four indices must belong to the same site and correlated shell. Thus
identically named shells on different sites are not coupled by a local
interaction. The dressed matrix is

$$
\boldsymbol\chi=
\left[\mathbb 1-\boldsymbol\chi^0\boldsymbol\Gamma\right]^{-1}
\boldsymbol\chi^0 .
$$

The feedback equation is $\chi=\chi^0+\chi^0\Gamma\chi$. After solving it,
a local probe with weights $v_l(\mathbf Q)=f_l(|\mathbf Q|)
 e^{2\pi i\mathbf Q\cdot\mathbf r_l}$ gives one Cartesian component
$\chi_{s,\alpha\alpha}=\tfrac12\sum_{lm}v_l\chi_{ll,mm}v_m^*$.
$\mathbf r_l$ is the fractional orbital center, $\mathbf Q$ reduced
extended-zone transfer, $f_l$ the dimensionless radial amplitude (one for
intrinsic total spin), and $\alpha=x,y,z$. The factor 1/2 is the analytic
spin trace, applied after orbital RPA; off-diagonal Cartesian components vanish
in this implicit-spin isotropic model.

Orbital-center phases are retained when the result is projected onto the
isotropic physical spin tensor. Its imaginary part is used for inelastic
scattering and its real zero-energy limit for quasistatic and bulk response.

## Parameters

| API name | Meaning | Unit | Acceptable input example |
| --- | --- | --- | --- |
| `U` | intraorbital Hubbard interaction | eV | `2.0` |
| `U_prime` | interorbital direct interaction | eV | `1.4` |
| `J_H` | Hund exchange | eV | `0.3` |
| `J_pair` | pair-hopping interaction | eV | `0.3` |

With `rotationally_invariant=true`, nfit derives
$U'=U-2J_H$ and $J_{\rm pair}=J_H$; the stored `U_prime` and `J_pair` values
are not fitted. With it disabled, all four parameters are independent.
Interaction inputs and fit results use eV. The canonical interaction vertex
stores meV.

Because this vertex dresses the orbital-pair response *before* the spin trace
is applied, $U$ needs no extra factor: for a single correlated orbital the
zero-temperature, zero-width uniform instability is at $U D_\uparrow(\mu)=1$,
where $D_\uparrow$ is the per-spin DOS in states/(meV cell) at chemical
potential $\mu$. At finite temperature replace it by the Fermi-derivative
average in the [Lindhard static limit](lindhard.md#degenerate-transitions-and-the-static-limit). That is the same scale as the $I$ of
[Scalar Stoner RPA](stoner_rpa.md), so the two interaction parameters are
directly comparable.

## Configuration

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `response_component` | sibling `lindhard` component supplying the electronic model and bare-response settings | `""` | `"Bare response"` |
| `singular_tolerance` | relative singular-value threshold for an RPA pole | `1e-12` | `1e-10` |
| `near_pole_tolerance` | relative minimum singular value below which an evaluated denominator is flagged | `1e-3` | `0.01` |
| `static_stability_warning_margin` | sampled static margin below which the result is marked near instability | `0.05` | `0.1` |
| `reject_sampled_static_instability` | give sampled zero-energy crossings a finite fitting penalty | `false` | `true` |
| `correlated_shells` | shell labels included in the interaction; empty selects every labelled shell | `[]` | `["M1_3d"]` |
| `rotationally_invariant` | enforce $U'=U-2J_H$ and $J_{\rm pair}=J_H$ | `true` | `false` |

The dressing replaces the referenced bare observable on its own dataset
scope. The broadening, mesh, chemical-potential policy, and normalization
remain defined once on the referenced Lindhard component. Orbital form factors
remain defined once on the referenced tight-binding basis.
The reported pole and static-margin diagnostics cover evaluated points.
Because finding the global electronic instability requires a separately
converged $\mathbf q$ scan, the sampled margin is not described as a full-BZ
stability proof. Rejection is opt-in and supplies a finite fit penalty at a
sampled zero-energy crossing. Post-fit diagnostics report the corresponding
probe at the referenced Lindhard component's `plot_q_reduced`.

## Calculable data

The model calculates single-crystal and powder inelastic neutron scattering,
single-crystal and powder quasistatic elastic scattering, uniform bulk
susceptibility, and linear-response magnetization.

Bulk comparison is optional. A local Hubbard--Hund RPA approximation may be
inadequate at $\mathbf Q=0$ or near strong-coupling and orbital-selective
regimes; disagreement is useful evidence about the approximation rather than
a requirement to include or exclude a bulk dataset.

## Scripting, fitting, and export

```python
from nfit import (
    bare_lindhard_susceptibility,
    correlated_basis_indices,
    hubbard_hund_spin_vertex,
    orbital_pair_operator_basis,
    project_implicit_spin_response,
    rpa_dress_susceptibility,
)

indices = correlated_basis_indices(model, ["M1_3d"])
operators = orbital_pair_operator_basis(model, list(indices))
bare_pairs = bare_lindhard_susceptibility(
    model, Q_reduced, energy_meV, mesh, operators,
    temperature_K=20.0,
    chemical_potential_meV=0.0,
    broadening_meV=2.0,
)
vertex = hubbard_hund_spin_vertex(
    model, indices, U=2.0, J_H=0.3, rotationally_invariant=True
)
dressed_pairs = rpa_dress_susceptibility(bare_pairs, vertex)
dressed_spin = project_implicit_spin_response(dressed_pairs, model, indices)
```

All independent interactions can use the standard fit, bounds, sharing, and
uncertainty machinery. Exported results record the selected basis, shell
locality, resolved rotational constraints, operator ordering, interaction
matrix, and RPA pole diagnostics.

The orbital-pair calculation uses the referenced response's bounded transition
batches, digest-aware eigensystem cache, and certified execution backend.
Automatic response-mesh symmetry reduction is intentionally disabled for this
matrix basis until its full orbital transformation can be certified; the
calculation records a full-mesh fallback.

## Current limits

- The standard preset requires an implicit-spin normal-state model. Explicit
  SOC/spinor models can use `matrix_rpa`; a spinor Hubbard--Hund vertex needs a
  separately documented basis transformation.
- The interaction is static and local.
- RPA does not include a self-energy, dynamical vertex correction, or ordered
  state beyond a diagnosed pole.

## References

- S. Graser *et al.*, *New J. Phys.* **11**, 025016 (2009),
  [doi:10.1088/1367-2630/11/2/025016](https://doi.org/10.1088/1367-2630/11/2/025016).
- A. Georges *et al.*, *Annu. Rev. Condens. Matter Phys.* **4**, 137 (2013),
  [doi:10.1146/annurev-conmatphys-020911-125045](https://doi.org/10.1146/annurev-conmatphys-020911-125045).
