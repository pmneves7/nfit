# Hubbard--Hund RPA

`hubbard_hund_rpa` constructs a local multiorbital spin-channel interaction on
the correlated shells of a referenced tight-binding model, dresses its
orbital-pair Lindhard response, and then projects the result onto physical
spin.

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

All four indices must belong to the same site and correlated shell. Thus
identically named shells on different sites are not coupled by a local
interaction. The dressed matrix is

$$
\boldsymbol\chi=
\left[\mathbb 1-\boldsymbol\chi^0\boldsymbol\Gamma\right]^{-1}
\boldsymbol\chi^0 .
$$

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

## Configuration

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `response_component` | sibling `lindhard` component supplying the electronic model and bare-response settings | `""` | `"Bare response"` |
| `singular_tolerance` | relative singular-value threshold for an RPA pole | `1e-12` | `1e-10` |
| `correlated_shells` | shell labels included in the interaction; empty selects every labelled shell | `[]` | `["M1_3d"]` |
| `rotationally_invariant` | enforce $U'=U-2J_H$ and $J_{\rm pair}=J_H$ | `true` | `false` |

The dressing replaces the referenced bare observable on its own dataset
scope. The broadening, mesh, chemical-potential policy, form factor, and
normalization remain defined once on the referenced Lindhard component.

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
