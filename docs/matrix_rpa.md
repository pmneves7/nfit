# Matrix RPA interaction

`matrix_rpa` applies a user-defined Hermitian interaction matrix to a
referenced [bare Lindhard response](lindhard.md). Use it for a compact
anisotropic spin interaction or to test a vertex supplied by another
calculation without changing the electronic Hamiltonian.

## Complex susceptibility

In the ordered $(S_x,S_y,S_z)$ basis,

$$
\boldsymbol\Gamma=g\,\mathbf V,\qquad
\boldsymbol\chi_s=
\left[\mathbb 1-\boldsymbol\chi^0_s\boldsymbol\Gamma\right]^{-1}
\boldsymbol\chi^0_s .
$$

The configured Hermitian matrix $\mathbf V$ is dimensionless and the fitted
scale $g$ is entered in eV. nfit converts the resulting vertex to canonical
meV before combining it with the bare response. The final inelastic response
is $\boldsymbol\chi_s''=\operatorname{Im}\boldsymbol\chi_s$; quasistatic and
bulk calculations use the real zero-energy limit.

For a completely general operator basis, scripts can construct an
`InteractionVertex` with `matrix_interaction_vertex`. The GUI preset currently
uses the physical Cartesian spin basis so neutron and bulk projection remain
unambiguous.

## Parameters

| API name | Meaning | Unit | Acceptable input example |
| --- | --- | --- | --- |
| `scale` | energy multiplying `vertex_matrix` | eV | `0.2` |

## Configuration

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `response_component` | sibling `lindhard` component supplying the bare spin response and its source settings | `""` | `"Bare response"` |
| `singular_tolerance` | relative singular-value threshold for an RPA pole | `1e-12` | `1e-10` |
| `vertex_matrix` | dimensionless Hermitian matrix ordered as $(S_x,S_y,S_z)$ | identity | `[[1,0,0],[0,0.8,0],[0,0,1.2]]` |
| `channel` | descriptive channel stored with the vertex | `"spin"` | `"spin"` |

The dressing consumes the referenced bare observable only on its own dataset
scope. The bare component remains available independently on other datasets.

## Calculable data

The model calculates single-crystal and powder inelastic neutron scattering,
single-crystal and powder quasistatic elastic scattering, uniform bulk
susceptibility, and linear-response magnetization. The referenced Lindhard
component supplies the form factor, normalization, powder average, and
electronic-response settings.

Bulk comparison is optional. A phenomenological matrix that describes
finite-$\mathbf Q$ fluctuations need not reproduce the uniform
susceptibility.

## Scripting, fitting, and export

```python
from nfit import matrix_interaction_vertex, rpa_dress_susceptibility

vertex = matrix_interaction_vertex(
    bare.operator_labels,
    [[0.20, 0, 0], [0, 0.16, 0], [0, 0, 0.24]],
    energy_unit="eV",
    channel="spin",
)
dressed = rpa_dress_susceptibility(bare, vertex)
```

The scale can be fixed, fitted, and shared using the standard machinery.
Projects, fit results, reports, and scripts retain the matrix, operator order,
channel, unit conversion, and pole diagnostics.
The referenced response supplies the cache, batching, backend certification,
and convergence settings. Certified implicit-spin bare responses may use
little-group mesh reduction before this Cartesian interaction is applied.

## Current limits

- The GUI preset uses a static $3\times3$ Cartesian spin matrix and one
  fittable scale.
- Momentum- or frequency-dependent vertices are available through the public
  `InteractionVertex` API but not yet through this GUI preset.
- A near-singular RPA denominator is reported, not continued into an ordered
  phase.

## References

- S. Graser *et al.*, *New J. Phys.* **11**, 025016 (2009),
  [doi:10.1088/1367-2630/11/2/025016](https://doi.org/10.1088/1367-2630/11/2/025016).
