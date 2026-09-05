# Matrix RPA interaction

`matrix_rpa` applies a user-defined Hermitian interaction matrix to a
referenced [bare Lindhard response](lindhard.md). Use it for a compact
anisotropic spin interaction or to test a vertex supplied by another
calculation without changing the electronic Hamiltonian.

## Complex susceptibility

The starting ansatz is linear interaction feedback,
$\boldsymbol\chi=\boldsymbol\chi^0+\boldsymbol\chi^0\boldsymbol\Gamma\boldsymbol\chi$,
with static bands and a static local vertex. $\chi^0$ is the bare Lindhard
response in meV$^{-1}$ per model cell and $\chi$ the dressed response in the
same units, evaluated at the same momentum and energy. The identity below
is $3\times3$, and $S_x,S_y,S_z$ are the dimensionless spin operators defined
in [Physics conventions](physics_conventions.md#spin-operators-and-equilibrium-averages).

In the ordered $(S_x,S_y,S_z)$ basis,

$$
\boldsymbol\Gamma=g\,\mathbf V,\qquad
\boldsymbol\chi_s=
\left[\mathbb 1-\boldsymbol\chi^0_s\boldsymbol\Gamma\right]^{-1}
\boldsymbol\chi^0_s .
$$

The configured Hermitian matrix $\mathbf V$ is dimensionless and the fitted
scale $g$ (`scale`) is entered in eV; this energy-valued $g$ is unrelated
to the dimensionless Landé factor. nfit converts the resulting vertex to canonical
meV before combining it with the bare response. The final inelastic response
is the absorptive tensor $(\boldsymbol\chi_s-\boldsymbol\chi_s^\dagger)/(2i)$,
which equals ordinary $\operatorname{Im}\chi_s$ for scalar or diagonal channels;
quasistatic and
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
| `near_pole_tolerance` | relative minimum singular value below which an evaluated denominator is flagged | `1e-3` | `0.01` |
| `static_stability_warning_margin` | sampled static margin below which the result is marked near instability | `0.05` | `0.1` |
| `reject_sampled_static_instability` | give sampled zero-energy crossings a finite fitting penalty | `false` | `true` |
| `vertex_matrix` | dimensionless Hermitian matrix ordered as $(S_x,S_y,S_z)$ | identity | `[[1,0,0],[0,0.8,0],[0,0,1.2]]` |
| `channel` | descriptive channel stored with the vertex | `"spin"` | `"spin"` |

The dressing consumes the referenced bare observable only on its own dataset
scope. The bare component remains available independently on other datasets.
Pole diagnostics cover every evaluated response point. The static stability
margin uses evaluated zero-energy points only and is therefore not a global
Brillouin-zone proof. Enabling rejection turns a sampled crossing into a large
finite fit penalty. Post-fit diagnostics report the corresponding zero-energy
probe at the referenced Lindhard component's `plot_q_reduced`.

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
