# Scalar Stoner RPA

`stoner_rpa` dresses a referenced [bare Lindhard response](lindhard.md) with
one isotropic spin interaction. It is the smallest interaction model for
testing whether a band susceptibility is enhanced toward an itinerant magnetic
instability.

## Complex susceptibility

For the Cartesian bare spin tensor $\boldsymbol\chi^0_s(\mathbf Q,E)$,

$$
\boldsymbol\chi_s=
\left[\mathbb 1-\boldsymbol\chi^0_s I\right]^{-1}
\boldsymbol\chi^0_s .
$$

$I$ is entered in eV and converted to meV before multiplication by
$\chi^0_s$, which is in meV$^{-1}$ per primitive electronic cell. The
dissipative response used for inelastic scattering is
$\boldsymbol\chi_s''=\operatorname{Im}\boldsymbol\chi_s$. The real static
limit is used for quasistatic elastic and bulk comparisons.

The implementation records the operator order, interaction matrix, RPA
multiplication order, smallest relative singular value, condition number, and
sampled static stability margin. A denominator within `singular_tolerance` of
a pole raises an error instead of returning an unstable finite number.

## Parameters

| API name | Meaning | Unit | Acceptable input example |
| --- | --- | --- | --- |
| `I` | scalar Stoner interaction in each Cartesian spin channel | eV | `0.35` |

`I` may be fixed, fitted, or shared across datasets using the common model
parameter controls.

## Configuration

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `response_component` | sibling `lindhard` component supplying $\chi^0_s$, the tight-binding model, mesh, broadening, and dataset conversion | `""` | `"Bare response"` |
| `singular_tolerance` | relative singular-value threshold used to identify an RPA pole | `1e-12` | `1e-10` |
| `near_pole_tolerance` | relative minimum singular value below which an evaluated denominator is flagged | `1e-3` | `0.01` |
| `static_stability_warning_margin` | sampled static margin below which the result is marked near instability | `0.05` | `0.1` |
| `reject_sampled_static_instability` | give sampled zero-energy crossings a finite fitting penalty | `false` | `true` |

The dressing replaces its referenced bare component on the datasets to which
the dressing applies. The same bare component may still be compared directly
with other datasets by giving the two components different dataset scopes.

At an evaluated zero-energy point, nfit reports
$1-\lambda_{\max}$ using the Hermitian part of
$\boldsymbol\chi^0\boldsymbol\Gamma$. This is a sampled diagnostic, not proof
of stability throughout the Brillouin zone. If rejection is enabled, a sampled
crossing produces the same kind of large finite fitting penalty used by the
Heisenberg RPA model. Model energy plots identify near sampled poles or
instabilities. Post-fit diagnostics repeat the zero-energy test at
`plot_q_reduced` and include its margin, feedback ratio, and relative minimum
singular value in the GUI and fit report.

## Calculable data

The model calculates single-crystal and powder inelastic neutron scattering,
single-crystal and powder quasistatic elastic scattering, uniform bulk
susceptibility, and linear-response magnetization. Form factor, neutron
polarization, fluctuation--dissipation conversion, normalization, and dataset
scale are inherited from the referenced bare-response calculation.

The bulk comparison is optional. A scalar Stoner enhancement is deliberately
simple and may fail to describe the finite-$\mathbf Q$ response and
$\mathbf Q=0$ susceptibility with one value of $I$.

## Scripting, fitting, and export

The dressing API is independent of Qt:

```python
from nfit import rpa_dress_susceptibility, scalar_stoner_vertex

vertex = scalar_stoner_vertex(bare.operator_labels, 0.35, energy_unit="eV")
dressed = rpa_dress_susceptibility(bare, vertex)
chi_prime = dressed.chi_prime
chi_double_prime = dressed.chi_double_prime
```

The GUI component references a named bare response, participates in
simultaneous fitting, and is serialized with its dependency. Fit reports and
exported results retain the interaction convention and canonical vertex.
The referenced response also supplies bounded transition batches, digest-aware
eigensystem reuse, backend certification, symmetry policy, and numerical
convergence plots.

## Current limits

- The scalar vertex has no orbital, momentum, or frequency dependence.
- A pole is diagnosed but not regularized into an ordered-state theory.
- Self-energy and vertex corrections beyond the stated RPA equation are not
  included.

## References

- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*, Springer
  (1985), [doi:10.1007/978-3-642-82499-9](https://doi.org/10.1007/978-3-642-82499-9).
