# Scalar Stoner RPA

`stoner_rpa` dresses a referenced [bare Lindhard response](lindhard.md) with
one isotropic spin interaction. It is the smallest interaction model for
testing whether a band susceptibility is enhanced toward an itinerant magnetic
instability.

## Complex susceptibility

RPA means random-phase approximation: the feedback ansatz is
$\chi=\chi^0+\chi^0\Gamma\chi$. It sums repeated interactions in the response
while retaining the independent bands and occupation factors of
[the Lindhard model](lindhard.md#complex-susceptibility).
$\mathbf Q$ is experimental momentum, $E$ is transfer in meV, and
$S_\alpha$ is the dimensionless spin defined in
[Physics conventions](physics_conventions.md#spin-operators-and-equilibrium-averages).
$\mathbb1$ is the identity on the three Cartesian spin channels; multiplication
by a scalar vertex below means $\Gamma\mathbb1$.

For the Cartesian bare spin tensor $\boldsymbol\chi^0_s(\mathbf Q,E)$,

$$
\boldsymbol\chi_s=
\left[\mathbb 1-\boldsymbol\chi^0_s\,\Gamma\right]^{-1}
\boldsymbol\chi^0_s ,
\qquad
\Gamma=2I .
$$

$I$ is entered in eV and converted to meV before it enters the vertex
$\Gamma$, which multiplies $\chi^0_s$ in meV$^{-1}$ per primitive electronic
cell. The dissipative response used for inelastic scattering is
$(\boldsymbol\chi_s-\boldsymbol\chi_s^\dagger)/(2i)$; on scalar or diagonal
channels this equals $\operatorname{Im}\chi_s$. The real static
limit is used for quasistatic elastic and bulk comparisons.

### Why the vertex is $2I$

For one local orbital, $n_\sigma=c_\sigma^\dagger c_\sigma$ counts electrons
of spin $\sigma$, $n=n_\uparrow+n_\downarrow$, and
$S_z=(n_\uparrow-n_\downarrow)/2$. Fermion creation/annihilation and Pauli
matrices are defined on the [tight-binding](tight_binding.md#reciprocal-space-hamiltonian-and-bands)
and [spin](tight_binding_spin.md) pages. The identity
$n_\uparrow n_\downarrow=n^2/4-S_z^2$ fixes the density-to-spin convention.

$\chi^0_s$ is the susceptibility of the *dimensionless* spin operator
$S^\alpha$. Since $S_z=(n_\uparrow-n_\downarrow)/2$ carries a factor $1/2$ per
spin index, an on-site interaction written for the densities,
$H=I\,n_\uparrow n_\downarrow$, becomes $-I S_z^2$ up to a charge term, so the
irreducible vertex conjugate to $S^\alpha$ is $2I$. For an implicit-spin model
the zero-temperature, zero-width uniform limit is
$\chi^0_s=D_\uparrow(\mu)/2$, where $D_\uparrow$ is the per-spin density of
states in states/(meV model cell) and $\mu$ is chemical potential in meV.
At finite temperature use the Fermi-derivative average defined on the
[Lindhard page](lindhard.md#degenerate-transitions-and-the-static-limit).
Thus in this uniform limit the RPA denominator is

$$
1-2I\chi^0_s=1-I D_\uparrow(\mu),
$$

and the instability sits at the textbook Stoner criterion
$I D_\uparrow(\mu)=1$. With this convention $I$ is directly comparable with the
Hubbard $U$ of [Hubbard--Hund RPA](hubbard_hund_rpa.md), which dresses the
orbital-pair response *before* the same spin trace is applied. The applied
factor is recorded in the exported vertex provenance as
`spin_channel_vertex_factor`.

When orbitals have different magnetic form factors, nfit retains the
form-factor-weighted neutron probe and the unweighted total spin as separate
operators. RPA dresses the total-spin channel, then projects back to the probe:

$$
\chi_{ff}=\chi^0_{ff}+\chi^0_{fS}\Gamma
(1-\chi^0_{SS}\Gamma)^{-1}\chi^0_{Sf}.
$$

Here $S$ labels the intrinsic total-spin probe and $f$ labels the
form-factor-weighted spin probe, not the Fermi function. $\chi^0_{fS}$ is
the bare cross response of those two probes, with the second operator
adjointed, and $\chi^0_{Sf}$ the reverse cross response.
All blocks have units meV$^{-1}$ per model cell; for Cartesian tensors,
$1$ in the inverse means the spin-space identity. Orbital label $a$ indexes
the dimensionless radial amplitudes $f_a(Q)$ defined on the
[Lindhard page](lindhard.md#spin-and-neutron-projection).

Consequently $f_a(Q)$ appears in the measured numerator and interference
terms, but never in the Stoner denominator or stability criterion.

The implementation records the operator order, interaction matrix, RPA
multiplication order, smallest relative singular value, condition number, and
sampled static stability margin. A denominator within `singular_tolerance` of
a pole raises an error instead of returning an unstable finite number.
For an implicit-spin isotropic response, nfit evaluates the equivalent scalar
closed form pointwise and stores extrema and counts rather than per-point
singular-value arrays; matrix and explicit-spin responses retain the general
batched solve.

## Parameters

| API name | Meaning | Unit | Acceptable input example |
| --- | --- | --- | --- |
| `I` | Stoner interaction; instability at $I D_\uparrow(\mu)=1$ | eV | `0.35` |

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
$1-\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of
$(X+X^\dagger)/2$ with $X=\boldsymbol\chi^0\boldsymbol\Gamma$ at $E=0$.
The eigenvalue and margin are dimensionless. This is a sampled diagnostic, not proof
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
