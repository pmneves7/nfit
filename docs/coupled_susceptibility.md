# Coupled susceptibilities

The `coupled_susceptibility` component mixes two existing complex scalar
magnetic responses through a bilinear interaction. It is a composition model:
the two referenced components supply their own momentum, energy, crystal, and
form-factor dependence, while this component supplies only their coupling.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `coupling` | bilinear interaction energy $g$ between sectors $A$ and $B$ | meV |

The configuration fields `response_a` and `response_b` name two distinct,
enabled model components. `singular_tolerance` is a positive dimensionless
numerical threshold for rejecting a sampled pole of the coupled response.

## Complex susceptibility

$O_A,O_B$ are dimensionless scalar spin-component operators for the two
sectors, with compatible normalization and uncoupled susceptibilities
$\chi_A(\mathbf Q,E),\chi_B(\mathbf Q,E)$ in meV$^{-1}$.
Their response is to conjugate fields in meV. The bilinear ansatz
$H_{AB}=-gO_AO_B$ gives induced fields
$h_A^{\rm eff}=h_A+g\langle O_B\rangle$ and
$h_B^{\rm eff}=h_B+g\langle O_A\rangle$ in linear response.
This yields the $2\times2$ inverse matrix in sector order $(A,B)$ below.
The real energy $g$ is unrelated to the dimensionless Landé factor.

For uncoupled scalar responses $\chi_A$ and $\chi_B$, nfit solves

$$
\boldsymbol\chi^{-1}=
\begin{pmatrix}
\chi_A^{-1} & -g\\
-g & \chi_B^{-1}
\end{pmatrix}.
$$

If the neutron-visible operator is
$O=F_A(\mathbf Q)O_A+F_B(\mathbf Q)O_B$, the returned susceptibility is

$$
\chi_O=
\frac{F_A^2\chi_A+F_B^2\chi_B+2F_AF_Bg\chi_A\chi_B}
{1-g^2\chi_A\chi_B}.
$$

$F_A$ and $F_B$ are real, dimensionless signed magnetic form-factor amplitudes.
The displayed numerator assumes these real amplitudes, as used by this
component; complex probe weights would require adjoints and absolute squares. Consequently,
`coupling=0` reduces to the additive neutron response of the two source
components, including their separate form factors. The sign of $g$ matters
through the interference term. The coupling is applied to the causal complex
susceptibilities before the Bose factor, neutron cross-section constant,
polarization factor, and dataset scale.

The sampled response is rejected when
$|1-g^2\chi_A\chi_B|$ falls below `singular_tolerance` relative to the terms
forming the denominator. This protects the optimizer at a pole; it is not a
claim that stability has been established between sampled points.

## Calculable data

The component calculates single-crystal inelastic and quasistatic elastic
magnetic scattering. Elastic data evaluate both sources at $E=0$ before
coupling them.

Composable sources are currently the scalar forms of:

- `local_relaxational`;
- `mmp_relaxational`;
- `generalized_paramagnon`;
- `conserved_ferromagnetic`;
- `heisenberg_rpa`; and
- another acyclic `coupled_susceptibility` component.

Tensor or Zeeman Heisenberg-RPA responses and powder coupling are deliberately
not coerced into the scalar contract. The two source components are consumed
on the coupled component's dataset scope, so their bare observables are not
also added a second time. Their parameters remain fit parameters and the bare
components can still apply independently to other datasets.

Because two sector amplitudes, two line shapes, and $g$ may be correlated,
start by fitting or fixing the source responses, then release `coupling`.
`coupling=0` is the direct nested-model check.

## Scripting and export

Create the two source `ModelComponentSpec` objects first, then add
`ModelComponentSpec(type="coupled_susceptibility", ...)` with their component
names in `response_a` and `response_b`. Component dependencies, parameter
sharing, project serialization, and editable workflow export use the same
machinery as electronic-response dressings. Dependency cycles and missing or
incompatible sources are rejected during compilation.
