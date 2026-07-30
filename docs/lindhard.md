# Bare Lindhard spin susceptibility

`lindhard` calculates the complex particle--hole response of a referenced
`tight_binding` component. Use it when measured magnetic fluctuations can be
compared with the noninteracting band response before a Stoner or
Hubbard--Hund interaction dressing is introduced.

The tight-binding component remains a separate electronic-structure model.
Its onsite, hopping, and spin--orbit coefficients can be fixed or fitted
through the same simultaneous fit, while `lindhard` owns the response
broadening and response-specific settings.

## Complex susceptibility

For one-particle operators $O_A$, the generalized bare response is

$$
\chi^0_{AB}(\mathbf q,E)=
-\sum_{\mathbf k}w_{\mathbf k}\sum_{nm}
\frac{f(\epsilon_{n\mathbf k})-f(\epsilon_{m,\mathbf k+\mathbf q})}
{E+\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}+i\eta}
M^A_{nm}M^{B*}_{nm},
$$

where $w_{\mathbf k}$ are normalized mesh weights, $f$ is the Fermi function,
$\eta>0$ is the lifetime broadening in meV, and

$$
M^A_{nm}=
\langle n,\mathbf k|O_A|m,\mathbf k+\mathbf q\rangle .
$$

The corresponding dissipative matrix is the imaginary part

$$
\chi^{0\prime\prime}_{AB}=
\sum_{\mathbf k,n,m}w_{\mathbf k}
\frac{[f(\epsilon_{n\mathbf k})-f(\epsilon_{m,\mathbf k+\mathbf q})]\eta}
{[E+\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}]^2+\eta^2}
M^A_{nm}M^{B*}_{nm}.
$$

This convention gives positive diagonal $\chi^{0\prime\prime}$ for a
positive-energy absorption process. The response obeys the corresponding
causal frequency and $\mathbf q\leftrightarrow-\mathbf q$ conjugation
relation; a Hermitian scalar response at a symmetry-equivalent wavevector has
$\chi^0(-E)=\chi^0(E)^*$. At $E=0$, exactly degenerate intraband terms use the
Fermi-function derivative rather than the undefined ratio $0/0$.

`orbital_pair_operator_basis(model)` constructs the complete ordered
$|a\rangle\langle b|$ basis and its conjugate map.
`bare_lindhard_susceptibility` evaluates any declared one-particle operator
basis. The model component uses `bare_spin_susceptibility`, which projects
directly onto Cartesian $S_x$, $S_y$, and $S_z$.

## Spin and neutron projection

For an extended-zone transfer $\mathbf Q$, the magnetic operator contains the
orbital-center phase

$$
S_\alpha(\mathbf Q)=
\sum_{ab}
e^{2\pi i\mathbf Q\cdot\mathbf r_a}
(S_\alpha)_{ab}|a\rangle\langle b|.
$$

The Hamiltonian is evaluated at $\mathbf q=\mathbf Q\bmod\mathbf G$, but the
operator retains the full $\mathbf Q$. An explicit spinor model supplies its
spin matrices. A spin-independent model uses the equivalent isotropic
spin-degenerate response without expanding the Hamiltonian to a redundant
spinful basis.

For neutron scattering, nfit contracts the Cartesian tensor with

$$
P_{\alpha\beta}(\mathbf Q)
=\delta_{\alpha\beta}-\widehat Q_\alpha\widehat Q_\beta .
$$

At $\mathbf Q=0$, the orientation-independent limit is
$P_{\alpha\beta}=\tfrac23\delta_{\alpha\beta}$. The dataset layer then applies
the magnetic form factor, fluctuation--dissipation factor, normalization, and
dataset scale described in [Physics conventions](physics_conventions.md).

## Parameters

| API name | Meaning | Unit | Acceptable input example |
| --- | --- | --- | --- |
| `broadening` | positive particle--hole lifetime broadening $\eta$ | meV | `2.0` |

The broadening regularizes the response but is not a replacement for checking
mesh convergence. It may be fitted. Onsite, hopping, and spin--orbit fit
parameters remain on the referenced tight-binding component.

## Configuration

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `electronic_component` | enabled sibling `tight_binding` component supplying $H(\mathbf k)$ | `""` | `"Bands"` |
| `response_mesh` | full-zone integration-mesh sizes | `[16, 16, 16]` | `[48, 48, 24]` |
| `response_mesh_shift` | offsets in mesh steps | `[0, 0, 0]` | `[0.5, 0.5, 0.5]` |
| `chemical_potential_mode` | use the source chemical potential or solve from filling | `"source"` | `"source"` or `"filling"` |
| `filling_per_cell` | electron count per primitive cell in filling mode | `1.0` | `3.0` |
| `response_backend` | eigensystem backend used by the reference response | `"numpy"` | `"numpy"`, `"threaded"`, or `"cupy"` |
| `response_workers` | bounded CPU worker count | `1` | `8` |
| `response_max_batch_mb` | eigensystem temporary-memory target | `256.0` | `512.0` |
| `powder_orientations` | deterministic sphere directions for powder averaging | `50` | `96` |
| `formula_units_per_cell` | formula units in the primitive electronic cell for molar bulk conversion | `1.0` | `2.0` |
| `ion` | tabulated magnetic form-factor ion; empty applies none | `""` | `"Fe2"` |
| `form_factor_coefficients` | custom form-factor coefficients instead of `ion` | `""` | `"0.0263,34.96,0.3668,15.94,0.6188,5.594,-0.0119"` |
| `bulk_g_factor` | Landé factor for bulk conversion | `2.0` | `2.1` |
| `plot_q_reduced` | extended-zone $\mathbf Q$ for the model energy scan | `[0, 0, 0]` | `[0.5, 0.5, 0]` |
| `plot_energy_min_meV` | lower plotted energy transfer | `-100.0` | `-50.0` |
| `plot_energy_max_meV` | upper plotted energy transfer | `100.0` | `50.0` |
| `plot_energy_points` | plotted energy samples | `401` | `501` |
| `plot_temperature_K` | plotted response temperature | `10.0` | `20.0` |

The mesh must match the periodic dimensionality of the electronic model. A
three-entry mesh or shift is also accepted for a reduced-dimensional model;
nfit selects its periodic axes. Phase 4 uses the full mesh as its numerical
reference. Accelerated eigensystem backends are explicit and recorded in
provenance; response-level symmetry reduction is deferred until the
Hamiltonian, magnetic operators, and requested observable can all be
certified.

Filling mode solves one chemical potential at each required temperature. It
therefore requires $T>0$ in the current implementation. Source mode uses
`chemical_potential_meV` from the tight-binding component.

## Calculable data

The model calculates:

- single-crystal and powder inelastic neutron scattering from
  $\chi^{0\prime\prime}_s(\mathbf Q,E)$;
- single-crystal and powder quasistatic elastic magnetic scattering from the
  static $\chi^{0\prime}_s(\mathbf Q,0)$;
- uniform bulk susceptibility; and
- magnetic moment or magnetization in linear response to an applied field.

Powder calculations average the tensor response and neutron polarization over
the configured sphere directions. Bulk calculations use the isotropic
$\operatorname{Tr}\boldsymbol\chi_s(\mathbf 0,0)/3$ response, convert from the
primitive-cell normalization to a formula-unit normalization, and then follow
the bulk units on the physics-conventions page.

The bulk comparison is optional. A bare band response may be intentionally
insufficient at $\mathbf q=0$, especially near an interaction-driven magnetic
instability. That discrepancy can motivate an interaction dressing rather
than requiring the bulk dataset to be omitted.

## Scripting, fitting, and export

The calculation API is independent of Qt:

```python
import numpy as np

from nfit import bare_spin_susceptibility, k_mesh

mesh = k_mesh(model, [32, 32, 32], symmetry="full")
energy_meV = np.linspace(-40.0, 40.0, 401)
Q_reduced = np.broadcast_to([0.5, 0.5, 0.0], (energy_meV.size, 3))
result = bare_spin_susceptibility(
    model,
    Q_reduced,
    energy_meV,
    mesh,
    temperature_K=20.0,
    chemical_potential_meV=0.0,
    broadening_meV=2.0,
)
chi_prime = result.chi_prime
chi_double_prime = result.chi_double_prime
```

`SusceptibilityResult` records reduced $\mathbf q$, extended-zone $\mathbf Q$,
complex values, operator labels, normalization, thermodynamic settings, mesh,
backend, and provenance. `to_dict()` and `from_dict()` provide a portable
round trip for modest results.

The GUI model plot shows the real and imaginary isotropic response at
`plot_q_reduced`. **Copy script** exports the electronic source, response
settings, calculation, and rendering calls as editable Python. The standard
fit machinery saves and exports the linked electronic and response component
states, fitted parameters, dataset bindings, and calculated fit channels.

## Current limits

- The kernel evaluates the full band-pair sum and can be expensive for large
  orbital bases or dense meshes.
- Response-level symmetry reduction, transition chunking, and distributed
  execution are not yet enabled.
- This component is the bare bubble with finite lifetime broadening. Separate
  scalar Stoner, matrix, and Hubbard--Hund RPA components can dress it;
  self-energy and superconducting extensions are not yet included.
- The basis is orthonormal. Nonorthogonal overlap matrices are not supported.

## References

- J. Lindhard, *Kgl. Danske Videnskab. Selskab, Mat.-Fys. Medd.* **28**,
  no. 8 (1954).
- S. Graser *et al.*, *New J. Phys.* **11**, 025016 (2009),
  [doi:10.1088/1367-2630/11/2/025016](https://doi.org/10.1088/1367-2630/11/2/025016).
