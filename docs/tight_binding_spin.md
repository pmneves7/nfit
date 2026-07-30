# Spin and spin--orbit coupling

Spin is an optional extension of the spatial-orbital basis. nfit keeps the
basis spin implicit unless an explicit representation is needed, avoiding a
factor-of-two increase in Hamiltonian dimension for an SU(2)-symmetric
paramagnet.

## Spin treatments

`spin_treatment` accepts four policies:

| Setting | Resolved basis | Use and assumptions |
| --- | --- | --- |
| `"auto"` | implicit without SOC; spinor with active SOC | recommended default |
| `"implicit"` | $N$ states | spin-degenerate, SU(2)-symmetric Hamiltonian |
| `"collinear"` | $2N$ states | explicit up/down labels and spin operators, but no generated spin mixing |
| `"spinor"` | $2N$ states | SOC or other explicitly constructed spin-mixing terms |

For an explicit basis, nfit uses orbital-major order

$$
(a\uparrow,a\downarrow,b\uparrow,b\downarrow,\ldots).
$$

A spin-independent spatial Hamiltonian and every named spatial parameter are
lifted as

$$
H_{\rm spin}(\mathbf R)=H_{\rm orb}(\mathbf R)\otimes I_2,
\qquad
P_{p,\rm spin}(\mathbf R)=P_{p,\rm orb}(\mathbf R)\otimes I_2 .
$$

The physical spin operators are

$$
S_\alpha=I_N\otimes\frac{\sigma_\alpha}{2}.
$$

Thus `collinear` alone does not create exchange splitting or independent
spin-up and spin-down hopping coefficients. Those require explicitly supplied
spin-dependent matrices. Generated hoppings remain spin independent.

In the implicit representation, later response calculations apply the
two-fold degeneracy and spin trace analytically. This path is both faster and
less memory intensive.

## Onsite spin--orbit coupling

Each active `SpinOrbitTerm` adds

$$
H_{{\rm SOC},m}
=\lambda_m\sum_{\alpha=x,y,z}L_{m,\alpha}\otimes S_\alpha
$$

to a named orbital manifold $m$. The coefficient $\lambda_m$ is an electronic
energy: it is normally entered in eV, stored in meV, and can use the same fit,
bounds, and sharing controls as hopping parameters.

| Field | Meaning | Example |
| --- | --- | --- |
| `identifier` | stable parameter identifier | `"M1_d:soc:lambda"` |
| `label` | readable name | `"M1_d λ L·S"` |
| `manifold_label` | spatial manifold receiving SOC | `"M1_d"` |
| `value_meV` | canonical coupling $\lambda_m$ | `25.0` |
| `bounds_meV` | optional canonical bounds | `[0.0,100.0]` |
| `fit` | optimizer selection | `false` |
| `prescription` | source of $L_x,L_y,L_z$ | `"auto"`, `"atomic"`, `"projected"`, or `"effective"` |
| `orbital_operators` | explicit local angular-momentum matrices | complex array of shape `(3,n,n)` |

The prescriptions make different physical assumptions:

- `atomic` uses the exact angular-momentum matrices of a complete
  $(2l+1)$-state analytic shell and is rejected for a truncated shell.
- `projected` evaluates $P L_\alpha P$ in a selected symmetry subspace.
  Angular momentum may be quenched, and virtual coupling through excluded
  orbitals is absent.
- `effective` uses explicitly supplied Hermitian orbital operators. This is
  the route for an effective, custom, or characterized numerical basis.
- `auto` chooses `atomic` for a complete analytic shell and `projected` for
  an analytic subspace.

For analytic manifolds, `harmonic_transform` fixes the basis convention. nfit
constructs the angular-momentum matrices in that convention, projects them,
and rotates their Cartesian components from the manifold's local frame to the
crystal frame.

Explicit `implicit` or `collinear` treatment is rejected when SOC is active
because neither representation can contain the required spin mixing.

## Time-reversal validation

The builder treats its spinor model as nonmagnetic and validates

$$
H(\mathbf k)
=U_\Theta H(-\mathbf k)^*U_\Theta^\dagger,
\qquad
U_\Theta=I_N\otimes i\sigma_y .
$$

`spinor_time_reversal_residual` reports the normalized residual and
`validate_spinor_time_reversal` applies the threshold. This catches an
inconsistent spinor construction; it does not support an intentionally
time-reversal-breaking magnetic Hamiltonian.

For a spatial operation $g$, nfit can construct the double-group action

$$
D_{\rm spinor}(g)=D_{\rm orbital}(g)\otimes D_{1/2}(g).
$$

Spin is treated as an axial vector under improper operations.

## Scripted SOC

```python
from nfit import (
    set_tight_binding_soc_term,
    set_tight_binding_spin_treatment,
)

set_tight_binding_spin_treatment(component, "auto")
set_tight_binding_soc_term(
    component,
    "M1_d",
    enabled=True,
    prescription="auto",
    value=0.025,
    energy_unit="eV",
    lower=0.0,
    upper=0.10,
)
```

The resolved basis stays $N$ dimensional until the SOC term is enabled, then
becomes a $2N$ spinor basis.

## Choosing the representation

Use implicit spin for a spin-degenerate paramagnetic model without SOC. Use a
spinor basis when SOC affects the band structure, anisotropic spin matrix
elements, or neutron response. Choose a truncated projected shell only when
its omitted states are sufficiently remote; otherwise a complete shell or an
imported numerical model is safer.

The implemented Hubbard--Hund response currently assumes an implicit
spin-degenerate orbital model. General spinor interactions can be represented
with the user-defined matrix RPA dressing, but nfit does not yet construct a
spinor Hubbard--Hund vertex automatically.
