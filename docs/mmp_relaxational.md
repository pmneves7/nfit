# MMP relaxational response

`mmp_relaxational` describes one isotropic relaxational peak near a known
ordering vector. It is useful for nearly antiferromagnetic metals when the data
do not warrant a full crystal exchange model.

## Response

Let $\Delta\mathbf q=\mathbf q-\mathbf Q_0$ in Å$^{-1}$ and
$A(\mathbf q)=1+\xi^2|\Delta\mathbf q|^2$. The complex response is

$$
\chi(\mathbf q,E)=
\frac{\chi_{\rm pk}}{A(\mathbf q)-iE/E_{\rm sf}},
$$

and

$$
\chi''(\mathbf q,E)=
\chi_{\rm pk}
\frac{E/E_{\rm sf}}{A(\mathbf q)^2+(E/E_{\rm sf})^2}.
$$

The multiplication between $\chi_{\rm pk}$ and the final fraction is implicit.
The API name `omega_sf` stores the energy
$E_{\rm sf}=\hbar\omega_{\rm sf}$, not an angular frequency.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `chi_pk` | static susceptibility at $\mathbf Q_0$ | meV$^{-1}$ |
| `xi` | isotropic correlation length | Å |
| `omega_sf` | relaxation energy at $\mathbf Q_0$ | meV |
| `q0_h`, `q0_k`, `q0_l` | peak center $\mathbf Q_0$ | r.l.u. |

For comparison with bulk data, two fixed model settings specify the conversion
from spin susceptibility:

| API name | Meaning | Unit |
| --- | --- | --- |
| `bulk_g_factor` | Landé factor $g$ | dimensionless |
| `magnetic_ions_per_formula_unit` | equivalent magnetic ions represented by the response | ions/f.u. |

These settings do not alter the momentum-dependent response. In a joint
neutron and bulk fit, `bulk_g_factor` should agree with the Landé factor in the
neutron dataset's spectral convention.

The intrinsic momentum dependence is the Lorentzian peak about
$\mathbf Q_0$. The measured magnetic neutron intensity is additionally
modulated by the magnetic form factor. Because $\mathbf Q_0$ is
specified in reciprocal-lattice units while $\xi$ is in Å, the crystal lattice
is needed to form the dimensionless product
$\xi|\mathbf q-\mathbf Q_0|$.

## Calculable data

The model calculates:

- single-crystal inelastic neutron scattering;
- single-crystal quasistatic elastic magnetic scattering;
- uniform bulk susceptibility; and
- magnetic moment or magnetization in linear response to an applied field.

Inelastic calculations use the imaginary part above. Quasistatic elastic
calculations use

$$
\chi'(\mathbf q,0)=\frac{\chi_{\rm pk}}{A(\mathbf q)}.
$$

The bulk prediction is the extrapolated uniform limit

$$
\chi_{\rm uniform}
=\chi'(\mathbf 0,0)
=\frac{\chi_{\rm pk}}
{1+\xi^2|\mathbf Q_0|^2}.
$$

This comparison can expose a failure of the finite-$\mathbf Q$ model. The MMP
form is an expansion about $\mathbf Q_0$, however, and need not remain valid
all the way to $\mathbf q=0$. Bulk conversion uses the stated Landé factor and
number of magnetic ions per formula unit. The model's ability to calculate
this extrapolation does not require the user to include bulk data in a fit.

## Fitting and identifiability

Inelastic data can constrain the spin-fluctuation energy. Elastic and bulk
data do not constrain `omega_sf`; the bulk response alone generally constrains
only a combination of $\chi_{\rm pk}$, $\xi$, and $\mathbf Q_0$. Use
`generalized_paramagnon` when powder averaging, anisotropy, several peak
centers, a non-Lorentzian spatial profile, or propagating dynamics are needed.

## Scripting

```python
from nfit import mmp_susceptibility

chi = mmp_susceptibility(
    q_offset_sq_inv_angstrom2,
    energy_meV,
    chi_pk=3.0,
    xi=2.5,
    omega_sf=1.8,
)
```

Use `ModelComponentSpec(type="mmp_relaxational", ...)` for fitting. Project
files, workflow scripts, fit-result export, reports, and model/residual
channels use the shared model machinery.

## References

- A. J. Millis, H. Monien, and D. Pines, *Phys. Rev. B* **42**, 167
  (1990), [doi:10.1103/PhysRevB.42.167](https://doi.org/10.1103/PhysRevB.42.167).
- P. Monthoux and D. Pines, *Phys. Rev. B* **47**, 6069
  (1993), [doi:10.1103/PhysRevB.47.6069](https://doi.org/10.1103/PhysRevB.47.6069).
