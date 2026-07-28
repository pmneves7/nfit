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

An optional magnetic form factor is fixed model configuration. Momentum
conversion requires reciprocal-basis or lattice metadata.

## Elastic and inelastic data

Inelastic fits use the imaginary part above. Single-crystal elastic fits use

$$
\chi'(\mathbf q,0)=\frac{\chi_{\rm pk}}{A(\mathbf q)}.
$$

Elastic-only data cannot constrain `omega_sf`. Use
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

## References

- A. J. Millis, H. Monien, and D. Pines, *Phys. Rev. B* **42**, 167
  (1990), [doi:10.1103/PhysRevB.42.167](https://doi.org/10.1103/PhysRevB.42.167).
- P. Monthoux and D. Pines, *Phys. Rev. B* **47**, 6069
  (1993), [doi:10.1103/PhysRevB.47.6069](https://doi.org/10.1103/PhysRevB.47.6069).
