# Spin-fluctuation models

nfit provides four magnetic-response models. Start with the least structured
model that resolves the features in the data.

| Model | Main response | Typical use |
| --- | --- | --- |
| `local_relaxational` | $\chi=\chi_{\rm loc}/(1-iE/\Gamma)$ | Local or momentum-independent fluctuations |
| `mmp_relaxational` | $\chi=\chi_{\rm pk}/(1+\xi^2q^2-iE/E_{\rm sf})$ | One isotropic peak in a nearly antiferromagnetic metal |
| `generalized_paramagnon` | $\chi=(\chi_{\rm pk}/A)/(1-a_EE^2/A-iE/[\Gamma_0A^z])$ | Anisotropic peaks, critical slowing down, and damped propagating modes |
| `heisenberg_rpa` | $\boldsymbol\chi=[\mathbb 1-\chi_0J(\mathbf Q)]^{-1}\chi_0$ | Dispersive fluctuations constrained by a crystal and exchange network |

```{toctree}
:maxdepth: 1
:caption: Model details

local_relaxational
mmp_relaxational
generalized_paramagnon
heisenberg_rpa
```

For inelastic data, each kernel returns one Cartesian component of the
dissipative spin susceptibility $\chi''_s(\mathbf Q,E)$ in `spin^2/meV`.
The dataset convention determines
whether nfit compares that response directly with $\chi''$ data or converts it
to a neutron cross section. See [Physics conventions](physics_conventions.md)
for the Bose factor, polarization, form factor, Landé factor, and absolute
normalization.

Backgrounds are separate additive model components.

The [local relaxational](local_relaxational.md),
[MMP relaxational](mmp_relaxational.md),
[generalized paramagnon](generalized_paramagnon.md), and
[Heisenberg RPA](heisenberg_rpa.md) pages define their parameters, complex
susceptibilities, dissipative responses, elastic limits, scripts, and
references.

## Magnetic form factor

nfit uses the standard analytic approximation

$$
f(s)=Ae^{-as^2}+Be^{-bs^2}+Ce^{-cs^2}+D,
\qquad s=\frac{|\mathbf Q|}{4\pi},
$$

with coefficients for common $3d$, $4d$, $4f$, and $5f$ ions. Select a
tabulated ion in the model editor or provide custom
$(A,a,B,b,C,c,D)$ coefficients.

$f$ and $A,B,C,D$ are dimensionless; $s=|\mathbf Q|/(4\pi)$ has units
Å$^{-1}$, so $a,b,c$ have units Å$^2$.

Evaluating $|\mathbf Q|$ requires lattice or UB metadata when coordinates are
in reciprocal-lattice units. Use the same ion when a dataset conversion and a
model both apply or remove a form factor.

## Temperature series

Temperature is read from each dataset. A model that requires detailed balance
cannot evaluate a dataset without a valid temperature.

Experimental calibration belongs to the dataset, not the response model. Keep
the dataset scale fixed at 1 for normalized data. For unnormalized data it may
be fitted independently or shared by datasets in one dataset group.

For an unconstrained temperature series, keep structural parameters and
exchange constants global while sharing $\chi_0$ and $\Gamma_0$ per dataset or
temperature group. The fitted trajectories can then be compared with the
self-consistent alternatives described in
[Theory: sum rules and self-consistency](theory_notes.md).

## Units

| Quantity | Unit |
| --- | --- |
| $E$, $\Gamma$, $\Gamma_0$, $E_{\rm sf}$, $E_0$, $J_i$ | meV |
| $\chi_{\rm loc}$, $\chi_{\rm pk}$, $\chi_0$ | meV$^{-1}$ in the model normalization |
| $\xi$ | Å |
| $a_E=1/E_0^2$ | meV$^{-2}$ |
| $H,K,L$ | r.l.u. |
| $T$ | K |

## References

- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*
  (Springer, 1985).
- A. J. Millis, H. Monien, and D. Pines, *Phys. Rev. B* **42**, 167
  (1990), [doi:10.1103/PhysRevB.42.167](https://doi.org/10.1103/PhysRevB.42.167).
- P. Monthoux and D. Pines, *Phys. Rev. B* **47**, 6069
  (1993), [doi:10.1103/PhysRevB.47.6069](https://doi.org/10.1103/PhysRevB.47.6069).
- P. J. Brown, “Magnetic form factors,” *International Tables for
  Crystallography*, Vol. C, §4.4.5; see the
  [ILL tables](https://www.ill.eu/sites/ccsl/ffacts/).
