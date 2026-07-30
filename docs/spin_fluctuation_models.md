# Spin-fluctuation models

nfit provides eight magnetic-response models. Start with the least structured
response that resolves the features in the data. Electronic Hamiltonians are
documented separately under
[Electronic-structure models](electronic_structure_models.md).

| Model | Main response | Typical use | Calculable data |
| --- | --- | --- | --- |
| `local_relaxational` | $\chi=\chi_{\rm loc}/(1-iE/\Gamma)$ | Local or momentum-independent fluctuations | Single-crystal and powder neutron scattering; bulk linear response |
| `mmp_relaxational` | $\chi=\chi_{\rm pk}/(1+\xi^2q^2-iE/E_{\rm sf})$ | One isotropic peak in a nearly antiferromagnetic metal | Single-crystal neutron scattering; extrapolated bulk linear response |
| `generalized_paramagnon` | $\chi=(\chi_{\rm pk}/A)/(1-a_EE^2/A-iE/[\Gamma_0A^z])$ | Anisotropic peaks, critical slowing down, and damped propagating modes | Single-crystal and powder neutron scattering; extrapolated bulk linear response |
| `heisenberg_rpa` | $\boldsymbol\chi=[\mathbb 1-\chi_0J(\mathbf Q)]^{-1}\chi_0$ | Dispersive fluctuations constrained by a crystal and exchange network | Single-crystal and powder neutron scattering; bulk linear response |
| `lindhard` | $\chi^0=-\sum_{\mathbf k,n,m}(f_{n\mathbf k}-f_{m,\mathbf k+\mathbf q})M_AM_B^*/(E+\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}+i\eta)$ | Bare particle--hole response of a linked tight-binding model | Single-crystal and powder neutron scattering; bulk linear response |
| `stoner_rpa` | $\boldsymbol\chi=[\mathbb 1-\boldsymbol\chi^0 I]^{-1}\boldsymbol\chi^0$ | Minimal isotropic enhancement of a band response | Single-crystal and powder neutron scattering; bulk linear response |
| `matrix_rpa` | $\boldsymbol\chi=[\mathbb 1-\boldsymbol\chi^0 g\mathbf V]^{-1}\boldsymbol\chi^0$ | User-supplied anisotropic spin interaction | Single-crystal and powder neutron scattering; bulk linear response |
| `hubbard_hund_rpa` | $\boldsymbol\chi=[\mathbb 1-\boldsymbol\chi^0\boldsymbol\Gamma_{U,U',J_H,J_{\rm pair}}]^{-1}\boldsymbol\chi^0$ | Local multiorbital correlations on labelled shells | Single-crystal and powder neutron scattering; bulk linear response |

```{toctree}
:maxdepth: 1
:caption: Model details

local_relaxational
mmp_relaxational
generalized_paramagnon
heisenberg_rpa
lindhard
stoner_rpa
matrix_rpa
hubbard_hund_rpa
```

For inelastic data, each magnetic-response kernel supplies the dissipative
spin susceptibility $\chi''_s(\mathbf Q,E)$ in `spin^2/meV` in its documented
site, formula-unit, or primitive-cell normalization.
The dataset convention determines
whether nfit compares that response directly with $\chi''$ data or converts it
to a neutron cross section. See [Physics conventions](physics_conventions.md)
for the Bose factor, polarization, form factor, Landé factor, and absolute
normalization.

Backgrounds are separate additive model components.

The [local relaxational](local_relaxational.md),
[MMP relaxational](mmp_relaxational.md),
[generalized paramagnon](generalized_paramagnon.md), and
[Heisenberg RPA](heisenberg_rpa.md),
[bare Lindhard](lindhard.md), [scalar Stoner](stoner_rpa.md),
[matrix RPA](matrix_rpa.md), and
[Hubbard--Hund RPA](hubbard_hund_rpa.md) pages define their parameters, complex
susceptibilities, dissipative responses, elastic limits, scripts, and
references.

Bulk comparison is optional. For models constructed near finite-momentum
peaks, the $\mathbf Q=0$ result is an extrapolation whose failure can itself be
a useful model diagnostic.

## Units

| Quantity | Unit |
| --- | --- |
| $E$, $\Gamma$, $\Gamma_0$, $E_{\rm sf}$, $E_0$, $J_i$ | meV |
| $\chi_{\rm loc}$, $\chi_{\rm pk}$, $\chi_0$ | meV$^{-1}$ in the model normalization |
| $\xi$ | Å |
| $a_E=1/E_0^2$ | meV$^{-2}$ |
| $H,K,L$ | r.l.u. |
| $T$ | K |
