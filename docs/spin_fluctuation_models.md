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

## Units

| Quantity | Unit |
| --- | --- |
| $E$, $\Gamma$, $\Gamma_0$, $E_{\rm sf}$, $E_0$, $J_i$ | meV |
| $\chi_{\rm loc}$, $\chi_{\rm pk}$, $\chi_0$ | meV$^{-1}$ in the model normalization |
| $\xi$ | Å |
| $a_E=1/E_0^2$ | meV$^{-2}$ |
| $H,K,L$ | r.l.u. |
| $T$ | K |
