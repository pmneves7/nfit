# Curie--Weiss susceptibility

`curie_weiss` describes the molar susceptibility of a paramagnet over a
temperature interval where a Curie--Weiss law is appropriate. It is a bulk
model and does not predict neutron-scattering intensity.

## Response

The starting ansatz is Curie response to a molecular field proportional to
magnetization. Solving $M=(C_v/T)(H+\lambda_{\rm mf}M)$ gives
$M/H=C_v/(T-C_v\lambda_{\rm mf})$. In rationalized SI, $M,H$ are in A/m,
$C_v$ is a volume Curie coefficient in K, and $\lambda_{\rm mf}$ is a
dimensionless molecular-field coefficient. Hence
$\Theta_{\rm CW}=C_v\lambda_{\rm mf}$. The fitted molar law below uses $C$
on the dataset's mole basis; it does not fit $\lambda_{\rm mf}$ separately.

For $n_{\rm ion}$ identical rigid moments per formula unit with spin quantum
number $S$ and Landé factor $g$,
$C_{\rm SI}=\mu_0N_A n_{\rm ion}(g\mu_B)^2S(S+1)/(3k_B)$
in m$^3$ K/mol, with SI constants from
[Physics conventions](physics_conventions.md#constants-coordinates-and-mathematical-notation).
This interprets $C$ only for that local-moment ansatz. Numerically
$C_{\rm SI}=4\pi\times10^{-6}C_{\rm CGS}$ for $C_{\rm CGS}$ in cm$^3$ K/mol.
The inferred effective moment per formula unit is
$\mu_{\rm eff}=\sqrt{3k_BC_{\rm SI}/(\mu_0N_A)}$;
per-ion interpretation requires the ion count rather than silently assuming one.

The molar susceptibility is

$$
\chi_{\rm mol}(T)=\frac{C}{T-\Theta_{\rm CW}},
$$

where $T$ is absolute temperature, $C$ is the molar Curie constant, and
$\Theta_{\rm CW}$ is the Curie--Weiss temperature. Positive and negative
$\Theta_{\rm CW}$ conventionally indicate net ferromagnetic and
antiferromagnetic mean-field interactions, respectively, but do not determine
the ordering wavevector or microscopic exchange network.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `curie_constant` | molar Curie constant $C$ | cm$^3$ K/mol |
| `theta_CW` | Curie--Weiss temperature $\Theta_{\rm CW}$ | K |

The response is evaluated in cm$^3$/mol and converted to rationalized SI
m$^3$/mol when that is the dataset unit.

## Calculable data

The model calculates molar bulk susceptibility as a function of temperature
for a `magnetization` dataset whose selected fit channel is susceptibility. It
does not calculate magnetic moment versus field, neutron scattering, or heat
capacity.

## Fitting and identifiability

The fitted temperature interval must not cross $T=\Theta_{\rm CW}$. Both
$C$ and $\Theta_{\rm CW}$ become poorly determined when the interval is too
narrow or too far above $|\Theta_{\rm CW}|$. A temperature-independent
background susceptibility is not part of this model; add a constant component
when such a term is physically justified.

The separate Curie--Weiss analysis provides fit-window selection and
diagnostic $\chi$ and $1/\chi$ datasets. The registered model component is
useful when Curie--Weiss susceptibility is part of a simultaneous workspace
fit.

## Scripting

```python
from nfit import curie_weiss_susceptibility

chi_cm3_per_mol = curie_weiss_susceptibility(
    temperature_K,
    curie_constant_cm3_K_per_mol=0.42,
    theta_CW_K=-18.0,
)
```

Use `ModelComponentSpec(type="curie_weiss", ...)` for fitting. Project files,
workflow scripts, fit-result export, reports, and model/residual channels use
the shared model machinery.

## References

- P. Curie, *Propriétés magnétiques des corps à diverses températures*
  (Gauthier-Villars, 1895).
- P. Weiss, “L'hypothèse du champ moléculaire et la propriété
  ferromagnétique,” *J. Phys. Théor. Appl.* **6**, 661 (1907),
  [doi:10.1051/jphystap:019070060066100](https://doi.org/10.1051/jphystap:019070060066100).
