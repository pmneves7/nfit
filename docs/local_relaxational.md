# Local relaxational response

`local_relaxational` is the smallest magnetic-response model in nfit. It is
appropriate when intrinsic momentum dependence is unresolved or when a local
relaxor is a useful baseline.

## Response

For transferred energy $E$,

$$
\chi(E)=\frac{\chi_{\rm loc}}{1-iE/\Gamma},
\qquad
\chi''(E)=\chi_{\rm loc}\frac{\Gamma E}{E^2+\Gamma^2}.
$$

This convention gives $\chi(-E)=\chi(E)^*$ and positive $\chi''$ at positive
energy. The peak of $\chi''$ occurs at $E=\Gamma$.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `chi_loc` | static local susceptibility $\chi'(0)$ | meV$^{-1}$ |
| `gamma` | relaxation energy $\Gamma$ | meV |

The susceptibility has no intrinsic momentum dependence. Momentum dependence
in the magnetic neutron intensity appears only through the magnetic form
factor.

For comparison with bulk data, two fixed model settings specify the conversion
from spin susceptibility:

| API name | Meaning | Unit |
| --- | --- | --- |
| `bulk_g_factor` | Landé factor $g$ | dimensionless |
| `magnetic_ions_per_formula_unit` | equivalent magnetic ions represented by the response | ions/f.u. |

These settings do not change the intrinsic susceptibility. In a joint neutron
and bulk fit, `bulk_g_factor` should agree with the Landé factor in the neutron
dataset's spectral convention.

## Calculable data

The model calculates:

- single-crystal and powder inelastic neutron scattering;
- single-crystal and powder quasistatic elastic magnetic scattering;
- uniform bulk susceptibility; and
- magnetic moment or magnetization in linear response to an applied field.

For neutron scattering, inelastic calculations use $\chi''(E)$ and
quasistatic elastic calculations use $\chi'(0)=\chi_{\rm loc}$. Because the
response is independent of momentum, its uniform static limit is also

$$
\chi_{\rm uniform}=\chi_{\rm loc}.
$$

Bulk conversion uses the Landé factor and number of magnetic ions per formula
unit stated with the model, together with the sample normalization and units
stored by the dataset.

## Fitting and identifiability

Inelastic data can constrain both $\chi_{\rm loc}$ and $\Gamma$. Elastic and
bulk-susceptibility data constrain only $\chi_{\rm loc}$; they contain no
information about the relaxation energy. The model has no intrinsic
temperature dependence, so a temperature series requires per-dataset or
grouped values of $\chi_{\rm loc}$ unless another relation is imposed.

## Scripting

```python
from nfit import local_relaxational_susceptibility

chi = local_relaxational_susceptibility(
    energy_meV,
    chi_loc=2.0,
    gamma=3.0,
)
chipp = chi.imag
```

Use `ModelComponentSpec(type="local_relaxational", ...)` for fitting. Project
files, workflow scripts, fit-result export, reports, and model/residual
channels use the shared model machinery.
