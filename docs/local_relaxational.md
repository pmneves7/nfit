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

The model has no intrinsic momentum dependence. Its optional magnetic form
factor is fixed configuration, while experimental normalization is owned by
the dataset.

## Elastic and inelastic data

Inelastic fits use $\chi''(E)$ and the dataset’s declared spectral convention.
Elastic fits use the quasistatic response
$\chi'(0)=\chi_{\rm loc}$ before the form factor and cross-section conversion.
Elastic-only data therefore cannot constrain `gamma`.

The model supports single-crystal and powder inelastic or elastic datasets.

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

Use `ModelComponentSpec(type="local_relaxational", ...)` for fitting and
workflow export.
