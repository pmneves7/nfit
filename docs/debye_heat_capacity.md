# Debye phonon heat capacity

`debye_heat_capacity` describes the lattice heat capacity of an isotropic
elastic solid with a Debye cutoff.

## Response

The molar heat capacity is

$$
C_D(T)=9nR\left(\frac{T}{\Theta_D}\right)^3
\int_0^{\Theta_D/T}
\frac{x^4e^x}{(e^x-1)^2}\,dx .
$$

$T$ is absolute temperature, $\Theta_D$ is the Debye temperature, $n$ is the
number of atoms represented per formula unit, $R$ is the molar gas constant,
and $x$ is dimensionless. The high-temperature limit is $3nR$ and the
low-temperature response is proportional to $T^3$.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `debye_temperature` | Debye temperature $\Theta_D$ | K |
| `oscillator_count` | represented atoms per formula unit $n$ | dimensionless |

Both parameters are positive.

## Calculable data

The model calculates molar heat capacity $C(T)$ and, when that fit channel is
selected, $C(T)/T$. It applies only to `heat_capacity` datasets and does not
calculate magnetic susceptibility or neutron scattering.

## Fitting and identifiability

Data extending through a substantial fraction of $\Theta_D$ are needed to
separate $\Theta_D$ and $n$ reliably. In a restricted low-temperature window,
the model approaches a single $T^3$ coefficient, so the two parameters are
strongly correlated. Fix `oscillator_count` to the known atom count when the
model represents the complete lattice contribution.

## Scripting

```python
from nfit import debye_heat_capacity

heat_capacity = debye_heat_capacity(
    temperature_K,
    debye_temperature_K=300.0,
    oscillator_count=7.0,
)
```

The function returns mJ/(mol K). Use
`ModelComponentSpec(type="debye_heat_capacity", ...)` for fitting. Project
files, workflow scripts, fit-result export, reports, and model/residual
channels use the shared model machinery.

## References

- P. Debye, “Zur Theorie der spezifischen Wärmen,” *Ann. Phys.* **344**,
  789 (1912),
  [doi:10.1002/andp.19123441404](https://doi.org/10.1002/andp.19123441404).
