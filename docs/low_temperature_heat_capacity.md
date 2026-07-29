# Low-temperature electronic and phonon heat capacity

`low_temperature_heat_capacity` is the leading normal-state expansion of the
heat capacity of a metal at temperatures well below the relevant phonon and
electronic scales.

## Response

The molar heat capacity is

$$
C(T)=\gamma T+\beta T^3,
\qquad
\frac{C(T)}{T}=\gamma+\beta T^2.
$$

The linear term is the electronic contribution and the cubic term is the
leading Debye phonon contribution.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `sommerfeld_gamma` | electronic Sommerfeld coefficient $\gamma$ | mJ/(mol K$^2$) |
| `debye_beta` | leading phonon coefficient $\beta$ | mJ/(mol K$^4$) |

Both coefficients are nonnegative.

## Calculable data

The model calculates molar heat capacity $C(T)$ and, when selected, $C(T)/T$.
It applies only to `heat_capacity` datasets and does not calculate magnetic
susceptibility or neutron scattering.

## Fitting and identifiability

A plot of $C/T$ versus $T^2$ is linear within the validity range. Curvature
indicates that higher-order phonon terms, magnetic contributions,
superconductivity, or another low-energy scale may be important. The fitted
$\beta$ may be converted to a Debye temperature only after specifying the
number of atoms represented per formula unit.

## Scripting

```python
from nfit import low_temperature_heat_capacity

heat_capacity = low_temperature_heat_capacity(
    temperature_K,
    sommerfeld_mJ_mol_K2=400.0,
    beta_mJ_mol_K4=0.08,
)
```

The function returns mJ/(mol K). Use
`ModelComponentSpec(type="low_temperature_heat_capacity", ...)` for fitting.
Project files, workflow scripts, fit-result export, reports, and model/residual
channels use the shared model machinery.

## References

- A. Sommerfeld, “Zur Elektronentheorie der Metalle auf Grund der
  Fermischen Statistik,” *Z. Phys.* **47**, 1 (1928),
  [doi:10.1007/BF01391052](https://doi.org/10.1007/BF01391052).
- P. Debye, “Zur Theorie der spezifischen Wärmen,” *Ann. Phys.* **344**,
  789 (1912),
  [doi:10.1002/andp.19123441404](https://doi.org/10.1002/andp.19123441404).
