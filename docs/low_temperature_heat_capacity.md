# Low-temperature electronic and phonon heat capacity

`low_temperature_heat_capacity` is the leading normal-state expansion of the
heat capacity of a metal at temperatures well below the relevant phonon and
electronic scales.

## Response

This ansatz adds a Sommerfeld expansion for normal-state quasiparticles to
the leading three-dimensional Debye phonon term. $T$ is temperature in K
and $C$ molar heat capacity in mJ/(mol K), per mole of formula units.
As for the [Debye model](debye_heat_capacity.md), the thermodynamic expansion
is for fixed volume; using it for measured constant-pressure data assumes
that the difference is negligible in the fitted range.

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

For a smooth quasiparticle density of states $D_{\rm fu}(\mu)$ counting both
spins per formula unit per joule, the Sommerfeld approximation gives
$\gamma=(\pi^2/3)N_Ak_B^2D_{\rm fu}(\mu)$ in J/(mol K$^2$).
$\mu$ is chemical potential and $k_BT$ must be small compared with the scale
on which the DOS varies. A DOS per model cell must first be divided by its
formula-unit count and converted from states/meV to states/J. Multiply the
resulting $\gamma$ by 1000 for the fitted mJ units. This relation interprets
the fitted coefficient; the model does not automatically compute it from bands.
The [Debye ansatz](debye_heat_capacity.md) gives
$\beta=12\pi^4nR/(5\Theta_D^3)$, with $n$ atoms per formula unit,
$R=8314.46261815324$ mJ/(mol K), and Debye temperature $\Theta_D$ in K.

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
