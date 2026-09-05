# Heat-capacity data and models

Quantum Design PPMS Heat Capacity `.dat` files import as the **Heat capacity**
data type. Every exported column and the raw header are retained. `MASS` and
`MOLWGHT` header records seed the editable sample mass and formula-unit molar
mass.

Throughout this page, $C(T)$ is molar heat capacity and $T$ is absolute
temperature in K; molar ordinates use mJ/(mol K) per mole of formula units.

The dataset panel contains a **PPMS source units** selector matching the PPMS
choices: `uJ/K`, `uJ/(mol K)`, `mJ/(g K)`, `J/(g K)`, `cal/(g K)`,
`mJ/(mol K)`, `J/(mol K)`, `cal/(mol K)`, `J/(gat K)`, and `cal/(gat K)`.
The column header seeds the selector when possible. Sample-total `uJ/K` uses
both mass and molar mass; mass-specific units use molar mass; already molar
units need neither. Gram-atom units also require **Atoms / formula unit**.

Enabling molar normalization creates **Heat capacity** in `mJ/(mol K)` and
**C/T** in `mJ/(mol K^2)`, with propagated uncertainty. It also creates
**Temperature squared** as an alternate coordinate. Select it in the existing
viewer coordinate selector to draw `C/T` versus `T^2` without a dedicated plot
mode.

## Models

| Model | Main response | Typical use |
| --- | --- | --- |
| `debye_heat_capacity` | Debye phonon integral | Lattice heat capacity over a broad temperature range |
| `low_temperature_heat_capacity` | Linear electronic plus cubic phonon terms | Electronic and leading phonon terms at low temperature |

Both models calculate either $C$ or $C/T$ according to the selected fit
channel. Their equations, parameters, calculable data, fitting limitations,
scripts, and references are documented separately.

```{toctree}
:maxdepth: 1
:caption: Heat-capacity models

debye_heat_capacity
low_temperature_heat_capacity
```

## Low-temperature analysis

The **Low-temperature C/T fit** analysis performs a weighted linear fit between
editable `Tmin` and `Tmax`. It reports $\gamma$, $\beta$, their covariance, and

$$
\Theta_D=\left(\frac{12\pi^4 nR}{5\beta}\right)^{1/3},
$$

Here $\gamma$ is the electronic coefficient in mJ/(mol K$^2$), $\beta$ the
phonon coefficient in mJ/(mol K$^4$), $n$ the dimensionless number of atoms
per formula unit, and $R=N_Ak_B=8314.46261815324$ mJ/(mol K).
Thus $\Theta_D$ is the Debye temperature in K. This inference requires
$\beta>0$ and the [low-temperature Debye ansatz](low_temperature_heat_capacity.md).
The analysis takes $n$ from **Atoms / formula unit**. Its diagnostic opens as
`C/T` versus `T^2` with the fitted line and fit-window boundaries.

## Magnetic heat capacity

The Heisenberg RPA response does not define a unique thermodynamic free energy,
so nfit does not derive magnetic heat capacity from it. Such a calculation
requires a conserving free-energy functional and converged full-zone,
full-energy thermodynamics; a finite measured neutron window is insufficient.
