# Heat-capacity data and models

Quantum Design PPMS Heat Capacity `.dat` files import as the **Heat capacity**
data type. Every exported column and the raw header are retained. `MASS` and
`MOLWGHT` header records seed the editable sample mass and formula-unit molar
mass.

Throughout this page, $C(T)$ is molar heat capacity and $T$ is absolute
temperature.

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

## Fit components

**Debye phonon heat capacity** evaluates

$$
C_D(T)=9nR\left(\frac{T}{\Theta_D}\right)^3
\int_0^{\Theta_D/T}\frac{x^4e^x}{(e^x-1)^2}\,dx .
$$

Here $T$ is absolute temperature, $\Theta_D$ is the Debye temperature, $n$ is
the number of atoms per formula unit, $R$ is the molar gas constant, and $x$
is a dimensionless integration variable. The high-temperature limit is $3nR$.
Use $R$ in the same energy units as the heat-capacity ordinate. The model is
registered only for heat-capacity datasets.

**Low-temperature heat capacity** evaluates

$$
C(T)=\gamma T+\beta T^3,
\qquad \frac{C}{T}=\gamma+\beta T^2.
$$

Both models automatically return either $C$ or $C/T$ according to the selected
fit channel. $\gamma$ is the electronic coefficient in `mJ/(mol K^2)` and
$\beta$ is the cubic phonon coefficient in `mJ/(mol K^4)`.

## Low-temperature analysis

The **Low-temperature C/T fit** analysis performs a weighted linear fit between
editable `Tmin` and `Tmax`. It reports $\gamma$, $\beta$, their covariance, and

$$
\Theta_D=\left(\frac{12\pi^4 nR}{5\beta}\right)^{1/3},
$$

using the analysis parameter **Atoms / formula unit**. Its diagnostic opens as
`C/T` versus `T^2` with the fitted line and fit-window boundaries.

## Magnetic heat capacity

The Heisenberg RPA response does not define a unique thermodynamic free energy,
so nfit does not derive magnetic heat capacity from it. Such a calculation
requires a conserving free-energy functional and converged full-zone,
full-energy thermodynamics; a finite measured neutron window is insufficient.
