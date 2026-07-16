# Heat-capacity data and models

Quantum Design PPMS Heat Capacity `.dat` files import as the **Heat capacity**
data type. Every exported column and the raw header are retained. `MASS` and
`MOLWGHT` header records seed the editable sample mass and formula-unit molar
mass.

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

The parameters are the Debye temperature $\Theta_D$ and oscillator count $n$.
Here $n$ is the number of atoms per formula unit, so the high-temperature limit
is $3nR$; use $n=7$ for LiV$_2$O$_4$. The model is registered only for
heat-capacity datasets.

**Low-temperature heat capacity** evaluates

$$
C(T)=\gamma T+\beta T^3,
\qquad \frac{C}{T}=\gamma+\beta T^2.
$$

Both models automatically return either $C$ or $C/T$ according to the selected
fit channel.

## Low-temperature analysis

The **Low-temperature C/T fit** analysis performs a weighted linear fit between
editable `Tmin` and `Tmax`. It reports $\gamma$, $\beta$, their covariance, and

$$
\Theta_D=\left(\frac{12\pi^4 nR}{5\beta}\right)^{1/3},
$$

using the analysis parameter **Atoms / formula unit**. Its diagnostic opens as
`C/T` versus `T^2` with the fitted line and fit-window boundaries.

## Magnetic thermodynamics roadmap

The present Heisenberg RPA response does not by itself define a unique
thermodynamic free energy. A simultaneous magnetic heat-capacity fit requires
an explicit conserving free-energy functional consistent with the closure:

1. evaluate the RPA/SCR/TAC free energy on the same full-zone grid and energy
   cutoff used by the closure;
2. include the closure's stationary or double-counting term rather than
   differentiating fitted temperature-dependent parameters as external inputs;
3. calculate $S_{mag}=-\partial F/\partial T$ and
   $C_{mag}=-T\partial^2F/\partial T^2$ with grid and temperature convergence
   checks;
4. enforce $S_{mag}(0)=0$ and the appropriate total entropy budget;
5. expose the checked result to the joint neutron, susceptibility, and
   heat-capacity evaluator.

Finite measured neutron windows constrain the response but cannot replace the
full-zone and full-energy thermodynamic integral.
