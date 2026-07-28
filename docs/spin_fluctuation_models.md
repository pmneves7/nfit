# Spin-fluctuation models

nfit provides three magnetic-response models:

| Model | Momentum dependence | Typical use |
| --- | --- | --- |
| `local_relaxational` | form factor only | Local or nearly momentum-independent fluctuations |
| `mmp_relaxational` | one peak with a correlation length | Nearly antiferromagnetic metals near a known ordering vector |
| `heisenberg_rpa` | exchange matrix on a crystal lattice | Dispersive fluctuations constrained by crystal symmetry |

```{toctree}
:maxdepth: 1
:caption: Model details

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

## Local relaxational model

The local model is

$$
\chi''(E)=\chi_{\rm loc}\frac{\Gamma E}{E^2+\Gamma^2}.
$$

$E$ is transferred energy, $\Gamma$ is the relaxation energy, and
$\chi_{\rm loc}$ is the static local susceptibility. The response peaks at
$E=\Gamma$ and has no momentum dependence beyond the magnetic form factor. It
therefore applies to both single-crystal and powder data.

For elastic data, nfit uses the quasistatic result
$S(\mathbf Q)\simeq k_BT\chi'(\mathbf Q,0)$. The local model therefore has no
momentum dependence beyond the form factor, and elastic-only data do not
constrain $\Gamma$.

## MMP relaxational model

The Millis--Monien--Pines form describes a relaxational peak centered at
$\mathbf Q_0$:

$$
\chi''(\mathbf q,E)=
\chi_{\rm pk}
\frac{E/E_{\rm sf}}
{\left[1+\xi^2|\mathbf q-\mathbf Q_0|^2\right]^2+(E/E_{\rm sf})^2}.
$$

Here $\mathbf q$ is physical momentum transfer, $\mathbf Q_0$ is the peak
position, $\chi_{\rm pk}$ is the static peak susceptibility, $\xi$ is the
correlation length in Å, momentum distance is in Å$^{-1}$, and $E_{\rm sf}$ is
the spin-fluctuation energy in meV. The API name `omega_sf` stores the energy
$E_{\rm sf}=\hbar\omega_{\rm sf}$, not an angular frequency.

This model is useful when the peak position is known and the data do not
require a crystallographic exchange network.

The same static peak form is available for single-crystal elastic data.
Elastic-only data do not constrain `omega_sf`.

## Heisenberg RPA model

`heisenberg_rpa` dresses a local relaxational response with real-space
exchange:

$$
\chi(\mathbf Q,E)=
\left[\mathbb 1-\chi_0(E)J(\mathbf Q)\right]^{-1}\chi_0(E),
\qquad
\chi_0(E)=\frac{\chi_0}{1-iE/\Gamma_0}.
$$

Exchange parameters belong to symmetry-distinct bond orbits generated from the
crystal structure. The model supports multiple magnetic sites, powder
averaging, anisotropic interactions, dipoles, applied fields,
self-consistency closures, and joint magnetometry fits.

For single-crystal and powder elastic data, the model evaluates
$\chi'(\mathbf Q,0)$ directly and applies the quasistatic cross section.
Powder evaluation requires lattice parameters and performs the same
orientation average as the inelastic model. Elastic-only data do not constrain
$\Gamma_0$.

The phase convention, mode weights, tensor extension, closures, and fitting
parameters are documented separately in
[Heisenberg RPA](heisenberg_rpa.md).

## Magnetic form factor

nfit uses the standard analytic approximation

$$
f(s)=Ae^{-as^2}+Be^{-bs^2}+Ce^{-cs^2}+D,
\qquad s=\frac{|\mathbf Q|}{4\pi},
$$

with coefficients for common $3d$, $4d$, $4f$, and $5f$ ions. Select a
tabulated ion in the model editor or provide custom
$(A,a,B,b,C,c,D)$ coefficients.

$f$ and $A,B,C,D$ are dimensionless; $s=|\mathbf Q|/(4\pi)$ has units
Å$^{-1}$, so $a,b,c$ have units Å$^2$.

Evaluating $|\mathbf Q|$ requires lattice or UB metadata when coordinates are
in reciprocal-lattice units. Use the same ion when a dataset conversion and a
model both apply or remove a form factor.

## Temperature series

Temperature is read from each dataset. A model that requires detailed balance
cannot evaluate a dataset without a valid temperature.

Experimental calibration belongs to the dataset, not the response model. Keep
the dataset scale fixed at 1 for normalized data. For unnormalized data it may
be fitted independently or shared by datasets in one dataset group.

For an unconstrained temperature series, keep structural parameters and
exchange constants global while sharing $\chi_0$ and $\Gamma_0$ per dataset or
temperature group. The fitted trajectories can then be compared with the
self-consistent alternatives described in
[Theory: sum rules and self-consistency](theory_notes.md).

## Units

| Quantity | Unit |
| --- | --- |
| $E$, $\Gamma$, $\Gamma_0$, $E_{\rm sf}$, $J_i$ | meV |
| $\chi_{\rm loc}$, $\chi_{\rm pk}$, $\chi_0$ | meV$^{-1}$ in the model normalization |
| $\xi$ | Å |
| $H,K,L$ | r.l.u. |
| $T$ | K |

## References

- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*
  (Springer, 1985).
- A. J. Millis, H. Monien, and D. Pines, *Phys. Rev. B* **42**, 167
  (1990), [doi:10.1103/PhysRevB.42.167](https://doi.org/10.1103/PhysRevB.42.167).
- P. Monthoux and D. Pines, *Phys. Rev. B* **47**, 6069
  (1993), [doi:10.1103/PhysRevB.47.6069](https://doi.org/10.1103/PhysRevB.47.6069).
- P. J. Brown, “Magnetic form factors,” *International Tables for
  Crystallography*, Vol. C, §4.4.5; see the
  [ILL tables](https://www.ill.eu/sites/ccsl/ffacts/).
