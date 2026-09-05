# Relaxational and inertial Heisenberg RPA

The `heisenberg_rpa` model couples local spins through a crystallographic
exchange network. A continuous inertia parameter selects either the historical
relaxational response or an exchange-paramagnon response with damped
propagating modes. Both limits share one crystal, exchange network, static
susceptibility, stability criterion, and fitting workflow.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `chi0` | static single-site susceptibility $\chi_0$ | meV$^{-1}$ |
| `gamma0` | bare single-site relaxation energy $\Gamma_0$ | meV |
| `inverse_mode_energy_sq` | local inertial coefficient $a_E=1/E_0^2$; zero is exactly relaxational | meV$^{-2}$ |
| exchange-orbit label, such as `J1` | exchange assigned to every bond in that orbit | meV |

The crystal and exchange editor determines the orbit labels. Tensor exchange,
single-ion anisotropy, dipolar interactions, Zeeman coupling, and
self-consistency closures add the parameters described in their sections
below.

## Scalar model

The exchange ansatz is
$H_{\rm ex}=-\tfrac12\sum_{ij}J_{ij}\mathbf S_i\cdot\mathbf S_j$,
where $i,j$ index magnetic sites including lattice translations,
$\mathbf S_i$ is the dimensionless spin, and $J_{ij}$ is an energy in meV.
The ordered-pair sum counts each bond twice, hence the half factor; positive
scalar $J_{ij}$ favors parallel spins. A local causal response is postulated,
then dressed by the induced exchange field:
$\boldsymbol\chi=\chi_0(E)\mathbb1+\chi_0(E)J\boldsymbol\chi$.
This random-phase (molecular-field feedback) approximation fixes the response
below; it is not a spin-wave expansion about an ordered ground state.

For $N$ magnetic sites in the chosen crystallographic cell,

$$
\chi_0(E)=\frac{\chi_0}{1-a_EE^2-iE/\Gamma_0},
$$

and

$$
\boldsymbol\chi(\mathbf Q,E)=
\left[\mathbb 1-\chi_0(E)J(\mathbf Q)\right]^{-1}\chi_0(E)\mathbb1.
$$

$\chi_0$ is the static single-site susceptibility in meV$^{-1}$,
$\Gamma_0$ is its relaxation energy in meV, and $a_E$ is nonnegative in
meV$^{-2}$. Setting $a_E=0$ recovers the original relaxational model. For
$a_E>0$, the bare local natural energy is $E_0=1/\sqrt{a_E}$. Each fitted
exchange $J_i$ is in meV, so $J_i\chi_0$ is dimensionless. $E$ is transferred energy,
$\mathbf Q$ is momentum transfer, $N$ is the number of magnetic sites, and
$\mathbb 1$ is the $N\times N$ identity matrix. $\boldsymbol\chi$ is an $N\times N$ sublattice matrix for one Cartesian
spin component, in meV$^{-1}$. The measured scalar below is
$\chi=N^{-1}\phi^\dagger\boldsymbol\chi\phi$, where $\phi_a=1$ for every
site. “Scalar model” means isotropic in spin space; it does not discard
sublattice indices.

### Exchange phase convention

A bond connects site $a$ at fractional position $\mathbf r_a$ to site $b$ in
the cell displaced by integer vector $\mathbf n$. nfit uses

$$
J_{ab}(\mathbf Q)=
\sum_{(a,b,\mathbf n)}
J_{\rm bond}
\exp\!\left[
2\pi i\,\mathbf Q\cdot
(\mathbf r_b+\mathbf n-\mathbf r_a)
\right].
$$

This extended-zone convention includes the sublattice offset in
$J(\mathbf Q)$. Each physical bond is stored once and its Hermitian conjugate
is added automatically. Here $\mathbf Q=(H,K,L)$ is in reciprocal-lattice
units, $\mathbf r_a$ and $\mathbf r_b$ are fractional site coordinates,
$\mathbf n$ is an integer lattice translation, and $J_{\rm bond}$ is the
exchange energy assigned to that bond orbit.

### Mode response

Diagonalize
$J(\mathbf Q)=U\Lambda U^\dagger$, where the columns of the unitary matrix
$U$ are eigenvectors and the diagonal entries $\lambda_\nu$ of $\Lambda$ are
exchange eigenvalues in meV. Define
$\delta_\nu(\mathbf Q)=1-\lambda_\nu(\mathbf Q)\chi_0$. Mode $\nu$ has

$$
\chi_{\mathbf Q\nu}(E)=
\frac{\chi_0}{\delta_\nu(\mathbf Q)-a_EE^2-iE/\Gamma_0},
$$

and contributes

$$
\chi''(\mathbf Q,E)=
\sum_\nu w_\nu(\mathbf Q)
\frac{\chi_0E/\Gamma_0}
{[\delta_\nu(\mathbf Q)-a_EE^2]^2+(E/\Gamma_0)^2}.
$$

Its static susceptibility is $\chi_0/\delta_\nu$. In the relaxational limit,
$\Gamma_\nu=\Gamma_0\delta_\nu$ and the last equation reduces to
$\chi_0\Gamma_0E/(E^2+\Gamma_\nu^2)$. With $a_E>0$, the undamped
exchange-mode energy is

$$
E_\nu(\mathbf Q)=\sqrt{\delta_\nu(\mathbf Q)/a_E}.
$$

Exchange therefore softens the same static denominator into either critical
relaxation or a propagating paramagnon, without introducing a second model
type or a separate interaction dressing.

The dimensionless neutron weight is

$$
w_\nu(\mathbf Q)=
\frac{1}{N}\left|\sum_a U_{a\nu}(\mathbf Q)\right|^2,
\qquad
\sum_\nu w_\nu=1.
$$

The sublattice sum is uniform because the site offsets already appear in the
exchange phase. Adding site phases again would double-count them. This pairing
makes the observable invariant under an origin shift or an equivalent enlarged
cell.

### Stability and limiting behavior

The paramagnetic model requires

$$
1-\lambda_\nu(\mathbf Q)\chi_0>0
$$

for every mode. Approaching zero produces a divergent static susceptibility
and critical slowing. Invalid trial parameters receive a large finite
least-squares penalty.

Useful checks are:

- with every $J_i=0$, the model reduces to the selected local relaxational or
  inertial response;
- $\chi''(\mathbf Q,-E)=-\chi''(\mathbf Q,E)$;
- at small positive energy, the response is enhanced by
  $[1-\lambda_\nu\chi_0]^{-2}$; and
- for $a_E=0$, the leading high-energy tail is
  $\chi_0\Gamma_0/E$, independent of exchange.

Positive $J$ favors the wavevector where the largest eigenvalue of
$J(\mathbf Q)$ is maximal under this convention.

## Calculable data

The model calculates:

- single-crystal and powder inelastic neutron scattering;
- single-crystal and powder quasistatic elastic magnetic scattering;
- uniform bulk susceptibility; and
- magnetic moment or magnetization, including field-dependent tensor and
  self-consistent responses when those extensions are enabled.

All channels are evaluations of the same response parameters and may be
fitted jointly. Inelastic data constrain the relaxation scale; elastic and
bulk data evaluate static limits and cannot determine $\Gamma_0$ by
themselves.

## Crystal and exchange network

The GUI keeps the workflow in three tabs: **Structure and exchange**,
**Response and fit**, and **Advanced**. The first tab stores lattice
parameters, space group, magnetic sites, and a bond cutoff.
**Generate symmetry orbits** expands the magnetic sites and groups
symmetry-equivalent bonds into parameters `J1`, `J2`, and so on.
Symmetry-inequivalent orbits at the same distance receive suffixes such as
`J3a` and `J3b`.

Regenerating the network preserves values for orbit labels that remain present.
The complete site and bond configuration is serialized in the project and is
available through the crystal API.

### Primitive-cell reduction

For a centered conventional cell, nfit detects translations that preserve both
the magnetic sites and every labeled bond orbit. It evaluates one
representative site per translation class while leaving the user-visible cell,
coordinates, bond table, and fitted parameters unchanged.

The reduction is exact for the scalar model and for translationally compatible
anisotropic exchange and single-ion anisotropy. It is disabled when the
network does not possess the required translation symmetry or when dipolar
evaluation requires the full Bravais-lattice description.

## Powder average

Powder data supply $Q=|\mathbf Q|$ rather than a direction. nfit evaluates the
single-crystal response on deterministic approximately equal-area sphere
directions, converts each Cartesian momentum to model HKL, and averages the
result.

The default uses 50 directions. The **Advanced** tab exposes
`powder_orientations`, which may be set to another value of at least 6.
Increasing it increases cost proportionally. A lattice or UB matrix is
required.

The orientation count is a numerical approximation like an integration mesh or
a lifetime broadening, so it is converged the same way — see
[Powder angular convergence](powder_convergence.md).

## Tensor interactions

Anisotropic models promote the interaction to a
$3N\times3N$ Hermitian matrix:

$$
\mathbb J(\mathbf Q)=\sum_p\theta_pP_p(\mathbf Q).
$$

The composite row/column indices are $(a,\alpha)$ for site $a=1,\ldots,N$
and Cartesian spin component $\alpha=x,y,z$. The scalar ansatz becomes
$\boldsymbol\chi=[\chi_0(E)^{-1}I_{3N}-\mathbb J]^{-1}$ at zero field;
$\mathbb J$ has units meV and $\boldsymbol\chi$ meV$^{-1}$.

$\theta_p$ is the fitted strength of tensor parameter $p$, and
$P_p(\mathbf Q)$ is its symmetry-generated structure matrix. Exchange and onsite coefficients
have units meV and dimensionless structure matrices; the dipole coefficient
has units meV Å$^3$ and its structure matrix has units Å$^{-3}$.

Available terms are:

- symmetry-allowed symmetric and Dzyaloshinskii--Moriya exchange on each bond
  orbit;
- quadratic single-ion anisotropy allowed by each site point group;
- an Ewald-summed dipole--dipole tensor; and
- a Zeeman term from the dataset magnetic field.

With no tensor terms, nfit uses the scalar path unchanged. At zero field, one
Hermitian eigendecomposition of $\mathbb J(\mathbf Q)$ serves every energy at a
given momentum. A field makes the local propagator gyrotropic and requires a
linear solve at every fitted point.

The unpolarized observable contracts the dissipative tensor with

$$
\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta.
$$

$\alpha,\beta$ label Cartesian components, $\delta_{\alpha\beta}$ is the
Kronecker delta, and $\hat{\mathbf Q}$ is the unit momentum-transfer vector.
Its isotropic limit is exactly $2\chi''_s$, matching the scalar one-component
convention. See [Physics conventions](physics_conventions.md#tensor-anisotropic-interactions)
for frames and signs.

### Zeeman response

Let $\hat{\mathbf b}$ be the Cartesian unit field direction and
$B\ge0$ its magnitude in tesla. With the electron moment
$\boldsymbol\mu=-g\mu_B\mathbf S$, the Zeeman Hamiltonian is
$H_Z=+E_L S_{\hat b}$, with Larmor energy
$E_L=g\mu_BB$ in meV, dimensionless $g$, and
$\mu_B=0.05788381$ meV/T. Choose orthonormal transverse axes
$\hat{\mathbf x},\hat{\mathbf y}$ with
$\hat{\mathbf x}\times\hat{\mathbf y}=\hat{\mathbf b}$.

The ansatz relaxes and precesses the departure from the instantaneous
field-induced equilibrium spin. In circular channels, it is
$\hbar\dot s_\pm=-(\Gamma_\perp\pm iE_L)(s_\pm-\chi_\perp h_\pm)$,
where $s_\pm$ and $h_\pm$ are coefficients along
$(\hat{\mathbf x}\pm i\hat{\mathbf y})/\sqrt2$ of the induced spin and its
conjugate meV field. For time dependence $e^{-iEt/\hbar}$ this gives

$$
\chi_\parallel(E)=\frac{\chi_0}{1-iE/\Gamma_0},\qquad
\chi_\pm(E)=\chi_\perp
\frac{\Gamma_\perp\pm iE_L}{\Gamma_\perp-i(E\mp E_L)}.
$$

$\chi_\perp=$ `chi_perp_ratio` $\times\chi_0$ is the static transverse
susceptibility in meV$^{-1}$ and
$\Gamma_\perp=$ `gamma_perp_ratio` $\times\Gamma_0$ its relaxation energy
in meV. The Cartesian block has
$\chi_{xx}=\chi_{yy}=(\chi_++\chi_-)/2$,
$\chi_{xy}=-\chi_{yx}=-i(\chi_+-\chi_-)/2$, and
$\chi_{zz}=\chi_\parallel$. It is rotated into the crystal frame before RPA.

Both transverse static responses equal $\chi_\perp$, and their absorptive
parts are $\chi_\perp\Gamma_\perp E/[\Gamma_\perp^2+(E\mp E_L)^2]$,
nonnegative at positive energy. The numerator is essential: shifting only
the denominator would violate the equilibrium static limit and allow negative
absorption. The positive circular absorption peaks at
$\sqrt{E_L^2+\Gamma_\perp^2}$; the full transverse Cartesian response sums
both circular channels and need not peak at exactly that value.
At zero field and unit ratios the model returns the local scalar relaxor.

This is a phenomenological precessing relaxor with adjustable static
susceptibilities, not a calculation of saturation or thermal level populations.
Finite-field spectra and closures fitted with versions before 0.80.5 should
be refitted: those versions used shifted relaxors without the equilibrium
numerator. A field direction and magnitude must be defined in the dataset.

## Self-consistency closures

Bare RPA fits $\chi_0$ and $\Gamma_0$ directly. A closure instead solves an
internal local response from a full-zone moment constraint:

- **Onsager** solves a reaction field $\lambda(T)$ so the moment equals a fixed
  or fitted target.
- **SCR** uses
  $\chi_{0,\rm eff}^{-1}(T)=\chi_0^{-1}+u\langle m^2\rangle(T)$.
- **TAC** conserves the zero-point plus thermal amplitude.

$\lambda$ and the mode-coupling coefficient $u$ have energy units, and
$\langle m^2\rangle$ is the dimensionless component-summed fluctuating spin
amplitude.

The closure configuration selects its energy cutoff. The **Advanced** tab
shows the $N\times N\times N$ Brillouin-zone grid only when a closure is
active, because bare RPA evaluates requested data points directly without a
full-zone grid. With both a closure and a Zeeman term, the tab also exposes
the field-on energy quadrature. Solved quantities such as $\lambda(T)$,
$\chi_{0,\rm eff}$, and $\langle m^2\rangle$ appear in fit diagnostics.

Closures use finite-difference gradients and support the relaxational scalar,
tensor, and field-on models. Sum-rule closures and the Zeeman propagator
currently require `inverse_mode_energy_sq=0`, because their finite-frequency
moment equations have not yet been generalized to the inertial local
propagator. Their assumptions and sum rules are discussed in
[Theory: sum rules and self-consistency](theory_notes.md).

## Bulk susceptibility and magnetometry

The uniform static response is $\chi(\mathbf 0,0)$. Magnetization datasets can
be fitted jointly with inelastic datasets while sharing exchange and local
response parameters.

For the scalar model, the moment per magnetic ion is

$$
\frac{m}{\mu_B}
=g^2\mu_B^{({\rm meV/T})}
\chi_{\rm uniform}^{({\rm meV}^{-1})}B^{({\rm T})}.
$$

Here $m$ is magnetic moment, $B$ is magnetic flux density, and
$\mu_B^{({\rm meV/T})}=0.05788\ {\rm meV/T}$ is the numerical value of the
Bohr magneton in these units. $B$ is signed along the dataset's measurement
direction. For an absolute sample-moment channel, nfit also applies the sample
amount and number of magnetic ions on the selected molar basis. Selecting
susceptibility predicts $\chi_{\rm mol}$ directly; selecting moment also
multiplies by field. See [Physics conventions](physics_conventions.md) for the
rationalized-SI relation $M=\chi_{\rm SI}H$.

When the crystal contains every chemical site and element, nfit derives the
reduced formula and divides the expanded magnetic-site count by the formula
units in the crystallographic cell. This gives magnetic ions per formula unit
without confusing a centered conventional cell with one formula unit.
`bulk.sites_per_fu` remains an explicit override. Older magnetic-only crystal
definitions do not contain enough composition information for this inference;
they retain the historical expanded-magnetic-site fallback until an override
or complete crystal is supplied.

## Fitting a temperature series

For an unconstrained series, use per-dataset or grouped sharing for $\chi_0$
and $\Gamma_0$ while keeping exchange constants global. With a closure,
$\chi_0(T)$ is derived from a smaller set of global parameters.

Temperature and field are read from each dataset. Keep dataset scale fixed at
1 for normalized data; otherwise fit a dataset scale, independently or shared
within a dataset group.

## Analytic derivatives

The scalar evaluator differentiates the resolvent rather than its eigenvectors.
With

$$
A=\mathbb 1-\chi_0(E)J,\qquad
x=A^{-1}\phi,\qquad
z=A^{-\dagger}\phi,
$$

an exchange-orbit derivative is

$$
\frac{\partial\chi}{\partial J_o}
=\frac{\chi_0(E)^2}{N}z^\dagger P_o(\mathbf Q)x,
\qquad
P_o=\frac{\partial J}{\partial J_o}.
$$

$\phi=(1,\ldots,1)^T$ is the dimensionless uniform sublattice vector for the neutron observable, $J_o$ is
the exchange assigned to orbit $o$, and $P_o$ is that orbit's exchange
structure matrix. $A^{-\dagger}=(A^\dagger)^{-1}$, $x,z$ are dimensionless
solve vectors, and $P_o$ is dimensionless; the derivative has units meV$^{-2}$.
The expression remains well defined at degenerate exchange eigenvalues and
reuses the model factorization. Tensor mode currently uses central differences.

## Scripting and export

The numerical API exposes `build_rpa_geometry`,
`heisenberg_rpa_susceptibility`, `heisenberg_rpa_chipp`, and the related
derivative and exchange-matrix functions. Fitting uses
`ModelComponentSpec(type="heisenberg_rpa", ...)`; project files, workflow
scripts, fit-result export, reports, and model/residual channels use the shared
model machinery.

## References

- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*
  (Springer, 1985).
- N. R. Bernhoeft and G. G. Lonzarich, *J. Phys.: Condens. Matter* **7**,
  7325 (1995), [doi:10.1088/0953-8984/7/37/006](https://doi.org/10.1088/0953-8984/7/37/006).
- D. Dahlbom *et al.*, “Sunny.jl: A Julia Package for Spin Dynamics”
  (2025), [arXiv:2501.13095](https://doi.org/10.48550/arXiv.2501.13095).
- K. A. Ross *et al.*, *Phys. Rev. B* **84**, 064430
  (2011), [doi:10.1103/PhysRevB.84.064430](https://doi.org/10.1103/PhysRevB.84.064430).
- M. Enjalran and M. J. P. Gingras, *Phys. Rev. B* **70**, 174426
  (2004), [doi:10.1103/PhysRevB.70.174426](https://doi.org/10.1103/PhysRevB.70.174426).
