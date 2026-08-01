# Heisenberg RPA

The `heisenberg_rpa` model couples local relaxational spins through a
crystallographic exchange network. This page defines its phase convention,
observable, extensions, and fitting controls.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `chi0` | static single-site susceptibility $\chi_0$ | meV$^{-1}$ |
| `gamma0` | bare single-site relaxation energy $\Gamma_0$ | meV |
| exchange-orbit label, such as `J1` | exchange assigned to every bond in that orbit | meV |

The crystal and exchange editor determines the orbit labels. Tensor exchange,
single-ion anisotropy, dipolar interactions, Zeeman coupling, and
self-consistency closures add the parameters described in their sections
below.

## Scalar model

For $N$ magnetic sites in the chosen crystallographic cell,

$$
\chi_0(E)=\frac{\chi_0}{1-iE/\Gamma_0},
$$

and

$$
\chi(\mathbf Q,E)=
\left[\mathbb 1-\chi_0(E)J(\mathbf Q)\right]^{-1}\chi_0(E).
$$

$\chi_0$ is the static single-site susceptibility in meV$^{-1}$ and
$\Gamma_0$ is its relaxation energy in meV. Each fitted exchange $J_i$ is also
in meV, so $J_i\chi_0$ is dimensionless. $E$ is transferred energy,
$\mathbf Q$ is momentum transfer, $N$ is the number of magnetic sites, and
$\mathbb 1$ is the $N\times N$ identity matrix. The scalar $\chi$ is one
Cartesian spin-susceptibility component.

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
exchange eigenvalues in meV. Mode $\nu$ has

$$
\chi_{\mathbf Q\nu}=
\frac{\chi_0}{1-\lambda_\nu(\mathbf Q)\chi_0},
\qquad
\Gamma_\nu=
\Gamma_0[1-\lambda_\nu(\mathbf Q)\chi_0],
$$

and contributes

$$
\chi''(\mathbf Q,E)=
\sum_\nu w_\nu(\mathbf Q)
\frac{\chi_0\Gamma_0E}{E^2+\Gamma_\nu^2}.
$$

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

- with every $J_i=0$, the model reduces to the local relaxational response;
- $\chi''(\mathbf Q,-E)=-\chi''(\mathbf Q,E)$;
- at small positive energy, the response is enhanced by
  $[1-\lambda_\nu\chi_0]^{-2}$; and
- the leading high-energy tail is $\chi_0\Gamma_0/E$, independent of exchange.

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

$\theta_p$ is the fitted strength of tensor parameter $p$, and
$P_p(\mathbf Q)$ is its symmetry-generated structure matrix.

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

In the field frame, the longitudinal response is relaxational. The transverse
circular responses are centered at the Larmor energy

$$
E_L=g\mu_BB,
$$

where $g$ is dimensionless and
$\mu_B=0.05788\ {\rm meV/T}$. The parameters `chi_perp_ratio` and
`gamma_perp_ratio` set transverse-to-longitudinal ratios. A field direction and
magnitude must be defined in the dataset conditions.

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

Closures use finite-difference gradients and support scalar, tensor, and
field-on models. Their assumptions and sum rules are discussed in
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

$\phi$ is the uniform sublattice vector for the neutron observable, $J_o$ is
the exchange assigned to orbit $o$, and $P_o$ is that orbit's exchange
structure matrix. The expression remains well defined at degenerate bands and
reuses the model factorization. Tensor mode currently uses central differences.

## Scripting and export

The numerical API exposes `build_rpa_geometry`, `heisenberg_rpa_chipp`, and
the related derivative and exchange-matrix functions. Fitting uses
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
