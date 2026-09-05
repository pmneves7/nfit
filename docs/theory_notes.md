# Sum rules and self-consistency

This page explains the optional closures used with
[`heisenberg_rpa`](heisenberg_rpa.md). It separates exact response identities
from the assumptions that make a model self-consistent.

## Bare RPA

Without a closure, nfit evaluates

$$
\boldsymbol\chi(\mathbf Q,E)
=\left[\mathbb 1-\chi_0(E)\mathbf J(\mathbf Q)\right]^{-1}\chi_0(E),
\qquad
\chi_0(E)=\frac{\chi_0}{1-iE/\Gamma_0}.
$$

$E$ is transferred energy in meV, $\mathbf Q$ is momentum transfer,
$\mathbf J$ is the sublattice exchange matrix in meV, $\mathbb1$ is its
identity, $\chi_0$ is a scalar static local spin susceptibility in meV$^{-1}$,
and $\Gamma_0>0$ is the local relaxation energy in meV. Multiplication by
$\chi_0(E)$ on the right means $\chi_0(E)\mathbb1$. The isotropic model has
$N_s$ sites and an $N_s\times N_s$ matrix per Cartesian component; the tensor
model uses the $3N_s$ site/component space. The local exponential-relaxation
ansatz and exchange Hamiltonian are defined on the
[Heisenberg page](heisenberg_rpa.md#scalar-model).

This is linear response around a fixed local propagator. The fitted
`chi0`, `gamma0`, and exchange parameters determine the response, but the
resulting spectral weight does not feed back into those parameters.

Bare RPA is a useful empirical baseline. It does not, however, conserve a
local moment, and its mean-field instability generally overestimates ordering
tendencies. A closure adds one scalar condition per temperature to address
one of these limitations.

## Exact response identities

### Total moment

Use the dimensionless spin and equilibrium average defined in
[Physics conventions](physics_conventions.md#spin-operators-and-equilibrium-averages).
For $N_s$ magnetic sites per cell, the site-diagonal correlation is

$$
S_{aa}^{\alpha\alpha}(\mathbf q,E)=\frac{1}{2\pi\hbar}
\sum_{\mathbf R}e^{2\pi i\mathbf q\cdot\mathbf R}
\int_{-\infty}^{\infty}dt\,e^{-iEt/\hbar}
\langle S_{0a}^{\alpha}(0)S_{\mathbf R a}^{\alpha}(t)\rangle .
$$

$a=1,\ldots,N_s$ labels a site within a cell, $\mathbf R$ an integer cell
translation, $0$ the home cell, $\alpha=x,y,z$ a Cartesian component, and
$\mathbf q$ dimensionless reduced momentum. $E$ is in meV; $\hbar$ and time
$t$ must use compatible units. This correlation has units meV$^{-1}$.
The per-site, component-summed total moment obeys

$$
\frac{1}{N_s}\sum_{a,\alpha}\int_{\rm BZ}d\bar q
\int_{-\infty}^{\infty}dE\,S_{aa}^{\alpha\alpha}(\mathbf q,E)
=\frac1{N_s}\sum_a\langle\mathbf S_{0a}^2\rangle,
\qquad
d\bar q=\frac{d^d q_{\rm cart}}{V_{\rm BZ}}.
$$

$d$ is the number of periodic dimensions and $V_{\rm BZ}$ is the reciprocal
volume (area/length for $d=2/1$), so the measure integrates to one. The BZ
average cancels all nonzero cell translations. For a rigid spin of quantum
number $S$, the dimensionless local value is $S(S+1)$. For itinerant electrons
it need not be fixed. The neutron-visible sublattice sum includes off-site
interference; it cannot replace this site trace in a multi-site cell.

In the paramagnetic state of these closures the mean spin vanishes, and the
fluctuation--dissipation theorem gives
$S_{aa}^{\alpha\alpha}=\chi''_{aa,\alpha\alpha}/\{\pi[1-e^{-E/(k_BT)}]\}$.
In a state with nonzero mean spin, the retarded response describes connected
fluctuations; recovering the full equal-time moment also requires the static
mean-spin contribution. $k_B$ is Boltzmann's constant in meV/K and $T$ is K.

Quantitative use requires two choices:

- The integral is over the full Brillouin zone, not only the measured region.
- A relaxational spectrum needs a physical high-energy cutoff $\Lambda$.
  Its local integral grows as
  $\chi_0\Gamma_0\ln(\Lambda/\Gamma_0)$.

### Static susceptibility

The zero-frequency Kramers--Kronig relation is

$$
\chi'(\mathbf Q,0)
=\frac{2}{\pi}\int_0^\infty
\frac{\chi''(\mathbf Q,E)}{E}\,dE.
$$

For the relaxational response this integral equals the fitted static
susceptibility at the same $\mathbf Q$. Bulk magnetometry probes
$\mathbf Q=0$; the required $g$, $\mu_B$, $\mu_0$, number-density, and molar
conversions are given in [Physics conventions](physics_conventions.md).

The Hohenberg--Brinkman first-moment sum rule provides a further consistency
check between the energy-weighted spectrum and exchange-weighted equal-time
correlations [8]. nfit does not currently use it as a fit constraint.

## Available closures

Each closure keeps the RPA response form and determines its local parameters
from one additional equation.

| closure | feedback or conserved quantity | suitable interpretation |
| --- | --- | --- |
| none | none | empirical RPA baseline |
| Onsager / spherical | fixed integrated moment target | fixed-length local moments |
| SCR | fluctuation amplitude renormalizes `chi0` | soft itinerant moments |
| TAC | zero-point plus thermal amplitude is fixed | quantum itinerant moments |

TPSC and Anderson-lattice physics are useful comparisons, but are not
implemented closures.

### Onsager or spherical closure

The closure shifts every exchange eigenvalue by a scalar reaction field:

$$
\mathbf J(\mathbf Q)\longrightarrow
\mathbf J(\mathbf Q)-\lambda(T)\mathbb 1.
$$

$\lambda(T)$ is a reaction-field energy in meV, $\mathbb1$ the site/component
identity, and the integrated quantity is $\langle m^2\rangle$ defined below,
including zero-point and thermal weight up to the specified cutoff.

nfit solves for $\lambda(T)$ so that the integrated moment equals
`moment_target` or the fitted `m2_total`. This replaces a strict site-by-site
constraint with an average one and prevents unlimited accumulation of weight
in a soft mode [1--4].

### Self-consistent renormalization

Moriya's SCR treats the moment amplitude as a soft field:

$$
\chi_{0,\rm eff}^{-1}(T)=\chi_{0,\rm bare}^{-1}+u\langle m^2\rangle(T),
$$

with

$$
\langle m^2\rangle(T)
=\int_{\rm BZ}d\bar q\int_0^\Lambda dE\,
\coth\!\left(\frac{E}{2k_BT}\right)
\frac{\mathcal X''(\mathbf q,E)}{\pi},\qquad
\mathcal X''=\frac1{N_s}\sum_{a,\alpha}\chi''_{aa,\alpha\alpha}.
$$

$\chi_{0,\rm bare}$ is the fitted reference `chi0`, not the already
renormalized value at zero temperature; zero-point weight also enters the
self-consistent solution at $T=0$. $\mathcal X''$ is the local site/component
trace in meV$^{-1}$, with no neutron polarization, form factor, or mode
interference weights. For an isotropic scalar sublattice response it is
$3\operatorname{Tr}_{\rm site}\chi''/N_s$. The hyperbolic factor is
$\coth x=(e^{2x}+1)/(e^{2x}-1)=1+2n_B(E,T)$ for $x=E/(2k_BT)$.

Here $\langle m^2\rangle$ is the component-summed fluctuating spin amplitude,
$\Lambda$ is the energy cutoff, and
$u$ is a mode-coupling energy because $m^2$ is dimensionless. Because
$\chi''$ depends on $\chi_0$, nfit solves this equation self-consistently.
`mode_coupling_u` controls $u$ [5, 6].

### Total-amplitude conservation

Takahashi's TAC closure conserves the combined quantum and thermal amplitude:

$$
\langle m^2\rangle_{\rm ZP}(T)
+\langle m^2\rangle_T(T)
=\text{constant}.
$$

In the same per-site spin convention,
$\langle m^2\rangle_{\rm ZP}=\int d\bar q\int_0^\Lambda dE\,\mathcal X''/\pi$
and $\langle m^2\rangle_T=\int d\bar q\int_0^\Lambda dE\,2n_B(E,T)\mathcal X''/\pi$.
ZP denotes zero-point and subscript $T$ the thermal contribution. Both can
depend on temperature through the self-consistent spectrum, even though the
zero-point kernel has no explicit Bose occupation. Their dimensionless sum
is the configured `total_amplitude`.

The fitted `total_amplitude` replaces an independently varied `chi0`.
`gamma0` remains a fitted parameter; nfit does not derive its full temperature
dependence [7].

## Parameters and diagnostics

| quantity | no closure | with a closure |
| --- | --- | --- |
| local static susceptibility | fitted `chi0` | derived by Onsager, SCR, or TAC |
| relaxation rate | fitted `gamma0` | still fitted |
| reaction field | absent | derived `lambda(T)` for Onsager |
| moment budget | diagnostic integral | `moment_target`, `m2_total`, or `total_amplitude` |
| energy cutoff | diagnostic choice | shared `energy_cutoff_mev` |

`chi0` is the local static susceptibility before intersite RPA enhancement. It
has units of inverse energy. Without a closure it is fitted and used directly.
With SCR it is the fitted bare reference susceptibility; with TAC it seeds the
self-consistent solve.

`chi0_eff` is the local susceptibility actually inserted into the RPA
denominator. It equals `chi0` without a closure and is derived by SCR or TAC.
`lambda_shift` is the Onsager reaction-field energy. Onsager subtracts it from
every interaction eigenvalue,
$\lambda_\nu(\mathbf Q)\rightarrow\lambda_\nu(\mathbf Q)-\lambda_{\rm shift}$,
to enforce the configured moment sum rule. It is zero for the other modes.

$\lambda_\nu(\mathbf Q)$ is interaction eigenvalue $\nu$ in meV, distinct
from the scalar reaction field. $D_\nu$, $D_{\min}$, and $r_{\max}$ below are
dimensionless; the extrema run over every sampled momentum and eigenmode.

For each sampled wavevector and mode, define

$$
D_\nu(\mathbf Q)
=1-\left[\lambda_\nu(\mathbf Q)-\lambda_{\rm shift}\right]
\chi_{0,\rm eff}.
$$

nfit reports

$$
r_{\max}=\max_{\mathbf Q,\nu}
\left[\lambda_\nu(\mathbf Q)-\lambda_{\rm shift}\right]\chi_{0,\rm eff},
\qquad
D_{\min}=1-r_{\max}.
$$

`stability_margin` is $D_{\min}$: it is positive in the stable region, zero at
the RPA boundary, and negative beyond it. The report also stores the critical
sampled HKL and mode. With a closure, nfit stores both the bare margin and the
effective self-consistent margin. If the closure has no solution, the bare
margin remains available and the record states that an effective closure was
not obtained. The BZ grid is finite, so this is the smallest *sampled*
denominator rather than a continuous optimization over reciprocal space.

The fit report also records the integrated moment, static susceptibilities,
and `chi0 * gamma0`. These trends can help choose a model:

- nearly constant integrated moment suggests an Onsager/local-moment model;
- a growing amplitude that renormalizes inverse susceptibility suggests SCR;
- redistribution between zero-point and thermal weight suggests TAC;
- a collapsing moment with a saturating linewidth may require a
  valence-fluctuation model not supplied by these closures.

Compare any closure with the unconstrained fit using both fit quality and
parameter count. A closure is most useful when it explains a temperature
series with shared parameters, not merely when it reproduces one dataset.

## Numerical details

Moment integrals use a midpoint-shifted Monkhorst--Pack grid configured by
`config["closure"]["bz_grid"]`. This grid spans the full Brillouin zone and is
independent of the measured points. `energy_cutoff_mev` sets $\Lambda$.

With no applied field, nfit integrates the mode Lorentzians using analytic or
Matsubara forms checked against adaptive quadrature. With a field, it
integrates $\operatorname{Tr}\chi''$ numerically. Closure results are cached
across repeated evaluations; see [Performance notes](performance.md) for the
computational implications.

For a molar CGS comparison, the scalar model conversion used internally is

$$
\chi_{\rm mol}\,[{\rm emu/mol}]
=C\,g^2\chi_{\rm spin}\,[{\rm meV}^{-1}],
\qquad
C\approx0.0323\ {\rm emu\,meV/mol},
$$

per mole of magnetic sites. Here $\chi_{\rm spin}=\chi'(\mathbf0,0)$ is
one Cartesian component and $C=N_A\mu_{B,\rm CGS}^2/\varepsilon_{\rm meV,erg}$,
with $\mu_{B,\rm CGS}=9.2740100783\times10^{-21}$ erg/G and
$\varepsilon_{\rm meV,erg}=1.602176634\times10^{-15}$ erg.
“emu/mol” is the package's conventional CGS susceptibility shorthand for
cm$^3$/mol (emu/(Oe mol)), not a molar magnetic moment. Absolute sample normalization additionally
requires the sample amount and the number of magnetic sites on the chosen
molar basis.

## References

1. T. H. Berlin and M. Kac, "The spherical model of a ferromagnet",
   Phys. Rev. **86**, 821 (1952). <https://doi.org/10.1103/PhysRev.86.821>
2. R. Brout and H. Thomas, "Molecular field theory, the Onsager reaction
   field and the spherical model", Physics Physique Fizika **3**, 317 (1967).
   <https://doi.org/10.1103/PhysicsPhysiqueFizika.3.317>
3. D. A. Garanin and B. Canals, "Classical spin liquid: Exact solution for
   the infinite-component antiferromagnet on the kagomé lattice",
   Phys. Rev. B **59**, 443 (1999).
   <https://doi.org/10.1103/PhysRevB.59.443>
4. P. H. Conlon and J. T. Chalker, "Absent pinch points and emergent clusters:
   Further neighbor interactions in the pyrochlore Heisenberg
   antiferromagnet", Phys. Rev. B **81**, 224413 (2010).
   <https://doi.org/10.1103/PhysRevB.81.224413>
5. T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*, Springer
   Series in Solid-State Sciences 56 (Springer, Berlin, 1985).
6. G. G. Lonzarich and L. Taillefer, "Effect of spin fluctuations on the
   magnetic equation of state of ferromagnetic or nearly ferromagnetic
   metals", J. Phys. C **18**, 4339 (1985).
   <https://doi.org/10.1088/0022-3719/18/22/017>
7. Y. Takahashi, "On the origin of the Curie--Weiss law of the magnetic
   susceptibility in itinerant electron ferromagnetism",
   J. Phys. Soc. Jpn. **55**, 3553 (1986).
   <https://doi.org/10.1143/JPSJ.55.3553>
8. P. C. Hohenberg and W. F. Brinkman, "Sum rules for the frequency spectrum
   of linear magnetic chains", Phys. Rev. B **10**, 128 (1974).
   <https://doi.org/10.1103/PhysRevB.10.128>
9. N. D. Mermin and H. Wagner, "Absence of ferromagnetism or
   antiferromagnetism in one- or two-dimensional isotropic Heisenberg
   models", Phys. Rev. Lett. **17**, 1133 (1966).
   <https://doi.org/10.1103/PhysRevLett.17.1133>
10. I. A. Zaliznyak and S.-H. Lee, "Magnetic neutron scattering", in *Modern
    Techniques for Characterizing Magnetic Materials*, edited by Y. Zhu
    (Springer, Boston, 2005).
11. J. M. Lawrence, P. S. Riseborough, and R. D. Parks, "Valence fluctuation
    phenomena", Rep. Prog. Phys. **44**, 1 (1981).
    <https://doi.org/10.1088/0034-4885/44/1/001>
12. E. Holland-Moritz, D. Wohlleben, and M. Loewenhaupt, "Neutron
    spectroscopy of intermediate-valence compounds", Phys. Rev. B **25**,
    7482 (1982). <https://doi.org/10.1103/PhysRevB.25.7482>
13. Y. M. Vilk and A.-M. S. Tremblay, "Non-perturbative many-body approach to
    the Hubbard model and single-particle pseudogap",
    J. Phys. I France **7**, 1309 (1997).
    <https://doi.org/10.1051/jp1:1997135>
