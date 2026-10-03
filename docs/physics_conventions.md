# Physics conventions

nfit uses energy transfer $E = E_i-E_f=\hbar\omega$ in meV, with positive
$E$ denoting neutron energy loss. Momentum is in reciprocal-lattice units or
Å$^{-1}$, temperature is in K, and a double-differential cross section is per
steradian and per meV. Model kernels produce the dissipative response
$\chi''(\mathbf Q,E)$ before instrumental factors are applied.

Electronic-structure inputs and plots use eV by default, following common
DFT and Wannier conventions. nfit converts them once to canonical meV when an
`ElectronicModel` is built or imported. Neutron energy transfer,
spin-fluctuation linewidths, magnetic exchange, susceptibilities, and response
kernels remain in meV. Every unit-neutral electronic adapter must declare its
input energy unit; nfit never infers a unit from a numerical magnitude.

Common symbols are:

| symbol | meaning |
| --- | --- |
| $E_i,E_f,E$ | incident, final, and transferred neutron energy |
| $\mathbf k_i,\mathbf k_f$ | incident and final neutron wavevectors; $k_i,k_f$ are their magnitudes |
| $\mathbf Q=\mathbf k_i-\mathbf k_f$ | momentum transfer; $Q=|\mathbf Q|$ and $\hat{\mathbf Q}=\mathbf Q/Q$ |
| $T$, $k_B$ | absolute temperature and Boltzmann constant |
| $\omega$, $\hbar$ | angular frequency and reduced Planck constant, with $E=\hbar\omega$ |
| $g$, $\mu_B$, $\mu_0$ | dimensionless Landé factor, Bohr magneton, and vacuum permeability |
| $B$, $H$, $M$ | magnetic flux density (T), magnetic field strength (A/m), and magnetization (A/m) |
| $N_A$ | Avogadro constant |
| $\alpha,\beta$ | Cartesian components $x,y,z$ |

## Constants, coordinates, and mathematical notation

Numerical energies in response equations are in meV. A kelvin is converted
into energy by $k_B T$, never by setting $k_B=1$ without changing units.
The following values document the numerical constants used by the package;
rounded implementation values are not claims of exact metrological values.

| Symbol | Definition and value |
| --- | --- |
| $k_B$ | Boltzmann constant, exactly $1.380649\times10^{-23}$ J/K; response kernels use $0.08617333262$ meV/K |
| $h$, $\hbar=h/(2\pi)$ | Planck constant, exactly $h=6.62607015\times10^{-34}$ J s; $\hbar\simeq0.658211957$ meV ps |
| $e$ | positive elementary charge, exactly $1.602176634\times10^{-19}$ C; an electron has charge $-e$ |
| $\mu_B=e\hbar/(2m_e)$ | Bohr magneton; bulk conversion uses $9.2740100783\times10^{-24}$ J/T, and the Zeeman kernel uses $0.05788381$ meV/T; $m_e$ is electron rest mass |
| $\mu_0$ | vacuum permeability in N/A$^2$; bulk conversion uses the approximation $4\pi\times10^{-7}$ |
| $N_A$ | entities per mole, exactly $6.02214076\times10^{23}$ mol$^{-1}$ |
| $R=N_Ak_B$ | molar gas constant, $8.31446261815324$ J/(mol K), or $8314.46261815324$ mJ/(mol K) |
| $r_0=e^2/(4\pi\epsilon_0m_ec^2)$ | classical electron radius, approximately $2.81794\times10^{-15}$ m; $\epsilon_0$ is vacuum permittivity and $c=299792458$ m/s is the speed of light |
| $\gamma$ | dimensionless neutron moment in nuclear magnetons, approximately $-1.913$; $\boldsymbol\mu_n=\gamma\mu_N\boldsymbol\sigma$, with $\mu_N=e\hbar/(2m_p)$ and proton rest mass $m_p$ |

One meV is exactly $1.602176634\times10^{-22}$ J, one Å is $10^{-10}$ m,
one barn is $10^{-28}$ m$^2$, and one mbarn is $10^{-3}$ barn. The cross-section
kernel uses $(\gamma r_0/2)^2=0.07265$ barn directly. The modern SI fixes
$h,e,k_B,N_A,c$; $\mu_0$, particle masses, and magnetons remain measured
quantities. See the [SI defining constants](https://www.nist.gov/pml/special-publication-330/sp-330-section-2)
and [NIST CODATA tables](https://physics.nist.gov/cuu/Constants/).

Native DGS/MDE conversion pins Mantid 6.16's numerical constants:
$h_M=6.62606896\times10^{-34}$ J s, neutron mass
$m_{n,M}=1.674927211\times10^{-27}$ kg, and energy conversion
$J_M=1.602176487\times10^{-22}$ J/meV. For incident or final neutron
kinetic energy $E_n$ in meV and wavevector magnitude $k$ in Å$^{-1}$,
$E_n=A_M k^2$, where
$A_M=10^{20}(h_M/(2\pi))^2/(2m_{n,M}J_M)$ has units meV Å$^2$.
MDNorm's reciprocal coefficient is evaluated separately as
$B_M=8\pi^2m_{n,M}J_M10^{-20}/h_M^2$, in Å$^{-2}$/meV, rather than computing
$1/A_M$; algebraic cancellation can change floating-point rounding at bin
boundaries. These pinned constants are used by both event-precision policies;
high precision does not select a newer set of physical constants.

Here $i^2=-1$, a star denotes complex conjugation, $X^T$ transpose, and
$X^\dagger=(X^*)^T$ the Hermitian adjoint. $\mathbb1$ or $I_N$ is the
identity on the stated $N$-dimensional space, $\operatorname{Tr}X=\sum_aX_{aa}$
is the trace, and $X\otimes Y$ is the tensor (Kronecker) product with elements
$(X\otimes Y)_{a\sigma,b\sigma'}=X_{ab}Y_{\sigma\sigma'}$.
$\delta_{ab}$ is one for equal indices and zero otherwise.
A ket $|v\rangle$ is a state vector; its bra is $\langle v|=|v\rangle^\dagger$.
A matrix element $\langle v|O|w\rangle$ is $\sum_{ab}v_a^*O_{ab}w_b$.
Symbols reused in separate models (for example interaction $g$ and Landé $g$)
have the local definitions and units stated on those pages.

For direct-lattice columns $A=(\mathbf a_1,\mathbf a_2,\mathbf a_3)$ in Å,
the reciprocal matrix is $B=2\pi A^{-T}$ in Å$^{-1}$, so
$\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$. Fractional position $\mathbf r$
and reduced momentum $\mathbf q$ are dimensionless:
$\mathbf r_{\rm cart}=A\mathbf r$ and $\mathbf q_{\rm cart}=B\mathbf q$.
Thus $\mathbf q_{\rm cart}\cdot\mathbf r_{\rm cart}=2\pi\mathbf q\cdot\mathbf r$.
Reciprocal-lattice units (r.l.u., HKL) always refer to a declared cell.
The Brillouin zone (BZ) is one primitive reciprocal cell; $\mathbf G$ denotes
an integer reciprocal translation in reduced coordinates. The electronic
$\mathbf q$ is the representative of extended transfer $\mathbf Q$ modulo
$\mathbf G$. Physical momentum is $\hbar\mathbf Q_{\rm cart}$; the conventional
scattering label “momentum transfer” usually means its wavevector.
The instrument UB convention omits $2\pi$: $\mathbf Q_{\rm sample}=2\pi UB(H,K,L)^T$.
Here $U$ is the sample orientation rotation and the $B$ inside UB is the
crystallographic reciprocal basis without $2\pi$, unlike the electronic $B$ above.
Internally, raw SNS instrument coordinates use beam $+z$ and vertical $+y$.
ISAW UB files use the IPNS frame with beam $+x$ and vertical $+z$; nfit applies
the same row permutation as Mantid's `LoadIsawUB` while reading and its inverse
while writing.

## Spin operators and equilibrium averages

A dimensionless spin of quantum number $s=0,\tfrac12,1,\ldots$ acts on
$|s,m\rangle$, with $m=-s,-s+1,\ldots,s$, as

$$
S_z|s,m\rangle=m|s,m\rangle,\qquad
S_\pm|s,m\rangle=\sqrt{s(s+1)-m(m\pm1)}\,|s,m\pm1\rangle,
$$

where $S_\pm=S_x\pm iS_y$, with a zero result outside the allowed $m$ range.
Consequently $[S_\alpha,S_\beta]=i\sum_\gamma\epsilon_{\alpha\beta\gamma}S_\gamma$
and $\mathbf S^2=s(s+1)\mathbb1$. Here $[X,Y]=XY-YX$, and the Levi-Civita
symbol $\epsilon_{xyz}=1$ is antisymmetric and vanishes for repeated indices.
Physical angular momentum is $\hbar\mathbf S$. The scalar spin-length input
called $S$ in sum rules is this quantum number $s$, not an operator or a
structure factor.

For an electron, $s=1/2$ and $S_\alpha=\sigma_\alpha/2$ in the ordered
$(|\uparrow\rangle,|\downarrow\rangle)$ basis, where

$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

These are the dimensionless Pauli matrices. An orbital basis of size $N$
uses $I_N\otimes\sigma_\alpha/2$. Orbital angular momentum $\mathbf L$ obeys
the same ladder construction with integer $l$, and total angular momentum is
$\mathbf J=\mathbf L+\mathbf S$, all dimensionless here. A spin-only moment
is $-g\mu_B\mathbf S$. Within an isolated total-$J$ multiplet the projected
moment is $-g_J\mu_B\mathbf J$; an effective spin model must declare which
multiplet or pseudospin it represents. A projected subspace need not retain
the full angular-momentum algebra.

A thermal average means $\langle O\rangle=\operatorname{Tr}(\rho O)$, with
$\rho=Z^{-1}\exp[-\hat H/(k_BT)]$ and
$Z=\operatorname{Tr}\exp[-\hat H/(k_BT)]$ for fixed particle number.
$\hat H$ is the many-body Hamiltonian in energy units and $Z$ the dimensionless
partition function. For electrons with fluctuating particle number use
$\hat H-\mu\hat N$ in the exponential, where $\mu$ is chemical potential and
$\hat N$ the electron-number operator. Time evolution is
$O(t)=e^{i\hat Ht/\hbar}Oe^{-i\hat Ht/\hbar}$.
Connected fluctuations are $\delta O=O-\langle O\rangle$; static ordered
moments give an additional elastic contribution to unconnected correlations.

Independent electron levels of energy $\varepsilon$ have occupation
$f(\varepsilon)=[e^{(\varepsilon-\mu)/(k_BT)}+1]^{-1}$.
At $T=0$, nfit uses $f=1$ below $\mu$, $0$ above, and $1/2$ at equality.
Bosons of positive excitation energy $E$ have occupation
$n_B(E,T)=[e^{E/(k_BT)}-1]^{-1}$; $n_B+1$ is the energy-loss population
factor. $f$ here is unrelated to the magnetic form factor $f(Q)$.

## Microscopic spin response

### Dynamic spin correlation function

nfit follows the Fourier-transform convention in Squires, chapter 7,
Eq. 7.73. For equivalent magnetic ions at equilibrium positions
$\mathbf R_l$, let $\mathbf S$ denote the dimensionless
spin operator. The dynamic spin correlation function per magnetic ion is

$$
S^{\alpha\beta}_s(\mathbf Q,E)
=
\frac{1}{2\pi\hbar}
\sum_l \exp(i\mathbf Q\mathbin{\cdot}\mathbf R_l)
\int_{-\infty}^{\infty} dt\,
\exp(-iEt/\hbar)
\left\langle
S^\alpha_0(0)S^\beta_l(t)
\right\rangle .
$$

$l$ labels magnetic ions, $\mathbf R_l$ is the equilibrium position of ion
$l$ relative to ion 0, $t$ is time, and angle brackets denote a thermal
equilibrium average.

Because $\mathbf S$ is dimensionless, $S^{\alpha\beta}_s$ has dimensions of
inverse energy. nfit labels this normalization `spin^2/meV` when $E$ is in
meV; `spin^2` identifies the operator convention rather than an additional SI
dimension. The phase signs, operator order, and factor $1/(2\pi\hbar)$ are
part of the Fourier-transform convention. The phase
$\mathbf Q\cdot\mathbf R_l$ is dimensionless:
$\mathbf Q$ and $\mathbf R_l$ in this equation are physical inverse-length
and length vectors. For a crystal basis, the sum is generalized to both site
indices and normalized by the number of reference magnetic ions; nfit's
multi-sublattice mode weights use that same per-site normalization.

Squires Eq. 7.73 also includes the positional factor
$\langle e^{-i\mathbf Q\cdot\mathbf u_0(0)}
e^{i\mathbf Q\cdot\mathbf u_l(t)}\rangle$. nfit's spin-fluctuation kernels
use the rigid-lattice magnetic response, so this factor is one. A
Debye--Waller or magnetovibrational correction, when needed, must therefore
be included explicitly in the dataset reduction or model rather than being
hidden inside $S_s^{\alpha\beta}$.

### Linear response and the fluctuation--dissipation theorem

The dynamic susceptibility describes the response to a weak magnetic
perturbation. Let $h^\beta(\mathbf Q,E)$ be the energy-like field conjugate to
$S^\beta(-\mathbf Q)$, so that the perturbing Hamiltonian contains
$-h^\beta S^\beta$. For $\boldsymbol\mu=-g\mu_B\mathbf S$, the Zeeman
energy is $-\boldsymbol\mu\cdot\mathbf B=+g\mu_B\mathbf S\cdot\mathbf B$,
so this choice of conjugate spin field has $\mathbf h=-g\mu_B\mathbf B$.
The two minus signs cancel in moment-to-field susceptibility.
To first order,

$$
\delta\langle S^\alpha(\mathbf Q,E)\rangle
=
\sum_\beta \chi_{s,\alpha\beta}(\mathbf Q,E)
h^\beta(\mathbf Q,E).
$$

For example, for Fourier operators
$S_\alpha(\mathbf Q)=\sum_l e^{-i\mathbf Q\cdot\mathbf R_l}S_{l\alpha}$
in a periodic system of $N_{\rm ion}$ equivalent ions, the retarded definition is

$$
\chi_{s,\alpha\beta}(\mathbf Q,E)=\frac{i}{\hbar N_{\rm ion}}
\int_0^\infty dt\,e^{i(E+i0^+)t/\hbar}
\langle[S_\alpha(\mathbf Q,t),S_\beta(-\mathbf Q,0)]\rangle.
$$

The positive infinitesimal $0^+$ has energy units and ensures convergence.
The perturbation at this wavevector couples to the conjugate Fourier operator;
$\delta\langle S_\alpha(\mathbf Q)\rangle/N_{\rm ion}$ is the response per ion.
For a scalar channel this convention produces positive absorption at positive
$E$ and agrees with the correlation convention above. For a tensor the
absorptive part means $(\boldsymbol\chi-\boldsymbol\chi^\dagger)/(2i)$,
not elementwise imaginary parts of complex off-diagonal entries.
These are equilibrium spectral properties of an exact retarded response;
approximations such as constant-width Lindhard broadening need not preserve
all of them at finite width. See the [Lindhard response](lindhard.md#complex-susceptibility).

The retarded susceptibility is causal. For a scalar component, its real part $\chi'_s$ is the
in-phase, reactive response, and $\chi'_s(\mathbf Q,0)$ is the static
susceptibility. Its imaginary part $\chi''_s$ is the out-of-phase,
dissipative response measured by inelastic scattering; “imaginary” describes
the phase of the response function. With the spin normalization above, both
parts have units `spin^2/meV`.

Bulk measurements instead use
$\delta M_\alpha=\sum_\beta\chi^{\alpha\beta}_{MH}\delta H_\beta$.
Magnetic-moment, sample-normalization, and field-unit factors make this SI or
molar susceptibility numerically distinct from $\chi_s$; see
[SI bulk susceptibility and magnetization](#si-bulk-susceptibility-and-magnetization).

In nfit's per-unit-energy convention, the fluctuation--dissipation theorem is

$$
S_s^{\alpha\beta}(\mathbf Q,E)=
\frac{1}{\pi}\,
\frac{\chi''_{s,\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_BT)]}.
$$

In this convention, $1/\pi$ relates $S_s$ to $\chi''_s$ and is not included
in the definition of $\chi''_s$. The Bose or detailed-balance denominator is
evaluated stably near $E=0$. For reciprocal scalar channels the dissipative
response is odd in energy at fixed $\mathbf Q$. In general detailed balance
relates opposite momenta and exchanged tensor indices:
$S_s^{\alpha\beta}(\mathbf Q,-E)=e^{-E/(k_BT)}
S_s^{\beta\alpha}(-\mathbf Q,E)$.
Strictly elastic mean-spin or nondecaying connected correlations at $E=0$
must be handled separately; multiplying by $1-e^{-E/(k_BT)}$ annihilates
such a delta function, so a finite-energy absorptive spectrum alone cannot
reconstruct its weight.

### Spin and magnetic-moment susceptibility

For the dimensionless spin operator used above, the magnetic moment operator
is $\boldsymbol\mu=-g\mu_B\mathbf S$. The Landé factor $g$ is dimensionless,
and $\mu_B$ supplies the magnetic-moment unit. The corresponding physical
moment susceptibility is

$$
\chi''_{\rm moment}(\mathbf Q,E)
=(g\mu_B)^2\chi''_s(\mathbf Q,E).
$$

It is useful to define the moment response normalized by $\mu_B^2$,

$$
\bar\chi''_\mu
\equiv \frac{\chi''_{\rm moment}}{\mu_B^2}.
$$

This quantity has dimensions of inverse energy. Its numerical ordinate is what
nfit labels in `mu_B^2/meV`. Combining the definitions gives

$$
\chi''_{\rm moment}
=\mu_B^2\bar\chi''_\mu
=(g\mu_B)^2\chi''_s,
\qquad
\bar\chi''_\mu=g^2\chi''_s.
$$

Thus $g^2$ converts numerical ordinates from `spin^2/meV` to
`mu_B^2/meV`, while the physical conversion is $(g\mu_B)^2$. The unit
declaration determines how nfit applies this relation:

- Model kernels produce $\chi''_s$. A model curve requested in
  `mu_B^2/meV` is multiplied by the dataset's $g^2$; one requested in
  `spin^2/meV` is unchanged.
- An imported response declared in `spin^2/meV` is converted by $g^2$ when
  nfit calculates a cross section in the moment convention, and the converted
  response is represented in `mu_B^2/meV`.
- An imported response declared in `mu_B^2/meV` already represents
  $\bar\chi''_\mu$ and is used without another factor of $g^2$.

The dataset parameter `g_factor` is unitless and may differ from 2. The same
spin-to-moment relation accounts for the factor $(g/2)^2$ in the cross-section
formula below.

## Neutron cross sections

### Inelastic magnetic scattering

With this per-ion definition, the magnetic cross section is

$$
\frac{d^2\sigma}{d\Omega\,dE} =
\frac{k_f}{k_i}(\gamma r_0)^2
\left(\frac{g}{2}\right)^2 |f(Q)|^2
\sum_{\alpha\beta}
(\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta)
S_s^{\alpha\beta}(\mathbf Q,E).
$$

$d\Omega$ is detector solid angle, $f(Q)$ is the dimensionless magnetic
amplitude form factor, $\delta_{\alpha\beta}$ is the Kronecker delta,
$\gamma$ is the neutron magnetic-moment coefficient, and $r_0$ is the
classical electron radius. The tensor in parentheses projects out moment
components parallel to $\mathbf Q$.

For a sample of $N$ equivalent magnetic ions, Squires writes the total
cross section with an additional factor $N$ outside the per-ion correlation
function. nfit instead carries the selected per-magnetic-ion, per-formula-unit,
or per-unit-cell normalization as dataset metadata.

Substituting the fluctuation--dissipation relation gives the form used for an
unpolarized measurement in the dipole approximation:

$$
\frac{d^2\sigma}{d\Omega\,dE} =
\frac{k_f}{k_i}\frac{(\gamma r_0)^2}{\pi}
\left(\frac{g}{2}\right)^2 |f(Q)|^2
\sum_{\alpha\beta}
(\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta)
\frac{\chi''_{s,\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_B T)]},
$$

The kinematic ratio, form factor, projector, $g$, and Bose denominator are
dimensionless. The remaining area coefficient times the inverse-energy
susceptibility therefore gives cross section per energy. Since

$$
(\gamma r_0)^2\left(\frac{g}{2}\right)^2
=\left(\frac{\gamma r_0}{2}\right)^2g^2,
\qquad
\left(\frac{\gamma r_0}{2}\right)^2=0.07265\ {\rm barn},
$$

the spin-response form has area coefficient
$0.07265\,g^2\ {\rm barn}$. For the physical moment response, the same
normalization is

$$
C_\mu
=\frac{(\gamma r_0/2)^2}{\mu_B^2}
=0.07265\ \frac{{\rm barn}}{\mu_B^2},
$$

so $C_\mu\chi''_{\rm moment}$ has units barn per energy. The relation between
the spin and moment conventions is defined above.

See Berk's NIST review for the general double-differential and magnetic
correlation-function formalism,
Squires for the magnetic cross section and linear-response convention, and
Welch *et al.* for an explicit absolute-unit derivation
([Berk 1993](https://doi.org/10.6028/jres.098.002);
[Squires, chapter 7](https://doi.org/10.1017/CBO9781139107808.008);
[Welch *et al.* 2022](https://doi.org/10.1103/PhysRevB.105.094402)).

### Quasistatic elastic magnetic scattering

An elastic diffraction measurement that does not resolve a narrow quasielastic
response effectively integrates over its energy dependence. The resulting
equal-time correlation is

$$
S_s^{\alpha\beta}(\mathbf Q)
=
\int_{-\infty}^{\infty}
S_s^{\alpha\beta}(\mathbf Q,E)\,dE.
$$

The energy integral selects the $t=0$ correlation, so
$S_s^{\alpha\beta}(\mathbf Q)$ is the spatial Fourier transform of simultaneous
spin products. It represents an instantaneous snapshot of the spatial
correlations, not necessarily a time-independent configuration. An experiment
averages such snapshots over its finite illuminated volume and over times long
compared with the characteristic fluctuation time. For an equilibrium ergodic
system, this space-time average is equivalent to the ensemble average in the
definition
([Boothroyd, §7.5.1](https://doi.org/10.1093/oso/9780198862314.003.0007)).

When the experimental energy acceptance contains the whole peak and the
magnetic linewidth is small compared with $k_BT$, the classical limit of the
fluctuation--dissipation theorem and the Kramers--Kronig relation give

$$
S_s^{\alpha\beta}(\mathbf Q)
\simeq
k_BT\,\chi'_{s,\alpha\beta}(\mathbf Q,0).
$$

This is commonly called the **static approximation**. “Quasistatic” emphasizes
that the collected response may have a finite linewidth below the experiment's
energy resolution.

nfit uses this quasistatic expression for `single_crystal_elastic` and
`powder_elastic` datasets. It evaluates the model's static susceptibility
directly, avoiding a numerical energy integral, and predicts

$$
\frac{d\sigma}{d\Omega}
\simeq
(\gamma r_0)^2\left(\frac{g}{2}\right)^2 |f(Q)|^2 k_BT
\sum_{\alpha\beta}
(\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta)
\chi'_{s,\alpha\beta}(\mathbf Q,0).
$$

The expression requires a positive dataset temperature and the conditions
above. It describes diffuse elastic or resolution-limited quasielastic
magnetic scattering, rather than magnetic Bragg intensity from a static
ordered moment or the quantum low-temperature limit. Elastic-only data
constrain the static susceptibility but not a model's relaxation rate; combine
them with inelastic data to fit that rate.

### Polarization convention

The tensor projector is
$\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta$. When the response is
represented by a scalar, its meaning determines the polarization factor:

- **One isotropic Cartesian component:** $\chi''=\chi''_{xx}=\chi''_{yy}
  =\chi''_{zz}$ gives $P=2$. nfit's Heisenberg, MMP, and local-relaxational
  models use this convention and return the component in `spin^2/meV`.
- **Isotropic trace:** $\chi''=\sum_\alpha\chi''_{\alpha\alpha}$ gives
  $P=2/3$. This remains available for imported data that were reduced with a
  trace convention.
- **Already corrected:** use $P=1$ only when the stored response has already
  had the polarization contraction removed.
- **Custom scalar:** a positive user-supplied factor is recorded in the
  dataset convention.

For an anisotropic response, the full tensor contraction is required; a single
scalar $P$ is not generally physical.

### Form factor and kinematics

The magnetic amplitude form factor is $f(Q)$, so intensity contains
$|f(Q)|^2$. nfit's tabulated $\langle j_0\rangle$ approximation follows the
International Tables/ILL convention
([Brown, International Tables C §4.4.5](https://www.ill.eu/sites/ccsl/ffacts/)).
It has the form

$$
f(s)=Ae^{-as^2}+Be^{-bs^2}+Ce^{-cs^2}+D,
\qquad
s=\frac{|\mathbf Q|}{4\pi}.
$$

$f$ and $A,B,C,D$ are dimensionless. Since $s$ has units Å$^{-1}$,
$a,b,c$ have units Å$^2$. Evaluating $|\mathbf Q|$ from reciprocal-lattice
coordinates requires the crystal lattice or a UB-derived reciprocal basis.

### Orbital moments and the dipole approximation

$\langle j_0\rangle$ alone is the spin-only form factor. When the moment
carries orbital angular momentum, nfit uses the dipole approximation

$$
f(Q)=\langle j_0(Q)\rangle
+\left(\frac{2}{g_J}-1\right)\langle j_2(Q)\rangle,
\qquad
\langle j_2\rangle(s)=s^2\left[Ae^{-as^2}+Be^{-bs^2}+Ce^{-cs^2}+D\right].
$$

The radial integral is
$\langle j_\ell(Q)\rangle=\int_0^\infty dr\,r^2|R_{\rm ion}(r)|^2j_\ell(Qr)$,
where $r$ is radius in Å, $R_{\rm ion}$ is a radial orbital normalized by
$\int r^2|R_{\rm ion}|^2dr=1$, and $j_\ell$ is a spherical Bessel function.
Specifically $j_0(x)=\sin(x)/x$ and
$j_2(x)=(3/x^3-1/x)\sin(x)-3\cos(x)/x^2$, with their continuous values at zero.
Their arguments $x=Qr$ are dimensionless. The coefficients in the $j_2$
fit are a separate table: its $A,B,C,D$ carry Å$^2$ to cancel $s^2$;
its $a,b,c$ also carry Å$^2$. They are not the $j_0$ coefficients.
For an LS-coupled ion with $J>0$ and electron spin factor approximated by 2,
$g_J=1+[J(J+1)+S(S+1)-L(L+1)]/[2J(J+1)]$;
$L,S,J$ here are angular-momentum quantum numbers, not matrices.

The leading $s^2$ is part of the tabulation and makes $\langle j_2(0)\rangle=0$.
Writing the electron moment as $\boldsymbol\mu=-\mu_B(\langle\mathbf L\rangle+2\langle\mathbf
S\rangle)$ and projecting onto $\mathbf J$ gives
$\langle L\rangle=(2-g_J)J$ and $2\langle S\rangle=2(g_J-1)J$, so the orbital
part weights $\langle j_0\rangle+\langle j_2\rangle$ and the spin part weights
$\langle j_0\rangle$; their ratio is the coefficient above.

$g_J$ is the **ion's Landé factor**, set in each tight-binding orbital's
magnetic form-factor profile. It is deliberately separate from the dataset `g_factor`,
which converts a spin-operator response to a magnetic moment and may be fitted.
The default $g_J=2$ returns $\langle j_0\rangle$ exactly and never consults the
$\langle j_2\rangle$ table, so spin-only models are unaffected. The correction
is not small for rare earths: Yb$^{3+}$ has $g_J=8/7$, giving a
$\langle j_2\rangle$ weight of $0.75$.

The built-in tables cover common $3d$, $4d$, $4f$, and $5f$ magnetic ions
(97 with $\langle j_0\rangle$, 95 of those also with $\langle j_2\rangle$;
Pr$^{3+}$ and O$^{1-}$ are spin-only). They do **not** include any $5d$ ion
(Re, Os, Ir, Pt, W, Ta), Ru$^{2+}$, Ru$^{3+}$, Rh$^{3+}$, or Ce$^{3+}$ — that
is a gap in Section 4.4.5 itself, not a transcription omission, so those ions
need explicit coefficients from the literature. `nfit.available_ions()` and
`nfit.available_dipole_ions()` list what is tabulated.

For direct geometry,

$$
\frac{k_f}{k_i}=\sqrt{\frac{E_i-E}{E_i}},
$$

and for fixed-final-energy indirect geometry,

$$
\frac{k_f}{k_i}=\sqrt{\frac{E_f}{E_f+E}}.
$$

The dataset setting records whether this factor is included in the imported
double-differential cross section or was removed upstream to make
an $S(\mathbf Q,E)$-like quantity. Included data require fixed $E_i$ or $E_f$
for conversion to $\chi''$. This agrees with Mantid's documented `CorrectKiKf`
direction: multiplying a cross section by $k_i/k_f$ removes the phase-space
factor to obtain a dynamic structure factor
([Mantid `CorrectKiKf`](https://docs.mantidproject.org/nightly/algorithms/CorrectKiKf-v1.html)).

For native DGS He-3 tube efficiency, the correction is
$[1-\exp(-a\lambda)]^{-1}$, where final neutron wavelength $\lambda=2\pi/k_f$
is in Å and absorption coefficient $a$ is in Å$^{-1}$. The instrument definition
supplies tube pressure in atm, temperature in K, and wall thickness and cylinder
dimensions in metres. Mantid precision uses source-ordered ray geometry for
supported local cylinders; `high_precision` uses their nominal radius and the
equivalent denominator $-\operatorname{expm1}(-a\lambda)$. These conventions can
give different correction weights; neither alternative alone establishes
physical calibration accuracy. Shape support and fallback behavior are listed
under [numerical reduction policies](data_import.md#numerical-reduction-policies).

## SI bulk susceptibility and magnetization

### Microscopic-to-bulk SI relation

The microscopic spin susceptibility responds to its conjugate Zeeman energy,
whereas rationalized SI bulk susceptibility is defined by
$M=\chi_{\rm SI}H$. Their comparison begins with the zero-frequency
Kramers--Kronig relation for the component parallel to the applied field:

$$
\chi'_{s,ii}(\mathbf Q,0)
=\frac{2}{\pi}\int_0^\infty
\frac{\chi''_{s,ii}(\mathbf Q,E)}{E}\,dE .
$$

Here $E$, $dE$, and the inverse-energy unit of $\chi''_s$ must be expressed
consistently. The index $i$ labels the Cartesian field component. Both sides
have dimensions of inverse energy.
The displayed positive-energy form assumes reciprocity at that momentum;
otherwise the signed-frequency integral is required. It relates dynamic
retarded limits and does not replace a separately specified thermodynamic
static limit for an exactly conserved uniform quantity, as discussed for
[Lindhard susceptibility](lindhard.md#degenerate-transitions-and-the-static-limit).

Bulk susceptibility is the uniform response and therefore requires
$\mathbf Q=0$. It is not obtained by integrating a finite-$\mathbf Q$ neutron
spectrum over momentum. The `local_relaxational`, `mmp_relaxational`,
`generalized_paramagnon`, and `heisenberg_rpa` models evaluate their uniform
static limits and can share response parameters between bulk and neutron
datasets. The `curie_weiss` model instead fits molar susceptibility directly.

For MMP and generalized-paramagnon models, the bulk value is an extrapolation
of a response constructed around one or more finite-$\mathbf Q$ peaks. That
form need not remain quantitatively valid at $\mathbf Q=0$. Comparing the
extrapolation with bulk susceptibility can therefore test the model's
insufficiency; it is optional and does not imply that the finite-$\mathbf Q$
form is expected to describe the bulk response.

For $\chi'_s$ in J$^{-1}$ per magnetic ion and a number density
$n_{\rm mag}$ in m$^{-3}$ of equivalent magnetic ions,

$$
\frac{M_i}{H_i}
=\chi_{{\rm SI},ii}
=\mu_0 n_{\rm mag}(g\mu_B)^2
\chi'_{s,ii}(\mathbf 0,0).
$$

The result is the dimensionless SI volume susceptibility $M/H$. The factor
$\mu_0$ enters because the microscopic Zeeman perturbation couples to $B$,
with $B\simeq\mu_0H$ in the linear, weak-susceptibility limit. For a molar
susceptibility normalized per mole of formula units,

$$
\chi_{{\rm mol},ii}
=\mu_0N_A n_{\rm ion/f.u.}(g\mu_B)^2
\chi'_{s,ii}(\mathbf 0,0),
$$

where $n_{\rm ion/f.u.}$ is the number of equivalent magnetic ions per formula
unit and $\chi'_s$ is again in J$^{-1}$. This equation has units m$^3$/mol.
A susceptibility quoted per mole of magnetic ions omits
$n_{\rm ion/f.u.}$. The corresponding susceptibility volume per ion is

$$
\chi_{\rm SI,ion}
=\mu_0(g\mu_B)^2\chi'_s(\mathbf 0,0)
=\mu_0\mu_B^2\bar\chi'_\mu(\mathbf 0,0),
$$

which has units m$^3$ per magnetic ion, not a dimensionless bulk
susceptibility.

nfit stores energies in meV. Let
$\varepsilon_{\rm meV}=1.602176634\times10^{-22}\ {\rm J}$ be the energy of
1 meV, and let $\widetilde\chi'_s$ and
$\widetilde{\bar\chi}'_\mu$ denote the dimensionless numerical values quoted
in `1/meV` and `mu_B^2/meV`, respectively. The implemented SI conversion is
then

$$
\chi_{{\rm SI},ii}
=\frac{\mu_0n_{\rm mag}(g\mu_B)^2}{\varepsilon_{\rm meV}}\,
\widetilde\chi'_{s,ii}
=\frac{\mu_0n_{\rm mag}\mu_B^2}{\varepsilon_{\rm meV}}\,
\widetilde{\bar\chi}'_{\mu,ii}.
$$

The division by $\varepsilon_{\rm meV}$ converts a numerical value quoted per
meV; it is omitted when the susceptibility still carries its physical
inverse-energy unit.

These equations place $\mu_0$ in the conversion to rationalized-SI $M/H$.
The neutron cross section above uses the microscopic response in
`spin^2/meV` or `mu_B^2/meV` and contains no additional $\mu_0$. The
spin-fluctuation models' rationalized-SI bulk predictions include
$\mu_0(g\mu_B)^2$. Welch *et al.* derive the per-magnetic-ion relation
explicitly. Experimentally, a finite energy window gives only a partial
Kramers--Kronig integral, and finite-$\mathbf Q$ neutron data must be
extrapolated to $\mathbf Q=0$ before comparison with a bulk magnetometer.

### CGS and SI data units

Bulk magnetic data use explicit CGS/SI conversions: one emu of magnetic dipole
moment is $10^{-3}\ {\rm A\,m^2}$; in vacuum, a field reported as
$B=1\ {\rm T}$ corresponds to $H=10^4\ {\rm Oe}$; and molar susceptibility
obeys `1 cm^3/mol (CGS) = 4 pi 10^-6 m^3/mol (SI)`. For a sample with $n$
moles of formula units, a measured dipole moment $m_{\rm sample}$ is converted
to Bohr magnetons per formula unit as

$$
\frac{m_{\rm sample}}
{nN_A\mu_B},
$$

provided $m_{\rm sample}$ and $\mu_B$ are expressed in the same moment unit
(for example emu). The result is the numerical moment in
`mu_B/f.u.`.

For moment channels, $B$ is the signed field component along the dataset's
declared measurement direction. Reversing the field therefore reverses the
linear-response moment. A molar susceptibility prediction does not multiply
by field. Sample mass and formula-unit molar mass are needed when converting
between a sample-total moment and a molar or formula-unit channel; they are
not additional factors in an already molar susceptibility.

## Data representations and normalization

### Channel names and units

Every imported signal and auxiliary channel records a value, one-sigma
uncertainty, physical quantity type, and unit.

- Energy transfer is displayed as $\Delta E$ with its stored unit, normally
  meV. The stable project/script axis key is `DeltaE`.
- Unclassified measured INS signal is displayed as $I(\mathbf Q,E)$.
- An inelastic cross-section channel is displayed as
  $d^2\sigma/(d\Omega\,dE)$, with units such as
  `mbarn/(sr meV f.u.)` (or the selected magnetic-ion/unit-cell basis).
- Dynamical susceptibility is displayed simply as $\chi''$. Magnetic-moment
  units render the Bohr magneton as $\mu_{\mathrm B}^2$, for example
  $\mu_{\mathrm B}^2/(\mathrm{meV\ f.u.})`; `spin^2` is retained when that
  convention is explicitly selected.
- Signal and uncertainty plots always include a unit. Missing or explicitly
  arbitrary units are displayed as `(a.u.)`. Published normalized intensity
  such as `1/(meV atom)` retains that dimensionful, non-cross-section unit.
  Without an absolute cross-section calibration, nfit can apply the Bose,
  form-factor, polarization, and kinematic shape corrections, but the paired
  $\chi''$ channel remains in arbitrary units.

The cross section and dynamic structure factor are related but are not
interchangeable labels. nfit therefore does not call an arbitrary or
cross-section-valued channel $S(\mathbf Q,E)$ unless its metadata explicitly
identifies that response. Mantid likewise defines normalized direct-geometry
inelastic output as the double-differential cross section
$d^2\sigma/(dE\,d\Omega)$, while the neutron-scattering relation contains
$S(\mathbf Q,E)$ as a separate response function
([Mantid MDNorm](https://docs.mantidproject.org/v6.1.0/concepts/MDNorm.html);
[ORNL introduction to neutron spin echo](https://neutrons.ornl.gov/sites/default/files/LS_Introduction_to_NSE_2019NXS-R.pdf)).

### Measurement estimator contracts

The [public measurement statistics API](measurement_statistics.md) declares a
target quantity, compatible unit labels, support, and dependencies before
combination. Explicit contracts select corresponding project and viewer paths;
unmarked saved workflows retain compatibility provenance. Units below are physical declarations; no automatic unit
conversion or instrument-based estimator selection is performed.

For continuous observations $y_i$ of a common value $\mu$, with known positive
independent variances $s_i^2$ in squared observation units, the Gaussian
inverse-variance estimator is

$$
a_i=\frac{1/s_i^2}{\sum_j1/s_j^2},\qquad
\hat\mu=\sum_i a_i y_i,\qquad
\operatorname{Var}(\hat\mu)=\sum_i a_i^2s_i^2.
$$

The $a_i$ are dimensionless. Equal-weight propagation instead uses $a_i=1/n$
for $n$ included observations. Neither estimator multiplies precision weights
by an unrelated exposure. For Gaussian precision weights,
$\chi^2=\sum_i(y_i-\hat\mu)^2/s_i^2$ has $n-1$ degrees of freedom under the
common-value model. A large value diagnoses model/error disagreement; variance
is not automatically inflated. A varying response produces a precision-weighted
response, rather than a uniform coordinate average or a bin-center value.
Correlated observations generally require generalized least squares, not these
diagonal precision weights.

For samples $y_i=f(x_i)$ at strictly increasing coordinates $x_i$, explicitly
assume piecewise linear interpolation. Let $\phi_i(x)$ be its nodal basis
function and $D$ the covered part of the requested interval. Then

$$
b_i=\int_D\phi_i(x)\,dx,\qquad
J=\sum_i b_i y_i,\qquad
\bar f_D=J/|D|.
$$

$b_i$ and $|D|$ have coordinate units; $J$ has observation times coordinate
units. The mean retains observation units. Independent-node variance is
$\sum_i b_i^2s_i^2$ for $J$, divided by $|D|^2$ for the mean. A node shared
by adjacent interpolation segments has one coefficient $b_i$, not independent
errors in each segment. Missing or masked nodes break their adjacent segments;
widely spaced valid nodes still use the selected interpolation. No extrapolation
or interpolation-model uncertainty is inferred. The default requires full
interval support; an explicit covered-only target reports $|D|$ and
$|D|/(x_{\max}-x_{\min})$, where these bounds describe the requested interval.

For linear reconstructed observations, write fluctuations as
$\delta y_i=\sum_k A_{ik}\delta z_k$, where primitive sources $z_k$ with
different IDs are independent and have variance $u_k^2$. $A_{ik}$ carries
observation units divided by source units. For a declared linear combination
$Y=\sum_i a_i y_i$,

$$
\operatorname{Var}(Y)=\sum_k\left(\sum_i a_i A_{ik}\right)^2u_k^2.
$$

Repeated source coefficients are combined before squaring. Signed coefficients
can cancel shared uncertainty; distinct sources do not cancel by identity.
The sparse `SourceTerm` representation must reproduce each observation's
diagonal variance and include every represented dependency. It does not infer
missing correlations or declare different calibration sources independent
unless the contract makes that assumption. Required reconstruction terms reject
missing data by default; omission explicitly changes the target to available
terms.

For pooled count numerator $C$, exposure $N>0$, numerator variance $V_C$,
exposure variance $V_N$, and covariance $K_{CN}$, first-order ratio propagation
gives

$$
\operatorname{Var}(C/N)\simeq
\frac{V_C}{N^2}+\frac{C^2V_N}{N^4}-\frac{2CK_{CN}}{N^3}.
$$

$N$ has numerator units divided by intensity units, and $K_{CN}$ has numerator
times exposure units. Shared source sensitivities propagate this joint Jacobian
without subtracting inconsistent marginal errors. A common normalizer error does
not decrease as independent repeated noise. This delta-method variance is
approximate, conditional on the supplied model, and unsuitable as an exact
confidence interval near poorly constrained denominators. It does not establish
that a plug-in ratio is unbiased or optimal. Known exposure sets $V_N=K_{CN}=0$.

Additive continuous-mean payloads retain $S=\sum_i w_i y_i$,
$Q=\sum_i w_i^2s_i^2$, $W=\sum_i w_i$, and the number of observations.
The mean is $S/W$ and its independent variance is $Q/W^2$. The $w_i$ are
dimensionless, with a recorded reference scale for precision weighting.
Combining separately initialized precisions first reconciles that scale.
Shared-source propagation replaces the diagonal expression by the source
coefficient formula above. Uncertain count normalization instead retains both
numerator and exposure primitives; rate sensitivities alone cannot recover the
variance of a differently pooled ratio.

### Declared fitting likelihoods

Independent Gaussian fitting uses $(y_i-f_i)/s_i$ for model predictions $f_i$,
with positive supplied standard deviations $s_i$. For represented covariance
$\Sigma$ in squared observation units, generalized least squares (GLS) minimizes
$\mathbf r^T\Sigma^{-1}\mathbf r$, where $r_i=y_i-f_i$. nfit factors a bounded
selected covariance; singular dependence requires a primitive-source fit.

For explicitly audited independent integer counts $n_i$ with known exposure
$N_i$ and constant dimensionless event weight $w>0$, corrected numerator is
$C_i=w n_i$ and predicted count is $\lambda_i=N_i f_i/w$. Poisson deviance is

$$
D=2\sum_i\left[\lambda_i-n_i+n_i\log(n_i/\lambda_i)\right].
$$

Counts and $\lambda_i$ are dimensionless; $f_i$ has intensity units and $N_i$
has corrected numerator per intensity units. The logarithmic term is zero when
$n_i=0$, so a measured empty cell contributes $2\lambda_i$. Positive counts with
zero prediction have infinite deviance. Corrected weighted events or reused
symmetry copies do not automatically satisfy this model. Reported Poisson fit
parameter covariance uses expected Fisher information; its asymptotic standard
errors are distinct from exact low-count intervals. Dataset importance weights
do not create extra observations or calibrated likelihood-ratio probabilities.

### Pooling normalized count histograms

Let $C_i$ be the corrected count numerator in cell $i$, $N_i>0$ its known
exposure denominator (charge times detector-trajectory normalization in the
native direct-geometry path), and $I_i=C_i/N_i$ its normalized intensity.
Integration of these measured cells reports

$$
I_{\mathrm{pool}}=\frac{\sum_i C_i}{\sum_i N_i}
=\frac{\sum_i N_i I_i}{\sum_i N_i},\qquad
\sigma_{\mathrm{pool}}^2=\frac{\sum_i N_i^2\sigma_i^2}{(\sum_i N_i)^2}.
$$

$N_i$ has units of count numerator divided by intensity; its absolute scale
cancels from the pooled intensity. $\sigma_i$ is the stored one-sigma intensity
uncertainty. The variance formula propagates independent stored diagonal
variances and treats exposure as known. It does not reconstruct covariance
from fractional event sharing or repeated symmetry copies. Measured zero-count
cells contribute their exposure and stored uncertainty; unmeasured or masked
cells do not contribute.

For independent uncorrected counts with a common intensity, summing counts and
exposures retains the information needed for a Poisson likelihood. A zero-count
measurement contributes exposure. Evaluate a confidence interval after pooling;
do not add confidence endpoints as variances. If intensity varies within the
region, pooling estimates an exposure-weighted average of those intensities.
A uniform spatial average or integral is a different target: it requires
geometric weights and assumptions about unsampled portions. Sparse coverage
cannot establish those portions by itself.

Uniform weighting coincides with exposure pooling only for equal exposure.
For uncorrected Poisson counts with a common underlying intensity $I$,
$\operatorname{Var}(I_i)=I/N_i$, so weights based on the *expected* inverse
variance are proportional to $N_i$. Weights estimated from observed counts
fluctuate with the counts, can suppress zero-count measurements, and need not
produce the pooled estimator. Event corrections and non-Poisson uncertainties
also break that simple equivalence. Summing normalized intensities,
$\sum_i I_i$, is a different quantity and overweights weakly exposed cells.

#### Covered empty cells and confidence intervals

Native DGS and MDE histograms store additive count numerator $C$, numerator
variance $V$, and known exposure $N$ as immutable channels. For independent
corrected events, $C=\sum_j w_j$ and $V=\sum_j v_j$, where $w_j$ is the
dimensionless corrected weight and $v_j$ its separately stored variance. For
known correction factors applied to unit-count events, $v_j=w_j^2$ before
storage rounding. Mantid precision rounds weight and variance independently,
so the stored variance must not be reconstructed by squaring the stored weight.
The observed event standard error is $\sqrt{V}/N$. A covered empty cell has
$C=V=0$ and positive $N$. Zero observed variance does not establish certainty
about its unknown intensity. A barely exposed empty fringe cell can have a
large confidence upper endpoint even without pooling.

Confidence endpoints are separate from standard errors. For independent
integer counts $n$, a known constant positive weight $w$, and known exposure
$N$, `nfit.histogram_statistics.poisson_rate_interval` returns the central
Garwood interval at confidence $c$. Let $\alpha=1-c$ and $\chi^2_{\nu,p}$ be the
$p$ quantile of a chi-squared distribution with $\nu$ degrees of freedom:

$$
I_{\mathrm{lower}}=\frac{w}{2N}\chi^2_{2n,\alpha/2},\qquad
I_{\mathrm{upper}}=\frac{w}{2N}\chi^2_{2(n+1),1-\alpha/2}.
$$

The lower endpoint is zero for $n=0$. At $c=0.682689492137$, the zero-count
upper endpoint is approximately $1.841w/N$. This is an equal-tailed interval,
not the previous Feldman–Cousins endpoint convention. Compute it once from the
final pooled observation. Heterogeneous correction weights, correlated copies,
signed backgrounds, and uncertain exposure require a different likelihood;
this API does not infer that they satisfy the Poisson model.
See [Garwood's original construction](https://doi.org/10.1093/biomet/28.3-4.437).

#### Event copies and covariance

Copies of one event are perfectly correlated. If copies with coefficients
$a_j$ fall in one output bin, their contribution to its variance is
$V_{\mathrm{source}}(\sum_j a_j)^2$, including the cross terms
$2a_ja_kV_{\mathrm{source}}$. Native raw-DGS and MDE binning defaults to
`independent_copies`, which follows Mantid's diagonal convention: two identical
unit copies contribute $2V_{\mathrm{source}}$. This convention does not make
copies physically independent. The optional `within_bin_covariance` policy adds
these cross terms using the exact signal-kernel bin assignments, giving
$4V_{\mathrm{source}}$ for the same two copies in one final bin.
Event contributions are counted separately under either policy.

The default event histogram does **not** retain covariance between different bins.
If symmetry copies land in separate bins that are later integrated together,
adding stored diagonal variances misses those cross terms. Fractional histogram
assignment, shared monitors/vanadium, reused backgrounds, and CORELLI
reconstruction also require dependencies beyond this representation. Metadata
records the selected policy and, for the covariance option, corrected within-bin
pairs and unrepresented cross-bin pairs. Do not
interpret diagonal pooling as exact uncertainty for correlated cells. An optional
`SourceDependencies` payload can retain represented cross-bin sources through
subsequent aggregation; current native DGS caches do not construct this payload
for every event or reconstruct shared detector-calibration uncertainty.
Rebinning cached original events directly onto the final requested grid with
`within_bin_covariance` recovers same-event covariance within those final bins
without a dense matrix. It does not supply the covariance needed when subsequent
operations combine different bins.

Native CORELLI finite-energy reconstruction currently accumulates fractional
and symmetry contributions separately. Two identical copies of one reconstructed
event contribute twice the diagonal variance; physically their shared-event
variance is four times the single-copy variance. Different energy hypotheses
also share measured neutrons. If signed reconstruction weights cancel, omitted
covariance can instead overstate the uncertainty of a final cut. CORELLI has no
native complete source-dependency payload or corresponding DGS
`within_bin_covariance` treatment. Its charge/duty normalization and optional
pointwise corrections are separate from four-dimensional trajectory exposure.

The diagnostics `benchmarks/benchmark_histogram_uncertainty.py` and
`benchmarks/benchmark_uncertainty_paths.py` separate event variance, exposure,
confidence inference, and estimator behavior. Their unit tests neither import
nor execute Mantid or Shiver. Full estimator and dependency propagation is
tracked in the [measurement pipeline plan](measurement_pipeline_plan.md).

### Derived channels, calibration, and normalization

For count data, `Signal units / mbarn` gives the imported signal units per
`mbarn/(sr meV)` on the selected sample basis; zero denotes an uncalibrated
signal.

The imported signal is retained as a named channel. When temperature is known,
the paired scattering-cross-section and $\chi''$ channels are generated with
propagated one-sigma errors. Each standard error is multiplied by the absolute
value of the same pointwise conversion Jacobian as its signal, including the
Bose, form-factor, polarization, Landé-factor, calibration, and kinematic
terms. `Plot and fit` chooses the primary observable; both remain independently
selectable with their error bars in the data viewer. Models read the
primary channel's quantity and unit and evaluate in that representation.

The data normalization basis is metadata, not a hidden correction to imported
values. Beam-flux and illuminated-sample calibration must already refer to the
same formula-unit, magnetic-ion, or unit-cell basis selected in the GUI. A
microscopic model may nevertheless have a different native basis. In
particular, a Lindhard calculation is extensive per electronic model cell, so
nfit explicitly divides the model prediction by the resolved formula units or
represented reference magnetic centers per model cell before comparing it with
the declared dataset basis. That model conversion is recorded separately from
the data calibration and from the magnetic form factor. For other per-atom
data, an optional element or site label changes the displayed normalization
suffix; it does not multiply or divide the imported data.
These definitions and the distinction between correlation functions,
$\chi''$, cross section, and absolute normalization follow general neutron
scattering references rather than any material-specific paper
([Squires, chapters 7-8](https://doi.org/10.1017/CBO9781139107808.009);
[Xu, Xu, and Tranquada 2013](https://doi.org/10.1063/1.4818323)).

## Tensor (anisotropic) interactions

These apply when the `heisenberg_rpa` model is run with anisotropic exchange,
single-ion anisotropy, dipole–dipole, or Zeeman terms (see
[Heisenberg RPA](heisenberg_rpa.md#tensor-interactions)).

- **Polarization contraction.** The unpolarized channel uses
  $W_{\alpha\beta} = \delta_{\alpha\beta} - \hat Q_\alpha \hat Q_\beta$
  with $\hat Q$ the Cartesian unit momentum transfer (from
  `rlu_to_inv_angstrom_matrix`, stamped on each fit point). The isotropic limit
  reduces to the scalar model's $P = 2$ one-component polarization factor,
  keeping the intensity continuous at the isotropic limit.
- **Dissipative tensor.** The tensor response is
  $\chi''_{\alpha\beta} = (\chi_{\alpha\beta} - \chi^*_{\beta\alpha})/2i$; its
  off-diagonal entries can be complex. A Zeeman field breaks time reversal;
  Dzyaloshinskii--Moriya exchange is itself time-reversal even and may permit
  momentum-dependent chiral correlations without a field. A real symmetric
  unpolarized projector does not measure the antisymmetric part.
- **Polarization metadata.** The per-dataset schema is
  `dataset.parameters["polarization"]` = `{direction, frame, channel}`.
- **Rank-2 tensors and improper operations.** Spin–spin tensors transform as
  $T \to R\,T\,R^{\mathsf T}$ with the *proper or improper* Cartesian rotation
  $R = L\,R_{\text{frac}}\,L^{-1}$ of the symmetry op; the two axial-vector
  $\det R$ factors cancel, so no sign flip is applied for improper ops. A bond
  reversed by its generating op contributes $R\,T^{\mathsf T}R^{\mathsf T}$.
  Here $T$ is the Cartesian interaction tensor, $R_{\rm frac}$ is a fractional
  crystallographic symmetry operation, and $L$ maps fractional direct-space
  vectors to Cartesian coordinates.
- **Field frames.** The applied field is entered as a magnitude in tesla plus a
  direction given as either a direct $[u\,v\,w]$ vector (converted with the
  direct lattice matrix) or a reciprocal $(H\,K\,L)$ vector (converted with the
  reciprocal matrix), then normalized to a Cartesian unit vector. In a cubic
  cell $[1\,1\,1]$ and $(1\,1\,1)$ coincide; in lower symmetry they differ.
- **Units.** Field $B$ is in tesla. The Larmor angular frequency obeys
  $\hbar\omega_L=g\mu_BB$; nfit stores the corresponding Larmor energy
  $E_L=\hbar\omega_L$ in meV, using
  $\mu_B=0.05788\,\text{meV/T}$. The dipole strength $D_{\mathrm{dip}}$ is in
  meV·Å³, with physical default $(\mu_0/4\pi)(g\mu_B)^2$; this product has
  dimensions energy times volume.
- **Sign convention.** The interaction matrix is defined by
  $H=-\tfrac12\sum_{ij,\alpha\beta}S_i^\alpha\mathbb J_{ij}^{\alpha\beta}S_j^\beta$, so positive
  couplings favour the ordering where the largest eigenvalue
  $\lambda_{\max}(\mathbf{Q})$ of $\mathbb{J}(\mathbf{Q})$ peaks; the RPA
  instability is at $\lambda_{\max}\chi_0 \to 1$.
  $\lambda_{\max}$ has energy units and $\chi_0$ inverse-energy units, so their
  product is dimensionless. With a self-consistency closure, the denominator is
  $1-[\lambda_\nu(\mathbf Q)-\lambda_{\rm shift}]\chi_{0,\rm eff}$.
  `chi0_eff` is the local susceptibility used by the response, and the Onsager
  reaction field `lambda_shift` is an energy subtracted from every interaction
  eigenvalue. nfit reports the smallest sampled denominator as
  `stability_margin`. This is the same sign convention as the scalar model.
- **Field-on stability.** If the static local response is anisotropic, define
  $C=X_0(0)$ in meV$^{-1}$ and
  $C_N=I_N\otimes C$ for $N$ magnetic sites. The dimensionless Hermitian
  feedback matrix is
  $F(\mathbf Q)=C_N^{1/2}[\mathbb J(\mathbf Q)-\lambda_{\rm shift}I_{3N}]C_N^{1/2}$.
  Stability requires every eigenvalue $\rho_\nu(F)<1$, and
  `stability_margin` is $1-\max_{\mathbf Q,\nu}\rho_\nu$ over the sampled
  momenta. This reduces to the scalar expression only when $C=\chi_0 I_3$.
  The static bulk tensor is the uniform-site contraction of
  $[C_N^{-1}-\mathbb J(\mathbf0)+\lambda_{\rm shift}I_{3N}]^{-1}$ divided
  by $N$. A longitudinal projection can depend on transverse local
  susceptibilities when exchange mixes Cartesian components.
- **Dipole sign.** The dipolar Hamiltonian is
  $H=+\tfrac12 D_{\mathrm{dip}}\sum_{i\neq j}\mathbf S_i\,\mathbb
  T(\mathbf r_{ij})\,\mathbf S_j$ with
  $\mathbb T=(\delta_{\alpha\beta}-3\hat r_\alpha\hat r_\beta)/r^3$, which is
  the *opposite* sign to the exchange convention above. nfit therefore assembles
  $-\mathbb T(\mathbf Q)$ into $\mathbb J(\mathbf Q)$, so a **positive**
  $D_{\mathrm{dip}}$ is the physical point-dipole interaction and favours
  head-to-tail alignment along the shortest lattice direction.

In the tensor Hamiltonians, $i,j$ label sites including their cell translations,
and $\alpha,\beta=x,y,z$. For dipoles,
$\mathbf r_{ij}=\mathbf R_j-\mathbf R_i$, $r=|\mathbf r_{ij}|$ in Å,
$\hat{\mathbf r}=\mathbf r_{ij}/r$, and $\mathbb T$ has units Å$^{-3}$.
The half factors remove double counting of ordered site pairs.
An antisymmetric exchange matrix can be specified without sign ambiguity by
$H_{\rm DM}=\sum_{i<j}\mathbf D_{ij}\cdot(\mathbf S_i\times\mathbf S_j)$;
with the negative exchange sign above this means
$\mathbb J_{ij}^{\alpha\beta}=-\sum_\gamma\epsilon_{\gamma\alpha\beta}D_{ij}^\gamma$.
$\mathbf D_{ji}=-\mathbf D_{ij}$ and $\mathbf D$ has units meV. The GUI's fitted
tensor coefficients multiply displayed invariant matrices; convert those
matrices using this equation before assigning a named DM-vector component.
For an onsite symmetric matrix $\mathbb J_{ii}$, its contribution is
$-\tfrac14\sum_{\alpha\beta}\mathbb J_{ii}^{\alpha\beta}\{S_i^\alpha,S_i^\beta\}$,
where $\{X,Y\}=XY+YX$. Thus an anisotropy written as $K(S_i^z)^2$ corresponds
to $\mathbb J_{ii}^{zz}=-2K$ in this Hamiltonian convention.

## Electronic-response conventions

- **Cartesian spin response.** `bare_spin_susceptibility` returns the
  susceptibility of the dimensionless spin operator. For an implicit-spin model
  it evaluates the spin trace analytically, giving
  $\chi^0_{s,\alpha\alpha}(\mathbf0,0)=D_{\uparrow,T}(\mu)/2$ for intrinsic
  total spin. $D_{\uparrow,T}$ is the per-spin DOS averaged with $-df/d\varepsilon$,
  in states/(meV model cell); the zero-temperature implementation uses an
  $\eta$-broadened derivative. Only the converged zero-temperature, zero-width
  limit becomes the unsmeared $D_\uparrow(\mu)/2$. See the
  [Lindhard static limit](lindhard.md#degenerate-transitions-and-the-static-limit).
- **RPA vertices.** The vertex conjugate to $S^\alpha$ is twice an on-site
  density interaction, so `stoner_rpa` uses $\Gamma=2I$ and its instability is
  the textbook $I D_\uparrow(\mu)=1$. `hubbard_hund_rpa` dresses the
  orbital-pair response before the spin trace and so uses $U$ directly; the two
  interaction parameters are on the same scale. See
  [Scalar Stoner RPA](stoner_rpa.md).
- **Density of states.** `density_of_states` includes the model's spin
  degeneracy: an implicit-spin total integrates to $2N_{\rm basis}$ states per
  primitive cell and matches the electron count from `electron_filling`. A
  collinear or spinor model already carries spin in its basis and uses a
  degeneracy of one. The factor is recorded as `provenance["spin_degeneracy"]`.

Model-specific form-factor and exchange symbols are defined in
[Spin-fluctuation models](spin_fluctuation_models.md) and
[Heisenberg RPA](heisenberg_rpa.md).
