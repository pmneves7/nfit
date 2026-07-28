# Physics conventions

## Inelastic magnetic neutron scattering

nfit uses energy transfer $E = E_i-E_f=\hbar\omega$ in meV; positive $E$
means neutron energy loss. Momentum is in reciprocal-lattice units or
Å$^{-1}$, temperature is in K, and a double-differential cross section is per
steradian and per meV. Model kernels produce the dissipative response
$\chi''(\mathbf Q,E)$ before instrumental factors.

Common symbols are:

| symbol | meaning |
| --- | --- |
| $E_i,E_f,E$ | incident, final, and transferred neutron energy |
| $\mathbf k_i,\mathbf k_f$ | incident and final neutron wavevectors; $k_i,k_f$ are their magnitudes |
| $\mathbf Q=\mathbf k_i-\mathbf k_f$ | momentum transfer; $Q=|\mathbf Q|$ and $\hat{\mathbf Q}=\mathbf Q/Q$ |
| $T$, $k_B$ | absolute temperature and Boltzmann constant |
| $\omega$, $\hbar$ | angular frequency and reduced Planck constant, with $E=\hbar\omega$ |
| $g$, $\mu_B$, $\mu_0$ | dimensionless Landé factor, Bohr magneton, and vacuum permeability |
| $B$, $H$, $M$ | magnetic flux density, magnetic field strength, and magnetization |
| $N_A$ | Avogadro constant |
| $\alpha,\beta$ | Cartesian components $x,y,z$ |

### Definition of the dynamic spin correlation function

nfit follows the Fourier-transform convention in Squires, chapter 7,
Eq. 7.73. For equivalent magnetic ions at equilibrium positions
$\mathbf R_l$, let $\mathbf S=\hat{\mathbf J}/\hbar$ denote the dimensionless
spin operator. The dynamic spin correlation function **per magnetic ion** is

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

Thus $S^{\alpha\beta}_s$ has physical dimensions of inverse energy and is
labelled `spin^2/meV` when $E$ is in meV. Here `spin^2` records the operator
normalization; it is not an additional SI dimension.
The phase signs, operator order, and the factor $1/(2\pi\hbar)$ are all part
of the convention. The phase $\mathbf Q\cdot\mathbf R_l$ is dimensionless:
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
$-h^\beta S^\beta$. For a magnetic field, $h$ contains the factor $g\mu_B B$.
To first order,

$$
\delta\langle S^\alpha(\mathbf Q,E)\rangle
=
\sum_\beta \chi_{s,\alpha\beta}(\mathbf Q,E)
h^\beta(\mathbf Q,E).
$$

The retarded susceptibility is causal. Its real part $\chi'_s$ is the
in-phase, reactive response; $\chi'_s(\mathbf Q,0)$ is the static
susceptibility. Its imaginary part $\chi''_s$ is the out-of-phase,
dissipative response measured by inelastic scattering. “Imaginary” refers to
the phase of the response function, not to an imaginary-valued spin. With
dimensionless spin operators, both parts have units `spin^2/meV`.

For a bulk measurement one may instead write
$\delta M_\alpha=\sum_\beta\chi^{\alpha\beta}_{MH}\delta H_\beta$.
That SI or molar susceptibility includes magnetic-moment, amount-of-sample,
and field-unit conversions; it is not numerically identical to $\chi_s$.
The conversion used by nfit is given later on this page under the relation to
SI susceptibility.

In nfit's per-unit-energy convention, the fluctuation--dissipation theorem is

$$
S_s^{\alpha\beta}(\mathbf Q,E)=
\frac{1}{\pi}\,
\frac{\chi''_{s,\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_BT)]}.
$$

The $1/\pi$ appears exactly once and is not part of $\chi''$ itself. The
Bose/detailed-balance denominator is evaluated stably near $E=0$. The
odd-in-$E$ dissipative response then gives the correct balance between
neutron energy loss and gain.

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

The scalar spin-fluctuation kernels return a **single Cartesian component** of
the spin-operator susceptibility,
$\chi''_s=\chi''_{s,xx}=\chi''_{s,yy}=\chi''_{s,zz}$, in
`spin^2/meV`. For an unpolarized measurement in the dipole approximation, nfit
uses

$$
\frac{d^2\sigma}{d\Omega\,dE} =
\frac{k_f}{k_i}\frac{(\gamma r_0)^2}{\pi}
\left(\frac{g}{2}\right)^2 |f(Q)|^2
\sum_{\alpha\beta}
(\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta)
\frac{\chi''_{s,\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_B T)]},
$$

where $\chi''_{s,\alpha\beta}$ is the dimensionless-spin response per unit
energy. The kinematic ratio, form factor, projector, $g$, and Bose denominator
are dimensionless, so the area coefficient times $\chi''_s$ gives cross
section per energy, as required.
Since

$$
(\gamma r_0)^2\left(\frac{g}{2}\right)^2
=\left(\frac{\gamma r_0}{2}\right)^2g^2,
\qquad
\left(\frac{\gamma r_0}{2}\right)^2=0.07265\ {\rm barn},
$$

the spin-response form uses the area coefficient
$0.07265\,g^2\ {\rm barn}$. Equivalently, when a *physical* moment response
is used,

$$
C_\mu
=\frac{(\gamma r_0/2)^2}{\mu_B^2}
=0.07265\ \frac{{\rm barn}}{\mu_B^2},
$$

so $C_\mu\chi''_{\rm moment}$ has units barn per energy. These are the same
physical normalization written in two unit conventions. The code multiplies
numerical values in `spin^2/meV` by $g^2$; values already expressed in
`mu_B^2/meV` receive no additional $g^2$.

See Berk's NIST review for the general double-differential and magnetic
correlation-function formalism,
Squires for the magnetic cross section and linear-response convention, and
Welch *et al.* for an explicit absolute-unit derivation
([Berk 1993](https://doi.org/10.6028/jres.098.002);
[Squires, chapter 7](https://doi.org/10.1017/CBO9781139107808.008);
[Welch *et al.* 2022](https://doi.org/10.1103/PhysRevB.105.094402)).

### Quasistatic elastic magnetic scattering

An elastic diffraction measurement can collect a narrow quasielastic response
without resolving its energy dependence. The measured equal-time correlation
is then

$$
S_s^{\alpha\beta}(\mathbf Q)
=
\int_{-\infty}^{\infty}
S_s^{\alpha\beta}(\mathbf Q,E)\,dE.
$$

If the magnetic linewidth is small compared with $k_BT$ and the experimental
energy acceptance contains the whole peak, the classical limit of the
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

The approximation requires a positive dataset temperature. It describes diffuse
elastic or resolution-limited quasielastic magnetic scattering, not magnetic
Bragg intensity from a static ordered moment. It is also inappropriate when
the linewidth is comparable to $k_BT$, when the instrument accepts only part
of the peak, or in the quantum low-temperature limit. Elastic-only data
constrain the static susceptibility but not a model's relaxation rate; combine
them with inelastic data if that rate is to be fitted.

### Polarization convention

The tensor projector is always
$\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta$. A scalar polarization factor
depends on what the scalar $\chi''$ means:

- **One isotropic Cartesian component:** $\chi''=\chi''_{xx}=\chi''_{yy}
  =\chi''_{zz}$ gives $P=2$. This is the scalar convention used by nfit's
  Heisenberg, MMP, and local-relaxational models.
- **Isotropic trace:** $\chi''=\sum_\alpha\chi''_{\alpha\alpha}$ gives
  $P=2/3$. This remains available for imported data that were reduced with a
  trace convention.
- **Already corrected:** use $P=1$ only when the stored response has already
  had the polarization contraction removed.
- **Custom scalar:** a positive user-supplied factor is recorded in the
  dataset convention.

For an anisotropic response, the full tensor contraction is required; a single
scalar $P$ is not generally physical.

### Magnetic moment, spin, and the Landé factor

The Landé factor $g$ is **dimensionless**. It is not measured in Bohr
magnetons. The magnetic moment operator is
$\boldsymbol\mu=-g\mu_B\mathbf S$, so $g\mu_B$ carries moment units while
$g$ alone does not. The dimensionally complete conversion from the spin
response to the physical magnetic-moment response is

$$
\chi''_{\rm moment}(\mathbf Q,E)
=(g\mu_B)^2\chi''_s(\mathbf Q,E).
$$

Here `spin^2/meV` records the operator convention; because $\mathbf S$ is
defined as $\hat{\mathbf J}/\hbar$, `spin^2` is dimensionless. Define the
unit-normalized moment response

$$
\bar\chi''_\mu
\equiv \frac{\chi''_{\rm moment}}{\mu_B^2}.
$$

It has dimensions of inverse energy, and its numerical ordinate is what nfit
labels in `mu_B^2/meV`. The physical and unit-normalized conversions are
therefore

$$
\chi''_{\rm moment}
=\mu_B^2\bar\chi''_\mu
=(g\mu_B)^2\chi''_s,
\qquad
\bar\chi''_\mu=g^2\chi''_s.
$$

The last equality is dimensionally valid because both sides have dimensions
of inverse energy; it is also the $g^2$ conversion between numerical ordinates
in the two declared unit conventions. The nfit parameter `g_factor` remains
unitless. This explains the factor $(g/2)^2$ in Squires Eq. 7.73: together
with the magnetic scattering constant, it converts the dimensionless-spin
response to a magnetic-moment response.

If imported $\chi''$ is expressed in `mu_B^2/meV`, it is a
**magnetic-moment** susceptibility and already contains the moment conversion.
nfit therefore applies no additional $g^2$ when converting that imported
channel. If the imported response is instead declared in `spin^2/meV`, nfit
multiplies its numerical values by $g^2$ exactly once and changes the declared
unit to `mu_B^2/meV` when calculating the cross section. This is the numerical
unit conversion defined above; physically, the factor is $(g\mu_B)^2$.

The model kernels themselves are spin responses. When a dataset requests a
model curve in `mu_B^2/meV`, nfit multiplies the model by the dataset's Landé
$g^2$; when it requests `spin^2/meV`, it does not. This explicit choice avoids
both omitting $g^2$ and double counting it. The user-set Landé factor may differ
from 2.

### Relation to SI susceptibility and the role of $\mu_0$

The microscopic spin susceptibility above responds to the conjugate Zeeman
energy. Rationalized SI bulk susceptibility instead uses $M=\chi_{\rm SI}H$.
The zero-frequency Kramers--Kronig relation for the component parallel to the
applied field is

$$
\chi'_{s,ii}(\mathbf Q,0)
=\frac{2}{\pi}\int_0^\infty
\frac{\chi''_{s,ii}(\mathbf Q,E)}{E}\,dE .
$$

Here $E$, $dE$, and the inverse-energy unit of $\chi''_s$ must be expressed
consistently. The index $i$ labels the Cartesian field component. Both sides
have dimensions of inverse energy.

Bulk susceptibility is the uniform response, so it requires
$\mathbf Q=0$. If $\chi'_s$ is expressed in J$^{-1}$ per magnetic ion, then
for a number density $n_{\rm mag}$ in m$^{-3}$ of equivalent magnetic ions,

$$
\frac{M_i}{H_i}
=\chi_{{\rm SI},ii}
=\mu_0 n_{\rm mag}(g\mu_B)^2
\chi'_{s,ii}(\mathbf 0,0).
$$

The right-hand side is dimensionless, as required for the SI volume
susceptibility $M/H$. The factor $\mu_0$ enters because the microscopic
Zeeman perturbation couples to $B$, with $B\simeq\mu_0H$ in the linear,
weak-susceptibility limit. For a molar susceptibility normalized per mole of
formula units,

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

The explicit division by $\varepsilon_{\rm meV}$ belongs only in this
unit-stripped numerical-value equation. It must not be combined with a
susceptibility that still carries its physical inverse-energy unit.

The factor $\mu_0$ is therefore required for rationalized-SI $M/H$ units. It
is **not** an additional factor in the neutron cross section once $\chi''$ is
already in `spin^2/meV` or `mu_B^2/meV`; inserting it there would mix SI bulk
and microscopic neutron conventions. nfit's molar
`cm^3/mol` to `m^3/mol` conversion includes the rationalized-SI $4\pi$ factor,
and the Heisenberg RPA bulk prediction includes the corresponding
$\mu_0(g\mu_B)^2$ conversion. Welch *et al.* derive this relation explicitly
for susceptibility per magnetic ion. Experimentally, a finite energy window
gives only a partial Kramers--Kronig integral, and finite-$\mathbf Q$ neutron
data must be extrapolated to $\mathbf Q=0$ before comparison with a bulk
magnetometer.

### Form factor and kinematics

The magnetic **amplitude** form factor is $f(Q)$, so intensity contains
$|f(Q)|^2$. nfit's tabulated $\langle j_0\rangle$ approximation follows the
International Tables/ILL convention
([Brown, International Tables C §4.4.5](https://www.ill.eu/sites/ccsl/ffacts/)).

For direct geometry,

$$
\frac{k_f}{k_i}=\sqrt{\frac{E_i-E}{E_i}},
$$

and for fixed-final-energy indirect geometry,

$$
\frac{k_f}{k_i}=\sqrt{\frac{E_f}{E_f+E}}.
$$

The dataset setting records whether this factor is still **included** in the
imported double-differential cross section or was **removed upstream** to make
an $S(\mathbf Q,E)$-like quantity. Included data require fixed $E_i$ or $E_f$
for conversion to $\chi''$. This agrees with Mantid's documented `CorrectKiKf`
direction: multiplying a cross section by $k_i/k_f$ removes the phase-space
factor to obtain a dynamic structure factor
([Mantid `CorrectKiKf`](https://docs.mantidproject.org/nightly/algorithms/CorrectKiKf-v1.html)).

### Units, normalization, and channels

Every imported MDHisto signal and auxiliary channel follows the same contract
as bulk-susceptibility and heat-capacity data: value, one-sigma uncertainty,
physical quantity type, and unit.

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
  arbitrary units are displayed as `(a.u.)`.
- Published normalized intensity such as `1/(meV atom)` is retained as a
  dimensionful but non-cross-section signal. Its paired $\chi''$ channel has
  the correct Bose/correction shape but remains arbitrary unless an absolute
  cross-section calibration is supplied.
- Uncalibrated data remain `arb. units`. nfit can remove or apply the Bose,
  form-factor, polarization, and kinematic **shape**, but does not label the
  result absolute.
- For count data, `Signal units / mbarn` is the explicit calibration in
  imported signal units per `mbarn/(sr meV)` on the selected sample basis.
  Zero means uncalibrated.

The cross section and dynamic structure factor are related but are not
interchangeable labels. nfit therefore does not call an arbitrary or
cross-section-valued channel $S(\mathbf Q,E)$ unless its metadata explicitly
identifies that response. Mantid likewise defines normalized direct-geometry
inelastic output as the double-differential cross section
$d^2\sigma/(dE\,d\Omega)$, while the neutron-scattering relation contains
$S(\mathbf Q,E)$ as a separate response function
([Mantid MDNorm](https://docs.mantidproject.org/v6.1.0/concepts/MDNorm.html);
[ORNL introduction to neutron spin echo](https://neutrons.ornl.gov/sites/default/files/LS_Introduction_to_NSE_2019NXS-R.pdf)).

The imported signal is retained as a named channel. When temperature is known,
the paired scattering-cross-section and $\chi''$ channels are generated with
propagated one-sigma errors. Each standard error is multiplied by the absolute
value of the same pointwise conversion Jacobian as its signal, including the
Bose, form-factor, polarization, Landé-factor, calibration, and kinematic
terms. `Plot and fit` chooses the primary observable; both remain independently
selectable with their error bars in the data viewer. Models read the
primary channel's quantity and unit and evaluate in that representation.

The normalization basis is metadata, not a hidden atom-count conversion.
Beam-flux and illuminated-sample calibration must already refer to the same
formula-unit, magnetic-ion, or unit-cell basis selected in the GUI.
For per-atom data, an optional element or site label changes the displayed
normalization suffix; it does not multiply or divide the data.
These definitions and the distinction between correlation functions,
$\chi''$, cross section, and absolute normalization follow general neutron
scattering references rather than any material-specific paper
([Squires, chapters 7-8](https://doi.org/10.1017/CBO9781139107808.009);
[Xu, Xu, and Tranquada 2013](https://doi.org/10.1063/1.4818323)).

### Bulk-data unit conversions

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

The model-specific form-factor and exchange symbols are defined in
[Spin-fluctuation models](spin_fluctuation_models.md) and
[Heisenberg RPA](heisenberg_rpa.md).

## Tensor (anisotropic) interactions

These apply when the `heisenberg_rpa` model is run with anisotropic exchange,
single-ion anisotropy, dipole–dipole, or Zeeman terms (see
[Heisenberg RPA](heisenberg_rpa.md#tensor-interactions)).

- **Polarization weight.** The unpolarized channel uses
  $W_{\alpha\beta} = \delta_{\alpha\beta} - \hat Q_\alpha \hat Q_\beta$
  with $\hat Q$ the Cartesian unit momentum transfer (from
  `rlu_to_inv_angstrom_matrix`, stamped on each fit point). The isotropic limit
  reduces **exactly** to the scalar model's $P = 2$ one-component polarization
  factor, so there is no intensity jump when an infinitesimal anisotropy is
  switched on. The dissipative tensor is
  $\chi''_{\alpha\beta} = (\chi_{\alpha\beta} - \chi^*_{\beta\alpha})/2i$; its
  antisymmetric (chiral) part is nonzero only when time reversal is broken (a
  Zeeman field or a DM term). Channels are computed internally so polarized
  (SF/NSF/chiral) fitting can be added later; the per-dataset schema is
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
- **Sign convention.** Positive couplings favour the ordering where the largest
  eigenvalue $\lambda_{\max}(\mathbf{Q})$ of $\mathbb{J}(\mathbf{Q})$ peaks; the
  RPA instability is at $\lambda_{\max}\chi_0 \to 1$.
  $\lambda_{\max}$ has energy units and $\chi_0$ inverse-energy units, so their
  product is dimensionless. With a self-consistency closure, the denominator is
  $1-[\lambda_\nu(\mathbf Q)-\lambda_{\rm shift}]\chi_{0,\rm eff}$.
  `chi0_eff` is the local susceptibility used by the response, and the Onsager
  reaction field `lambda_shift` is an energy subtracted from every interaction
  eigenvalue. nfit reports the smallest sampled denominator as
  `stability_margin`. This continues the scalar convention and is pinned by
  limiting-case tests.
