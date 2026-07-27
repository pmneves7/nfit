# Physics conventions

## Inelastic magnetic neutron scattering

nfit uses energy transfer $E = E_i-E_f=\hbar\omega$ in meV; positive $E$
means neutron energy loss. Momentum is in reciprocal-lattice units or
Å$^{-1}$, temperature is in K, and a double-differential cross section is per
steradian and per meV. Model kernels produce the dissipative response
$\chi''(\mathbf Q,E)$ before instrumental factors.

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
\frac{\chi''_{\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_B T)]},
$$

where $\chi''_{\alpha\beta}$ is the dimensionless-spin response per unit energy.
Since

$$
(\gamma r_0)^2\left(\frac{g}{2}\right)^2
=\left(\frac{\gamma r_0}{2}\right)^2g^2,
\qquad
\left(\frac{\gamma r_0}{2}\right)^2=0.07265\ {\rm barn},
$$

the code uses the numerically equivalent $0.07265\,g^2$ barn form. If the
susceptibility is already expressed as a magnetic-moment response
$\chi''_\mu=g^2\chi''_s$ in `mu_B^2/meV`, no additional $g^2$ is applied.

This is a susceptibility **per unit energy** convention. The fluctuation-
dissipation theorem is

$$
S^{\alpha\beta}(\mathbf Q,E)=
\frac{1}{\pi}\,
\frac{\chi''_{s,\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_BT)]}.
$$

The $1/\pi$ must appear exactly once; it is not part of $\chi''$ itself. The
Bose/detailed-balance factor uses a numerically stable
$1-\exp(-E/k_BT)$ evaluation. The odd-in-$E$ $\chi''$ then produces the correct
detailed balance between energy loss and gain. See Berk's NIST review for the
general double-differential and magnetic correlation-function formalism,
Squires for the magnetic cross section and linear-response convention, and
Welch *et al.* for an explicit absolute-unit derivation
([Berk 1993](https://doi.org/10.6028/jres.098.002);
[Squires, chapter 8](https://doi.org/10.1017/CBO9781139107808.009);
[Welch *et al.* 2022](https://doi.org/10.1103/PhysRevB.105.094402)).

### Polarization convention

The tensor projector is always
$\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta$. A scalar polarization factor
depends on what the scalar $\chi''$ means:

- **One isotropic Cartesian component:** $\chi''=\chi''_{xx}=\chi''_{yy}
  =\chi''_{zz}$ gives $P=2$. This is the scalar convention used by nfit's
  Heisenberg, MMP, local-relaxational, and paramagnon models.
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

If imported $\chi''$ is expressed in `mu_B^2/meV`, it is a
**magnetic-moment** susceptibility and already contains the moment conversion.
nfit therefore applies no additional $g^2$ when converting that imported
channel. If the imported response is instead declared in `spin^2/meV`, nfit
multiplies it by $g^2$ exactly once when calculating the cross section:

$$
\chi''_{\mu_B^2}=g^2\chi''_{\rm spin^2}.
$$

The model kernels themselves are spin responses. When a dataset requests a
model curve in `mu_B^2/meV`, nfit multiplies the model by the dataset's Landé
$g^2$; when it requests `spin^2/meV`, it does not. This explicit choice avoids
both omitting $g^2$ and double counting it. The user-set Landé factor may differ
from 2.

### Relation to SI susceptibility and the role of $\mu_0$

The microscopic spin susceptibility above responds to the conjugate Zeeman
energy. Rationalized SI bulk susceptibility instead uses $M=\chi_{\rm SI}H$.
For a response per magnetic ion,

$$
\chi_{\rm SI,ion}(\mathbf Q,E)
=\frac{\mu_0(g\mu_B)^2}{1\ {\rm meV\ in\ joules}}\,
\chi_s(\mathbf Q,E),
$$

and a molar value additionally carries Avogadro's number and the chosen number
of magnetic ions per formula unit. Equivalently,

$$
\chi_{\rm SI,ion}
=\frac{\mu_0\mu_B^2}{1\ {\rm meV\ in\ joules}}\,\chi_\mu .
$$

Thus the user's $\mu_0$ observation is correct for conversion to proper MKS/SI
$M/H$ units. It is **not** an additional factor in the neutron cross section
once $\chi''$ is already in `spin^2/meV` or `mu_B^2/meV`; inserting it there
would mix SI bulk and microscopic neutron conventions. nfit's molar
`cm^3/mol` to `m^3/mol` conversion includes the rationalized-SI $4\pi$ factor,
and the Heisenberg RPA bulk prediction includes the corresponding
$\mu_0(g\mu_B)^2$ conversion. Welch *et al.* derive this relation explicitly
for susceptibility per magnetic ion.

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

- Absolute cross section is displayed as `mbarn/sr/meV/f.u.` (or the selected
  magnetic-ion/unit-cell basis).
- Absolute dynamical susceptibility is displayed as
  `mu_B^2/meV/f.u.` (or `spin^2/meV/f.u.` when explicitly selected).
- Published normalized intensity such as `1/meV/V` is retained as a
  dimensionful but non-cross-section signal. Its paired $\chi''$ channel has
  the correct Bose/correction shape but remains arbitrary unless an absolute
  cross-section calibration is supplied.
- Uncalibrated data remain `arb. units`. nfit can remove or apply the Bose,
  form-factor, polarization, and kinematic **shape**, but does not label the
  result absolute.
- For count data, `Signal units / mbarn` is the explicit calibration in
  imported signal units per `mbarn/(sr meV)` on the selected sample basis.
  Zero means uncalibrated.

The imported signal is retained as a named channel. When temperature is known,
the paired scattering-cross-section and $\chi''$ channels are generated with
propagated one-sigma errors. `Plot and fit` chooses the primary observable;
both remain independently selectable in the data viewer. Models read the
primary channel's quantity and unit and evaluate in that representation.

The normalization basis is metadata, not a hidden atom-count conversion.
Beam-flux and illuminated-sample calibration must already refer to the same
formula-unit, magnetic-ion, or unit-cell basis selected in the GUI.
For per-atom data, an optional label such as `V` changes the displayed suffix
from `/magnetic ion` to `/V`; it does not multiply or divide the data.
These definitions and the distinction between correlation functions,
$\chi''$, cross section, and absolute normalization follow general neutron
scattering references rather than any material-specific paper
([Squires, chapters 7-8](https://doi.org/10.1017/CBO9781139107808.009);
[Xu, Xu, and Tranquada 2013](https://doi.org/10.1063/1.4818323)).

### Digitized powder data and the LiV2O4 examples

The digitizer CSV importer is general: it records the quantity, units,
normalization basis, temperature, fixed cut coordinate, and kinematic state
specified by the user. It does not infer a convention from a material, author,
or filename. The following papers are therefore provenance and worked import
examples, not the sources of nfit's physical conventions.

For the supplied LiV2O4 files, Lee *et al.* report Q-E maps and constant-E
intensity cuts as normalized magnetic intensity in `1/meV/V`, while their
constant-Q spectra are $\chi''$ in `mu_B^2/meV/V`
([Lee *et al.* 2001](https://doi.org/10.1103/PhysRevLett.86.5554), especially
Figs. 1 and 2). Here `V` means per vanadium atom, not per LiV2O4 formula unit.

Tomiyasu *et al.* report absolute constant-E cuts as
$(k_i/k_f)d^2\sigma/(d\Omega\,dE)$ in `mbarn/sr/meV/V`
([Tomiyasu *et al.* 2014](https://doi.org/10.1103/PhysRevLett.113.236402),
Fig. 1). Because the published ordinate has already been multiplied by
$k_i/k_f$, import it with **k_f/k_i removed upstream**. Their derived
$\chi''$ is in `mu_B^2/meV/V` (Fig. 2). The importer records these choices but
does not silently apply paper-specific conventions.

Digitized three-column cuts carry the digitized one-sigma error bars. Color-map
matrices do not contain an uncertainty layer, so import assigns the explicit
user-selected uniform map uncertainty and masks NaN pixels. Consequently, an
absolute intensity scale does not by itself imply statistically calibrated
map uncertainties.

Bulk magnetic data use explicit CGS/SI conversions: `1 emu = 10^-3 A m^2`,
`1 T = 10^4 Oe` for the applied-field convention, and molar susceptibility
obeys `1 cm^3/mol (CGS) = 4 pi 10^-6 m^3/mol (SI)`.
For a sample with `n` moles of formula units, a measured magnetic moment in
emu is normalized as `M / (n N_A mu_B)` to obtain `mu_B/f.u.`.

Additional conventions used by the spin-fluctuation model family (see
[Spin-fluctuation models](spin_fluctuation_models.md) for the full math):

- Magnetic form factors use the $\langle j_0 \rangle$ analytic approximation
  $f(s) = A e^{-a s^2} + B e^{-b s^2} + C e^{-c s^2} + D$ with
  $s = |Q|/4\pi$ in Å⁻¹ (`nfit.form_factors`).
- Exchange Fourier transforms use the extended-zone phase convention:
  $J(\mathbf{Q})_{ab} = \sum J_{\text{bond}}
  \exp[2\pi i\, \mathbf{Q}\cdot(\mathbf{r}_b + \mathbf{n} - \mathbf{r}_a)]$
  with fractional site positions inside the phases. Because these matrix
  elements already carry the full pair phases, the RPA neutron weights use the
  *uniform* sublattice sum $|\sum_a U_{a\nu}|^2/N$ (no additional site phases),
  which makes the observable exactly independent of the cell description and
  keeps the mode weights summing to one. See
  [Spin-fluctuation models](spin_fluctuation_models.md).
- Temperature enters only through the Bose factor and is read from each
  dataset (`PointData4D.temperature`), never from fit parameters.

## Tensor (anisotropic) interactions

These apply when the `heisenberg_rpa` model is run with anisotropic exchange,
single-ion anisotropy, dipole–dipole, or Zeeman terms (see
[Spin-fluctuation models](spin_fluctuation_models.md#tensor-anisotropic-interactions)).

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
- **Field frames.** The applied field is entered as a magnitude in tesla plus a
  direction given as either a direct $[u\,v\,w]$ vector (converted with the
  direct lattice matrix) or a reciprocal $(H\,K\,L)$ vector (converted with the
  reciprocal matrix), then normalized to a Cartesian unit vector. In a cubic
  cell $[1\,1\,1]$ and $(1\,1\,1)$ coincide; in lower symmetry they differ.
- **Units.** Field $B$ in tesla; Larmor energy $\omega_L = g\,\mu_B\,B$ in meV
  with $\mu_B = 0.05788\,\text{meV/T}$; the dipole strength $D_{\mathrm{dip}}$ in
  meV·Å³ with physical default $(\mu_0/4\pi)(g\mu_B)^2$.
- **Sign convention.** Positive couplings favour the ordering where the largest
  eigenvalue $\lambda_{\max}(\mathbf{Q})$ of $\mathbb{J}(\mathbf{Q})$ peaks; the
  RPA instability is at $\lambda_{\max}\chi_0 \to 1$. This continues the scalar
  convention and is pinned by limiting-case tests.
