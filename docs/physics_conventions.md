# Physics conventions

## Inelastic magnetic neutron scattering

nfit uses energy transfer $E = E_i-E_f=\hbar\omega$ in meV; positive $E$
means neutron energy loss. Momentum is in reciprocal-lattice units or
Å$^{-1}$, temperature is in K, and a double-differential cross section is per
steradian and per meV. Model kernels produce the dissipative response
$\chi''(\mathbf Q,E)$ before instrumental factors.

For an unpolarized measurement in the dipole approximation, nfit uses

$$
\frac{d^2\sigma}{d\Omega\,dE} =
\frac{k_f}{k_i}\frac{C}{\pi}|f(Q)|^2
\sum_{\alpha\beta}
(\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta)
\frac{\chi''_{\alpha\beta}(\mathbf Q,E)}
{1-\exp[-E/(k_B T)]},
\qquad
C=\left(\frac{\gamma r_0}{2}\right)^2
=0.07265\ {\rm barn}/\mu_B^2.
$$

This is a susceptibility **per unit energy** convention. The $1/\pi$ belongs
to nfit's fluctuation-dissipation convention; it must not be inserted a second
time when importing an already calculated $\chi''$. The Bose/detailed-balance
factor uses a numerically stable $1-\exp(-E/k_BT)$ evaluation. The odd-in-$E$
$\chi''$ then produces the correct detailed balance between energy loss and
gain. See Berk's NIST review for the general double-differential and magnetic
correlation-function formalism and Squires for the magnetic cross section and
linear-response convention
([Berk 1993](https://doi.org/10.6028/jres.098.002);
[Squires, chapter 8](https://doi.org/10.1017/CBO9781139107808.009)).

### Polarization convention

The tensor projector is always
$\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta$. A scalar polarization factor
depends on what the scalar $\chi''$ means:

- **Isotropic trace:** $\chi''=\sum_\alpha\chi''_{\alpha\alpha}$ gives
  $P=2/3$. This is the scalar convention used by nfit's Heisenberg, MMP, and
  local-relaxational models.
- **One isotropic Cartesian component:** $\chi''=\chi''_{xx}=\chi''_{yy}
  =\chi''_{zz}$ gives $P=2$.
- **Already corrected:** use $P=1$ only when the stored response has already
  had the polarization contraction removed.
- **Custom scalar:** a positive user-supplied factor is recorded in the
  dataset convention.

For an anisotropic response, the full tensor contraction is required; a single
scalar $P$ is not generally physical.

### Magnetic moment, spin, and the Landé factor

If $\chi''$ is expressed in `mu_B^2/meV`, it is a **magnetic-moment**
susceptibility and already contains the moment conversion. nfit therefore
applies no additional $g^2$. If the response is instead declared in
`spin^2/meV`, nfit multiplies it by $g^2$ exactly once when calculating the
cross section:

$$
\chi''_{\mu_B^2}=g^2\chi''_{\rm spin^2}.
$$

This explicit choice avoids both omitting $g^2$ and double counting it.

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
  $W_{\alpha\beta} = \tfrac13(\delta_{\alpha\beta} - \hat Q_\alpha \hat Q_\beta)$
  with $\hat Q$ the Cartesian unit momentum transfer (from
  `rlu_to_inv_angstrom_matrix`, stamped on each fit point). The per-component
  $1/3$ makes the isotropic limit reduce **exactly** to the scalar model's
  $P = 2/3$ polarization factor, so there is no intensity jump when an
  infinitesimal anisotropy is switched on. The dissipative tensor is
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
