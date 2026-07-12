# Physics conventions

The package separates dynamical susceptibility from measured neutron intensity.

- Model functions return chi''(Q,E).
- Cross-section helpers convert chi'' into intensity-like quantities.
- H, K, and L are reciprocal lattice units.
- E is energy transfer in meV. Positive E denotes neutron energy loss.
- Temperature is in K.

The initial intensity convention is

$$
I(Q,E) =
s |f(Q)|^2 P(Q)
\frac{\chi''(Q,E)}{1-\exp[-E/(k_B T)]}
+ B(Q,E).
$$

The scale, magnetic form factor, polarization factor, and background are kept
explicit so that normalization assumptions remain visible.

Additional conventions used by the spin-fluctuation model family (see
[Spin-fluctuation models](spin_fluctuation_models.md) for the full math):

- The polarization factor for isotropic (Heisenberg) spins is $P = 2/3$.
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

