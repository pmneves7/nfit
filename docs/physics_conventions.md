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

