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

