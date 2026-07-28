# Generalized paramagnon and damped mode

`generalized_paramagnon` is a continuous phenomenological model for diffuse
spin fluctuations near one or more momentum-space centers. It covers
relaxational paramagnons and damped propagating modes without switching
response functions.

Use it when a local or isotropic MMP response is too restrictive but a
microscopic Heisenberg or electronic model is not justified by the data.

## Spatial response

For a peak center $\mathbf Q_c$, define the Cartesian momentum offset
$\Delta\mathbf q$ in Å$^{-1}$ and

$$
A(\mathbf q)=
1+\left[\Delta\mathbf q^T C\Delta\mathbf q\right]^{p/2},
\qquad C=LL^T.
$$

$L$ is a lower-triangular matrix in the crystallographic Cartesian frame:

$$
L=
\begin{pmatrix}
\xi_x&0&0\\
\xi_{yx}&\xi_y&0\\
\xi_{zx}&\xi_{zy}&\xi_z
\end{pmatrix}.
$$

This parameterization keeps $C$ positive semidefinite while its diagonal and
off-diagonal elements are fitted. The singular values of $L$ are the principal
correlation lengths. The default $p=2$ gives an Ornstein--Zernike spatial
profile.

`q0_h`, `q0_k`, and `q0_l` define a fitted primary center. Fixed
`center_offsets` add crystallographically related centers. With
`center_combination="sum"` their responses add; `"nearest"` assigns each
momentum to its smallest spatial kernel. `periodic=true` minimizes the fitted
spatial kernel over nearby reciprocal-lattice images, including for
non-orthogonal cells.

## Complex dynamical susceptibility

The static susceptibility and momentum-dependent relaxation energy are

$$
\chi_{\rm static}(\mathbf q)=\frac{\chi_{\rm pk}}{A(\mathbf q)},
\qquad
\Gamma(\mathbf q)=\Gamma_0 A(\mathbf q)^z.
$$

The full causal response is

$$
\chi(\mathbf q,E)=
\frac{\chi_{\rm pk}/A(\mathbf q)}
{1-[a_E/A(\mathbf q)]E^2-iE/[\Gamma_0A(\mathbf q)^z]}.
$$

Writing $x=1-a_EE^2/A$ and
$y=E/[\Gamma_0A^z]$, the fitted dissipative part is

$$
\chi''(\mathbf q,E)=
\frac{\chi_{\rm pk}}{A}\frac{y}{x^2+y^2}.
$$

The sign convention gives $\chi(-E)=\chi(E)^*$, an even real part, and an odd
imaginary part.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `chi_peak` | static susceptibility of one center | meV$^{-1}$ |
| `gamma0` | relaxation energy at a center, $\Gamma_0$ | meV |
| `relaxation_power` | exponent $z$ in $\Gamma=\Gamma_0A^z$ | dimensionless |
| `inverse_mode_energy_sq` | inertial coefficient $a_E=1/E_0^2$ | meV$^{-2}$ |
| `xi_x`, `xi_y`, `xi_z` | diagonal entries of $L$ | Å |
| `xi_yx`, `xi_zx`, `xi_zy` | off-diagonal entries of $L$ | Å |
| `q0_h`, `q0_k`, `q0_l` | fitted primary center | r.l.u. |

The diagonal correlation parameters, susceptibility, relaxation energy,
relaxation power, and inertial coefficient have nonnegative fit bounds;
`gamma0` must be strictly positive when evaluated.

Fixed configuration contains `spatial_power`, `center_offsets`,
`center_combination`, `periodic`, `powder_orientations`, optional lattice
parameters, and the magnetic form factor.

## Important limits

- Local relaxation: set all entries of $L$ and $a_E$ to zero. With one center,
  $\chi=\chi_{\rm pk}/(1-iE/\Gamma_0)$.
- MMP response: use $L=\xi\mathbb 1$, $p=2$, $z=1$, one center, and $a_E=0$.
- Damped propagating mode: use $a_E>0$. At a peak center,
  $E_0=1/\sqrt{a_E}$ and the equivalent DHO damping coefficient is
  $\gamma_{\rm DHO}=E_0^2/\Gamma_0$. The damping ratio is
  $\zeta=\gamma_{\rm DHO}/(2E_0)$.

The inertial coefficient is fitted instead of $E_0$ so the relaxational limit
is the finite boundary `inverse_mode_energy_sq=0`, rather than
$E_0\rightarrow\infty$. Fit diagnostics report $E_0$ and the damping
quantities when the coefficient is nonzero.

## Dataset comparison

Inelastic datasets use $\chi''$. Elastic datasets use the exact static limit

$$
\chi'(\mathbf q,0)=\frac{\chi_{\rm pk}}{A(\mathbf q)}
$$

and then the common quasistatic cross-section conversion. Elastic-only data
cannot constrain `gamma0`, `relaxation_power`, or
`inverse_mode_energy_sq`.

Powder datasets average the same response over deterministic sphere
directions. Lattice or reciprocal-basis metadata is required to relate the
centers in r.l.u. to physical momentum. The Bose factor, magnetic form factor,
polarization, and dataset scale remain outside the susceptibility kernel.

## Scripting and plots

The numerical API returns the complex response:

```python
from nfit import (
    generalized_paramagnon_susceptibility,
    paramagnon_spatial_kernel,
)

A = paramagnon_spatial_kernel(
    q_offsets_inv_angstrom,
    correlation_cholesky_angstrom=L,
    spatial_power=2,
)
chi = generalized_paramagnon_susceptibility(
    A,
    energy_meV,
    chi_peak=3.0,
    gamma0=2.0,
    relaxation_power=1.0,
    inverse_mode_energy_sq=0.04,
)
```

`generalized_paramagnon_energy_scan` calculates $\chi'$ and $\chi''$ at
specified values of $A(\mathbf q)$.
`render_generalized_paramagnon_energy_scan` renders the two panels, and
`generalized_paramagnon_energy_scan_script` produces an editable GUI-free
script. The same provider is available through
`model_plot_definitions("generalized_paramagnon")`.

Fitting uses `ModelComponentSpec(type="generalized_paramagnon", ...)`. Project
files, workflow scripts, fit-result export, reports, and model/residual plot
channels use the shared model machinery.

## References

- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*
  (Springer, 1985),
  [doi:10.1007/978-3-642-82499-9](https://doi.org/10.1007/978-3-642-82499-9).
- A. J. Millis, H. Monien, and D. Pines, *Phys. Rev. B* **42**, 167
  (1990), [doi:10.1103/PhysRevB.42.167](https://doi.org/10.1103/PhysRevB.42.167).
- S. M. Hayden *et al.*, *Phys. Rev. Lett.* **66**, 821
  (1991), [doi:10.1103/PhysRevLett.66.821](https://doi.org/10.1103/PhysRevLett.66.821).
