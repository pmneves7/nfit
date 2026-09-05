# Generalized paramagnon and damped mode

`generalized_paramagnon` is a continuous phenomenological model for diffuse
spin fluctuations near one or more momentum-space centers. It covers
relaxational paramagnons and damped propagating modes without switching
response functions.

Use it when a local or isotropic MMP response is too restrictive but a
microscopic Heisenberg or electronic model is not justified by the data.

Here $\chi=\chi'+i\chi''$ is one isotropic Cartesian component of the
response of dimensionless spin, in meV$^{-1}$ per magnetic ion. $\chi'$ is
reactive and $\chi''$ absorptive; $i^2=-1$ and $E=\hbar\omega$ is transfer
in meV. Spin matrices, the conjugate energy field, and the Fourier convention
are defined in [Physics conventions](physics_conventions.md#spin-operators-and-equilibrium-averages).

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

$\Delta\mathbf q=\mathbf q_{\rm cart}-\mathbf Q_{c,\rm cart}$ after converting
both momenta with the reciprocal lattice. $C$ is a correlation metric in Å$^2$,
$L$ has entries in Å, superscript $T$ means transpose, and $p>0$ is the fixed
dimensionless spatial exponent (`spatial_power`). Thus $A\ge1$ is dimensionless.

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

The ansatz takes a static peak and adds damping linear in frequency and,
optionally, an inertial term quadratic in frequency to its inverse response.
$\chi_{\rm pk}$ is one-center static amplitude in meV$^{-1}$,
$\Gamma_0>0$ the center relaxation energy in meV, $z\ge0$ a dimensionless
relaxation exponent (not automatically a universal critical exponent), and
$a_E\ge0$ an inverse-square-energy coefficient in meV$^{-2}$.

The full causal response is

$$
\chi(\mathbf q,E)=
\frac{\chi_{\rm pk}/A(\mathbf q)}
{1-[a_E/A(\mathbf q)]E^2-iE/[\Gamma_0A(\mathbf q)^z]}.
$$

Writing the dimensionless real denominator parts $x=1-a_EE^2/A$ and
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

For comparison with bulk data, two fixed model settings specify the conversion
from spin susceptibility:

| API name | Meaning | Unit |
| --- | --- | --- |
| `bulk_g_factor` | Landé factor $g$ | dimensionless |
| `magnetic_ions_per_formula_unit` | equivalent magnetic ions represented by the response | ions/f.u. |

These settings do not alter the dynamical susceptibility. In a joint neutron
and bulk fit, `bulk_g_factor` should agree with the Landé factor in the neutron
dataset's spectral convention.

The diagonal correlation parameters, susceptibility, relaxation energy,
relaxation power, and inertial coefficient are nonnegative; $\Gamma_0$ must be
strictly positive.

The exponent $p$, the set of related peak centers, and whether the response is
periodic are choices that define the spatial form rather than adjustable
dynamical parameters. Powder calculations average this response over
orientations. The crystal lattice converts peak centers in r.l.u. to physical
momentum, and the magnetic form factor supplies the additional $|\mathbf Q|$
dependence of the neutron intensity.

## Important limits

- Local relaxation: set all entries of $L$ and $a_E$ to zero. With one center,
  $\chi=\chi_{\rm pk}/(1-iE/\Gamma_0)$.
- MMP response: use $L=\xi\mathbb 1$, $p=2$, $z=1$, one center, and $a_E=0$.
- Damped propagating mode: use $a_E>0$. At a peak center,
  $E_0=1/\sqrt{a_E}$ and the equivalent damped-harmonic-oscillator (DHO) damping energy is
  $\gamma_{\rm DHO}=E_0^2/\Gamma_0$. The damping ratio is
  $\zeta=\gamma_{\rm DHO}/(2E_0)$.

The inertial coefficient is fitted instead of $E_0$ so the relaxational limit
is the finite boundary `inverse_mode_energy_sq=0`, rather than
$E_0\rightarrow\infty$. Fit diagnostics report $E_0$ and the damping
quantities when the coefficient is nonzero.

## Calculable data

The model calculates:

- single-crystal and powder inelastic neutron scattering;
- single-crystal and powder quasistatic elastic magnetic scattering;
- uniform bulk susceptibility; and
- magnetic moment or magnetization in linear response to an applied field.

Inelastic calculations use $\chi''$. Quasistatic elastic calculations use the
static limit

$$
\chi'(\mathbf q,0)=\frac{\chi_{\rm pk}}{A(\mathbf q)}
$$

for each included center. The bulk prediction evaluates the same sum at
$\mathbf q=0$. This makes a joint comparison possible, but does not assert
that a peak expansion developed at finite momentum is quantitatively valid at
the zone center. Calculating the extrapolation does not require the user to
include bulk data in a fit.

Powder datasets average the same response over orientations. The
fluctuation--dissipation relation, polarization factor, magnetic form factor,
and experimental normalization then convert the intrinsic susceptibility into
the measured neutron intensity, as described in
[Physics conventions](physics_conventions.md).

## Fitting and identifiability

Inelastic data are needed to constrain `gamma0`, `relaxation_power`, and
`inverse_mode_energy_sq`. Elastic and bulk data constrain only the static
spatial response. A bulk value alone usually constrains a combination of peak
amplitude, correlation metric, and center positions rather than those
quantities separately. Bulk conversion uses the stated Landé factor and number
of magnetic ions per formula unit.

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
