# Conserved ferromagnetic paramagnon

The `conserved_ferromagnetic` model describes long-wavelength fluctuations of
a nearly ferromagnetic metal when total spin is conserved on the experimental
time scale. Its defining feature is a relaxation rate that vanishes at the
ferromagnetic wavevector. It is therefore distinct from a local relaxor or an
MMP antiferromagnetic peak, both of which retain a finite relaxation rate at
their peak center.

## Parameters

| API name | Meaning | Unit |
| --- | --- | --- |
| `chi_uniform` | static susceptibility at the primary ferromagnetic center, $\chi_{\rm u}$ | meV$^{-1}$ |
| `gamma_scale` | conserved relaxation-energy scale, $\Gamma_s$ | meV |
| `xi_x`, `xi_y`, `xi_z` | diagonal entries of the correlation-metric Cholesky factor $L$ | Å |
| `xi_yx`, `xi_zx`, `xi_zy` | off-diagonal entries of $L$ | Å |
| `q0_h`, `q0_k`, `q0_l` | primary peak center $\mathbf Q_0$ | r.l.u. |

The positive-semidefinite correlation metric is $C=LL^T$. For Cartesian
momentum offset $\Delta\mathbf q$ in Å$^{-1}$, define

$$
\rho^2=\Delta\mathbf q^TC\Delta\mathbf q.
$$

$\rho$ is dimensionless. Periodic reciprocal-lattice images and multiple
domain centers use the same conventions as the generalized-paramagnon model.

## Complex susceptibility

The static Ornstein--Zernike response and momentum-dependent relaxation rate
are

$$
\chi(\rho,0)=\frac{\chi_{\rm u}}{1+\rho^2},\qquad
\Gamma(\rho)=\Gamma_s\rho^m(1+\rho^2),
$$

with the causal response

$$
\chi(\rho,E)=\chi(\rho,0)
\frac{\Gamma(\rho)}{\Gamma(\rho)-iE}.
$$

The `damping_kind` configuration selects $m=1$ (`clean`) for ballistic
Landau damping or $m=2$ (`diffusive`) for spin diffusion. `gamma_scale`
absorbs the corresponding power of the inverse correlation length, so it
always has meV units in this parameterization.

At $\rho=0$, $\chi(0,0)=\chi_{\rm u}$ but the finite-energy response is zero.
This noncommuting static and dynamic limit is the conservation law, not a
numerical regularization. Data that do not approach the peak closely will
usually constrain a combination of `gamma_scale` and the correlation lengths
more strongly than either quantity separately.

## Calculable data

The model calculates:

- single-crystal and powder inelastic neutron scattering;
- single-crystal and powder quasistatic elastic magnetic scattering; and
- uniform susceptibility or linear-response magnetization.

For inelastic data, nfit converts $\chi''=\operatorname{Im}\chi$ using the
dataset's spectral convention, Bose factor, isotropic polarization factor,
magnetic form factor, and scale. Elastic and bulk channels use the $E=0$
static limit. A lattice or reciprocal-basis matrix is required when momentum
coordinates are in r.l.u.

Powder averaging uses `powder_orientations` deterministic sphere directions.
`center_offsets` adds fixed offsets to the fitted $\mathbf Q_0$;
`center_combination="nearest"` uses the nearest center at each momentum,
whereas `"sum"` adds responses from distinct domains.

## Scripting and export

Use `ModelComponentSpec(type="conserved_ferromagnetic", ...)` in compiled fit
workflows. The public numerical functions
`conserved_ferromagnetic_susceptibility` and
`conserved_ferromagnetic_chipp` accept the already scaled radius $\rho$ and
are useful for testing or custom scripts.

## References

- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*
  (Springer, 1985).
- N. R. Bernhoeft and G. G. Lonzarich, *J. Phys.: Condens. Matter* **7**,
  7325 (1995), [doi:10.1088/0953-8984/7/37/006](https://doi.org/10.1088/0953-8984/7/37/006).
