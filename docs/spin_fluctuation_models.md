# Spin-fluctuation models

nfit provides ten magnetic-response models. Start with the least structured
response that resolves the features in the data. Electronic Hamiltonians are
documented separately under
[Electronic-structure models](electronic_structure_models.md).

The table names each model's starting ansatz. Follow its link for equations,
operator definitions, parameter units, and validity limits; shared spin and
response conventions are defined in [Physics conventions](physics_conventions.md).

| Model | Starting ansatz | Calculable data |
| --- | --- | --- |
| [Local relaxational](local_relaxational.md) | One local variable with exponential relaxation | Single-crystal and powder neutron scattering; bulk linear response |
| [MMP](mmp_relaxational.md) | Isotropic Ornstein--Zernike peak with relaxational dynamics | Single-crystal neutron scattering; extrapolated bulk linear response |
| [Generalized paramagnon](generalized_paramagnon.md) | Anisotropic static peak with damping and optional inertia | Single-crystal and powder neutron scattering; extrapolated bulk linear response |
| [Conserved ferromagnetic](conserved_ferromagnetic.md) | Long-wavelength spin response with vanishing uniform relaxation rate | Single-crystal and powder neutron scattering; bulk linear response |
| [Heisenberg RPA](heisenberg_rpa.md) | Local propagators coupled by a crystallographic exchange matrix | Single-crystal and powder neutron scattering; bulk response |
| [Coupled susceptibilities](coupled_susceptibility.md) | Two scalar response sectors with bilinear feedback | Single-crystal neutron scattering |
| [Lindhard](lindhard.md) | Normalized independent-particle bubble with explicit probe matrix elements | Single-crystal and powder neutron scattering; bulk linear response |
| [Stoner RPA](stoner_rpa.md) | Spin-channel feedback vertex twice the fitted density interaction | Single-crystal and powder neutron scattering; bulk linear response |
| [Matrix RPA](matrix_rpa.md) | A fixed Hermitian Cartesian interaction matrix times a fitted energy | Single-crystal and powder neutron scattering; bulk linear response |
| [Hubbard--Hund RPA](hubbard_hund_rpa.md) | Local orbital-pair vertex, dressed before physical spin projection | Single-crystal and powder neutron scattering; bulk linear response |

```{toctree}
:maxdepth: 1
:caption: Model details

local_relaxational
mmp_relaxational
generalized_paramagnon
conserved_ferromagnetic
heisenberg_rpa
coupled_susceptibility
lindhard
stoner_rpa
matrix_rpa
hubbard_hund_rpa
```

For inelastic data, each magnetic-response kernel supplies the dissipative
spin susceptibility $\chi''_s(\mathbf Q,E)$ in `spin^2/meV` in its documented
site, formula-unit, or primitive-cell normalization.
The dataset convention determines
whether nfit compares that response directly with $\chi''$ data or converts it
to a neutron cross section. See [Physics conventions](physics_conventions.md)
for the Bose factor, polarization, form factor, Landé factor, and absolute
normalization.

Every model that applies a magnetic form factor reads the same three settings:
`ion` selects a tabulated ion, `form_factor_coefficients` and
`form_factor_j2_coefficients` override the $\langle j_0\rangle$ and
$\langle j_2\rangle$ coefficients, and `form_factor_g_J` selects the dipole
approximation. `form_factor_g_J` defaults to 2, which is the spin-only
$\langle j_0\rangle$ form factor; set it to the ion's Landé factor for a moment
with an orbital contribution. It is independent of the dataset `g_factor`,
which converts a spin response to a magnetic moment.

Backgrounds are separate additive model components.

The [local relaxational](local_relaxational.md),
[MMP relaxational](mmp_relaxational.md),
[generalized paramagnon](generalized_paramagnon.md),
[conserved ferromagnetic paramagnon](conserved_ferromagnetic.md),
[Heisenberg RPA](heisenberg_rpa.md), and
[coupled susceptibilities](coupled_susceptibility.md),
[bare Lindhard](lindhard.md), [scalar Stoner](stoner_rpa.md),
[matrix RPA](matrix_rpa.md), and
[Hubbard--Hund RPA](hubbard_hund_rpa.md) pages define their parameters, complex
susceptibilities, dissipative responses, elastic limits, scripts, and
references.

Bulk comparison is optional. For models constructed near finite-momentum
peaks, the $\mathbf Q=0$ result is an extrapolation whose failure can itself be
a useful model diagnostic.

Model parameter tables distinguish canonical response energies in meV from
electronic interaction inputs in eV, and dimensionless Landé factors from
energy-valued coupling constants. Never transfer a parameter solely by its
symbol between models.
