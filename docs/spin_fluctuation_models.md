# Spin-fluctuation models

This page documents the physics models for itinerant, nearly magnetically
ordered systems: a fully local relaxational spin, the Millis–Monien–Pines
(MMP) nearly-antiferromagnetic form, and a Heisenberg-coupled RPA model whose
exchange constants live on symmetry-distinct bond orbits. All three compute
the imaginary part of the dynamic susceptibility $\chi''(\mathbf{Q}, E)$ and
convert it to measured intensity through the shared cross-section convention.

## From susceptibility to intensity

Every model in this family produces

$$
I(\mathbf{Q}, E) = s\, \tfrac{2}{3}\, |f(Q)|^2\,
\frac{\chi''(\mathbf{Q}, E)}{1 - e^{-E/k_B T}},
$$

where

- $s$ is a fittable overall **scale** for unnormalized data,
- $2/3$ is the polarization (orientation) factor for **isotropic (Heisenberg)
  spins**: unpolarized neutrons couple only to spin components perpendicular
  to $\mathbf{Q}$, and the isotropic average of
  $\sum_{\alpha} (1 - \hat{Q}_\alpha^2)$ is $2/3$,
- $|f(Q)|^2$ is the magnetic form factor (below),
- $[1 - e^{-E/k_B T}]^{-1}$ is the detailed-balance (Bose) factor connecting
  $\chi''$ to $S(\mathbf{Q}, E)$; the odd-in-$E$ $\chi''$ times this factor
  satisfies detailed balance automatically.

The **temperature** is read from each dataset (`PointData4D.temperature`).
Fitting a spin-fluctuation model against a dataset without a valid temperature
raises an error naming the fix: set the per-dataset temperature in the GUI
(dataset details, "T (K)") or import data carrying temperature metadata.
Backgrounds are separate additive components (`constant_background`,
`linear_background`), not part of these models.

## Magnetic form factor

The `metallix.form_factors` module tabulates the $\langle j_0 \rangle$
analytic approximation

$$
f(s) = A e^{-a s^2} + B e^{-b s^2} + C e^{-c s^2} + D, \qquad
s = \frac{|Q|}{4\pi} \ \text{in Å}^{-1},
$$

for the common $3d$, $4d$, $4f$, and $5f$ ions, keyed by labels such as
`"Mn2"`, `"Fe2"`, `"Yb3"`, `"U4"`. Custom coefficients
$(A, a, B, b, C, c, D)$ from the ILL tables override the built-in table.
Computing $|Q|$ in Å$^{-1}$ requires lattice metadata on the fit points; the
GUI attaches the group's lattice parameters automatically.

Sources: P. J. Brown, *International Tables for Crystallography*, Vol. C,
Section 4.4.5; the ILL form-factor tables
(<https://www.ill.eu/sites/ccsl/ffacts/>); coefficients transcribed from the
public-domain `periodictable` package.

## Model 1: local relaxational spin (`local_relaxational`)

A totally local spin relaxing at rate $\Gamma$:

$$
\chi''(E) = \chi_{\mathrm{loc}}\, \frac{\Gamma E}{E^2 + \Gamma^2}.
$$

$\chi''$ peaks at $E = \Gamma$; there is no $\mathbf{Q}$ dependence beyond the
form factor, so the model also applies to powder data. `scale` and
`chi_loc` are exactly degenerate — fix one.

## Model 2: MMP relaxational (`mmp_relaxational`)

The phenomenological susceptibility of a nearly antiferromagnetic metal
[Millis, Monien & Pines 1990; Monthoux & Pines 1993]:

$$
\chi(\mathbf{q}, \omega) =
\frac{\chi_{\mathrm{pk}}}{1 + \xi^2 |\mathbf{q} - \mathbf{Q}_0|^2
- i \omega/\omega_{\mathrm{sf}}},
$$

$$
\chi''(\mathbf{q}, E) = \chi_{\mathrm{pk}}\,
\frac{E/\omega_{\mathrm{sf}}}
{\left(1 + \xi^2 |\mathbf{q} - \mathbf{Q}_0|^2\right)^2
+ (E/\omega_{\mathrm{sf}})^2},
$$

with correlation length $\xi$ in Å, $|\mathbf{q} - \mathbf{Q}_0|$ in
Å$^{-1}$ (computed from HKL through the lattice matrix), and
spin-fluctuation energy $\omega_{\mathrm{sf}}$ in meV. At $\mathbf{Q}_0$ the
lineshape is relaxational with $\Gamma = \omega_{\mathrm{sf}}$; away from
$\mathbf{Q}_0$ the response broadens and weakens. This form describes, e.g.,
the normal-state response of optimally doped BaFe$_{1.85}$Co$_{0.15}$As$_2$
[Inosov et al. 2010].

## Model 3: Heisenberg RPA (`heisenberg_rpa`)

The centerpiece model couples local relaxational spins through real-space
Heisenberg exchange in the random phase approximation. The single-site
dynamic susceptibility is

$$
\chi_0(\omega) = \frac{\chi_0}{1 - i\omega/\Gamma_0},
$$

and the RPA sums the exchange to all orders:

$$
\chi(\mathbf{Q}, \omega) =
\left[\mathbb{1} - \chi_0(\omega)\, J(\mathbf{Q})\right]^{-1} \chi_0(\omega).
$$

### Exchange Fourier transform and phase convention

With $N$ magnetic sites at fractional positions $\mathbf{r}_a$ in the unit
cell, $J(\mathbf{Q})$ is the $N \times N$ Hermitian matrix

$$
J(\mathbf{Q})_{ab} = \sum_{\text{bonds } (a, b, \mathbf{n})}
J_{\text{bond}}\,
e^{2\pi i\, \mathbf{Q} \cdot (\mathbf{r}_b + \mathbf{n} - \mathbf{r}_a)},
$$

with $\mathbf{Q} = (H, K, L)$ in r.l.u. and $\mathbf{n}$ the integer cell
offset of the bond target. This is the "extended zone" convention (sublattice
offsets inside the phases, as in Sunny.jl). Each physical bond is stored once;
its Hermitian conjugate is added automatically.

### Mode decomposition

Diagonalizing $J(\mathbf{Q}) = U \Lambda U^\dagger$ decouples the RPA into
relaxational modes:

$$
\chi''(\mathbf{Q}, E) = \sum_\nu w_\nu(\mathbf{Q})\,
\chi_{\mathbf{Q}\nu}\, \frac{\Gamma_\nu E}{E^2 + \Gamma_\nu^2},
$$

$$
\chi_{\mathbf{Q}\nu} = \frac{\chi_0}{1 - \lambda_\nu(\mathbf{Q})\, \chi_0},
\qquad
\Gamma_\nu = \Gamma_0 \left[1 - \lambda_\nu(\mathbf{Q})\, \chi_0\right],
$$

with neutron structure-factor weights

$$
w_\nu(\mathbf{Q}) = \frac{1}{N}
\left| \sum_a U_{a\nu}(\mathbf{Q})\,
e^{2\pi i\, \mathbf{Q} \cdot \mathbf{r}_a} \right|^2,
\qquad \sum_\nu w_\nu(\mathbf{Q}) = 1 .
$$

The weight sum rule holds exactly because $U$ is unitary and the same phase
convention enters $J(\mathbf{Q})$ and the weights; the test suite enforces it
as a phase-convention lock.

### Physics and conventions

- **Sign convention:** positive $J$ favors ordering at the wavevector where
  $\lambda_{\max}(\mathbf{Q})$ is largest (e.g. a single positive
  nearest-neighbor $J$ on a Bravais lattice favors $\mathbf{Q} = 0$,
  ferromagnetism; negative $J$ favors the zone boundary).
- **Stoner-like criterion:** the static susceptibility of mode $\nu$ diverges
  when $\lambda_\nu(\mathbf{Q})\, \chi_0 \to 1$. Evaluation raises an error on
  the ordered side ($1 - \lambda \chi_0 \le 0$); during optimization such
  trial parameters are mapped to a huge misfit so the fit stays in the
  paramagnetic region.
- **Critical slowing:** $\Gamma_\nu = \Gamma_0 (1 - \lambda_\nu \chi_0)$
  softens toward the incipient ordering vector — the standard relaxational
  phenomenology of nearly ordered itinerant magnets [Moriya 1985; Bernhoeft &
  Lonzarich 1995]. In the limit of all $J = 0$ the model reduces exactly to
  `local_relaxational`.

### Symmetry-distinct bond orbits

Exchange constants are defined per **bond orbit**: the set of bonds mapped
onto each other by the space group, which must share one $J$. Orbits are
generated from the crystal (lattice, space group, magnetic Wyckoff sites) up
to a cutoff distance and labeled `J1`, `J2`, ... by increasing bond length.
When symmetry-inequivalent orbits occur at the *same* distance they get letter
suffixes — on the pyrochlore lattice the twelve third neighbors split into
`J3a` and `J3b` (six bonds each per site), which are independent fit
parameters, following the Sunny.jl convention. Each orbit label becomes one
fit parameter of the component, with the same sharing/limit/constraint
machinery as any other parameter.

### Units

| Quantity | Unit |
| --- | --- |
| $E$, $\Gamma_0$, $\omega_{\mathrm{sf}}$, $J_i$ | meV |
| $\chi_0$, $\chi_{\mathrm{loc}}$, $\chi_{\mathrm{pk}}$ | meV$^{-1}$ (up to the intensity normalization) |
| $\xi$ | Å |
| $H, K, L$ | r.l.u. |
| $T$ | K |

$J \chi_0$ is dimensionless, so the instability criterion is unit-free.

## Temperature-dependent fitting

Model parameters are shared across datasets through the standard sharing
modes. To fit a temperature series, set each dataset's temperature (GUI
dataset details or `dataset.parameters["temperature"]`), then choose
`per_dataset` (or `grouped`) sharing for the parameters expected to vary with
temperature — typically `chi0` and `gamma0` — while keeping the exchange
constants and `scale` global:

```python
from metallix import FitDatasetInput, ModelComponentSpec, compile_fit_problem

component = ModelComponentSpec(
    name="rpa",
    type="heisenberg_rpa",
    parameters={"scale": 1.0, "chi0": 0.3, "gamma0": 4.0, "J1": 0.0},
    fit_parameters={"chi0": True, "gamma0": True, "J1": True},
    sharing={
        "chi0": {"mode": "per_dataset"},
        "gamma0": {"mode": "per_dataset"},
    },
    config={
        "ion": "Yb3",
        "site_positions": [[0.0, 0.0, 0.0]],
        "orbits": [{"label": "J1", "bonds": [
            {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]},
            {"site_i": 0, "site_j": 0, "offset": [0, 1, 0]},
            {"site_i": 0, "site_j": 0, "offset": [0, 0, 1]},
        ]}],
    },
)
compiled = compile_fit_problem(
    [component],
    [FitDatasetInput("T2K", points_2K), FitDatasetInput("T50K", points_50K)],
)
```

The compiled problem then contains `rpa.chi0[T2K]`, `rpa.chi0[T50K]`, one
global `rpa.J1`, and each dataset's Bose factor uses its own temperature.

For scripted orbit generation from a CIF file:

```python
from metallix import crystal_from_cif, generate_bond_orbits, orbits_to_config, sites_to_config

crystal = crystal_from_cif("pyrochlore.cif")
sites, orbits = generate_bond_orbits(crystal, ["Yb1"], cutoff_angstrom=7.2)
component.config["site_positions"] = sites_to_config(sites)
component.config["orbits"] = orbits_to_config(orbits)  # J1, J2, J3a, J3b, ...
```

## References

1. T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*, Springer
   Series in Solid-State Sciences 56 (Springer, Berlin, 1985).
2. A. J. Millis, H. Monien, and D. Pines, "Phenomenological model of nuclear
   relaxation in the normal state of YBa₂Cu₃O₇", Phys. Rev. B **42**, 167
   (1990). <https://doi.org/10.1103/PhysRevB.42.167>
3. P. Monthoux and D. Pines, "YBa₂Cu₃O₇: A nearly antiferromagnetic Fermi
   liquid", Phys. Rev. B **47**, 6069 (1993).
   <https://doi.org/10.1103/PhysRevB.47.6069>
4. D. S. Inosov et al., "Normal-state spin dynamics and temperature-dependent
   spin-resonance energy in optimally doped BaFe₁.₈₅Co₀.₁₅As₂", Nat. Phys.
   **6**, 178 (2010). <https://doi.org/10.1038/nphys1483>
5. N. R. Bernhoeft and G. G. Lonzarich, "Scattering of slow neutrons from
   long-wavelength magnetic fluctuations in UPt₃", J. Phys.: Condens. Matter
   **7**, 7325 (1995). <https://doi.org/10.1088/0953-8984/7/37/006>
6. P. J. Brown, "Magnetic form factors", *International Tables for
   Crystallography*, Vol. C, Section 4.4.5; ILL tables:
   <https://www.ill.eu/sites/ccsl/ffacts/>
7. D. Dahlbom et al., Sunny.jl — symmetry-distinct bond convention:
   <https://github.com/SunnySuite/Sunny.jl>
