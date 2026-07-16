# Analysis Window

The Analysis Window runs non-fitting operations on reduced datasets while
keeping inputs unchanged. Open it from a workspace, dataset, or `Analyses` node
with **Open Analysis Window**. Successful dataset-like results appear in
`Datasets / Derived data`; the authoritative recipe remains under `Analyses`.

Projects must be saved before a run. Array outputs live in
`<project>.nfit-assets/analyses/<analysis-id>/`; project JSON stores relative
manifests and provenance. **Save As** copies the asset tree. Changing a recipe
or input marks the latest result stale without deleting it.

## Curie-Weiss fitting

The **Curie-Weiss fit** operation accepts unit-aware point-list susceptibility
data and exposes only `Tmin` and `Tmax` as fit controls. The input must be an
absolute molar susceptibility in `cm^3/mol` or rationalized SI `m^3/mol`; for
MPMS magnetization data, first enable **Plot and fit susceptibility** and the
sample mass/molar-mass normalization in the dataset panel.

The operation fits the untransformed susceptibility to
`chi(T) = C / (T - theta_CW)`. This avoids the statistical bias introduced by
least-squares fitting after taking `1/chi`. If point uncertainties are
available, the fit is weighted with them and treats them as absolute standard
uncertainties. The parameter covariance is propagated to `C`, `theta_CW`, and
`mu_eff = sqrt(3 k_B C / (N_A mu_B^2))`; results report `C` in `cm^3 K/mol`,
`theta_CW` in K, and `mu_eff` in `mu_B/f.u.`.

The **Curie-Weiss diagnostic** output retains the full temperature range. Open
it in the data viewer and switch the channel between **Susceptibility** and
**Inverse susceptibility**. The matching fitted curve is shown automatically,
and dashed vertical lines mark the fitted `Tmin` and `Tmax`.

The same expression is also available as the ordinary **Curie-Weiss
susceptibility** model component. It applies only to magnetization datasets
whose selected fit channel is absolute molar susceptibility.

## Low-temperature heat-capacity fitting

The **Low-temperature C/T fit** operation accepts molar heat capacity or C/T,
fits `C/T = gamma + beta T^2` between editable `Tmin` and `Tmax`, and reports
the Sommerfeld coefficient, Debye coefficient, and the Debye temperature for
the supplied atom count. Its diagnostic defaults to C/T versus T squared.
See [Heat-capacity data and models](heat_capacity.md) for import units and the
ordinary Debye and low-temperature fit components.

## Bragg integration

Bragg integration accepts three momentum dimensions, or 4D data. For 4D data,
leaving both elastic-energy bounds blank uses the measured energy bin nearest
zero; set both bounds to integrate an explicit elastic window. Peaks come from space-group-filtered crystal
HKLs or supplied H, K, L positions. Methods are exact-overlap HKL boxes,
deterministic subvoxel ellipsoids, and local Gaussian fits. Regions may use HKL
or inverse-angstrom widths. Optional shells subtract a volume-scaled local
background and propagate peak and background variance in quadrature. Neighbor
peak regions can be excluded.

Results contain `H`, `K`, `L`, `I`, `dI`, `Background`, `I/dI`, peak/background
coverage, and status. Dataset metadata must declare `signal_semantics` as
`density` or `bin_integral`; quantitative integration refuses `unknown`.

## Spectral integration

The **INS absolute conversion** operation creates a derived dataset before any
integration. It converts a measured signal to either an absolute differential
cross section in `barn/(sr meV)` or dynamic susceptibility in `mu_B^2/meV`, per
the selected normalization basis. Supply the measured-signal scale in signal
units per `barn/(sr meV)` (for example from vanadium or nuclear-Bragg
normalization), temperature, magnetic form factor, and polarization convention.
The output records the complete source and target conventions in provenance.
The converter uses

```text
d2sigma/dOmega/dE = (kf/ki) (0.07265 barn/mu_B^2)
                     |f(Q)|^2 P(Q) chi''
                     / [pi (1 - exp(-E/k_B T))].
```

The cross section carries steradians; `chi''` does not. A result is marked
absolute only when its normalization basis is known.

Physical total-moment, QFI, and static-susceptibility results require absolute
scale, a known normalization basis, and an explicit `spectral_convention`.
Uncalibrated data may use `weighted_integral_arbitrary_units`.

nfit uses `E = hbar*omega` in meV and `chi''(Q,E)` per unit energy:

```text
total moment:  coth(E / 2 k_B T) / pi
QFI:           4 tanh(E / 2 k_B T) / pi
static chi:    2 / (pi E)
```

The convention records representation, units, normalization, moment units,
form-factor/polarization/Bose/kinematic states, absolute scale, and Landé
factor. Corrections apply only when marked `included`. QFI from `mu_B^2` data
requires `g_factor` and divides by `g^2`; nfit never assumes `g=2`. Invalid
form-factor divisions become missing coverage.

When `spin_S` is supplied, nfit also emits normalized QFI using
`nQFI = f_Q / (12 S^2)`. The formula is stored in provenance; nfit does not
infer `S` from an ion label or automatically claim an entanglement depth.

Energy reduction returns a scalar, a `PointListData` curve, or an `MDHistoData`
map according to remaining dimensionality. Maps expose energy and weighted
coverage in the viewer. Per-zone total moment uses reciprocal-Bravais-lattice
Wigner-Seitz assignment with deterministic subvoxel boundary sampling. Partial
and coverage-corrected estimates remain distinct.

## Script API

Pure functions such as `integrate_bragg_peaks()` and
`spectral_energy_reduce()` do not construct Qt objects. Registry workflows use
`available_analysis_types()`, `default_analysis_parameters()`,
`validate_analysis()`, and `run_analysis_operation()`.

Future raw-TOF reduction and general dataset subtraction remain upstream and
can emit reduced derived datasets through the same recipe, artifact,
fingerprint, and coverage system.
