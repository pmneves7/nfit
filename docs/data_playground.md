# Data Playground

The Data Playground runs non-fitting operations on reduced datasets while
keeping inputs unchanged. Open it from a workspace, dataset, or `Analyses` node
with **Open in Data Playground**. Successful dataset-like results appear in
`Datasets / Derived data`; the authoritative recipe remains under `Analyses`.

Projects must be saved before a run. Array outputs live in
`<project>.nfit-assets/analyses/<analysis-id>/`; project JSON stores relative
manifests and provenance. **Save As** copies the asset tree. Changing a recipe
or input marks the latest result stale without deleting it.

## Bragg integration

Bragg integration accepts three momentum dimensions, or 4D data with an
explicit elastic energy window. Peaks come from space-group-filtered crystal
HKLs or supplied H, K, L positions. Methods are exact-overlap HKL boxes,
deterministic subvoxel ellipsoids, and local Gaussian fits. Regions may use HKL
or inverse-angstrom widths. Optional shells subtract a volume-scaled local
background and propagate peak and background variance in quadrature. Neighbor
peak regions can be excluded.

Results contain `H`, `K`, `L`, `I`, `dI`, `Background`, `I/dI`, peak/background
coverage, and status. Dataset metadata must declare `signal_semantics` as
`density` or `bin_integral`; quantitative integration refuses `unknown`.

## Spectral integration

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
