# Analysis Window

The Analysis Window runs non-fitting operations on reduced datasets while
keeping inputs unchanged. Open it from a workspace, dataset, or `Analyses` node
with **Open Analysis Window**. Successful dataset-like results appear in
`Datasets / Derived data`; the authoritative recipe remains under `Analyses`.

The configuration and output areas are separated by a draggable horizontal
divider. The initial layout favors the configuration area; drag the divider to
make more room for a results table or diagnostic view when needed.

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

The Bragg controls are grouped into peak selection, elastic volume, integration
region, background, quality filters, Gaussian fit, and numerical settings.
Controls that do not apply to the selected method are hidden. The default is an
ellipsoidal integration with a local background shell, partial-edge reporting,
and moderately permissive coverage thresholds so a first run normally produces
inspectable results rather than silently discarding most reflections.

Results contain `H`, `K`, `L`, `I`, `dI`, `Background`, `I/dI`, peak/background
coverage, acceptance, and a rejection status bitmask. Gaussian results also
retain fitted center, amplitude, baseline, three widths, and reduced chi-square.
Optional quality filters reject peaks below a selected `I/dI`, above an
absolute background threshold, or below either coverage threshold. Rejected
measurements remain in the output for diagnosis; they are not replaced by
`NaN`. Status bits identify low peak coverage (1), low background coverage (2),
low or non-finite `I/dI` (4), excessive background (8), and Gaussian failure
(16). Multiple reasons are combined by addition.

The **Results** and **Diagnostics** tables can be sorted in ascending or
descending order by clicking a column header. Both keep `I` and `dI` beside
the H, K, L indices. The **Export .int** action writes accepted reflections as
a CSV-formatted `.int` file with `H,K,L,I,dI` columns. Exported H, K, and L
values are rounded to their nearest integer, while `I` and `dI` retain their
floating-point precision. The **Diagnostics** tab
keeps a second selectable table beside three local data planes and three axis
profiles. Integration and background regions are drawn over the data; Gaussian
runs also overlay the fitted profiles. These diagnostics are rebuilt from the
saved recipe, input dataset, and output artifact, so they remain available when
the project is reopened. **View input with peaks** opens the ordinary data
viewer with accepted reflections as green open circles and rejected reflections
as red crosses.

The **Analysis progress** window records the current reflection, its HKL,
coverage, `I/dI`, and running accepted/rejected counts. On completion it stays
open with the generated, integrated, accepted, and rejected reflection totals;
use **Close** when you have finished reviewing the activity log.

Analysis tables remain owned by the analysis and are not automatically inserted
into the workspace's fitting datasets. **Add to datasets** explicitly creates a
disabled **Bragg reflections** dataset when a table is needed elsewhere.

Dataset metadata may declare `signal_semantics` as `density` or `bin_integral`.
Density is the nfit default because normalized rebinning returns a
variance-weighted mean rather than a sum. Mantid MDHistoWorkspace imports,
native MDEvent/raw-TOF reductions, normalized rebins, and legacy archives
without an explicit convention therefore use `density`. Quantitative
integration multiplies these values by the physical bin volume. Use the dataset
**Signal convention** control to select `bin_integral` only when each upstream
array value is already the total intensity contained in that bin. An explicitly
`unknown` convention still blocks quantitative integration.

## Bose-Einstein elastic separation

**Bose-Einstein elastic separation** accepts two identically binned histogram
datasets. Select the dataset whose temperature should define the output as the
primary input and the other temperature as the secondary input. Both datasets
must have positive temperatures in their **Conditions** fields, and
the temperatures must differ.

For each nonzero energy-transfer bin, nfit solves

```text
I_1(Q,E) = I_elastic(Q,E) + f(E,T_1) A(Q,E)
I_2(Q,E) = I_elastic(Q,E) + f(E,T_2) A(Q,E)
```

where `f(E,T)` is `n(E,T)+1` on the neutron-energy-loss side and `n(|E|,T)`
on the energy-gain side. The main output is `f(E,T_1) A`, the inelastic signal
at the primary dataset temperature; a second output stores the inferred
temperature-independent component. Exactly zero-transfer bins are assigned to
the temperature-independent output because the Bose factor is singular there.
Input variances are propagated through the two linear combinations under the
assumption that the measurements are statistically independent.

## Spherical averaging

**Spherical average** converts a single-crystal inelastic histogram to a
powder inelastic `|Q|, DeltaE` dataset. The energy bins are preserved and the
number of radial bins is editable. Source bins are combined with
inverse-variance weights; the output uncertainty is the standard uncertainty
of that weighted mean. A lattice or UB matrix is required when the source
momentum coordinates are in reciprocal-lattice units.

## Angle-energy background

**Angle-energy background** operates on two or more MDEvent runs. Use **Select
runs...** to choose the sample-rotation measurements. Each run is reduced to a
common `|Q|, DeltaE` grid and normalized by its proton charge. In every bin,
the run intensities are sorted and the lowest fraction is averaged; the default
is 20 percent. This follows the per-bin low-intensity selection used by
Shiver's Mantid
[`GenerateGoniometerIndependentBackground`](https://docs.mantidproject.org/v6.11.0/algorithms/GenerateGoniometerIndependentBackground-v1.html)
workflow. The
statistical variances of the selected runs are propagated, while the additional
uncertainty from deciding which runs belong to the selected order statistic is
reported as excluded from the error model.

The result is a powder inelastic dataset that can be inspected directly or
attached to another dataset through its **Backgrounds** branch. For raw events,
nfit's current implementation uses proton-charge normalization; future
instrument-specific reduction can replace that normalization without changing
the saved analysis recipe or background interface.

## Spectral integration

The dataset details panel now provides the ordinary import/view/fit route:
temperature-aware inelastic datasets expose paired scattering-cross-section and
$\chi''$ channels in arbitrary or absolute units. Use that path when the same
dataset should be switchable between representations during plotting or
fitting.

The **INS absolute conversion** operation creates a derived dataset before any
integration. It converts a measured signal to either an absolute differential
cross section in `barn/(sr meV)` or dynamic susceptibility in `mu_B^2/meV`, per
the selected normalization basis. Supply the measured-signal scale in signal
units per `barn/(sr meV)` (for example from vanadium or nuclear-Bragg
normalization), temperature, magnetic form factor, and polarization convention.
The output records the complete source and target conventions in provenance.
Use this analysis operation when an immutable converted dataset is desired for
downstream analysis recipes rather than a paired channel on the imported
dataset.
The converter uses

```text
d2sigma/dOmega/dE = (kf/ki) (0.07265 barn/mu_B^2)
                     |f(Q)|^2 P(Q) chi''
                     / [pi (1 - exp(-E/k_B T))].
```

This form takes `chi''` in `mu_B^2/meV`. For a spin response, replace
`chi''` by `g^2 chi''_spin`, equivalently using
`(gamma r_0)^2 (g/2)^2`. The default scalar convention is one isotropic
Cartesian component and therefore `P = 2`; `P = 2/3` denotes a
three-component trace. The cross section carries steradians; `chi''` does not.
A result is marked absolute only when its normalization basis is known.

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

Future raw-TOF reduction and single-crystal-to-single-crystal subtraction can
emit corrected derived datasets through the same recipe, artifact, fingerprint,
and coverage system.
