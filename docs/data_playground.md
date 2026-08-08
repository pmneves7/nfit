# Analysis Window

The Analysis Window runs non-fitting operations without changing the input
datasets. Open it from a workspace, dataset, or `Analyses` node. Results appear
under `Datasets / Derived data`; the saved recipe and provenance remain under
`Analyses`.

Save the project before running an analysis. Materialized array outputs are
stored inside the `.nfit` project under `assets/analyses/<analysis-id>/`, so
moving or using **Save As** on the project keeps its analysis results with it.
Clone and histogram-arithmetic outputs are instead saved as virtual,
source-linked recipes and are recomputed from their underlying data when used.

Every operation receives the same prepared dataset used for viewing and
fitting. Dataset and inherited masks, rebinning, attached backgrounds, scale,
and the selected physical channel therefore affect analyses consistently.
Enabled group composites also appear as inputs, labeled **live composite**.

## Histogram arithmetic

**Histogram arithmetic** creates an editable sum or difference of two
identically binned histograms. Choose the left and right inputs, select addition
or subtraction, and set the right-hand scale $a$. Subtraction evaluates
$I_{\rm left}-aI_{\rm right}$ and propagates statistically independent
variances as

$$
\sigma^2=\sigma_{\rm left}^2+a^2\sigma_{\rm right}^2.
$$

Both source datasets or live composites remain independently viewable. The
derived dataset panel begins with editable left source, right source,
arithmetic operation, and right-hand scale controls. Its output binning and
symmetry are pushed into both underlying source reductions before the
arithmetic is evaluated; nfit does not rebin cached source composites. The
derived dataset owns its masks independently of the source datasets.

## Dataset clones

**Dataset clone** creates a traceable comparison branch from one dataset or
live composite. The derived output starts disabled with fit weight zero, while
its source remains independently viewable and usable for fitting. Configure
masks, rebinning, and reciprocal-space symmetry on the derived dataset without
changing the source. The source selector appears at the top of the derived
dataset panel and remains editable after creation.

For a live-composite source, the clone retains the source run membership,
calibration scales, and background recipe, but owns its masks and output grid.
nfit applies that grid directly to the underlying event or point data; it does
not reuse the source composite's bins, symmetry, or fit/display masks. This
allows, for example, a disabled, fit-weight-zero, fine m-3m-symmetrized and
unmasked comparison beside a coarser unsymmetrized dataset used for fitting.
Changing a source or recipe invalidates the derived cache and recomputes it on
the next view, fit, or explicit rebin.

## Curie--Weiss fitting

**Curie--Weiss fit** accepts absolute molar susceptibility in `cm^3/mol` or
rationalized-SI `m^3/mol` and fits

$$
\chi(T)=\frac{C}{T-\theta_{\rm CW}}.
$$

$T$ is temperature, $C$ is the Curie constant, and $\theta_{\rm CW}$ is the
Curie--Weiss temperature. Set `Tmin` and `Tmax` to select the fit interval.
nfit fits $\chi$, rather than $1/\chi$, to avoid bias from transforming the
uncertainties.

The report gives $C$ in `cm^3 K/mol`, $\theta_{\rm CW}$ in K, their covariance,
and the effective moment. To display the dimensions explicitly, the
rationalized-SI relation is

$$
\frac{\mu_{\rm eff}}{\mu_B}
=\frac{1}{\mu_B}
\sqrt{\frac{3k_BC_{\rm SI}}{\mu_0N_A}},
$$

where $C_{\rm SI}$ is the Curie constant converted to `m^3 K/mol`, $k_B$ is
the Boltzmann constant, $N_A$ is the Avogadro constant, $\mu_0$ is the vacuum
permeability, and $\mu_B$ is the Bohr magneton. nfit performs the fit and the
algebraically equivalent moment calculation in CGS units. The diagnostic can
display either susceptibility or inverse susceptibility with the fitted
interval marked.

For MPMS moment data, first enable susceptibility and sample normalization in
the dataset panel. The same expression is also available as a model component
for simultaneous fitting.

## Low-temperature heat capacity

**Low-temperature C/T fit** fits
$C/T=\gamma+\beta T^2$ between `Tmin` and `Tmax`. It reports the electronic
coefficient $\gamma$, phonon coefficient $\beta$, and Debye temperature for
the supplied number of atoms per formula unit. See
[Heat-capacity data and models](heat_capacity.md) for definitions and units.

## Bragg integration

Bragg integration accepts three momentum dimensions or a four-dimensional
momentum-energy dataset. For 4D data, blank energy bounds use the bin nearest
zero; two supplied bounds define an elastic window.

Peaks may come from space-group-filtered crystal reflections or an explicit
HKL list. Available methods are:

| method | integration region |
| --- | --- |
| Gaussian (default) | local 3D Gaussian plus constant or linear baseline |
| HKL box | exact overlap with a rectangular region |
| ellipsoid | deterministic subvoxel overlap |

Widths may be in HKL or inverse angstroms. An optional shell estimates a
volume-scaled local background, propagates its variance, and can exclude
neighboring peak regions. Dataset `signal_semantics` determines whether nfit
integrates a density over physical bin volume or treats each bin as an
already-integrated value. An `unknown` convention blocks quantitative
integration.

Results retain HKL, intensity $I$, standard uncertainty $\delta I$,
$I/\delta I$, coverage, acceptance, and rejection reasons. Box and ellipsoid
results report the measured raw and shell-background integrals over the peak
region. For a Gaussian fit, $I$ is the analytic integral of the fitted Gaussian
over all reciprocal space. `FitWindowRaw`, `FitWindowBackground`, and
`FitWindowPeak` are separate finite integrals over the measured voxels used by
the fit; their domain is therefore explicit. Gaussian results also retain
fitted centers, widths, amplitude, baseline density, and reduced chi-squared.
Rejected reflections remain visible for diagnosis. The status bitmask is:

| bit | reason |
| ---: | --- |
| 1 | low peak coverage |
| 2 | low background coverage |
| 4 | low or non-finite $I/\delta I$ |
| 8 | excessive background |
| 16 | Gaussian fit failure |

**Export .int** writes accepted reflections as `H K L I dI`, rounding HKL to
the nearest integers while retaining floating-point intensity and uncertainty.
Diagnostics show local planes and profiles with the integration regions;
Gaussian runs also show fitted profiles. **View input with peaks** marks
accepted and rejected reflections in the data viewer. **Add to datasets**
creates a disabled Bragg reflection dataset when the table is needed by
another workflow.

## Bose--Einstein elastic separation

This operation accepts two identically binned datasets at distinct positive
temperatures $T_1$ and $T_2$. At each nonzero transferred energy it solves

$$
I_j(\mathbf Q,E)
=I_{\rm elastic}(\mathbf Q,E)+f(E,T_j)A(\mathbf Q,E),
\qquad j\in\{1,2\},
$$

where $I_j$ is the measured intensity, $I_{\rm elastic}$ is the
temperature-independent part, $A$ is the underlying inelastic amplitude, and
$f$ is $n(E,T)+1$ on the neutron-energy-loss side or $n(|E|,T)$ on the
energy-gain side. Here
$n(E,T)=[\exp(E/k_BT)-1]^{-1}$ is the Bose occupation.

The main result is the inelastic signal at the primary dataset temperature;
a second result stores $I_{\rm elastic}$. The exactly zero-energy bin is
assigned to the latter because the Bose factor is singular. Independent input
variances are propagated through both linear combinations.

## Spherical averaging

**Spherical average** converts a single-crystal inelastic histogram to a
powder $|\mathbf Q|,E$ dataset: the intensity expected after averaging the
measured single-crystal volume over directions, as for a ground powder. It
preserves energy bins and weights every source voxel by its physical
reciprocal-space volume and fractional overlap with each spherical shell.
Odd-grid subvoxel sampling estimates boundary overlaps. Uncertainties are
propagated with the same geometric weights; statistical precision does not
change the physical average. If the source stores bin integrals, nfit first
divides by each four-dimensional bin volume; an unknown signal convention is
rejected.

The `powder_coverage` channel gives the measured reciprocal volume divided by
the full shell volume. Values below one mean that some powder orientations were
not represented by the single-crystal data, so the result averages only the
measured portion. A lattice or UB matrix is required when the source momentum
coordinates are in reciprocal-lattice units.

## Angle-energy background

This operation estimates a powder background from two or more MDEvent runs.
Each run is reduced to a common $|\mathbf Q|,E$ grid. nfit divides the summed
intensity and its uncertainty by proton charge and by
$\Delta|\mathbf Q|\,\Delta E$, so the result is an intensity density rather
than a bin integral. In each bin, it averages the lowest selected fraction of
run densities; the default is 20%. Statistical variances are propagated, but
uncertainty in choosing the order-statistic subset is not included.

The method follows Shiver's Mantid
[`GenerateGoniometerIndependentBackground`](https://docs.mantidproject.org/v6.11.0/algorithms/GenerateGoniometerIndependentBackground-v1.html)
workflow. Attach the result through a dataset's **Backgrounds** branch.

## Absolute conversion and spectral integration

Ordinary inelastic datasets can expose paired cross-section and $\chi''$
channels directly. **INS absolute conversion** instead creates an immutable
derived dataset for a downstream analysis recipe. It requires the measured
signal scale, temperature, normalization basis, form factor, and polarization
convention.

For moment susceptibility in `mu_B^2/meV`, the conversion is

$$
\frac{d^2\sigma}{d\Omega\,dE}
=\frac{k_f}{k_i}
\frac{0.07265\ {\rm barn}/\mu_B^2}{\pi}
|f(Q)|^2P(\mathbf Q)
\frac{\chi''(\mathbf Q,E)}{1-e^{-E/(k_BT)}}.
$$

$k_i$ and $k_f$ are the incident and final neutron wavevector magnitudes,
$f(Q)$ is the magnetic form factor, and $P$ is the polarization factor. For a
spin response, replace $\chi''$ by $g^2\chi''_s$. The complete convention is
defined in [Physics conventions](physics_conventions.md). When the stored cross
section includes $k_f/k_i$, its spectral convention must also provide the fixed
incident energy for direct geometry or fixed final energy for indirect
geometry.

Spectral reductions use these energy kernels:

| result | kernel multiplying $\chi''(\mathbf Q,E)\,dE$ |
| --- | --- |
| total moment | $\coth[E/(2k_BT)]/\pi$ |
| quantum Fisher information | $4\tanh[E/(2k_BT)]/\pi$ |
| static susceptibility | $2/(\pi E)$ |

Physical results require absolute scale, known normalization, and an explicit
spectral convention. QFI from moment units requires `g_factor` and divides by
$g^2$; nfit never assumes $g=2$. If spin length $S$ is supplied, nfit also
reports normalized QFI $f_Q/(12S^2)$, where $f_Q$ is the QFI density.

Energy reduction returns a scalar, curve, or map according to the remaining
dimensions. Per-zone integration assigns reciprocal space through
Wigner--Seitz cells and reports coverage separately from any
coverage-corrected estimate.

## Script API

Pure functions such as `integrate_bragg_peaks()` and
`spectral_energy_reduce()` do not construct Qt objects. Registry workflows use
`available_analysis_types()`, `default_analysis_parameters()`,
`validate_analysis()`, and `run_analysis_operation()`.
`prepare_analysis_inputs()` constructs the canonical prepared inputs for a
saved recipe, and `run_project_analysis()` runs it directly. Right-click a
saved analysis to copy or save a readable script containing its full dataset
dependency closure. For live-composite inputs, that script reloads the saved
project and reruns the editable project recipe so hierarchical dependencies are
not flattened into snapshots.

Native raw-TOF reductions use the same recipe, artifact, fingerprint, and
coverage system. Future analysis directions are listed in
[Planned features](planned_features.md).
