# Importing and preparing data

This page covers the project-explorer controls used before fitting. Scientific
definitions and unit conversions are collected in
[Physics conventions](physics_conventions.md).

## Importing files

Select a workspace or dataset group, choose the data type and importer, then use
**Add files**. Numbered acquisitions can instead be generated from a path,
prefix, suffix, and numor expression:

```text
409981:409995
409981:409992,409994:409995
409981:3:409995
```

These mean an inclusive range, two comma-separated ranges, and a stepped range.
nfit checks that all generated paths exist before adding any of them.

Nested dataset groups have their own enabled state and optional composite.
Their bulk controls can set every descendant's fit weight or calibration scale,
and **Share fitted scale** can tie those dataset scales during fitting.
Disabling a group excludes its descendants without changing their individual
enabled states.

Run-heavy collections are compact in the project explorer. Collections with
more than twelve direct datasets start collapsed and expose lazy pages of 50
runs; expand a page to create those individual run nodes for inspection or
editing. **Expand groups** opens the structural hierarchy but deliberately
leaves run pages collapsed. The collection details panel shows 20 direct runs
per page. A parent with child collections lists those immediate children rather
than flattening every descendant run into one long table. Empty **Masks** and
**Backgrounds** folders are omitted; select the dataset or collection and use
the add action to create the first item. Background containers use a coral
folder accent so they remain distinct from ordinary, mask, model, fit,
analysis, and plot folders throughout this hierarchy. Analyses and Plots also
use their own folder colors for quick navigation.

### Point-list data

MPMS magnetization, PPMS heat capacity, and powder diffraction imports retain
their source columns and header metadata. The dataset panel identifies
coordinates and measured channels and lets the user correct quantity types and
units.

For MPMS data, sample mass and molar mass seed the absolute normalization.
Choose whether the plotted and fitted observable is moment or susceptibility
and declare the input moment and field units. Molar susceptibility is available
in `cm^3/mol` or rationalized SI `m^3/mol`; the conversion includes
$4\pi\times10^{-6}$.

See [Heat-capacity data and models](heat_capacity.md) for PPMS normalization
and heat-capacity fit components.

### Digitized powder INS

The powder digitizer importer accepts:

- a headerless `x, signal, sigma` cut; or
- a `y\x` matrix with $Q$ columns, energy rows, and signal values.

For a cut, specify whether `x` is $Q$ at fixed energy or energy at fixed $Q$.
For every file, supply temperature, the signal representation, units, and
normalization basis. A matrix has no uncertainty layer, so **Map σ** supplies a
uniform one-sigma uncertainty. NaNs are masked.

With a complete convention, nfit exposes paired cross-section and $\chi''$
channels without changing the imported values. Powder Heisenberg models average
the single-crystal response over momentum directions and require lattice
parameters for the reciprocal-coordinate conversion.

## NIST NCNR MACS NeXus data

The MACS importer recognizes the instrument from NeXus content rather than the
filename suffix. Selecting one or many MACS files creates two sibling dataset
collections:

- **MACS SPEC** contains the fixed-final-energy, energy-analyzed detector
  stream as HKLE points. Energy transfer is $\Delta E=E_i-E_f$ in meV.
- **MACS DIFF** contains the unanalysed remainder of the scattered beam. It is
  not combined with SPEC. Following DAVE, nfit assigns DIFF coordinates using
  the elastic approximation $E_f=E_i$ and $\Delta E=0$. This is a coordinate
  approximation, not a claim that DIFF resolved the outgoing energy.

Both streams use the 20 MACS angular offsets
$2\theta_d=\mathrm{Kidney}-76^\circ+8^\circ d$, where $d=0,\ldots,19$.
nfit converts $E_i$ and $E_f$ to wavevectors with
$E=2.0721246 k^2$ (meV for $k$ in inverse angstrom), rotates the resulting
$\mathbf Q=\mathbf k_i-\mathbf k_f$ by the recorded sample A3 angle, and solves
for HKL using the lattice and orientation vectors stored in the file. The A3
offset is an explicit import setting because a correction used during analysis
may not be stored in the acquisition file. For the supplied LiV2O4 example,
enter `66.5` degrees to reproduce the shown MSlice orientation.

Counts and Poisson one-sigma uncertainties are multiplied by the stored
per-stream detector-efficiency corrections and normalized to the chosen monitor
target; the default is $10^6$ monitor counts. A zero-count point receives the
uncertainty corresponding to one count so it cannot acquire infinite fit
weight.

### MACS detector masks

File masks remain part of the immutable imported point mask. SPEC additionally
uses three analyzer checks:

1. the NeXus `specDetector/roiMask`;
2. DAVE's per-scan analyzer-angle comparison, with a default one-degree
   tolerance for analyzer-two-theta files; and
3. an unresponsive-channel test based on nonzero-count occupancy across at
   least 20 scan points.

Every rejected analyzer is retained in the arrays but marked ineligible, and
its 1-based channel number, reason, and measured occupancy are stored in
metadata. Additional 1-based SPEC channels can be entered manually. These
analyzer masks do not propagate to DIFF: its independent NeXus ROI mask is used.
In `Ef3p7_et_1.2_244.nxs.ng0`, SPEC channel 19 is identified as unresponsive
because only 6.9% of its scan points are nonzero; the corresponding DIFF channel
remains valid.

Each SPEC and DIFF collection has its own live composite enabled. Importing a
batch therefore prepares one SPEC composite and one DIFF composite, without
mixing their different energy semantics. Configure the HKLE output grid on each
collection and choose **Rebin now**. The same operation is scriptable:

```python
from nfit import DataGroup, import_dataset_paths

files = ["run244.nxs.ng0", "run245.nxs.ng0"]
workspace = DataGroup("MACS series")
options = {
    path: {"a3_offset_deg": 66.5, "monitor_target": 1_000_000.0}
    for path in files
}
entries = import_dataset_paths(
    workspace,
    files,
    data_type="single_crystal_inelastic",
    importer_options=options,
)
```

The import options, selected stream, geometry constants, normalization, and
mask provenance are stored with every entry and reproduced by dataset workflow
scripts.

## Compatible direct-geometry spectrometer data

This importer supports raw event NeXus files from compatible direct-geometry
spectrometers, including CNCS, HYSPEC, and SEQUOIA. It requires the expected
event banks and run logs plus an embedded Mantid instrument definition. It is
not a universal direct-geometry NeXus importer.

Compatible runs are stored as a file-backed dataset group. The shared setup
holds the UB matrix, detector mask, processed vanadium file, and optional
$E_i$ and $T_0$ overrides. The accepted energy-transfer interval is expressed
as fractions of each run's $E_i$ and defaults to
$-0.95E_i\leq\Delta E\leq0.95E_i$. The same limits define event selection and
detector-trajectory normalization. Individual detector-event runs are not
plotted directly; enable the group composite and rebin them to an HKLE
histogram.

nfit streams detector banks and event chunks rather than loading the complete
event table. It obtains detector geometry and flight paths from the embedded
instrument definition, applies available detector-efficiency and $k_i/k_f$
corrections, converts $\mathbf Q=\mathbf k_i-\mathbf k_f$ to HKL, and normalizes
by retained proton charge and detector-trajectory coverage.

### Raw TOF reduction sequence

For each enabled run, nfit:

1. reads run logs, instrument geometry, detector IDs, and orientation;
2. resolves $E_i$ and $T_0$ from explicit overrides, monitor analysis, or an
   instrument time-zero formula;
3. combines detector-mask and processed-vanadium exclusions;
4. rejects bad pulses using the configured charge threshold;
5. converts accepted event TOF to final energy, $\Delta E$, and sample-frame
   momentum, then applies the configured $\Delta E/E_i$ limits;
6. applies detector-efficiency and optional $k_i/k_f$ corrections;
7. bins corrected events and their variances; and
8. divides by independently accumulated trajectory coverage.

Bins without detector coverage are masked. Covered bins with no events are
measured zeros and retain a finite uncertainty.

## MDEvent data

Importing MDEvent NeXus creates one lightweight entry per experiment or run.
The shared event table stays on disk. Enable **Combine datasets**, configure the
four output axes and bounds, and select **Rebin now** to create the normalized
HKLE dataset.

Each coordinate-axis row is an HKLE basis vector. The four rows must be linearly
independent; momentum rows use only H, K, and L, while the energy row uses E.
Bounds and resolution are expressed in that basis.

An incident-energy override changes normalization trajectories. A time-zero
override cannot move coordinates already stored in an MDEvent workspace.

### Measured-zero uncertainties

Event histograms distinguish:

- bins with one or more accepted events;
- covered bins with zero accepted events; and
- bins with no detector coverage.

For nonempty bins, nfit stores

```text
signal = sum(signal_i) / D
sigma  = sqrt(sum(errorSquared_i)) / D
```

where $D$ is the normalization denominator. An empty covered bin has no event
variance, so nfit uses

```text
sigma_zero = 1.29 * representative_event_scale / D
```

The representative scale is the root-mean-square event uncertainty over the
accepted dataset. The factor 1.29 is the 68.27% Feldman--Cousins upper endpoint
for zero observed events and zero known background. Uncovered bins remain
masked. This fitting convention avoids assigning infinite weight to a measured
zero; it does not make the underlying Poisson interval symmetric.

## UB matrices

**Crystal orientation > UB setup** edits lattice parameters, orientation
vectors `u` and `v`, and the full $3\times3$ UB matrix. **Calculate from lattice
and u/v** places `u` along the incident beam and uses `u` and `v` to define the
horizontal scattering plane.

UB maps `[h,k,l]` to reciprocal momentum in inverse angstrom, with
$|\mathbf Q'|=1/d$. **UB from NeXus** reads embedded orientation metadata.
**UB from ISAW** and **Save ISAW** use the conventional transposed three-row
ISAW representation.

Apply the dialog to a dataset for a dataset-specific orientation or to a group
for shared orientation and composite HKL conversion.

## Dataset details and physical conventions

Selecting a dataset shows its axes, source, crystal information, data summary,
and imported metadata. The **Fit bins** count uses the same prepared view as the
optimizer, including masks, invalid values, and invalid uncertainties.

For inelastic data, **INS representations** records:

1. whether the imported signal is intensity/cross section or $\chi''$;
2. its units and amount-of-sample normalization;
3. form-factor and polarization states;
4. whether the response uses spin or magnetic-moment units; and
5. whether $k_f/k_i$ remains included.

Set temperature and field in **Conditions**. Imported moment response already
contains $g^2$; imported spin response receives it once. The full conversion is
given in [Physics conventions](physics_conventions.md).

`Save dataset` and `Save rebin to disk` write portable `.npz` archives
containing axes, values, uncertainties, masks, metadata, and dataset conditions.

## Rebinning and composites

Rebinning affects both viewing and fitting. It supports:

- **Step** or **Bins** resolution;
- inverse-variance or uniform averaging;
- fractional bin overlap;
- optional nonuniform edges on any individual output axis;
- a minimum effective source-sample threshold;
- projected HKLE coordinate bases;
- point-group symmetry expansion; and
- bounded batch sizes for temporary working memory.

An imported MDEvent collection can reduce directly either to a projected
single-crystal HKLE histogram or to a powder $|\mathbf Q|,E$ histogram. Both
paths accumulate event weights and use proton charge plus detector-trajectory
coverage for normalization. The powder path bins radially without first
allocating a sparse four-dimensional volume, which is the appropriate route
for a fixed-angle environment/background run.

Masks are applied before binning. Automatic rebinning is used for modest jobs;
larger jobs remain pending until **Rebin now** or until an operation requires
current rebinned data. The batch target controls temporary work, not the
persistent output-grid allocation.

**Minimum coverage** masks an output bin when the measured source support
occupies less than the selected fraction of its requested geometric volume.
The default is 0.9. Coverage is saved as an independent auxiliary channel, and
the resulting coverage mask is used consistently for viewing, histogram
exports, and fitting. Changing the cutoff does not rewrite the source file or
nfit masks.

**Edges (optional)** accepts a strictly increasing list for one axis, such as
`[-2, -1, 0, 0.5, 2]`. That axis uses the listed edges while blank axes retain
the selected uniform **Step** or **Bins** resolution. Mixed grids are useful for
irregular energy-transfer sampling and for nonuniform temperature or magnetic-
field series. **Minimum samples** independently masks bins with too few source
observations; fractional binning counts the summed fractional contribution.
This statistical threshold complements geometric minimum coverage rather than
replacing it.

For gridded inputs, nfit propagates native-bin volume and any existing
fractional coverage through the rebin. A legacy histogram without fractional
coverage falls back to covered/uncovered native bins. Point collections that
do not define source-cell geometry cannot provide sub-bin geometric coverage;
their populated-bin behavior is unchanged.

**Copy settings** and **Paste settings** transfer compatible rebin recipes.
**Create dataset from rebin** materializes an independent project dataset.
For a dataset collection, **Create dataset from composite** stores the current
composite inside the `.nfit` archive. Project-owned materializations load lazily
when the project is reopened and can serve as a downstream composite input or
background without keeping the source event files in memory.

A composite combines compatible enabled descendants into one effective
dataset. Each source signal and uncertainty receive its positive dataset
calibration scale, then the sources are averaged using their fit weights. Use a
dataset or group background for subtraction. Gridded MDEvent outputs also carry
their detector-trajectory normalization denominator. Select **Normalization
denominator** averaging when combining separately reduced angle ranges to
reproduce numerator/denominator accumulation rather than an arithmetic or
inverse-variance mean. When a composite is active, its constituents are not
fitted separately.

A parent collection with no direct datasets can combine the live composites of
its enabled child collections. This is useful for keeping separate angle ranges
such as `34` and `70` independently configurable while exposing their corrected
combination as one dataset. Child masks, rebin settings, backgrounds, and scales
remain visible in the project tree; changing one invalidates the parent result.

## Masks

File masks and nfit masks remain separate but are combined for fitting. All GUI
mask parameters describe the region to exclude:

- coordinate and energy/$|Q|$ ranges;
- projected boxes and ellipsoids; and
- acoustic-phonon cones around one or more Bragg centers.

Zero-width defaults are inactive. **Invert** excludes the complement.
**Additive** removes a region from earlier nfit-mask contributions but cannot
undo a file mask. Disabled masks do not participate.

Large datasets may use manual mask application. Opening the viewer, fitting, or
pressing **Apply masks now** always resolves pending masks first.

## Backgrounds

A dataset or composite can subtract one or more gridded backgrounds. Each
background has an enabled state and scale. Powder backgrounds also select
linear or nearest interpolation.

For a single-crystal target, nfit interpolates the powder background in
$|Q|$ and energy and propagates independent uncertainties:

$$
\sigma_{\rm corrected}^2
=\sigma_{\rm data}^2+a^2\sigma_{\rm background}^2,
$$

where $a$ is the background scale. Values outside the background domain are
masked rather than extrapolated.

For a single-crystal background, reduce sample and background independently
onto identical axes and bins. nfit then subtracts corresponding bins without
interpolation and uses the same variance equation above. A mismatched shape,
axis name, unit, or edge is rejected explicitly. This supports environments
measured over comparable angle ranges. A group background can point directly to
another collection's enabled **live composite**, so sample and background
recipes remain separately visible and editable without materializing an
intermediate snapshot.

A dataset background is applied before that dataset's scale. A group background
is subtracted once after the group composite is formed. Use **Spherical
average** in the [Analysis Window](data_playground.md) to create a powder
background from single-crystal data.
