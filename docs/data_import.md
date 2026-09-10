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

Delimited text import recognizes instrument preambles and quoted comma- or
tab-separated labels. Every data row must have the same number of columns as
the header. Blank fields and common missing-value markers such as `N/A` and
`NaN` become missing numeric values; malformed text reports its source line and
column instead of silently changing the dataset shape.

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
  stream as HKLE points. Energy transfer is $\Delta E=E_i-E_f$ in meV. As in
  DAVE, $E_f$ is reconstructed from the mean aligned analyzer angle and the
  analyzer crystal spacing as
  $E_f=81.8042/[2d_A\sin(A_5)]^2$, where $d_A$ is the analyzer-plane spacing in
  Å and $A_5$ is the analyzer Bragg angle in degrees. This avoids stale common
  final-energy logs; the recorded value is retained separately in metadata for
  inspection.
- **MACS DIFF** contains the unanalysed remainder of the scattered beam. It is
  not combined with SPEC. Following DAVE, nfit assigns DIFF coordinates using
  the elastic approximation $E_f=E_i$ and $\Delta E=0$. This is a coordinate
  approximation, not a claim that DIFF resolved the outgoing energy.

Both streams use the 20 MACS angular offsets
$2\theta_d=\mathrm{Kidney}-76^\circ+8^\circ d$, where $d=0,\ldots,19$
is the detector index, $2\theta_d$ the scattering angle, and `Kidney` the
recorded analyzer-bank angle in degrees.
nfit converts $E_i$ and $E_f$ to wavevectors with
$E_{\rm kin}=[\hbar^2/(2m_n)]k^2$ with
$\hbar^2/(2m_n)\simeq2.0721246$ meV Å$^2$; $m_n$ is neutron rest mass,
$E_{\rm kin}$ means $E_i$ or $E_f$ rather than energy transfer, and $k$ is
in Å$^{-1}$. It then rotates the resulting
$\mathbf Q=\mathbf k_i-\mathbf k_f$ by the recorded sample A3 angle, and solves
for HKL using the lattice and orientation vectors stored in the file. The A3
offset is an explicit import setting because a correction used during analysis
may not be stored in the acquisition file. For the supplied LiV2O4 example,
enter `66.5` degrees to reproduce the shown MSlice orientation.

Counts and Poisson one-sigma uncertainties are multiplied by the stored
per-stream detector-efficiency corrections and normalized to the chosen monitor
target; the default is $10^6$ monitor counts. A zero-count point receives the
uncertainty corresponding to one count so it cannot acquire infinite fit
weight. Each point also retains the effective normalization denominator
$D=M/(T c)$, where $M$ is its live monitor count, $T$ is the selected monitor
target, and $c$ is the multiplicative detector-efficiency correction. The
central nfit rebinner always includes $D$ in the data weight. Uniform averaging
therefore uses $D$, while inverse-variance averaging uses $D/\sigma^2$.

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
mixing their different energy semantics. The import dialog can either add a
later batch to compatible existing SPEC and DIFF collections (the default) or
create separate collections. Source-backed MACS entries retain a lightweight
detector-point count, so collection summaries and rebin work estimates remain
correct while the numerical arrays are unloaded. Configure the HKLE output grid
on each collection and choose **Rebin now**. The imported lattice and UB matrix
are promoted to each collection, and its rebinned composite is a gridded
dataset that opens directly in the data viewer. The same operation is
scriptable:

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
    stream_group_mode="reuse",
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

**Crystal orientation** displays the active UB matrix directly. **UB setup**
edits lattice parameters, orientation vectors `u` and `v`, and the full
$3\times3$ UB matrix. **Calculate from lattice
and u/v** places `u` along the incident beam and uses `u` and `v` to define the
horizontal scattering plane.

UB maps `[h,k,l]` to reciprocal momentum in inverse angstrom, with
$\mathbf Q'=UB(H,K,L)^T$ and $|\mathbf Q'|=1/d$ for a reflection of
plane spacing $d$ in Å. Physical scattering wavevector is
$\mathbf Q=2\pi\mathbf Q'$; $U$ is the orientation rotation and $B$ here
is the crystallographic reciprocal basis without $2\pi$. **UB from NeXus** reads embedded orientation metadata.
**UB from ISAW** and **Save ISAW** use the conventional transposed three-row
ISAW representation.

Apply the dialog to a dataset for a dataset-specific orientation or to a group
for shared orientation and composite HKL conversion.

## Dataset details and physical conventions

Selecting a dataset organizes its controls into **Overview**, **Physics**,
**Binning & channels**, and **Metadata** tabs instead of one long settings
page. The overview contains identity, conditions, data, and source information;
the physics tab contains crystal orientation and signal conventions; and the
binning tab contains axes, rebinning, and point-list channels. The **Fit bins**
count uses the same prepared view as the optimizer, including masks, invalid
values, and invalid uncertainties.

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

## Metadata dimensions

Select the dataset collection containing a series (for example, **MACS SPEC**)
and use **Metadata dimensions → Add dimension…** to select the channel.
Added axes appear after DeltaE in the composite rebinner. Enable **Combine
enabled datasets into one effective dataset** to view the result.

Choose a numeric metadata channel, an axis name, and its units. The channel
selector lists paths from the first enabled source; the selected channel must
exist in every enabled source. Paths such as `entry/data/temp/average_value`
or `entry>data>temp>value` read the original NeXus file. `parameters/temperature`
reads the dataset's configured temperature in kelvin (K), while `temperature`
reads the loaded point temperatures.

**One value per dataset: mean** or **median** assigns the selected channel's
summary to every point in that dataset. **One value per measured point** accepts
scalar values, arrays aligned to the loaded points or histogram cells, and
MACS scan columns under `entry/data` repeated over their detector channels.
Timestamped `value` logs are not scan columns. Asynchronous logs require an
explicit time-alignment adapter; nfit does not interpolate or guess alignment.

Enter **Discrete coordinates**, such as `5, 10, 20, 30, 40, 50`, to use nominal
temperatures. A reading of 10.24 K maps wholly to 10 K when the **Assignment
tolerance** is at least 0.24 K. This tolerance is a maximum distance in the
coordinate's units, not a bin width. Outliers, equal-distance ties, missing
channels, nonfinite values, and incompatible units produce errors. Units label
the values and do not convert them. Leave the coordinates blank to retain the
exact sorted unique values. **Preview assignments** shows every source's
readings and target coordinates before applying.

Metadata-axis rows appear with the physical axes under **Rebin settings** and
use the same two selectors. **Discrete** retains the assigned values; Step,
Bins, and Edges construct ordinary grids. **Tolerance** derives a compact set
of bin centers from nearby assigned values.
This rebin tolerance is distinct from the metadata recipe's assignment
tolerance: it controls clustering after the source channel has been assigned.

Sources assigned to the same coordinate share the existing composite's weights,
physical normalization, masks, scales, symmetry, and spatial grid. Physical and
metadata coordinates are passed together to one N-dimensional central rebin;
metadata values are not first reduced into independent condition slices.
Multiple metadata dimensions produce a grid of combinations; combinations with
no measurements remain masked. Temperature coordinates in K are also supplied
to fitting and paired spectral-channel conversion.

Each added rebin row defaults to **Discrete**, retaining the assigned coordinates.
Select **Step**, **Bins**, or **Edges** to combine them into broader bins. Step and
bin-count limits are first/last bin centers, as for momentum and energy; a
one-bin count uses the limits as integration edges. Explicit edges can be
nonuniform. Interior edges belong to the bin on their right, and the final edge
is inclusive. Coordinates outside the edges are excluded; empty bins stay masked.
Nominal-coordinate assignment, when configured, happens before this binning.
Metadata assignment defaults to **Discrete**. For Step, Bins, and Edges grids,
it can instead be changed to **Fractional**, distributing a measurement between
adjacent metadata bins with the same weighting and uncertainty propagation used
for physical axes. Discrete and Tolerance grids require discrete assignment.
Weights and uncertainties are combined directly from the original samples,
rather than by averaging existing condition slices. Return to
**Discrete** to recover the original coordinates. **Copy settings** and
**Paste settings** include metadata recipes when used between composite panels.

Choose the new dimension as a plot axis, hidden-axis selection or integration
range, waterfall axis, or tiled-slice axis in the data viewer. Automatic tiling
shows one panel per metadata coordinate. The stored coordinates remain exact even when their
spacing is uneven. Boundaries around these coordinates are display cells;
they do not imply interpolation between conditions. Materialized outputs retain
the axes, but further rebinning must be performed on the original collection.
Configure these dimensions on the collection holding neutron point datasets or
histograms; stacking already stacked or hierarchical composites and raw event
time-log alignment are not currently supported.

Save the project and use **Copy composite script** to export an editable recipe.
The [workflow API](workflow_scripts.md#composite-workflows) uses the same
implementation without Qt widgets.

## Rebinning and composites

Rebinning affects both viewing and fitting. It supports:

- multiple named rebin configurations for each dataset or dataset collection;
- per-axis **Discrete**, **Step**, **Bins**, **Edges**, or **Tolerance** modes;
- inverse-variance or uniform averaging;
- independently selectable fractional or discrete assignment on Step, Bins, and Edges grids;
- exact or tolerance-clustered whole-bin assignment;
- a minimum effective source-sample threshold;
- projected HKLE coordinate bases;
- point-group symmetry expansion; and
- bounded batch sizes for temporary working memory.

The **Binning** selector above the settings chooses which named configuration
is being edited. **Add…** creates a visualization configuration from the fit
configuration, while **Duplicate** copies the selected configuration. Rename or
remove visualization configurations as needed, and select **Use for fitting**
to designate a different configuration as the sole fit rebin. The remaining
enabled configurations have zero fit weight: they are not prepared during
optimizer or sampler iterations, but nfit evaluates the fitted model on them
once afterward for plotting. A dataset and a collection each have their own
independent list of named configurations.
When combining histograms, nfit reconstructs physical coordinates from each
source's saved axis vectors before projecting into the output basis. This
preserves peak positions and coverage when combining already rebinned HHL
histograms or histograms with custom axis labels.
Viewer aliases for named binnings preserve live analysis ownership, so
source-linked clone and histogram-arithmetic datasets remain evaluable after a
project is reopened.

Every physical axis has two independent selectors. **Grid** constructs the bin
coordinates with Discrete, Step, Bins, Edges, or Tolerance. **Mode** controls
whether a point contributes fractionally to neighboring bins or wholly to one
bin. Fractional is the default for physical Step, Bins, and Edges grids;
Discrete and Tolerance grids require discrete assignment and disable the second
selector. **Discrete** derives exact coordinate centers. **Tolerance** clusters
nearby recorded coordinates and derives one center per cluster. For nominal
MACS energy scans, setting DeltaE to Tolerance with `0.1` meV separates scans
near -0.4, 0, 0.4, 1.2, 2.0, 2.8, and 3.6 meV without creating unmeasured
intervening slices. Metadata axes default to discrete assignment but offer the
same fractional choice for Step, Bins, and Edges grids. Empty bins remain masked.

For four-dimensional single-crystal data, **Momentum coordinates** exposes the
complete $3\times3$ momentum block. Its rows define the three output directions
in physical H, K, and L coordinates; energy transfer remains a separate fixed
coordinate and cannot be mixed into those rows. The rows must form an
invertible basis. For example, `[[1,1,0], [0,0,1], [1,-1,0]]` produces axes
labelled `[H,H,0]`, `[0,0,L]`, and `[K,-K,0]`.

**Min center** and **Max center** specify the first and last centers of a uniform
grid, in the axis units. For example, energy centers `-0.2` to `1.4` with step
`0.2` produce nine bins with outer edges `-0.3` and `1.5` meV. Step mode keeps
the exact spacing and stops at the last center at or below the maximum; Bins
mode places the requested number of centers including both endpoints. A
one-bin integration in Bins mode instead uses the endpoints as interval edges.
Explicit nonuniform edge lists and viewer integration ranges also remain edges.
Saved uniform-grid limits use this same center convention when reopened.
If an older project contains an obsolete automatic-limit marker beside an
explicit numeric bound, the numeric bound is shown and used; clearing that
field explicitly restores automatic endpoint selection.

Limits are blank by default. A blank endpoint is obtained from
the minimum or maximum projected data coordinate. Uniform edges are placed at
half-step offsets so the corresponding bin centers lie on integer multiples of
the step, including zero on the extended grid. Entering a number makes only
that endpoint explicit. When symmetry is enabled, automatic endpoints include
every generated symmetry image; explicit endpoints remain unchanged. Turning
**Apply symmetry** off and back on preserves both the expression and its
notation mode. Point-cloud preview resolution adapts to the complete composite
range, so combining nominally discrete energy scans does not turn small
within-run energy jitter into an impractically large output volume.

An imported MDEvent collection can reduce directly either to a projected
single-crystal HKLE histogram or to a powder $|\mathbf Q|,E$ histogram. Both
paths accumulate event weights and use proton charge plus detector-trajectory
coverage for normalization. The powder path bins radially without first
allocating a sparse four-dimensional volume, which is the appropriate route
for a fixed-angle environment/background run.

Masks are applied before binning. The rebin panel separates **Rebin settings**,
including physical and metadata-axis rows, **Metadata dimensions** for defining
metadata sources and nominal assignments, and expandable **Bin information**.
The information tab reports each axis's count, limits, centers, edges, grid
mode, assignment mode, total bins, and numeric payload estimate; after a current
rebin it uses the exact cached grid.

Automatic rebinning turns off when an edit raises the estimate above 5,000,000
point contributions or 2,000,000 output bins. It may be manually re-enabled;
otherwise the job remains pending until **Rebin now** or an operation requires
current rebinned data. **Workers** sets the saved worker ceiling for this
configuration; the rebinner can use fewer workers. **Benchmark this rebin…**
compares batch targets and worker ceilings using isolated runs of the full
current configuration, without updating live data. Review the timing and process
peak-memory table before choosing **Apply recommendation**. The same controls
are available for composites. Defaults for new configurations live in
**File → Preferences → Performance**; see [benchmark details](performance.md#performance-preferences-and-benchmarks).

All rebin workflows use the same compact progress dialog. When a task prepares
several datasets or dataset-group composites, it uses two levels: the upper bar
reports completed viewer entries and the lower bar reports progress within the
named current entry. A one-entry task hides the redundant upper status and bar.
Both levels show elapsed time and, once their progress is determinate, estimated
time remaining. Large integer counters use grouped thousands, and resource
details below the lower status report output bins, CPUs, and estimated working
memory. Symmetry-equivalent duplicates still count as examined work, so detailed
progress reaches completion even when those duplicates are omitted from the
histogram.

Native MDEvent reductions report event accumulation, detector-trajectory setup,
detector-normalization integration, output finalization, derived-data
evaluation, and viewer preparation as separate stages. Event counters include
symmetry-expanded contributions rather than only source rows. Consequently a
completed event counter does not conceal a subsequent normalization pass, and
the status continues to advance while a large viewer is being constructed.

The batch target controls temporary work, not the
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

**Copy settings** and **Paste settings** transfer the selected compatible rebin
recipe.
**Create dataset from rebin** materializes an independent project dataset.
For a dataset collection, **Create dataset from composite** stores the current
composite inside the `.nfit` archive. Project-owned materializations load lazily
when the project is reopened and can serve as a downstream composite input or
background without keeping the source event files in memory.

A composite combines compatible enabled descendants into one effective
dataset. Each source signal and uncertainty receive its positive dataset
calibration scale, then the sources are averaged using their fit weights and any
physical normalization denominator carried by the data. **Uniform** omits an
additional uncertainty factor; **Inverse variance** additionally applies
$1/\sigma^2$. Gridded MDEvent outputs carry their detector-trajectory
normalization denominator, and MACS points carry their monitor- and
detector-efficiency-based denominator. Use a dataset or group background for
subtraction. When a composite is active, its constituents are not fitted
separately.

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

where $a$ is the dimensionless fixed background scale and each $\sigma$
is a one-sigma uncertainty in the signal unit. This assumes independent
sample/background values and does not propagate uncertainty in $a$. Values outside the background domain are
masked rather than extrapolated.

For a group background referencing raw neutron point data, nfit bins the
reference onto the sample's resolved momentum/energy grid automatically,
including its coordinate basis, symmetry, averaging mode, and bin edges.
The source's own viewer-rebin settings are preserved but do not override the
group background's required alignment. Temperature-series composites
subtract this reference from each metadata slice and retain the enabled reference
run at its own temperature. An isolated reference subtracted from itself at
scale 1 remains visible as zero signal and zero uncertainty: the same observations
cancel exactly. Zero uncertainty excludes that identity check from weighted fitting.
Without metadata dimensions, references are excluded from the pooled sample mean.
Disabling the background restores the measured reference signal and immediately
refreshes open viewers, including in manual rebin mode.
After changing the sample grid, the reference follows
it on the next rebin. Check the energy limits when copying settings between series.

Use **Reload data** on a dataset to reread its configured source without changing
its masks, backgrounds, rebin recipe, or other project settings. On a dataset
collection, the same action reloads every descendant that has a source. Derived
composites and open data viewers refresh after the reload. The scripting
equivalents are ``reload_dataset_data(dataset)`` and ``reload_data_group(group)``.

Select a **Masks** or **Backgrounds** folder to enable or disable every item it
contains with one checkbox. A Backgrounds folder also provides a shared **Scale**
editor; a blank value means its background scales differ. The corresponding
scripting calls are ``set_mask_collection_enabled(owner, enabled)`` and
``set_background_collection(owner, enabled=..., scale=...)``.

For an already gridded or explicitly rebinned single-crystal background, reduce
sample and background independently onto identical axes and bins.
nfit then subtracts corresponding bins without
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
