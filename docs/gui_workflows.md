# GUI workflows

The graphical interface is a project explorer plus a data viewer. It is meant
to make common exploratory work easier without replacing scripts: every GUI
operation should correspond to a readable project file or a Python operation
that can be repeated later.

## Data Playground analyses

The project tree includes an `Analyses` branch after `Fits`. Use **Open in Data
Playground** on a workspace, dataset, or analysis node for non-fitting Bragg and
spectral operations. Recipes are non-destructive, run through the background
task framework, and create linked `Derived data` entries. Tree labels
distinguish never-run, fresh, stale, failed, and unavailable results.

See [Data Playground](data_playground.md) for normalization, coverage, output,
and script conventions.

## Launch

After installing the package in the `nfit` environment, launch the project
explorer with:

```bash
nfit
```

During local development, the explicit environment interpreter is:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m nfit.project_gui
```

When launched from a terminal, Ctrl+C sends an interrupt that closes the Qt
event loop cleanly with the standard interrupt exit status.

## Project explorer

### Dataset importing

At the top level of a data group, **Dataset importing** provides a persistent
file-based import workflow. Enable **Import datasets from files** to use its
controls. **Add files** opens a standard multi-file browser and adds every
selected file as a dataset entry in that data group.

For numbered acquisitions, fill in **Path**, **Prefix**, **Suffix**, and
**Numors**, then press **Import datasets**. Numors accepts inclusive ranges and
comma-separated pieces:

```text
409981:409995
409981:409992,409994:409995
409981:3:409995
```

The first imports every number in the interval, the second skips 409993, and
the third imports every third number. nfit verifies that every constructed file
exists before importing any of them. **Clear datasets** removes all direct and
nested datasets from the data group after confirmation, while retaining that
group's models, masks, and fit history.

Raw direct-geometry time-of-flight NeXus files such as `SEQ_409981.nxs.h5`
are imported together as a file-backed raw-run dataset group. Enable its
composite and choose the four HKLE coordinate axes, limits, and resolution to
stream the event banks directly into a plotted and fitted HKLE histogram. Raw
runs are intentionally not listed in the data viewer by themselves: detector
events are not yet a meaningful plotted dataset until this reduction completes.

### Raw direct-geometry TOF data

The **Raw TOF shared setup** panel stores one UB matrix, optional vanadium
normalization file, optional detector mask, and Ei/T0 overrides for all selected
runs. nfit reads the source-to-sample distance and detector pixel positions from
each file's NeXus instrument definition, calculates final energy from the TOF
remaining after the incident flight path, forms `Q = k_i - k_f`, rotates into
the sample frame, and converts to HKL using `(2*pi*UB)^-1`. Raw IDs are
processed bank by bank and in bounded event chunks, so the source event table is
never copied into memory.

Each run is normalized by its retained proton-pulse charge in microampere-hours.
Before this kinematic factor, nfit applies Mantid's wavelength-dependent He-3
tube-efficiency correction whenever the embedded instrument definition supplies
the tube pressure, temperature, wall thickness, diameter, and orientation. It
multiplies both the event and its uncertainty by the inverse detector
efficiency; unsupported detector definitions are left unchanged. The default
**Apply ki/kf correction** then multiplies an accepted event by the
incident-to-final wavevector ratio; its variance receives the squared total
factor, matching Mantid direct-geometry reduction. Both a processed vanadium
file and a mask file act as detector masks in this Shiver-compatible workflow: zero,
negative, or invalid values exclude the detector from events and trajectory
coverage rather than rescaling its signal.

When available, nfit estimates Ei and T0 separately for each run from monitor
locations in its embedded instrument definition, using one-microsecond bins and
Mantid GetEi v2's peak-width, rebinning, background, and first-moment analysis.
The calculation is not tied to a particular monitor name: it follows IDF
monitor order, including IDs whose locations are defined from a run log. For
instruments such as CNCS and HYSPEC that define Mantid's `t0_formula`, nfit
uses the requested Ei and evaluates that formula instead of fitting two monitor
peaks. The **T0 override** is in microseconds and is subtracted from each
raw event TOF. Bins with trajectory coverage but no accepted events retain a
zero signal and receive nfit's 68% Feldman-Cousins upper-limit uncertainty,
scaled by that bin's normalization. Progress reports the number and percentage
of raw events reduced.

#### Raw TOF reduction sequence

For a raw direct-geometry group, nfit performs the following operations in this
order for every selected run:

1. It reads the run logs, embedded instrument definition (IDF), detector IDs and
   pixel positions, source-to-sample distance, and sample orientation. Event
   banks remain on disk and are read in bounded chunks.
2. It determines `Ei` and `T0`. An explicit group override wins; otherwise nfit
   uses Mantid GetEi v2 from the IDF monitor layout, or the IDF's Mantid
   `t0_formula` for formula-driven instruments such as CNCS and HYSPEC.
3. It combines the processed-vanadium and explicit-mask files as binary detector
   exclusions. A detector with a zero, negative, invalid, or explicitly masked
   value is excluded from both the event numerator and normalization coverage.
4. It applies the bad-pulse rule to raw events. With the default 95% threshold,
   a pulse is retained when its proton charge is at least 95% of the run's mean
   pulse charge. The denominator uses the sum of those same retained charges,
   converted from pC to microampere-hours.
5. For each retained detector event, it subtracts `T0`, subtracts the incident
   flight time `2286.4 * L1 / sqrt(Ei)` microseconds, and obtains
   `Ef = (2286.4 * L2 / t_f)^2`. Events with nonpositive final time or outside
   the default energy range `-0.95 Ei <= DeltaE = Ei - Ef <= 0.95 Ei` are
   discarded.
6. It converts accepted events to `Q = k_i - k_f` in the laboratory frame,
   rotates by the run goniometer, converts to HKL with `(2*pi*UB)^-1`, and then
   projects HKLE into the four configured rebin coordinate axes.
7. When an IDF defines a cylindrical He-3 detector with tube pressure,
   temperature, wall thickness, and radius, nfit multiplies the event by
   `1 / (1 - exp(-alpha * lambda_f))`, where `lambda_f = 2*pi/k_f` and `alpha`
   is Mantid's path-length-dependent tube coefficient. It then applies the
   optional `ki/kf` factor. The event variance receives the square of the full
   product of these corrections. Detectors without complete He-3 IDF metadata
   retain unit efficiency.
8. It sums corrected event weights and their squared weights in each output
   bin, while recording the unweighted number of accepted events separately.
9. It constructs the MDNorm-style denominator independently by tracing every
   unmasked detector's allowed `-0.95 Ei` to `+0.95 Ei` trajectory through HKLE
   bins and accumulating its retained proton charge times the energy interval.
10. Finally, it divides the event sum and square-root variance by that
    denominator. Bins without trajectory coverage are masked; covered bins with
    zero accepted events retain signal zero and receive the documented 68%
    Feldman-Cousins upper-limit uncertainty.

This mirrors the relevant Shiver/Mantid direct-geometry sequence without
requiring Mantid at runtime. The result metadata records whether the He-3 and
`ki/kf` corrections were enabled.

### UB setup for single crystals

Every single-crystal dataset and dataset group exposes **Crystal orientation >
UB setup**. The dialog edits unit-cell lengths and angles, two reciprocal-lattice
orientation vectors `u` and `v`, and the complete 3 x 3 UB matrix. **Calculate
from lattice and u/v** places `u` along the incident beam (`+x`), uses `u` and
`v` to define the horizontal scattering plane, and takes `+z` as vertically
upward. UB maps the column vector `[h,k,l]` to `Q'` in inverse angstrom, where
`|Q'| = 1/d`.

**UB from NeXus** reads orientation and lattice metadata from either MDEvent or
processed MDHisto NeXus files. **UB from ISAW** reads the usual `.mat` format;
the first three rows in that format are the transpose of UB. **Save ISAW**
writes the same IPNS convention, followed by lattice parameters, cell volume,
and the conventional zero row. Hover text on each control summarizes its units
and coordinate convention.

Applying the dialog to an individual dataset stores a dataset-specific
orientation. Applying it to a dataset group stores shared orientation metadata;
for an MDEvent or raw-TOF group it also updates the shared UB used for HKL
conversion and marks the composite rebin stale. Applying it at the top data-group
level updates the shared sample lattice and orientation.

### MDEvent single-crystal data

Importing an MDEvent NeXus file as **Single-crystal inelastic** creates a
dataset group containing one lightweight entry per experiment/run. The event
table stays file-backed: nfit stores the source path and run index instead of
copying shared instrument, orientation, and event data into every entry.

The **MDEvent shared setup** panel controls the UB matrix, detector mask,
vanadium normalization, incident-energy override, and time-zero override for
the whole group. If the source directory contains exactly one file beginning
with `van`, nfit initially selects it for both masking and normalization. Zero,
negative, and invalid detector values are omitted from both event data and
normalization coverage.

Enable **Combine datasets** on the imported group to produce the normalized
HKLE dataset. Large groups default to manual rebinning: choose the HKLE limits
and resolution and press **Rebin now**. The initial grid is deliberately modest
(20 bins on H, K, and L and 50 on DeltaE), since all 4D working arrays scale
with the product of the four bin counts.

The composite controls appear above the run table. Each **Coord axis** row is a
four-component HKLE direction and the four rows must form a linearly independent
basis. Momentum rows may combine H, K, and L, for example `[1, 1, 0, 0]`,
`[0, 0, 1, 0]`, and `[1, -1, 0, 0]`; the energy row remains
`[0, 0, 0, 1]`. Bounds and resolution refer to coordinates in this basis.

Individual file-backed MDEvent run entries are not offered to the data viewer:
they are unnormalized event partitions rather than meaningful plotted datasets.
Enable compositing and rebin the group to view its normalized HKLE result.
Import shows progress while run metadata is read. **Rebin now** opens progress
immediately and reports preparation, event scanning, and detector normalization;
determinate progress labels show both the completed fraction and percentage.

nfit reads the NeXus/HDF5 structures directly and does not import or invoke
Mantid. An Ei override changes normalization trajectories. A T0 override is
retained for future raw-event conversion, but cannot move coordinates already
stored in an MDEvent workspace.

#### Measured-zero uncertainties

An MDEvent table contains one row for every detected event. It does **not**
contain a row for every output histogram bin. In each event row, column 0 is
the event's signal contribution and column 1 is its variance contribution,
named `errorSquared` in the file. Both values are supplied by the program that
created the MDEvent file; they are not generated from an nfit fit weight.

nfit keeps two questions separate when it creates a histogram:

1. Did any detector trajectory measure this bin?
2. If the bin was measured, how many events landed in it?

This distinction produces three cases:

- **One or more events:** nfit sums the events' signal contributions and
  variance contributions. With normalization denominator `D`, the result is
  `signal = sum(signal_i) / D` and
  `sigma = sqrt(sum(errorSquared_i)) / D`.
- **Detector coverage but no events:** this is a **measured zero**. No event row
  exists for the bin, so there is no `signal_i` or `errorSquared_i` to read from
  it. nfit stores signal `0`, preserves `num_events = 0`, and assigns the finite
  uncertainty described below.
- **No detector coverage:** this is missing data, not a measured zero. The bin
  remains NaN/masked and is excluded from viewing and fitting as data.

Assigning uncertainty zero to a measured zero would make it appear to be known
exactly and would give it infinite statistical influence in a fit. Put simply,
nfit says: **take the uncertainty this bin would have if it contained one
representative event, multiply it by `1.29`, and use that as the uncertainty
for zero observed events.** The factor is the upper endpoint of the
68.27% Feldman--Cousins confidence interval `[0, 1.29]` for observing zero
events with zero known background (Table II of
[Feldman and Cousins](https://arxiv.org/pdf/physics/9711021)). In detail:

```text
representative_event_scale = sqrt(
    sum(errorSquared_i for all accepted events) / number_of_accepted_events
)

sigma_zero = 1.29 * representative_event_scale / D
```

The representative one-event scale comes from the nonzero accepted events
elsewhere in the requested MDE volume, because the empty bin itself has no
event rows. The bin's own normalization `D` is then applied, so this is the
uncertainty nfit would assign to one representative event in that particular
bin. When the file uses the common convention `errorSquared_i = signal_i^2`,
the representative scale is the root-mean-square of those events' signal
contributions. For ordinary unweighted events, every `signal_i` and
`errorSquared_i` is `1`, so the rule reduces to `sigma_zero = 1.29 / D`.

For example, if two events in a bin have `(signal, errorSquared)` values
`(1.2, 1.44)` and `(0.8, 0.64)`, and `D = 1000`, that bin has signal
`(1.2 + 0.8) / 1000 = 0.002` and uncertainty
`sqrt(1.44 + 0.64) / 1000 = 0.001442`. A covered bin with no events has no such
rows. If the representative event scale estimated from the accepted dataset is
`1.778`, its stored result is signal `0` and uncertainty
`1.29 * 1.778 / 1000 = 0.00229`.

This zero-count value is an effective, symmetric fitting uncertainty. It does
not claim that the underlying Poisson interval is symmetric. It is also
unrelated to the dataset **Fit weight**, dataset scale factor, or detector
normalization value.

#### Project organization

The left tree organizes a project into top-level workspaces. A workspace holds
datasets, models, and fit history. Datasets may also contain nested dataset
groups and masks, so large experiments can be kept close to the physical or
sample-series structure the user thinks in.

Common actions are available from the bottom-left buttons, the right-side action
buttons, keyboard shortcuts, drag and drop, and right-click context menus:

- create, rename, reorder, and delete workspaces, datasets, dataset groups,
  masks, and models where the item type allows it,
- import datasets from files or drag them into a workspace or dataset group;
  dropping a dataset into an empty project creates its first workspace. Drop a
  single `.nfit` project file anywhere in the tree to open it after the usual
  unsaved-changes prompt,
- copy and paste datasets between workspaces and masks between datasets,
- save datasets or whole projects,
- open or refresh the data viewer for a selected workspace or dataset,
- add masks and model components,
- run fits and keep the resulting history in the `Fits` tree.

The tree supports multi-selection so several items of the same kind can be
deleted at once. Shift-click extends a contiguous range from the previously
clicked item and Ctrl/Command-click toggles individual items; the selection is
automatically kept to a single kind, so a range of fit results (or datasets,
masks, or models) never sweeps in the enclosing workspace or the
`Datasets`/`Models`/`Fits` folder headers. Pressing `Delete` or the `Delete`
button then removes every selected item in one action. Extending a fit selection
with a modifier held does not restore each fit's state as you go, so building a
selection to prune old history stays fast.

The `File` menu supports New, Open, Recent projects, Save, Save As, Close, and
Quit. On platforms with standard shortcuts, these use the expected New/Open/Save
bindings, with Command as the default modifier on macOS and Control elsewhere.
Close and Quit prompt before discarding unsaved project changes.

## Dataset details

Selecting a dataset shows structured details in the right panel. The sections
include the dataset type, axes, crystal information, data summary, source file,
and imported metadata. Metadata is shown as an expandable tree so nested fields,
instrument logs, sample-environment information, timestamps, temperature, field,
and similar provenance remain inspectable without flattening everything into a
long text block. For binned MDHisto data, the data summary reports the total
bin count and a `Fit bins` count computed from the same prepared view used by
the optimizer, so file masks, inherited group masks, dataset masks, empty bins,
and invalid uncertainties are reflected in the number of bins that actually
participate in a fit.

Point-list datasets such as magnetization or powder elastic data expose editable
coordinate/channel configuration. MDHisto datasets expose rebin settings in the
Axes panel when rebinning is supported. Rebinning can be enabled for viewing and
fitting. The current rebin can be materialized as a new independent project
dataset with `Create dataset from rebin`, or written directly to disk with
`Save rebin to disk`.
`Save dataset` and `Save rebin to disk` write portable nfit `.npz` archives.
They can be added back through the normal dataset import flow; the archive
restores its signal, uncertainties, masks, axes, metadata, and saved dataset
temperature/field context. Archives saved by older nfit versions may not carry
temperature or field because those values were not written at the time.
All enabled file, inherited group, and dataset masks are applied before point or
MDHisto data are rebinned, so excluded data do not contribute to rebinned bin
averages; disabled masks are ignored. The materialized dataset keeps the source
dataset's temperature override. The Rebin panel's `Mean` selector controls how
multiple source points are averaged inside each output bin: `Inverse variance`
uses `1/sigma^2` weights and is the default, while `Uniform` keeps a simple
mean. Fractional binning still applies the geometric fractional contribution on
top of the selected mean weighting and is enabled by default. The Rebin panel's
`Batch target` control sets the approximate per-batch working-memory target in
MB; it defaults to 192 MB. Smaller batches usually use less temporary memory but
require more computational time. The target is not a cap on total rebinner
memory use, because source arrays, coordinates, output grids, and bookkeeping
also consume memory. The best value depends on dataset size, output grid size,
dimensionality, and available memory.
The `Automatic rebinning` checkbox controls whether edits to rebin bounds,
vectors, bin counts, weighting, or batching immediately recompute the cached
rebinned view. nfit estimates the work from source points, output-bin count,
and fractional-neighbor contributions; small datasets default to automatic
rebinning, while large datasets default to manual rebinning. When automatic
rebinning is off, edits are marked pending and the currently cached rebin stays
visible until `Rebin now` is pressed. Operations that require current rebinned
data, including fitting, opening the data viewer, materializing a rebinned
dataset, and saving a rebinned dataset, force the pending rebin first. Large
explicit rebin jobs show a progress dialog driven by the rebinner batches.
The Rebin panel can also apply symmetry before binning. A space-group entry
uses its point-group rotations only: screw/glide translations are deliberately
discarded and nfit does not add inversion unless it belongs to the selected
point group. Symmetry always acts on physical HKL coordinates before any custom
output-axis projection; energy transfer is unchanged. Choose the input mode
and enter one of the following forms:

- `P -1` as a space group, or `-1` as a point group.
- `x,y,z;-x,-y,-z` as an exact semicolon-separated Jones-faithful operation list.
- `rotate(order=3, axis=[1,1,1])` for a geometric generator about a direct-lattice direction.
- `mirror(plane=(0,0,1))` for a geometric generator across a reciprocal-lattice plane.

Multiple geometric generators are separated with semicolons and nfit closes the
generated group. Geometric generators require complete lattice parameters.
The resolved operation count is shown beside the editor and contributes to the
rebin work estimate. Symmetry-expanded rebins stream one transformed batch at a
time rather than materializing every image in memory.
The rebin table has one `Resolution` column with a `Step`/`Bins` selector;
`Step` is the default. Switching the selector derives the displayed quantity
from the current bounds and resolution. In `Step` mode, changing bounds keeps
the requested step size exactly; if the range is not an exact multiple of that
step, the final bin is shorter. In `Bins` mode, changing bounds keeps the bin
count and updates the displayed step.
For 4D MDHisto data, the `Coord axis` defaults follow the displayed physical
axis when possible rather than a blind diagonal matrix: for example `DeltaE`
starts as `[0, 0, 0, 1]`, `[H,-H,0]` starts as `[1, -1, 0, 0]`, `[0,0,L]`
starts as `[0, 0, 1, 0]`, and `[H,H,0]` starts as `[1, 1, 0, 0]`. The `Axis`
column shows only the scalar variable (`H`, `K`, `L`, or `E`). The plotted axis
name is generated from that variable and its coordinate vector, so an `H` row
with `[1, 1, 1, 0]` becomes `[H,H,H]`.

Together, the four coordinate vectors define a complete HKLE basis; they are
not four independent, unnormalized dot products. nfit solves each source point
in that basis, which prevents a direction such as `[1, 1, 1, 0]` from changing
scale merely because its vector has length greater than one. Whenever a vector
is edited, all four output bounds are recalculated to contain the full source
volume. In `Step` mode, the requested step is preserved by updating bin counts;
in `Bins` mode, the bin count is preserved and the step is recalculated.
Once configured, coordinate axes persist across project reloads, fit-history
selection, and model edits. nfit generates default axes only for a dataset that
has no saved rebin-axis configuration; changing coordinate vectors remains an
explicit rebin-panel action.
The vectors must remain linearly independent. Momentum rows may contain only
H, K, and L components, while the energy row may contain only a nonzero E
component; nfit rejects coordinate systems that mix energy and momentum.

Select the top-level `Datasets` node or any nested dataset group to see its
descendant dataset count, total loaded data points, dataset types, fit weights,
scale factors, and `Composite dataset` panel. Composite controls are not shown
on the workspace node, which remains the owner of models and fit history. When
all enabled datasets in the selected collection hold the same kind of data,
the panel can combine them into one effective rebinned dataset. Composite mode uses
the same rebin controls as an individual dataset, but applies them after
collecting valid points from every enabled constituent dataset. For each source
dataset, nfit first multiplies the signal by the dataset scale factor and the
uncertainty by the absolute value of that scale factor; use a negative scale
factor to subtract a dataset from the composite. The inverse-variance weight is
then multiplied by the dataset fit weight, so larger fit weights make that
dataset count more strongly in the composite average.

When composite mode is enabled on the top-level `Datasets` node, the entire
dataset tree behaves like a single dataset for plotting and fitting. When it is
enabled on a nested dataset group, only that group's descendants are replaced
by its composite; datasets and groups beside it remain separate fit/viewer
inputs. The individual constituents of an active composite are not fitted
separately. To inspect constituents, open the data viewer from a dataset inside
the composite; that viewer bypasses composites and exposes the underlying
datasets.
Composite rebin controls use the same batching and automatic/manual behavior as
individual dataset rebins. Large composites default to manual rebinning: edits
are marked pending, the cached composite is reused during passive refreshes,
and `Rebin now`, fitting, or opening the data viewer forces the current
composite rebin. Large composite rebins also report progress by batch.

A **Sample environment** panel sits between the Dataset and Axes panels of the
dataset details. It holds the per-dataset temperature and applied magnetic
field:

- `T (K)` sets a per-dataset sample temperature override, stored in
  `dataset.parameters["temperature"]`. Spin down to the minimum ("(from data)")
  to defer to any temperature imported with the data. Physics models that need
  the Bose factor read the temperature from the fit points and raise a clear
  error when no valid temperature is available.
- `Field (T)` sets the applied magnetic field magnitude in tesla; `(none)` (the
  minimum) means zero field. The frame selector chooses whether the direction
  is a direct-lattice `[u v w]` vector (the usual experimental statement, e.g.
  B ∥ [111]) or a reciprocal `(H K L)` vector, and the direction box takes three
  components such as `1 1 0`. Only the orientation matters; the magnitude sets
  the strength. The field is stored as
  `dataset.parameters["magnetic_field"] = {"magnitude_T", "direction", "frame"}`
  and converted to a Cartesian tesla vector on the fit points using the data
  group's lattice (so a group lattice is required to orient it). The Zeeman term
  of spin-fluctuation models reads it and raises a clear error when it is
  needed but missing. For cubic crystals the two frames give the same
  directions; for lower symmetry they differ.
- `k_f/k_i included` is checked by default, meaning the reduced intensity
  already contains the neutron kinematic factor in the magnetic cross-section
  convention. When unchecked, nfit multiplies the signal and uncertainty by
  `k_f/k_i` before plotting and fitting, using `Ei`/`incident_energy` or
  `Ef`/`final_energy` metadata when available. For energy transfer
  `E = Ei - Ef`, nfit uses `sqrt((Ei - E) / Ei)` from Ei or
  `sqrt(Ef / (Ef + E))` from Ef. With neither energy recorded, nfit leaves the
  dataset unchanged.

Magnetic inelastic models use the absolute prefactor
`gamma_0^2 |f(Q)|^2 / pi`, with `gamma_0 = 0.073 / mu_B`, as well as the Bose
factor and polarization factor. The dataset-level kinematic option completes
this convention without embedding instrument metadata in a model component.

When the data group defines lattice parameters, they are attached to the fit
points automatically so form factors and |Q|-dependent models work without
per-dataset setup.

The dataset title row includes dataset `Fit weight` and `Scale` controls. With
`Fit scale` unchecked, `Scale` is a fixed data transform: nfit multiplies the
dataset signal by the scale factor and its uncertainty by the absolute value of
the scale factor before viewing and fitting. With `Fit scale` checked, `Scale`
is instead the initial guess for an optimizer parameter named for that dataset;
the fit compares `scale * signal` to the model and uses `abs(scale) * sigma` as
the uncertainty. After the fit, the optimized scale is written back to the
dataset and stored in the fit snapshot. Because zero scale makes the data-side
uncertainty singular, use a nonzero initial scale.

## Masks and models

Selecting a mask or model shows its type selector and parameter editor at the
top of the right panel. Parameter tooltips come from the same registry that
defines defaults and examples, so a user can inspect what each field means
without guessing from the widget label alone.

Masks are applied together. File-provided masks remain separate from nfit
masks; nfit masks are applied as their own analysis mask layer. The data
viewer exposes `combined_mask`, `file_mask`, and `nfit_mask` channels:
`combined_mask` is the effective file-or-nfit exclusion mask used for
display and fitting, while the other two channels show provenance. The data
viewer's `Apply Masks` checkbox controls whether masked regions are hidden in
the viewer, but fitting still excludes masked data. When a dataset is rebinned,
these source masks are applied before binning; the rebinned view's output mask
then marks empty or invalid rebinned bins.

All GUI mask parameters describe the region to exclude. Coordinate and
energy/`|Q|` ranges mask points inside the intersection of their active
intervals; `[0, 0]` means that dimension is inactive, and a new range mask with
only inactive dimensions masks nothing. Zero box widths, zero ellipsoid radii,
and zero phonon-cone slope are likewise neutral defaults. `Invert` deliberately
switches to masking outside the specified region. `Additive` removes the
specified region from earlier nfit-mask contributions, but cannot unmask a
file-provided mask. Disabled masks never contribute. Older saved broad no-op
sentinels such as `[-1e99, 1e99]` and `[0, 1e99]` remain recognized as inactive.

For histogram datasets, a coordinate-range mask's **Coordinate axes** default
to the dataset's physical bin-axis vectors, exactly as they do in the rebinning
controls. Projected axes are preserved rather than converted to canonical
coordinates; for example, `[H,-H,0]` defaults to `[1,-1,0,0]`. This makes each
mask interval operate directly along its corresponding displayed bin axis.

A **Phonon cone** mask can use either one Bragg center, such as `[1,1,0]`, or
a list of centers, such as `[[1,1,0], [2,2,0], [3,3,0]]`. The same slope and
radius are applied around every center, and the union of those acoustic-phonon
cones is masked. Centers are interpreted in physical reciprocal-lattice HKL
coordinates, even when a histogram is displayed or rebinned using projected
axes such as `[H,H,H]` or `[L,L,-2L]`; nfit reconstructs physical HKLE before
evaluating the cone. A legacy single-center value remains fully supported.

Mask application is automatic by default for datasets with at most 5 million
points. Larger datasets default to manual application so editing a mask does
not repeatedly scan the full dataset or block the GUI. The mask editor shows
an **Automatic mask application** checkbox, an **Apply masks now** button, and
whether mask changes are current or pending. In manual mode, passive GUI
updates reuse the last exactly matching masked result when one is available;
otherwise they use an inexpensive file-mask-only view until masks are applied.
Opening the data viewer, starting a fit, or pressing **Apply masks now** always
applies every enabled dataset mask and inherited group mask first. Turning on
automatic application opts the dataset back into recalculation as mask
settings change. This controls when masks are materialized, not which masks
participate in viewing, rebinning, or fitting.

Models can have multiple components in one workspace. Model parameters include
controls for whether they are fitted and whether they are shared globally across
datasets. Each parameter can also define an optional plot label used as the
default nickname in fit diagnostics; plain text and Matplotlib mathtext/LaTeX
style labels such as `$\\Gamma$` are accepted. More granular constraints and
linking should continue to be expressed as explicit, scriptable model
configuration rather than hidden widget state. The model editor scrolls when a
component has many sections or parameters, so fit-parameter rows keep normal
editor height as generated Heisenberg RPA exchange orbits are added.

The spin-fluctuation models (`local_relaxational`, `mmp_relaxational`,
`heisenberg_rpa`; see [Spin-fluctuation models](spin_fluctuation_models.md))
add a form-factor picker with all tabulated magnetic-ion entries plus a
`Custom...` choice; the custom coefficient field appears only when that choice
is selected. The Heisenberg RPA model additionally shows a structured crystal
editor:

- **Crystal** — lattice parameters and space group, with `Import CIF...`
  (loads lattice, space group, and atomic sites from a CIF file, and copies
  them onto the data group) and `Use group crystal` (copies the group's stored
  crystal into the model). Bare numeric or Hermann-Mauguin space groups use
  gemmi's reference setting; append an explicit setting such as `:1` or `:2`
  when a non-reference origin choice is intended.
- **Atomic Sites** — an editable table of Wyckoff sites with fractional
  coordinates, a magnetic-ion selector per site, and a `Magnetic` checkbox
  marking which sites carry spins.
- **Exchange Bonds** — a bond-length cutoff and a `Generate symmetry orbits`
  button. Generation expands the magnetic sites through the space group,
  enumerates bonds up to the cutoff, groups them into symmetry-distinct orbits
  (`J1`, `J2`, `J3a`, `J3b`, ...), and adds one exchange fit parameter per
  orbit to the Fit Parameters grid. Values of orbits whose labels persist are
  kept across regeneration; a read-only table grows with the orbit count, up
  to a capped visible height, and summarizes each orbit's distance and
  multiplicity.

All crystal and bond state is plain data in the model component's
configuration, so it serializes with the project file and can equally be set
from a script.

## Fit timeline

The `Fits` tree stores initial conditions, fit results, current states, and
nested timelines. Selecting a fit state restores the dataset/mask/model
configuration stored with that fit. Running a fit creates only the new fit
result; a sibling `Current state` is created lazily only when the user edits the
latest result at that timeline level. The active fit state is shown with a
different tree style from the ordinary cursor selection, so users can tell which
fit state is applied even after selecting a model or dataset elsewhere.

Running `Fit now` from the current state or from the end of a timeline appends a
new fit result at that level. Running from an earlier result, or explicitly
checking `Branch timeline`, creates a nested timeline so alternative fitting
attempts remain organized. When a fit succeeds, globally shared fitted
parameters are written back into the live model component and the result
snapshot, so selecting the model after the fit shows the best-fit values in the
parameter editor. Parameters fitted separately per dataset or group are stored
on the component as scoped fitted values because one editor field cannot
represent several fitted numbers.

If a fit fails, the failed result is kept for diagnostics but the mutable
scientific state remains available as `Current state`: a failed run from an
existing fit state creates or reuses the following `Current state`, while a
failed run from `Current state` keeps that state active for immediate edits.

When a fit state is selected, `Branch timeline` stays above the scrollable
details area. Optimizer/loss controls, differential-evolution settings, and
posterior settings appear as panels inside the same scrollable details area as
the fit state summary. Fit-result and initial-state parameter tables sit in an
adjustable pane so the user can drag the divider to give more room to the
parameter list or to the metadata/details below it.

Changing scientific state creates or updates the current timeline state:
datasets, masks, model components, parameter values, fitted/fixed flags,
sharing, limits, constraints, scale factors, and dataset weights are all part of
the fit snapshot. Editing the latest fit result at a level moves the active fit
state to that level's `Current state` and records the edit there; editing an
earlier result creates a nested timeline. Changing only fit-engine settings does
not create an edit branch by itself. This lets users rerun the same scientific
state with a different loss, initializer, or posterior sampler and compare the
resulting fit entries side by side.

The fit editor follows a simple opt-in pipeline:

1. `Least squares` always runs and is the default fast workflow.
2. `Differential evolution initialization` may be enabled when starting values
   are uncertain. It searches bounded parameter space before least-squares
   polishing and therefore requires finite bounds on every fitted parameter.
3. `Sample posterior with emcee` may be enabled after least squares to estimate
   posterior intervals and correlations around the fitted solution.

Initialization and posterior sampling are optional. For simple fits with good
starting values, users can leave differential evolution and emcee disabled and
just run least squares. For more complex fits, emcee can be run as part of the
fit pipeline or later from an existing fit result.

The `Loss` control selects the cost function: the rule that turns residuals
into a number the optimizer tries to make small. A residual is the difference
between data and model divided by the data uncertainty, so a residual of `1`
means the model is about one standard deviation away from the data point. The
available losses are:

| Loss | What it does | When to use it |
| --- | --- | --- |
| `linear` | Ordinary chi-squared. Every residual is squared, so points with residual `10` count one hundred times more than points with residual `1`. | Use this for clean data, well-understood error bars, and final reported fits where the usual least-squares uncertainties should be easy to interpret. |
| `soft_l1` | A smooth robust loss. Small residuals behave like ordinary chi-squared, while large residuals grow more slowly. | A good first robust choice when a few pixels, spurions, or imperfectly masked regions are pulling the fit away from the main signal. It is gentler than the more aggressive robust losses. |
| `huber` | Chi-squared near zero, then roughly linear for residuals above the loss scale. | Use when you want a simple compromise: trust most points, but stop very large residuals from dominating. It is often easy to explain and less aggressive than `cauchy` or `arctan`. |
| `cauchy` | Strongly down-weights large residuals. Very bad points add only slowly increasing cost. | Use for exploratory fits when there are clear outliers or model-mismatch regions that should have little leverage. It can hide systematic problems, so inspect residuals before trusting the result. |
| `arctan` | The most aggressive option here. Very large residuals contribute almost a capped amount. | Use only as a last-resort diagnostic when severe outliers otherwise prevent convergence. If this changes the scientific conclusion, improve masks/modeling and refit with a less aggressive loss. |

`Loss scale` sets where robust losses begin to treat a point as "large." With
normalized residuals, `1.0` is approximately one standard deviation. Increasing
it makes robust losses behave more like ordinary least squares; decreasing it
makes down-weighting begin earlier.

Longer fit runs open or reuse a progress window and run in a background Qt
worker so the GUI can keep repainting and responding while initialization,
least squares, or emcee is active. The progress window reports the active
stage, iteration or residual-evaluation count, current cost when available, and
current time per step plus current parameter values in a table, with a short
stage log below; the divider between the table and log is draggable. The
time-per-step value is reported for differential-evolution
initialization, least-squares residual evaluations, and emcee posterior
sampling. The `Cancel` button requests cancellation at the next optimizer or
sampler progress update. If cancellation happens during emcee after one or more
samples have been recorded, nfit stores the partial raw chain on the fit result
just like a completed posterior run; the posterior diagnostics can inspect it,
and `Append emcee` can continue from the last saved walker positions. Starting
another fit resets the same progress window instead of opening duplicates. When
a fit pipeline finishes, the progress window stays open so the final stage,
parameters, and log remain available until the user closes it. A compact
progress log is also stored in the fit metadata.

`DE workers` and `emcee workers` control optional parallel worker threads for
differential-evolution objective evaluations and emcee log-probability
evaluations. The default is `1`, which keeps execution serial and predictable.
Use `-1` to let nfit choose a conservative CPU-based value, currently one
less than the available processor count capped at eight workers. Values greater
than one use an internal thread pool; this is most useful when model evaluation
spends substantial time in NumPy/SciPy code.

Fit metadata is displayed in structured panels for optimizer configuration,
goodness-of-fit values, stored fit channels, arbitrary metadata, and the saved
snapshot. A `Fit results` panel presents best-fit parameter values in a compact
table with covariance-derived standard errors and, when emcee samples are
available, posterior median and 16/84 percentiles. Least-squares results also
include chi-squared, reduced chi-squared, covariance/correlation summaries, and
per-dataset contributions in the structured goodness-of-fit panel. Posterior
summaries include emcee settings, acceptance fractions, credible intervals, and
posterior correlations. Stored fit and residual channels are available in the
data viewer when their shapes still match the current dataset view. Fit results
also record a warning when a variable parameter finishes at a finite configured
lower or upper limit. The completion state of the fit-progress window names the
affected parameters, and their final values are shown in red in the progress
table, saved fit results, current-state parameter table, and model parameter
editor. A limit hit is a diagnostic that the optimum may lie outside the
allowed range, not a claim that the result is invalid.

When a fit state or result is selected, **Copy fit script** and **Save fit
script** generate a readable Python program that loads the saved project and
restores that exact state without constructing a GUI. The script defaults to
restoring the stored state and printing its fit metadata. Set its explicit
`RUN_FIT = True` option to rerun the optimizer and append a new result. These
actions require saving the project first so dataset sources and the selected
fit ID have portable references.

The fit editor and fit results share one `Posterior` panel. It configures emcee
for sampling immediately after a fit; when a fit result is selected, the same
controls also rerun emcee from the best-fit parameters, append additional steps
to a stored raw chain, or change burn-in/thinning after the fact. Result-only
checkboxes control which stored result representation is displayed and exported:
`Use emcee uncertainties, correlations, and asymmetry` replaces least-squares
standard errors with the asymmetric 16--84% emcee interval and uses the emcee
correlation matrix in `Fit diagnostics`; `Use best sample` applies the
highest-log-probability stored emcee sample to the selected result's live model
and any open data-viewer overlay. Unchecking it restores the saved
least-squares values. The least-squares snapshot and timeline remain unchanged,
and both choices are stored with the fit result so its table, diagnostics, and
exported report remain consistent after reopening a project. These
posterior-only operations update the selected fit result's posterior summaries
and diagnostics without running least squares again and without creating a new
timeline point.
Changing burn-in or thinning simply reinterprets the stored raw chain; rerun
replaces the stored posterior after an overwrite-confirmation dialog when
samples already exist; append continues from the final walker positions
and extends the stored chain. Posterior rerun and append also use the background
worker/progress window. The shared worker control defaults to `-1`, which
selects an automatic CPU-based worker count; set the value to `1` for serial
emcee evaluations.

Fit results with covariance estimates or stored emcee samples expose a `Fit
diagnostics` button. The diagnostics window includes a covariance heatmap when
a covariance matrix is available, or a correlation heatmap for older fit entries
that only stored correlations. Diagnostic plots use compact labels such as `p1`, `p2`,
... by default. A label table at the bottom of the diagnostics window shows the
full parameter names and editable plot labels; edits immediately redraw the
covariance, trace, and corner plots for that fit result. When emcee samples are
stored, the same window also includes trace and corner-style posterior tabs. The
trace tab uses the raw emcee chain when available, plotting each walker as its
own colored line against `MCMC step` and marking the selected burn-in step.
Older entries that only stored flattened samples fall back to a single `sample`
axis. Corner histograms are drawn as unfilled step histograms with best-fit and
uncertainty marker lines plus value/error titles using a proper plus-minus
symbol with stacked asymmetric bounds. The lower-triangle panels combine
density shading, iso-density contour lines, faint individual samples, and solid
best-fit reference crosshairs so overplotted posterior pile-ups remain visible.
When either posterior display checkbox is selected, those reference lines use
the corresponding selected emcee values rather than the least-squares values.

## Data viewer

## Saved plots

Every workspace has a **Plots** tree section. Use the data viewer's **Plot /
Create saved plot** action to preserve the current visual state as an editable
figure recipe. Open the tree entry for a clean presentation window, or choose
**Edit in data viewer** to restore the recipe into the full interactive controls.
The saved-plot window keeps controls hidden until **Plot / Open plot controls**
is selected; its same menu can copy/save the figure or a backend-only generating
script.

The data viewer supports dataset switching, channel selection, mask toggling,
axis selection, hidden-axis slicing/integration, color scale and limit controls,
cursor readouts, histogram box cuts, 1D line styling, model overlays, figure copy,
and script export.
Box cuts are inverse-variance weighted profiles rather than summed intensities.
For each displayed bin, the viewer combines the values across the selected box
using weights of `1 / sigma^2` and draws the propagated standard error,
`1 / sqrt(sum(1 / sigma^2))`. Masked bins and bins with non-finite or
non-positive uncertainties are excluded from both the mean and its error bar.
`Plot smoothing` provides independent Gaussian sigma controls for the displayed
X and Y directions, measured in bin widths. A value of zero disables smoothing
on that direction. Smoothing is applied after slicing/integration and only to
the plotted channels; it does not change the dataset, fitting inputs, rebinned
data, or numerical data exports. Masked bins remain masked and do not contribute
to neighboring smoothed values. Figure scripts include the smoothing settings
because they reproduce the visual plot.
When enabled model components have a complete parameter set, the viewer can
calculate and show the current model and residual channels from the `Show
model` control even before an optimization has been run. Stored fit-result
channels are still reused when no current model can be evaluated and the stored
channels remain compatible with the current dataset view.
For 2D fit comparisons, histogram box cuts show inverse-variance weighted
data+fit cuts with propagated data error bars along both plotted axes; when
residuals are enabled, residual cuts are shown below
the residual panel and at the far right.

### 3D PyVista mode

For a gridded dataset with three or more dimensions, the `Visualization`
selector enables `3D PyVista`. This mode renders the selected dataset in the
same data-viewer window and leaves the standard `2D slices` mode available for
cuts and detailed inspection. Point-list datasets must first be rebinned onto a
regular grid before volumetric rendering.

Choose three distinct dataset axes for the displayed X, Y, and Z coordinates.
Every remaining dimension has the same center, width, low/high range, and
`Integrate range` controls used by the 2D slicer. With integration off, the
nearest bin to `Value` is selected. With integration on, bins between `Range
low` and `Range high` are summed into the 3D volume. This is how, for example,
an adjustable energy interval of a 4D reciprocal-space dataset can be viewed
as a volume. Masks and empty bins are applied before a remaining dimension is
integrated, and bins with no valid contribution stay transparent.

The X/Y/Z limit controls crop the rendered volume using bin centers while
retaining complete cells and their bounding edges. `Reset axis limits` restores
the full selected axes. `Equal data units` preserves the dataset's true axis
length ratios: one coordinate unit has the same visual length on X, Y, and Z.
`Custom` enables independent scale factors for stretching or squishing the
rendered axes without changing scalar values or coordinate labels.

The 3D `Plot smoothing` controls independently set Gaussian sigma along the
displayed X, Y, and Z directions in bin widths. They affect interactive
rendering, still images, and movies only. Volume-grid and surface-model exports
are generated from the unsmoothed numerical channels, and masked cells remain
transparent. Axis scaling is likewise applied to a temporary render grid so the
volumetric data itself stretches or squishes rather than only moving its bounds.

`Volume` performs direct volume rendering. `Isosurface` extracts a surface at
the selected level of the opacity channel. The scene uses a white background
with black orientation and coordinate axes. The color and opacity channels are
independent: keep `Link opacity to color` checked to drive both from one
channel, or uncheck it to select a different opacity channel. Each channel has
its own clipped value range. The color mapping curve maps normalized color
values to positions in the selected colormap, while the opacity mapping curve
maps normalized opacity values from transparent to opaque. Drag curve points,
click to add one, and right-click an interior point to remove it. Masked or
non-finite values remain transparent regardless of the transfer curves.

`Export 3D model` saves an isosurface as VTK PolyData, PLY, STL, or a glTF
scene. A volumetric view is saved as a VTK rectilinear grid (`.vtr`) because a
ray-cast volume is scalar field data rather than a polygon mesh. `Export still
image` saves the current camera view as PNG. `Rotation axis` selects displayed
X, Y, or Z, or the direction currently vertical on screen. Z is the default.
`Animate rotation` previews a continuous orbit around the selected axis. The
movie controls set frame rate and duration; `Export rotation MP4` renders one
complete orbit around that same axis at the current camera distance, then
restores the original view. MP4 encoding is supplied by nfit's
`imageio-ffmpeg` dependency.

For powder and magnetization-style datasets, cursor readouts hide coordinate
quantities that are not meaningful for that data type. `|Q|` readout is shown
when valid `q` information is present or can be computed from `2theta` and
wavelength.

The exported script is part of the package philosophy: GUI-produced figures
should be reproducible from editable Python code. When a GUI feature changes the
plot state, the script-export path should be updated at the same time.

## Hover text standard

Every user-interactable control in the project explorer and data viewer should
have useful hover text. A good tooltip explains what the control changes, when
the change takes effect, important constraints, and an example when the input is
not obvious. Registry-backed controls, such as mask parameters, model
parameters, model configuration, optimizers, and future resolution models,
should render hover text from the same metadata that defines defaults and
scriptable validation.

Tests audit the main GUI surfaces for missing tooltips. When adding a new
control, update the tooltip and the relevant documentation or wiki page in the
same change.
