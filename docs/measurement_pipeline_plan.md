# Measurement pipeline implementation plan

This temporary plan tracks the agreed import, reduction, binning, and uncertainty
refactor. Scientific correctness takes priority over retaining an incorrect
recipe. nfit must support event spectrometers, continuous-wave neutron instruments,
diffractometers, and continuous measurements such as susceptibility versus
temperature. Instrument names alone must not determine the statistical estimator.

## Working agreement

- Implement one numbered checkpoint at a time. After each checkpoint, report
  evidence, limitations, and whether it is complete or needs smaller steps.
  **Wait for Paul to authorize the next checkpoint.** Approval of this plan does
  not authorize running all remaining checkpoints.
- Keep the table below current. Split a checkpoint before proceeding if its
  scope cannot be validated in one coherent change.
- Update focused user/developer documentation, scripting APIs, and meaningful
  tests together. Keep numerical services independent of Qt.
- Work on `main`, preserve unrelated changes, increment the package version,
  validate, commit, and push each completed change. Synchronize the existing
  ORNL installation and its documentation; retain the desktop launcher and keep
  application installations outside experiment data folders.
- Record changed scientific assumptions and invalidate affected caches explicitly.
  Preserve provenance when migrating existing projects; do not silently retain
  known incorrect math for compatibility.
- Keep the main NiO project unchanged during this refactor. Use separate test
  projects beside it when project-level validation needs persisted changes;
  a pared-down selection is appropriate for interactive checks.
- Check speed, peak memory, lazy loading, and reuse when changing numerical payloads
  or caching. Reject small speed gains that require disproportionate complexity.
- Remove this page and its navigation links after all checkpoints are accepted.
  Keep the resulting scientific conventions and workflows in permanent docs.

## Checkpoints

| Checkpoint | Scope | Status |
| --- | --- | --- |
| 1A | Reproducible uncertainty diagnostic baseline | Complete |
| 1B | Trace the unsubtracted NiO uncertainty example through histogram, slice, and cut | Complete |
| 1C | Separate accumulated event variance from low-count confidence intervals | Complete |
| 1D | Match Mantid DGS reduction and histogram statistics; expose deviations | Complete |
| 1D1 | Audit converter; validate matched histograms and same-bin copy covariance | Complete |
| 1D2 | Make Mantid histogram conventions the default and native alternatives explicit | Complete |
| 1D3 | Resolve remaining monitor peak-fit differences against the reference | Complete |
| 2A | Define explicit measurement and estimator contracts | Pending |
| 2B | Apply the contracts consistently to reductions, cuts, and exports | Pending |
| 3A | Persist complete source and reduction recipes | Pending |
| 3B | Edit recipes through one settings schema and targeted cache invalidation | Pending |
| 4A | Add basic run expressions and resolved-source preview | Pending |
| 4B | Add grouped, repeated, and stacked run expressions | Pending |
| 5A | Separate Sources, Reduction, Binning, and Plot controls | Pending |
| 5B | Expose statistical diagnostics and complete script replay | Pending |
| 6A | Validate instrument families and continuous measurements | Pending |
| 6B | Migrate projects, verify performance, and remove this plan | Pending |

Statuses are **Pending**, **Pending review**, **In progress**, **Complete**, or
**Needs subdivision**.
Completion means the stated deliverables have passed their checks and have been
committed, pushed, and synchronized. It does not mean the next checkpoint may
start automatically.

## 1. Establish and correct uncertainty statistics

### 1A — Diagnostic baseline

Deliver a private-data-free executable reference and tests for independent
Poisson counts with known exposure, unequal exposures, measured empty cells,
unmeasured cells, masks, and independent background subtraction. Include a seeded
repeated-sampling check and an optional comparison of previously exported nfit
and Mantid histogram slabs. Compare numerator, numerator variance, exposure,
intensity, uncertainty, and coverage separately; matching intensity is insufficient.

This checkpoint records current behavior and independent analytic references.
It does not alter production reduction, binning, or viewer behavior. A diagnostic
must label a modeled policy separately from a measured comparison.

**Initial evidence (2026-10-01):** nfit replaces the uncertainty of covered empty
DGS/MDE cells with a Feldman–Cousins upper limit, then histogram pooling squares
and sums those limits as independent variances. A confidence limit is not a
symmetric standard deviation. With 25 independent unit-weight events, 100 equal
exposures, and 99 empty fine cells, the observed Poisson standard error is 0.05;
the current per-cell floor model followed by nfit pooling gives approximately
0.13775. The same observation placed directly into one coarse cell gives 0.05.
These values use arbitrary consistent count/exposure units.

That establishes a partition-dependent algorithmic issue. It does not yet prove
which operation produced the exact screenshot. Paul reports excessive errors in
both subtracted and unsubtracted NiO and requests using unsubtracted NiO first.

Run the reference from the repository root:

```bash
PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
  benchmarks/benchmark_histogram_uncertainty.py \
  --output /tmp/nfit_uncertainty_reference.json
```

**Measured baseline:** previously exported unsubtracted 617-run NiO 3bar slabs
from shared Shiver MDE input have identical central-region counts and nonempty
numerator variances agreeing to numerical precision. For $|K|\leq0.1$ r.l.u.
and $6\leq E\leq14$ meV, pooling hidden momentum axes gives standard-error
ratios of 1.658 (median), 3.012 (95th percentile), and 7.392 (maximum), compared
with Mantid event variance using the same exposure. Empty covered cells account
for the added variance in this region. Outside it, small event-count differences
remain at bin boundaries with float32 edge rounding. These historical slabs do
not reproduce the exact energy cut and are not a fresh reduction benchmark.

The report is saved in
`benchmarks/results/histogram-uncertainty-baseline.json`; it records edges,
checksums, assumptions, support, and separate exposure/variance comparisons.
The reference benchmark's `--nfit-slab`, `--mantid-slab`, and `--mantid-data`
options reproduce the supplied-slab analysis. See `benchmarks/README.md` for the
array contract.

**Completion record:** version 0.105.5. The diagnostic and packaging tests pass
(17 tests), together with Ruff, byte-compilation, whitespace checks, and the
Sphinx build with warnings treated as errors. The committed baseline changes
only diagnostics and documentation. Local source and the existing ORNL
application/help are synchronized for this checkpoint.

**Review gate:** checkpoint 1A is complete. The uncertainty stage is deliberately
split into 1A–1D; Paul authorized 1B on 2026-10-01. The comparison prioritizes poorly covered fringes.

### 1B — Trace the NiO example

Reproduce the unsubtracted NiO slice and energy cut with explicit edges,
symmetry, masks, and aggregation settings. Trace raw events or shared MDE input,
histogram finalization, hidden-axis integration, viewer coarsening, and ordinary
and rotated cuts. Export the sufficient statistics at each boundary and compare
against Mantid using identical input and bins. Separate normalization differences
from event variance differences, including the previously identified per-run
versus first-run incident-energy convention.

**Acceptance:** identify the operations responsible for inflated error bars,
quantify their contribution, and show which discrepancies remain. Do not assume
the screenshot's source grid or exact recipe from its appearance. Use the shared
MDE input to isolate binning before repeating the native raw reduction.

**Measured findings (2026-10-01):** matched shared-MDE histograms isolate the
zero-count confidence-limit substitution, trajectory incident-energy convention,
and estimator change in cuts. In the bottom exposure decile, 90.0% of fine
cells are empty; empty-cell prescriptions contribute 87.8% of current numerator
variance. Positive-event map errors have median current/reference ratio 2.434
at the same exposure. Fringe fine-cell counts and nonempty event variances
agree with Mantid to numerical precision. A common trajectory energy brings
fringe exposure ratios from a maximum 3.524 to 1.000740. The full
`benchmarks/results/nio-uncertainty-fringes.md` and accompanying aggregate JSON
record support, exact edges, checksums, path comparisons, and remaining
fine-cell boundary discrepancies. Source slabs remain outside the repository.

A weakly exposed measured zero can legitimately have a large confidence upper
limit. Mantid's zero accumulated event variance does not make its unknown
intensity certain. The propagation defect and interval construction must be
addressed separately. For a spatial cut, count pooling estimates an
exposure-weighted response; a uniform spatial mean or integral has a different
target and assumptions about sparse coverage.

**Completion record:** version 0.105.7 adds the path diagnostic and synthetic
tests, clarifies permanent documentation, and retains production behavior.
All 30 uncertainty-diagnostic and packaging tests pass, along with Ruff,
byte-compilation, whitespace checks, and Sphinx with warnings treated as errors.
Manual Mantid reference runs are separate from pytest; no unit tests import or
execute Mantid/Shiver. Local and ORNL source/help are synchronized after commit.
The old native cube has reduction-cache version 1; current native reduction is
version 2. Fresh raw-reduction validation remains in 1D. Exact screenshot
recreation remains limited by its unavailable source recipe.

**Review gate:** checkpoint 1B is complete. Paul authorized 1C and 1D on
2026-10-01, including a Mantid-compatible default trajectory energy convention
and current-build performance measurements.

### 1C — Event variance and low-count inference

Retain actual accumulated event variance independently of confidence limits.
For independent weighted events with dimensionless corrected weights $w_j$, the
numerator is $C=\sum_j w_j$ and its observed counting variance is
$V=\sum_j w_j^2$. With a known exposure $N$, expressed in the recipe's exposure
units, intensity is $I=C/N$ and its diagonal variance is $V/N^2$.

Carry these statistics through histogram caches and pooled reductions. Treat
measured zero counts and missing exposure distinctly. Provide low-count intervals
or likelihoods as separately named outputs; do not present zero observed variance
as certainty about an unknown rate. Do not remove errors generically based on
`num_events == 0`: signed background differences and reconstructed data require
source-specific statistics.

**Acceptance:** direct coarse binning and fine-then-coarse pooling agree in
numerator, variance, and exposure under the same independence assumptions.
Empty-bin confidence intervals are constructed for the final requested estimate.
Serialization, lazy loading, cache invalidation, units, and script replay agree.

**Implementation record:** version 0.106.0 retains explicit $C$, $V$, and $N$
channels in native and MDE histograms, compressed caches, and lazy project loading.
Covered empty cells retain zero observed event variance rather than a confidence
limit substituted into that variance. Marked histogram slices and coarsening pool
these statistics; generic continuous measurements retain their existing estimator.
A separate public Poisson-rate interval helper requires independent integer counts,
a known exposure, and a known constant event weight. It is not automatically
applied to heterogeneous weighted events or symmetry-expanded counts.

Trajectory normalization defaults to the first participating run's full-precision
incident energy, matching MDNorm. Each-run normalization remains selectable in the
GUI and public API. Raw event reconstruction still uses each run's resolved Ei/T0.
Affected histogram signatures invalidate old results; the normalization convention
does not invalidate reduced-event caches. Viewer channels expose the numerator,
variance numerator, exposure, and observed standard error, with explicit semantics
for event contributions versus contributing histogram cells.

The full suite passed 2,321 tests with one skip; subsequent focused tests cover
mask partitions, same-bin covariance through both event backends, Qt labels, and
lazy cached-project round trips. Complete grouped event workflow script generation
remains a pre-existing gap tracked in `planned_features.md`; the scientific APIs
and saved settings are Qt-independent. Regular and rotated box cuts remain in 2B.

### 1D — Mantid-compatible DGS reduction and histogramming

**Scope agreed 2026-10-02:** finish the Mantid-compatible reference path first.
Make intentional native alternatives selectable, saved, and documented. Cross-bin
source dependency propagation moves to 2B, where cuts and pooling acquire an
explicit statistical contract. This change of scope does not claim that dependency
propagation is already implemented.

Validate native reduction and shared-MDE binning at the statistics level: incident
energy $E_i$ (meV), time zero $T_0$ (µs), retained pulses and charge, detector
geometry, vanadium, masks, weighted signal $C$, accumulated variance $V$, event
contributions, exposure $N$, and coverage. Prioritize fringe cells and the lowest
exposure decile. Use both six-operation 3bar and twelve-operation 3barm symmetry.
Compare the same input, bounds, actual bin counts, and symmetry; matching aggregate
intensity alone is insufficient. Manual Mantid jobs remain separate from pytest.

**Earlier evidence:** versions 0.106.0–0.106.1 corrected empty-bin error-floor
propagation, added sufficient-statistic channels, and diagnosed the remaining
monitor and event-boundary differences. Their measurements are retained in
`benchmarks/results/dgs-reference-current.md` and its aggregate JSON. The earlier
remaining −708 cube contributions and one thin-slice contribution were not a
parity claim.

#### 1D2 — Explicit histogram conventions

The reference defaults reproduce Mantid's float32 weighted-event storage,
independently accumulated event weight and variance, coordinate transforms, bin
edges, trajectory midpoint assignment, and independent symmetry-copy variance.
These numerical conventions are implemented in nfit without calling Mantid.
Reduced-event caches retain separate weight and variance columns. Version 5
invalidates older five-column caches and interim caches with nominal He-3 geometry. Histogram signatures include effective
policies and their numerical version.

The He-3 reference path reproduces the radius obtained from Mantid's distant
ray and source-ordered absorption correction for checked local cylinders. The
cylinder's bottom and height are part of this calculation. Unsupported declared
shape rotations or transverse offsets fail explicitly; incomplete IDFs without
height retain a documented nominal-radius fallback.

Selectable alternatives retain double-precision event arithmetic, nominal He-3
tube radius, stable absorption arithmetic, and same-bin symmetry-copy covariance. They are scientific choices, not a promise of improved
accuracy for every dataset. Same-bin covariance is appropriate when a source event
and its transformed copies are contributions to the same final estimate; it does
not propagate dependencies between different histogram bins through later cuts.
For that optional calculation, bin the original cached events directly onto the
final requested grid. Nonuniform explicit axes have no BinMD equivalent and retain
their supplied boundaries; uniform-grid reference parity is not claimed for them.

The GUI exposes the policies after import. Public setters validate atomically,
and the policy script export replays the choices without Qt. The snippet does not
include source import and binning recipes; complete grouped workflow export
remains in 5B. Changing monitor or event precision regenerates affected raw
caches, while changing symmetry variance retains reusable events.

#### 1D3 — Monitor calibration

The default follows GetEi's histogram variance, reciprocal-width multiplication,
peak-tail derivative parentheses, and floating-point stopping rules. In
particular, a negative cancellation residual has the reference NaN comparison
behavior rather than raising a Python exception and silently substituting nominal
$E_i$ and $T_0$. All 617 NiO runs agree with the reference to floating-point
roundoff: maximum absolute differences are $3.979\times10^{-13}$ meV in $E_i$
and $1.910\times10^{-11}$ µs in $T_0$, with no calibration failures.

The nonnegative derivative-variance formulation remains an optional monitor
policy. Its peak selection can differ slightly; numerical stability alone does
not establish that it estimates the peak more accurately. Physical conversion
constants and arithmetic order are pinned to the measured Mantid reference.

**Validated shared-MDE results:** all 91,584,578 cells in the 617-run 3bar cube
have identical event counts, signal numerators, and variance numerators. Exposure
relative L2 difference is $1.95\times10^{-10}$; the maximum relative difference
is $1.20\times10^{-5}$ in an energy-boundary cell. The lowest exposure decile
has maximum relative uncertainty difference $1.61\times10^{-11}$. Its engine
wall time is 244.20 s in nfit and 751.00 s in Mantid on the same host, excluding
Mantid's separate 118.43 s MDE load. This isolates histogramming and does not yet
establish fresh raw-reduction parity.

**Validation record:** the full-suite run passed 2,412 tests with one skip for
an unavailable CuPy electronic-response backend. Final focused geometry,
reduced-cache, and architecture checks passed 122 tests. Ruff, byte-compilation,
whitespace checks, and Sphinx with warnings treated as errors passed. Neither
production code nor unit tests import or execute Mantid/Shiver; external engine
comparisons remain manual diagnostics.

**Final fresh-raw validation:** all 617 NiO runs were reduced once into temporary,
dataset-owned disk caches. Event contributions, signal numerators, and variance
numerators match Mantid literally in every 91,584,578 cube cell and every
130,235 fine-cut cell. Coverage and the 13 event contributions outside exposure
also match. The largest cube uncertainty residual is 12.029 ppm at an $E=0$
boundary; the lowest-exposure decile has maximum relative uncertainty difference
$5.35\times10^{-11}$. The fine cut's selected low-coverage region agrees to
$3.92\times10^{-13}$ relative uncertainty. Exposure has a small numerical
residual; bitwise equality is not claimed for it.

Fresh reduction plus twelve-operation fine binning and cache creation took
371.46 s. The subsequent six-operation cube took 238.93 s with 617 cache hits
and no reduction. Temporary caches occupied 10.70 GiB; fine-operation peak RSS
was 3.49 GiB and cube peak RSS was 41.83 GiB. Both saved NiO projects remained
unchanged and all temporary reduced caches were removed. The standard sequential
Mantid reduction estimate is about 139 minutes, extrapolated from three measured
runs; it is not a measured full 617-run reduction. Full measurements, input and
symmetry contracts, residuals, and limitations are recorded in
`benchmarks/results/dgs-parity-final.md` and its aggregate JSON.

The screenshot's saved sample recipe named **Al2O3 3barm zoom** resolves to a
different set of matrices from the explicitly supplied twelve-operation 3barm
recipe. The comparison uses the supplied operations and exact selected grid;
the report records the saved-generator difference independently of parity.
Future GUI work will expose resolved operations alongside recipe names.

**Completion record:** version 0.107.0. The existing ORNL application and help
files use the same source as the local checkout; the desktop launcher is retained.
Affected reduced-event caches regenerate under version 5, while policy and
algorithm versions invalidate affected histograms. Other measurement pipelines
retain their previous conventions.

**Review gate:** phase 1D is complete and awaits review. Start 2A only when Paul
authorizes it.

## 2. Measurement and estimator contracts

### 2A — Choose according to the measurement

Specify public contracts for:

- **Counts with known exposure:** pool count numerators and exposures before
  division when estimating a common intensity within the bin.
- **Independent continuous measurements with supplied errors:** an
  inverse-variance mean where the common-value and independence assumptions apply.
- **Samples of a continuous function:** an explicitly defined coordinate-interval
  average or integral with sampling-width weights and stated interpolation.
- **Signed or correlated reconstructed measurements:** propagate the linear
  contributions and their dependencies without imposing Poisson errors on the
  reconstructed values.

Define the target quantity, units, missing-data behavior, coverage, and uncertainty
for each choice. Distinguish a mean, an integral, and a count sum. Use statistical
assumptions to recommend defaults; let users inspect and override those assumptions.
Test heterogeneity within a bin and uncertain normalizers explicitly before claiming
an estimator has the most accurate uncertainty for every acquisition type.

### 2B — One numerical path

Use the same GUI-independent services for rebinning, hidden-axis slicing,
coarsening, regular/rotated cuts, region summaries, fits, and exported profiles.
Carry masks and support consistently across numerator, variance, and exposure.
Preserve or replay source dependencies when slices or cuts reunite symmetry
copies, repeated sources, or fractional assignments. Validate shared calibration
and background dependencies separately. Use bounded source-aware storage or
cached-event replay rather than a dense 4D covariance matrix.
Preserve sufficient statistics in derived views rather than inventing event counts.

**Acceptance:** equivalent GUI and script operations agree; changing an
intermediate grid does not change the final estimate under the declared model.
Existing saved choices migrate explicitly and retain their recorded provenance.

## 3. Persistent source and reduction recipes

### 3A — Complete reproducible configuration

Persist source expressions and resolved files/runs, shared reduction defaults,
per-run overrides, resolved automatic values, and algorithm/calibration provenance.
Keep source identity and acquisition geometry per run, including mixed instruments.
Separate acquisition/reduction settings from coordinate transforms, histogram
recipes, and display settings. Choose the ownership of UB and related transforms
based on the scientific dependency graph.

### 3B — Editable settings and cache dependencies

One authoritative settings schema supplies validation, GUI controls, serialization,
public scripting, tooltips, and cache keys. Every supported adjustable reduction
setting remains visible and editable after import; show automatic choices alongside
resolved values. Support physically valid negative time-zero overrides without
using a numerical sentinel for automatic mode.

Invalidate and reconstruct only affected run reductions and dependent binnings.
Keep bounded, lazy, dataset-owned reduced-event caches with streamed on-disk
blocks; retain the uncompressed storage chosen for fast reuse. Adding or removing
one run must not require reducing unrelated runs again. Calibration edits and geometry
reuse require complete dependency signatures.

**Acceptance:** saved edits replay without Qt, affected caches rebuild, unrelated
caches remain reusable, and project reopening is lazy.

## 4. Run-expression selection

### 4A — Basic expressions

Provide a standard source selector with directory, prefix, suffix, optional number
padding, run expression, and expanded preview. Support lists, inclusive ranges,
strides, and summed groups. Preview grouping, missing files, duplicate identities,
and available metadata before import. Retain explicit file selection for irregular
names, containers, and files without run numbers.

### 4B — Advanced grouping

Read the GRASP `numor_parse.m` grammar for repeated, blocked, cumulative, and stacked
selections. Describe supported syntax and its nfit meaning; preserve intentional
semantics rather than parser bugs. Distinguish grouping from summing already divided
intensities. Preserve each run's energy, geometry, exposure, and source identity.
Repeated sources must retain their correlations; empty slots mean missing data.

**Acceptance:** expansion is deterministic and bounded, errors are actionable,
large ranges remain responsive, GUI and scripts resolve identically, and grouping
does not alter the statistical meaning of the measurements.

## 5. GUI and scripting workflow

### 5A — Separate responsibilities

Organize dataset groups around **Sources**, **Reduction**, **Binning and
combination**, and **Plots and cuts**. Retain the useful current import controls.
Show shared defaults, mixed values, overrides, and resolved automatic results.
Preview resolved symmetry operations and their coordinate convention; a recipe
name alone must not imply a particular group orientation.
Use SEQUOIA's separate data/slice recipes as design inspiration while supporting
other acquisition types through instrument adapters.

### 5B — Diagnostics and complete replay

Expose counts/numerator, exposure, coverage, statistical uncertainty, confidence
intervals, and relevant provenance as appropriate. Clearly distinguish scientific
smoothing of numerator/exposure from display interpolation and propagate the
introduced dependencies. Export complete editable workflows through public APIs.

**Acceptance:** no changeable reduction setting is hidden; dataset edits can
retrigger reduction and binning independently; controls have tested tooltips;
GUI actions round-trip without constructing widgets.

## 6. Cross-instrument acceptance and cleanup

### 6A — Representative measurements

Validate SEQUOIA and HYSPEC DGS, CORELLI reconstruction, MACS measurements,
continuous-wave diffraction, and synthetic densely sampled susceptibility versus
temperature. Include uneven coverage, repeated measurements, independent and
shared backgrounds, fractional/discrete binning, and symmetry. Match Mantid where
it implements the same model; analytic and repeated-sampling references decide
correctness when conventions differ.

### 6B — Migration, performance, and completion

Validate existing projects and a documented migration path. Measure native reduction,
first binning, cached-event rebinning, cache size, peak memory, and reopen latency.
Report measured Mantid/Shiver job timings separately from estimated conventional
single-job timings, with the same data/settings and explicit uncertainty assumptions.
Geometry and calibration reuse must verify equivalence for mixed instrument inputs.

Once all checkpoints are accepted, replace temporary findings with permanent
physics/workflow documentation and delete this plan and its navigation references.
