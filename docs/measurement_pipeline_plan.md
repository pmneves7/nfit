# Measurement pipeline implementation plan

This temporary plan tracks the agreed import, reduction, binning, and uncertainty
refactor. Scientific correctness takes priority over retaining an incorrect
recipe. nfit must support event spectrometers, continuous-wave neutron instruments,
diffractometers, and continuous measurements such as susceptibility versus
temperature. Instrument names alone must not determine the statistical estimator.

## Working agreement

- Paul authorized the current DGS-focused work on 2026-10-03: investigate HYSPEC
  background correctness, assess the statistical motivation of departures from
  Mantid, and profile/improve HYSPEC and SEQUOIA reduction and binning. CORELLI
  and MACS adapter work is deferred. Review completed checkpoints and remaining
  decisions with Paul; new scientific defaults still need an explicit decision.
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
- Assess optional DGS precision, monitor-fitting, trajectory-energy and
  symmetry-variance treatments in the authorized DGS investigation. Existing
  compatibility defaults stay fixed until the 6A2 decision, except Paul's
  approved Shiver-compatible time-weighted sample angles. Continue performance
  work and conditional statistical validation after that correction. Accept
  observed-extrema float32 clipping as rounding; defer continuous-rotation
  event reconstruction. Mathematical fixes and speedups must state assumptions.
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
| 2A | Define explicit measurement and estimator contracts | Complete |
| 2B | Apply the contracts consistently to reductions, cuts, and exports | Complete |
| 2B1 | Share box-profile estimation and retain profile statistics | Complete |
| 2B2 | Preserve estimator statistics through project rebinning and derived views | Complete |
| 2B3 | Preserve/replay source dependencies through cuts and arithmetic | Complete |
| 2B4 | Complete region, fitting, export, and migration parity | Complete |
| 3A | Persist complete source and reduction recipes | Complete |
| 3B | Edit recipes through one settings schema and targeted cache invalidation | Complete |
| 4A | Add basic run expressions and resolved-source preview | Complete |
| 4B | Add grouped, repeated, and stacked run expressions | Complete |
| 5A | Separate Sources, Reduction, Binning, and Plot controls | Complete |
| 5B | Expose statistical diagnostics and complete script replay | Complete |
| 6A | Finish DGS correctness and background validation | Complete; accepted by Paul |
| 6A-P | Complete realistic performance comparisons and useful speedups | Complete; accepted by Paul |
| 6A1 | Validate optional numerical and statistical methods | Complete; controlled checks reviewed, limitations recorded |
| 6A2 | Review evidence and choose future defaults with Paul | Complete |
| 6B-S | Make resident cubes and project storage safe; add Resource Manager | Pending review |
| 6B | Validate migration, finish documentation, and remove this plan | Pending |

Statuses are **Pending**, **Pending review**, **In progress**, **Complete**,
**Deferred**, or **Needs subdivision**.
Completion means the stated deliverables have passed their checks and have been
committed, pushed, and synchronized. It does not mean the next checkpoint may
start automatically.

Stage 6 uses the five checkpoints above, with detailed checklists below.
CORELLI and MACS follow-ups remain deferred; they do not block the current DGS
work. Earlier stage 6 subcheckpoint names are retained only as evidence references.

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

Selectable alternatives retain double-precision event arithmetic and same-bin
symmetry-copy covariance. They are scientific choices, not a promise of improved
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

**Review gate:** phase 1D is complete. Paul authorized 2A on 2026-10-02, alongside
independent diagnostics of the trajectory residual and performance candidates.

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

**Implementation boundary:** deliver a versioned public `MeasurementContract`,
immutable one-bin `MeasurementEstimate`, sparse primitive `SourceTerm` sensitivities,
and reference estimators independent of Qt and instrument adapters. Existing project
binning, slicing, fits and exports retain their current dispatch until 2B. Contracts
serialize complete assumptions without attaching new defaults to existing datasets.
Unit labels are explicit; callers perform physical unit conversion before combining.
Uncertain-normalizer propagation is explicitly first-order, not a universal likelihood.

**Completion record (2026-10-02, 0.108.0):** the public contracts and scalar
reference estimators are implemented in `measurement_contracts.py` and
`measurement_statistics.py`, with examples in `docs/measurement_statistics.md`.
Analytic tests cover unequal exposure, measured zeros, coordinate support,
shared backgrounds/calibration, signed reconstruction, uncertain normalization,
heterogeneity diagnostics, serialization, and immutable inputs. Architecture
tests enforce GUI independence and authoritative public exports. The focused
contract, architecture, and packaging run passed 125 tests; histogram and dataset
regressions passed another 31. Ruff, byte-compilation, whitespace checks, and
Sphinx with warnings treated as errors passed. No test invokes Mantid or Shiver.

The separately authorized audits identified the entire trajectory residual as
float32 boundary leakage and measured a parity-preserving fused-kernel prototype.
See `benchmarks/results/dgs-trajectory-boundary.md` and
`benchmarks/results/dgs-parity-speedup-audit.md`. Neither audit changed production
scientific settings. Alternative validation and default adoption remain at 6A1
and 6A2. Checkpoint 2B requires Paul's next authorization.

### 2B — One numerical path

Paul authorized 2B on 2026-10-02. The audit requires four reviewable checkpoints:

- **2B1:** one prepared-view service for regular and rotated box profiles,
  retaining count numerator, variance, exposure, contributions, masks and support;
  use it in static/Qt plots, live cut viewers and existing CSV export. Preserve
  legacy continuous weighting explicitly. Region sums retain their stated linear
  target.
- **2B2:** attach and preserve contracts/sufficient statistics through project
  rebinning, slices/coarsening and derived containers. Test direct/staged
  equivalence for each supported estimator and document when source replay is
  required rather than silently redistributing histogram centers.
- **2B3:** bounded source-dependency propagation or cached-source replay for
  fractional assignments, symmetry copies, shared calibration and backgrounds.
- **2B4:** route region estimates, fit preparation/model projection and exports
  through the complete contract; validate GUI/script parity, compatibility
  migration and recorded provenance.

Paul subsequently authorized continuing through the remaining stage 2 checkpoints.
Review stage 2 before starting stage 3.

**2B1 completion record (2026-10-02, 0.109.0):** `measurement_profiles.py`
provides bounded grouped estimates and immutable prepared payloads. Static and
Qt regular/rotated box cuts share this service; model overlays use observation
weights and record prediction provenance. Live cut viewers retain C,V,N and source
contribution semantics. CSV exports retain available statistics and a versioned
contract sidecar; historical three-column array export remains available.
Measured zero-count cells contribute exposure. Masks, coverage cutoffs, missing
values, stable precision means, and unsupported declarations are explicit.
Statistics metadata survives slicing; smoothing invalidates the unsmoothed
contract. Tests cover analytic references, coarsening followed by cuts, rotated
selection, archive/live-view preservation and GUI/script/export equivalence.

**Validation:** the full suite passed 2,486 tests with one optional GPU skip.
After the final profile-mask and tooltip changes, the focused profile, statistics,
export, architecture, Qt and packaging suite passed 136 tests. Ruff,
byte-compilation, diff checks and the Sphinx build with warnings treated as errors
passed. These tests do not invoke Mantid or Shiver.

Only diagonal histogram variances are available in this checkpoint. Shared-source
and uncertain-normalizer contracts fail explicitly until the corresponding
payload/replay support exists. Background differences, waterfalls, region sums,
full fit likelihoods and legacy migration remain in 2B2–2B4. Scientific alternative
validation/default adoption remains at 6A1–6A2. The next completion record covers
the jointly validated 2B2–2B4 integration.

**Cluster completion/storage record:** the requested background-project rebuild
finished all nine histograms, validated 2,775 v5 reduced-event caches and lazy
archive backings, and installed the 89,869,291,072-byte result. The previous file
is backed up in IPTS-37189; the separate NiO-and-sapphire project is unchanged.
Task-owned scientific archives and diagnostics were moved from home and `/tmp`
to IPTS diagnostics storage. Home usage is 7.9 GiB. The desktop launcher now only
starts the existing application. Scientific staging is allocated beside the
owning project/source or selected output when needed,
without cluster-specific paths or startup mkdir calls. An unavailable experiment
mount affects the operation, not an empty application launch. Node23's home-data
IPTS link was confirmed dangling; node01 can access the same experiment. Repair
of that node's mount requires a working filesystem alias or administrators;
direct node23 access remains unavailable for verification.

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

**2B2–2B4 completion record (2026-10-02, 0.110.0):** explicitly declared count
and continuous measurements retain additive statistics through original-point
binning, aligned histogram coarsening/composites, hidden-axis slicing, displayed
coarsening, regular/rotated profiles, waterfalls, fit preparation and CSV export.
Original one-coordinate sampled functions use interval interpolation and retain
shared nodal sensitivities. Intermediate means preserve their original weights;
independently initialized precision scales reconcile before combination.
Continuous measurements do not require neutron event counts to establish support.

Bounded immutable sparse dependencies and numerator/exposure bundles retain
represented shared uncertainty through cuts, arithmetic, deterministic calibration
and archives. Region and Bragg linear sums combine signed coefficients before
squaring. Final-grid raw-DGS/MDE replay reuses reduced-event caches under recorded
numerical policies. Lazy archive loading remains dataset-owned; copies share
immutable factors. Grid planning does not accumulate a dummy output histogram.
Model projection uses observation weights and propagated observation errors,
without declaring those errors to be fitted-model uncertainty.

Gaussian remains the default fit objective. Optional bounded nonsingular GLS and
audited constant-weight integer Poisson deviance retain their declarations through
the project compiler, saved settings, reports and script API. Measured zeros
participate in the count objective; expected-Fisher parameter errors are explicitly
asymptotic. Unmarked saved recipes retain compatibility provenance. Declared
derivations invalidate stale targets rather than silently falling back to a mean.

**Explicit boundaries:** current large DGS caches do not reconstruct every event's
cross-bin lineage or shared detector-calibration factors. Represented payloads or
final-grid replay are required; no missing covariance is inferred. Multidimensional
function interpolation, declared metadata-axis stacking, cross-reflection Bragg
covariance, singular primitive-source fits and full source-adapter enrichment remain
in the later workflow/acceptance work. Unsupported operations raise explicit replay
requirements. Uncertain exposure remains first-order ratio propagation. Smoothing
is a plot preview; declared scientific aggregation needs an appropriate source
model. Scientific alternative validation and default adoption remain in 6A1–6A2.

**Validation:** the full regression suite passed 2,582 tests with one optional CuPy
skip. Subsequent focused runs cover the final grid-planning, immutable-copy,
shared model-overlay and signed-conversion checks. Ruff, byte-compilation,
whitespace checks and Sphinx with warnings treated as errors passed. Ordinary
tests neither import nor call Mantid/Shiver. The existing local and ORNL application,
source mirror and help are synchronized; scientific project data stay in IPTS.
Neither NiO project was rebuilt or rewritten in this stage.

**Review gate:** stage 2 is complete within these explicit supported models.
Paul authorized stages 3 and 4 on 2026-10-02. Keep this document until the full refactor
has been accepted; then remove it and its navigation entries.

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

**Stage 3 completion record (0.112.0):** native DGS, CORELLI and MDEvent
imports and project saves persist metadata-only source/reduction recipes with
current run membership, shared defaults, per-run overrides, resolved automatic
values and per-run geometry/calibration provenance. A schema supplies scientific
validation, serialization and focused Qt controls; signed T0 uses explicit
automatic mode. UB belongs to the group coordinate transform. DGS cache edits
invalidate only effectively changed run reductions; histogram signatures include
calibration identities and effective overrides. CORELLI verifies each current
embedded geometry before reuse. Reopening continues to bind lazy event assets.

Editable standalone native-group scripts replay source/reduction/grid settings
without Qt; project-backed composite scripts expose editable native sample and
background recipes while retaining topology. Complete standalone arbitrary
composite export remains 5B. CORELLI reconstruction still depends on its requested
energy hypotheses and does not use the DGS laboratory-event cache. Source file
identities use size and nanosecond modification/change times, with full embedded
geometry hashes; large acquisition files are not content-hashed.

See [Source and reduction recipes](reduction_recipes.md) for current usage.
Stage 4 is authorized; scientific alternative/default decisions remain 6A1–6A2.

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

**Stage 4 completion record (0.113.0):** the public bounded resolver supports
GRASP-inspired basic, summed, blocked, repeated, empty and second-index
expressions, plus legacy `start:stride:end` syntax. Directory/prefix/suffix and
padding are saved alongside resolved appearances. A worker-thread preview reports
missing/repeated identities and optional metadata without reading event arrays;
the table is bounded to 1,000 rows. The GUI delegates atomic import to the same
public service as editable scripts.

Flat import creates one ordinary nfit dataset group with unique physical source
membership. Optional expression grouping creates ordinary dataset subfolders;
there is no additional project grouping framework. Repeated raw/MDE sources
within a summed subfolder retain a correlated multiplicity coefficient, without
inventing counting precision. Cross-subfolder aliases retain physical identity
and are rejected when a composite would merge them without shared covariance.
Acquisition lineage survives histogram and fit preparation; separate positive-
weight fit blocks reject overlapping tracked acquisitions. The check is
conservative until primitive-level independence or joint covariance is represented.
CORELLI/ordinary repeated coaddition likewise requires an adapter extension;
unmarked legacy entries retain their existing behavior. Empty retained folders
are disabled. Mixed reduction families use separate dataset groups.

Saved source expressions are visible in collection details. Selection scripts
resolve the current directory; saved reduction recipes retain their resolved
logical membership. Sources and reduced caches remain lazy. See
[Selecting numbered sources](source_selection.md). Stage 5 follows below;
alternative validation/default adoption remain 6A1–6A2.

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

**Stage 5 completion record (0.114.0):** ordinary dataset groups now present
Sources, Reduction, Binning and combination, and Plots and cuts separately.
Shared defaults, mixed effective values, per-run overrides and resolved
automatic calibration remain visible through the authoritative settings schema.
Resolved symmetry matrices state their reciprocal-HKL coordinate convention.
Measurement combination labels explain the target and weights without changing
estimator defaults.

Source expressions can edit existing groups through `update_source_selection`.
Edits inspect only new files, retain unchanged entries and reduced caches, and
invalidate changed combinations. Missing inputs and dangling saved background
references fail atomically. The GUI checks all project workspaces; scripting
callers supply external reference roots when needed. Existing expression groups
remain ordinary subfolders.

The viewer exposes count numerators, numerator variance, exposure, coverage,
standard uncertainty and a bounded metadata-only provenance report. Audited
independent constant-weight Poisson models additionally offer final-bin 68.27%
Garwood bounds, recomputed after integration/coarsening. Weighted DGS,
symmetrized, subtracted or uncertain-normalizer data are not assigned an
unsupported exact interval. Display smoothing remains a plot approximation.
The optional `smooth_count_histogram` API smooths numerator/exposure with
represented primitive dependencies; it does not change defaults or infer missing
cross-bin covariance. Representative-data validation remains at 6A.

Standalone composite workflows now contain original source descriptors, native
reduction recipes, nested topology, ancestor masks/lattice, selected grids,
scales and linked background closure, including external workspaces. They do
not require a saved project or Qt. Source-less replacements, custom callable
transforms and live derived-analysis inputs fail explicitly and remain separate
export extensions. Scientific background context fixes prevent inherited-root
self cycles, repeated root subtraction and squared reference result scales;
versioned affected background caches recompute while reduced-event caches remain
reusable. Genuine cycles still fail.

Paul authorized the representative-measurement survey in 6A. Scientific
alternative validation and default adoption remain 6A1–6A2; no main NiO project
is rebuilt for this checkpoint.

## 6. DGS acceptance and completion

Finish SEQUOIA and HYSPEC first. The five checkpoints below replace the nested
stage 6 tracking list. Review each coherent change with Paul before starting the
next. Keep scientific originals unchanged and all ORNL scientific outputs in
IPTS folders. Mantid/Shiver comparisons are manual validation jobs, outside
production nfit and its unit tests.

### 6A — Finish DGS correctness and background validation

**Status: Complete; accepted by Paul on 2026-10-04.** Geometry, timing,
directional background reconstruction and cached-background profile uncertainties
are validated. Time-weighted angle replay matches fresh Mantid histogramming;
remaining accepted differences have explicit attribution.

Completed work:

- [x] Resolve each raw DGS run's logged geometry, timing and masks. Reuse geometry
  only after complete equivalence checks, including mixed-instrument inputs.
- [x] Validate raw SEQUOIA and both HYSPEC detector-bank configurations against
  matched Shiver recipes, with recorded energy-boundary differences.
- [x] Average the HYSPEC dummy sample in laboratory coordinates and replay it at
  each sample rotation. Preserve dummy geometry for dummy events/normalization
  and sample geometry for sample events/normalization.
- [x] Preserve shared dummy-event correlations through supported cached-field
  cuts, hierarchical bank composites, asynchronous GUI replay and exports.

Final checks:

- [x] Match Mantid's duration/accepted-time averaging of raw DGS rotation logs.
  Run 505784 matches the reference angle exactly. Record effective angles and
  convention in public recipe provenance and GUI details; reduction version 7
  invalidates prior reduced-event/binned caches lazily. Native tests cover pause
  and bad-pulse intervals, unequal durations, repeated/backdated timestamps,
  legacy constants, source/calibration reuse and saved lazy-cache reopening.
- [x] Repeat the matched 34° and 70° pilots with six and twelve symmetry copies.
  Compare event numerator, variance, contribution counts, exposure, masks,
  signal and uncertainty in every cell; prioritize low-coverage fringes and
  the HHH/energy cut at K=0±0.03 and L=0.33±0.02 r.l.u.
- [x] Attribute remaining trajectory boundaries and support differences.
  Paul accepts the known observed-extrema float32 event clipping as immaterial
  rounding; no loss-emulation option is needed. Separate that rounding from
  recipe differences before acceptance.
- [x] Review the DGS correctness results with Paul. Complete sample covariance
  between bins created by symmetry is explicitly deferred to a later checkpoint;
  it does not block this accepted reduction/background scope.

**Corrected angle issue:** nfit previously used 48.5016271525° for run 505784.
Version 0.116.3 uses Mantid's pause-filtered time mean of
48.504392463390396°, verified against the real raw timestamps. With a 95%
bad-pulse threshold the native resolved angle is 48.50439215956369°; pulse
filter settings therefore remain part of the recipe. The other 23 pilot
rotations, Ei, T0 and UB already agreed. Fresh matched six/twelve-copy
replays give exposure relative L2 differences of 5.2×10⁻¹⁵–3.5×10⁻¹⁴ with
identical support in both banks. Count excess is confined to accepted
energy-edge clipping (32/30 contributions for six copies; 64/60 for twelve).
The old loaded-MDE exposure discrepancy also occurs within Mantid itself.
Its save/load parameter precision changes detector directions; native replay of
the actual loaded directions reproduces the residual, while UB, sample rotation
matrices and charge are unchanged. Preserve the original physical geometry.
See `benchmarks/results/hyspec-mde-roundtrip.md` for attribution and the fresh
resident controls. Continuous rotation remains deferred.

**Background scope:** the dummy has the same mount, glue and aluminum without
crystals; its coarse 180° sweep averages orientation effects. Retain detector
direction and energy dependence. Encoder offsets between acquisitions are within
instrument tolerance and do not justify substituting sample geometry into the
dummy denominator. Current averaging is charge weighted. Paul prefers an explicit
angular-average option for the dummy and has scheduled it for a later checkpoint.

Exact cached-field replay estimates the sample-exposure-weighted subtracted
field and combines each shared background event's coefficients before squaring.
Independent primitive references validate both banks, rotated cuts and coverage
fringes. This certifies represented background covariance **within each final
profile bin**. Sample uncertainty retains its recorded policy; covariance between
final bins and unrecorded calibration dependencies remain unavailable. Unsupported
composites and box sums retain explicitly labeled diagonal uncertainty or require
full source replay. See `docs/planned_features.md` for those remaining treatments.

Evidence for former 6A-R, R2 and R3a/R3b is retained in:

- `benchmarks/results/hyspec-dgs-parity.md`
- `benchmarks/results/hyspec-background-analysis.md`
- `benchmarks/results/hyspec-background-physical-assumptions.md`
- `benchmarks/results/hyspec-cached-background-profiles.json`
- `benchmarks/results/hyspec-bank-background-profiles.json`
- `benchmarks/results/dgs-user-workflows-node19.md` for the remaining angle and
  histogram differences (formerly 6A-P4).

### 6A-P — Complete realistic performance comparisons and useful speedups

**Status: Complete; accepted by Paul on 2026-10-04.** Matched full
SEQUOIA/HYSPEC workflows, every-cell comparisons, trajectory pooling,
guarded symmetry-cache expansion and the
remaining-cost profile are complete. Ordered event parallelism was tested but
its small workflow gain does not justify adoption. Code and documentation are
synchronized locally and on ORNL; scientific default changes remain in 6A2.

**Background follow-up (2026-10-06): implemented; awaiting Paul's review.**
Original background entries retain reduced-event caches across binnings and
sample datasets referencing the same background group. A matched native trial
with 90 sample runs and all 11 dummy files pools exactly matching trajectory
exposures across files: background preparation/replay improves 140.29 → 99.34 s
(29.2%). Every cell passes the numerical gate; support, counts, masks and edges
are literal, with signal/exposure/uncertainty differences below 2.35e-15 relative.
Different grids still require event replay; separately imported copies still
have separate reduction caches. Full-project profiling and cross-entry cache
sharing remain separate questions. Evidence is in
`benchmarks/results/hyspec-background-pooling-node21.md` and the earlier cache
ownership/batching report `hyspec-background-batching-node21.md`.

- [x] Fuse event projection/accumulation and reuse exact-text mask parsing.
- [x] Reuse successfully inspected scalar metadata and exact geometry arithmetic
  with complete source, policy, mask and resolved-geometry checks.
- [x] Parallelize large cached histogram reads while retaining the validated
  original file descriptor, checksums, memory limits and platform fallbacks.
- [x] Measure the full 617-run native SEQUOIA workflow before/after those changes,
  including saves, reopen, lazy event-cache reuse and later twelve-copy binning.
- [x] Measure matched SEQUOIA and both HYSPEC bank pilots in ordinary Shiver jobs.
- [x] Finish the full 617-run Shiver reduction, saves/reopens, and one/six/twelve
  symmetry-copy histograms. Validate arrays and the low-coverage cut before
  publishing the complete cross-engine comparison.
- [x] Validate and measure duplicate-trajectory pooling against the current
  time-weighted-angle build. Require literal event statistics, identical exposure
  support and every nonzero exposure cell within 10⁻¹² relative.
- [x] Measure representative full HYSPEC workflows after the 6A corrections.
- [x] Measure disjoint-bin event parallelism with all 617 real cached sources.
  Keep the production path unchanged: its 14.5 s event-pass gain is only about
  5% of the complete rebin/save workflow and adds separate numerical paths.
- [x] Reuse compatible cached symmetry contributions and calculate only the
  missing operations. Validate unchanged source/grid/policy signatures, full
  statistics and recomputed masks against complete replay; measure saved workflows.
- [x] Profile the remaining dominant costs, particularly trajectory normalization.
  Adopt further speedups only when real workflows show a worthwhile gain and
  preserve the selected numerical treatment; otherwise record the limiting cost.
- [x] Review timings, numerical differences and remaining opportunities with Paul.
  Accept the measured speedups and omit the ordered-event parallel prototype;
  its approximately 5% whole-workflow gain does not justify the added complexity.

**Current measured workflows:** same node, 64-thread ceilings, matching settings,
loading and saving included. The reference is an ordinary sequential Shiver
reduction with normal internal threading, followed by Mantid MDNorm.

| Workflow | nfit | Shiver/Mantid |
| --- | ---: | ---: |
| SEQUOIA 617 runs, first saved six-operation dataset | 8m16s | 4h25m16s |
| SEQUOIA, later twelve-operation bin and save | 3m19s | 52m26s |
| HYSPEC 34° bank, 361 runs, first saved dataset | 1m25s | 23m55s |
| HYSPEC, later twelve-operation bin and save | 34.4s | 1m42s |

Checked trajectory pooling removes 34.8% of full SEQUOIA trajectory tasks.
Compatible symmetry expansion reuses six operations and computes six missing
ones. A separate matched saved-project control improves 260.527 → 200.654 s
for SEQUOIA with the final workload guard; HYSPEC improves 49.665 → 41.626 s.
These controls retain existing histograms and differ from the standard two-bin
workflow above. Counts, C/V, masks and exposure support are exact over both full
grids; exposure changes only through summation rounding. The standard native
six/twelve gates agree in every nonzero exposure cell within 1.61×10⁻¹⁴
relative, including the NiO low-coverage cut and fringes.

A full617 normalization sweep favors 64 workers over 32 (median 127.526 versus
142.320 s). Ordered event parallelism saves 14.463 s in the isolated real event
pass but only about 5% of the complete saved workflow, so it is not adopted.
The instrumented raw profile confirms geometry reuse and many small fresh-bank
chunks. Fresh detector-chunk batching is planned for a later performance
checkpoint. Measure complete reduction/bin/save workflows and preserve event
order, cache contents, full-grid/fringe numerical results, memory bounds,
progress and cancellation. No gain is assumed before that experiment; producer
concurrency or fused reconstruction needs stronger evidence before implementation.

The final cost guard skips partial-cache loading when a large grid has too few
source events to repay its scans. The 24-run selection correctly retains full
replay, avoiding the measured regression. The final suite passes 3,234 tests
with one optional CuPy skip. Detailed stage receipts, support/uncertainty gates
and accepted Shiver clipping/geometry-rounding differences are in:

- `benchmarks/results/dgs-user-workflows-node19.md`
- `benchmarks/results/dgs-symmetry-cache-node19.md`
- `benchmarks/results/dgs-remaining-performance.md`

Report loading, reduction, event-cache/MDE persistence, first binning, saving,
reopening and later rebins separately. The primary Shiver comparison uses one
ordinary sequential reduction job with its normal internal threading and bins
its resident MDE; do not add an unnecessary reload to that headline. Filesystem
caches are uncontrolled. Paul's remembered two-hour reduction and thirty-minute
binning are context, not measured baselines. Keep profiling separate from timings.

Details and scalar receipts for former 6A-P1/P2/P3 are in
`benchmarks/results/dgs-fused-event-performance.md` and
`benchmarks/results/dgs-user-workflows-node19.md`. The full reference's
`seq-full617-shiver-reference-01/receipt.json` records successful completion.
The initial reference workflow takes 4h25m16s including setup, sequential raw
reduction, six-copy binning and saving; later twelve-copy MDNorm takes 49m56s
before its histogram save. The six-hour process total also includes independent
one/six/twelve cached-MDE rebins, reopen operations and untimed diagnostics, so
it is not the initial saved-dataset headline.

### 6A1 — Validate optional numerical and statistical methods

**Status: Complete; controlled evidence reviewed and follow-ups deferred.**

- [x] Compare optional precision, stable monitor fitting, trajectory-energy and
  symmetry-variance treatments against analytic or simulated truth.
- [x] Compare counting versus continuous estimators, unequal exposure,
  observed versus expected precision weights, source-copy covariance and final-bin
  low-count intervals against analytic expectations and repeated sampling.
- [x] Assess dummy-angle weighting and geometry/calibration-transfer assumptions
  using the recorded physical configuration.
- [x] Include uneven coverage, repeated sources, final cuts and fringes; record
  speed, memory, bias and uncertainty performance for each declared target.
- [x] Document which methods improve an estimate under which assumptions, and
  which only reproduce a different convention.
- [x] Review the evidence and remaining physical-calibration gaps with Paul.

Include the confirmed 12.03 ppm energy-boundary residual and Python/compiled
trajectory classification difference as explicit cases. Agreement with another
convention, smaller errors or smoother plots alone does not establish accuracy.
CORELLI/MACS alternatives remain deferred. Do not change defaults in this checkpoint.

The controlled reports are `benchmarks/results/dgs-estimator-alternatives.md`
and `benchmarks/results/dgs-numerical-alternatives.md`. They validate pooling for
independent counts with known exposure and a common rate, supplied-variance
precision weighting for Gaussian measurements, and primitive covariance for
transformed copies. With unequal copy multiplicity, counting each independent
unweighted source once also lowers common-rate variance in the controlled case;
DGS adoption would need the corresponding physical trajectory-union exposure,
not just a changed variance channel. Per-run energy and angular weighting can improve or worsen
an estimate depending on the actual acquisition model. Real calibration truth,
angular stationarity and complete sample-cut covariance remain acceptance gaps;
the simulations do not certify them. Higher precision improves fidelity to fixed
input values and retains tiny internal fringe segments; this need not improve
physical accuracy at the instrument's calibration tolerance. Stable monitor
variance avoids invalid arithmetic, but does not demonstrate better Ei/T0 in the
controlled pulse examples or validate HYSPEC's separate T0-formula branch.

### 6A2 — Review evidence and choose future defaults with Paul

**Status: Complete; defaults and deferred follow-ups accepted.**

Paul accepted DGS correctness on 2026-10-04 and deferred covariance between
symmetry-related sample bins to a later checkpoint. Review other statistical
choices separately; covariance work requires its own follow-up authorization
and complete propagation through final cuts before any default adoption.

**Accepted estimator policy (2026-10-04):** retain exposure-pooled normalization
for native DGS counting data, including covered zero-count measurements in the
exposure sum. Other measurement models retain estimators matched to their
declared target and assumptions. Pooling estimates an exposure-weighted field
when intensity varies within a bin; an equal-coordinate average is a separate
target requiring a sampling model.

**Accepted monitor policy (2026-10-04):** retain Mantid-compatible peak fitting
as the default; keep the stable derivative-variance treatment optional. The
real-data fallback failures were in the earlier Python translation: negative
roundoff raised an exception, whereas Mantid propagated NaN into its peak-tail
stopping comparison and completed calibration. The corrected compatibility path
calibrates all 617 NiO runs without fallbacks and matches Mantid Ei/T0 to roundoff.
The stable alternative avoids the intermediate NaN, but changes some peak tails;
controlled tests have not demonstrated better calibration accuracy.

**Accepted trajectory-energy policy (2026-10-04):** retain the first
participating run's Ei for normalization trajectories, following Mantid MDNorm;
keep each-run Ei optional. Raw reconstruction continues to use each run's own
resolved Ei/T0. A shared energy override retains precedence. Paul considers the
observed stable-monitor changes too small to matter practically; the accepted
Mantid monitor default remains unchanged.

**Accepted He-3 convention (2026-10-04):** both event-precision modes use Mantid's
tube radius and efficiency formula; no alternative correction is justified by
the evidence. High precision retains float64 weights. Only affected legacy
high-precision raw-event and histogram caches invalidate on recomputation.
Mantid defaults and existing scientific project files remain unchanged.

**Accepted background weighting (2026-10-04):** retain sample-exposure
weighting after background subtraction. Defer an optional covariance-aware
common-intensity estimator to a later statistical checkpoint. That estimator
requires a declared common response and complete shared-background dependencies;
it must not silently replace the exposure-weighted field target.

**Accepted cache migration (2026-10-04):** Paul does not require historical
default preservation. Missing DGS policy fields use current defaults. Reduction
version 8 and histogram policy version 4 invalidate prior DGS caches, including
derived backgrounds, when requested. Explicit choices remain effective. No
scientific project is rewritten by this application update.

**Completed performance follow-up (2026-10-04):** Paul authorized the fused
high-precision HKLE path, zero-count and fractional studies, and GUI audit while
away. The fused path retains three float64 transform stages, ordered accumulation
and shared physical membership. HYSPEC's full 361-run saved-MDE bin/save/reopen
workflow is 1.7–2.0 times faster on node19; warm six/twelve-copy event kernels
are about 3 times faster. Event numerator, variance, counts and masks are exact;
exposure, intensity and uncertainties agree to roundoff across all finite cells.
Evidence and scope are in `benchmarks/results/dgs-high-precision.md`.

**Completed statistical and GUI review (2026-10-04):** controlled zero-count
checks confirm retained exposure and valid unweighted Poisson intervals; weighted
DGS and shared-background intervals still require validated forward models.
A conditional physical-count exposure design is documented for future work.
Fractional assignment improves smooth-field variance but can bias peaks, bridge
unmeasured regions and understate cut uncertainties without cross-bin covariance.
Native DGS remains discrete. The study fixed general fractional endpoint and
streamed-sum masking bugs, with targeted cache invalidation. Supported reduction
controls are visible and scriptable; native DGS controls now display actual pooled
normalization and discrete assignment. Broader contract, interval, likelihood and
smoothing editors remain scripting-only and are listed in permanent documentation.
Evidence is in `benchmarks/results/dgs-zero-count-uncertainty.md` and
`benchmarks/results/dgs-fractional-binning.md`.

These investigations are complete. Implementing weighted DGS intervals,
matched fractional trajectory kernels, covariance propagation and the broader
statistical GUI remains later work requiring review.

**Fractional point-binning follow-up (2026-10-04):** Paul authorized correctness
and performance improvements to existing general fractional assignment, with
neighboring-bin covariance deliberately omitted. The implementation retains
multilinear center weights, consistent numerator and averaging weight, and
squared full coefficients for diagonal variance. An independent scalar oracle
covers 1D–7D, nonuniform/mixed axes, exact centers and physical edges, integrated
axes, streaming, invalid inputs, and serial/dense/sparse accumulation. Deposited
physical exposure survives point rebins through GUI and scripting paths.
Compiled accumulation now accepts explicit edges; redundant integrated-axis
neighbors, unnecessary bound scans and identity stream projections are removed.
General point/histogram numerical caches are invalidated lazily. Native DGS
remains discrete. Performance evidence is recorded in
`benchmarks/results/fractional-point-binning.md`. Zero-count treatment and
cross-bin covariance remain deferred. Code completion does not adopt a native
DGS fractional estimator or rewrite any scientific project.

**Deferred statistical follow-ups:** these require later checkpoints and do not
block acceptance of the intensity estimator.

- [ ] Establish error bars for covered bins with zero counts. Validate the
  uncertainty or interval definition and repeated-sampling coverage for the
  declared count model, including weighted events and backgrounds; distinguish
  positive exposure with zero events from absent coverage.
- [ ] Test whether fractional binning improves estimates of declared bin
  intensities. Use known intensity fields and uneven coverage, gradients,
  peaks and boundaries; compare bias, mean-squared error and uncertainty
  coverage with discrete binning. Apply fractional coefficients consistently
  to signal, exposure and source-dependent uncertainty before considering a
  default change.
- [ ] Add explicit angular averaging of measured dummy backgrounds, as requested
  by Paul on 2026-10-04. Normalize each acquisition using its own exposure and
  geometry, then apply declared angular weights. Retain detector/energy
  dependence, per-angle provenance and shared-source uncertainty. Distinguish
  the mean of measured angles from a continuous angular integral, validate
  against original per-angle acquisitions and preserve the current saved
  projects until that later checkpoint.

- [x] Present the accepted methods, assumptions, limitations and compatibility paths.
- [x] Agree which defaults change and which treatments remain optional.
- [x] Adopt current defaults for absent DGS settings and invalidate old DGS
  caches for lazy recomputation; preserve explicit saved choices.

This decision follows validation and complete dependency propagation for the
selected workflow. Distinguish correctness fixes from adopting a new estimator.

### 6B — Validate migration, finish documentation, and remove this plan

**Status: Pending; after accepted DGS checkpoints and default decisions.**

- [ ] Validate representative existing projects, portable recipe replay, cache
  invalidation, lazy loading and GUI/script equivalence.
- [ ] Confirm performance, peak memory, project size, reopening and reuse after
  the final changes. Keep measured reference times separate from estimates.
- [ ] Finish permanent physics, workflow and limitation documentation; synchronize
  local/ORNL applications and documentation.
- [ ] Review final acceptance with Paul, then delete this plan and navigation links.

#### 6B-S — Resident data and resource safety

Paul authorized this follow-up after the large SEQUOIA project exhausted memory
with several data viewers open. Whole hypercubes remain resident for responsive
switching and sliders; partial-cube viewing and slider caching are excluded.

- [x] Share immutable prepared payloads between viewers and count distinct array
  storage, including models, analysis, retained child views and compressed data.
- [x] Admit managed allocations using expanded archive headers, resolved grids,
  working-buffer estimates and process-wide reservations. Stop over-budget
  operations and offer Resource Manager without automatic eviction or overcommit.
- [x] Add a flat sortable resource table, Shift/Ctrl multi-selection, explicit
  load/unload actions and confirmed cache deletion. Display parents, shared
  owners and viewers below the actions; protect unsaved canonical data.
- [x] Expose decimal-GB budgets, CPU limits and future temporary-storage location
  beside project/system monitoring. Inspect unloaded resources without decoding
  numerical arrays or scanning scientific staging directories.
- [x] Keep large GUI I/O on guarded workers, support cooperative cancellation
  and reserve atomic replacement disk space while preserving the prior archive.
  Preserve standard project compression and lazy reduced-event caches.
- [x] Complete automated release validation.
- [ ] Review interactive behavior with Paul before continuing 6B migration work.

Release 0.117.0 passes 3,604 tests with one optional-backend skip, Ruff,
byte-compilation, diff checks and the strict Sphinx build. Permanent behavior and
scripting documentation is in `resources.md`. Interactive acceptance remains
pending; this checkpoint does not authorize starting 6B migration work.

Follow-up fixes for the first interactive review:

- [x] Preserve the saved state when refreshing a selected fit-history row.
- [x] Save relative source/calibration references and rebase older shared paths
  without decoding or recomputing numerical caches; preserve destination access
  metadata when replacing an existing archive.
- [x] Group scientific staging into lazy, private sessions under `.nfit-work`;
  preserve live, foreign and unmarked legacy folders during cleanup.
- [x] Launch offline Help with the system desktop environment instead of bundled
  library search paths; report opener failures without blocking the GUI.
- [x] Complete automated validation: release 0.117.1 passes 3,678 tests with
  two optional-platform/backend skips, Ruff, byte-compilation, diff checks and
  the strict Sphinx build.
- [x] After Paul closed the older GUI and authorized cleanup, remove its 1,355
  loose reduced-event staging folders (27.45 GB); preserve saved project caches.
- [ ] Review the updated ORNL application with Paul.

### Deferred instrument work

The initial survey covered continuous measurements, reduced CW tables and real
MACS/CORELLI inputs; it did not certify their complete native statistical paths.
Evidence remains in `benchmarks/results/measurement-acceptance.md`.

- **MACS (former 6A-M):** retain count numerator, declared variance, exposure and
  source identity through SPEC/DIFF histograms and final cuts; keep legacy
  weighting reproducible.
- **CORELLI (former 6A-C):** preserve each neutron's reconstruction/copy
  dependencies through final cuts, or require source replay. Changing energy-bin
  centers is not simply summing prior reconstruction hypotheses.
- **Other raw adapters:** DMC, D33, SANS-I, GP-SANS and WAND² files are future
  inputs. Reduced WAND² histogram import is supported; new raw reducers require
  separate feature work. These instruments must not use the DGS reducer.

Resume deferred work only when Paul authorizes it. Future non-DGS acceptance
must exercise the same measurement contracts without assuming DGS statistics.
