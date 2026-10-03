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
  compatibility defaults stay fixed until the 6A2 decision; mathematical
  correctness fixes and performance improvements must state their assumptions.
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
| 6A | Validate instrument families and continuous measurements | Survey complete; adapter follow-ups required |
| 6A-R | Resolve raw-DGS geometry/timing and validate HYSPEC conversion | Core complete; historical subtraction split into 6A-R2 |
| 6A-R2 | Validate directional HYSPEC background reconstruction and uncertainties | Analysis complete; cached final-cut dependencies split into 6A-R3 |
| 6A-R3 | Preserve background source correlations and target through final cuts | Complete within the declared cached-field scope |
| 6A-R3a | Persist bounded replay recipes and expose exact cached-field aggregation | Complete |
| 6A-R3b | Integrate parent composites and asynchronous GUI cuts/exports | Complete; checkpoint review |
| 6A-M | Preserve MACS count/exposure targets through final profiles | Deferred by Paul |
| 6A-C | Supply CORELLI reconstruction/copy covariance or require replay | Deferred by Paul |
| 6A1 | Validate optional numerical and statistical treatments | DGS investigation in progress; other families deferred |
| 6A-P | Profile current HYSPEC/SEQUOIA workflows and adopt verified major speedups | In progress |
| 6A-P1 | Fuse DGS projection and ordered accumulation with numerical parity | Complete; includes exact-text mask parsing reuse |
| 6A-P2 | Measure current raw reduction and SEQUOIA workflows on ORNL | Native full workflow and matched pilot complete; full Shiver reference pending |
| 6A-P3 | Reuse scalar inspection and geometry arithmetic; parallel snapshot reads | Complete; full SEQUOIA and both HYSPEC bank checks passed |
| 6A-P4 | Correct HYSPEC run-angle averaging and isolate reference boundary differences | Pending scientific checkpoint review |
| 6A2 | Review and adopt future defaults | Pending |
| 6B | Migrate projects, verify performance, and remove this plan | Pending |

Statuses are **Pending**, **Pending review**, **In progress**, **Complete**,
**Deferred**, or **Needs subdivision**.
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

## 6. Cross-instrument acceptance and cleanup

### 6A — Representative measurements

Validate SEQUOIA and HYSPEC DGS, CORELLI reconstruction, MACS measurements,
continuous-wave diffraction, and synthetic densely sampled susceptibility versus
temperature. Include uneven coverage, repeated measurements, independent and
shared backgrounds, fractional/discrete binning, and symmetry. Match Mantid where
it implements the same model; analytic and repeated-sampling references decide
correctness when conventions differ.

**Survey checkpoint (0.114.1):** analytic and repeated-acquisition tests validate
continuous common-response and coordinate targets, uneven support, repeated
temperatures, shared interpolation nodes, and independent/shared backgrounds
through final cuts. Synthetic reduced CW tables exercise public import,
coordinate preparation, binning and final cuts. Real MACS SPEC/DIFF, HYSPEC MDE,
CORELLI raw data and SEQUOIA references are examined read-only; the bounded
scripts and aggregate receipts are documented in
`benchmarks/results/measurement-acceptance.md`.

This survey does not establish complete cross-instrument acceptance. Native
MACS histograms lose their count/exposure model before later profiles; native
CORELLI omits reconstruction/copy covariance; raw HYSPEC needs run-log-dependent
instrument geometry and correct evaluation of its T0 formula. These are adapter
implementation gaps, distinct from optional estimator/default adoption. **6A needs the follow-ups below before it
can be marked complete.** Existing compatibility defaults and original science
projects remain unchanged. The viewer now exposes CORELLI's recorded covariance
and normalization limitations through its metadata-only provenance report.

The supplied DMC, D33, SANS-I, GP-SANS and raw WAND² files establish useful
future adapter inputs. They do not have native reducers in nfit. Reduced WAND²
histogram import is supported; MACS DIFF supplies the real CW diffraction
example for this survey. Adding the other raw adapters is separate feature work.

#### 6A-R — Dynamic raw-DGS geometry and timing

Paul authorized this checkpoint with HYSPEC remaining in the DGS workflow.
Use HYSPEC's saved reduction recipes and fresh Mantid reference jobs to validate
the same acquisition and settings; reduced MDE-only checks are insufficient.

Resolve instrument-definition locations/rotations from each run's referenced
logs before conversion. Geometry reuse must compare complete resolved geometry,
including logged moderator/detector positions, rather than static XML alone.
Qualify raw-DGS detection by measurement type: WAND², CORELLI, MACS and SANS
must retain their separate reduction paths even when they store event banks.
Validate synthetic same-XML/different-log and mixed-instrument cases, then the
available HYSPEC raw runs against their saved reduction with identical
Ei/T0, goniometer, energy window, filters and corrections. Unsupported dynamic
definitions must fail explicitly. Correct the HYSPEC T0 formula's exponent
syntax and make failed automatic timing evaluation visible rather than silently
returning zero. Preserve SEQUOIA parity.

Implemented run-log geometry resolution, corrected HYSPEC timing, raw TOF
selection and relative Tank rotation. Reduction controls, script replay and
resolved provenance cover the HYSPEC window and optional offset. Geometry reuse
checks complete resolved definitions. Saved MDE detector masks now constrain
HKLE, powder and background normalization trajectories; their omission had
allowed exposure from masked tube tips. Reduced-event version 6 and DGS/MDE
numerical-policy version 3 invalidate affected older caches on demand.

Fresh references use the installed Shiver data-to-MDE recipe with UB, rather
than HYSPEC histogram autoreduction. Runs 506277, 506278 and 505555 cover the
70° and 34° detector settings. Common energy, QSample, signal and variance
event fields agree bit for bit. Loading the original MDE reproduces its
histogram numerator, variance and event counts exactly. Native raw reduction
also agrees exactly when Mantid uses the requested physical energy bounds.
Normalization differs by at most $2.15\times10^{-11}$ relative and final-cut
signal/uncertainty by at most $2.28\times10^{-13}$ relative.

The unmodified Shiver recipe excludes 1–2 events per run when float32 energies
round outside its observed-extrema MD box. Widening only Mantid's energy box
restores precisely the native events. This is documented separately from
reconstruction accuracy; a strict loss-emulation option remains a review
decision. Original scientific projects and inputs remain unchanged.

The three-run SEQUOIA powder regression preserves signal, variance and counts
after accounting for four already documented Mantid conversion-boundary
omissions; fringe bins agree without that adjustment. Exposure differs by at
most $8.21\times10^{-11}$ relative. Five event energies differ by at most
$3.73\times10^{-9}$ meV; QLab, weights and variances agree exactly. This bounded
regression does not replace the earlier full HKLE validation.

Detailed receipts and the unresolved saved-histogram comparison are in
`benchmarks/results/hyspec-dgs-parity.md`, its aggregate JSON, and
`measurement-acceptance-sequoia-raw-regression.json`. No scientific original was
saved or rebuilt during this checkpoint.

Validation: 2,895 tests passed; one CuPy availability test skipped. Ruff,
byte-compilation, diff checks and warning-as-error Sphinx build passed. Manual
engine comparisons remain outside pytest and production code.

#### 6A-R2 — Directional HYSPEC background reconstruction

Paul reports that subtraction in `HYSPEC_all.nfit` removed the background well,
while the historical Mantid subtraction looked worse. Use its existing
`Workspace1/Group1` collections as the reference configuration: all four enabled
background links use `measured_events`, despite their legacy names containing
“powder”. The separate `powder averages` branch uses center projection. Preserve
both branches and the original project during validation.

The physical target is a background fixed in laboratory coordinates, replayed
as if acquired at every sample angle. Retain each measured event's full lab-frame
momentum and energy, detector acceptance, and bank geometry. Use sample-angle
exposure weights, matching detector masks, energy coverage, UB, output edges and
symmetry. A radial background cannot recover directional structure lost by
averaging over equal momentum modulus and energy. Check the assumption that the
background is stable over sample rotation and that each source matches the
sample temperature and bank configuration.

Validate signal, exposure and uncertainty against explicit synthetic acquisitions
with directional contrast at equal momentum modulus and energy, unequal sample
charges and repeated angles. Copies of the same measured background event must
retain their common source identity. Extend checks through final integrated cuts
and low-coverage fringes; same-voxel coefficient merging alone does not certify
cross-voxel covariance. The current replay does not store that covariance or a
certified additive count/variance payload. Assess these gaps before changing the
scientific contract.

Use the four historical histograms to attribute discrepancies, without making
visual agreement with their subtraction an acceptance criterion. The available
`histograms/export_metallix.py` passes MDE backgrounds directly to `MDNorm`; it
does not establish that their upstream construction used a radial background.
Audit that construction, source membership, angle weighting and variance
propagation before attributing the residuals. Keep validated optional treatments
and default adoption in 6A1/6A2; observed-extrema compatibility remains a separate
review decision.

**Completion record (0.114.4):** complete real backgrounds for all four sample
configurations agree with an independent event/covariance oracle at nine selected
angles, including fringes and the lowest-exposure decile. Direct final-grid
replay matches the full covariance reference. Unequal charges, source scales,
repeated angles and both replay backends pass analytic tests. First-run Ei is
now resolved before singleton trajectories and source-mask partitioning;
background/composite caches recompute under the corrected convention.

Matched historical membership exposes a different normalization model: Mantid
replays background lab directions but uses sample detector geometry in its
background denominator. The 0.0261° / 0.00137° bank differences strongly affect
tiny trajectory intersections. Substituting that denominator explains most of
the largest oversubtraction (−0.100 native versus −6.233 saved; −6.294 with the
historical geometry). Signal RMS discrepancy falls by about 74%, but complete
historical parity is not established. Keep the successful directional treatment
and review explicit compatibility/calibration-transfer alternatives in 6A1/6A2.
See `benchmarks/results/hyspec-background-analysis.md` and its two scalar receipts.
Original projects and sources remain unchanged. Full suite: 2,905 passed, one
unavailable-CuPy skip; Ruff, compilation, diff checks and Sphinx passed.

#### 6A-R3 — Background final-cut correlations

Cached measured-background voxels lack cross-voxel source dependencies and a
certified additive payload. Preserve source correlations and an explicit
statistical target through profiles, cuts and exports, or use automatic replay
on the final grid. Direct final-grid scripting replay is already correct for the
recorded within-bin model. A cached-cut diagonal approximation must be visible
and must not claim source-aware uncertainty. Analytic variance is 2 where
separate diagonal pooling gives 1.25; the real sparse 50 K / 34° energy-profile
uncertainty is up to 2.12 times the diagonal result. Do not apply a fixed empirical
correction. The following split implements the explicit API before its asynchronous
GUI and hierarchical-composite consumers.

**R3a, authorized 2026-10-03:** Newly computed directional backgrounds retain
bounded source/transform/mask recipes, source-event digests, and lazy exposure
channels. An explicit scripting operation aggregates the original cached cells
with sample exposure after subtraction, combining each background primitive's
signed coefficients before squaring. Preserve the field target
`sum(N_sample * (I_sample - B)) / sum(N_sample)`; separately pooling background
and sample components is a distinct target. Ordinary previews remain responsive
and identify diagonal uncertainty. Validate analytic truth, masked/low-exposure
regions, serialization, stale sources and bounded repeated-query caching.

**R3a acceptance:** `replay_cached_background_profile` and compact lazy recipes
are implemented. Complete 50 K / 34° dummy-source validation with nine sample
angles agrees with an independent original-voxel oracle to 1.63 × 10⁻¹⁴
relative variance at the peak. A separate projected coverage-edge region agrees
to 6.20 × 10⁻¹⁶; the lowest-exposure decile also passes. Cold queries take
2.44–2.64 s and repeated identical queries 10–13 ms. Combined sample/background
sigma exceeds the diagonal approximation by up to 10.5% overall, 13.5% in the
lowest-exposure decile, and 5.2% at projected edges. These ratios concern the
sample-exposure-weighted difference, distinct from R2's background-only target.
Original files remain unchanged. The source/recipe receipt is
`benchmarks/results/hyspec-cached-background-profiles.json`.

Only background covariance within the requested final profile bins is replayed.
Sample uncertainty retains its recorded policy; cross-profile-bin covariance
and unrecorded calibration/exposure dependencies remain unavailable. Changed
payloads invalidate recipes even when transforms copy their metadata. Oversized
optional recipes preserve ordinary histograms with an explicit replay-unavailable
reason. Unsupported derived background operands require complete source replay.
Final validation includes the full 2,948-test suite and focused post-review gates.

**Geometry assessment — complete, accepted 2026-10-03:** Paul confirmed that
sample and dummy bank angles differ because the bank was repositioned and its
encoder reaches each target within the instrument tolerance. Retain sample
geometry for sample events and normalization, and dummy geometry for dummy
events and normalization. Replay the orientation-averaged dummy at each sample
rotation. These small positioning differences do not justify detector remapping
or mixing source-event geometry with another acquisition's normalization.

**R3b — complete 2026-10-03:** Aligned exposure-weighted hierarchical bank
composites preserve each component's exposure, signed scale and shared background
source coefficients. Native-grid regular and rotated viewer box cuts replay
asynchronously after a debounced selection; obsolete results are discarded and
pending exports are disabled. Completed profile exports and editable figure
scripts use the public original-grid selection API. Unsupported treatments retain
explicit diagonal previews and reasons. Construction and project loading do not
read event sources.

**R3b acceptance:** A nested public dataset-group composite of both 50 K HYSPEC
banks reads the complete 4,744,544-event and 4,624,140-event dummy sources, with
nine sample rotations per bank. Eight regular/12°-rotated x/y profiles across
support and projected coverage fringes agree with an independent primitive
coefficient oracle: variance discrepancies are at most 2.073 × 10⁻¹⁴ relative
to the peak, and 1.089 × 10⁻¹⁵ at projected fringes. Exposure and masks agree
literally. Both banks contribute event counts and variance to unmasked support
bins; only the 34° bank contributes to the selected outer union fringe.
Source files and scientific module hashes remain unchanged during validation.
The bounded grid, acceptance masks, component contributions and scalar receipt
are recorded in `benchmarks/results/hyspec-bank-background-profiles.json`. The
full local suite passes with 3,024 tests and one unavailable CuPy backend skip;
GUI/script round trips, export guards, metadata stacking and public API regression
checks pass. Ruff, compilation, diff checks and the strict Sphinx build pass.

The replay target is the sample-exposure-weighted cached subtracted field, so
both mean and uncertainty can differ from a precision-weighted preview. Exact
claims concern represented background covariance within each final profile bin.
Sample uncertainty retains its recorded policy; covariance between final bins
and unrecorded source/calibration dependencies remain unavailable. Regridded,
metadata-stacked, inverse-variance and unsupported dependent composites require
full source replay. Box sums remain labeled diagonal. These remaining treatments
are tracked in `docs/planned_features.md` and are outside this checkpoint.

Paul clarified that the HYSPEC background is a dummy sample: the same mount,
glue and aluminum without crystals. The coarse 180° sweep intentionally averages
orientation-dependent effects. Retain detector direction and energy dependence
while averaging dummy orientations. The current merge is charge weighted; the
50 K / 34° background differs modestly from equal-angle weighting. Physical
assessment and controlled geometry/calibration tests are recorded in
`benchmarks/results/hyspec-background-physical-assumptions.md`. New scientific
defaults remain gated by 6A2.

#### 6A-M — Native MACS statistical payloads

Retain original numerator, its explicitly declared variance, exposure and source
identity through the native point histogram and every final-cut path. Present
the target as an acquisition choice rather than inferring it from the instrument
name. Validate SPEC and DIFF against original counts, covered zeros and exposure;
direct/staged/profile estimates of the same target must agree. Keep the legacy
error-floor/weighting treatment reproducible. Any new default and its migration
remain a separate decision in 6A2.

#### 6A-C — CORELLI reconstruction dependencies

Accumulate coefficients of each physical neutron before squaring when
fractional/symmetry copies meet. Represent dependencies across reconstructed
energy channels or require source replay for final scientific cuts that need
them. A wider energy bin reconstructs a new channel at a different center; it
does not by itself replay a linear sum of previous energy hypotheses. Validate
signed cancellation, positive covariance, overlapping copies, final cuts and
fringes against independent primitive references. Keep charge/duty and pointwise
calibration normalization distinct from full trajectory coverage.

**Current authorization:** 6A-R core and 6A-R2 analysis are complete. Paul
authorized DGS background follow-up, assessment of alternatives, and performance
work on 2026-10-03. CORELLI and MACS are deferred. R3a supplies exact scientific
aggregation without synchronous event replay on every GUI interaction; R3b
integrates asynchronous consumers and parent-composite propagation. Review each
coherent completion with Paul. Strict extrema compatibility and future default
changes remain explicit decisions in 6A2.

### 6A1 — Validate alternatives

Compare optional high precision, stable monitor fitting, per-run trajectory Ei,
and covariance treatments against analytic references, known calibration or
simulated truth, and the representative instrument families. Prioritize fringes,
coverage boundaries, repeated sources, and final cuts. Compare speed and memory
as well as signal and uncertainty. Agreement with a different convention alone,
smaller errors, or a smoother image does not establish greater accuracy.

The confirmed 12.03 ppm energy-boundary residual and Python/compiled trajectory
classification difference are explicit validation cases. Numerical performance
candidates must preserve the chosen treatment; their synthetic gains require
real-workload confirmation before adoption.

### 6A2 — Choose defaults

Review the evidence with Paul before changing any scientific default. State the
measurement assumptions behind each choice and retain a reproducible compatibility
path. Preserve previously resolved settings through explicit project migration;
changing a default must not silently reinterpret projects with absent legacy fields.
This gate follows complete dependency propagation and cross-instrument validation.

### 6A-P — DGS performance

**Authorized 2026-10-03; in progress:** With experiment mounts restored on
node19, run realistic current-build SEQUOIA and HYSPEC workflows on the same
hardware. Use the installed ordinary sequential Shiver reduction, allowing its
normal internal threading. Include source loading, reduction, event-cache or MDE
persistence, native histogramming, saved result reopening, and subsequent rebins.
Separate first saved-dataset time from later reuse workflows; do not force an
extra MDE reload into a resident Shiver reduction-to-histogram headline.
The remembered two-hour reduction and thirty-minute binning are informal context,
not measured benchmark inputs. Begin with a bounded matched pilot, then measure
the full 617-run NiO workflow and representative HYSPEC configurations. Preserve
original projects and keep all scientific artifacts in IPTS benchmark folders.
Profile separately from ordinary timings. Validate every C/V/count/exposure cell
and low-coverage cuts before attributing improvements to a candidate. Report
rounding/boundary differences explicitly. Commit coherent validated changes,
update both installations and review this checkpoint before proceeding.

Profile matched inputs and policies before diagnosing a regression. Separate
event reconstruction, event projection/accumulation, trajectory normalization,
background replay, archive I/O, and first-call compilation. The fused uniform-grid
Mantid projector preserves literal C/V/count/mask results. Complete local HYSPEC
histogramming of 49.4 million events and 361 runs improved from 16.485 to 3.913 s
with six symmetry copies, or 6.001 to 2.899 s without symmetry, on the same
bounded grid and eight-worker budget. An exact-text bounded mask parse cache
removes repeated parsing while retaining all source and geometry checks.
Exposure is computed by the unchanged
algorithm; differences are at its existing parallel addition precision.

`benchmarks/results/dgs-fused-event-performance.md` records the scope, scalar
receipts and fallback coverage. These timings exclude raw reconstruction and
archive persistence. Current raw/SEQUOIA and matched Mantid measurements use
restored experiment mounts on node19; older benchmarks cannot establish a
current regression. Remaining event reads/filtering, raw reconstruction, and
full-volume normalization require fresh isolated cluster profiles before adding
more cache machinery. Do not adopt speculative micro-optimizations.

**P2/P3 current evidence:** On node19, the matched 24-run saved-dataset workflow
takes 51.066 s in nfit 0.116.0 versus 565.660 s in one ordinary Shiver job.
C/V/counts and coverage support agree exactly for six/twelve copies. The
low-coverage cut agrees near 10⁻¹³; tiny whole-cube exposure differences remain
documented separately. Full 617-run native candidate acceptance preserves
literal C/V/counts/edges and every-cell exposure within 10⁻¹² relative. Initial
saved-dataset time improves 567.152 → 503.554 s, and saved histogram access
39.706 → 7.926 s. Subsequent twelve-copy binning remains approximately 258–263 s;
its 172–173 s trajectory interval remains the main cost.

The validated candidates cache only successfully inspected scalar metadata and
the geometry's exact distances/directions, with source identity, monitor policy,
mask and complete geometry checks. Parallel cached-array reads retain the
validated original descriptor across atomic saves. No numerical convention,
saved schema or compression choice changes. The fused projector dispatches for
the full SEQUOIA grid; rounded output edges do not disable it. Metadata threading
is not adopted: its local smoke was slower and HDF5/Python serialization limits
the plausible gain. Full-suite acceptance: 3113 passed, one optional GPU skip.

Both HYSPEC detector-bank twelve-run pilots pass the same original-cell
six/twelve-copy gate, with identical C/V/counts/coverage and roundoff-only N
differences. Their roughly nine-second initial workflows show no material total
speedup; small pilots are dominated by setup and first-call compilation. Their
matched Shiver pilot receipts and the full SEQUOIA reference remain separate
measurements, rather than extrapolated full-job claims.

The ordinary Shiver HYSPEC pilots finish in 47.220/49.820 s, but reference
parity is incomplete: 34° has count migrations and approximately 3×10⁻⁴
relative-L2 exposure differences, and 70° twelve-copy exposure differs in two
support cells. These differences also exist in the unchanged native baseline.
Run 505784 exposes one concrete cause: native omega uses the arithmetic
48.5016271525° average, while Mantid's time-weighted average is 48.5043924634°.
The raw timestamps reproduce Mantid's value. Other 23 pilot rotations and Ei,
T0 and UB agree. P4 must reproduce duration/accepted-time averaging through a
public reduction convention, invalidate affected caches, isolate that run's
histogram change, and then classify remaining clipping/support differences.
Do not present the performance result as completed HYSPEC cross-engine parity.

`benchmarks/results/dgs-user-workflows-node19.md` records complete scope, actual
saves/reopens, scalar evidence locations and unresolved full-reference work.
Launch the full ordinary Shiver reduction and one/six/twelve-copy bins last,
with timings/progress/resource logs inside the IPTS folder. Keep 6A-P open
until its full-job comparison and HYSPEC scope have been reviewed; do not proceed
to selecting alternative defaults.

### 6B — Migration, performance, and completion

Validate existing projects and a documented migration path. Measure native reduction,
first binning, cached-event rebinning, cache size, peak memory, and reopen latency.
Report measured Mantid/Shiver job timings separately from estimated conventional
single-job timings, with the same data/settings and explicit uncertainty assumptions.
Geometry and calibration reuse must verify equivalence for mixed instrument inputs.

Once all checkpoints are accepted, replace temporary findings with permanent
physics/workflow documentation and delete this plan and its navigation references.
