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
- Check speed, peak memory, lazy loading, and reuse when changing numerical payloads
  or caching. Reject small speed gains that require disproportionate complexity.
- Remove this page and its navigation links after all checkpoints are accepted.
  Keep the resulting scientific conventions and workflows in permanent docs.

## Checkpoints

| Checkpoint | Scope | Status |
| --- | --- | --- |
| 1A | Reproducible uncertainty diagnostic baseline | Complete |
| 1B | Trace the unsubtracted NiO uncertainty example through histogram, slice, and cut | Pending |
| 1C | Separate accumulated event variance from low-count confidence intervals | Pending |
| 1D | Validate DGS reference statistics and covariance boundaries | Pending |
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

Statuses are **Pending**, **In progress**, **Complete**, or **Needs subdivision**.
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
split into 1A–1D; 1B has not started and requires Paul's go-ahead.

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

### 1D — DGS reference and dependency validation

Test native reduction and shared-MDE binning against Mantid at the statistics
level, including incident energy, time zero, beam deadtime/pause/bad-pulse removal,
charge accounting, vanadium, mask, and trajectory normalization. Read the SEQUOIA
recipes in `shared/MDE_data_reduction_backup` as reference examples, rather than
as a requirement to preserve a wrong estimator.

Validate weighted events and known correlated copies separately. Repeated events,
symmetry copies, fractional assignment, shared calibration, and backgrounds must
not acquire artificial independence. Document which dependencies are represented
and which remain outside the current model. Add source/per-bin sparse or low-rank
representations where justified; avoid a dense covariance matrix for a large 4D
histogram. Split implementation from validation if needed.

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
Keep compressed, lazy, dataset-owned reduced-event caches; adding or removing one
run must not require reducing unrelated runs again. Calibration edits and geometry
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
