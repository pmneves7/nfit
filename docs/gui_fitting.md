# Models and fitting

This page describes the project-explorer fitting workflow. Model equations are
documented in [Spin-fluctuation models](spin_fluctuation_models.md); the
underlying objects and script API are described in
[Modeling and fitting pipeline](modeling_pipeline.md).

## Model components

A workspace model is the sum of its enabled components. Add components from the
**Models** branch, then set each parameter's value, fitted/fixed state, bounds,
and sharing:

- **Global** uses one value across the fit.
- **Per dataset** fits an independent value for each dataset.
- **Grouped** shares values within selected dataset groups.

Dataset fit weights control statistical influence. A zero-weight dataset is
excluded from optimization but evaluated once afterward for visualization.
Disabling a dataset or dataset group has the same fitting behavior: its model
is not evaluated during optimizer or sampler iterations, it contributes no
chi-squared, and it is omitted from fit reports. nfit evaluates it once after a
fit so the fitted model can still be viewed. **Use best sample** also evaluates
disabled data with the selected posterior parameters.

The dataset **Scale** is an experimental calibration, not a model parameter.
Keep it fixed at 1 for normalized data. For unnormalized data, **Fit scale**
uses the displayed value as the initial guess. On a nested dataset group,
**Share fitted scale** fits one common calibration for all descendants;
unchecking it leaves their scales fitted independently. The fitted value is
written back to every dataset that shares it.

Fit weights are useful when datasets with very different point counts should
have comparable influence. They are user-selected importance weights, not
additional measurements. Parameter uncertainties are therefore conditional on
those weights and are not necessarily repeated-experiment confidence intervals.
When weights represent importance rather than statistical precision, reduced
chi-squared is not a calibrated goodness-of-fit statistic.

The model editor marks fitted values near a finite bound in red. Treat this as a
diagnostic that the optimum may lie outside the allowed interval.

### Constraints

The workspace constraint table defines equality and inequality relationships
between qualified global parameters. Exact relationships remove the dependent
parameter from the optimizer; inequalities introduce a nonnegative offset and
do not reduce the number of fitted parameters.

Use **Check constraints** before fitting. See
[Fit constraints](fit_constraints.md) for expression syntax, validation, and
degrees-of-freedom accounting.

### Heisenberg crystal editor

The Heisenberg RPA component includes:

- lattice parameters, space group, CIF import, and group-crystal import;
- magnetic atomic sites and ion form factors; and
- symmetry-generated exchange-bond orbits.

Generating orbits creates one exchange parameter per distinct orbit while
preserving values for labels that remain present. Crystal and bond state is
ordinary model configuration and is saved in the project.

## Fit timeline

The **Fits** tree preserves the scientific state associated with each result.
Selecting a state restores its datasets, masks, models, parameter settings,
constraints, weights, and scales.

Running from the latest state appends a result. Running from an earlier result,
or enabling **Branch timeline**, creates a nested branch so alternative fits do
not overwrite one another. Editing the latest result creates a sibling
**Current state**; editing an earlier result creates a branch.

A failed result is retained for diagnosis while its editable scientific state
remains available. **Clear history** keeps the active state as a new initial
state and removes the older result tree after confirmation.

## Fit pipeline

The stages are opt-in:

1. **Least squares** always runs.
2. **Differential evolution initialization** optionally searches bounded
   parameter space before least squares. Every varied parameter needs finite
   bounds.
3. **Sample posterior with emcee** optionally estimates posterior intervals and
   correlations after least squares.

The loss function and robust-loss scale are explained in
[Modeling and fitting pipeline](modeling_pipeline.md#simultaneous-least-squares).
Use ordinary `linear` loss for a final chi-squared fit when the uncertainty
model is trusted. Robust losses are useful for diagnosis, but should not replace
appropriate masks or a better physical model.

**Parameter uncertainty** controls the local least-squares covariance:

- **Use absolute data uncertainties** is the default. It trusts the supplied
  one-sigma uncertainties and uses $(J^\mathsf{T}J)^{-1}$ without rescaling.
- **Estimate scale from residuals** multiplies the covariance by
  $\chi_\nu^2$. This treats the residual scatter as an estimate of one missing
  global noise scale. It may absorb underestimated statistical uncertainty,
  systematics, correlations, outliers, and model mismatch, but cannot identify
  or model any of them separately. It can also reduce reported uncertainties
  when $\chi_\nu^2<1$.

This choice affects covariance-derived standard errors, not stored data
uncertainties or posterior intervals.

Long operations run in a background worker and report the active stage,
evaluation count, cost, current parameters, and time per step. **Terminate**
requests a stop at the next progress update. If least squares is still running,
nfit stores the lowest-objective parameter set evaluated so far as a cancelled
fit and evaluates its model and residual channels. It does not report local
covariance errors for this unconverged result. A completed least-squares result
is not discarded if posterior sampling is terminated later.

Parallel worker controls apply to differential evolution and emcee objective
evaluations. `1` is serial; `-1` selects a conservative automatic CPU count.
See [Performance notes](performance.md) before increasing worker counts on
large problems.

## Results

Each fit result stores:

- best-fit parameters and covariance-derived standard errors;
- chi-squared, reduced chi-squared, and per-dataset contributions;
- covariance or correlation information;
- model and residual channels;
- optimizer and posterior settings; and
- the complete scientific-state snapshot.

Parameters that end at a configured bound are named in the fit status and
shown in red. Stored model and residual channels can be opened in the data
viewer while their coordinates remain compatible with the current dataset
view.

Fit reports and result tables round displayed uncertainties for readability;
the project and generated scripts retain full floating-point values. See
[Fit reports](fit_reports.md) for export details.

## Posterior sampling

The **Posterior** panel can:

- run emcee from a least-squares result;
- replace an existing stored chain;
- append steps from its final walker positions; or
- reinterpret the stored chain with different burn-in and thinning.

**Use emcee uncertainties** switches result tables, diagnostics, and reports to
the stored 16--84% intervals and posterior correlations. **Use best sample**
temporarily evaluates the live model at the highest-log-probability stored
sample. Neither option changes the saved least-squares snapshot.

If a terminated emcee run has produced samples, nfit keeps the partial raw
chain so it can be inspected or extended.

## Diagnostics

**Fit diagnostics** displays covariance or correlation heatmaps. With emcee
samples it also provides walker traces and corner-style posterior plots.
Editable compact labels affect only these plots.

Trace plots mark the selected burn-in. Corner plots show distributions,
best-fit references, and symmetric or asymmetric reported intervals. Inspect
acceptance fractions, mixing, correlations, residuals, and sensitivity to
bounds before interpreting posterior intervals.

## Scripts

**Copy fit script** and **Save fit script** generate GUI-free Python for the
workspace's live fit state. The script reloads and prepares source datasets,
rebuilds the current model and constraints, and runs the active optimizer
configuration once. It does not replay fit-history branches.
