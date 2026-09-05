# Fit reports

Every stored fit result can be exported as a LaTeX or PDF report. The report
summarizes the saved fit configuration, statistics, model, and conventions; it
is a starting point for methods documentation, not a substitute for scientific
review.

## Exporting

Right-click a fit result in the project tree and choose **Export report…**,
or use the **Export report…** button in the fit result's details pane. The
save dialog offers two formats:

- **PDF report (`.pdf`)** — the LaTeX source is compiled with a locally
  installed TeX engine and the finished PDF opens in your default viewer.
- **LaTeX source (`.tex`)** — the standalone document itself, ready to
  compile, edit, or paste into a manuscript.

Everything in the report comes from the fit entry's stored snapshot,
statistics, and diagnostics — no recomputation — so reports can be exported
from any historical fit result, including after reloading a project.

## Contents

1. **Fit summary** — status, optimizer, timestamps, total and per-dataset
   $\chi^2$ / reduced $\chi^2$, the number of points per dataset, dataset
   temperatures, applied fields, and fit weights.
2. **Crystal structure** — space group, lattice parameters, and the magnetic
   sites with their form-factor ions (one block per `heisenberg_rpa`
   component).
3. **Model Hamiltonian** — assembled from the active terms and documented
   term by term: the Heisenberg exchange orbit table (distance, multiplicity,
   representative bond, fitted $J \pm \sigma$), anisotropic-exchange basis
   matrices with their symmetric/Dzyaloshinskii–Moriya classification and
   fitted coefficients, single-ion anisotropy tensors, the dipole–dipole
   strength compared to the physical $(\mu_0/4\pi)(g\mu_B)^2$ value, the
   Zeeman term with per-dataset fields, the RPA dynamic-response equations
   with the fitted $\chi_0$/$\Gamma_0$ (per dataset when shared that way),
   the self-consistency closure with its defining equation, and the
   cross-section convention (the fluctuation-dissipation $1/\pi$, the
   one-component $P=2$ polarization factor, $(g/2)^2$, form factor, Bose
   factor, and the separate $\mu_0$ conversion to SI $M/H$).
   A tight-binding component contributes its basis count, model digest, and
   readable onsite and hopping terms with canonical values, uncertainties,
   fit state, sharing, and bounds.
4. **Fitted parameters** — every parameter with value, standard error,
   varied/fixed status, sharing scope, and limits; posterior median and
   16/84 percentiles when emcee sampling ran. When the fit result's posterior
   display choices are enabled, the report instead uses asymmetric emcee
   16/84 uncertainties and/or the selected highest-log-probability emcee sample
   as its displayed best fit. Selecting the best sample also applies it to the
   live selected model; the saved least-squares result remains unchanged and is
   restored when the selection is cleared.
5. **Physics diagnostics** — the per-dataset derived quantities
   ($\mu_{\text{eff}}^2$, static susceptibilities, distance to instability,
   closure internals). Electronic-RPA entries identify their sampled
   zero-energy stability margin and relative pole distance; this configured
   wavevector probe is not labelled as a global Brillouin-zone bound.
6. **Methods** and a bibliography citing only the references relevant to the
   terms actually used.

In the report notation, $J\pm\sigma$ is an exchange energy with one-sigma
standard error in meV; this $\sigma$ is not a cross section.
$\chi^2$ and reduced $\chi^2$ are defined under
[Fit constraints](fit_constraints.md#reduced-chi-squared).
$\mu_{\rm eff}^2$ is a squared effective magnetic moment; its conversion from
the dimensionless integrated spin amplitude uses $(g\mu_B)^2$ and the stated
site/molar normalization. The remaining Hamiltonian and cross-section symbols
are defined in [Physics conventions](physics_conventions.md) and the linked
model pages.

The document is a plain `article` and needs only the `amsmath`, `amssymb`,
`booktabs`, `longtable`, and `geometry` packages — present in any standard
TeX distribution, including minimal ones.

### Reported precision

Human-facing parameter values in the report use uncertainty-aware rounding.
Each symmetric standard error, or each side of an asymmetric posterior error,
is rounded to two significant figures. The associated parameter value is
rounded to the decimal place supported by that uncertainty; for asymmetric
errors, the larger-sided uncertainty sets the value's displayed precision.
Posterior median and 16/84 percentiles use the same convention. Values without
an uncertainty retain the general six-significant-digit display.

This formatting changes only the generated report. Project files, fit result
metadata, scripts, and other machine-readable exports retain the stored
floating-point values. Least-squares standard errors are local covariance
estimates conditional on the model, supplied data uncertainties, and
user-selected dataset weights. Reports state whether the covariance trusts
absolute data uncertainties or estimates one global scale from the residuals.
If weights encode importance rather than statistical precision, reduced
chi-squared is not statistically calibrated.
Residual scaling can reflect underestimated statistical error, systematics,
correlations, outliers, or model mismatch, but it cannot distinguish them.
Posterior intervals are likewise conditional on the model, weights, and priors.

## TeX engine requirement

PDF compilation uses whichever of `pdflatex`, `tectonic`, `xelatex`, or
`lualatex` is found on your `PATH` (in that order). If none is installed the
export shows the install options and still writes the `.tex` next to the
requested PDF path, so nothing is lost:

- macOS: `brew install --cask basictex` (≈100 MB) or the full MacTeX.
- Conda environments: `conda install -c conda-forge tectonic` (a small,
  self-contained engine that fetches packages on demand).
- Linux: `apt install texlive-latex-recommended` or your distribution's
  equivalent.

## Scripting

The renderer is a pure function, usable without the GUI:

```python
from nfit.report import render_fit_report_latex, compile_latex_pdf

tex = render_fit_report_latex(fit_entry, group_name="MyGroup")
compile_latex_pdf(tex, "report.pdf")
```

`fit_entry` is any stored `FitTimelineEntry` (e.g. from
`group.fits[...]children` after `load_project`). Old fit entries recorded by
earlier nfit versions render with `--` placeholders for statistics that were
not yet persisted.
