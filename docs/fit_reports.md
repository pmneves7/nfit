# Fit reports

Every stored fit result can be exported as a publication-grade LaTeX or PDF
report — a complete record of what was fit, comprehensive enough to serve as
the methods/supplementary section of a peer-reviewed paper.

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
   cross-section convention (polarization factor, form factor, Bose factor).
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
   closure internals).
6. **Methods** and a bibliography citing only the references relevant to the
   terms actually used.

The document is a plain `article` and needs only the `amsmath`, `amssymb`,
`booktabs`, `longtable`, and `geometry` packages — present in any standard
TeX distribution, including minimal ones.

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
