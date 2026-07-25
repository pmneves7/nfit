# nfit documentation

`nfit` analyzes reduced magnetic-scattering and related experimental data
in physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy transfer. The
name stands for neutron fitting and N-dimensional fitting. The
package should not be tied to one instrument or one file format: Mantid MDHisto
NeXus files are the first adapter, while text exports, triple-axis cuts, powder
data, and other reduced formats should enter through the same generic fitting
pipeline.

Nfit was authored by Paul M. Neves (Johns Hopkins University,
pneves1@jhu.edu) with use of LLM coding tools.

Current package version: 0.17.0.

Heat-capacity import, normalization, fitting, and analysis are documented in
[Heat-capacity data and models](heat_capacity.md).

## MDEvent measured-zero uncertainties

For normalized MDEvent data, a detector-covered bin with zero observed events
is displayed as a measured signal of zero with a finite conservative Poisson
uncertainty. An empty bin has no event row or stored event variance of its own;
nfit estimates its uncertainty scale from accepted events elsewhere in the
requested volume. Only bins without detector coverage are masked. See
[Measured-zero uncertainties](gui_workflows.md#measured-zero-uncertainties)
for a plain-language explanation, the exact convention, and a numerical
example.

```{toctree}
:maxdepth: 2

getting_started
data_philosophy
gui_workflows
data_playground
heat_capacity
plotting
physics_conventions
spin_fluctuation_models
theory_notes
modeling_pipeline
fit_constraints
fit_reports
performance
examples/synthetic_single_q
examples/hyspec_constant_background
api
```
