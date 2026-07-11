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

Current package version: 0.4.2.

```{toctree}
:maxdepth: 2

getting_started
data_philosophy
gui_workflows
physics_conventions
spin_fluctuation_models
modeling_pipeline
performance
examples/synthetic_single_q
examples/hyspec_constant_background
api
```
