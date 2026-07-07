# metallix documentation

`metallix` analyzes reduced magnetic-scattering and related experimental data
in physical coordinates such as `H`, `K`, `L`, `|Q|`, and energy transfer. The
package should not be tied to one instrument or one file format: Mantid MDHisto
NeXus files are the first adapter, while text exports, triple-axis cuts, powder
data, and other reduced formats should enter through the same generic fitting
pipeline.

```{toctree}
:maxdepth: 2

getting_started
data_philosophy
physics_conventions
modeling_pipeline
examples/synthetic_single_q
examples/hyspec_constant_background
api
```
