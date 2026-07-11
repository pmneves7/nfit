# Synthetic single-Q fit

The first reference example generates a synthetic overdamped paramagnon dataset,
adds Gaussian noise, and fits measured intensity. The susceptibility model and
cross-section wrapper are separate:

```python
from nfit.models import paramagnon_chipp
from nfit.cross_section import intensity_from_chipp
```

Run it with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/synthetic_single_q_fit.py
```

The example writes a compact JSON summary to
`examples/reference_cases/single_q_paramagnon/summary.json`.
