# HYSPEC constant-background fit

`examples/fit_hyspec_constant_background.py` demonstrates the current
preprocessing, model, resolution, and fitting framework on the bundled HYSPEC
example file:

```bash
PYTHONPATH=src python examples/fit_hyspec_constant_background.py
```

The script:

- imports `data/hyspec/3D_HHL_3meV_1p8K_metallix_m-3m.nxs`,
- flattens the HHL volume into `(H,K,L,E)` point data with
  `point_data_from_hyspec_hhl`,
- masks the elastic line from `-2` to `1` meV,
- masks a phonon-like cone centered at `(2,2,2)` RLU,
- fits a constant measured-intensity model,
- applies Gaussian energy resolution with polynomial FWHM coefficients
  `[0.6, 0.0, 0.0]` meV,
- prints a JSON summary with the best-fit constant, standard error,
  chi-squared, reduced chi-squared, degrees of freedom, and point counts.

By default, the script fits a reproducible subset of 5000 complete `Q`
trajectories so that the example runs quickly. Use all eligible trajectories
with:

```bash
PYTHONPATH=src python examples/fit_hyspec_constant_background.py --max-q-trajectories 0
```

The phonon velocity and resolution oversampling are configurable:

```bash
PYTHONPATH=src python examples/fit_hyspec_constant_background.py \
  --phonon-velocity 35.0 \
  --resolution-oversampling 5
```
