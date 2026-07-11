# HYSPEC constant-background fit

`examples/fit_hyspec_constant_background.py` demonstrates the current
preprocessing, model, resolution, and fitting framework on the bundled HYSPEC
example file:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/fit_hyspec_constant_background.py
```

The script:

- imports `data/hyspec/4D_test.nxs`,
- flattens the HHL volume into `(H,K,L,E)` point data with
  `point_data_from_hyspec_hhl`,
- packages the masked point data in a `DataGroup` and fits through a
  `FitModelSession`,
- masks the elastic line from `-2` to `1` meV,
- masks a phonon-like cone centered at `(2,2,2)` RLU,
- fits a constant measured-intensity model,
- applies Gaussian energy resolution with polynomial FWHM coefficients
  `[0.6, 0.0, 0.0]` meV,
- can attach a fit-comparison view to the dataset and open the slice viewer,
- prints a JSON summary with the best-fit constant, standard error,
  chi-squared, reduced chi-squared, degrees of freedom, and point counts.

By default, the script fits a reproducible subset of 5000 complete `Q`
trajectories so that the example runs quickly. Use all eligible trajectories
with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/fit_hyspec_constant_background.py --max-q-trajectories 0
```

The phonon velocity and resolution oversampling are configurable:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/fit_hyspec_constant_background.py \
  --phonon-velocity 35.0 \
  --resolution-oversampling 5
```

Open the interactive viewer after fitting with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/fit_hyspec_constant_background.py --view-fit
```

The attached comparison contains three matching views on the original HYSPEC
axes: the masked data points used in the fit, the fitted model values, and
`(data - fit) / sigma`. Use the "Show model" checkbox to open linked data/model
panels, choose the model and fit result from the dropdown menus, and toggle the
residual panel.
