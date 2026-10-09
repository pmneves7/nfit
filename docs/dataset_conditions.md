# Dataset conditions

A **Dataset condition** mask excludes entire runs from a dataset group using
one numeric value per run. Excluded runs contribute neither counts nor exposure
to its composite. Their **Enabled** flags and their individual data remain
available, so disabling the mask restores the original selection.

## Preview and exclude runs

Select a dataset group, add a mask, and choose **Dataset condition** as its type.
Choose a channel and press **Calculate values**:

- **Elastic line intensity** integrates the unsubtracted, corrected native DGS
  events within an energy-transfer window in meV. This channel currently
  requires runs imported through the native raw DGS importer.
- **Metadata** reads a numeric entry, parameter, or source-file channel. Choose
  **Minimum**, **Maximum**, or **Time average**. A scalar uses its own value.
  A series requires timestamped NeXus logs and run start/end times for a time
  average; nfit integrates the stepwise log over that interval. It does not
  substitute an arithmetic average when timestamps are missing.

Opening or leaving this editor displays the saved scalar preview without
reading source events or rechecking every run's reduction inputs. Source
freshness is checked by **Calculate values** and scientific preparation
(such as rebinning or fitting).

The plot includes disabled runs for inspection. Its x axis uses run numbers when
available and dataset order otherwise. The Matplotlib toolbar provides zoom,
pan, axes configuration, and figure export. **Y scale** also offers logarithmic
and symmetric logarithmic axes.

The condition row reads `left limit op value op right limit`. Both active
comparisons must hold. Blank limits are inactive; both blank means no exclusion.
For example, leave the left limit blank and use `value > 0.15` on the right to
exclude high elastic intensity. Use `2 <= value <= 10` to exclude a closed range.
Changing a limit moves its horizontal dashed line immediately. Red points match
this mask's condition; blue points do not. Other masks and Enabled flags can
still affect the final selection.

**Invert** excludes values outside the active condition. **Additive** restores
matching runs excluded by earlier dataset-condition masks. These rules are
ordered from ancestor groups to descendant groups. Additive conditions cannot
override Enabled flags, file masks, or coordinate masks. Disable a condition
with its existing mask Enabled control.

After editing a condition, rebin its composite to update the histogram. The
condition is part of saved projects and exported composite workflows. Original
runs remain available for inspection; excluded runs are omitted from independent
fit blocks.

Conditions also apply when constructing measured-background replay: excluded
sample runs supply no replay angles or exposure, and excluded background runs
supply no background counts or exposure. Coordinate masks can be used alongside
conditions; a condition alone does not mask histogram bins.

## Elastic normalization and cached values

For the elastic diagnostic, nfit uses

```{math}
J_{\mathrm{elastic}} = \frac{\sum_{i:\,E_{\min}\leq E_i<E_{\max}} w_i}
 {q\sum_{d\in\mathcal{D}} N_d}.
```

Here $E_i$ is event energy transfer in meV, $w_i$ is the corrected event weight
from the run's reduction settings, $q$ is accepted beam charge in
microampere-hours, and $\mathcal{D}$ contains detectors with positive
normalization after applying the detector masks. $N_d$ is the detector's
normalization weight in the importer's calibration convention. Without a
normalization file, $N_d=1$ for each accepted detector. The reported value is a
relative integrated intensity in that convention, suitable for comparisons
between similarly configured runs. It includes all accepted detectors and has
no momentum-region selection or division by energy-window width.

The diagnostic respects each run's energy/time-zero settings, beam-pulse
rejection, detector mask, vanadium normalization, detector-efficiency correction,
and incident/final wavevector correction. It uses the same reconstructed and
rounded events as native binning. Saved recipes that omit a numerical policy
use the current importer default, just as native reduction does; calculating
this diagnostic does not rewrite their settings. The requested window must lie
within every run's reduced energy domain. A run with no events in a covered window has value
zero; absent normalization raises an error. This diagnostic does not subtract a
background or apply a UB matrix, symmetry, dataset scale, or fit weight.

Existing reduced-event archives are read in bounded chunks without loading the
whole archive. Without a valid archive, nfit streams a fresh reduction to
calculate the scalar; this preview does not create a 4D histogram or a new
full-event cache. The stored diagnostic error is the square root of summed
corrected event variances divided by the same denominator, treating the
calibration and beam charge as fixed.

Only scalar diagnostics and preview rows are saved with the mask. Threshold
edits reuse them. Adding runs calculates values for the new runs; changes to
source files, reduction settings, calibration, detector masks, or the diagnostic
channel/window invalidate affected values. Preview calculation runs in a
cancellable background task and publishes a complete preview only on success.

Use **Calculate values** after changing diagnostic settings or adding runs.
Explicit binning and fitting also calculate missing values before applying the
selection. Metadata-only GUI summaries do not reduce data; a pending condition
can leave a new/changed run visible until its value has been calculated.

## Scripting

The GUI uses the same public functions available in scripts:

```python
from nfit import dataset_criterion_mask, dataset_criterion_preview

mask = dataset_criterion_mask(
    group, name="Reject excess elastic scattering",
    energy_min=-0.5, energy_max=0.5,
    right_operator=">", right_value=0.15,
)
rows = dataset_criterion_preview(group, mask)

# This edits the selection without repeating the diagnostic calculation.
mask.parameters["right_value"] = 0.16
```

For metadata, use `channel="metadata"`,
`source="entry/DASlogs/temperature/value"`, and
`statistic="min"`, `"max"`, or `"time_average"`. Paths beginning with
`metadata/` and `parameters/` access saved entry values. Source-file logs are limited to one-dimensional numeric channels of at most
32 MiB when decoded as float64; select a scalar aggregate for larger logs.
All selected runs must use consistent channel units; convert sources first if they differ.

`filter_dataset_criteria(root, entries)` returns the original entries surviving
the ordered conditions without changing them. It requires current cached
values by default. Use `prepare_dataset_criteria(root)` to calculate missing
active diagnostics, or `filter_dataset_criteria(..., compute=True)` to calculate
while selecting. Native raw-DGS, MDEvent, and CORELLI binners and composite
construction perform this preparation through their public entry points.
The elastic diagnostic remains specific to raw DGS; numeric metadata conditions
also support other measurement types.
