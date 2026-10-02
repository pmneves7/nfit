# Source and reduction recipes

Native raw DGS, CORELLI and MDEvent collections keep reproducible source and
reduction recipes. A recipe records resolved source files and logical runs,
shared settings, per-run overrides, automatic results, calibration fingerprints
and each run's acquisition geometry. It does not assume one instrument geometry
for a whole collection. Files remain external references; reduced events remain
lazy project assets.

## Edit shared settings and individual runs

The collection's reduction panel shows all supported settings from one schema.
Choose a run to override its inherited values. **Inherit** removes the override;
**Automatic** stores `None` explicitly. Automatic incident energy and time zero
are shown alongside their resolved values and calibration provenance. A previous
resolved result is marked stale after a relevant edit until reduction runs again.

Incident energy $E_i$ is in meV. Time zero $T_0$ is in microseconds and may be
negative. An explicit run $E_i$ overrides that run's reconstruction and trajectory
energy; other trajectories use the selected first-run or per-run convention.
An automatic run override cancels a shared manual value. The first-run convention
still uses the first participating run as its reference before mask partitioning.

MDEvent files already contain reconstructed coordinates. Their adjustable $E_i$
affects trajectory normalization only; stored $T_0$ is provenance, not an editable
control that could change those coordinates. CORELLI reconstruction exposes its
own timing, wavelength, calibration, pulse and efficiency settings. Its energy
hypotheses depend on the requested output grid; a DGS laboratory-event cache does
not represent those hypotheses.

Use the same public API in scripts:

```python
from nfit import set_reduction_settings, effective_reduction_config

set_reduction_settings(collection, {"bad_pulse_threshold": 95.0})
run = collection.datasets[0]
set_reduction_settings(collection, {"t0_override": -20.0},
                       dataset_ids=[run.id])
settings = effective_reduction_config(collection, run)
set_reduction_settings(collection, {"t0_override": None},
                       dataset_ids=[run.id], inherit=True)
```

Validation checks all affected effective configurations before committing an
edit. Setting names, defaults, units, validation and tooltips come from
`reduction_settings_schema(collection)`.

## Coordinate transforms and cache dependencies

The UB matrix is owned by the collection's coordinate transform, using the
[physics conventions](physics_conventions.md). DGS caches contain laboratory
momentum transfer, so changing UB, symmetry or output bins reuses those events.
Changing a run's effective TOF conversion, detector selection, pulse filtering
or efficiency calibration discards only that run's reduced cache. Other runs
remain reusable. Calibration file identities participate in histogram signatures
as well as event-cache signatures.

Raw geometry reuse checks the complete current embedded instrument definition.
CORELLI also checks each run before reusing detector geometry. MDEvent trajectory
batching compares actual detector arrays and calibration values. The cache
signature includes file path, size and nanosecond modification/change times;
these are lightweight fingerprints, not checksums of entire acquisition files.

Adding and removing runs changes histogram membership without discarding the
remaining DGS caches. Saved projects bind reduced-event references without loading
their numerical arrays. Event blocks remain uncompressed for fast bounded reuse.
An absent unchanged raw/calibration file can use its last cached fingerprint;
changed reduction settings require the original inputs to regenerate events.

## Save and replay

**Copy reduction recipe script** exports a complete editable source/reduction
recipe. Scientific settings live in `shared_defaults`, `per_run_overrides` and
`coordinate_transform`; the structural `config` retains legacy importer context.
Run identity includes the canonical source path and, for MDEvent containers, the
experiment index. Recipes retain resolved membership rather than silently
re-evaluating a directory expression on replay.

```python
from nfit import export_reduction_recipe, replay_reduction_recipe

recipe = export_reduction_recipe(collection)
recipe["shared_defaults"]["bad_pulse_threshold"] = 90.0
rebuilt = replay_reduction_recipe(recipe)
```

`reduction_workflow_script(collection, binning_config=...)` additionally exports
an editable binning recipe and runs from the original sources without Qt or a
saved project. An isolated native collection supports its own masks and scaling;
nested topology or linked backgrounds use `composite_workflow_script(...)`, whose
editable native recipes are applied to the saved workspace. Complete standalone
export of arbitrary composite topology remains separate workflow work.

Replaying freshly inspects acquisition metadata and preserves logical run IDs,
ordering and saved settings. It neither imports Mantid nor calls Shiver.
