# HYSPEC data

Large HYSPEC data files should usually live outside the git checkout. The
notebook looks first in:

```text
~/metallix-data/hyspec/raw/
```

You can override that location with:

```bash
export METALLIX_HYSPEC_DATA_DIR=/path/to/hyspec/raw
```

If you prefer a repo-local scratch location, put files here:

```text
data/hyspec/raw/
```

For the real Mantid `SaveMD` output, either layout is fine:

```text
~/metallix-data/hyspec/raw/<your_hyspec_savemd_file>.nxs
data/hyspec/raw/<your_hyspec_savemd_file>.nxs
```

The repo-local `raw/` directory and large NeXus/HDF5 files are ignored by git.
Keep this README tracked so the expected locations are visible without
committing acquired beamline data.

Useful smoke tests once the file is available:

```bash
python examples/import_hyspec_mdhisto_nxs.py
python examples/view_hyspec_mdhisto_slice.py
python examples/plot_hyspec_test_datasets.py --save-dir /tmp/hyspec-test-plots
```

The viewer can switch between same-shaped MDHisto channels (`signal`, `errors`,
`num_events`/multiplicity, `combined_mask`, `file_mask`, and `metallix_mask`)
from the Channel dropdown. The Figure panel can copy the current figure or
export a `plot_mdhisto_slice(...)` script that reproduces the current display
settings without launching the GUI.
When multiple datasets are loaded into the Qt viewer, the Dataset dropdown
switches between them and rebuilds the axis/integration controls for the
selected file.

`plot_hyspec_test_datasets.py` treats `1D_test.nxs` specially: because only one
dimension is non-singleton, it renders a line/errorbar plot instead of a
colormap. Higher-dimensional test files are rendered as MDHisto slice figures.

`load_mantid_mdhisto_nxs` does not copy bulky ancillary NeXus metadata by
default. Use `copy_metadata=True` only when you explicitly need the additional
metadata tree; otherwise the importer reads only axes plus `signal`,
`errors_squared`, `mask`, and `num_events` from `/MDHistoWorkspace/data`.
