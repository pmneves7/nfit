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
```

The viewer can switch between same-shaped MDHisto channels (`signal`, `errors`,
`num_events`/multiplicity, and `mask`) from the Channel dropdown. The Figure
panel can copy the current figure or export a `plot_mdhisto_slice(...)` script
that reproduces the current display settings without launching the GUI.
