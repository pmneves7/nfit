# HYSPEC data

Put local HYSPEC data files here:

```text
data/hyspec/raw/
```

For the real Mantid `SaveMD` output, a good layout is:

```text
data/hyspec/raw/<your_hyspec_savemd_file>.nxs
```

The `raw/` directory and large NeXus/HDF5 files are ignored by git. Keep this
README tracked so the expected location is visible without committing acquired
beamline data.
