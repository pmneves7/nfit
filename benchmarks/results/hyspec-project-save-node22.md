# Histogram save performance on ORNL

Measured on analysis-node22 on October 7, 2026, with a 16-worker ceiling and
actual HYSPEC diagnostic arrays. The histogram contains **20,921,628 bins**, occupying 690,413,724
bytes across signal, errors, event counts, mask, and normalization denominator.
No reduction or binning occurs in these timings.

| Operation | Before | After | Effect |
| --- | ---: | ---: | --- |
| Write a new histogram NPZ, median of two trials | 0.895 s | 0.684 s | 24% less elapsed time |
| Compression CPU time, median of two trials | 6.55 CPU s | 4.47 CPU s | 32% less CPU time |
| Save an existing NPZ spill into a project, one trial | 1.829 s | 0.111 s | 16.5 times faster |
| New histogram NPZ size | 143,070,573 bytes | 149,445,014 bytes | 4.5% larger |

The new writer uses lossless DEFLATE level 1 instead of level 6, and starts
parallel compression at 8 MiB instead of 32 MiB. The existing-spill comparison
uses the identical original NPZ: the old path decodes and recompresses it,
whereas the new path streams it into the project archive. This applies to
ordinary dataset, composite, and owned background histogram caches.

Timings include artifact publication, project archive copying, and filesystem
synchronization. Header/dtype/shape and numerical-byte comparisons run outside
the timed intervals. Every array agrees byte for byte, including NaNs, and the
reused NPZ agrees by SHA-256. The scientific project was not modified. Temporary
benchmark artifacts were deleted after recording results; receipts and logs
remain in the HYSPEC IPTS benchmark directory.

These are warm filesystem measurements of one representative histogram, not
an end-to-end time for the whole HYSPEC project. They omit the full cached
background recipe metadata, other histograms, and reduced-event assets. They
do not establish why the user's older GUI save on another node took hours.
The SSH connection landed on node22 and could not inspect the GUI process on
node23. The running version in the screenshot was 0.119.2, before the archive
progress coalescing fix in 0.119.3.

## Reproduction

`benchmarks/trial_histogram_save.py` takes an HDF5 diagnostic histogram, a new
output directory, an older `array_archive.py` snapshot, and a worker ceiling.
Run it with the candidate nfit build. On ORNL, select an output directory in
the experiment's IPTS folder. The input histogram is read-only, all temporary
scientific output stays in that directory, and no Mantid module is imported.

The original timing receipt is retained in
`benchmarks/results/hyspec-project-save-node22.json`.
