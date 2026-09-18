# Large mapped-artifact benchmark provenance

These reports compare the existing resident artifact reader with the bounded
four-worker mapped reader. Both reader processes ran on
`analysis-node21.sns.gov` with `NFIT_NUM_THREADS=4`, `OMP_NUM_THREADS=4`,
`OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and
`NUMEXPR_NUM_THREADS=4`. The interpreter was
`/SNS/users/paulneves/miniforge3/bin/python` (Python 3.14.7, NumPy 2.5.3).
The source checkout began at Git commit
`46daf6d18415d657f652e9f186b653d4ab47ba52`; the benchmarked uncommitted files
are identified by content hash below. The two benchmarked source hashes were
calculated from the local files immediately after the run; those files were the
sources supplied by `rsync`, but their hashes were not independently
recalculated on the remote host before its scratch directory was removed. The
benchmark harness was subsequently corrected to describe parallel inflation;
its current hash is also recorded below.

The OS page cache was not cleared. Results measure post-write,
OS-cache-dependent behavior and are not cold-disk claims. Each reader ran in a
fresh child process. Exact bounded-chunk hashes matched for all six numerical
arrays, archive SHA-256 remained unchanged, and each worker reported unchanged
source inode, size, and modification time.

Erratum: the raw result files' `measurement.decode_concurrency` string says the
mapped reader inflated members sequentially. That inherited label is stale:
the final rows used four bounded mapped-inflation workers, as independently
recorded by `measurement.mapped_workers: 4` and each mapped row's
`nfit_num_threads: "4"`. The raw reports remain byte-for-byte unchanged so the
recorded hashes continue to identify the measured results.

SHA-256 hashes:

- `benchmark_mapped_artifacts.py`: `7d905983869994d107db26a2fa841841700dfbe61b189be7c1fd3f160b787065`
- current corrected `benchmark_mapped_artifacts.py`: `aa6629e684a467f4877d5e100367d9d55b6991c0027feb16fe0d6a3cf3aa0d0e`
- `mapped_archive.py`: `b6b954d956c0fa5c698a1a5358d4520cafb23aff11f9231ba4dee01a670c2c4d`
- `prepare-1gb.json`: `d59f1844b9297e50550ef8b4b6dd2832fe5f276de455f84083bb2553c1e186ac`
- `prepare-10gb.json`: `5b2fff0b953b260a93a08bd2a2957db517b648f086379669675de5b1c1b41f5d`
- `results-1gb-parallel.json`: `1a9e031f90025d3878f9a9350f085498f4db1bc5cc797bf9df01033249d8c3a1`
- `results-10gb-parallel.json`: `512e394370fa8d0b2d156bdb24f09f3ec0ab0583fc5e1aca1b2f8b4efd7114cc`

Remote cleanup completed after downloading the reports. The shared root
`/SNS/users/paulneves/.cache/nfit-phase2-mapped-OqC5xL` was removed and its
absence verified. An earlier load-balanced attempt used
`/tmp/nfit-mapped-artifacts-Cz62Ea` before its creation hostname was recorded.
The initial cleanup probe suppressed per-node SSH failures, so it did not prove
absence on every node in that range. A subsequent load-balanced connection
verified the path absent on its selected node. See `cleanup-audit.txt` for the
later per-node reachability and path-check audit. No artifact from that
interrupted attempt was recovered or retained.
