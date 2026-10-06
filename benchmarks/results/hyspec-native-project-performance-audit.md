# Native HYSPEC project performance audit

## Scope

This audit covers the complete `HYSPEC_all_raw.nfit` workflow: raw sources,
automatic grid bounds, sample histograms, directional dummy replay, hierarchical
combinations, cached binnings and project persistence. The implementation was
inspected locally on October 6, 2026. ORNL measurements resumed on node21 after data mounts became available. The
[matched background trial](hyspec-background-batching-node21.md) confirmed and
fixed source-cache ownership and measured a larger-batch improvement. The active
GUI job was not interrupted and neither original scientific project was changed.
The subsequent [11-file background trial](hyspec-background-pooling-node21.md)
also pools matching trajectory exposures across source files, reducing its
background stage by 29.2% with every-cell numerical parity. Event replay still
processes each original source, preserving its uncertainty recipe.

The screenshot reports 27 configured work items, 5 completed after 88 minutes.
These are binnings counted by the batch workflow, despite the label saying
"dataset groups". Its detail timer starts at a new batch item, not a new
scientific stage. Therefore "elapsed 25m23s" is elapsed binning time; it cannot
be assigned solely to the currently displayed dummy run. One item's progress
can pass through both sample banks and their distinct background sources.

The [same-host HYSPEC engine benchmark](dgs-user-workflows-node19.md) used one
361-run sample, without background subtraction. It measured 85.221 seconds for
native initial reduction/binning/saving and 34.388 seconds for a later bin/save.
It does not predict the time for this full project's multiple grids, banks,
temperatures, directional backgrounds and derived data.

### Automatic-bounds passes

A read-only metadata trace of the native project's stale-binning inputs found
four distinct leaf bounds requests for each automatic live temperature
subtraction grid:

| Source collection | Selected runs |
| --- | ---: |
| Low temperature, 34° bank | 722 |
| Low temperature, 70° bank | 361 |
| 50 K, 34° bank | 361 |
| 50 K, 70° bank | 361 |

Their union supplies the shared subtraction grid. `Default` has no symmetry;
`Default copy` uses eight operations in the same HHL basis; `HHH_symm` uses a
different basis. These are distinct bounds requests. The progress message now
names each source collection and its run count. A regression test confirms that
an identical request does not reopen event streams, while a changed UB obtains
new extents from the existing reduced-event cache. The trace read metadata only;
it did not time these scans or profile the user's running GUI job.

## Execution inventory

| Stage | Current acceleration | What to measure next |
| --- | --- | --- |
| Run/log inspection, Ei/T0 and accepted-time angle | Serial per-run preparation with successful inspection reuse | Actual cache hits, repeated metadata reading, monitor fitting versus file access |
| Detector geometry and calibration | Checked immutable resolved-geometry reuse; array calculations | Repeated native replay payload decoding; never merge different bank positions by instrument name |
| Raw HDF5 reads, pulse selection and reconstruction | Serial banks/chunks; vectorized NumPy corrections | Read time, detector lookup and kinematics separately; independent run producers |
| Reduced-event cache construction | Bounded, uncompressed column blocks, one writer per run | Array copying, checksums and writing versus actual reconstruction |
| Automatic coordinate bounds | Streaming event passes, bounded extent cache | Repeated scans across projections/symmetries and derived grids |
| Cached-event reading and sample rotation | Streamed, serial chunks, vectorized arithmetic | Actual read/checksum cost versus rotation; no reduced-event decompression to accelerate |
| Sample event projection and accumulation | Fused compiled Numba, serial ordered updates | Scatter cost and grid locality; shared-bin parallel writes are unsafe |
| Sample trajectory exposure | Compiled parallel integration with checked task pooling | Integration, dense-array initialization/merge and worker limits |
| Background setup and detector matching | Serial preparation with acceptance/geometry deduplication | Repeated cache header/detector-array reads, transform and mask preparation |
| Background trajectory exposure | Parallel integration, pooled tasks across angles and source files after exact detector geometry/output-mask checks | Different grids still require integration; positive exposure weights can be summed for otherwise equal tasks |
| Background event reconstruction and detector recovery | Native reduced-event adapter; vectorized rays plus nearest-detector lookup | Coupling of input chunk size to replay scratch; many tiny reads/queries |
| Background mapping and same-event collision handling | Compiled parallel mapping and per-event hashes | Batch size, thread/barrier overhead and scratch allocation |
| Background histogram updates | Compiled serial ordered scatter | Its fraction of replay time, not the CPU use of mapping alone |
| Hierarchical combination and finalization | Array arithmetic; signature-checked histogram reuse | Duplicate requests on identical grids; full-grid passes on sparse outputs |
| Histogram loading and project saving | Parallel large-member decoding/compression; unchanged asset copying | New artifacts versus reused assets, CRC/copy cost and actual storage bandwidth |

Compiled code can occupy one core while already being much faster than a Python
loop. Increasing concurrency requires demonstrating a whole-workflow gain and
preserving event numerator, variance, counts, bin edges and exposure support,
including low-coverage fringes. Worker-private full histogram grids are not an
acceptable default for these large outputs.

## Specific candidates

1. **Decouple native input/reduction batches from replay batches.** In
   `dgs_background_sources.reduced_event_stream`, the requested replay rows
   determine both the cache iterator slice size and the fresh reducer's byte
   budget. Replay's one-million-transform-task cap gives only 346 rows for
   361 angles and eight operations. A cache miss then requests approximately
   32 KiB reconstruction chunks, despite a much larger workflow byte budget.
   Cached inputs load their stored event blocks once, but detector recovery
   still receives the small slices. Read/reconstruct/recover detector IDs in
   larger bounded blocks, then slice for replay. Preserve order, rejection,
   geometry checks, event weights and covariance recipes exactly.
2. **Measure larger replay batches within admitted memory.** The transform-task
   cap can limit batching before the byte budget does. Larger batches reduce
   allocations and parallel dispatch, but increase scratch and cancellation
   latency. Validate on the real large grid and under resource pressure.
3. **Check duplicate raw reduction across dataset entries.** The native project
   has 5,527 raw source appearances but 1,850 unique files. Its separate crystal
   and powder workflows can refer to identical acquisitions. Reduced caches are
   currently looked up on the individual dataset entry, not through a shared
   project registry of complete reduction identities. Different entries can
   therefore reconstruct the same file separately even when all reduction
   settings agree. The profiler counts completed cache producers by full
   reduction-signature hash to establish the actual duplicated work. Potential
   sharing must include source/calibration identities and reduction settings;
   differing geometry or corrections must retain separate results.
4. **Measure repeated setup and exposure across related bins.** A reduced-event
   cache avoids reconstruction, not trajectory integration or dummy replay on
   a new output grid. Instrument sample exposure/trajectory payload preparation
   and nested requests before adding another cache. A legitimate different-grid
   request must not be mistaken for redundant computation.
5. **Integrate independent raw-run producers only after full-grid validation.**
   [Existing node23 trials](dgs-raw-pipeline-trials-node23.md) found 18–26% initial
   workflow savings with four processes. This remains a prototype. Threads,
   prefetch and fused reconstruction showed much smaller gains. Cached rebinning
   does not benefit from parallelizing work already cached.

### Local replay batch screen

A warmed synthetic control uses 12,000 events, 2,888 transforms, the existing
benchmark's 48×48×12×32 grid, varied weights, acceptance flags and exclusions.
Three sequential repeats per configuration use the current compiled mapping,
collision and scatter kernels. It excludes file I/O, native detector recovery,
trajectory integration and saving; it is not an ORNL or complete-job timing.
All numerator, variance and count arrays are literally identical to the
346-row/one-worker reference. Workers are requested kernel workers.

| Workers | Rows/batch | Median seconds | Reported replay scratch |
| --- | ---: | ---: | ---: |
| 1 | 346 | 0.55929 | 53.4 MiB |
| 1 | 692 | 0.53400 | 106.7 MiB |
| 1 | 1000 | 0.51782 | 154.2 MiB |
| 4 | 346 | 0.18412 | 53.4 MiB |
| 4 | 692 | 0.14982 | 106.7 MiB |
| 4 | 1000 | 0.14442 | 154.2 MiB |
| 8 | 346 | 0.13572 | 53.4 MiB |
| 8 | 692 | 0.08735 | 106.7 MiB |
| 8 | 1000 | 0.08288 | 154.2 MiB |

The eight-worker kernel saves about 39% with 1000 rows. This supports a real-data
trial, not a claimed 39% project speedup. Reported scratch comes from
`replay_scratch_bytes`; output grids, detector flags, input blocks and hash slots
also consume memory. The existing one-million-task cap was changed only in the
manual control, not in production. The receipt is
`hyspec-replay-batch-screen-local.json`.

## Full-project measurement

`benchmarks/profile_dgs_project.py` runs the public load/save workflow on an
isolated project copy. Saving prepares every stale configured binning and its
background. It records HDF5 reads, inspection, detector lookup, raw generator
resumes, cache production/reading, bounds, event accumulation, trajectory work,
background mapping/scatter, compression/decode workers and archive copying.
Generator measurements exclude consumer time between yields. Input arrays and
function arguments are not retained by instrumentation.

Use a new output directory within the same IPTS, and run after competing GUI or
CLI rebuilds finish. Without `--execute` it prints a plan and writes nothing.
The ordinary run retains compatible caches; `--fresh-reduction` discards caches
only in the in-memory copy, for a deliberately fresh reconstruction profile.
Neither mode changes the source project. A cached input may require little new
work: inspect call counts and cache state rather than calling that a full rebin.

```bash
PYTHONPATH=src /path/to/nfit/python benchmarks/profile_dgs_project.py \
  /SNS/HYS/IPTS-36860/shared/nfit/HYSPEC_all_raw.nfit \
  /SNS/HYS/IPTS-36860/shared/nfit/benchmarks/full-project-profile-01 \
  --workers 16 --execute
```

For the existing desktop runtime, pass the argument list as JSON in
`NFIT_DGS_PROJECT_PROFILE_ARGS` and use `nfit --run-script`.
Receipts include module hashes, source identity, progress transitions, process
CPU time, peak resident memory and a main-thread cProfile report. Component
intervals include nested work and worker intervals overlap: **do not add them**.
Component CPU is caller-thread time, so low CPU there does not prove a compiled
parallel kernel is idle. Total process CPU includes the other threads. First
compilation/cache restoration and uncontrolled filesystem caches remain part of
this diagnostic workflow. Profiled results are not headline benchmark timings.

After identifying the major components, run sequential, unprofiled baseline and
candidate workflows on the same node. Accept speedups only with full numerical
channel/support gates and bounded resource/cancellation behavior.
