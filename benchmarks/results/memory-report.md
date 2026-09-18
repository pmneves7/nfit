# SEQUOIA memory investigation

Measured September 17, 2026; fixes prepared for nfit 0.96.3.

## Implemented fixes

- Cache lookups reuse one live immutable result after resident eviction,
  compressed restoration, or disk restoration. Weak references avoid retaining
  an extra copy. Replacement and clearing invalidate the lookup correctly.
- The shared cache budget accounts for results still held by viewers or other
  callers. Repeated references to the same array object are counted once.
- Decompressed and newly rebinned arrays are frozen before constructing immutable
  containers, avoiding defensive copies of buffers already owned by the operation.
- Regular MDHisto rebins above 250,000 source bins generate coordinates, validity
  masks, normalization weights, and geometric coverage in bounded chunks.
  Affine grid corners determine automatic limits, including symmetry images.
  Small grids and discrete/tolerance clustering retain the direct path.
- Streaming worker pools shut down on cancellation and errors.

Scientific precision, project formats, scripting entry points, and the single
CPU/RAM preference pair remain unchanged. No phase-two storage backend or
dependency was added to nfit.

## Data and measurement

The input is the saved SEQUOIA project
`YMn2_SEQ_IPTS-32969.nfit`, member
`assets/binnings/composite-1ea19bbbe0ea49c5a5a8a2e8879382e2-5f0a6b2ed3f64894805c8486ec5f7f6f/data.npz`.
Its full shape is `(201, 172, 172, 89)`. The remote tests use a contiguous central
`(96, 96, 96, 48)` crop: **42,467,328 bins**, 1.62 GiB of numerical payload,
including both auxiliary channels. Every channel and corresponding axis is
cropped identically; scientific values are unchanged.

Final trials run on **analysis-node21.sns.gov**, Python 3.14.7, NumPy 2.5.3,
Numba 0.67.0, with four CPUs and an isolated 16 GiB managed-memory setting.
The comparison uses pre-fix source from `ec7920f` and the final phase-one code.
Each trial starts a fresh process and completes a sufficiently large numerical
warmup before timing. Rebin trials alternate baseline/current code, three times.
Input loading and output saving are outside the operation timer. Process peak
RSS includes imports, input loading, and warmup; it is not an incremental
workspace estimate. No live project or real preference file is modified.

## Phase-one results

| Operation | Before, s | After, s | Before peak RSS, GiB | After peak RSS, GiB |
| --- | ---: | ---: | ---: | ---: |
| Four retained lookups of the same compressed binning | 14.36 | 3.37 | 9.00 | 2.53 |
| Regular rebin, roughly half as many bins per axis | 8.79 | 7.52 | 9.34 | 3.00 |
| Same rebin with inversion symmetry (`P -1`) | 132.81 | 124.98 | 12.46 | 3.73 |

Rebin rows are medians of three trials. The cache row is one sequence of four
lookups, holding every returned result live. It uses an isolated compressed
cache, with construction/compression outside the lookup timer. The old cache
returned four independent payloads totaling **6.49 GiB**; the fixed cache returns
one shared payload totaling **1.62 GiB**. After the initial decode, fixed lookups
took 1–13 microseconds versus about 3.6 seconds each previously. The cache budget
now sees the live decoded payload as well as its compressed backing; the old
accounting saw only the compressed bytes. This trial isolates lookup reuse and
does not model GUI eviction prompts or total application memory.

Regular rebin peak RSS fell **68%**, with a **15% shorter runtime**. The symmetry
case used **70% less peak RSS** and ran **6% faster**. These are measured operation
results on a representative crop, not a prediction of the entire project's RSS.
The full cached grids were not all opened concurrently.

Raw trials are in `memory-sequoia.json`. The output arrays were compared between
baseline and current code, including signal, errors, event counts, coverage,
and normalization; relative and absolute tolerances were both `1e-12`. Masks and
axis arrays were checked exactly. The project size and modification time were
checked against the preparation manifest afterward. These checks and the crop
manifest are saved in `memory-sequoia-validation.json`; numerical source hashes
are in `memory-source-hashes.json`.

The historical 400 GB application peak was not reproduced. The earlier archive
inspection found 12 saved binning artifacts totaling about 128.6 GiB expanded
inside a 14.9 GiB project. Keeping many active grids, old viewer results, and
temporary arrays can therefore still consume substantial RAM after these fixes.
The managed RAM allowance cannot free arrays an active viewer still owns.

## Storage evaluation only

The local storage comparison uses a further `(48, 48, 48, 24)` crop of the same
SEQUOIA sample. It contains the original float64 signal, errors, and event counts:
63.70 MB total. Signal values include 19.13% NaNs and 3.23% exact zeros. Coverage
and normalization channels are not included in this smaller storage experiment.
Times are medians of three fresh-process trials on macOS arm64.

| Candidate | Stored size, MB | Write/build, ms | Full signal read, ms | ROI sum, ms |
| --- | ---: | ---: | ---: | ---: |
| Resident NumPy | 63.70 | 4.14 | 1.22 | 0.19 |
| NumPy `.npy` memmap | 63.70 | 8.09 | 2.62 | 0.22 |
| HDF5 gzip | 30.81 | 704.26 | 64.11 | 15.94 |
| HDF5 LZF | 34.54 | 126.45 | 20.77 | 4.42 |
| Blosc2 ZSTD, level 5 | 30.48 | 141.22 | 8.73 | 1.96 |
| Blosc2 LZ4 + bitshuffle, level 1 | 32.86 | 61.07 | 8.63 | 1.71 |
| In-memory Blosc2 LZ4 + bitshuffle | 32.99 compressed payload | 25.21 | 7.37 | 1.07 |

Every persisted read follows that process's write: these are **warm page-cache
measurements**, not cold-disk or memory-pressure tests. In-memory compressed size
counts compressed payload bytes, excluding container overhead. Peak process RSS
in the JSON includes original inputs and correctness-check temporaries; it does
not measure the incremental retained memory of a storage candidate. Full reads
materialize a NumPy signal array; ROI sums materialize the selected ROI before
NumPy reduction. Native Blosc2 lazy/fused reductions were not benchmarked.
All candidate payloads and requested slices were checked for exact equality.

## What makes sense for a second phase

One further compute optimization is worth testing before changing storage:
recognizing uniformly spaced coverage edges could allow the existing fused
Numba accumulator to replace repeated NumPy partial histograms. The current
explicit-edge coverage path also supports nonuniform grids, which must retain
their present behavior. This optimization is not part of the delivered fixes.

Keep active numerical work in resident NumPy arrays. The real sample shows that
lossless compression can nearly halve payload storage, but decoding adds latency:
in-memory Blosc2 was about six times slower than resident NumPy for full signal
reads and roughly five times slower for the tested ROI sum. Disk-backed Blosc2
LZ4 was about 3.3 times slower than warm memmap for full reads and eight times
slower for the ROI sum. It is a promising inactive-result tier, not evidence for
a universal replacement of hot arrays. Blosc2 supports chunked arrays and lazy
computation, which warrant separate tests before making a backend decision.
See the [Blosc2 overview](https://blosc.org/python-blosc2/getting_started/overview.html).

The next prototype should compare a read-only `.npy` mapping on local scratch
disk with Blosc2 LZ4 storage for inactive binnings, promoting only actively used
data into the managed RAM cache. NumPy mapping is the lowest-risk starting point
for compatibility with current array consumers. It lets the OS reclaim file
pages, but cached pages still count toward resident memory, and page faults under
pressure can slow access. The compressed `.nfit` archive cannot simply become a
memory mapping; a separate backing artifact and lifecycle policy are required.
See [NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

Before implementation, measure actual viewer reductions under memory pressure,
on local scratch and GPFS, including different slice orientations and repeated
navigation. Audit implicit `np.asarray` calls to avoid silently loading complete
grids. Retain the one CPU/RAM preference pair; backend, chunk, and promotion
policies should be automatic. No choice here guarantees zero slowdown when
working data exceed physical RAM.

Mantid provides a useful design precedent: file-backed MDEvent workspaces keep
the box index in memory and load events on demand, while documenting slower
binning/slicing for some disk-backed workloads. This is an event-storage design,
not evidence that arbitrary dense MDHisto arrays can be compressed without cost.
See [Mantid's file-backed MD workspaces](https://docs.mantidproject.org/nightly/concepts/MDWorkspace.html#file-backed-mdworkspaces).

## Reproduction

`benchmarks/prepare_sequoia_memory_sample.py` creates the cropped NPZ and a source
manifest using only NumPy and the standard library. It refuses to overwrite the
project or an existing output. Run `benchmarks/benchmark_memory.py` in separate
processes with `PYTHONPATH` selecting the baseline or current source, an isolated
`NFIT_PERFORMANCE_FILE`, and the same `--input` NPZ. Operations are `cache`,
`rebin`, and `symmetry`; use `--output` to retain numerical results for comparison.
`benchmarks/benchmark_storage_candidates.py --input-fixture ...` accepts a small
NPZ containing signal/errors/num_events. Blosc2 is optional and was installed
only into temporary benchmark storage.

Raw storage measurements are in `storage-candidates-sequoia-real.json`; the
synthetic comparison is in `storage-candidates-synthetic.json`. The small-grid
check is in `memory-small-arrays.json`: 15,360 bins retained the direct path,
with median runtimes of 25.2 ms before and 26.2 ms after (overlapping trial
ranges; too small a difference to establish a speed regression).

Validation: the complete suite passed **1,781 tests**, with one optional CuPy
test skipped. The latest benchmark/preparation/packaging checks also passed.
Ruff, byte compilation, `git diff --check`, and the Sphinx build with warnings
treated as errors passed.
