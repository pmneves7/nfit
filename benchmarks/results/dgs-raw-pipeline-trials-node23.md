# Native DGS raw-pipeline trials on node23

## Scope

Manual process-local experiments against installed nfit 0.117.0, followed by
the 95% filtering check on 0.117.1, on analysis-node23, October 5, 2026.
Original projects and application numerical
code are unchanged. Scientific staging, snapshots, logs and outputs are inside
the respective SEQUOIA/HYSPEC IPTS `shared/nfit/benchmarks` directories.

Each job imports raw sources, reconstructs and caches events, calculates a
six-operation histogram and saves a native project. It then reopens that project,
accesses its saved histogram, calculates a twelve-operation binning and saves
again. Initial saved-dataset time and later rebin/save time are reported
separately. Diagnostic exports/comparisons are untimed. Process producer startup
and the extra cache read are included in the initial workflow.
The cached rebin/save column excludes reopening and accessing the saved primary
histogram; those operations are included in the combined user-workflow total.

Trials run sequentially with eight normalization workers and one BLAS thread.
Small sweeps use two rounds in reversed order; the first eight-run SEQUOIA screen
uses one round. The calibration-snapshot 256-run process comparison is repeated
in reverse order. OS caches are unflushed and other
node users are uncontrolled, so small differences are not reliable evidence for
adoption. Numerical module hashes and source identities are retained per job.

The 32-run SEQUOIA and HYSPEC selections span their configured rotation ranges.
The 256-run selection spans the 617-run NiO collection. These are coarse-grid
screening trials, not timings for every binning of the large GUI project. They
do not rerun Mantid. Existing same-node engine comparisons remain in
[the user-workflow report](dgs-user-workflows-node19.md).

## Results and recommendation

Four independent run producers are worth pursuing for large raw reductions.
The repeated 256-run SEQUOIA comparison contains 155,728,368 raw events and
390,728 output cells. This screening cube uses momentum steps 0.12, 0.12 and
0.08 r.l.u. and an energy step of 2 meV, with the project's three projection
vectors. It is deliberately smaller than the large GUI project's grids.

| Execution order | Serial initial workflow | Four-process initial workflow | Initial time saved | Serial cached rebin/save | Four-process cached rebin/save |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial, then processes | 139.17 s | 103.10 s | 25.9% | 45.75 s | 43.70 s |
| Processes, then serial | 133.01 s | 103.15 s | 22.5% | 43.65 s | 44.34 s |

These two pairs use the earlier benchmark recipe's 0% bad-pulse threshold;
the separate 95% comparison below checks the main project's filtering policy.
The first pair's reduction/cache/binning stage drops from 109.78 to
77.06 seconds; import/setup and final saving are also included in the headline
139.17/103.10 seconds. The cached-rebin stage has no producers to parallelize,
so it shows no consistent gain. The two-stage workflow saves about 16–21%.

Each producer's maximum resident set is about 1.1 GiB, in addition to the parent.
These are individual process peaks, not a measured simultaneous aggregate peak.
Four full histogram grids are **not** allocated. The parent waits for completed
per-run cache references and bins through the existing ordered accumulator.

### Main-project pulse filtering

The same 256 sources and immutable calibration snapshot, with the main project's
95% bad-pulse threshold, were repeated on stable installed nfit 0.117.1:

| Workflow | Serial | Four processes | Time saved |
| --- | ---: | ---: | ---: |
| Import, reduce, cache, six-copy binning, save | 138.99 s | 113.62 s | 18.3% |
| Cached twelve-copy rebinning and save | 45.64 s | 44.54 s | 2.4% |
| Complete workflow, including reopen and saved-histogram access | 185.34 s | 158.87 s | 14.3% |

This pair runs processes first, then serial. Counts, raw event payloads and
exposure support pass the same numerical gates as the earlier trials.
The earlier attempt overlapped deployment of the workspace fixes; its native
module identity check failed, so its times are excluded. The successful repeat
uses one unchanged build throughout both cases. The cached-stage difference is
small and is not evidence of faster cached-event accumulation.

### Other strategies and production scope

Smaller jobs do not amortize the process setup. On 32 SEQUOIA runs, serial
initial workflow times were 27–28 seconds and four-process times 41–49 seconds.
For 32 HYSPEC runs, serial was about 9 seconds and four processes about 21–22
seconds. A production implementation should keep a serial path for smaller
collections and admit a bounded number of producers using operation CPU/RAM
budgets. It needs portable worker launch, joined cancellation, guarded cache
publication and source-identity checks before consumption. The prototype's
AST extraction must not become the production implementation.

The remaining strategies do not justify application complexity:

- Two/four reconstruction threads, coalescing and prefetch showed no convincing
  whole-workflow SEQUOIA gain on the initial eight-run screen.
- Dense lookup and fused kinematics stayed close to serial times on repeated
  32-run SEQUOIA/HYSPEC jobs. Their best HYSPEC combinations were only about
  5–8% better, including saving.
- Whole-run threads gained roughly 3% for 32 SEQUOIA runs and 8% for 32 HYSPEC
  runs; the larger 256-run SEQUOIA gain was about 5%. Separate runtimes avoid
  [h5py's process-wide HDF5 lock](https://docs.h5py.org/en/stable/threads.html),
  which limits the thread-based pipeline.
- Linear pulse membership on 64 SEQUOIA runs with 95% filtering took 43.30
  seconds in both rounds, versus 46.16 and 43.73 seconds serially. That small
  difference is not a reason to add another pulse-selection implementation.

No prototype is enabled in the desktop application. Trial tooling and receipts
are retained so the substantial process gain can be reproduced and integrated
with the resource and cache lifecycle services. Cached-event parallel scatter
still needs an algorithm that avoids shared-bin races and per-worker full-grid
copies; the previously measured bin-owner prototype's small gain remains
insufficient. These results do not establish a speedup for the full 617-run
collection, fine grids, background replay or every binning in the GUI project.

## Strategies

| Trial | Work changed | Main additional cost |
| --- | --- | --- |
| Coalescing | Deliver larger contiguous fresh-event blocks to the unchanged accumulator | Another bounded copy and delayed consumer progress |
| Prefetch | Overlap one reducer with cache writing and accumulation | One extra live block and thread coordination |
| Chunk workers | One HDF reader and two/four ordered reconstruction workers | Bank-sized task dispatch; h5py calls remain serialized |
| Dense ID map | Index the owning geometry's detector rows directly when IDs are compact | Bounded immutable map; sparse IDs retain searchsorted |
| Fused kinematics | Combine final energy and laboratory momentum arithmetic | Compilation and another numerical path; correction ufuncs stay unchanged |
| Run threads | Produce per-run caches with two/four workers, then native binning | Extra cache read; HDF calls still share a process-wide lock |
| Run processes | Separate existing runtimes produce per-run caches | Startup, repeated geometry setup, extra cache read |
| Linear pulse membership | Expand intersected pulse ranges rather than per-event pulse search | Pulse-range checks and a bounded boolean mask |

Every run resolves its own complete geometry and HYSPEC preprocessing. Detector
IDs or instrument names are not used as geometry identity. Writers, buffers and
worker dataset objects are private; completed references are published on the
caller thread. Neither run strategy creates private full histogram grids.

## Numerical acceptance

Every candidate is compared with its own series/round baseline. Histogram
counts and edges are literal; finite/exposure support must match exactly.
Weighted numerator C, variance numerator V and exposure N use a 10⁻¹² relative
tolerance to allow changed sum grouping. Intensity is C/N and stored standard
uncertainty is sqrt(V)/N, so these checks include low-exposure cells rather than
only peak pixels. C and V have event-weight units and squared units; N follows
the configured calibration/exposure convention. Counts are dimensionless.

Saved raw caches are compared independently, including events outside the output
cube. Reduction signatures, scalar headers, detector directions, solid-angle
payloads, charges, corrected event weights and variance match literally.
Laboratory momentum coordinates (Å⁻¹) and energy transfer (meV) allow 10⁻¹²
relative/absolute arithmetic roundoff. Accepted event order and raw totals must
match. Unit checks include both precision policies, invalid/duplicate/sparse
detector IDs, empty chunks, bank boundaries, pulse boundaries, bad pulses,
pause/deadtime filtering and early generator closure. They do not load Mantid.

## Source-identity interruption

The first 256-run process trial is excluded. Its processed-vanadium file kept
the same size and modification timestamp but changed its inode change timestamp
during the job. That invalidated all preproduced cache signatures, correctly
triggered raw reduction again and failed the benchmark cache-hit assertion.
The cause of that filesystem metadata change is not established. The later
repeat uses a benchmark-owned calibration snapshot and records its checksum.
Original calibration files are not edited.

## Receipts

Remote roots:

- `/SNS/SEQ/IPTS-37189/shared/nfit/benchmarks/raw-pipeline-node23-20261005`
- `/SNS/HYS/IPTS-36860/shared/nfit/benchmarks/raw-pipeline-node23-20261005`

Each case retains `receipt.json`, `trial.json`, progress/resource logs, the native
project, two diagnostic histograms and numerical reports. Immutable `code-v*`
directories retain the actual prototype versions used. The failed bootstrap
screen has no timing result and is excluded.

[Scalar receipts](dgs-raw-pipeline-trials-node23.json) retain 67 successful
workflows, 48 candidate-to-baseline raw-cache comparisons and 96 full-histogram
comparisons, together with source/build checks, series settings and phase times.
No scientific arrays are included in that summary. The committed manual tools
also add a fail-fast source-identity check before publishing producer references
and process-group cleanup, independently tested without external engines.
