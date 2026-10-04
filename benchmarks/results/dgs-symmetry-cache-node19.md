# Cached DGS symmetry expansion on node19

The 0.116.5 candidate reuses the additive statistics of a compatible cached
six-operation histogram when constructing the same-grid twelve-operation
histogram. It computes the six missing operations, joins event numerator C,
variance V, exposure N and contribution counts, then reconstructs signal,
uncertainty, coverage and threshold masks.

This uses existing lazy, unscaled, unsubtracted histogram caches. Complete
source, reduction, geometry, calibration, weights, masks, grid and policy
signatures must agree except for symmetry. Operation multisets must match
exactly, including multiplicity; the largest compatible subset is preferred.
Only native raw DGS HKLE independent-copy statistics are additive in this path.
Reduced MDE masks, copy covariance, dependencies, automatic bounds, metadata
axes, powder and hierarchical composites retain full replay.

## Matched saved-project measurements

Both jobs start from the same saved benchmark project, load its primary
six-operation histogram, calculate a **new** twelve-operation histogram from
saved events, and save a new project containing all existing histograms.
Equivalent whole-histogram reuse is disabled only for that timed new request.
Original projects remain unchanged. All reduced-event caches hit; no raw
reduction occurs. The control and candidate use the same harness and limits.

The host is analysis-node19: Xeon Gold 6246R, 32 physical/64 logical CPUs.
Both use a 64-thread ceiling, 200000 MiB managed RAM and one BLAS thread.
Jobs run sequentially; filesystem caches are uncontrolled. Normal buffered
saves are included; diagnostic scans and exports are excluded.

| Stage | SEQUOIA control | SEQUOIA reuse | HYSPEC control | HYSPEC reuse |
| --- | ---: | ---: | ---: | ---: |
| Metadata loading | 1.325 s | 1.254 s | 0.715 s | 0.643 s |
| Saved primary histogram access | 3.690 s | 3.666 s | 0.335 s | 0.426 s |
| New binning setup | 0.233 s | 0.240 s | 0.112 s | 0.128 s |
| Cached-event twelve-operation histogram | 215.565 s | 160.899 s | 38.765 s | 30.316 s |
| Save new project | 39.629 s | 43.816 s | 9.661 s | 10.045 s |
| Complete saved-project workflow | **260.527 s** | **209.932 s** | **49.665 s** | **41.626 s** |

The histogram calculation saves **25.4% / 21.8%**; the complete saved workflows
save **19.4% / 16.2%** for SEQUOIA/HYSPEC. These are measured paired jobs,
not estimates of a twofold speedup from halving symmetry work.

SEQUOIA uses all 617 sources and 91,584,578 cells. Exposure work falls from
486,374,976 to 243,187,488 trajectories, with normalization intervals
126.676 → 68.376 s. Event intervals improve 78.784 → 62.895 s; reading all
saved events and rotating their coordinates remains necessary. Statistics
validation/joining adds finalization work: 8.763 → 23.116 s. Peak RSS is
49,340.5 / 47,219.1 MiB; saved reduced-event payloads remain 11.520 GB.

HYSPEC uses all 361 sources in the 34° bank and 3,365,793 cells. Its smaller
exposure calculation and unavoidable event reads leave less reusable work.
Peak RSS is 1,115.5 / 1,092.3 MiB; reduced-event payloads remain 3.988 GB.

These paired jobs retain an existing twelve-operation cache as well as the
new one, so their archive sizes and saving work differ from the standard
initial-six/later-twelve benchmark. Cross-engine headline ratios use that
standard workflow in [the full workflow report](dgs-user-workflows-node19.md).

## Every-cell acceptance

Both full grids pass exact contribution counts, edges and exposure support.
Saved mask checksums agree exactly; signal/error finite counts are unchanged.
C and V also happen to be literal in these real datasets. N changes only
through floating-point addition grouping: maximum relative nonzero-cell
differences are **1.70×10⁻¹⁵ / 1.69×10⁻¹⁴**, with relative L2 differences
2.72×10⁻¹⁶ / 2.06×10⁻¹⁵. SEQUOIA retains the same nine event-containing
cells without positive exposure; HYSPEC has none.

The NiO HHH/energy cut at K=0±0.03 and L=0.33±0.02 r.l.u. also retains exact
C/V/counts/support. Signal and uncertainty agree at rounding precision across
coverage fringes, the lowest-exposure decile and the central low-coverage region.

The acceptance gate permits 10⁻¹² relative C/V/N differences with zero absolute
tolerance, while requiring literal counts/edges and exact exposure/nonfinite
support. This explicitly allows changed summation association without weakening
the existing stricter native gate. Signed numerator, partial zero-exposure
contributions, source/configuration invalidation, threshold masks, scales,
background separation, lazy reopening, duplicate operations and low-RAM fallback
have separate controlled tests. Independent scientific review found no blocking
issue; the final full suite passed 3,234 tests with one optional CuPy skip.

## Workload guard

An instrumented 24-run job on the same 91.6-million-cell SEQUOIA grid exposed
a regression: validating and joining full-grid statistics cost more than the
event and trajectory work saved. Its incremental histogram took 30.779 s;
the ordinary matched control took 20.523 s. The profile is diagnostic and is
not an unprofiled timing comparison.

The final implementation checks workload before loading a partial cache.
For grids of at least 250,000 cells it requires known source counts and at
least eight saved event projections per cell. Smaller grids retain the cheap
incremental path. Missing counts, insufficient work or insufficient RAM request
full replay. The threshold is a conservative cost heuristic, not a scientific
parameter or a guarantee of a particular speedup.

The final guarded 24-run job correctly used full replay: 19.063 s binning and
30.735 s including loading and saving, versus 20.523 / 32.739 s for its control.
No improvement is attributed to the small timing difference. The guarded
617-run job retained six cached and six computed operations: 153.954 s binning
and 200.654 s including loading and saving, versus 215.565 / 260.527 s for
the matched control. It saves 28.6% of binning and 23.0% of that workflow.
All 617 event caches hit; no raw reconstruction occurred.

Both final guarded grids pass literal C/V/counts/edges and saved mask digests,
with identical exposure support. Maximum relative nonzero exposure differences
are 1.80×10⁻¹⁵ for the small full-replay control and 1.70×10⁻¹⁵ for full617.
The low-coverage NiO cut and fringes retain rounding-level signal and uncertainty
agreement. Input projects and original source identities remain unchanged.
The final guard receipts and comparison summaries are in the JSON report's
`workload_guard` section. Original full receipts remain on the cluster.

Scalar stage receipts, module/harness hashes and full-grid comparisons are in
[dgs-symmetry-cache-node19.json](dgs-symmetry-cache-node19.json). Scientific
archives remain in each experiment's `shared/nfit/benchmarks/6A-P-node19` folder:

- SEQUOIA: `seq-full617-cached12-v011604-control-02` and
  `seq-full617-cached12-v011605-symmetry-reuse`.
- HYSPEC: `hys34-full361-cached12-v011604-control` and
  `hys34-full361-cached12-v011605-symmetry-reuse`.

Source-candidate jobs use frozen module hashes in the existing desktop runtime;
its installed release metadata is 0.116.4 during these candidate measurements.
No separate application installation is made in an experiment folder.
