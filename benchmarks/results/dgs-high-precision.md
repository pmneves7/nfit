# Fused high precision DGS event binning

The manual [benchmark](../benchmark_dgs_high_precision.py) compares the old
NumPy projection plus ordered compiled accumulator with fused float64 projection
and accumulation. Both use the same events, weights, variance, axes, UB,
symmetry operations and requested physical edges. No Mantid is imported.

## Implementation and numerical checks

The compiled HKLE path retains the three transform stages and source accumulation
order, with fast-math disabled. It handles uniform and arbitrary physical edges;
uniform estimates are corrected against saved boundaries. The compiled general
and fused event accumulators share one physical membership implementation.
There are no event-sized projected-coordinate temporaries and no per-worker
histogram copies. The existing projection path remains the fallback without JIT.

Tests cover layouts/strides, masks, signed unequal weights, supplied variances,
streaming, exact/adjacent edges, nonfinite inputs, one/six/twelve symmetry copies,
within-bin copy covariance, irregular grids, fallback and module ownership.
High-precision histograms record a projection version: rebuilding them reuses
current reduced events. Scalar and BLAS dot products can differ by roundoff;
this does not promise identical membership for every hypothetical point exactly
on a transformed boundary.

## Measured costs

The HYSPEC input is the full `Ei15meV_50K_240Hz_s2_70.nxs`: 361 experiments,
49,415,626 stored events, approximately 2.2 GiB. Kernel measurements select its
first 600,000 events and alternate seven warm passes. Initial compilation/cache
loading and source loading are recorded separately. Warm kernel timing excludes
histogram clearing, trajectory normalization, loading and saving.

| Host and scope | Previous | Fused | Improvement |
| --- | ---: | ---: | ---: |
| ORNL node19, full HYSPEC MDE bin/save/reopen/view, reference first | 68.92 s | 35.05 s | 1.97× |
| ORNL node19, same workflow, fused first | 64.88 s | 38.13 s | 1.70× |
| Local Mac, full workflow, reference first | 24.22 s | 11.75 s | 2.06× |
| Local Mac, same workflow, fused first | 21.04 s | 12.66 s | 1.66× |

Warm HYSPEC event kernels improve 2.43–2.50× for one symmetry copy,
2.93–3.01× for six, and 2.84–3.01× for twelve.

The SEQUOIA cube has 67×134×101×101 cells. On its first 600,000 stored events,
five alternating warm passes give 1.70× / 1.71× / 1.67× gains for one/six/twelve
copies (76.0→44.6 ms, 440.6→257.2 ms, and 875.6→523.3 ms). All event numerators,
variance numerators and counts match exactly. The larger dense grid has a
different memory-access cost; this is a kernel measurement, not a full SEQUOIA
raw-reduction or saved-project workflow.

The HYSPEC grid has 40×50×25×57 cells and twelve symmetry operations. The
full-workflow intervals include source metadata, all events, trajectory exposure,
histogram construction, native project save, metadata reopen and cached
histogram access. They do not include fresh raw reduction or an existing
original scientific project. Filesystem caches are uncontrolled; reverse-order
controls show the gain persists. BLAS is limited to one thread. ORNL uses the
existing desktop runtime and staged source on analysis-node19. HYSPEC and
SEQUOIA scientific storage stays within their IPTS directories. The recorded
HYSPEC control finished before the SEQUOIA benchmark started.

For the warm 600,000-event pass, tracked temporary allocation falls from about
50.36 MiB to about 2 KiB. The input array and three common histogram accumulators
remain allocated in both paths. This is a temporary NumPy/Python allocation
measurement, not total RSS or a claim about whole-project memory.

Event numerator, event variance numerator, count and mask arrays are exact in
both full HYSPEC comparisons. All finite exposure, intensity and error cells,
including fringes, agree to at most 2.7×10⁻¹⁵ relative on node19. Their differences
come from ordinary parallel trajectory summation; the event numerators are
identical. Source size and modification time remain unchanged.

Local scalar receipts are [dgs-high-precision-macos.json](dgs-high-precision-macos.json).
A bounded [node19 summary](dgs-high-precision-node19.json) records the reported
scalar results. Full ORNL receipts and new diagnostic projects remain under
`/SNS/HYS/IPTS-36860/shared/nfit/benchmarks/high-precision-20261004/`.
The SEQUOIA kernel diagnostic uses the existing `(1,1,1)`, `(1,-1,0)`,
`(1,1,-2)` cube settings and remains under its matching IPTS benchmark directory.

These measurements establish a substantial high precision HKLE binning gain.
They do not imply that raw reduction, detector-trajectory normalization or every
other plotting/rebinning operation accelerates by the same factor.
