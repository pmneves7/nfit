# Fractional point binning: correctness and speed

Measured on 2026-10-04 with an Apple M3 Max, 16 CPUs and 128 GiB RAM.
The benchmark compares nfit 0.116.16 with 0.116.15 (`8a530e9`). Platform,
Python, NumPy, Numba and all seven alternating timed repetitions are in
[fractional-point-binning.json](fractional-point-binning.json).

## Numerical treatment

General fractional assignment remains the tensor product of linear weights
between physical grid centers. Each accepted source point has total spatial
weight one. Boundary support saturates into an end bin; integrated axes have
one contribution. Normalized means deposit numerator and averaging weight
consistently. Deterministic sharing squares the complete coefficient for each
bin's variance. Neighboring-bin covariance is deliberately omitted.

Corrections include exact-center ghost support, uniform discrete membership at
physical edges and their adjacent representable values, nonfinite-coordinate
filtering, empty stream batches during automatic-limit discovery, and fully
explicit grids containing no finite source coordinates. Exposure-weighted
point means now retain deposited physical normalization through GUI and
scripting rebins; inverse-variance weights are not relabeled as exposure.
The general point/histogram numerical cache version changes from 1 to 2.
Native raw-DGS, MDE and CORELLI reduction cache signatures are unaffected.

An independent scalar oracle checks 1D, 2D, 3D, 5D and 7D, both averaging
policies, normalized means and sums, mixed/nonuniform grids, boundary and
center identity, integrated axes, invalid inputs, streaming, and dense/sparse
parallel partials. Separate integration tests cover exposure preservation and
per-axis fractional policy through project and scripting consumers.

## Timing scope

The deterministic fixtures contain 500,000 weighted observations, except the
rotating-area-detector fixture with 495,616 points. This fixture maps elastic
momentum transfer from a curved detector surface through 16 sample rotations.
Signals and source uncertainties are generated independently of coordinates;
it tests computation speed and parity, not reconstruction accuracy against an
unknown experimental intensity field. Uniform grids use 24 centers per axis;
the curved grid uses 80. The seven-dimensional case has three resolved axes
and four one-bin integrations. The streamed case uses 100,000-point batches.

Timers include the full public call: grid preparation, output allocation,
bounded accumulation and finalization. Source generation, initial JIT/cache
loading, scientific file I/O and instrument reduction are excluded. Each path
is warmed, then timed seven times in alternating order with BLAS limited to
one thread. Requested worker counts are 1 or 4. Complete signal, diagonal
uncertainty, sample contribution and normalization arrays agree with the old
implementation at `rtol=atol=1e-12`, with equal NaN masks on these fixtures.
Independent tests exercise the intentional boundary/center correctness changes.

## Median wall times

| Fixture | Workers | 0.116.15 (ms) | 0.116.16 (ms) | Speedup |
|---|---:|---:|---:|---:|
| uniform_3d | 1 | 35.570 | 30.138 | 1.180× |
| uniform_3d | 4 | 15.955 | 8.868 | 1.799× |
| uniform_4d | 1 | 71.882 | 56.345 | 1.276× |
| uniform_4d | 4 | 44.100 | 26.335 | 1.675× |
| mixed_4d | 1 | 46.876 | 40.264 | 1.164× |
| mixed_4d | 4 | 33.958 | 23.398 | 1.451× |
| nonuniform_3d | 1 | 202.496 | 75.696 | 2.675× |
| nonuniform_3d | 4 | 214.673 | 24.747 | 8.675× |
| integrated_7d | 1 | 359.760 | 31.381 | 11.464× |
| integrated_7d | 4 | 113.543 | 10.887 | 10.429× |
| curved_detector_3d | 1 | 35.399 | 24.445 | 1.448× |
| curved_detector_3d | 4 | 23.044 | 13.808 | 1.669× |
| streamed_3d | 1 | 28.446 | 31.616 | 0.900× |
| streamed_3d | 4 | 11.013 | 9.567 | 1.151× |

Explicit in-memory grids previously fell back to NumPy even when the caller
requested Numba. They now use bounded coordinate encoding and compiled
accumulation, including parallel workers. The 8.7× four-worker gain therefore
includes enabling that parallel compiled path. Integrated axes omit redundant
neighbor combinations in both backends. Uniform accumulation computes flat
neighbor offsets once per batch and axis locations once per point. Explicit
bounds/edges avoid full-source limit scans; identity stream projections avoid
unnecessary matrix products.

The single-worker streamed 3D fixture is 11% slower (28.446 to 31.616 ms),
including the added exact physical-center and edge corrections. Its four-worker
version is 15% faster. Full in-memory 3D/4D calls improve by 18–28% with one
worker and 67–80% with four. The curved detector fixture improves by 45–67%.
Alternative stencil expansions and unrolled deposits were measured and rejected
because their speed was worse. These are warm local point-binning measurements,
not measured SEQUOIA/HYSPEC or Mantid end-to-end speedups.

Reproduce from the repository with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_fractional_binning.py \
  --reference 8a530e9 --points 500000 --workers 1 4 > fractional.json
```
