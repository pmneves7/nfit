# Numerical benchmarks

## Lindhard acceleration

The three Lindhard scripts compare existing exact CPU backends, shifted-q
eigensystem reuse, and an isolated approximate spectral FFT experiment:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_lindhard_dispatch.py
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_lindhard_cache.py
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_lindhard_spectral.py
```

Run them separately to avoid CPU contention. They exclude import/JIT startup,
use one native BLAS thread, and record repeated timings and numerical errors.
See [the measured results and limitations](results/lindhard-acceleration.md).
`lindhard_spectral_candidate.py` is a benchmark experiment, not a selectable
production backend. Its dense transition storage is intended for small and
moderate operator bases.

## Rebinning

`benchmark_rebin.py` measures preparation, accumulation, normalization, total
wall time, and process peak RSS for a representative fractional 4D rebin.

```bash
PYTHONPATH=src python benchmarks/benchmark_rebin.py
PYTHONPATH=src python benchmarks/benchmark_rebin.py --sizes 100 500000 50000000
PYTHONPATH=src python benchmarks/benchmark_rebin.py --sizes 500000 --backend numpy
PYTHONPATH=src python benchmarks/benchmark_rebin.py --sizes 5000000 --workers 4
```

Run separate invocations for worker scaling (the `--workers` option accepts one
value per run), and vary `--dimensions`, `--bins-per-dim`, `--nonfractional`,
and `--strategy dense|sparse`. The output includes effective backend, reduction
strategy, workers, phase timings, throughput, and peak RSS.
Use `--coordinate-span` to create a deliberately low-occupancy region when
measuring sparse reduction; random coordinates over the full output range are
not a sparse workload merely because the output grid is large.

The 50-million-point tier is intentionally opt-in because its generated source
arrays require several gigabytes. Compare backends in separate processes: peak
RSS is a process-lifetime high-water mark, and the first Numba run may include
JIT compilation. Record CPU model, available cores, memory, operating system,
Python, NumPy, and Numba versions with published results.

`benchmark_rebin_modes.py` compares discrete assignment, fractional momentum
with discrete energy, and fractional assignment on all four axes. It uses the
same source and output basis and grid for every mode. Repeated timings separate
the first dispatch from resident-kernel performance:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_rebin_modes.py \
  --shape 96,80,64,24 --workers 8 --batch-mb 192 --repeats 3
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_rebin_modes.py \
  --project /absolute/path/to/project.nfit --member assets/binnings/example/data.npz \
  --workers 8 --batch-mb 192 --coarsen 2 --repeats 3
```

Run worker counts and batch sizes in separate processes. JSON lines report
source loading separately, per-mode elapsed time, resolved backend and workers,
output summaries, and process peak RSS. Fractional and discrete results differ
scientifically; only comparisons of the same mode should require numerical
equivalence. Native raw-DGS and MDE event histograms already assign events
discretely, so this comparison measures subsequent histogram rebinning.

See [SEQUOIA reduction and binning measurements](results/sequoia-performance.md).

`benchmark_event_accumulation.py` compares the ordered NumPy reference with the
compiled event accumulator, with exact output checks and compilation excluded
from repeated timings:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_event_accumulation.py --span 4
/Users/pmneves/anaconda3/envs/nfit/bin/python benchmarks/benchmark_event_accumulation.py --span 1.35
```

The default million-event workload uses a 64 × 64 × 64 × 48 grid and several
hundred MiB of arrays. Use `--events` and `--shape` to reduce it for smaller
machines. This measures event accumulation, excluding reduction, coordinate
projection, normalization, and archive reading.

## Histogram uncertainty diagnostics

`benchmark_histogram_uncertainty.py` compares nfit pooling against independent
Poisson count/exposure references. It reports unequal exposure, covered empty
versus unmeasured cells, independent background subtraction, and seeded sampling
variance/coverage. It separately models the historical nfit 0.105.7 DGS empty-cell confidence
limit substitution. nfit 0.106.0 stores accumulated event variance separately
from confidence intervals; the baseline diagnostic does not validate the old
substitution policy.

```bash
PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
  benchmarks/benchmark_histogram_uncertainty.py \
  --output /tmp/nfit_uncertainty_reference.json
```

The reference assumes known exposure and independent unit-weight events. It does
not establish low-count Gaussian coverage or shared-source covariance. The
[temporary implementation plan](../docs/measurement_pipeline_plan.md) records
review gates and the subsequent production work.

For matched 4D diagnostic slabs, supply all three paths:

```bash
PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
  benchmarks/benchmark_histogram_uncertainty.py \
  --nfit-slab /path/to/nfit.npz --mantid-slab /path/to/mantid.npz \
  --mantid-data /path/to/mantid-numerator.npz \
  --output /tmp/nfit_matched_uncertainty.json
```

Both normalized slabs require `signal`, `norm`, `counts`, `errors`, `numerator`,
and `edges0` through `edges3`. The Mantid count-space slab requires `numerator`
and `variance`. Arrays must have identical 4D shape and edges must match within
the reported float32 rounding tolerance. The default aggregation collapses axes
0 and 2; `--aggregation-axes` changes this. The central-region diagnostic assumes
axis 1 is [K,-K,0] in r.l.u. and axis 3 is energy transfer in meV.

The supplied-slab report compares `(nfit_error * nfit_norm)**2` with Mantid's
count-space variance, isolating exposure differences before comparing errors.
It records SHA256 checksums and grid differences and does not run Mantid.
[The historical NiO baseline](results/histogram-uncertainty-baseline.json)
contains aggregate statistics only; source slabs remain outside the repository.

`benchmark_uncertainty_paths.py` traces actual native hidden-axis pooling,
regular and rotated box cuts, display coarsening, and ROI sums. It classifies
geometric fringes and low exposure separately, reports coverage disagreement
before common-support filtering, and retains fine-cell comparisons even when
hidden-axis redistribution cancels in the map. Provide the same NPZ fields as
above; optional `mask` arrays are honored. Axes must be H, K, L, energy:

```bash
PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
  benchmarks/benchmark_uncertainty_paths.py \
  --nfit-slab /path/to/nfit.npz --mantid-slab /path/to/mantid.npz \
  --mantid-data /path/to/mantid-numerator.npz \
  --normalization-storage auxiliary --roi 0.13,0.23,3.5,25 \
  --output /tmp/nfit_uncertainty_paths.json
```

Use `metadata` for historical raw DGS histograms that lacked a normalization
channel, and `auxiliary` for current event histograms and historical MDE
histograms with that channel. The helper holds nfit exposure and intensity
fixed while changing only the variance prescription. Its floor-removal variant
is a diagnostic for unsubtracted event histograms, not a general correction.
Its JSON contains full maps and profiles and can include private derived data;
the committed [NiO fringe report](results/nio-uncertainty-fringes.md) retains
aggregate metrics only. Neither diagnostic nor its unit tests imports or runs
Mantid or Shiver. External engine measurements are separate manual runs.

For nfit 0.106.0 and later event-statistics slabs, add `--event-statistics`
to the path diagnostic. It preserves the additive numerator, variance, and
normalization contract through viewer coarsening. Omitting the flag traces
the historical histogram behavior. Ordinary inverse-variance box cuts and
covariance between different output bins remain separate diagnostic limits.

See [previous DGS reference measurements](results/dgs-reference-current.md)
for historical manual matched-MDE thin/cube binning, final fresh native reduction,
standard sequential Mantid/Shiver reduction estimates, calibration audits,
and fringe uncertainty diagnostics. The report separates measured phase times
from extrapolations and incomplete converter/covariance parity.

See [final SEQUOIA DGS parity validation](results/dgs-parity-final.md) for the
nfit 0.107.0 manual comparison with Mantid: identical per-cell event moments
from both common MDE and fresh raw data, quantified exposure residuals, fine
low-coverage diagnostics, transient reduced-event cache reuse, and current
measured timings versus the estimated ordinary sequential Mantid workflow.
The aggregate receipts exclude full event and diagnostic slab payloads.
