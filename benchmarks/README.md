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
