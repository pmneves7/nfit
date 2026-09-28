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
