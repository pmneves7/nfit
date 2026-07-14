# Rebin benchmarks

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
