"""Benchmark NumPy and Numba for the implicit scalar-spin Lindhard path.

The benchmark uses deterministic, dense multi-orbital tight-binding models on
a full uniform 2D mesh. It measures 1, 3, 101, and 401 paired energies at one
incommensurate Q, covering short requests and energy scans. Numba is measured
with one and four threads; NumPy uses one worker. The first call for each Numba
kernel is a warm-up and is excluded from measurements. Results are written as
JSON and include timing samples, backend agreement, and environment metadata.

Run from the repository root with the local environment::

    PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
      benchmarks/benchmark_lindhard_dispatch.py \
      --output benchmarks/results/lindhard-dispatch.json

The default cases can take several minutes. Use ``--bands`` and
``--energy-counts`` to run a smaller smoke case when changing this script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from nfit import BasisState, bare_spin_susceptibility, build_electronic_model, k_mesh

Q_REDUCED = np.array([0.137, 0.271, 0.0])
TEMPERATURE_K = 30.0
CHEMICAL_POTENTIAL_MEV = 7.0
BROADENING_MEV = 1.5
MESH_SHAPE = (32, 32)
WORKER_COUNTS = (1, 4)


def build_model(n_bands: int):
    """Create a reproducible dispersive model with weak orbital mixing."""
    rng = np.random.default_rng(20260928 + n_bands)
    basis = [
        BasisState(f"orbital-{index}", site="A", orbital=f"o{index}")
        for index in range(n_bands)
    ]
    onsite = np.diag(np.linspace(-48.0, 48.0, n_bands)).astype(np.complex128)
    random_onsite = rng.normal(size=(n_bands, n_bands)) + 1.0j * rng.normal(
        size=(n_bands, n_bands)
    )
    onsite += 1.5 * (random_onsite + random_onsite.conj().T) / 2.0

    def hopping(scale: float) -> np.ndarray:
        raw = rng.normal(size=(n_bands, n_bands)) + 1.0j * rng.normal(
            size=(n_bands, n_bands)
        )
        return scale * raw / np.sqrt(n_bands)

    return build_electronic_model(
        direct_lattice=np.diag([3.0, 4.0, 8.0]),
        basis=basis,
        hoppings={
            (0, 0, 0): onsite,
            (1, 0, 0): hopping(7.0),
            (0, 1, 0): hopping(5.0),
            (1, 1, 0): hopping(1.7),
        },
        orbital_centers=np.column_stack(
            [np.linspace(0.0, 0.25, n_bands), np.zeros(n_bands), np.zeros(n_bands)]
        ),
        periodic_axes=(0, 1),
        energy_unit="meV",
    )


def paired_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.repeat(Q_REDUCED[None, :], count, axis=0)
    energy = np.linspace(0.4, 9.0, count)
    return q, energy


def evaluate(model, mesh, q: np.ndarray, energy: np.ndarray, *, transition_backend: str, workers: int):
    return bare_spin_susceptibility(
        model,
        q,
        energy,
        mesh,
        temperature_K=TEMPERATURE_K,
        chemical_potential_meV=CHEMICAL_POTENTIAL_MEV,
        broadening_meV=BROADENING_MEV,
        backend="numpy",
        workers=workers,
        transition_backend=transition_backend,
        q_evaluation="direct",
        cache=None,
    )


def summarize_error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.abs(np.asarray(actual) - np.asarray(reference))
    scale = np.maximum(np.abs(reference), np.finfo(float).tiny)
    return {
        "max_absolute_complex_error": float(np.max(difference, initial=0.0)),
        "max_relative_complex_error": float(np.max(difference / scale, initial=0.0)),
    }


def environment() -> dict[str, object]:
    import numba

    physical_memory = None
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        pass
    thread_variables = {
        name: os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        )
        if name in os.environ
    }
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": physical_memory,
        "thread_environment": thread_variables,
    }


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def benchmark(*, bands: tuple[int, ...], energy_counts: tuple[int, ...], repeats: int) -> dict:
    cases = []
    for n_bands in bands:
        model = build_model(n_bands)
        mesh = k_mesh(model, MESH_SHAPE, symmetry="full")

        # Warm both Numba kernels on this model signature. JIT compilation,
        # imports, and model/mesh construction do not enter reported timings.
        warm_q, warm_energy = paired_points(2)
        for workers in WORKER_COUNTS:
            evaluate(
                model,
                mesh,
                warm_q,
                warm_energy,
                transition_backend="numba",
                workers=workers,
            )

        for count in energy_counts:
            q, energy = paired_points(count)
            reference = evaluate(
                model,
                mesh,
                q,
                energy,
                transition_backend="numpy",
                workers=1,
            ).values_per_meV_cell
            variants = []
            for backend, workers in (("numpy", 1), ("numba", 1), ("numba", 4)):
                samples = []
                result = None
                for _ in range(repeats):
                    started = perf_counter()
                    result = evaluate(
                        model,
                        mesh,
                        q,
                        energy,
                        transition_backend=backend,
                        workers=workers,
                    )
                    samples.append(perf_counter() - started)
                assert result is not None
                variants.append(
                    {
                        "transition_backend": backend,
                        "workers": workers,
                        "seconds": samples,
                        "median_seconds": float(np.median(samples)),
                        "minimum_seconds": float(np.min(samples)),
                        "resolved_transition_backends": result.provenance[
                            "transition_backend"
                        ],
                        **summarize_error(reference, result.values_per_meV_cell),
                    }
                )
            cases.append(
                {
                    "bands": n_bands,
                    "mesh_shape": list(MESH_SHAPE),
                    "mesh_points": int(mesh.reduced_coordinates.shape[0]),
                    "energy_count": count,
                    "unique_q_count": 1,
                    "q_reduced": Q_REDUCED.tolist(),
                    "energy_mev": [float(energy.min()), float(energy.max())],
                    "variants": variants,
                }
            )
    return {
        "benchmark": "implicit_scalar_lindhard_numpy_vs_numba",
        "seed": 20260928,
        "repeats": repeats,
        "temperature_K": TEMPERATURE_K,
        "chemical_potential_meV": CHEMICAL_POTENTIAL_MEV,
        "broadening_meV": BROADENING_MEV,
        "q_evaluation": "direct",
        "backend": "numpy eigensystem; transition contraction varied",
        "warmup": "both Numba serial and parallel kernels per band count; excluded",
        "eigensystem_cache": "warm per band count and Q; included state is reused during timing",
        "completed_response_cache": "disabled (cache=None) for every call",
        "native_blas_thread_limit": 1,
        "environment": environment(),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=parse_int_list, default=(1, 6, 12))
    parser.add_argument("--energy-counts", type=parse_int_list, default=(1, 3, 101, 401))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/lindhard-dispatch.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with threadpool_limits(limits=1, user_api="blas"):
        result = benchmark(
            bands=args.bands,
            energy_counts=args.energy_counts,
            repeats=args.repeats,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
