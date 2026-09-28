"""Benchmark warm Lindhard calls with and without shifted-q cache reuse.

Run with the repository's ``nfit`` environment::

    /Users/pmneves/anaconda3/envs/nfit/bin/python \
        benchmarks/benchmark_lindhard_cache.py

The legacy cache emulates the former q-parallel path: it caches the base
eigensystem, but never stores shifted-q eigensystems. Both paths retain the
same cold-path q batching and response contraction.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import nfit.electronic_response as response_module
from nfit import (
    BasisState,
    ElectronicResponseCache,
    bare_spin_susceptibility,
    build_electronic_model,
    k_mesh,
)
from nfit.cache_utils import array_digest
from nfit.electronic_spin import spin_half_operators


class _LegacyShiftCache(ElectronicResponseCache):
    """Keep base eigensystem caching but discard every shifted eigensystem."""

    def __init__(self, *, base_coordinate_digest: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base_coordinate_digest = base_coordinate_digest

    def get(self, key: tuple[Any, ...]) -> Any | None:
        if (
            len(key) > 2
            and key[0] == "eigensystem"
            and key[2] != self._base_coordinate_digest
        ):
            return None
        return super().get(key)

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        if (
            len(key) > 2
            and key[0] == "eigensystem"
            and key[2] != self._base_coordinate_digest
        ):
            return
        super().put(key, value)


@contextlib.contextmanager
def _legacy_no_copy(enabled: bool):
    """Emulate the pre-fix view-only split in legacy timing samples."""

    original = response_module._owned_eigensystem
    if enabled:
        response_module._owned_eigensystem = lambda values: values
    try:
        yield
    finally:
        response_module._owned_eigensystem = original


def _model(bands: int):
    if bands % 2:
        raise ValueError("benchmark band count must be even")
    orbitals = bands // 2
    basis = [
        BasisState(
            f"o{index}_{spin}",
            site="A",
            orbital=f"o{index}",
            spin=spin,
        )
        for index in range(orbitals)
        for spin in ("up", "down")
    ]
    diagonal = np.linspace(-2.0, 2.0, orbitals)
    onsite = np.kron(np.diag(diagonal), np.eye(2))
    nearest_orbital = np.diag(np.linspace(-5.0, -2.0, orbitals))
    for index in range(orbitals - 1):
        nearest_orbital[index, index + 1] = 0.18
    nearest = np.kron(nearest_orbital, np.eye(2))
    transverse = np.kron(np.diag(np.linspace(-1.1, -0.4, orbitals)), np.eye(2))
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 3.0, 9.0]),
        basis=basis,
        hoppings={
            (0, 0, 0): onsite,
            (1, 0, 0): nearest,
            (0, 1, 0): transverse,
        },
        orbital_centers=np.zeros((bands, 3)),
        periodic_axes=(0, 1),
        spin_operators=spin_half_operators(orbitals),
        energy_unit="meV",
    )


def _workload(bands: int, mesh_shape: tuple[int, int], energies: int):
    model = _model(bands)
    mesh = k_mesh(model, mesh_shape)
    q_values = np.asarray(
        [
            [0.137, 0.213, 0.0],
            [0.281, 0.347, 0.0],
            [0.413, 0.159, 0.0],
            [0.091, 0.431, 0.0],
        ],
        dtype=float,
    )
    energy_grid = np.linspace(-8.0, 8.0, energies)
    q = np.repeat(q_values, energies, axis=0)
    energy = np.tile(energy_grid, q_values.shape[0])
    return model, mesh, q, energy


def _one_run(
    *,
    model: Any,
    mesh: Any,
    q: np.ndarray,
    energy: np.ndarray,
    legacy: bool,
    warm_calls: int,
    cache_bytes: int,
    expect_q_parallel: bool,
) -> tuple[list[float], np.ndarray, int, bool]:
    cache_arguments: dict[str, Any] = {
        "max_bytes": cache_bytes,
        "max_entries": warm_calls + 16,
    }
    if legacy:
        cache: ElectronicResponseCache = _LegacyShiftCache(
            base_coordinate_digest=array_digest(
                mesh.reduced_coordinates,
                np.float64,
            ),
            **cache_arguments,
        )
    else:
        cache = ElectronicResponseCache(**cache_arguments)
    settings = {
        "temperature_K": 30.0,
        "chemical_potential_meV": 0.1,
        "workers": 2,
        "max_batch_bytes": 64 * 1024**2,
        "transition_max_batch_bytes": 64 * 1024**2,
        "backend": "numpy",
        "transition_backend": "numpy",
        "cache": cache,
    }
    with _legacy_no_copy(legacy):
        reference = bare_spin_susceptibility(
            model,
            q,
            energy,
            mesh,
            broadening_meV=0.5,
            **settings,
        )
        q_parallel_applied = bool(
            reference.provenance.get("q_parallel_execution", {}).get("applied")
        )
        if q_parallel_applied != expect_q_parallel:
            raise RuntimeError(
                "benchmark q-parallel activation differed from expectation: "
                f"expected {expect_q_parallel}, got {q_parallel_applied}"
            )
        samples = []
        last_values = reference.values_per_meV_cell
        for index in range(warm_calls):
            started = time.perf_counter()
            result = bare_spin_susceptibility(
                model,
                q,
                energy,
                mesh,
                broadening_meV=0.7 + 0.013 * index,
                **settings,
            )
            samples.append(time.perf_counter() - started)
            last_values = result.values_per_meV_cell
    return samples, last_values, cache.entries, q_parallel_applied


def run_benchmark(*, rounds: int = 3, warm_calls: int = 5) -> dict[str, Any]:
    workloads = (
        ("q-parallel-cache-dominated", 48, (32, 32), 3, True),
        ("q-parallel-long-energy-scan", 12, (32, 32), 49, True),
    )
    records = []
    for label, bands, mesh_shape, energy_count, expect_q_parallel in workloads:
        model, mesh, q, energy = _workload(
            bands,
            mesh_shape,
            energy_count,
        )
        cache_bytes = 256 * 1024**2
        timings: dict[str, list[float]] = {"legacy": [], "cached": []}
        outputs: dict[str, np.ndarray] = {}
        entries: dict[str, int] = {}
        q_parallel_applied = False
        # Alternate order to reduce systematic first-run and thermal bias.
        for round_index in range(rounds):
            modes = ("legacy", "cached")
            if round_index % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                samples, values, entry_count, q_parallel_applied = _one_run(
                    model=model,
                    mesh=mesh,
                    q=q,
                    energy=energy,
                    legacy=mode == "legacy",
                    warm_calls=warm_calls,
                    cache_bytes=cache_bytes,
                    expect_q_parallel=expect_q_parallel,
                )
                timings[mode].extend(samples)
                outputs[mode] = values
                entries[mode] = entry_count
        legacy_median = statistics.median(timings["legacy"])
        cached_median = statistics.median(timings["cached"])
        records.append(
            {
                "name": label,
                "orbitals_and_bands": bands,
                "mesh_shape": list(mesh_shape),
                "mesh_points": int(mesh.reduced_coordinates.shape[0]),
                "offmesh_q_values": int(np.unique(q, axis=0).shape[0]),
                "paired_energy_points": int(energy.size),
                "workers": 2,
                "native_thread_limit_per_q_worker": 1,
                "q_parallel_applied": q_parallel_applied,
                "rounds": rounds,
                "warm_calls_per_round": warm_calls,
                "cache_bytes": cache_bytes,
                "legacy_warm_median_ms": 1000.0 * legacy_median,
                "cached_warm_median_ms": 1000.0 * cached_median,
                "speedup": legacy_median / cached_median,
                "legacy_cache_entries": entries["legacy"],
                "cached_cache_entries": entries["cached"],
                "max_abs_response_difference": float(
                    np.max(np.abs(outputs["legacy"] - outputs["cached"]))
                ),
                "legacy_samples_ms": [1000.0 * value for value in timings["legacy"]],
                "cached_samples_ms": [1000.0 * value for value in timings["cached"]],
            }
        )
    return {
        "benchmark": "warm Lindhard eta change with offmesh q-parallel response",
        "cache_comparison": (
            "legacy mode caches the base eigensystem only; cached mode also "
            "retains shifted-q eigensystems under the normal bounded LRU"
        ),
        "q_parallel_threshold_override": False,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warm-calls", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/lindhard-cache.json"),
    )
    args = parser.parse_args()
    result = run_benchmark(rounds=max(1, args.rounds), warm_calls=max(1, args.warm_calls))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
