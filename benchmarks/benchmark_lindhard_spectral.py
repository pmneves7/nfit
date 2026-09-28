"""Compare a benchmark-only spectral FFT prototype with production Lindhard.

Run with the project interpreter from the repository root. Timings exclude
imports and JIT startup, use one BLAS thread (up to four Numba workers), and include fresh electronic
preparation unless explicitly labeled as reusing a transition table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import scipy
from lindhard_spectral_candidate import prepare_transitions, spectral_sum
from threadpoolctl import threadpool_info, threadpool_limits

from nfit import (
    ElectronicResponseCache,
    bare_lindhard_susceptibility,
    build_electronic_model,
    k_mesh,
)
from nfit.electronic_response import spin_operator_matrices

CASES = {
    "short": (12, 48, 3, 0.5, False),
    "scan": (6, 48, 501, 0.5, False),
    "dense": (12, 48, 2001, 0.5, False),
    "narrow": (12, 32, 1001, 0.05, False),
    "tensor": (6, 32, 501, 0.5, True),
}


def make_model(bands, explicit_spin):
    rng = np.random.default_rng(20260928 + bands)

    def random_matrix():
        return (rng.normal(size=(bands, bands)) + 1j * rng.normal(size=(bands, bands))) / np.sqrt(bands)

    onsite = random_matrix()
    spin = None
    if explicit_spin:
        pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        spin = np.array([np.kron(np.eye(bands // 2), item / 2) for item in pauli])
        centers = np.repeat(np.linspace(0, 0.3, bands // 2), 2)
    else:
        centers = np.linspace(0, 0.3, bands)
    return build_electronic_model(
        direct_lattice=np.diag([3, 4, 8]), basis=[str(i) for i in range(bands)],
        hoppings={(0, 0, 0): 3 * (onsite + onsite.conj().T),
                  (1, 0, 0): 5 * random_matrix(), (0, 1, 0): 4 * random_matrix()},
        orbital_centers=np.column_stack((centers, centers * 0, centers * 0)),
        periodic_axes=(0, 1), spin_operators=spin, energy_unit="meV",
    )


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append(perf_counter() - start)
    return result, {"median_seconds": float(np.median(samples)), "samples_seconds": samples}


def error_metrics(reference, approximate):
    error = approximate - reference
    report = {}
    for name, ref, diff in (("complex", reference, error),
                            ("real", reference.real, error.real),
                            ("imag", reference.imag, error.imag)):
        report[f"{name}_max_absolute"] = float(np.max(np.abs(diff)))
        report[f"{name}_peak_normalized"] = float(np.max(np.abs(diff)) / max(np.max(np.abs(ref)), 1e-30))
    return report


def benchmark_case(name, repeats):
    bands, mesh_size, n_energy, eta, explicit_spin = CASES[name]
    model = make_model(bands, explicit_spin)
    mesh = k_mesh(model, [mesh_size, mesh_size])
    Q = np.array([1.173, 0.219, 0.0])
    energy = np.linspace(-30, 30, n_energy)
    basis, matrices, prefactor = spin_operator_matrices(model, Q)
    thermodynamics = {"temperature_K": 20.0, "chemical_potential_meV": 0.0}

    def reference(engine, width=eta, cache=None):
        backend, workers = {"numpy_1": ("numpy", 1), "numba_1": ("numba", 1), "numba_4": ("numba", 4)}[engine]
        response = bare_lindhard_susceptibility(
            model, np.tile(Q, (n_energy, 1)), energy, mesh, basis,
            operator_matrices_by_point=np.broadcast_to(matrices, (n_energy, *matrices.shape[1:])),
            **thermodynamics, broadening_meV=width, backend="numpy", workers=workers,
            transition_backend=backend, cache=cache, q_evaluation="direct",
        )
        return prefactor * response.values_per_meV_cell

    def prepare():
        return prepare_transitions(model, mesh, Q, matrices[0], **thermodynamics)

    references = {}
    # JIT/import costs are excluded for both engines; no completed-response cache.
    for backend in ("numpy_1", "numba_1", "numba_4"):
        reference(backend)
        exact, timing = measure(lambda backend=backend: reference(backend), repeats)
        references[backend] = timing
    best = min(references, key=lambda key: references[key]["median_seconds"])
    table, preparation = measure(prepare, repeats)
    variants = []
    for density in (4, 8, 16, 32):
        spectral_sum(table, energy, eta, bins_per_width=density)
        (candidate, memory), timing = measure(
            lambda density=density: spectral_sum(table, energy, eta, bins_per_width=density), repeats,
        )
        candidate *= prefactor

        def fresh(density=density):
            return spectral_sum(prepare(), energy, eta, bins_per_width=density)

        _, full = measure(fresh, repeats)
        record = {"bins_per_width": density, "evaluation": timing, "fresh_total": full,
                  "fresh_speedup_vs_best_direct": references[best]["median_seconds"] / full["median_seconds"],
                  **memory, "errors": error_metrics(exact, candidate)}
        # A scalar RPA stress check; not a certified stable full-zone model.
        if not explicit_spin:
            static = float(exact[n_energy // 2, 0, 0].real)
            if static > 0:
                record["rpa_checks"] = []
                for static_margin in (0.1, 0.01):
                    vertex = (1 - static_margin) / static
                    denominator = 1 - vertex * exact
                    dressed = exact / denominator
                    dressed_candidate = candidate / (1 - vertex * candidate)
                    record["rpa_checks"].append({
                        "requested_static_margin": static_margin,
                        "minimum_sampled_denominator_magnitude": float(np.min(np.abs(denominator))),
                        "errors": error_metrics(dressed, dressed_candidate),
                    })
        variants.append(record)
    # Distinct eta values defeat the completed-response cache. Include normal
    # eigensystem caching on the direct path; reuse transitions on the prototype.
    widths = [eta * value for value in (0.8, 1.0, 1.2)]
    cache = ElectronicResponseCache()
    reference(best, width=eta * 0.9, cache=cache)
    direct_seconds, fft_seconds, reuse_errors = [], [], []
    for i in range(repeats):
        for width in widths:
            width *= 1 + i * 0.01
            start = perf_counter()
            ref = reference(best, width=width, cache=cache)
            direct_seconds.append(perf_counter() - start)
            start = perf_counter()
            value, _ = spectral_sum(table, energy, width, bins_per_width=16)
            fft_seconds.append(perf_counter() - start)
            reuse_errors.append(error_metrics(ref, prefactor * value)["complex_peak_normalized"])
    return {
        "case": name, "bands": bands, "operators": len(matrices[0]), "mesh": [mesh_size, mesh_size],
        "energies": n_energy, "energy_interval_meV": [-30, 30], "Q_reduced": Q.tolist(),
        "broadening_meV": eta, **thermodynamics, "direct": references, "best_direct_engine": best,
        "prepare_transitions": preparation, "retained_transition_bytes": table.nbytes,
        "variants": variants, "width_scan": {
            "direct_seconds": direct_seconds, "spectral_seconds": fft_seconds,
            "median_speedup": float(np.median(direct_seconds) / np.median(fft_seconds)),
            "max_peak_normalized_error": max(reuse_errors),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/lindhard-spectral.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    with threadpool_limits(limits=1):
        report = {"environment": {"platform": platform.platform(), "machine": platform.machine(),
                                   "python": platform.python_version(), "numpy": np.__version__,
                                   "scipy": scipy.__version__, "numba": numba.__version__,
                                   "logical_cpus": os.cpu_count(), "native_threads": threadpool_info()},
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__), Path(__file__).with_name("lindhard_spectral_candidate.py"))},
                  "repeats": args.repeats, "cases": []}
        for case in args.cases:
            row = benchmark_case(case, args.repeats)
            report["cases"].append(row)
            print(json.dumps({"case": case, "best_direct": row["best_direct_engine"],
                              "variants": [{"density": v["bins_per_width"], "speedup": v["fresh_speedup_vs_best_direct"],
                                            "error": v["errors"]["complex_peak_normalized"]} for v in row["variants"]]}), flush=True)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["process_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
