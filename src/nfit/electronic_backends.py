"""Bounded execution backends for electronic Hamiltonians and eigensystems."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _parallel

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover - declared dependency is unavailable
    _threadpool_limits = None

try:
    from . import _electronic_cupy as _CUPY_BACKEND
except Exception:  # pragma: no cover - CuPy or a GPU is unavailable
    _CUPY_BACKEND = None

ElectronicBackend = Literal["auto", "numpy", "threaded", "cupy"]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_BACKEND = os.environ.get("NFIT_ELECTRONIC_BACKEND", "auto").lower()
_DEFAULT_MAX_BATCH_BYTES = 256 * 1024**2
_THREADED_WORK_THRESHOLD = 5.0e7


@dataclass(frozen=True)
class ElectronicEigensystem:
    """Eigenvalues, optional eigenvectors, and resolved execution metadata."""

    eigenvalues: FloatArray
    eigenvectors: ComplexArray | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ElectronicBackendValidation:
    """Numerical comparison of one backend with the NumPy reference."""

    requested_backend: str
    resolved_backend: str
    points: int
    max_absolute_error_meV: float
    max_relative_error: float
    relative_tolerance: float
    absolute_tolerance_meV: float
    passed: bool
    reason: str = ""


def set_electronic_backend(name: str) -> None:
    """Select ``auto``, ``numpy``, ``threaded``, or optional ``cupy`` execution."""

    global _BACKEND
    choice = str(name).strip().lower()
    if choice not in {"auto", "numpy", "threaded", "cupy"}:
        raise ValueError(
            "electronic backend must be auto, numpy, threaded, or cupy"
        )
    _BACKEND = choice


def electronic_backend() -> str:
    """Return the configured electronic execution backend."""

    return _BACKEND


def available_electronic_backends() -> tuple[str, ...]:
    """Return electronic backends usable in the current process."""

    choices = ["numpy", "threaded"]
    if _CUPY_BACKEND is not None:
        choices.append("cupy")
    return tuple(choices)


@lru_cache(maxsize=16)
def _executor(workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=int(workers),
        thread_name_prefix="nfit-electronic",
    )


def _resolved_workers(workers: int | None) -> int:
    if workers is not None and int(workers) < 0:
        raise ValueError("workers must be nonnegative")
    return _parallel.bounded_worker_count(workers)


def _resolved_backend(
    requested: str | None,
    *,
    n_points: int,
    n_basis: int,
    workers: int,
) -> str:
    choice = _BACKEND if requested is None else str(requested).strip().lower()
    if choice not in {"auto", "numpy", "threaded", "cupy"}:
        raise ValueError(
            "electronic backend must be auto, numpy, threaded, or cupy"
        )
    if choice == "numpy":
        return "numpy"
    if choice == "threaded":
        return "threaded" if workers > 1 else "numpy"
    if choice == "cupy":
        return "cupy" if _CUPY_BACKEND is not None else "numpy"
    # Keep automatic execution on the CPU reference implementations. CuPy
    # remains explicit so selecting a GPU is an intentional, validated choice.
    work = float(n_points) * float(n_basis) ** 3
    if workers > 1 and work >= _THREADED_WORK_THRESHOLD:
        return "threaded"
    return "numpy"


def _batch_size(
    n_points: int,
    n_basis: int,
    *,
    eigenvectors: bool,
    max_batch_bytes: int,
) -> int:
    if max_batch_bytes < 1:
        raise ValueError("max_batch_bytes must be positive")
    matrix_bytes = 16 * n_basis * n_basis
    output_bytes = 8 * n_basis
    if eigenvectors:
        output_bytes += 16 * n_basis * n_basis
    per_point = max(1, matrix_bytes + output_bytes)
    return max(1, min(int(n_points), int(max_batch_bytes) // per_point))


def _numpy_chunk(
    model: Any,
    coordinates: FloatArray,
    *,
    eigenvectors: bool,
) -> tuple[FloatArray, ComplexArray | None]:
    matrices = model.hamiltonian(coordinates)
    if eigenvectors:
        values, vectors = np.linalg.eigh(matrices)
        return values, vectors
    return np.linalg.eigvalsh(matrices), None


def _numpy_matrices_chunk(
    matrices: ComplexArray,
    *,
    eigenvectors: bool,
) -> tuple[FloatArray, ComplexArray | None]:
    """Diagonalize already-constructed matrices without model evaluation."""

    if eigenvectors:
        values, vectors = np.linalg.eigh(matrices)
        return values, vectors
    return np.linalg.eigvalsh(matrices), None


def _chunk_ranges(
    n_points: int,
    chunk_size: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + chunk_size, n_points))
        for start in range(0, n_points, chunk_size)
    )


def evaluate_eigensystem(
    model: Any,
    reduced_k: ArrayLike,
    *,
    eigenvectors: bool,
    backend: ElectronicBackend | str | None = None,
    workers: int | None = None,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
) -> ElectronicEigensystem:
    """Evaluate a full-precision eigensystem in deterministic bounded chunks.

    The NumPy and threaded paths call the same ``numpy.linalg`` routines on
    each independent Hermitian matrix. Threading only changes which leading
    coordinate chunks execute concurrently; results are concatenated in input
    order. CuPy is an explicit opt-in backend using complex128/float64.
    """

    coordinates = np.asarray(reduced_k, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("reduced_k must have shape (n_k, 3) and be finite")
    n_points = int(coordinates.shape[0])
    if n_points < 1:
        raise ValueError("reduced_k must contain at least one wavevector")
    n_basis = int(model.n_basis)
    worker_count = _resolved_workers(workers)
    resolved = _resolved_backend(
        backend,
        n_points=n_points,
        n_basis=n_basis,
        workers=worker_count,
    )
    batch = _batch_size(
        n_points,
        n_basis,
        eigenvectors=eigenvectors,
        max_batch_bytes=int(max_batch_bytes),
    )

    if resolved == "cupy":
        values, vectors, gpu_batch = _CUPY_BACKEND.evaluate_eigensystem(
            model,
            coordinates,
            eigenvectors=eigenvectors,
            max_batch_bytes=int(max_batch_bytes),
        )
        batch = int(gpu_batch)
        used_workers = 1
    else:
        used_workers = worker_count if resolved == "threaded" else 1
        if resolved == "threaded":
            dispatch = max(1, int(np.ceil(n_points / (used_workers * 4))))
            per_worker_budget = max(1, int(max_batch_bytes) // used_workers)
            batch = min(
                _batch_size(
                    n_points,
                    n_basis,
                    eigenvectors=eigenvectors,
                    max_batch_bytes=per_worker_budget,
                ),
                dispatch,
            )
        ranges = _chunk_ranges(n_points, batch)

        def calculate(bounds: tuple[int, int]):
            start, stop = bounds
            return _numpy_chunk(
                model,
                coordinates[start:stop],
                eigenvectors=eigenvectors,
            )

        if resolved == "threaded" and len(ranges) > 1:
            pieces = []
            # On macOS, calling Hamiltonian assembly (notably complex einsum)
            # concurrently with Apple Accelerate eigensolvers can expose
            # transient corrupted matrices. Build one bounded wave serially,
            # then diagonalize only immutable, already-built matrices in
            # parallel. A wave completes before the next model evaluation.
            for wave_start in range(0, len(ranges), used_workers):
                wave = ranges[wave_start : wave_start + used_workers]
                matrices = [
                    np.asarray(
                        model.hamiltonian(coordinates[start:stop]),
                        dtype=np.complex128,
                    )
                    for start, stop in wave
                ]
                limiter = (
                    _threadpool_limits(limits=1, user_api="blas")
                    if _threadpool_limits is not None
                    else contextlib.nullcontext()
                )
                with limiter:
                    pieces.extend(
                        _executor(used_workers).map(
                            lambda item: _numpy_matrices_chunk(
                                item,
                                eigenvectors=eigenvectors,
                            ),
                            matrices,
                        )
                    )
        else:
            pieces = [calculate(bounds) for bounds in ranges]
        values = np.concatenate([item[0] for item in pieces], axis=0)
        vectors = (
            np.concatenate(
                [item[1] for item in pieces if item[1] is not None],
                axis=0,
            )
            if eigenvectors
            else None
        )

    values = np.asarray(values, dtype=float)
    values.setflags(write=False)
    if vectors is not None:
        vectors = np.asarray(vectors, dtype=np.complex128)
        vectors.setflags(write=False)
    return ElectronicEigensystem(
        eigenvalues=values,
        eigenvectors=vectors,
        provenance={
            "requested_backend": (
                _BACKEND if backend is None else str(backend).strip().lower()
            ),
            "resolved_backend": resolved,
            "workers": used_workers,
            "batch_size": batch,
            "max_batch_bytes": int(max_batch_bytes),
            "precision": "float64/complex128",
            "approximation": "none",
            "thread_schedule": (
                "serial_hamiltonian_parallel_eigensolver_waves"
                if resolved == "threaded"
                else "serial"
            ),
        },
    )


def validate_electronic_backend(
    model: Any,
    reduced_k: ArrayLike,
    candidate_backend: ElectronicBackend | str,
    *,
    workers: int | None = None,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance_meV: float = 1.0e-8,
    require_requested_backend: bool = True,
) -> ElectronicBackendValidation:
    """Compare candidate eigenvalues with the serial NumPy reference."""

    rtol = float(relative_tolerance)
    atol = float(absolute_tolerance_meV)
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("absolute_tolerance_meV must be finite and nonnegative")
    coordinates = np.asarray(reduced_k, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    reference = evaluate_eigensystem(
        model,
        coordinates,
        eigenvectors=False,
        backend="numpy",
        workers=1,
        max_batch_bytes=max_batch_bytes,
    )
    candidate = evaluate_eigensystem(
        model,
        coordinates,
        eigenvectors=False,
        backend=candidate_backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
    )
    difference = np.abs(candidate.eigenvalues - reference.eigenvalues)
    maximum_absolute = float(np.max(difference))
    scale = np.maximum(np.abs(reference.eigenvalues), atol)
    maximum_relative = float(np.max(difference / scale))
    requested = str(candidate_backend).strip().lower()
    resolved = str(candidate.provenance["resolved_backend"])
    used_requested = resolved == requested or (
        requested == "auto" and resolved in available_electronic_backends()
    )
    values_pass = bool(
        np.allclose(
            candidate.eigenvalues,
            reference.eigenvalues,
            rtol=rtol,
            atol=atol,
        )
    )
    passed = values_pass and (used_requested or not require_requested_backend)
    reason = ""
    if require_requested_backend and not used_requested:
        reason = f"requested backend {requested!r} resolved to {resolved!r}"
    elif not values_pass:
        reason = "candidate eigenvalues exceed the configured tolerance"
    return ElectronicBackendValidation(
        requested_backend=requested,
        resolved_backend=resolved,
        points=int(coordinates.shape[0]),
        max_absolute_error_meV=maximum_absolute,
        max_relative_error=maximum_relative,
        relative_tolerance=rtol,
        absolute_tolerance_meV=atol,
        passed=passed,
        reason=reason,
    )
