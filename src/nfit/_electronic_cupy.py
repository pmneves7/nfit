"""Optional full-precision CuPy backend for electronic eigensystems."""

from __future__ import annotations

import cupy as cp
import numpy as np

if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware dependent
    raise RuntimeError("no CUDA/ROCm device available for electronic calculations")


def evaluate_eigensystem(
    model,
    coordinates,
    *,
    eigenvectors: bool,
    max_batch_bytes: int,
):
    """Evaluate the same Fourier Hamiltonian and Hermitian solver on a GPU."""

    translations = cp.asarray(model.translations, dtype=cp.float64)
    weights = cp.asarray(model.interpolation_weights, dtype=cp.float64)
    blocks = cp.asarray(model.resolved_hamiltonian_blocks, dtype=cp.complex128)
    n_points = int(coordinates.shape[0])
    n_basis = int(model.n_basis)
    per_point = 16 * n_basis * n_basis * (2 if eigenvectors else 1)
    batch = max(1, min(n_points, int(max_batch_bytes) // max(per_point, 1)))
    value_pieces = []
    vector_pieces = []
    for start in range(0, n_points, batch):
        stop = min(start + batch, n_points)
        k_points = cp.asarray(coordinates[start:stop], dtype=cp.float64)
        phase = cp.exp(2j * cp.pi * k_points @ translations.T)
        matrices = cp.einsum(
            "kr,r,rij->kij",
            phase,
            weights,
            blocks,
            optimize=True,
        )
        if eigenvectors:
            values, vectors = cp.linalg.eigh(matrices)
            vector_pieces.append(cp.asnumpy(vectors))
        else:
            values = cp.linalg.eigvalsh(matrices)
        value_pieces.append(cp.asnumpy(values))
    return (
        np.concatenate(value_pieces, axis=0),
        (
            np.concatenate(vector_pieces, axis=0)
            if eigenvectors
            else None
        ),
        batch,
    )
