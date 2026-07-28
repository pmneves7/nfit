"""Numba-accelerated per-point RPA kernels.

These fused kernels replace the memory-bandwidth-bound numpy contractions in
:mod:`nfit.spin_fluctuations` for large datasets. They stream over fitted
points, doing each point's small ``N x N`` mode/site contractions with only
``O(N)`` local scratch -- no ``(n_points, N, N)`` temporaries -- and parallelize
across points. The eigendecomposition of ``J(Q)`` is still done once (by LAPACK)
over the deduplicated unique-Q grid and passed in.

Importing this module requires ``numba``; :mod:`nfit.spin_fluctuations`
guards the import and falls back to numpy when it is unavailable. Results match
the numpy path to machine precision (locked by the test suite).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit, prange
from numba import set_num_threads as _numba_set_num_threads

_OPENMP_INIT_LOCK = threading.Lock()
_OPENMP_PARALLEL_INITIALIZED = False


@lru_cache(maxsize=1)
def _openmp_warning_controls():
    """Return LLVM/Intel OpenMP warning controls when that runtime is active."""

    prefix = Path(sys.prefix)
    candidates = [
        prefix / "lib" / "libomp.dylib",
        prefix / "lib" / "libomp.so",
        prefix / "lib" / "libomp.so.5",
        prefix / "Library" / "bin" / "libomp.dll",
    ]
    discovered = ctypes.util.find_library("omp")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        try:
            library = ctypes.CDLL(str(candidate))
            warnings_off = library.kmp_set_warnings_off
            warnings_on = library.kmp_set_warnings_on
            warnings_off.argtypes = []
            warnings_off.restype = None
            warnings_on.argtypes = []
            warnings_on.restype = None
            return warnings_off, warnings_on
        except (OSError, AttributeError):
            continue
    return None


def _run_parallel_kernel(kernel, *args):
    """Initialize Numba OpenMP without its deprecated-API informational line."""

    global _OPENMP_PARALLEL_INITIALIZED
    if _OPENMP_PARALLEL_INITIALIZED:
        return kernel(*args)
    with _OPENMP_INIT_LOCK:
        if _OPENMP_PARALLEL_INITIALIZED:
            return kernel(*args)
        controls = _openmp_warning_controls()
        if controls is None:
            result = kernel(*args)
        else:
            warnings_off, warnings_on = controls
            configured = os.environ.get("KMP_WARNINGS", "").strip().lower()
            warnings_were_enabled = configured not in {"0", "false", "off", "no"}
            warnings_off()
            try:
                result = kernel(*args)
            finally:
                warnings_on() if warnings_were_enabled else warnings_off()
        _OPENMP_PARALLEL_INITIALIZED = True
        return result


def initialize_num_threads(n_threads: int) -> None:
    """Set Numba's thread budget while quietly initializing its OpenMP layer."""

    global _OPENMP_PARALLEL_INITIALIZED
    with _OPENMP_INIT_LOCK:
        controls = _openmp_warning_controls()
        if controls is None:
            _numba_set_num_threads(int(n_threads))
        else:
            warnings_off, warnings_on = controls
            configured = os.environ.get("KMP_WARNINGS", "").strip().lower()
            warnings_were_enabled = configured not in {"0", "false", "off", "no"}
            warnings_off()
            try:
                _numba_set_num_threads(int(n_threads))
            finally:
                warnings_on() if warnings_were_enabled else warnings_off()
        _OPENMP_PARALLEL_INITIALIZED = True


@njit(cache=True, fastmath=False)
def _hermitian_jacobi(A, V, n):
    """In-place cyclic complex-Hermitian Jacobi on one matrix.

    ``A`` (working copy, driven toward diagonal) holds the eigenvalues on its
    diagonal on return; ``V`` (initialized to the identity) accumulates the
    eigenvectors as columns. Each off-diagonal element is annihilated by a
    phase rotation (making it real) followed by a real Givens rotation.
    """

    total = 0.0
    for i in range(n):
        for j in range(n):
            total += A[i, j].real * A[i, j].real + A[i, j].imag * A[i, j].imag
    if total <= 0.0:
        return
    threshold = 1e-24 * total
    for _sweep in range(40):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                off += A[p, q].real * A[p, q].real + A[p, q].imag * A[p, q].imag
        if off <= threshold:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                mag = abs(apq)
                if mag < 1e-300:
                    continue
                # Phase rotation: make A[p, q] real by scaling column/row q.
                ph = apq / mag
                cph = ph.conjugate()
                for k in range(n):
                    A[k, q] = A[k, q] * cph
                for k in range(n):
                    A[q, k] = A[q, k] * ph
                for k in range(n):
                    V[k, q] = V[k, q] * cph
                # Real Givens rotation to annihilate the (now real) A[p, q].
                app = A[p, p].real
                aqq = A[q, q].real
                x = A[p, q].real
                if app == aqq:
                    t = 1.0 if x >= 0.0 else -1.0
                else:
                    zeta = (app - aqq) / (2.0 * x)
                    t = (1.0 if zeta >= 0.0 else -1.0) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp = A[k, p]
                    akq = A[k, q]
                    A[k, p] = akp * c + akq * s
                    A[k, q] = -akp * s + akq * c
                for k in range(n):
                    apk = A[p, k]
                    aqk = A[q, k]
                    A[p, k] = c * apk + s * aqk
                    A[q, k] = -s * apk + c * aqk
                for k in range(n):
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = vkp * c + vkq * s
                    V[k, q] = -vkp * s + vkq * c


@njit(parallel=True, cache=True, fastmath=False)
def _batched_hermitian_eigh_parallel(matrices):
    """Batched Hermitian eigendecomposition (Jacobi), parallel over the batch.

    Returns ``(eigenvalues, eigenvectors)`` shaped like ``numpy.linalg.eigh``
    (eigenvectors as columns), but in *unspecified order and phase*. That is
    sufficient for the RPA observable, which sums over all modes and is
    invariant to eigenvector phase and to the basis within degenerate subspaces.
    Fuses the many tiny decompositions into one parallel kernel, avoiding the
    per-call LAPACK overhead that dominates for small matrices.
    """

    batch = matrices.shape[0]
    n = matrices.shape[1]
    eigenvalues = np.empty((batch, n), dtype=np.float64)
    eigenvectors = np.empty((batch, n, n), dtype=np.complex128)
    for b in prange(batch):
        A = matrices[b].copy()
        V = np.eye(n, dtype=np.complex128)
        _hermitian_jacobi(A, V, n)
        for i in range(n):
            eigenvalues[b, i] = A[i, i].real
            for j in range(n):
                eigenvectors[b, i, j] = V[i, j]
    return eigenvalues, eigenvectors


@njit(parallel=True, cache=True, fastmath=False)
def _rpa_value_kernel_parallel(lam, modes, point_index, energy, chi0, gamma0):
    """Return ``chi''`` per point (uniform-weight RPA mode sum)."""

    n_points = energy.shape[0]
    n_sites = lam.shape[1]
    chipp = np.empty(n_points, dtype=np.float64)
    inv_gamma = 1.0 / gamma0
    for p in prange(n_points):
        q = point_index[p]
        e = energy[p]
        f = chi0 / (1.0 - 1j * e * inv_gamma)
        phi_x = 0.0 + 0.0j
        for nu in range(n_sites):
            amp = 0.0 + 0.0j
            for a in range(n_sites):
                amp += modes[q, a, nu]
            amp_sq = amp.real * amp.real + amp.imag * amp.imag
            denom = 1.0 - f * lam[q, nu]
            phi_x += amp_sq / denom
        chipp[p] = (f * phi_x / n_sites).imag
    return chipp


@njit(parallel=True, cache=True, fastmath=False)
def _rpa_value_grad_kernel_parallel(lam, modes, point_index, energy, phases, chi0, gamma0):
    """Return ``chi''`` and its ``chi0``/``gamma0``/orbit gradients per point.

    ``phases`` is the ``(n_orbits, n_q, N, N)`` stack of orbit phase matrices in
    the same order as the returned orbit-gradient rows.
    """

    n_points = energy.shape[0]
    n_sites = lam.shape[1]
    n_orbits = phases.shape[0]
    chipp = np.empty(n_points, dtype=np.float64)
    grad_chi0 = np.empty(n_points, dtype=np.float64)
    grad_gamma0 = np.empty(n_points, dtype=np.float64)
    grad_j = np.empty((n_orbits, n_points), dtype=np.float64)
    inv_gamma = 1.0 / gamma0
    gamma_sq = gamma0 * gamma0
    for p in prange(n_points):
        q = point_index[p]
        e = energy[p]
        f = chi0 / (1.0 - 1j * e * inv_gamma)
        y = np.empty(n_sites, dtype=np.complex128)
        w = np.empty(n_sites, dtype=np.complex128)
        phi_x = 0.0 + 0.0j
        z_j_x = 0.0 + 0.0j
        for nu in range(n_sites):
            amp = 0.0 + 0.0j
            for a in range(n_sites):
                amp += modes[q, a, nu]
            c = np.conj(amp)
            amp_sq = amp.real * amp.real + amp.imag * amp.imag
            denom = 1.0 - f * lam[q, nu]
            y[nu] = c / denom
            w[nu] = c / np.conj(denom)
            phi_x += amp_sq / denom
            z_j_x += amp_sq * lam[q, nu] / (denom * denom)
        common = (phi_x + f * z_j_x) / n_sites
        chipp[p] = (f * phi_x / n_sites).imag
        grad_chi0[p] = (common * f / chi0).imag
        grad_gamma0[p] = (common * (-1j * e * f * f / (chi0 * gamma_sq))).imag

        # Site-basis resolvent vectors x = U y, z = U w.
        xs = np.empty(n_sites, dtype=np.complex128)
        zs = np.empty(n_sites, dtype=np.complex128)
        for a in range(n_sites):
            xa = 0.0 + 0.0j
            za = 0.0 + 0.0j
            for nu in range(n_sites):
                m = modes[q, a, nu]
                xa += m * y[nu]
                za += m * w[nu]
            xs[a] = xa
            zs[a] = za
        f_sq = f * f
        for o in range(n_orbits):
            acc = 0.0 + 0.0j
            for a in range(n_sites):
                cza = np.conj(zs[a])
                for b in range(n_sites):
                    acc += cza * phases[o, q, a, b] * xs[b]
            grad_j[o, p] = (f_sq * acc).imag / n_sites
    return chipp, grad_chi0, grad_gamma0, grad_j


def batched_hermitian_eigh(matrices):
    return _run_parallel_kernel(_batched_hermitian_eigh_parallel, matrices)


def rpa_value_kernel(lam, modes, point_index, energy, chi0, gamma0):
    return _run_parallel_kernel(
        _rpa_value_kernel_parallel,
        lam,
        modes,
        point_index,
        energy,
        chi0,
        gamma0,
    )


def rpa_value_grad_kernel(lam, modes, point_index, energy, phases, chi0, gamma0):
    return _run_parallel_kernel(
        _rpa_value_grad_kernel_parallel,
        lam,
        modes,
        point_index,
        energy,
        phases,
        chi0,
        gamma0,
    )
