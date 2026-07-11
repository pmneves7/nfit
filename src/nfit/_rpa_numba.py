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

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=False)
def rpa_value_kernel(lam, modes, point_index, energy, chi0, gamma0):
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
def rpa_value_grad_kernel(lam, modes, point_index, energy, phases, chi0, gamma0):
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
