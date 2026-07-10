"""CuPy (GPU) per-point RPA kernels.

Mirrors the numpy per-point contractions of :mod:`metallix.spin_fluctuations`
on the GPU. The batched Hermitian eigendecomposition of ``J(Q)`` is currently
still performed on the host (LAPACK) and its results are transferred in; the
GPU accelerates the bandwidth-heavy resolvent contractions, which dominate on
large datasets. A future refinement can keep the eigendecomposition on the GPU
too (``cupy.linalg.eigh``) to avoid the transfer.

Importing this module requires a working CuPy install *and* a visible CUDA/ROCm
device; :mod:`metallix.spin_fluctuations` guards the import so environments
without a GPU transparently fall back to numpy/numba. Results match the numpy
path to floating-point precision (the maths is identical), so the CPU tests act
as the correctness reference.

For very large problems the per-orbit gather materializes ``(n_points, N, N)``
arrays in device memory; if that exceeds the GPU, lower ``_GPU_POINT_BLOCK`` to
process the points in slices (the eigendecomposition is shared across slices).
"""

from __future__ import annotations

import cupy as cp  # noqa: F401  (import failure disables this backend)

# Fail fast at import if no device is visible, so the auto backend never selects
# a non-functional GPU path.
if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware dependent
    raise RuntimeError("no CUDA/ROCm device available for the CuPy RPA backend")

# Point-block size for the gradient's (block, N, N) device temporaries. ``None``
# processes all points at once; set to an int to bound device memory.
_GPU_POINT_BLOCK: int | None = None


def rpa_value(lam, modes, point_index, energy, chi0, gamma0):
    """Return ``chi''`` per point on the GPU (uniform-weight RPA mode sum)."""

    lam_d = cp.asarray(lam)
    modes_d = cp.asarray(modes)
    idx = cp.asarray(point_index)
    e = cp.asarray(energy)[:, None]
    n_sites = lam.shape[1]

    denom = 1.0 - lam_d * chi0
    chi_q = chi0 / denom
    gamma_q = gamma0 * denom
    weights = cp.abs(modes_d.sum(axis=1)) ** 2 / n_sites
    chi_pts = (weights * chi_q)[idx]
    gamma_pts = gamma_q[idx]
    chipp = (chi_pts * gamma_pts * e / (e * e + gamma_pts * gamma_pts)).sum(axis=1)
    return cp.asnumpy(chipp)


def rpa_value_grad(lam, modes, point_index, energy, phases, chi0, gamma0):
    """Return ``chi''`` and its chi0/gamma0/orbit gradients per point on the GPU."""

    lam_d = cp.asarray(lam)
    modes_d = cp.asarray(modes)
    phases_d = cp.asarray(phases)
    n_sites = lam.shape[1]
    n_orbits = phases.shape[0]
    n_points = energy.shape[0]

    amplitudes = modes_d.sum(axis=1)
    c = cp.conj(amplitudes)
    amp_sq = cp.abs(amplitudes) ** 2

    chipp = cp.empty(n_points, dtype=cp.float64)
    grad_chi0 = cp.empty(n_points, dtype=cp.float64)
    grad_gamma0 = cp.empty(n_points, dtype=cp.float64)
    grad_j = cp.empty((n_orbits, n_points), dtype=cp.float64)

    block = _GPU_POINT_BLOCK or n_points
    energy_all = cp.asarray(energy)
    idx_all = cp.asarray(point_index)
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        sel = slice(start, stop)
        idx = idx_all[sel]
        e = energy_all[sel]
        f = chi0 / (1.0 - 1j * e / gamma0)
        lam_p = lam_d[idx]
        c_p = c[idx]
        amp_sq_p = amp_sq[idx]
        denom = 1.0 - f[:, None] * lam_p
        y = c_p / denom
        w = c_p / cp.conj(denom)
        phi_x = cp.sum(amp_sq_p / denom, axis=1)
        z_j_x = cp.sum(amp_sq_p * lam_p / (denom * denom), axis=1)
        common = (phi_x + f * z_j_x) / n_sites
        chipp[sel] = (f * phi_x / n_sites).imag
        grad_chi0[sel] = (common * f / chi0).imag
        grad_gamma0[sel] = (
            common * (-1j * e * f * f / (chi0 * gamma0 * gamma0))
        ).imag
        modes_p = modes_d[idx]
        x_site = cp.einsum("pan,pn->pa", modes_p, y)
        conj_z = cp.conj(cp.einsum("pan,pn->pa", modes_p, w))
        f_sq = f * f
        for o in range(n_orbits):
            phase_p = phases_d[o][idx]
            z_p_x = cp.einsum("pa,pab,pb->p", conj_z, phase_p, x_site)
            grad_j[o, sel] = (f_sq * z_p_x).imag / n_sites

    return (
        cp.asnumpy(chipp),
        cp.asnumpy(grad_chi0),
        cp.asnumpy(grad_gamma0),
        cp.asnumpy(grad_j),
    )
