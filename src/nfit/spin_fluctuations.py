"""Dynamic susceptibility models for itinerant, nearly magnetically ordered metals.

This module implements the imaginary part of the dynamic spin susceptibility
chi''(Q, E) for three increasingly structured models. Conversion of chi'' to
measured neutron intensity (Bose factor, magnetic form factor, polarization
factor, overall scale) is handled separately by
:func:`nfit.cross_section.intensity_from_chipp`.

Models
------
1. **Local relaxational** (:func:`local_relaxational_chipp`): a fully local
   spin relaxing at rate ``Gamma``,

   ``chi''(E) = chi_loc * Gamma * E / (E^2 + Gamma^2)``.

   This is the single-site quasi-elastic Lorentzian used for local-moment
   systems and as the Q-independent limit of the coupled models below.

2. **Millis-Monien-Pines (MMP) relaxational** (:func:`mmp_chipp`): the
   phenomenological nearly-antiferromagnetic-liquid form

   ``chi(q, w) = chi_pk / (1 + xi^2 |q - Q0|^2 - i w / omega_sf)``

   so that

   ``chi''(q, w) = chi_pk (w/omega_sf) / [(1 + xi^2 |q - Q0|^2)^2 + (w/omega_sf)^2]``

   with correlation length ``xi`` in Angstrom and ``|q - Q0|`` in inverse
   Angstrom [Millis, Monien & Pines, PRB 42, 167 (1990); Monthoux & Pines,
   PRB 47, 6069 (1993)]. The same relaxational response describes the normal
   state of nearly antiferromagnetic metals such as the optimally doped iron
   pnictides [Inosov et al., Nat. Phys. 6, 178 (2010)].

3. **Heisenberg RPA** (:func:`heisenberg_rpa_chipp`): a single-site
   relaxational susceptibility ``chi0(w) = chi0 / (1 - i w / Gamma0)`` coupled
   through real-space Heisenberg exchange in the random phase approximation,

   ``chi(Q, w) = [1 - chi0(w) J(Q)]^{-1} chi0(w)``,

   where ``J(Q)`` is the lattice Fourier transform of the exchange couplings.
   For ``N`` magnetic sites in the unit cell ``J(Q)`` is an ``N x N`` Hermitian
   matrix; diagonalizing it (eigenvalues ``lambda_nu(Q)``, eigenvectors
   ``U(Q)``) decouples the RPA into modes with

   ``chi_Qnu = chi0 / (1 - lambda_nu chi0)``,
   ``Gamma_nu = Gamma0 (1 - lambda_nu chi0)``,

   so each mode is again relaxational and the relaxation rate softens as the
   Stoner-like criterion ``max_Q lambda_nu(Q) chi0 -> 1`` is approached --- the
   standard phenomenology of nearly ordered itinerant magnets [Moriya, *Spin
   Fluctuations in Itinerant Electron Magnetism* (Springer, 1985); Bernhoeft &
   Lonzarich, J. Phys.: Condens. Matter 7, 7325 (1995)]. The measured response
   sums the modes with neutron structure-factor weights

   ``chi''(Q, E) = sum_nu w_nu(Q) chi_Qnu Gamma_nu E / (E^2 + Gamma_nu^2)``,
   ``w_nu(Q) = |sum_a U_{a nu}(Q)|^2 / N``.

Phase convention
----------------
All phases are dimensionless with Q in reciprocal lattice units (H, K, L) and
site positions ``r_a`` fractional:

``J(Q)_{ab} = sum_{bonds (a, b, n)} J_bond exp(2 pi i (H,K,L) . (r_b + n - r_a))``

(the "extended zone" convention, matching Sunny.jl). Each physical bond is
stored once; its Hermitian conjugate is added automatically. Because the
extended-zone matrix elements already carry the full physical pair phases
``exp(2 pi i Q . (r_b + n - r_a))``, the neutron weight uses the *uniform*
sublattice sum ``|sum_a U_{a nu}|^2 / N`` -- adding site phases there would
double-count the sublattice offsets. This pairing makes the observable exactly
independent of the cell description: the same physical lattice described with
a doubled cell (twice the sites, half the r.l.u. unit) or a shifted origin
yields identical ``chi''``; the test suite locks this with cell-equivalence
tests. The sum rule ``sum_nu w_nu(Q) = 1`` holds because ``U`` is unitary.

Units
-----
Energies (``E``, ``Gamma``, ``omega_sf``, ``J``) are in meV;
``chi0``/``chi_loc``/``chi_pk`` are the susceptibility of one Cartesian
spin-operator component in 1/meV (so that ``J * chi0`` is dimensionless), up
to the overall intensity normalization. Conversion to a magnetic-moment
response multiplies by ``g^2``; conversion to rationalized SI ``M/H`` also
contains ``mu_0``. ``xi`` is in Angstrom and Q is in r.l.u.

References
----------
- T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*,
  Springer Series in Solid-State Sciences 56 (Springer, 1985).
- A. J. Millis, H. Monien, and D. Pines, Phys. Rev. B 42, 167 (1990).
- P. Monthoux and D. Pines, Phys. Rev. B 47, 6069 (1993).
- D. S. Inosov et al., Nature Physics 6, 178 (2010).
- N. Bernhoeft and G. G. Lonzarich, J. Phys.: Condens. Matter 7, 7325 (1995).
- D. Dahlbom et al., Sunny.jl, https://github.com/SunnySuite/Sunny.jl
  (symmetry-distinct bond convention).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _parallel

try:  # optional: scopes BLAS thread counts to avoid nested oversubscription
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover - threadpoolctl not installed
    _threadpool_limits = None

from .models import relaxational_chipp

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# --- compute backend selection --------------------------------------------
# The per-point RPA contractions can run on numpy (default, dependency-free), a
# fused numba kernel (large problems), or a CuPy GPU kernel. The backend is
# chosen per call by problem size so a 100-point dataset pays no JIT/GPU
# overhead, while a multi-million-point fit uses the accelerated path.
try:  # optional acceleration
    from . import _rpa_numba as _NUMBA_KERNELS
except Exception:  # pragma: no cover - numba not installed
    _NUMBA_KERNELS = None

try:  # optional GPU acceleration
    from . import _rpa_cupy as _CUPY_KERNELS
except Exception:  # pragma: no cover - cupy not installed / no GPU
    _CUPY_KERNELS = None

_RPA_BACKEND = os.environ.get("NFIT_RPA_BACKEND", "auto").lower()
# Below this many points the numpy path wins (kernel dispatch / JIT / host<->GPU
# transfer overhead is not amortized).
_NUMBA_MIN_POINTS = 50_000
_GPU_MIN_POINTS = 200_000


def set_rpa_backend(name: str) -> None:
    """Select the RPA compute backend: ``"auto"``, ``"numpy"``, ``"numba"``, ``"cupy"``.

    ``"auto"`` (the default, also settable via the ``NFIT_RPA_BACKEND``
    environment variable) picks numpy for small problems and the fastest
    available accelerated backend for large ones.
    """

    global _RPA_BACKEND
    choice = name.lower()
    if choice not in ("auto", "numpy", "numba", "cupy"):
        raise ValueError(f"unknown RPA backend {name!r}")
    _RPA_BACKEND = choice


def rpa_backend() -> str:
    """Return the configured RPA backend selection."""

    return _RPA_BACKEND


def available_rpa_backends() -> tuple[str, ...]:
    """Return the backends usable in this environment (always includes numpy)."""

    backends = ["numpy"]
    if _NUMBA_KERNELS is not None:
        backends.append("numba")
    if _CUPY_KERNELS is not None:
        backends.append("cupy")
    return tuple(backends)


def _select_rpa_backend(n_points: int) -> str:
    """Resolve the effective backend for a problem of ``n_points`` points."""

    choice = _RPA_BACKEND
    if choice == "numpy":
        return "numpy"
    if choice == "numba":
        return "numba" if _NUMBA_KERNELS is not None else "numpy"
    if choice == "cupy":
        return "cupy" if _CUPY_KERNELS is not None else "numpy"
    # auto: prefer GPU for the largest problems, then numba, else numpy.
    if _CUPY_KERNELS is not None and n_points >= _GPU_MIN_POINTS:
        return "cupy"
    if _NUMBA_KERNELS is not None and n_points >= _NUMBA_MIN_POINTS:
        return "numba"
    return "numpy"


# The batched Hermitian eigendecomposition of J(Q) dominates the cost for
# datasets with little energy-per-Q deduplication (e.g. 2D maps). For small
# matrices the fused numba Jacobi solver beats LAPACK by ~7x (no per-call
# overhead, parallel over the batch). It is used for small sublattice counts
# and large batches; larger matrices fall back to LAPACK (Jacobi is O(N^3) per
# sweep and LAPACK is more efficient there).
_NUMBA_EIGH_MAX_SITES = 16
_NUMBA_EIGH_MIN_BATCH = 2000


def _use_numba_eigh(n_batch: int, n_sites: int) -> bool:
    if _NUMBA_KERNELS is None or _RPA_BACKEND == "numpy":
        return False
    if n_sites > _NUMBA_EIGH_MAX_SITES:
        return False
    if _RPA_BACKEND in ("numba", "cupy"):
        return True
    return n_batch >= _NUMBA_EIGH_MIN_BATCH


# Batched ``eigh`` over the unique-Q grid dominates the RPA evaluation cost.
# numpy loops over the batch in a single thread, but each small Hermitian
# decomposition releases the GIL, so chunking across a thread pool can give a
# near-linear speedup. It is only worth the dispatch overhead when the total
# work M * N^3 is large enough: for tiny matrices (e.g. N=4 after primitive-cell
# reduction) the per-chunk cost is so small that threading is a net loss, so we
# gate on the work estimate rather than the batch count alone (measured
# crossover is around N=8 at M~1e5).
_EIGH_THREAD_WORK = 5.0e7

# Parallel worker budget for the RPA kernels. ``None`` means auto-detect. Auto
# uses the number of CPUs the process is actually *allowed* to run on -- on
# Linux that respects cgroup / cpuset / SLURM allocations via
# ``os.sched_getaffinity`` (so a 16-core job on a 128-core node uses 16, not
# 128, avoiding oversubscription of shared nodes) -- falling back to
# ``os.cpu_count`` elsewhere. Override with ``NFIT_NUM_THREADS`` or
# :func:`set_num_threads`.
def _detect_cpu_budget() -> int:
    return _parallel.detect_cpu_budget()


def _thread_budget() -> int:
    return _parallel.num_threads()


def set_num_threads(n: int | None) -> None:
    """Set the parallel worker budget for the RPA kernels (``None`` = auto).

    Applies to the batched-eigendecomposition thread pool and the numba kernel.
    The auto default respects the process's CPU allocation (cgroups/SLURM on
    Linux), so this is mainly for overriding on a shared machine or pinning a
    benchmark.
    """

    _parallel.set_num_threads(n)
    _eigh_executor.cache_clear()
    _apply_numba_threads()


def num_threads() -> int:
    """Return the resolved parallel worker budget."""

    return _thread_budget()


def _apply_numba_threads() -> None:
    if _NUMBA_KERNELS is None:
        return
    try:
        # set_num_threads is capped at NUMBA_NUM_THREADS (default os.cpu_count),
        # and the budget never exceeds that, so this is always valid.
        _NUMBA_KERNELS.initialize_num_threads(_thread_budget())
    except Exception:  # pragma: no cover - defensive
        pass


@lru_cache(maxsize=1)
def _eigh_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_thread_budget(), thread_name_prefix="nfit-eigh"
    )


# Align numba's thread count with the detected budget at import (it defaults to
# the full machine, ignoring cgroup/SLURM allocations).
_apply_numba_threads()


# Target complex elements per gradient block: block * N^2 stays near this, so
# the (block, N, N) temporaries hold ~2e6 complex numbers (~32 MB each, a few
# live at once). Small problems fall back to a single block.
_GRADIENT_BLOCK_ELEMENTS = 2.0e6


def _gradient_block_size(n_sites: int, n_points: int) -> int:
    """Point-block size that bounds the gradient's ``(block, N, N)`` temporaries."""

    budget = int(_GRADIENT_BLOCK_ELEMENTS / max(n_sites * n_sites, 1))
    return max(1, min(n_points, budget))


def _batched_eigh(matrices: ComplexArray) -> tuple[FloatArray, ComplexArray]:
    """Eigendecompose a stack of Hermitian matrices, threaded when large.

    Identical result to ``np.linalg.eigh`` (ascending eigenvalues, unitary
    eigenvectors); only the dispatch is parallelized across the leading axis.
    """

    workers = _thread_budget()
    batch = matrices.shape[0]
    n_sites = matrices.shape[-1]
    if workers <= 1 or batch * n_sites**3 < _EIGH_THREAD_WORK:
        return np.linalg.eigh(matrices)
    chunks = np.array_split(matrices, workers * 4, axis=0)
    # Pin the underlying BLAS/LAPACK to one thread per call for the duration of
    # the parallel map: our thread pool is already the batch-level parallelism,
    # and a multithreaded BLAS (MKL/OpenBLAS/Accelerate) would otherwise nest
    # ``workers x BLAS_threads`` threads -- catastrophic on many-core nodes.
    if _threadpool_limits is not None:
        limiter: Any = _threadpool_limits(limits=1, user_api="blas")
    else:  # pragma: no cover - threadpoolctl absent
        limiter = contextlib.nullcontext()
    with limiter:
        results = list(_eigh_executor().map(np.linalg.eigh, chunks))
    eigenvalues = np.concatenate([values for values, _ in results], axis=0)
    eigenvectors = np.concatenate([vectors for _, vectors in results], axis=0)
    return eigenvalues, eigenvectors


def local_relaxational_chipp(
    E: ArrayLike,
    *,
    chi_loc: float,
    gamma: float,
) -> FloatArray:
    """Local-spin relaxational ``chi''(E) = chi_loc Gamma E / (E^2 + Gamma^2)``.

    ``gamma`` (meV) must be positive; ``chi_loc`` carries the static local
    susceptibility (1/meV up to intensity normalization).
    """

    return relaxational_chipp(float(chi_loc), float(gamma), E)


def mmp_chipp(
    q_minus_q0_sq_inv_angstrom2: ArrayLike,
    E: ArrayLike,
    *,
    chi_pk: float,
    xi: float,
    omega_sf: float,
) -> FloatArray:
    """Millis-Monien-Pines relaxational ``chi''`` near an ordering vector.

    ``chi'' = chi_pk (E/omega_sf) / [(1 + xi^2 |q - Q0|^2)^2 + (E/omega_sf)^2]``

    with ``|q - Q0|^2`` supplied in inverse square Angstrom, ``xi`` in
    Angstrom, and ``omega_sf`` in meV [Millis, Monien & Pines, PRB 42, 167
    (1990)].
    """

    if omega_sf <= 0:
        raise ValueError("omega_sf must be positive")
    q_sq = np.asarray(q_minus_q0_sq_inv_angstrom2, dtype=float)
    x = np.asarray(E, dtype=float) / float(omega_sf)
    a = 1.0 + float(xi) ** 2 * q_sq
    return float(chi_pk) * x / (a * a + x * x)


@dataclass
class RpaGeometry:
    """Q-dependent, exchange-independent RPA precomputation for one dataset.

    Built once per (H, K, L) grid by :func:`build_rpa_geometry`; only the small
    per-Q eigenproblems depend on the exchange constants, so fitting re-uses
    this object across optimizer iterations. Points sharing the same Q (e.g.
    every energy bin of a constant-Q cut) are deduplicated: arrays are stored
    at ``n_q`` unique Q points and ``point_index`` maps each of the original
    points back to its unique Q row.
    """

    bond_phases: dict[str, ComplexArray]
    """Orbit label -> ``(n_q, n_sites, n_sites)`` Hermitian phase sums."""

    point_index: NDArray[np.intp]
    """``(n_points,)`` map from original points to unique Q rows."""

    n_sites: int

    n_q: int
    """Number of unique Q points the phase arrays are stored at."""

    unique_hkl: FloatArray | None = None
    """``(n_q, 3)`` HKL of each unique Q row (RLU); needed for the tensor path's
    Cartesian Q-hat polarization factor and for the Ewald dipole sum."""


def _site_positions(sites: ArrayLike) -> FloatArray:
    positions = np.atleast_2d(np.asarray(sites, dtype=float))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("site positions must be an (n_sites, 3) array of fractional coordinates")
    return positions


def _wrap_to_unit(vectors: FloatArray) -> FloatArray:
    """Map fractional vectors into [0, 1) with wrap-aware rounding."""

    wrapped = vectors - np.floor(vectors)
    wrapped[wrapped > 1.0 - 1e-12] = 0.0
    return wrapped


def _site_match_index(position: FloatArray, positions: FloatArray, tol: float) -> int:
    """Return the index of ``position`` in ``positions`` modulo 1, or -1."""

    delta = positions - position
    delta -= np.rint(delta)
    matches = np.nonzero(np.all(np.abs(delta) < tol, axis=1))[0]
    return int(matches[0]) if matches.size else -1


def _canonical_bond_key(
    site_i: int, site_j: int, offset: FloatArray
) -> tuple[int, int, tuple[float, ...]]:
    """Orientation-insensitive hashable key for one physical bond."""

    forward = (site_i, site_j, tuple(np.round(np.asarray(offset, dtype=float), 6)))
    backward = (site_j, site_i, tuple(np.round(-np.asarray(offset, dtype=float), 6)))
    return min(forward, backward)


def _translation_reduction(
    positions: FloatArray,
    orbits: Sequence[Mapping[str, Any]],
    tol: float,
) -> tuple[list[int], list[int], dict[int, int], int] | None:
    """Detect the primitive translational sub-cell of a site/bond network.

    Returns ``(reps, representative, rep_index, group_order)`` when the network
    folds onto a smaller cell, or ``None`` when it does not (or any consistency
    check fails). ``representative[i]`` is the lowest site index in ``i``'s
    translation coset; ``reps`` lists the distinct representatives; ``rep_index``
    maps a representative to its rank in ``reps``; ``group_order`` is the number
    of sites per coset. Shared by the scalar and tensor-carrying folds.
    """

    n_sites = positions.shape[0]
    if n_sites <= 1:
        return None

    wrapped = _wrap_to_unit(positions)

    def bond_keys(bond_list: Sequence[Mapping[str, Any]]) -> set | None:
        keys = set()
        for bond in bond_list:
            keys.add(
                _canonical_bond_key(
                    int(bond["site_i"]),
                    int(bond["site_j"]),
                    np.asarray(bond["offset"], dtype=float),
                )
            )
        return keys if len(keys) == len(bond_list) else None

    orbit_keys = []
    for orbit in orbits:
        keys = bond_keys(orbit.get("bonds", []))
        if keys is None:
            return None
        orbit_keys.append(keys)

    def translation_site_map(translation: FloatArray) -> list[tuple[int, FloatArray]] | None:
        """Map each site through ``+translation``: (new index, integer shift)."""

        mapping: list[tuple[int, FloatArray]] = []
        seen: set[int] = set()
        for index in range(n_sites):
            moved = positions[index] + translation
            target = _site_match_index(moved, positions, tol)
            if target < 0 or target in seen:
                return None
            seen.add(target)
            shift = np.rint(moved - positions[target])
            mapping.append((target, shift))
        return mapping

    def preserves_bonds(mapping: list[tuple[int, FloatArray]]) -> bool:
        for orbit, keys in zip(orbits, orbit_keys, strict=True):
            translated = set()
            for bond in orbit.get("bonds", []):
                i, shift_i = mapping[int(bond["site_i"])]
                j, shift_j = mapping[int(bond["site_j"])]
                offset = np.asarray(bond["offset"], dtype=float) + shift_j - shift_i
                translated.add(_canonical_bond_key(i, j, offset))
            if translated != keys:
                return False
        return True

    # Candidate translations: pairwise site differences mod 1.
    candidates: list[FloatArray] = []
    for a in range(n_sites):
        for b in range(n_sites):
            if a == b:
                continue
            t = _wrap_to_unit((wrapped[b] - wrapped[a])[np.newaxis, :])[0]
            if np.all(np.abs(t) < tol):
                continue
            if any(np.all(np.abs(t - known) < tol) for known in candidates):
                continue
            candidates.append(t)

    translations: list[tuple[FloatArray, list[tuple[int, FloatArray]]]] = []
    for t in candidates:
        mapping = translation_site_map(t)
        if mapping is not None and preserves_bonds(mapping):
            translations.append((t, mapping))
    if not translations:
        return None

    # Defensive group-closure check: composing two valid translations must be
    # the identity or another valid translation, or the detection is
    # inconsistent and we decline to reduce.
    valid = [t for t, _ in translations]
    for t1 in valid:
        for t2 in valid:
            composed = _wrap_to_unit((t1 + t2)[np.newaxis, :])[0]
            if np.all(np.abs(composed) < tol):
                continue
            if not any(np.all(np.abs(composed - t) < tol) for t in valid):
                return None

    # Partition sites into cosets under the translation group; the lowest
    # index of each coset is its representative.
    representative = list(range(n_sites))
    for _, mapping in translations:
        for index in range(n_sites):
            target = mapping[index][0]
            root = min(representative[index], representative[target])
            representative[index] = root
            representative[target] = root
    # Compress (one pass suffices after taking pairwise minima repeatedly).
    for _ in range(n_sites):
        changed = False
        for index in range(n_sites):
            if representative[representative[index]] < representative[index]:
                representative[index] = representative[representative[index]]
                changed = True
        if not changed:
            break

    reps = sorted(set(representative))
    n_reduced = len(reps)
    if n_reduced == n_sites:
        return None
    if n_sites % n_reduced != 0:
        return None
    rep_index = {site: rank for rank, site in enumerate(reps)}
    group_order = n_sites // n_reduced
    return reps, representative, rep_index, group_order


def _fold_orbits(
    positions: FloatArray,
    orbits: Sequence[Mapping[str, Any]],
    representative: list[int],
    rep_index: dict[int, int],
    group_order: int,
    *,
    carry_tensor: bool = False,
) -> list[dict[str, Any]] | None:
    """Fold each orbit's bonds onto the reduced cell.

    Returns the reduced orbits, or ``None`` if any orbit does not fold cleanly
    (bond count not divisible by ``group_order``). In ``carry_tensor`` mode each
    folded bond keeps the symmetry rotation and reversal of the representative
    full-cell bond it came from -- pure lattice translations do not rotate spins,
    so every bond folding to one key produces the same Cartesian tensor, and the
    carried ``rotation``/``reversed`` stay consistent with the stored bond
    orientation.
    """

    reduced_orbits: list[dict[str, Any]] = []
    for orbit in orbits:
        folded: dict[tuple, dict[str, Any]] = {}
        for bond in orbit.get("bonds", []):
            i = int(bond["site_i"])
            j = int(bond["site_j"])
            offset = np.asarray(bond["offset"], dtype=float)
            alpha = representative[i]
            beta = representative[j]
            # Preserve the physical displacement: r_j + offset - r_i must
            # equal r_beta + offset' - r_alpha.
            new_offset = (positions[j] + offset - positions[i]) + (
                positions[alpha] - positions[beta]
            )
            key = _canonical_bond_key(rep_index[alpha], rep_index[beta], new_offset)
            entry = {
                "site_i": rep_index[alpha],
                "site_j": rep_index[beta],
                "offset": [float(x) for x in new_offset],
            }
            if carry_tensor:
                rotation = bond.get("rotation")
                if rotation is not None:
                    entry["rotation"] = [
                        [float(x) for x in row] for row in np.asarray(rotation, dtype=float)
                    ]
                entry["reversed"] = bool(bond.get("reversed", False))
            folded[key] = entry
        if len(folded) * group_order != len(orbit.get("bonds", [])):
            return None
        reduced = dict(orbit)
        reduced["bonds"] = list(folded.values())
        reduced_orbits.append(reduced)
    return reduced_orbits


def reduce_site_network(
    site_positions: ArrayLike,
    orbits: Sequence[Mapping[str, Any]],
    *,
    tol: float = 1e-8,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Fold a site/bond network onto its primitive translational cell.

    When the magnetic sites and every labeled bond orbit are invariant under a
    fractional lattice translation (an F/I/C/R centering, or any accidental
    translation symmetry), the extended-zone ``J(Q)`` block-diagonalizes and
    the uniform neutron weight vector lies entirely in the untranslated
    sector. Evaluating with one representative site per translation class and
    fractional bond offsets therefore gives *identical* ``chi''`` at a fraction
    of the eigendecomposition cost (e.g. pyrochlore in the conventional cubic
    cell: 16 sites -> 4).

    Everything stays expressed in the caller's cell: Q remains in the same
    r.l.u., positions keep their fractional coordinates, and bond offsets
    simply become fractional vectors such as ``(1/2, 1/2, 0)``. The reduction
    is detected empirically from the site and bond lists themselves -- never
    from a space-group symbol -- so hand-edited networks that break the
    translation symmetry are left untouched. Returns the inputs unchanged
    whenever no valid translation exists or any consistency check fails.
    """

    positions = _site_positions(site_positions)
    original = (positions, [dict(orbit) for orbit in orbits])
    detected = _translation_reduction(positions, orbits, tol)
    if detected is None:
        return original
    reps, representative, rep_index, group_order = detected
    reduced_orbits = _fold_orbits(
        positions, orbits, representative, rep_index, group_order
    )
    if reduced_orbits is None:
        return original
    return positions[reps], reduced_orbits


def reduce_site_network_with_tensors(
    site_positions: ArrayLike,
    orbits: Sequence[Mapping[str, Any]],
    *,
    site_rotations: Sequence[Any] | None = None,
    sia: Mapping[str, Any] | None = None,
    dipole_enabled: bool = False,
    tol: float = 1e-8,
) -> tuple[FloatArray, list[dict[str, Any]], list[Any] | None, dict[str, Any] | None]:
    """Fold a tensor-carrying network onto its primitive translational cell.

    Like :func:`reduce_site_network`, but additionally carries the per-bond
    symmetry rotation/reversal (needed to rebuild the anisotropic exchange
    tensors), the per-site generating rotations (needed for single-ion
    anisotropy), and the single-ion ``sites`` index lists through the fold.
    Pure lattice translations do not rotate spins, so bond-resolved anisotropic
    exchange and on-site single-ion anisotropy are exactly invariant under the
    fold. **Dipole coupling is not folded**: its Ewald lattice sum depends on
    the Bravais lattice, and the reduced cell would need the (finer) primitive
    lattice vectors rather than the caller's conventional lattice, so when
    ``dipole_enabled`` the full cell is kept. Returns ``(positions, orbits,
    site_rotations, sia)`` unchanged whenever no reduction applies, a payload
    cannot fold cleanly, or the dipole term is active.
    """

    positions = _site_positions(site_positions)
    original = (
        positions,
        [dict(orbit) for orbit in orbits],
        list(site_rotations) if site_rotations is not None else None,
        dict(sia) if sia is not None else None,
    )
    if dipole_enabled:
        return original
    detected = _translation_reduction(positions, orbits, tol)
    if detected is None:
        return original
    reps, representative, rep_index, group_order = detected
    reduced_orbits = _fold_orbits(
        positions, orbits, representative, rep_index, group_order, carry_tensor=True
    )
    if reduced_orbits is None:
        return original

    reduced_rotations: list[Any] | None = None
    if site_rotations is not None:
        if len(site_rotations) != positions.shape[0]:
            return original
        reduced_rotations = [site_rotations[site] for site in reps]

    reduced_sia: dict[str, Any] | None = None
    if sia:
        reduced_sia = {}
        for class_label, spec in sia.items():
            if not spec:
                reduced_sia[class_label] = spec
                continue
            new_spec = dict(spec)
            indices = spec.get("sites")
            if indices is not None:
                folded_sites = sorted(
                    {rep_index[representative[int(index)]] for index in indices}
                )
                new_spec["sites"] = folded_sites
            reduced_sia[class_label] = new_spec

    return positions[reps], reduced_orbits, reduced_rotations, reduced_sia


def build_rpa_geometry(
    H: ArrayLike,
    K: ArrayLike,
    L: ArrayLike,
    site_positions: ArrayLike,
    orbits: Sequence[Mapping[str, Any]],
) -> RpaGeometry:
    """Precompute the Q-dependent phase factors entering ``J(Q)``.

    Parameters
    ----------
    H, K, L
        Momentum transfer in reciprocal lattice units, one value per point.
    site_positions
        ``(n_sites, 3)`` fractional coordinates of the magnetic sites in the
        unit cell.
    orbits
        Bond orbits as plain mappings ``{"label": str, "bonds": [{"site_i":
        int, "site_j": int, "offset": [n1, n2, n3]}, ...]}``. Each physical
        bond appears exactly once; the Hermitian conjugate term is added here.
        All bonds of one orbit share a single exchange constant.
    """

    positions = _site_positions(site_positions)
    n_sites = positions.shape[0]

    hkl = np.column_stack(
        [
            np.asarray(H, dtype=float).ravel(),
            np.asarray(K, dtype=float).ravel(),
            np.asarray(L, dtype=float).ravel(),
        ]
    )
    unique_hkl, point_index = np.unique(hkl, axis=0, return_inverse=True)
    point_index = np.asarray(point_index, dtype=np.intp).ravel()

    bond_phases: dict[str, ComplexArray] = {}
    for orbit in orbits:
        label = str(orbit["label"])
        if label in bond_phases:
            raise ValueError(f"duplicate bond orbit label {label!r}")
        phases = np.zeros((unique_hkl.shape[0], n_sites, n_sites), dtype=complex)
        bonds = orbit["bonds"]
        if not bonds:
            raise ValueError(f"bond orbit {label!r} contains no bonds")
        for bond in bonds:
            i = int(bond["site_i"])
            j = int(bond["site_j"])
            if not (0 <= i < n_sites and 0 <= j < n_sites):
                raise ValueError(
                    f"bond orbit {label!r} references site index outside "
                    f"0..{n_sites - 1}"
                )
            offset = np.asarray(bond["offset"], dtype=float)
            if offset.shape != (3,):
                raise ValueError(f"bond orbit {label!r} has a non-3-vector cell offset")
            delta = positions[j] + offset - positions[i]
            phase = np.exp(2j * np.pi * (unique_hkl @ delta))
            phases[:, i, j] += phase
            phases[:, j, i] += np.conj(phase)
        bond_phases[label] = phases

    return RpaGeometry(
        bond_phases=bond_phases,
        point_index=point_index,
        n_sites=n_sites,
        n_q=int(unique_hkl.shape[0]),
        unique_hkl=np.ascontiguousarray(unique_hkl, dtype=float),
    )


def rpa_exchange_matrix(
    geometry: RpaGeometry, j_values: Mapping[str, float]
) -> ComplexArray:
    """Assemble ``J(Q)`` (meV) at the unique Q points of ``geometry``."""

    matrix = np.zeros(
        (geometry.n_q, geometry.n_sites, geometry.n_sites), dtype=complex
    )
    for label, value in j_values.items():
        try:
            phases = geometry.bond_phases[label]
        except KeyError:
            raise KeyError(
                f"unknown bond orbit {label!r}; geometry defines "
                f"{sorted(geometry.bond_phases)}"
            ) from None
        matrix += float(value) * phases
    return matrix


def _validate_rpa_scalars(chi0: float, gamma0: float) -> None:
    if gamma0 <= 0:
        raise ValueError("gamma0 must be positive")
    if chi0 <= 0:
        raise ValueError("chi0 must be positive")


def _validate_rpa_energy(geometry: RpaGeometry, energy: FloatArray) -> None:
    if energy.shape != geometry.point_index.shape:
        raise ValueError(
            f"E has {energy.size} points but the geometry was built for "
            f"{geometry.point_index.size}"
        )


def _rpa_modes(
    geometry: RpaGeometry,
    j_values: Mapping[str, float],
    chi0: float,
    lambda_shift: float = 0.0,
) -> tuple[FloatArray, ComplexArray]:
    """Eigen-factor ``J(Q)`` at each unique Q and guard against instability.

    Returns ``(lam, modes)`` with ``lam`` the real eigenvalues ``(n_q,
    n_sites)`` and ``modes`` the unitary eigenvectors ``(n_q, n_sites,
    n_sites)`` (``modes[q, a, nu] = U_{a nu}(Q)``). Order and phase are
    unspecified; the RPA observable sums over all modes and is invariant to
    both, and to the basis within degenerate subspaces.

    The eigendecomposition is the hot kernel of the model. It is computed by a
    fused numba Jacobi solver for small matrices (much faster than per-call
    LAPACK) or by LAPACK otherwise, and cached on the geometry keyed by the
    exchange values so a least-squares iteration's back-to-back value and
    Jacobian evaluations share a single decomposition.

    Raises ``ValueError`` when the static RPA denominator ``1 - lambda chi0`` is
    non-positive anywhere, i.e. at or beyond the magnetic instability
    ``max_Q lambda_nu(Q) chi0 = 1``. ``lambda_shift`` is the Onsager reaction
    field (a rigid shift ``lambda -> lambda - lambda_shift`` applied by the
    caller); it enters only the stability check here -- the returned
    eigenvalues stay unshifted so the cache is keyed by ``J`` alone.
    """

    # Cache the decomposition on the geometry keyed by the exchange values.
    # lam/modes depend only on J(Q) (not chi0/gamma0), so the value and the
    # Jacobian at the same parameters -- which scipy evaluates back to back --
    # reuse one eigendecomposition instead of computing it twice.
    key = tuple(sorted((str(label), float(value)) for label, value in j_values.items()))
    cached = geometry.__dict__.get("_rpa_modes_cache")
    if cached is not None and cached[0] == key:
        lam, modes = cached[1], cached[2]
    else:
        exchange = rpa_exchange_matrix(geometry, j_values)
        if _use_numba_eigh(exchange.shape[0], exchange.shape[-1]):
            lam, modes = _NUMBA_KERNELS.batched_hermitian_eigh(
                np.ascontiguousarray(exchange)
            )
        else:
            lam, modes = _batched_eigh(exchange)
        geometry.__dict__["_rpa_modes_cache"] = (key, lam, modes)

    lam_effective = lam - lambda_shift if lambda_shift != 0.0 else lam
    if np.any(1.0 - lam_effective * chi0 <= 0.0):
        raise ValueError(
            "RPA instability: 1 - lambda(Q) * chi0 <= 0 "
            f"(max lambda * chi0 = {float(np.max(lam_effective * chi0)):.6g}); "
            "the parameters describe a magnetically ordered state"
        )
    return lam, modes


def _stacked_phases(geometry: RpaGeometry, labels: Sequence[str]) -> ComplexArray:
    """Return the ``(n_orbits, n_q, N, N)`` phase stack, memoized per geometry."""

    cache = geometry.__dict__.setdefault("_phase_stack_cache", {})
    key = tuple(labels)
    stack = cache.get(key)
    if stack is None:
        stack = np.ascontiguousarray(
            np.stack([geometry.bond_phases[label] for label in labels], axis=0)
        )
        cache[key] = stack
    return stack


def heisenberg_rpa_chipp(
    geometry: RpaGeometry,
    E: ArrayLike,
    *,
    chi0: float,
    gamma0: float,
    j_values: Mapping[str, float],
    lambda_shift: float = 0.0,
) -> FloatArray:
    """RPA ``chi''(Q, E)`` for relaxational local spins coupled by ``J(Q)``.

    See the module docstring for the full model and conventions. Raises
    ``ValueError`` when the RPA denominator ``1 - lambda_nu(Q) chi0`` is not
    positive at some Q, i.e. when the parameters are at or beyond the magnetic
    instability ``max_Q lambda_nu(Q) chi0 = 1``. ``lambda_shift`` is the
    Onsager reaction field of the sum-rule closures: a rigid, Q-independent
    shift ``J(Q) -> J(Q) - lambda_shift`` (equivalently of every eigenvalue);
    the default 0.0 leaves the computation untouched.
    """

    chi0 = float(chi0)
    gamma0 = float(gamma0)
    _validate_rpa_scalars(chi0, gamma0)
    energy = np.asarray(E, dtype=float).ravel()
    _validate_rpa_energy(geometry, energy)

    lam, modes = _rpa_modes(geometry, j_values, chi0, lambda_shift)
    if lambda_shift != 0.0:
        lam = lam - lambda_shift
    idx = geometry.point_index

    backend = _select_rpa_backend(energy.shape[0])
    if backend == "numba":
        return np.asarray(
            _NUMBA_KERNELS.rpa_value_kernel(
                lam, np.ascontiguousarray(modes), idx, energy, chi0, gamma0
            ),
            dtype=float,
        )
    if backend == "cupy":
        return np.asarray(
            _CUPY_KERNELS.rpa_value(lam, modes, idx, energy, chi0, gamma0), dtype=float
        )

    denominator = 1.0 - lam * chi0
    chi_q = chi0 / denominator
    gamma_q = gamma0 * denominator
    # Uniform sublattice weight: the extended-zone J(Q) already carries the
    # full pair phases, so the neutron amplitude of mode nu is sum_a U_{a nu}.
    weights = np.abs(modes.sum(axis=1)) ** 2 / geometry.n_sites

    chi_pts = (weights * chi_q)[idx]
    gamma_pts = gamma_q[idx]
    e = energy[:, np.newaxis]
    chipp = chi_pts * gamma_pts * e / (e * e + gamma_pts * gamma_pts)
    return np.asarray(chipp.sum(axis=1), dtype=float)


def heisenberg_rpa_chipp_and_gradients(
    geometry: RpaGeometry,
    E: ArrayLike,
    *,
    chi0: float,
    gamma0: float,
    j_values: Mapping[str, float],
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """RPA ``chi''`` together with its exact parameter gradients.

    Returns ``(chipp, gradients)`` where ``gradients`` maps ``"chi0"``,
    ``"gamma0"``, and each exchange orbit label to ``d chi'' / d parameter``
    (one array per fitted point). One eigendecomposition of ``J(Q)`` serves the
    value and every derivative.

    The derivatives use the resolvent identity ``chi = chi0(w) phi^dag A^{-1}
    phi / N`` with ``A = 1 - chi0(w) J(Q)``, ``phi`` the uniform sublattice
    vector, and ``chi0(w) = chi0 / (1 - i w / gamma0)``:

    - ``d chi / d theta_o = chi0(w)^2 (z^dag P_o x) / N`` where ``x = A^{-1}
      phi``, ``z = A^{-dag} phi`` and ``P_o = dJ/dtheta_o`` is the cached orbit
      phase matrix. No eigenvector derivatives appear, so the expression is
      exact even where ``J(Q)`` has degenerate (flat) bands.
    - ``chi0`` and ``gamma0`` enter through ``chi0(w)`` and through ``A``; the
      chain rule gives ``d chi / d chi0(w) = [phi^dag x + chi0(w) z^dag J x] /
      N`` times ``d chi0(w)/d chi0 = chi0(w)/chi0`` or ``d chi0(w)/d gamma0 =
      -i w chi0(w)^2 / (chi0 gamma0^2)``.

    Every future term of the model is ``theta_p P_p(Q)`` (single-ion
    anisotropy, anisotropic/tensor exchange, dipole-dipole, Zeeman), so the
    ``theta_o`` formula generalizes verbatim.
    """

    chi0 = float(chi0)
    gamma0 = float(gamma0)
    _validate_rpa_scalars(chi0, gamma0)
    energy = np.asarray(E, dtype=float).ravel()
    _validate_rpa_energy(geometry, energy)

    lam, modes = _rpa_modes(geometry, j_values, chi0)
    n_sites = geometry.n_sites
    idx = geometry.point_index
    labels = list(j_values)

    backend = _select_rpa_backend(energy.shape[0])
    if backend in ("numba", "cupy"):
        phases = _stacked_phases(geometry, labels)
        if backend == "numba":
            chipp, grad_chi0, grad_gamma0, grad_j = _NUMBA_KERNELS.rpa_value_grad_kernel(
                lam, np.ascontiguousarray(modes), idx, energy, phases, chi0, gamma0
            )
        else:
            chipp, grad_chi0, grad_gamma0, grad_j = _CUPY_KERNELS.rpa_value_grad(
                lam, modes, idx, energy, phases, chi0, gamma0
            )
        gradients = {
            "chi0": np.asarray(grad_chi0, dtype=float),
            "gamma0": np.asarray(grad_gamma0, dtype=float),
        }
        for i, label in enumerate(labels):
            gradients[label] = np.asarray(grad_j[i], dtype=float)
        return np.asarray(chipp, dtype=float), gradients

    # phi is the uniform sublattice vector; c = U^dag phi has components
    # c_nu = conj(sum_a U_{a nu}). Its squared modulus is the mode weight.
    amplitudes = modes.sum(axis=1)
    c = np.conj(amplitudes)
    amp_sq = np.abs(amplitudes) ** 2

    n_pts = energy.shape[0]
    chipp = np.empty(n_pts, dtype=float)
    gradients: dict[str, FloatArray] = {
        name: np.empty(n_pts, dtype=float) for name in ("chi0", "gamma0", *labels)
    }

    # Process points in blocks: the per-point site-basis resolvent vectors and
    # the per-orbit phase gathers materialize (block, N, N) complex arrays, so
    # chunking caps peak memory at a fixed budget regardless of dataset size.
    # The eigendecomposition above is over the (deduplicated) unique-Q grid and
    # is shared across all blocks.
    block = _gradient_block_size(n_sites, n_pts)
    for start in range(0, n_pts, block):
        stop = min(start + block, n_pts)
        sel = slice(start, stop)
        idx_c = idx[sel]
        energy_c = energy[sel]
        f = chi0 / (1.0 - 1j * energy_c / gamma0)  # chi0(w), per point
        lam_c = lam[idx_c]
        c_c = c[idx_c]
        amp_sq_c = amp_sq[idx_c]
        denom = 1.0 - f[:, np.newaxis] * lam_c  # A eigenvalues, per point x mode

        y = c_c / denom  # A^{-1} phi in the eigenbasis
        w = c_c / np.conj(denom)  # A^{-dag} phi in the eigenbasis

        phi_x = np.sum(amp_sq_c / denom, axis=1)  # phi^dag A^{-1} phi
        z_j_x = np.sum(amp_sq_c * lam_c / (denom * denom), axis=1)  # z^dag J x
        common = (phi_x + f * z_j_x) / n_sites  # d chi / d chi0(w)

        chipp[sel] = (f * phi_x / n_sites).imag
        gradients["chi0"][sel] = (common * f / chi0).imag
        gradients["gamma0"][sel] = (
            common * (-1j * energy_c * f * f / (chi0 * gamma0 * gamma0))
        ).imag

        # Site-basis resolvent vectors x = U y and z = U w, shared across orbits.
        modes_c = modes[idx_c]
        x_site = np.einsum("pan,pn->pa", modes_c, y)
        conj_z = np.conj(np.einsum("pan,pn->pa", modes_c, w))
        f_sq = f * f
        for label in labels:
            phase_c = geometry.bond_phases[label][idx_c]
            z_p_x = np.einsum("pa,pab,pb->p", conj_z, phase_c, x_site)
            gradients[label][sel] = (f_sq * z_p_x).imag / n_sites

    return chipp, gradients
