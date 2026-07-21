"""Fused Numba kernels for Tier-A moment sum rules."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def tier_a_matsubara_moment_sums(
    eigenvalues: np.ndarray,
    chi0: float,
    gamma0: float,
    temperature_K: float,
    cutoff_mev: float,
    kb_mev_per_k: float,
    tail2: float,
    tail4: float,
    tail6: float,
    n_terms: int,
) -> tuple[float, float]:
    """Return unnormalized zero-point and total sums over Tier-A modes."""

    kt = kb_mev_per_k * temperature_K
    nu1 = 2.0 * math.pi * kt
    nu_values = np.empty(n_terms, dtype=np.float64)
    nu_numerators = np.empty(n_terms, dtype=np.float64)
    for term_index in range(n_terms):
        nu = nu1 * (term_index + 1)
        nu_values[term_index] = nu
        nu_numerators[term_index] = nu * math.atan(cutoff_mev / nu)
    zero_sum = 0.0
    total_sum = 0.0
    for index in range(eigenvalues.size):
        denominator = 1.0 - eigenvalues[index] * chi0
        chi = chi0 / denominator
        gamma = gamma0 * denominator
        gamma_sq = gamma * gamma
        arctan_g = math.atan(cutoff_mev / gamma)
        zero = (
            chi
            * gamma
            / (2.0 * math.pi)
            * math.log1p((cutoff_mev / gamma) ** 2)
        )
        series = 0.0
        for term_index in range(n_terms):
            nu = nu_values[term_index]
            denominator_n = nu * nu - gamma_sq
            if abs(denominator_n) < 1e-10 * gamma_sq:
                term = (
                    arctan_g / (2.0 * gamma)
                    - cutoff_mev / (2.0 * (gamma_sq + cutoff_mev**2))
                )
            else:
                term = (
                    nu_numerators[term_index] - gamma * arctan_g
                ) / denominator_n
            series += term
        c2 = cutoff_mev - gamma * arctan_g
        c4 = gamma_sq * c2 - cutoff_mev**3 / 3.0
        c6 = gamma_sq * c4 + cutoff_mev**5 / 5.0
        integrated = (
            arctan_g / gamma
            + 2.0 * series
            + 2.0 * (c2 * tail2 + c4 * tail4 + c6 * tail6)
        )
        total = 2.0 * kt * chi * gamma / math.pi * integrated
        zero_sum += zero
        total_sum += total
    return zero_sum, total_sum
