"""Optional fused CPU kernels for generalized Lindhard contractions."""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def _contract_serial(
    weights,
    energy,
    broadening,
    delta_energy,
    occupation_difference,
    equal_energy,
    static_limit,
    matrix_elements,
):
    n_energy = energy.shape[0]
    n_operators = matrix_elements.shape[1]
    n_k = weights.shape[0]
    n_bands = delta_energy.shape[1]
    result = np.zeros(
        (n_energy, n_operators, n_operators),
        dtype=np.complex128,
    )
    for energy_index in range(n_energy):
        static_row = abs(energy[energy_index]) <= 1.0e-14
        for operator_a in range(n_operators):
            for operator_b in range(n_operators):
                total = 0.0j
                for k_index in range(n_k):
                    weight = weights[k_index]
                    for band_n in range(n_bands):
                        for band_m in range(n_bands):
                            if (
                                static_row
                                and equal_energy[k_index, band_n, band_m]
                            ):
                                kernel = static_limit[
                                    k_index,
                                    band_n,
                                    band_m,
                                ]
                            else:
                                kernel = -occupation_difference[
                                    k_index,
                                    band_n,
                                    band_m,
                                ] / complex(
                                    energy[energy_index]
                                    + delta_energy[
                                        k_index,
                                        band_n,
                                        band_m,
                                    ],
                                    broadening,
                                )
                            total += (
                                weight
                                * kernel
                                * matrix_elements[
                                    k_index,
                                    operator_a,
                                    band_n,
                                    band_m,
                                ]
                                * np.conj(
                                    matrix_elements[
                                        k_index,
                                        operator_b,
                                        band_n,
                                        band_m,
                                    ]
                                )
                            )
                result[energy_index, operator_a, operator_b] = total
    return result


@njit(cache=True, nogil=True, parallel=True)
def _contract_parallel(
    weights,
    energy,
    broadening,
    delta_energy,
    occupation_difference,
    equal_energy,
    static_limit,
    matrix_elements,
):
    n_energy = energy.shape[0]
    n_operators = matrix_elements.shape[1]
    n_k = weights.shape[0]
    n_bands = delta_energy.shape[1]
    result = np.zeros(
        (n_energy, n_operators, n_operators),
        dtype=np.complex128,
    )
    count = n_energy * n_operators * n_operators
    for flat_index in prange(count):
        energy_index = flat_index // (n_operators * n_operators)
        remainder = flat_index % (n_operators * n_operators)
        operator_a = remainder // n_operators
        operator_b = remainder % n_operators
        static_row = abs(energy[energy_index]) <= 1.0e-14
        total = 0.0j
        for k_index in range(n_k):
            weight = weights[k_index]
            for band_n in range(n_bands):
                for band_m in range(n_bands):
                    if (
                        static_row
                        and equal_energy[k_index, band_n, band_m]
                    ):
                        kernel = static_limit[k_index, band_n, band_m]
                    else:
                        kernel = -occupation_difference[
                            k_index,
                            band_n,
                            band_m,
                        ] / complex(
                            energy[energy_index]
                            + delta_energy[k_index, band_n, band_m],
                            broadening,
                        )
                    total += (
                        weight
                        * kernel
                        * matrix_elements[
                            k_index,
                            operator_a,
                            band_n,
                            band_m,
                        ]
                        * np.conj(
                            matrix_elements[
                                k_index,
                                operator_b,
                                band_n,
                                band_m,
                            ]
                        )
                    )
        result[energy_index, operator_a, operator_b] = total
    return result


def initialize_num_threads(workers: int) -> None:
    """Set the Numba pool size within its configured process limit."""

    maximum = int(numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(max(1, min(int(workers), maximum)))


def contract_lindhard(
    weights,
    energy,
    broadening,
    delta_energy,
    occupation_difference,
    equal_energy,
    static_limit,
    matrix_elements,
    *,
    parallel: bool,
):
    """Return one full operator contraction without kernel temporaries."""

    function = _contract_parallel if parallel else _contract_serial
    return function(
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(energy, dtype=np.float64),
        float(broadening),
        np.ascontiguousarray(delta_energy, dtype=np.float64),
        np.ascontiguousarray(occupation_difference, dtype=np.float64),
        np.ascontiguousarray(equal_energy, dtype=np.bool_),
        np.ascontiguousarray(static_limit, dtype=np.float64),
        np.ascontiguousarray(matrix_elements, dtype=np.complex128),
    )
