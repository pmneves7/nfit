"""Optional full-precision CuPy backend for electronic eigensystems."""

from __future__ import annotations

import hashlib

import cupy as cp
import numpy as np

if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware dependent
    raise RuntimeError("no CUDA/ROCm device available for electronic calculations")


def _coordinate_digest(coordinates) -> str:
    values = np.ascontiguousarray(coordinates, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _device_hamiltonian_components(
    model,
    coordinates,
    *,
    max_batch_bytes: int,
    cache=None,
):
    """Return reusable GPU ``H_0(k)`` and ``dH(k)/dp`` components."""

    n_points = int(coordinates.shape[0])
    n_basis = int(model.n_basis)
    n_components = len(model.parameter_blocks) + 1
    component_bytes = 16 * n_components * n_points * n_basis * n_basis
    if cache is None or component_bytes > int(cache.max_bytes):
        return None
    key = (
        "cupy_hamiltonian_components",
        model.structure_digest,
        _coordinate_digest(coordinates),
    )
    if cache is not None:
        cached = cache.get_device(key)
        if cached is not None:
            return cached
    names = tuple(model.parameter_blocks)
    blocks = cp.asarray(
        np.stack(
            (
                model.hamiltonian_blocks,
                *(model.parameter_blocks[name] for name in names),
            ),
            axis=0,
        ),
        dtype=cp.complex128,
    )
    translations = cp.asarray(model.translations, dtype=cp.float64)
    weights = cp.asarray(model.interpolation_weights, dtype=cp.float64)
    components = cp.empty(
        (n_components, n_points, n_basis, n_basis),
        dtype=cp.complex128,
    )
    per_point = max(
        1,
        16 * (int(model.translations.shape[0]) + n_components * n_basis**2),
    )
    batch = max(1, min(n_points, int(max_batch_bytes) // per_point))
    for start in range(0, n_points, batch):
        stop = min(start + batch, n_points)
        k_points = cp.asarray(coordinates[start:stop], dtype=cp.float64)
        phase = cp.exp(2j * cp.pi * k_points @ translations.T)
        phase *= weights[None, :]
        components[:, start:stop] = cp.einsum(
            "kr,crij->ckij",
            phase,
            blocks,
            optimize=True,
        )
    result = (names, components, batch)
    if cache is not None:
        cache.put_device(key, result)
    return result


def _device_eigensystem(
    model,
    coordinates,
    *,
    max_batch_bytes: int,
    cache=None,
):
    """Return a GPU-resident eigensystem for one parameter point."""

    key = (
        "cupy_eigensystem",
        model.content_digest,
        _coordinate_digest(coordinates),
    )
    if cache is not None:
        cached = cache.get_device(key)
        if cached is not None:
            return cached
    component_result = _device_hamiltonian_components(
        model,
        coordinates,
        max_batch_bytes=max_batch_bytes,
        cache=cache,
    )
    n_points = int(coordinates.shape[0])
    n_basis = int(model.n_basis)
    values = cp.empty((n_points, n_basis), dtype=cp.float64)
    vectors = cp.empty(
        (n_points, n_basis, n_basis),
        dtype=cp.complex128,
    )
    n_parameters = len(model.parameter_values)
    per_point = max(1, 16 * ((n_parameters + 2) * n_basis**2) + 8 * n_basis)
    batch = max(1, min(n_points, int(max_batch_bytes) // per_point))
    if component_result is None:
        translations = cp.asarray(model.translations, dtype=cp.float64)
        interpolation_weights = cp.asarray(
            model.interpolation_weights,
            dtype=cp.float64,
        )
        resolved_blocks = cp.asarray(
            model.resolved_hamiltonian_blocks,
            dtype=cp.complex128,
        )
        component_batch = 0
    else:
        names, components, component_batch = component_result
        coefficients = cp.asarray(
            [1.0, *(model.parameter_values[name] for name in names)],
            dtype=cp.float64,
        )
    for start in range(0, n_points, batch):
        stop = min(start + batch, n_points)
        if component_result is None:
            k_points = cp.asarray(coordinates[start:stop], dtype=cp.float64)
            phase = cp.exp(2j * cp.pi * k_points @ translations.T)
            matrices = cp.einsum(
                "kr,r,rij->kij",
                phase,
                interpolation_weights,
                resolved_blocks,
                optimize=True,
            )
        else:
            matrices = cp.tensordot(
                coefficients,
                components[:, start:stop],
                axes=(0, 0),
            )
        values[start:stop], vectors[start:stop] = cp.linalg.eigh(matrices)
    result = (
        values,
        vectors,
        {
            "requested_backend": "cupy",
            "resolved_backend": "cupy",
            "workers": 1,
            "batch_size": batch,
            "component_batch_size": component_batch,
            "hamiltonian_components_reused": component_result is not None,
            "max_batch_bytes": int(max_batch_bytes),
            "precision": "float64/complex128",
            "approximation": "none",
            "device_resident": True,
        },
    )
    if cache is not None:
        cache.put_device(key, result)
    return result


def _fermi_function(energy, chemical_potential_meV: float, temperature_K: float):
    shifted = energy - float(chemical_potential_meV)
    if temperature_K == 0.0:
        return cp.where(shifted < 0.0, 1.0, cp.where(shifted > 0.0, 0.0, 0.5))
    # Keep the constant local to avoid importing the CPU response module.
    k_b_mev_per_k = 0.08617333262
    argument = cp.clip(shifted / (k_b_mev_per_k * temperature_K), -700.0, 700.0)
    return 1.0 / (cp.exp(argument) + 1.0)


def _static_equal_energy_limit(
    energy,
    occupation,
    *,
    chemical_potential_meV: float,
    temperature_K: float,
    broadening_meV: float,
):
    if temperature_K > 0.0:
        return occupation * (1.0 - occupation) / (
            0.08617333262 * temperature_K
        )
    offset = energy - chemical_potential_meV
    return broadening_meV / (
        cp.pi * (offset * offset + broadening_meV * broadening_meV)
    )


def evaluate_lindhard(
    model,
    k,
    weights,
    q,
    energy,
    point_operators,
    *,
    temperature_K: float,
    chemical_potential_meV: float,
    broadening_meV: float,
    max_batch_bytes: int,
    transition_max_batch_bytes: int,
    permutations,
    cache=None,
):
    """Evaluate the generalized Lindhard contraction entirely on the GPU.

    Hamiltonian components, eigensystems, occupations, transition matrix
    elements, and response kernels remain device-resident. Only the completed
    susceptibility is copied back to NumPy.
    """

    base_values, base_vectors, base_provenance = _device_eigensystem(
        model,
        k,
        max_batch_bytes=max_batch_bytes,
        cache=cache,
    )
    weights_device = cp.asarray(weights, dtype=cp.float64)
    occupation_k = _fermi_function(
        base_values,
        chemical_potential_meV,
        temperature_K,
    )
    q = np.asarray(q, dtype=float)
    energy = np.asarray(energy, dtype=float)
    point_operators = np.asarray(point_operators, dtype=np.complex128)
    unique_q, inverse = np.unique(q, axis=0, return_inverse=True)
    n_operators = int(point_operators.shape[1])
    bands = int(model.n_basis)
    result = cp.empty(
        (q.shape[0], n_operators, n_operators),
        dtype=cp.complex128,
    )
    operator_keys = np.asarray(
        [_coordinate_digest(value.view(np.float64)) for value in point_operators],
        dtype=object,
    )
    execution_records = []
    transition_batch_sizes = []
    energy_batch_sizes = []
    commensurate_count = 0
    for q_index, q_value in enumerate(unique_q):
        point_indices = np.flatnonzero(inverse == q_index)
        operator_groups = [
            point_indices[operator_keys[point_indices] == key]
            for key in np.unique(operator_keys[point_indices])
        ]
        permutation = permutations[q_index]
        if permutation is None:
            shifted_values, shifted_vectors, shifted_provenance = (
                _device_eigensystem(
                    model,
                    np.asarray(k, dtype=float) + q_value[None, :],
                    max_batch_bytes=max_batch_bytes,
                    cache=cache,
                )
            )
        else:
            indices = cp.asarray(permutation, dtype=cp.int64)
            shifted_values = base_values[indices]
            shifted_vectors = base_vectors[indices]
            shifted_provenance = {
                **base_provenance,
                "resolved_backend": "commensurate_permutation",
                "source_backend": "cupy",
                "commensurate_q_reduced": q_value.tolist(),
            }
            commensurate_count += 1
        execution_records.append(shifted_provenance)
        occupation_q = _fermi_function(
            shifted_values,
            chemical_potential_meV,
            temperature_K,
        )
        per_k_bytes = max(
            1,
            16 * n_operators * bands * bands
            + 64 * bands * bands
            + 32 * bands,
        )
        transition_batch = max(
            1,
            min(int(k.shape[0]), transition_max_batch_bytes // per_k_bytes),
        )
        transition_batch_sizes.append(transition_batch)
        result[cp.asarray(point_indices, dtype=cp.int64)] = 0.0
        for start in range(0, int(k.shape[0]), transition_batch):
            stop = min(start + transition_batch, int(k.shape[0]))
            values_k = base_values[start:stop]
            values_q = shifted_values[start:stop]
            occupation_base = occupation_k[start:stop]
            occupation_shifted = occupation_q[start:stop]
            delta_energy = values_k[:, :, None] - values_q[:, None, :]
            occupation_difference = (
                occupation_base[:, :, None] - occupation_shifted[:, None, :]
            )
            equal = cp.abs(delta_energy) <= 1.0e-10
            static_limit = _static_equal_energy_limit(
                0.5 * (values_k[:, :, None] + values_q[:, None, :]),
                0.5
                * (
                    occupation_base[:, :, None]
                    + occupation_shifted[:, None, :]
                ),
                chemical_potential_meV=chemical_potential_meV,
                temperature_K=temperature_K,
                broadening_meV=broadening_meV,
            )
            batch_k = stop - start
            fixed_bytes = per_k_bytes * batch_k
            per_energy_bytes = max(1, 16 * batch_k * bands * bands)
            available = max(
                per_energy_bytes,
                transition_max_batch_bytes - fixed_bytes,
            )
            for operator_group in operator_groups:
                operators_device = cp.asarray(
                    point_operators[operator_group[0]],
                    dtype=cp.complex128,
                )
                matrix_elements = cp.einsum(
                    "kan,Aab,kbm->kAnm",
                    base_vectors[start:stop].conj(),
                    operators_device,
                    shifted_vectors[start:stop],
                    optimize=True,
                )
                energy_batch = max(
                    1,
                    min(operator_group.size, available // per_energy_bytes),
                )
                energy_batch_sizes.append(energy_batch)
                for energy_start in range(0, operator_group.size, energy_batch):
                    batch_indices = operator_group[
                        energy_start : energy_start + energy_batch
                    ]
                    transferred = cp.asarray(
                        energy[batch_indices],
                        dtype=cp.float64,
                    )
                    kernel = -occupation_difference[None, ...] / (
                        transferred[:, None, None, None]
                        + delta_energy[None, ...]
                        + 1.0j * broadening_meV
                    )
                    static_rows = cp.abs(transferred) <= 1.0e-14
                    kernel = cp.where(
                        static_rows[:, None, None, None] & equal[None, ...],
                        static_limit[None, ...],
                        kernel,
                    )
                    contraction = cp.einsum(
                        "k,eknm,kAnm,kBnm->eAB",
                        weights_device[start:stop],
                        kernel,
                        matrix_elements,
                        matrix_elements.conj(),
                        optimize=True,
                    )
                    result[cp.asarray(batch_indices, dtype=cp.int64)] += contraction
    return (
        cp.asnumpy(result),
        {
            "base_execution": base_provenance,
            "shifted_execution": execution_records,
            "transition_batch_size": (
                min(transition_batch_sizes) if transition_batch_sizes else 0
            ),
            "energy_batch_size": (
                min(energy_batch_sizes) if energy_batch_sizes else 0
            ),
            "commensurate_permutation_count": commensurate_count,
            "direct_shift_count": len(unique_q) - commensurate_count,
            "device_cache_entries": (
                0 if cache is None else cache.device_entries
            ),
        },
    )


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
