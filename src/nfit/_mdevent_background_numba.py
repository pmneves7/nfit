"""Compiled measured-event background replay.

Importing this private module requires Numba.  The public replay coordinator
keeps the optional-dependency fallback and batch-allocation policy.
"""

from __future__ import annotations

import numpy as np
from numba import config, get_num_threads, get_thread_id, njit, prange, set_num_threads, types
from numba.typed import List

# Per event-transform output uses 24 bytes.  This conservative allowance also
# covers an open-addressed worker hash table when workers do not exceed events:
# its power-of-two capacity is less than four times the transform count.
REPLAY_SCRATCH_BYTES_PER_TASK = 56


def replay_scratch_bytes(events: int, transforms: int) -> int:
    """Return a conservative temporary-array bound for a replay batch."""

    return max(0, int(events)) * max(0, int(transforms)) * REPLAY_SCRATCH_BYTES_PER_TASK


def effective_workers(workers: int) -> int:
    """Cap a requested worker count at Numba's configured process ceiling."""

    return max(1, min(int(workers), int(config.NUMBA_NUM_THREADS)))


@njit(inline="always")
def _bin_index(value, edges):
    if not np.isfinite(value) or value < edges[0] or value > edges[-1]:
        return -1
    if value == edges[-1]:
        return edges.size - 2
    return np.searchsorted(edges, value, side="right") - 1


@njit(inline="always")
def _hash_slot(flat, mask):
    """Avalanche a flat grid index before selecting a power-of-two slot."""

    value = np.uint64(flat) + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return np.int64(value & np.uint64(mask))


@njit(cache=True, fastmath=False, nogil=True, parallel=True)
def _map_transforms(
    lab,
    energies,
    detector_indices,
    inverses,
    energy_bounds,
    accepted,
    exclusions,
    edge0,
    edge1,
    edge2,
    edge3,
    shape,
    raw_bins,
):
    """Map transforms in parallel while hoisting typed-list array access."""

    transforms = inverses.shape[0]
    for transform in prange(transforms):
        transform_index = np.int64(transform)
        transform_accepted = accepted[transform_index]
        excluded = exclusions[transform_index]
        inverse = inverses[transform_index]
        low_energy = energy_bounds[transform_index, 0]
        high_energy = energy_bounds[transform_index, 1]
        for event in range(lab.shape[0]):
            detector = detector_indices[event]
            energy = energies[event]
            if detector < 0 or not np.isfinite(energy):
                continue
            if detector >= transform_accepted.size or not transform_accepted[detector]:
                continue
            if energy < low_energy or energy > high_energy:
                continue
            coordinate0 = (
                lab[event, 0] * inverse[0, 0]
                + lab[event, 1] * inverse[0, 1]
                + lab[event, 2] * inverse[0, 2]
            )
            coordinate1 = (
                lab[event, 0] * inverse[1, 0]
                + lab[event, 1] * inverse[1, 1]
                + lab[event, 2] * inverse[1, 2]
            )
            coordinate2 = (
                lab[event, 0] * inverse[2, 0]
                + lab[event, 1] * inverse[2, 1]
                + lab[event, 2] * inverse[2, 2]
            )
            index0 = _bin_index(coordinate0, edge0)
            index1 = _bin_index(coordinate1, edge1)
            index2 = _bin_index(coordinate2, edge2)
            index3 = _bin_index(energy, edge3)
            if index0 < 0 or index1 < 0 or index2 < 0 or index3 < 0:
                continue
            flat = ((index0 * shape[1] + index1) * shape[2] + index2) * shape[3] + index3
            if excluded.size and excluded[flat]:
                continue
            raw_bins[transform_index, event] = flat


@njit(cache=True, fastmath=False, nogil=True, parallel=True)
def _combine_collisions(raw_bins, weights, unique_bins, combined_weights, hash_slots):
    """Combine transform copies per event using worker-local sparse hashes."""

    for event in prange(raw_bins.shape[1]):
        slots = hash_slots[get_thread_id()]
        for slot_index in range(slots.size):
            slots[slot_index] = -1
        unique_count = 0
        for transform in range(raw_bins.shape[0]):
            flat = raw_bins[transform, event]
            if flat < 0:
                continue
            slot = _hash_slot(flat, slots.size - 1)
            while slots[slot] >= 0 and unique_bins[event, slots[slot]] != flat:
                slot = (slot + 1) & (slots.size - 1)
            collision = slots[slot]
            if collision < 0:
                unique_bins[event, unique_count] = flat
                combined_weights[event, unique_count] = weights[transform]
                slots[slot] = unique_count
                unique_count += 1
            else:
                combined_weights[event, collision] += weights[transform]


def _map_and_combine(
    lab,
    energies,
    detector_indices,
    inverses,
    energy_bounds,
    weights,
    accepted,
    exclusions,
    edge0,
    edge1,
    edge2,
    edge3,
    shape,
    raw_bins,
    unique_bins,
    combined_weights,
    hash_slots,
):
    """Run the two parallel replay phases."""

    _map_transforms(
        lab,
        energies,
        detector_indices,
        inverses,
        energy_bounds,
        accepted,
        exclusions,
        edge0,
        edge1,
        edge2,
        edge3,
        shape,
        raw_bins,
    )
    _combine_collisions(raw_bins, weights, unique_bins, combined_weights, hash_slots)


@njit(cache=True, fastmath=False, nogil=True)
def _scatter(unique_bins, combined_weights, signal, source_variance, numerator, variance, events):
    """Scatter in source-event order for deterministic floating-point sums."""

    for event in range(unique_bins.shape[0]):
        for column in range(unique_bins.shape[1]):
            flat = unique_bins[event, column]
            if flat < 0:
                break
            weight = combined_weights[event, column]
            numerator[flat] += signal[event] * weight
            variance[flat] += source_variance[event] * weight * weight
            # A zero combined weight is still an observed event in this bin.
            events[flat] += 1.0


def _boolean_array_list(values):
    if hasattr(values, "_numba_type_"):
        return values
    arrays = List.empty_list(types.Array(types.boolean, 1, "C", readonly=True))
    for value in values:
        array = np.ascontiguousarray(value, dtype=np.bool_).reshape(-1).view()
        array.setflags(write=False)
        arrays.append(array)
    return arrays


def prepare_replay_flags(accepted, exclusions, *, output_size: int | None = None):
    """Convert reusable acceptance and exclusion flags to Numba typed lists."""

    if len(accepted) != len(exclusions):
        raise ValueError("replay acceptance and exclusion counts differ")
    accepted_list = _boolean_array_list(accepted)
    exclusions_list = _boolean_array_list(exclusions)
    if output_size is not None and any(
        value.size not in (0, int(output_size)) for value in exclusions_list
    ):
        raise ValueError("replay exclusion arrays must match the flattened output size")
    return accepted_list, exclusions_list


def accumulate_replayed_events(
    lab,
    energies,
    detector_indices,
    signal,
    source_variance,
    inverses,
    energy_bounds,
    weights,
    accepted,
    exclusions,
    edges,
    shape,
    numerator,
    variance,
    events,
    *,
    workers: int,
):
    """Accumulate one bounded event batch and return the worker count used.

    ``accepted`` and ``exclusions`` are sequences of one-dimensional Boolean
    arrays, one per transform.  An empty exclusion array means no excluded
    output bins.  Detector indices address the corresponding acceptance array;
    negative and out-of-range indices are rejected.
    """

    lab = np.ascontiguousarray(lab, dtype=np.float64)
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    detector_indices = np.ascontiguousarray(detector_indices, dtype=np.int64)
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    source_variance = np.ascontiguousarray(source_variance, dtype=np.float64)
    inverses = np.ascontiguousarray(inverses, dtype=np.float64)
    energy_bounds = np.ascontiguousarray(energy_bounds, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    shape = np.ascontiguousarray(shape, dtype=np.int64)
    edge0, edge1, edge2, edge3 = (
        np.ascontiguousarray(edge, dtype=np.float64) for edge in edges
    )
    transform_count = inverses.shape[0]
    if not (
        lab.ndim == 2
        and lab.shape[1] == 3
        and energies.size == detector_indices.size == signal.size == source_variance.size == lab.shape[0]
        and inverses.shape == (transform_count, 3, 3)
        and energy_bounds.shape == (transform_count, 2)
        and weights.shape == (transform_count,)
        and len(accepted) == len(exclusions) == transform_count
        and shape.shape == (4,)
    ):
        raise ValueError("inconsistent measured-event replay kernel inputs")
    output_size = int(np.prod(shape, dtype=np.int64))
    prepared_flags = hasattr(accepted, "_numba_type_") and hasattr(exclusions, "_numba_type_")
    accepted_list, exclusions_list = prepare_replay_flags(
        accepted, exclusions, output_size=None if prepared_flags else output_size
    )
    if any(np.asarray(value).ndim != 1 or np.asarray(value).size != output_size for value in (numerator, variance, events)):
        raise ValueError("replay output arrays must be flat arrays matching shape")
    raw_bins = np.full((transform_count, lab.shape[0]), -1, dtype=np.int64)
    unique_bins = np.full((lab.shape[0], transform_count), -1, dtype=np.int64)
    combined_weights = np.empty((lab.shape[0], transform_count), dtype=np.float64)
    actual_workers = min(effective_workers(workers), max(1, lab.shape[0]))
    hash_capacity = 1
    while hash_capacity < 2 * transform_count:
        hash_capacity *= 2
    hash_slots = np.empty((actual_workers, hash_capacity), dtype=np.int64)
    previous = get_num_threads()
    set_num_threads(actual_workers)
    try:
        _map_and_combine(
            lab, energies, detector_indices, inverses, energy_bounds, weights,
            accepted_list, exclusions_list, edge0, edge1, edge2, edge3, shape,
            raw_bins, unique_bins, combined_weights, hash_slots,
        )
        _scatter(
            unique_bins, combined_weights, signal, source_variance,
            numerator, variance, events,
        )
    finally:
        set_num_threads(previous)
    return actual_workers
