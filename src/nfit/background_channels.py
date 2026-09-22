"""Background-subtraction channels and lazy original-data reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mdhisto import MDHistoChannel, MDHistoData

BACKGROUND_CHANNEL = "background"
UNSUBTRACTED_CHANNEL = "unsubtracted"
BACKGROUND_EXCEPTIONS_KEY = "background_original_exceptions_v1"
BACKGROUND_PROVENANCE_KEY = "background_channels_version"
BACKGROUND_PROVENANCE_VERSION = 1


@dataclass(frozen=True)
class BackgroundChannelSelection:
    values: np.ndarray
    errors: np.ndarray
    mask: np.ndarray
    label: str
    unit: str
    quantity_type: str


def available_background_channels(data: MDHistoData) -> tuple[str, ...]:
    """Return display channels supplied by retained background provenance."""

    if (
        BACKGROUND_CHANNEL not in data.auxiliary_channels
        or data.metadata.get(BACKGROUND_PROVENANCE_KEY)
        != BACKGROUND_PROVENANCE_VERSION
    ):
        return ()
    record = data.metadata.get(BACKGROUND_EXCEPTIONS_KEY)
    if record is not None:
        _validated_exceptions(record)
    return (BACKGROUND_CHANNEL, UNSUBTRACTED_CHANNEL)


def accumulate_background_channel(
    original: MDHistoData,
    result: MDHistoData,
    *,
    contribution: np.ndarray,
    contribution_errors: np.ndarray,
    valid: np.ndarray,
) -> MDHistoData:
    """Retain one scaled background contribution and exact sparse exceptions."""

    shape = original.shape
    contribution = np.asarray(contribution, dtype=float)
    contribution_errors = np.asarray(contribution_errors, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if contribution.shape != shape or contribution_errors.shape != shape or valid.shape != shape:
        raise ValueError("background contribution arrays must match the data shape")

    previous = (
        original.auxiliary_channels.get(BACKGROUND_CHANNEL)
        if original.metadata.get(BACKGROUND_PROVENANCE_KEY)
        == BACKGROUND_PROVENANCE_VERSION
        else None
    )
    if previous is None:
        values = _readonly_if_needed(contribution)
        errors = _readonly_if_needed(contribution_errors)
    else:
        values = np.asarray(previous.values, dtype=float) + contribution
        previous_errors = (
            np.zeros(shape, dtype=float)
            if previous.errors is None
            else np.asarray(previous.errors, dtype=float)
        )
        errors = np.sqrt(np.square(previous_errors) + np.square(contribution_errors))
    _freeze(values)
    _freeze(errors)

    metadata = dict(result.metadata)
    metadata[BACKGROUND_PROVENANCE_KEY] = BACKGROUND_PROVENANCE_VERSION
    indices = _unsafe_reconstruction_indices(original, result, values, errors, valid)
    if indices.size:
        prior = (
            background_channel(
                original,
                UNSUBTRACTED_CHANNEL,
                np.unravel_index(indices, shape),
            )
            if previous is not None
            else None
        )
        signal = (
            original.signal[np.unravel_index(indices, original.shape)]
            if prior is None
            else prior.values.ravel()
        )
        original_errors = (
            original.errors[np.unravel_index(indices, original.shape)]
            if prior is None
            else prior.errors.ravel()
        )
        original_mask = (
            original.mask[np.unravel_index(indices, original.shape)]
            if prior is None
            else prior.mask.ravel()
        )
        metadata[BACKGROUND_EXCEPTIONS_KEY] = _merge_exceptions(
            metadata.get(BACKGROUND_EXCEPTIONS_KEY), indices, signal, original_errors, original_mask
        )

    channels = dict(result.auxiliary_channels)
    unit = str(original.metadata.get("signal_unit", ""))
    quantity_type = str(original.metadata.get("signal_quantity_type", "unknown"))
    channels[BACKGROUND_CHANNEL] = MDHistoChannel(
        values,
        errors,
        label="Background",
        unit=unit,
        quantity_type=quantity_type,
    )
    return result.with_updates(metadata=metadata, auxiliary_channels=channels)


def record_background_original_exceptions(
    result: MDHistoData,
    original: MDHistoData,
    where: np.ndarray,
) -> MDHistoData:
    """Record exact originals where covariance makes variance inversion unsafe."""

    where = np.asarray(where, dtype=bool)
    if where.shape != result.shape:
        raise ValueError("background exception mask must match the data shape")
    indices = np.flatnonzero(where)
    if not indices.size:
        return result
    if available_background_channels(original):
        source = background_channel(
            original,
            UNSUBTRACTED_CHANNEL,
            np.unravel_index(indices, result.shape),
        )
        source_values = source.values.ravel()
        source_errors = source.errors.ravel()
        source_mask = source.mask.ravel()
    else:
        source_values = original.signal[np.unravel_index(indices, original.shape)]
        source_errors = original.errors[np.unravel_index(indices, original.shape)]
        source_mask = original.mask[np.unravel_index(indices, original.shape)]
    metadata = dict(result.metadata)
    metadata[BACKGROUND_EXCEPTIONS_KEY] = _merge_exceptions(
        metadata.get(BACKGROUND_EXCEPTIONS_KEY),
        indices,
        source_values,
        source_errors,
        source_mask,
    )
    return result.with_updates(metadata=metadata)


def preserve_reduced_background_original(
    result: MDHistoData, values: np.ndarray, errors: np.ndarray,
) -> MDHistoData:
    """Retain exact original reductions when covariance prevents inversion."""
    if not available_background_channels(result):
        return result
    metadata = dict(result.metadata)
    metadata.pop(BACKGROUND_PROVENANCE_KEY, None)
    metadata.pop(BACKGROUND_EXCEPTIONS_KEY, None)
    original = result.with_updates(
        signal=_readonly_if_needed(values), errors=_readonly_if_needed(errors),
        metadata=metadata, auxiliary_channels={},
    )
    channel = result.auxiliary_channels[BACKGROUND_CHANNEL]
    indices = _unsafe_reconstruction_indices(
        original, result, channel.values, channel.errors, ~result.mask,
    )
    if not indices.size:
        return result
    coordinates = np.unravel_index(indices, result.shape)
    metadata = dict(result.metadata)
    metadata[BACKGROUND_EXCEPTIONS_KEY] = _merge_exceptions(
        metadata.get(BACKGROUND_EXCEPTIONS_KEY), indices,
        original.signal[coordinates], original.errors[coordinates], original.mask[coordinates],
    )
    return result.with_updates(metadata=metadata)


def background_channel(
    data: MDHistoData,
    name: str,
    selection: Any = Ellipsis,
) -> BackgroundChannelSelection:
    """Read a stored background or lazily reconstruct unsubtracted data."""

    channel = data.auxiliary_channels.get(BACKGROUND_CHANNEL)
    if (
        channel is None
        or data.metadata.get(BACKGROUND_PROVENANCE_KEY)
        != BACKGROUND_PROVENANCE_VERSION
        or name not in {BACKGROUND_CHANNEL, UNSUBTRACTED_CHANNEL}
    ):
        raise KeyError(name)
    background_values = np.asarray(channel.values[selection], dtype=float)
    background_errors = np.zeros_like(background_values) if channel.errors is None else np.asarray(channel.errors[selection], dtype=float)
    mask = np.asarray(data.mask[selection], dtype=bool)
    if name == BACKGROUND_CHANNEL:
        return BackgroundChannelSelection(
            _readonly_if_needed(background_values),
            _readonly_if_needed(background_errors),
            _readonly_if_needed(mask),
            channel.label,
            channel.unit,
            channel.quantity_type,
        )

    difference = np.asarray(data.signal[selection], dtype=float)
    difference_errors = np.asarray(data.errors[selection], dtype=float)
    values = np.asarray(difference + background_values)
    errors = np.asarray(
        np.sqrt(
            np.maximum(
                np.square(difference_errors) - np.square(background_errors), 0.0
            )
        )
    )
    restored_mask = np.array(mask, copy=True)
    _apply_exceptions(data, selection, values, errors, restored_mask)
    manual_mask = data.metadata.get("nfit_mask")
    if isinstance(manual_mask, np.ndarray) and manual_mask.shape == data.shape:
        restored_mask |= manual_mask[selection]
    return _selection(values, errors, restored_mask, "Unsubtracted", channel.unit, channel.quantity_type)


def scale_background_channels(
    data: MDHistoData,
    factor: float | np.ndarray,
    *,
    unit: str | None = None,
    quantity_type: str | None = None,
) -> MDHistoData:
    """Scale retained background provenance without changing the primary data."""

    channel = data.auxiliary_channels.get(BACKGROUND_CHANNEL)
    if (
        channel is None
        or data.metadata.get(BACKGROUND_PROVENANCE_KEY)
        != BACKGROUND_PROVENANCE_VERSION
    ):
        return data
    scaled = np.asarray(factor, dtype=float)
    target_unit = channel.unit if unit is None else unit
    target_quantity = channel.quantity_type if quantity_type is None else quantity_type
    if (
        scaled.ndim == 0
        and float(scaled) == 1.0
        and target_unit == channel.unit
        and target_quantity == channel.quantity_type
    ):
        return data
    channels = dict(data.auxiliary_channels)
    scaled_values = _freeze(channel.values * scaled)
    scaled_errors = (
        None
        if channel.errors is None
        else _freeze(np.abs(scaled) * channel.errors)
    )
    channels[BACKGROUND_CHANNEL] = MDHistoChannel(
        scaled_values,
        scaled_errors,
        channel.label,
        target_unit,
        target_quantity,
    )
    metadata = dict(data.metadata)
    record = metadata.get(BACKGROUND_EXCEPTIONS_KEY)
    if record is not None:
        checked = _validated_exceptions(record)
        coordinates = np.unravel_index(checked["indices"], data.shape)
        exception_factor = np.broadcast_to(scaled, data.shape)[coordinates]
        metadata[BACKGROUND_EXCEPTIONS_KEY] = {
            "indices": checked["indices"],
            "signal": _readonly(checked["signal"] * exception_factor),
            "errors": _readonly(checked["errors"] * np.abs(exception_factor)),
            "mask": checked["mask"],
        }
    return data.with_updates(auxiliary_channels=channels, metadata=metadata)


def _apply_exceptions(data: MDHistoData, selection: Any, values: np.ndarray, errors: np.ndarray, mask: np.ndarray) -> None:
    record = data.metadata.get(BACKGROUND_EXCEPTIONS_KEY)
    if record is None:
        return
    checked = _validated_exceptions(record)
    selected = _selected_flat_indices(data.shape, selection).ravel()
    positions = np.searchsorted(checked["indices"], selected)
    matches = positions < checked["indices"].size
    matches[matches] &= checked["indices"][positions[matches]] == selected[matches]
    if not np.any(matches):
        return
    flat_values, flat_errors, flat_mask = values.flat, errors.flat, mask.flat
    exception_positions = positions[matches]
    selected_positions = np.flatnonzero(matches)
    flat_values[selected_positions] = checked["signal"][exception_positions]
    flat_errors[selected_positions] = checked["errors"][exception_positions]
    flat_mask[selected_positions] = checked["mask"][exception_positions]


def _selected_flat_indices(shape: tuple[int, ...], selection: Any) -> np.ndarray:
    if selection is Ellipsis:
        return np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    coordinates = np.indices(shape, sparse=True)
    broadcast = np.broadcast_arrays(*coordinates)
    selected_coordinates = tuple(np.asarray(axis[selection]) for axis in broadcast)
    return np.ravel_multi_index(selected_coordinates, shape)


def _merge_exceptions(record: Any, indices: np.ndarray, signal: np.ndarray, errors: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    current = _validated_exceptions(record) if record is not None else None
    indices = np.asarray(indices, dtype=np.int64)
    signal = np.asarray(signal, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if current is not None:
        indices = np.concatenate((current["indices"], indices))
        signal = np.concatenate((current["signal"], signal))
        errors = np.concatenate((current["errors"], errors))
        mask = np.concatenate((current["mask"], mask))
    ordered, first = np.unique(indices, return_index=True)
    return {
        "indices": _freeze(ordered),
        "signal": _freeze(signal[first]),
        "errors": _freeze(errors[first]),
        "mask": _freeze(mask[first]),
    }


def _unsafe_reconstruction_indices(
    original: MDHistoData,
    result: MDHistoData,
    accumulated_values: np.ndarray,
    accumulated_errors: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Find invalid or numerically unsafe bins with bounded temporary storage."""

    previous = (
        original.auxiliary_channels.get(BACKGROUND_CHANNEL)
        if original.metadata.get(BACKGROUND_PROVENANCE_KEY)
        == BACKGROUND_PROVENANCE_VERSION
        else None
    )
    found: list[np.ndarray] = []
    size = int(np.prod(original.shape))
    chunk_size = 1_000_000
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        source_signal = _flat_chunk(original.signal, start, stop)
        source_errors = _flat_chunk(original.errors, start, stop)
        if previous is None:
            expected_signal = source_signal
            expected_errors = source_errors
        else:
            expected_signal = source_signal + _flat_chunk(previous.values, start, stop)
            previous_errors = (
                0.0
                if previous.errors is None
                else _flat_chunk(previous.errors, start, stop)
            )
            expected_errors = np.sqrt(
                np.maximum(np.square(source_errors) - np.square(previous_errors), 0.0)
            )
        reconstructed_signal = (
            _flat_chunk(result.signal, start, stop)
            + _flat_chunk(accumulated_values, start, stop)
        )
        reconstructed_errors = np.sqrt(
            np.maximum(
                np.square(_flat_chunk(result.errors, start, stop))
                - np.square(_flat_chunk(accumulated_errors, start, stop)),
                0.0,
            )
        )
        unsafe = (~_flat_chunk(valid, start, stop)) & ~_flat_chunk(
            original.mask, start, stop
        )
        unsafe |= ~np.isclose(
            reconstructed_signal,
            expected_signal,
            rtol=1.0e-12,
            atol=0.0,
            equal_nan=True,
        )
        unsafe |= ~np.isclose(
            reconstructed_errors,
            expected_errors,
            rtol=1.0e-12,
            atol=0.0,
            equal_nan=True,
        )
        local = np.flatnonzero(unsafe)
        if local.size:
            found.append(local.astype(np.int64, copy=False) + start)
    return np.concatenate(found) if found else np.empty(0, dtype=np.int64)


def _validated_exceptions(record: Any) -> dict[str, np.ndarray]:
    if not isinstance(record, dict) or set(record) != {"indices", "signal", "errors", "mask"}:
        raise ValueError("invalid background original-exception record")
    arrays = {
        "indices": np.asarray(record["indices"], dtype=np.int64),
        "signal": np.asarray(record["signal"], dtype=float),
        "errors": np.asarray(record["errors"], dtype=float),
        "mask": np.asarray(record["mask"], dtype=bool),
    }
    if any(array.ndim != 1 for array in arrays.values()) or len({array.size for array in arrays.values()}) != 1:
        raise ValueError("invalid background original-exception arrays")
    return arrays


def _selection(values: np.ndarray, errors: np.ndarray, mask: np.ndarray, label: str, unit: str, quantity_type: str) -> BackgroundChannelSelection:
    return BackgroundChannelSelection(_freeze(values), _freeze(errors), _freeze(mask), label, unit, quantity_type)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _readonly_if_needed(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.flags.writeable:
        return _readonly(result)
    return result


def _freeze(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _flat_chunk(array: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Return at most one bounded flat chunk, including for non-contiguous arrays."""

    array = np.asarray(array)
    if array.flags.c_contiguous:
        return array.reshape(-1)[start:stop]
    flat_indices = np.arange(start, stop, dtype=np.int64)
    return array[np.unravel_index(flat_indices, array.shape)]
