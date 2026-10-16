"""Saved visualization channels for fit history, independent of GUI state.

Only fit/residual display arrays use this float32 codec. Scientific fit samples
and their storage belong to :mod:`nfit.fit_sampling`.
"""

from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np

FIT_CHANNEL_NAMES = ("fit", "residual")


def fit_channel_array(value: Any) -> np.ndarray | None:
    """Return one saved display channel as a float array without changing it.

    Accepts encoded project payloads, legacy lists, and live ndarrays. Missing
    or malformed channels return ``None`` so stale display data can be skipped.
    """

    if value is None:
        return None
    try:
        if isinstance(value, dict):
            return _decode_float_array(value)
        return np.asarray(value, dtype=float)
    except (KeyError, ValueError, TypeError, OverflowError, zlib.error):
        return None


_fit_channel_array = fit_channel_array


def _encode_float_array(value: Any) -> dict[str, Any]:
    """Encode a display array as compressed float32 base64 for project JSON.

    Already encoded payloads pass through without materializing their arrays.
    """

    if isinstance(value, dict) and value.get("__ndarray__"):
        return value
    array = np.asarray(value, dtype=np.float32)
    return {
        "__ndarray__": True,
        "dtype": "float32",
        "shape": [int(size) for size in array.shape],
        "data": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii"),
    }


def _decode_float_array(payload: dict[str, Any]) -> np.ndarray:
    """Decode an array stored by :func:`_encode_float_array`."""

    raw = zlib.decompress(base64.b64decode(payload["data"]))
    array = np.frombuffer(raw, dtype=np.dtype(str(payload.get("dtype", "float32"))))
    return array.reshape([int(size) for size in payload["shape"]]).astype(float)


def _fit_channel_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize the metadata fields declared by the saved channel schema."""

    metadata: dict[str, Any] = {"kind": str(entry.get("kind", "points"))}
    if entry.get("visualization_only"):
        metadata["visualization_only"] = True
    if entry.get("fit_channel"):
        metadata["fit_channel"] = str(entry["fit_channel"])
    return metadata


def _fit_channels_to_dict(channels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Serialize display channels, retaining existing compressed payloads."""

    payload: dict[str, Any] = {}
    for dataset_name, entry in (channels or {}).items():
        encoded = _fit_channel_metadata(entry)
        for channel_name in FIT_CHANNEL_NAMES:
            if entry.get(channel_name) is not None:
                encoded[channel_name] = _encode_float_array(entry[channel_name])
        payload[str(dataset_name)] = encoded
    return payload


def _fit_channels_from_dict(
    payload: Any, *, decode_arrays: bool = True
) -> dict[str, dict[str, Any]]:
    """Read saved channels, eagerly decoding by default for existing callers.

    Project loading uses ``decode_arrays=False`` to retain compressed history
    until a viewer requests a channel with :func:`fit_channel_array`. Legacy
    lists and ndarrays remain accepted in either mode.
    """

    channels: dict[str, dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return channels
    for dataset_name, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        decoded = _fit_channel_metadata(entry)
        for channel_name in FIT_CHANNEL_NAMES:
            value = entry.get(channel_name)
            if not decode_arrays and isinstance(value, dict) and value.get("__ndarray__"):
                decoded[channel_name] = value
                continue
            array = _fit_channel_array(value)
            if array is not None:
                decoded[channel_name] = array
        channels[str(dataset_name)] = decoded
    return channels
