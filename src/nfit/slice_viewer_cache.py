"""Lightweight cache validation for prepared slice-view data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .mdhisto import MDHistoData


_GRID_METADATA_ARRAYS = (
    "coverage_fraction",
    "normalization_denominator",
    "file_mask",
    "nfit_mask",
    "fit",
    "residual",
)


def mdhisto_source_identity(data: MDHistoData) -> tuple[Any, ...]:
    """Return identities for immutable arrays and replaceable metadata sources."""

    metadata_arrays = tuple(
        (name, id(value), getattr(value, "shape", None))
        for name in _GRID_METADATA_ARRAYS
        if (value := data.metadata.get(name)) is not None
    )
    auxiliary_arrays = tuple(
        (name, id(channel.values), id(channel.errors))
        for name, channel in sorted(data.auxiliary_channels.items())
    )
    return (
        id(data),
        id(data.signal),
        id(data.errors),
        id(data.mask),
        id(data.num_events),
        tuple((id(axis.values), axis.values.shape) for axis in data.axes),
        metadata_arrays,
        auxiliary_arrays,
        bool(data.metadata.get("zero_event_bins_are_measured", False)),
    )


def mdhisto_sources_are_cacheable(
    data: MDHistoData,
    *,
    visible_metadata_channels: Iterable[str] = (),
) -> bool:
    """Return whether cached numerical values cannot change in place.

    File and nfit masks are refreshed separately because they are cheap to
    slice and commonly remain mutable. Coverage, fit, and residual arrays can
    change the primary plotted values, so writable instances disable reuse.
    """

    if bool(getattr(data, "_arrays_mutable", False)) or any(
        array.flags.writeable
        for array in (data.signal, data.errors, data.mask, data.num_events)
    ):
        return False
    if any(axis.values.flags.writeable for axis in data.axes):
        return False
    mutable_names = {"coverage_fraction", "normalization_denominator"}
    mutable_names.update(visible_metadata_channels)
    for name in mutable_names:
        value = data.metadata.get(name)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            return False
        if value is not None and not isinstance(value, np.ndarray):
            return False
    return True
