"""Shared projection/accumulation dispatch for native DGS event reducers.

The numerical policy owns affine preparation. This service optionally fuses
that transform with ordered accumulation on uniform Mantid grids and physical float64 grids;
arbitrary projectors and grids retain the caller's established accumulation.
"""

import numpy as np

try:
    from numba import config as _numba_config

    if _numba_config.DISABLE_JIT:
        _compiled = _compiled_high_precision = None
    else:
        from ._dgs_event_numba import (
            accumulate_high_precision_projected_events as _compiled_high_precision,
        )
        from ._dgs_event_numba import (
            accumulate_uniform_projected_events as _compiled,
        )
except Exception:
    _compiled = _compiled_high_precision = None


def accumulate_projected_dgs_events(
    projector, q_sample, energy, weights, variances, shape,
    data_sum, variance_sum, event_count, *, fallback_accumulator,
    enabled=None, bin_indices=None,
):
    """Accumulate one streamed event chunk and one symmetry copy in source order.

    ``projector`` is the callable returned by ``prepare_dgs_event_projector``.
    The fallback callback is passed by the reducer, so this service imports no
    importer facade. Same-event copy covariance uses these exact bin indices
    through the existing correction service. Histogram arrays are caller-owned
    writable accumulators; event and scientific configuration arrays are read-only.
    """
    affine = getattr(projector, "_dgs_uniform_affine", None)
    high_precision = getattr(projector, "_dgs_float64_projection", None)
    if ((affine is not None and _compiled is not None)
            or (high_precision is not None and _compiled_high_precision is not None)):
        prepared_shape = affine[2] if affine is not None else tuple(len(edge)-1 for edge in high_precision[3])
        q_sample, energy, weights = np.asarray(q_sample), np.asarray(energy), np.asarray(weights)
        variances = None if variances is None else np.asarray(variances)
        enabled = None if enabled is None else np.asarray(enabled)
        if (energy.ndim != 1 or q_sample.shape != (energy.size, 3) or weights.shape != energy.shape
                or (variances is not None and variances.shape != energy.shape)
                or (enabled is not None and enabled.shape != energy.shape)
                or (bin_indices is not None and bin_indices.shape != energy.shape)):
            raise ValueError("Projected DGS events require matching one-dimensional event arrays and N×3 Q")
        shape = np.asarray(shape, dtype=np.int64)
        if shape.shape != (4,) or tuple(shape) != prepared_shape or any(output.size != int(np.prod(shape))
                                     or not output.flags.c_contiguous or not output.flags.writeable
                                     for output in (data_sum, variance_sum, event_count)):
            raise ValueError("Projected DGS events require a four-dimensional grid and writable contiguous sums")
        outputs = (data_sum.ravel(), variance_sum.ravel(), event_count.ravel(), bin_indices)
        if affine is not None:
            matrix, offset, _ = affine
            _compiled(q_sample, energy, weights, variances, enabled, matrix, offset, shape, *outputs)
        else:
            _compiled_high_precision(q_sample, energy, weights, variances, enabled, *high_precision, shape, *outputs)
        return
    coordinates, edges = projector(q_sample, energy)
    fallback_accumulator(coordinates, weights, variances, edges, shape,
        data_sum, variance_sum, event_count, enabled=enabled, bin_indices=bin_indices)
