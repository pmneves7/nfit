"""Physical axis scaling for two-dimensional histogram views."""

from __future__ import annotations

import numpy as np

from .analysis.coordinates import physical_axis_vectors, rlu_to_q_matrix
from .mdhisto import MDHistoData


def mdhisto_axes_aspect(
    data: MDHistoData,
    x_dim: int,
    y_dim: int,
    mode: str = "fit",
    *,
    lattice_parameters: dict[str, float] | None = None,
) -> float | str:
    """Return Matplotlib's y/x display scale for the selected axis metric.

    ``q`` makes a unit along either displayed direction occupy space in
    proportion to its reciprocal-space length. It does not shear oblique axes.
    """

    if mode == "fit":
        return "auto"
    if mode not in {"rlu", "q"}:
        raise ValueError(f"unknown axes ratio mode: {mode!r}")
    if data.axes[x_dim].kind != "momentum" or data.axes[y_dim].kind != "momentum":
        raise ValueError("physical axes ratios require two momentum axes")
    vectors = physical_axis_vectors(data)[:, :3]
    if mode == "q":
        metadata = dict(data.metadata)
        if lattice_parameters is not None:
            metadata["lattice_parameters"] = lattice_parameters
        vectors = vectors @ rlu_to_q_matrix(metadata).T
    lengths = np.linalg.norm(vectors[[x_dim, y_dim]], axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("displayed momentum axes need nonzero direction vectors")
    return float(lengths[1] / lengths[0])
