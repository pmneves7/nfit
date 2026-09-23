"""Live 1D data-viewer windows for histogram box profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import PointListData
from .mdhisto import MDHistoAxis, MDHistoData


@dataclass(frozen=True)
class CutViewerContext:
    data: MDHistoData | PointListData
    x_dim: int
    y_dim: int
    axis_label: Callable[[int], str]
    dataset_name: str


def _profile_dataset(
    context: CutViewerContext,
    axis: str,
    cut: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> MDHistoData:
    coordinates, values, errors = cut
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size > 1:
        mids = (coordinates[:-1] + coordinates[1:]) / 2
        edges = np.r_[
            coordinates[0] - (mids[0] - coordinates[0]),
            mids,
            coordinates[-1] + (coordinates[-1] - mids[-1]),
        ]
    else:
        edges = np.array([coordinates[0] - 0.5, coordinates[0] + 0.5])
    dimension = context.x_dim if axis == "x" else context.y_dim
    source_axis = context.data.axes[dimension] if isinstance(context.data, MDHistoData) else None
    return MDHistoData(
        axes=(
            MDHistoAxis("Profile", np.array([0.0, 1.0]), "", "unknown"),
            MDHistoAxis(
                f"Box {axis} · {source_axis.name if source_axis else context.axis_label(dimension)}",
                edges,
                source_axis.units if source_axis else "",
                source_axis.kind if source_axis else "unknown",
            ),
        ),
        signal=np.asarray(values, dtype=float)[None, :],
        errors=np.asarray(errors, dtype=float)[None, :],
        mask=np.zeros((1, len(values)), dtype=bool),
        num_events=np.ones((1, len(values)), dtype=float),
    )


class LiveBoxCutViewers:
    """Own two reusable 1D viewers without changing project datasets."""

    def __init__(self, viewer_factory: Callable[..., Any], on_closed: Callable[[], None]):
        self._viewer_factory = viewer_factory
        self._on_closed = on_closed
        self.viewers: dict[str, Any] = {}

    def close(self) -> None:
        viewers = list(self.viewers.values())
        self.viewers.clear()
        for viewer in viewers:
            viewer.set_close_callback(None)
            viewer.window.close()

    def update(
        self,
        context: CutViewerContext,
        cuts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    ) -> None:
        for axis in ("x", "y"):
            cut = cuts.get(axis)
            if cut is None or len(cut[0]) == 0:
                continue
            data = _profile_dataset(context, axis, cut)
            name = f"{context.dataset_name} — {axis} box cut"
            viewer = self.viewers.get(axis)
            if viewer is None:
                viewer = self._viewer_factory(data, dataset_names=[name], x_dim=1, y_dim=0)
                viewer.window.setWindowTitle(name)
                viewer.set_close_callback(self._on_closed)
                self.viewers[axis] = viewer
                viewer.show()
            else:
                viewer.replace_datasets(data, dataset_names=[name])
