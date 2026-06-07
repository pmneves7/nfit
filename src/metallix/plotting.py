from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .dataset import PointData4D


def plot_energy_cut(
    data: PointData4D,
    *,
    ax=None,
    model_values: ArrayLike | None = None,
    label: str = "data",
):
    """Plot intensity versus energy transfer with optional model overlay."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    order = np.argsort(data.E)
    ax.errorbar(
        data.E[order],
        data.intensity[order],
        yerr=data.sigma[order],
        fmt="o",
        ms=4,
        label=label,
    )
    if model_values is not None:
        model = np.asarray(model_values, dtype=float)
        ax.plot(data.E[order], model[order], "-", label="model")
    ax.set_xlabel("Energy transfer E (meV)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return ax


def plot_q_cut(
    data: PointData4D,
    coordinate: str = "H",
    *,
    ax=None,
    model_values: ArrayLike | None = None,
    label: str = "data",
):
    """Plot intensity versus one reciprocal-lattice coordinate."""

    import matplotlib.pyplot as plt

    if coordinate not in {"H", "K", "L"}:
        raise ValueError("coordinate must be one of 'H', 'K', or 'L'")
    x = getattr(data, coordinate)
    order = np.argsort(x)
    if ax is None:
        _, ax = plt.subplots()
    ax.errorbar(x[order], data.intensity[order], yerr=data.sigma[order], fmt="o", ms=4, label=label)
    if model_values is not None:
        model = np.asarray(model_values, dtype=float)
        ax.plot(x[order], model[order], "-", label="model")
    ax.set_xlabel(f"{coordinate} (RLU)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return ax


def plot_2d_map(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    *,
    ax=None,
    xlabel: str = "H (RLU)",
    ylabel: str = "K (RLU)",
    clabel: str = "Intensity",
    gridsize: int = 60,
):
    """Plot a 2D map from point-cloud values.

    The helper uses ``tricontourf`` for irregular point clouds and falls back to
    scatter if triangulation is not possible.
    """

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    try:
        triangulation = mtri.Triangulation(x_arr, y_arr)
        artist = ax.tricontourf(triangulation, v_arr, levels=gridsize)
    except Exception:
        artist = ax.scatter(x_arr, y_arr, c=v_arr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = ax.figure.colorbar(artist, ax=ax)
    cbar.set_label(clabel)
    return ax

