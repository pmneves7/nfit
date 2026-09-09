"""Public plotting API and opt-in Qt slice-viewer launcher.

The plotting implementation is kept in :mod:`nfit.plotting_core`, which has no
dependency on Qt presentation modules. This compatibility facade preserves the
long-standing ``nfit.plotting`` import path while keeping the optional Qt
launcher at the outer edge of the dependency graph.
"""

from __future__ import annotations

from typing import Any

from . import plotting_core as _core
from .mdhisto import MDHistoData
from .plotting_core import *  # noqa: F403 - compatibility facade
from .plotting_core import MDHistoSliceViewer


def slice_viewer(data: MDHistoData, **kwargs: Any) -> MDHistoSliceViewer:
    """Open an interactive Qt MD histogram slice viewer.

    Parameters are forwarded to :class:`nfit.qt_slice_viewer.QtMDHistoSliceViewer`.
    The returned viewer object owns the Qt window. In scripts, call
    ``viewer.run()`` to start the Qt event loop; in notebooks with ``%gui qt``,
    keeping the returned object assigned is usually sufficient.

    ``viewer = slice_viewer(data)``
    """

    from .qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(data, **kwargs)
    viewer.show()
    return viewer


def __getattr__(name: str) -> Any:
    """Delegate private compatibility imports to the implementation module."""

    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_core)})


__all__ = [
    *(name for name in vars(_core) if not name.startswith("_")),
    "slice_viewer",
]
