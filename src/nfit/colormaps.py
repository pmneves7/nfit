"""Shared scientific and custom colormap catalogs."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

MATPLOTLIB_IMAGE_COLORMAPS = (
    "viridis",
    "magma",
    "plasma",
    "cividis",
    "turbo",
    "grey",
    "Spectral",
)

MATPLOTLIB_VOLUME_COLORMAPS = (
    "viridis",
    "magma",
    "plasma",
    "cividis",
    "turbo",
    "coolwarm",
    "grey",
    "Spectral",
)

MATPLOTLIB_WATERFALL_COLORMAPS = (
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "cividis",
    "turbo",
    "tab10",
    "Dark2",
    "Set1",
    "Blues",
    "Reds",
    "Spectral",
)

# Matplotlib's standard ColorBrewer-derived scientific families.  Keep these
# separate from the short, long-standing menu prefixes so the GUI can show
# useful section breaks without changing the order of existing choices.
MATPLOTLIB_SEQUENTIAL_COLORMAPS = (
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
)

MATPLOTLIB_DIVERGING_COLORMAPS = (
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
)

MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS = (
    "cubehelix",
    "CMRmap",
    "gnuplot2",
    "twilight",
    "twilight_shifted",
)

MATPLOTLIB_QUALITATIVE_COLORMAPS = (
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
)

NFIT_CONTINUOUS_COLORMAPS = (
    "young_rdbu",
    "young_ylbkcy",
    "young_quadratic",
)

# Control points transcribed from the user's YoungColorMap.m; the MATLAB
# implementation divides by 256; the registered maps use a centered midpoint.
_YOUNG_RDBU = (
    (5, 48, 97),
    (32, 100, 170),
    (65, 145, 194),
    (144, 196, 221),
    (209, 229, 240),
    (246, 246, 246),
    (253, 219, 199),
    (243, 163, 128),
    (213, 95, 76),
    (176, 23, 42),
    (103, 0, 31),
)
_YOUNG_YLBKCY = (
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (0, 0, 191),
    (0, 0, 127),
    (0, 0, 63),
    (0, 0, 0),
    (63, 0, 0),
    (127, 0, 0),
    (191, 0, 0),
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
)
_YOUNG_QUADRATIC = (
    (0, 255, 255),
    (0, 206.55, 252.45),
    (0, 163.2, 244.8),
    (0, 124.95, 232.05),
    (0, 91.8, 214.2),
    (0, 63.75, 191.25),
    (0, 40.8, 163.2),
    (0, 22.95, 130.05),
    (0, 10.2, 91.8),
    (0, 2.55, 48.45),
    (0, 0, 0),
    (48.45, 2.55, 0),
    (91.8, 10.2, 0),
    (130.05, 22.95, 0),
    (163.2, 40.8, 0),
    (191.25, 63.75, 0),
    (214.2, 91.8, 0),
    (232.05, 124.95, 0),
    (244.8, 163.2, 0),
    (252.45, 206.55, 0),
    (255, 255, 0),
)


def _register_nfit_colormaps() -> None:
    definitions = {
        # Retain the old name for stored-plot and script compatibility, but do
        # not expose this exact alias of Matplotlib's bwr in the GUI catalog.
        "bluewhitered": (
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        "young_rdbu": tuple(
            tuple(channel / 256.0 for channel in color) for color in _YOUNG_RDBU
        ),
        "young_ylbkcy": tuple(
            tuple(channel / 256.0 for channel in color) for color in _YOUNG_YLBKCY
        ),
        "young_quadratic": tuple(
            tuple(channel / 256.0 for channel in color)
            for color in _YOUNG_QUADRATIC
        ),
    }
    for name, colors in definitions.items():
        if name not in colormaps:
            colormaps.register(LinearSegmentedColormap.from_list(name, colors))


_register_nfit_colormaps()


def _named_colorcet_colormaps() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return Colorcet's short named maps split by continuous/categorical use."""

    try:
        import colorcet as cc
    except ImportError:  # Keep source checkouts usable before dependencies install.
        return (), ()
    continuous: list[str] = []
    categorical: list[str] = []
    for name, cmap in cc.cm_n.items():
        if name.endswith("_r"):
            continue
        target = categorical if isinstance(cmap, ListedColormap) else continuous
        target.append(f"cet_{name}")
    return tuple(continuous), tuple(categorical)


COLORCET_CONTINUOUS_COLORMAPS, COLORCET_CATEGORICAL_COLORMAPS = (
    _named_colorcet_colormaps()
)


def _cmcrameri_sequential_colormaps() -> tuple[str, ...]:
    """Register full-resolution Scientific Colour Maps via cmcrameri.

    Names follow cmcrameri 1.10's sequential classification, excluding its
    categorical, cyclic, diverging, and multi-sequential families.
    """

    try:
        from cmcrameri import cm
    except ImportError:  # Allow source checkouts before dependencies install.
        return ()
    names = (
        "batlow", "batlowW", "batlowK", "devon", "lajolla", "bamako",
        "davos", "bilbao", "nuuk", "oslo", "grayC", "hawaii", "lapaz",
        "tokyo", "buda", "acton", "turku", "imola", "glasgow", "lipari",
        "navia",
    )
    return tuple(f"cmc.{name}" for name in names if name in cm.cmaps)


CMCRAMERI_SEQUENTIAL_COLORMAPS = _cmcrameri_sequential_colormaps()


def user_colormap_directory() -> Path:
    """Return the drop-in RGB-table directory (override with NFIT_COLORMAP_DIR)."""

    return Path(os.environ.get("NFIT_COLORMAP_DIR", str(Path.home() / "nfit_colormaps"))).expanduser()


def load_colormap_file(path: str | Path) -> str:
    """Register a CSV/whitespace RGB table and its reverse; return its user_ name.

    Rows are evenly spaced from low to high. Values must be finite and within
    0–1, or integer 0–255 if any value exceeds one. Hash comments are allowed.
    Existing registered names cannot be replaced.
    """

    path = Path(path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", path.stem) or path.stem.endswith("_r"):
        raise ValueError("Use letters, digits, underscores or hyphens; the filename must not end in _r.")
    lines = path.read_text(encoding="utf-8-sig").replace(",", " ").splitlines()
    rgb = np.loadtxt(lines, comments="#", ndmin=2)
    if rgb.shape[0] < 2 or rgb.shape[1] != 3 or not np.all(np.isfinite(rgb)):
        raise ValueError("Expected at least two rows of three finite RGB values.")
    if np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError("RGB values must lie within 0–1 or 0–255.")
    if np.max(rgb) > 1:
        if not np.all(rgb == np.floor(rgb)):
            raise ValueError("RGB values above one must use integer 0–255 units.")
        rgb = rgb / 255.0
    name = f"user_{path.stem}"
    if name in colormaps or f"{name}_r" in colormaps:
        raise ValueError(f"Colormap already registered: {name}")
    cmap = LinearSegmentedColormap.from_list(name, rgb, N=max(256, len(rgb)))
    colormaps.register(cmap)
    colormaps.register(cmap.reversed())
    return name


def _load_user_colormaps() -> tuple[str, ...]:
    names = []
    directory = user_colormap_directory()
    try:
        paths = sorted(directory.iterdir()) if directory.is_dir() else []
    except OSError as exc:
        warnings.warn(f"Cannot read colormap folder {directory}: {exc}", stacklevel=2)
        return ()
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {".csv", ".txt", ".rgb"}:
            continue
        try:
            names.append(load_colormap_file(path))
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.warn(f"Skipping colormap {path.name}: {exc}", stacklevel=2)
    return tuple(names)


USER_COLORMAPS = _load_user_colormaps()


def _new_colormaps(
    names: tuple[str, ...], existing: tuple[str, ...]
) -> tuple[str, ...]:
    """Return catalog entries not already present in a menu's leading group."""

    return tuple(name for name in names if name not in existing)


_IMAGE_SEQUENTIAL_COLORMAPS = _new_colormaps(
    MATPLOTLIB_SEQUENTIAL_COLORMAPS, MATPLOTLIB_IMAGE_COLORMAPS
)
_IMAGE_DIVERGING_COLORMAPS = _new_colormaps(
    MATPLOTLIB_DIVERGING_COLORMAPS, MATPLOTLIB_IMAGE_COLORMAPS
)
_VOLUME_SEQUENTIAL_COLORMAPS = _new_colormaps(
    MATPLOTLIB_SEQUENTIAL_COLORMAPS, MATPLOTLIB_VOLUME_COLORMAPS
)
_VOLUME_DIVERGING_COLORMAPS = _new_colormaps(
    MATPLOTLIB_DIVERGING_COLORMAPS, MATPLOTLIB_VOLUME_COLORMAPS
)
_WATERFALL_SEQUENTIAL_COLORMAPS = _new_colormaps(
    MATPLOTLIB_SEQUENTIAL_COLORMAPS, MATPLOTLIB_WATERFALL_COLORMAPS
)
_WATERFALL_DIVERGING_COLORMAPS = _new_colormaps(
    MATPLOTLIB_DIVERGING_COLORMAPS, MATPLOTLIB_WATERFALL_COLORMAPS
)
_WATERFALL_QUALITATIVE_COLORMAPS = _new_colormaps(
    MATPLOTLIB_QUALITATIVE_COLORMAPS, MATPLOTLIB_WATERFALL_COLORMAPS
)

IMAGE_COLORMAPS = (
    MATPLOTLIB_IMAGE_COLORMAPS
    + _IMAGE_SEQUENTIAL_COLORMAPS
    + _IMAGE_DIVERGING_COLORMAPS
    + MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS
    + COLORCET_CONTINUOUS_COLORMAPS
    + CMCRAMERI_SEQUENTIAL_COLORMAPS
    + NFIT_CONTINUOUS_COLORMAPS
    + USER_COLORMAPS
)
IMAGE_COLORMAP_GROUPS = (
    MATPLOTLIB_IMAGE_COLORMAPS,
    _IMAGE_SEQUENTIAL_COLORMAPS,
    _IMAGE_DIVERGING_COLORMAPS,
    MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS,
    COLORCET_CONTINUOUS_COLORMAPS,
    CMCRAMERI_SEQUENTIAL_COLORMAPS,
    NFIT_CONTINUOUS_COLORMAPS,
    USER_COLORMAPS,
)
VOLUME_COLORMAPS = (
    MATPLOTLIB_VOLUME_COLORMAPS
    + _VOLUME_SEQUENTIAL_COLORMAPS
    + _VOLUME_DIVERGING_COLORMAPS
    + MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS
    + COLORCET_CONTINUOUS_COLORMAPS
    + CMCRAMERI_SEQUENTIAL_COLORMAPS
    + NFIT_CONTINUOUS_COLORMAPS
    + USER_COLORMAPS
)
VOLUME_COLORMAP_GROUPS = (
    MATPLOTLIB_VOLUME_COLORMAPS,
    _VOLUME_SEQUENTIAL_COLORMAPS,
    _VOLUME_DIVERGING_COLORMAPS,
    MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS,
    COLORCET_CONTINUOUS_COLORMAPS,
    CMCRAMERI_SEQUENTIAL_COLORMAPS,
    NFIT_CONTINUOUS_COLORMAPS,
    USER_COLORMAPS,
)
WATERFALL_COLORMAPS = (
    MATPLOTLIB_WATERFALL_COLORMAPS
    + _WATERFALL_SEQUENTIAL_COLORMAPS
    + _WATERFALL_DIVERGING_COLORMAPS
    + MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS
    + COLORCET_CONTINUOUS_COLORMAPS
    + CMCRAMERI_SEQUENTIAL_COLORMAPS
    + NFIT_CONTINUOUS_COLORMAPS
    + _WATERFALL_QUALITATIVE_COLORMAPS
    + COLORCET_CATEGORICAL_COLORMAPS
    + USER_COLORMAPS
)
WATERFALL_COLORMAP_GROUPS = (
    MATPLOTLIB_WATERFALL_COLORMAPS,
    _WATERFALL_SEQUENTIAL_COLORMAPS,
    _WATERFALL_DIVERGING_COLORMAPS,
    MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS,
    COLORCET_CONTINUOUS_COLORMAPS,
    CMCRAMERI_SEQUENTIAL_COLORMAPS,
    NFIT_CONTINUOUS_COLORMAPS,
    _WATERFALL_QUALITATIVE_COLORMAPS,
    COLORCET_CATEGORICAL_COLORMAPS,
    USER_COLORMAPS,
)
WATERFALL_DISCRETE_COLORMAPS = {
    "tab10",
    "Dark2",
    "Set1",
    *MATPLOTLIB_QUALITATIVE_COLORMAPS,
    *COLORCET_CATEGORICAL_COLORMAPS,
}


def populate_qt_colormap_combo(combo, groups: tuple[tuple[str, ...], ...]) -> None:
    """Fill a Qt combo with previews and separators between nonempty groups."""

    from matplotlib import colormaps
    from PySide6 import QtCore, QtGui, QtWidgets

    def open_folder() -> None:
        directory = user_colormap_directory()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory.resolve()))):
                raise OSError(f"Could not open {directory}")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(combo, "Colormap folder", str(exc))

    combo.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
    folder_action = QtGui.QAction("Open custom colormap folder…", combo)
    folder_action.setToolTip("Add RGB .csv, .txt or .rgb tables, then restart nfit to load them.")
    folder_action.triggered.connect(open_folder)
    combo.addAction(folder_action)

    preview_width = 96
    preview_height = 14
    combo.setIconSize(QtCore.QSize(preview_width, preview_height))

    for group in groups:
        if not group:
            continue
        if combo.count():
            combo.insertSeparator(combo.count())
        for name in group:
            registered_name = "gray" if name == "grey" else name
            cmap = colormaps.get_cmap(registered_name)
            image = QtGui.QImage(
                preview_width,
                preview_height,
                QtGui.QImage.Format.Format_ARGB32,
            )
            painter = QtGui.QPainter(image)
            for position in range(preview_width):
                rgba = cmap(position / (preview_width - 1))
                painter.fillRect(
                    position,
                    0,
                    1,
                    preview_height,
                    QtGui.QColor.fromRgbF(*rgba),
                )
            painter.end()
            combo.addItem(QtGui.QIcon(QtGui.QPixmap.fromImage(image)), name)
