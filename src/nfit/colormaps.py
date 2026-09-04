"""Shared Matplotlib and Colorcet colormap catalogs."""

from __future__ import annotations

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

NFIT_CONTINUOUS_COLORMAPS = (
    "bluewhitered",
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

IMAGE_COLORMAPS = (
    MATPLOTLIB_IMAGE_COLORMAPS
    + COLORCET_CONTINUOUS_COLORMAPS
    + NFIT_CONTINUOUS_COLORMAPS
)
IMAGE_COLORMAP_GROUPS = (
    MATPLOTLIB_IMAGE_COLORMAPS,
    COLORCET_CONTINUOUS_COLORMAPS,
    NFIT_CONTINUOUS_COLORMAPS,
)
WATERFALL_COLORMAPS = (
    MATPLOTLIB_WATERFALL_COLORMAPS
    + COLORCET_CONTINUOUS_COLORMAPS
    + NFIT_CONTINUOUS_COLORMAPS
    + COLORCET_CATEGORICAL_COLORMAPS
)
WATERFALL_COLORMAP_GROUPS = (
    MATPLOTLIB_WATERFALL_COLORMAPS,
    COLORCET_CONTINUOUS_COLORMAPS,
    NFIT_CONTINUOUS_COLORMAPS,
    COLORCET_CATEGORICAL_COLORMAPS,
)
WATERFALL_DISCRETE_COLORMAPS = {
    "tab10",
    "Dark2",
    "Set1",
    *COLORCET_CATEGORICAL_COLORMAPS,
}


def populate_qt_colormap_combo(combo, groups: tuple[tuple[str, ...], ...]) -> None:
    """Fill a Qt combo with previews and separators between nonempty groups."""

    from matplotlib import colormaps
    from PySide6 import QtCore, QtGui

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
