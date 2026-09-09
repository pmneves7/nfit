"""Installation-local preferences shared by nfit GUI windows."""

from __future__ import annotations

from PySide6 import QtCore

from .colormaps import IMAGE_COLORMAPS, WATERFALL_COLORMAPS

CONTINUOUS_COLORMAP_KEY = "colormaps/continuous_default"
WATERFALL_COLORMAP_KEY = "colormaps/waterfall_default"
DEFAULT_CONTINUOUS_COLORMAP = "viridis"
DEFAULT_WATERFALL_COLORMAP = "viridis"


def application_settings() -> QtCore.QSettings:
    """Return settings for this local nfit installation/user account."""

    return QtCore.QSettings("nfit", "nfit")


def _colormap_setting(
    key: str,
    *,
    available: tuple[str, ...],
    fallback: str,
    settings: QtCore.QSettings | None = None,
) -> str:
    store = application_settings() if settings is None else settings
    value = str(store.value(key, fallback))
    return value if value in available else fallback


def default_continuous_colormap(settings: QtCore.QSettings | None = None) -> str:
    """Return the local default for image-like continuous plots."""

    return _colormap_setting(
        CONTINUOUS_COLORMAP_KEY,
        available=IMAGE_COLORMAPS,
        fallback=DEFAULT_CONTINUOUS_COLORMAP,
        settings=settings,
    )


def default_waterfall_colormap(settings: QtCore.QSettings | None = None) -> str:
    """Return the local default for waterfall trace sequences."""

    return _colormap_setting(
        WATERFALL_COLORMAP_KEY,
        available=WATERFALL_COLORMAPS,
        fallback=DEFAULT_WATERFALL_COLORMAP,
        settings=settings,
    )


def set_default_continuous_colormap(
    name: str,
    settings: QtCore.QSettings | None = None,
) -> None:
    """Persist the local default for image-like continuous plots."""

    if name not in IMAGE_COLORMAPS:
        raise ValueError(f"Unknown continuous colormap {name!r}")
    (application_settings() if settings is None else settings).setValue(
        CONTINUOUS_COLORMAP_KEY,
        name,
    )


def set_default_waterfall_colormap(
    name: str,
    settings: QtCore.QSettings | None = None,
) -> None:
    """Persist the local default for waterfall trace sequences."""

    if name not in WATERFALL_COLORMAPS:
        raise ValueError(f"Unknown waterfall colormap {name!r}")
    (application_settings() if settings is None else settings).setValue(
        WATERFALL_COLORMAP_KEY,
        name,
    )
