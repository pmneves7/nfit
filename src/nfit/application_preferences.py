"""Installation-local preferences shared by nfit GUI windows."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from .colormaps import IMAGE_COLORMAPS, WATERFALL_COLORMAPS

CONTINUOUS_COLORMAP_KEY = "colormaps/continuous_default"
WATERFALL_COLORMAP_KEY = "colormaps/waterfall_default"
PRELOAD_VIEWER_DATA_KEY = "viewer/preload_all_data"
VIEWER_FONT_SIZE_KEY = "viewer/font_size"
VIEWER_AXIS_LINEWIDTH_KEY = "viewer/axis_linewidth"
DEFAULT_CONTINUOUS_COLORMAP = "viridis"
DEFAULT_WATERFALL_COLORMAP = "viridis"
DEFAULT_VIEWER_FONT_SIZE = 12.0
DEFAULT_VIEWER_AXIS_LINEWIDTH = 1.5


class _LinuxApplicationSettings:
    """Small QSettings-compatible store without advisory filesystem locks."""

    _lock = threading.RLock()

    def __init__(self) -> None:
        config_root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        self._path = config_root / "nfit" / "settings.json"
        self._status = QtCore.QSettings.Status.NoError
        self._legacy = QtCore.QSettings("nfit", "nfit")
        self._legacy.setAtomicSyncRequired(False)

    def fileName(self) -> str:
        return str(self._path)

    def isAtomicSyncRequired(self) -> bool:
        return False

    def status(self) -> QtCore.QSettings.Status:
        return self._status

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            self._status = QtCore.QSettings.Status.FormatError
            return {}
        if not isinstance(payload, dict):
            self._status = QtCore.QSettings.Status.FormatError
            return {}
        return payload

    def _write(self, values: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(values, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError) as exc:
            self._status = (
                QtCore.QSettings.Status.AccessError
                if isinstance(exc, OSError)
                else QtCore.QSettings.Status.FormatError
            )
        else:
            self._status = QtCore.QSettings.Status.NoError

    def value(
        self,
        key: str,
        default: Any = None,
        type: type | None = None,
    ) -> Any:
        with self._lock:
            values = self._read()
            value = values[key] if key in values else self._legacy.value(key, default)
        if type is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if type is not None and value is not None and not isinstance(value, type):
            try:
                return type(value)
            except (TypeError, ValueError):
                return default
        return value

    def setValue(self, key: str, value: Any) -> None:
        with self._lock:
            values = self._read()
            values[str(key)] = value
            self._write(values)

    def remove(self, key: str) -> None:
        with self._lock:
            values = self._read()
            prefix = f"{key}/"
            filtered = {
                name: value
                for name, value in values.items()
                if name != key and not name.startswith(prefix)
            }
            self._write(filtered)

    def sync(self) -> None:
        """Match QSettings; writes are already flushed by ``setValue``."""


def application_settings() -> QtCore.QSettings | _LinuxApplicationSettings:
    """Return settings for this local nfit installation/user account."""

    if sys.platform.startswith("linux"):
        return _LinuxApplicationSettings()
    return QtCore.QSettings("nfit", "nfit")


def _colormap_setting(
    key: str,
    *,
    available: tuple[str, ...],
    fallback: str,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> str:
    store = application_settings() if settings is None else settings
    value = str(store.value(key, fallback))
    return value if value in available else fallback


def default_continuous_colormap(
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> str:
    """Return the local default for image-like continuous plots."""

    return _colormap_setting(
        CONTINUOUS_COLORMAP_KEY,
        available=IMAGE_COLORMAPS,
        fallback=DEFAULT_CONTINUOUS_COLORMAP,
        settings=settings,
    )


def default_waterfall_colormap(
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> str:
    """Return the local default for waterfall trace sequences."""

    return _colormap_setting(
        WATERFALL_COLORMAP_KEY,
        available=WATERFALL_COLORMAPS,
        fallback=DEFAULT_WATERFALL_COLORMAP,
        settings=settings,
    )


def preload_viewer_data(
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> bool:
    """Return whether newly opened viewers should preload all selectable data."""

    store = application_settings() if settings is None else settings
    return bool(store.value(PRELOAD_VIEWER_DATA_KEY, False, type=bool))


def _positive_finite_setting(
    key: str,
    fallback: float,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None,
) -> float:
    store = application_settings() if settings is None else settings
    try:
        value = float(store.value(key, fallback))
    except (TypeError, ValueError, OverflowError):
        return fallback
    return value if value > 0.0 and math.isfinite(value) else fallback


def default_viewer_font_size(
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> float:
    """Return the local default font size for newly opened data viewers, in points."""

    return _positive_finite_setting(VIEWER_FONT_SIZE_KEY, DEFAULT_VIEWER_FONT_SIZE, settings)


def default_viewer_axis_linewidth(
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> float:
    """Return the local default axes and tick line width for new data viewers."""

    return _positive_finite_setting(
        VIEWER_AXIS_LINEWIDTH_KEY, DEFAULT_VIEWER_AXIS_LINEWIDTH, settings
    )


def set_default_continuous_colormap(
    name: str,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
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
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> None:
    """Persist the local default for waterfall trace sequences."""

    if name not in WATERFALL_COLORMAPS:
        raise ValueError(f"Unknown waterfall colormap {name!r}")
    (application_settings() if settings is None else settings).setValue(
        WATERFALL_COLORMAP_KEY,
        name,
    )


def set_preload_viewer_data(
    enabled: bool,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> None:
    """Persist whether newly opened viewers preload all selectable data."""

    (application_settings() if settings is None else settings).setValue(
        PRELOAD_VIEWER_DATA_KEY,
        bool(enabled),
    )


def _set_positive_finite_setting(
    key: str,
    value: float,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None,
    name: str,
) -> None:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if normalized <= 0.0 or not math.isfinite(normalized):
        raise ValueError(f"{name} must be a positive finite number")
    (application_settings() if settings is None else settings).setValue(key, normalized)


def set_default_viewer_font_size(
    value: float,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> None:
    """Persist the local default font size for new data viewers, in points."""

    _set_positive_finite_setting(VIEWER_FONT_SIZE_KEY, value, settings, "Font size")


def set_default_viewer_axis_linewidth(
    value: float,
    settings: QtCore.QSettings | _LinuxApplicationSettings | None = None,
) -> None:
    """Persist the local default axes and tick line width for new data viewers."""

    _set_positive_finite_setting(
        VIEWER_AXIS_LINEWIDTH_KEY, value, settings, "Axes line width"
    )
