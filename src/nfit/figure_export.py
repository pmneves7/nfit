"""Figure image export helpers that do not depend on a GUI toolkit."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

DEFAULT_FIGURE_DPI = 600

_FORMAT_PRIORITY = (
    "Portable Network Graphics",
    "Encapsulated Postscript",
    "Tagged Image File Format",
    "Scalable Vector Graphics",
    "Joint Photographic Experts Group",
    "Portable Document Format",
    "Postscript",
    "WebP Image Format",
)
_EXCLUDED_GROUPS = {"Raw RGBA bitmap", "PGF code for LaTeX"}


def _validate_dpi(dpi: Real) -> float:
    if isinstance(dpi, bool) or not isinstance(dpi, Real):
        raise ValueError("dpi must be a finite positive number")
    value = float(dpi)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("dpi must be a finite positive number")
    return value


def save_figure(
    figure: Any,
    path: str | Path,
    *,
    dpi: Real = DEFAULT_FIGURE_DPI,
) -> Path:
    """Save a Matplotlib-like figure at ``dpi`` and return its destination.

    The output format comes from the filename suffix. A missing suffix is
    treated as PNG. The figure's size, DPI, axes, and limits are left intact.
    """

    resolved_dpi = _validate_dpi(dpi)
    destination = Path(path).expanduser()
    suffix = destination.suffix.lower().lstrip(".")
    if not suffix:
        destination = destination.with_suffix(".png")
        suffix = "png"
    supported = figure.canvas.get_supported_filetypes()
    if suffix not in supported:
        choices = ", ".join(f".{item}" for item in sorted(supported))
        raise ValueError(
            f"unsupported figure format '.{suffix}'; supported formats: {choices}"
        )
    figure.savefig(destination, format=suffix, dpi=resolved_dpi)
    return destination


def figure_export_filters(figure: Any) -> dict[str, str]:
    """Return Qt save-dialog filters mapped to their default filename suffix.

    Backend group names come directly from Matplotlib. Common output formats
    appear first in a predictable order; any remaining supported groups follow
    alphabetically. Aliases are combined into one filter per format group.
    """

    grouped = figure.canvas.get_supported_filetypes_grouped()
    groups = [
        (name, sorted(set(extensions)))
        for name, extensions in grouped.items()
        if name not in _EXCLUDED_GROUPS and extensions
    ]
    priority = {name: index for index, name in enumerate(_FORMAT_PRIORITY)}
    groups.sort(key=lambda item: (priority.get(item[0], len(priority)), item[0]))
    filters: dict[str, str] = {}
    for name, extensions in groups:
        patterns = " ".join(f"*.{extension}" for extension in extensions)
        filters[f"{name} ({patterns})"] = f".{extensions[0]}"
    return filters


def figure_export_script(
    script: str,
    path: str | Path | None,
) -> str:
    """Add a high-resolution save step before a generated script's final show.

    ``None`` leaves the script unchanged. The path is embedded as a Python
    string literal so spaces, quotes, and non-ASCII characters remain safe.
    """

    if path is None:
        return script
    show_statement = "plt.show()"
    position = script.rfind(show_statement)
    if position < 0:
        raise ValueError("figure script must end with plt.show()")
    save_lines = (
        "from nfit import save_figure\n"
        f"save_figure(plt.gcf(), {str(path)!r}, dpi={DEFAULT_FIGURE_DPI!r})\n"
    )
    return script[:position] + save_lines + script[position:]
