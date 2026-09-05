"""Machine-local performance defaults and scoped CPU allocation (no Qt required)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def performance_settings_path() -> Path:
    """Return the machine-local preferences file; override with NFIT_PERFORMANCE_FILE."""
    return Path(os.environ.get("NFIT_PERFORMANCE_FILE", "~/.config/nfit/performance.json")).expanduser()


def load_performance_settings() -> dict[str, int]:
    """Read preferences. Zero means automatic (192 MiB or available CPU allocation)."""
    try:
        value = json.loads(performance_settings_path().read_text())
        return _validated(value)
    except (OSError, ValueError, TypeError, AttributeError):
        return {"max_batch_mb": 0, "workers": 0}


def _validated(value) -> dict[str, int]:
    result = {key: int(value.get(key, 0)) for key in ("max_batch_mb", "workers")}
    if not 0 <= result["max_batch_mb"] <= 1_048_576 or not 0 <= result["workers"] <= 4096:
        raise ValueError("Batch memory must be 0–1048576 MiB and workers 0–4096.")
    return result


def save_performance_settings(*, max_batch_mb: int = 0, workers: int = 0) -> None:
    """Atomically save defaults for newly initialized rebin configurations."""
    values = _validated({"max_batch_mb": max_batch_mb, "workers": workers})
    path = performance_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(values, stream, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_rebin_performance(config: dict) -> None:
    """Snapshot machine defaults without replacing existing saved values."""
    if "max_batch_mb" in config and "workers" in config:
        return

    from ._parallel import num_threads

    settings = load_performance_settings()
    config.setdefault("max_batch_mb", settings["max_batch_mb"] or 192)
    config.setdefault("workers", settings["workers"] or num_threads())
