"""Machine-local performance defaults and scoped CPU allocation (no Qt required)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def peak_process_memory_mib() -> float:
    """Return the process's peak resident memory in MiB on each supported OS."""
    if sys.platform != "win32":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024**2 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 1024**2


def performance_settings_path() -> Path:
    """Return the machine-local preferences file; override with NFIT_PERFORMANCE_FILE."""
    return Path(os.environ.get("NFIT_PERFORMANCE_FILE", "~/.config/nfit/performance.json")).expanduser()


def load_performance_settings() -> dict[str, int]:
    """Read preferences. Zero selects the documented automatic value."""
    try:
        value = json.loads(performance_settings_path().read_text())
        return _validated(value)
    except (OSError, ValueError, TypeError, AttributeError):
        return {"max_batch_mb": 0, "workers": 0, "transient_memory_percent": 0}


def _validated(value) -> dict[str, int]:
    result = {
        key: int(value.get(key, 0))
        for key in ("max_batch_mb", "workers", "transient_memory_percent")
    }
    if (
        not 0 <= result["max_batch_mb"] <= 1_048_576
        or not 0 <= result["workers"] <= 4096
        or not 0 <= result["transient_memory_percent"] <= 80
    ):
        raise ValueError(
            "Batch memory must be 0–1048576 MiB, workers 0–4096, and transient memory 0–80%."
        )
    return result


def save_performance_settings(
    *,
    max_batch_mb: int = 0,
    workers: int = 0,
    transient_memory_percent: int = 0,
) -> None:
    """Atomically save defaults for newly initialized rebin configurations."""
    values = _validated(
        {
            "max_batch_mb": max_batch_mb,
            "workers": workers,
            "transient_memory_percent": transient_memory_percent,
        }
    )
    path = performance_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(values, stream, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def transient_rebin_memory_limit_bytes(available_memory: int | None = None) -> int:
    """Return the machine-local ceiling for rebin batches and worker buffers."""

    if available_memory is None:
        try:
            import psutil

            available_memory = int(psutil.virtual_memory().available)
        except (ImportError, AttributeError):
            available_memory = None
    percent = load_performance_settings()["transient_memory_percent"] or 25
    if available_memory is None:
        return 512 * 1024**2
    return max(1, int(available_memory) * int(percent) // 100)


def initialize_rebin_performance(config: dict) -> None:
    """Snapshot machine defaults without replacing existing saved values."""
    if "max_batch_mb" in config and "workers" in config:
        return

    from ._parallel import num_threads

    settings = load_performance_settings()
    config.setdefault("max_batch_mb", settings["max_batch_mb"] or 192)
    config.setdefault("workers", settings["workers"] or num_threads())
