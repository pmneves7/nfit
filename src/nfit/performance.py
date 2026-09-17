"""Machine-local performance defaults and scoped CPU allocation (no Qt required)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

REBIN_RESULT_BYTES_PER_BIN = 33
REBIN_CACHE_WARNING_FRACTION = 0.8
_DEFAULT_TRANSIENT_MEMORY_PERCENT = 25
_MEBIBYTE = 1024**2
_SCOPED_BATCH_BYTES: ContextVar[int | None] = ContextVar(
    "nfit_batch_bytes", default=None
)


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


def _darwin_available_memory_bytes() -> int | None:
    """Return free and reclaimable macOS memory from ``vm_stat``."""

    try:
        output = subprocess.check_output(
            ["/usr/bin/vm_stat"], text=True, timeout=5
        )
        page_match = re.search(r"page size of (\d+) bytes", output)
        if page_match is None:
            return None
        pages = {
            name: int(value)
            for name, value in re.findall(
                r"^Pages (free|inactive|speculative):\s+(\d+)\.",
                output,
                re.MULTILINE,
            )
        }
        if not pages:
            return None
        return int(page_match.group(1)) * sum(pages.values())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _windows_available_memory_bytes() -> int | None:
    """Return Windows' available physical-memory counter."""

    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)
    except (AttributeError, OSError):
        pass
    return None


def available_memory_bytes() -> int | None:
    """Return reclaimable physical memory, constrained by Linux cgroups."""

    available = None
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError, ValueError):
        if sys.platform == "darwin":
            available = _darwin_available_memory_bytes()
        elif sys.platform == "win32":
            available = _windows_available_memory_bytes()
        else:
            try:
                available = int(
                    os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                )
            except (AttributeError, OSError, ValueError):
                pass

    if sys.platform == "linux":
        for limit_path, usage_path in (
            (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
            (
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ),
        ):
            try:
                limit_text = limit_path.read_text().strip()
                if limit_text != "max":
                    remaining = max(
                        0, int(limit_text) - int(usage_path.read_text().strip())
                    )
                    available = (
                        remaining if available is None else min(available, remaining)
                    )
                    break
            except (OSError, ValueError):
                continue
    return None if available is None else max(1, available)


def assess_output_rebin_memory(
    output_bins: int, *, max_batch_bytes: int = 192 * 1024**2
) -> tuple[int, int | None, bool]:
    """Estimate peak array workspace for an ordinary dataset/grid rebin."""

    estimate = max(int(output_bins), 0) * 96 + max(int(max_batch_bytes), 0)
    available = available_memory_bytes()
    return estimate, available, available is not None and estimate > available // 2


def estimate_rebin_result_bytes(output_bins: int) -> int:
    """Estimate the numerical payload retained for an MDHisto rebin result."""

    return max(int(output_bins), 0) * REBIN_RESULT_BYTES_PER_BIN


def assess_rebin_cache_memory(
    result_bytes: list[int] | tuple[int, ...],
    *,
    current_cache_bytes: int,
    cache_limit_bytes: int,
) -> tuple[int, int, int, bool]:
    """Estimate whether a batch will approach the shared result-cache ceiling."""

    added = sum(max(int(value), 0) for value in result_bytes)
    current = max(int(current_cache_bytes), 0)
    limit = max(int(cache_limit_bytes), 0)
    projected = current + added
    warn = bool(
        result_bytes
        and limit > 0
        and projected >= int(limit * REBIN_CACHE_WARNING_FRACTION)
    )
    return added, projected, limit, warn


def performance_settings_path() -> Path:
    """Return the machine-local preferences file; override with NFIT_PERFORMANCE_FILE."""
    return Path(os.environ.get("NFIT_PERFORMANCE_FILE", "~/.config/nfit/performance.json")).expanduser()


def _raw_performance_settings() -> dict:
    """Read the preference file without exposing its storage schema."""
    try:
        value = json.loads(performance_settings_path().read_text())
        if not isinstance(value, dict):
            raise ValueError("Performance preferences must be an object.")
        return value
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def load_performance_settings() -> dict[str, int]:
    """Read legacy rebin defaults.

    This compatibility API retains its original three keys.  New callers should
    use :func:`load_resource_limits` for the application-wide CPU and RAM
    ceilings shown in Preferences.
    """

    raw = _raw_performance_settings()
    try:
        return _validated(
            {
                "max_batch_mb": raw.get("max_batch_mb", 0),
                "workers": raw.get("workers", raw.get("cpu_limit", 0)),
                "transient_memory_percent": raw.get("transient_memory_percent", 0),
            }
        )
    except (TypeError, ValueError, AttributeError):
        return {"max_batch_mb": 0, "workers": 0, "transient_memory_percent": 0}


def _validated_resource_limits(value: dict) -> dict[str, int]:
    result = {
        "cpu_limit": int(value.get("cpu_limit", value.get("workers", 0))),
        "ram_limit_mb": int(value.get("ram_limit_mb", 0)),
    }
    if not 0 <= result["cpu_limit"] <= 4096 or not 0 <= result["ram_limit_mb"] <= 1_048_576:
        raise ValueError(
            "CPU limit must be 0–4096 and RAM limit must be 0–1048576 MiB."
        )
    return result


def load_resource_limits() -> dict[str, int]:
    """Return global resource ceilings, with zero selecting automatic limits.

    ``workers`` in older preference files is treated as ``cpu_limit``.  Older
    percentage-based RAM settings remain active through
    :func:`scientific_memory_limit_bytes`; they are deliberately not converted
    to a fixed amount while merely reading preferences.
    """

    try:
        return _validated_resource_limits(_raw_performance_settings())
    except (TypeError, ValueError, AttributeError):
        return {"cpu_limit": 0, "ram_limit_mb": 0}


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
    cpu_limit: int | None = None,
    ram_limit_mb: int | None = None,
) -> None:
    """Atomically save resource limits and legacy rebin defaults.

    ``max_batch_mb``, ``workers``, and ``transient_memory_percent`` are kept
    for scripts and existing project workflows.  New preferences should pass
    ``cpu_limit`` and ``ram_limit_mb``.  A zero resource limit selects nfit's
    automatic policy.
    """
    values = _validated(
        {
            "max_batch_mb": max_batch_mb,
            "workers": workers,
            "transient_memory_percent": transient_memory_percent,
        }
    )
    limits = _validated_resource_limits(
        {
            "cpu_limit": values["workers"] if cpu_limit is None else cpu_limit,
            "ram_limit_mb": 0 if ram_limit_mb is None else ram_limit_mb,
        }
    )
    # Keep the legacy worker value synchronized so old scripts which read this
    # file through ``load_performance_settings`` see the global CPU ceiling.
    values["workers"] = limits["cpu_limit"]
    path = performance_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({**values, **limits}, stream, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_resource_limits(*, cpu_limit: int = 0, ram_limit_mb: int = 0) -> None:
    """Save global resource limits while retaining hidden legacy rebin defaults.

    Saving an explicit Auto RAM choice clears a migrated percentage setting, so
    Auto consistently means nfit's normal conservative policy.
    """

    legacy = load_performance_settings()
    legacy["transient_memory_percent"] = 0
    save_performance_settings(
        **legacy,
        cpu_limit=cpu_limit,
        ram_limit_mb=ram_limit_mb,
    )


def scientific_memory_limit_bytes(available_memory: int | None = None) -> int:
    """Return nfit's managed cache and temporary-workspace RAM ceiling.

    This is a shared budget for numerical caches and operations that opt into
    it.  It cannot cap Python, Qt, mapped files, libraries, or all process RSS.
    A fixed user ceiling is additionally bounded by currently available RAM.
    """

    if available_memory is None:
        available_memory = available_memory_bytes()
    limits = load_resource_limits()
    fixed = limits["ram_limit_mb"] * _MEBIBYTE
    if fixed:
        return max(1, min(fixed, int(available_memory))) if available_memory is not None else fixed
    percent = (
        load_performance_settings()["transient_memory_percent"]
        or _DEFAULT_TRANSIENT_MEMORY_PERCENT
    )
    if available_memory is None:
        return 512 * 1024**2
    return max(1, int(available_memory) * int(percent) // 100)


def rebin_memory_limit_bytes(available_memory: int | None = None) -> int:
    """Compatibility name for nfit's managed scientific-memory ceiling."""

    return scientific_memory_limit_bytes(available_memory)


def transient_rebin_memory_limit_bytes(available_memory: int | None = None) -> int:
    """Compatibility name for the total rebin-memory ceiling."""

    return rebin_memory_limit_bytes(available_memory)


def operation_worker_count(
    total_bytes: int,
    *,
    bytes_per_worker: int = 16 * _MEBIBYTE,
    min_parallel_bytes: int = 8 * _MEBIBYTE,
    requested_workers: int | None = None,
    memory_limit_bytes: int | None = None,
) -> int:
    """Choose a bounded worker count for a large-array operation.

    The result honors the global CPU ceiling through ``num_threads()`` and the
    managed scientific-memory ceiling.  It is intentionally a planning helper,
    not a guarantee about total process memory use.
    """

    work = max(0, int(total_bytes))
    per_worker = int(bytes_per_worker)
    threshold = max(0, int(min_parallel_bytes))
    if per_worker < 1:
        raise ValueError("bytes_per_worker must be positive")
    if work < threshold:
        return 1
    from ._parallel import num_threads

    cpu_workers = num_threads()
    if requested_workers is not None:
        cpu_workers = min(cpu_workers, max(1, int(requested_workers)))
    budget = (
        scientific_memory_limit_bytes()
        if memory_limit_bytes is None
        else max(1, int(memory_limit_bytes))
    )
    memory_workers = max(1, budget // per_worker)
    return max(1, min(cpu_workers, memory_workers))


def operation_batch_bytes(
    *,
    minimum_bytes: int = 8 * _MEBIBYTE,
    maximum_bytes: int = 512 * _MEBIBYTE,
    budget_fraction: int = 8,
) -> int:
    """Choose an internal array-batch target from the central RAM allowance."""

    if minimum_bytes < 1 or maximum_bytes < minimum_bytes or budget_fraction < 1:
        raise ValueError("invalid automatic batch policy")
    limit = scientific_memory_limit_bytes()
    scoped = _SCOPED_BATCH_BYTES.get()
    if scoped is not None:
        return max(1, min(int(scoped), limit))
    return max(
        1,
        min(
            limit,
            max(
                int(minimum_bytes),
                min(int(maximum_bytes), limit // int(budget_fraction)),
            ),
        ),
    )


@contextmanager
def batch_budget(max_batch_bytes: int | None):
    """Temporarily select an internal batch target for calibration only."""

    token = _SCOPED_BATCH_BYTES.set(
        None if max_batch_bytes is None else max(1, int(max_batch_bytes))
    )
    try:
        yield
    finally:
        _SCOPED_BATCH_BYTES.reset(token)


def initialize_rebin_performance(config: dict) -> None:
    """Snapshot machine defaults without replacing existing saved values."""
    if "max_batch_mb" in config and "workers" in config:
        return

    from ._parallel import num_threads

    settings = load_performance_settings()
    config.setdefault("max_batch_mb", settings["max_batch_mb"] or 192)
    config.setdefault("workers", settings["workers"] or num_threads())
