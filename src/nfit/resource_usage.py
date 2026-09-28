"""Lightweight process and system resource sampling for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceUsageSnapshot:
    """A point-in-time resource usage summary, with percentages on a 0–100 scale."""

    process_cpu_percent: float
    process_memory_percent: float
    system_cpu_percent: float
    system_memory_percent: float


class ResourceUsageSampler:
    """Read process and system usage on demand without creating a worker thread."""

    def __init__(self) -> None:
        # Load psutil only when monitoring starts, not during ordinary package imports.
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self._logical_cpu_count = max(int(psutil.cpu_count(logical=True) or 1), 1)
        # psutil computes CPU percentages from deltas, so prime both baselines.
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def sample(self) -> ResourceUsageSnapshot:
        """Return current usage using nonblocking psutil reads."""
        process_cpu = self._process.cpu_percent(interval=None)
        system_cpu = self._psutil.cpu_percent(interval=None)
        process_memory = self._process.memory_info().rss
        system_memory = self._psutil.virtual_memory()

        process_cpu_percent = min(
            max(float(process_cpu) / self._logical_cpu_count, 0.0), 100.0
        )
        process_memory_percent = min(
            max(float(process_memory) / max(float(system_memory.total), 1.0) * 100.0, 0.0),
            100.0,
        )
        return ResourceUsageSnapshot(
            process_cpu_percent=process_cpu_percent,
            process_memory_percent=process_memory_percent,
            system_cpu_percent=min(max(float(system_cpu), 0.0), 100.0),
            system_memory_percent=min(max(float(system_memory.percent), 0.0), 100.0),
        )
