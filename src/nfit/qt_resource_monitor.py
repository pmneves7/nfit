"""GUI-only resource display with bounded, off-thread sampling."""

from __future__ import annotations

from threading import Event, Thread

from PySide6 import QtCore, QtWidgets

from .app_distribution import application_version
from .resource_usage import ResourceUsageSampler

_SAMPLE_SECONDS = 2.0
_TOOLTIP = (
    "Resource usage, refreshed every 2 seconds. CPU is a percentage of total "
    "logical CPU capacity (0–100%). Project readings cover the nfit process, "
    "including its current project and open viewers. Project RAM is resident memory "
    "as a percentage of physical RAM; system RAM is the system used-memory "
    "percentage. Child processes are excluded from nfit readings."
)


def _sample_resources(stop: Event, latest: list) -> None:
    """Own the sampler on this thread; retain only the most recent reading."""
    try:
        sampler = ResourceUsageSampler()
    except Exception:
        latest[0] = None
        return
    while not stop.wait(_SAMPLE_SECONDS):
        try:
            latest[0] = sampler.sample()
        except Exception:
            latest[0] = None


class ResourceMonitor(QtWidgets.QLabel):
    """Start only while visible and stop without blocking the GUI on shutdown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resource_monitor")
        self.setToolTip(_TOOLTIP)
        self.setContentsMargins(8, 0, 8, 0)
        self._version = application_version()
        self._latest = [None]
        self._stop = Event()
        self._thread = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(_SAMPLE_SECONDS * 1000))
        self._timer.timeout.connect(self._refresh)
        self._refresh()
        # The worker owns no Qt objects and cannot touch a destroyed widget.
        self.destroyed.connect(self._stop.set)

    def showEvent(self, event):
        super().showEvent(event)
        if self._thread is None:
            self.destroyed.disconnect(self._stop.set)
            self._stop = Event()
            self._latest = [None]
            self._refresh()
            self.destroyed.connect(self._stop.set)
            self._thread = Thread(
                target=_sample_resources,
                args=(self._stop, self._latest),
                name="nfit-resource-monitor",
                daemon=True,
            )
            self._thread.start()
            self._timer.start()

    def hideEvent(self, event):
        self._stop.set()
        self._timer.stop()
        self._thread = None
        super().hideEvent(event)

    @QtCore.Slot()
    def _refresh(self):
        usage = self._latest[0]
        if usage is None:
            self.setText(f"nfit {self._version} | Project CPU —  RAM —   |   System CPU —  RAM —")
            return
        self.setText(
            f"nfit {self._version} | Project CPU {usage.process_cpu_percent:.1f}%  "
            f"RAM {usage.process_memory_percent:.1f}%   |   "
            f"System CPU {usage.system_cpu_percent:.1f}%  "
            f"RAM {usage.system_memory_percent:.1f}%"
        )


def add_resource_monitor(toolbar):
    """Place the monitor after an expanding spacer at the toolbar's right edge."""
    spacer = QtWidgets.QWidget(toolbar)
    spacer.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    toolbar.addWidget(spacer)
    monitor = ResourceMonitor(toolbar)
    toolbar.addWidget(monitor)
    return monitor
