"""Settled, cancellable source replay for live background box profiles.

Only the GUI thread renders. A daemon worker owns one captured immutable query;
new interactions replace one pending query and cooperatively cancel the old one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread

import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Signal, Slot

from .cached_background_replay import CACHED_BACKGROUND_REPLAY
from .mdhisto import MDHistoData
from .slice_viewer_cache import mdhisto_source_identity


class _Cancelled(Exception):
    pass


@dataclass(frozen=True)
class _Request:
    key: tuple
    data: MDHistoData
    kwargs: dict


def _background_relevant(data):
    metadata = getattr(data, "metadata", {})
    return bool(metadata.get(CACHED_BACKGROUND_REPLAY)
                or metadata.get("background_profile_uncertainty")
                or any(item.get("projection", {}).get("mode") == "measured_events"
                       for item in metadata.get("background_subtractions", [])))


class BackgroundProfileReplay(QObject):
    """Automatically refine diagonal previews after the selection settles."""

    completed = Signal(object, object, object)

    def __init__(self, viewer, *, debounce_ms=180, replay=None):
        super().__init__(viewer.window)
        self.viewer = viewer
        self.status = "inactive"
        self.reason = ""
        self._replay = replay
        self._key = None
        self._ready = None
        self._failed_key = None
        self._failure_reason = ""
        self._pending = None
        self._active = None
        self._cancel = None
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._launch)
        self.completed.connect(self._finished)
        viewer.window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.viewer.window and event.type() == QEvent.Type.Close:
            self.close()
        return False

    def close(self):
        """Cancel without joining a running scientific operation."""
        self._closed = True
        self._timer.stop()
        self._pending = None
        if self._cancel is not None:
            self._cancel.set()

    def _capture(self, extents):
        viewer = self.viewer
        data = viewer.model.data
        if not isinstance(data, MDHistoData) or not _background_relevant(data):
            return None, ""
        if not viewer.hist_axes_check or not viewer.hist_axes_check.isChecked():
            return None, "Box profile panels are hidden"
        if not data.metadata.get(CACHED_BACKGROUND_REPLAY):
            return None, data.metadata.get("background_replay_unavailable_reason", "Legacy background has no source replay recipe")
        if viewer.model.channel != "signal":
            return None, "Source replay supports the signal channel"
        if not viewer.model.masked:
            return None, "Source replay requires Apply Masks"
        if viewer.smoothing_x or viewer.smoothing_y:
            return None, "Smoothed profiles require a separate covariance target"
        if any(int(value) != 1 for value in viewer.display_step_factors.values()):
            return None, "Display coarsening requires a separate covariance target"
        if viewer._fit_panels_active():
            return None, "Fit overlays do not yet use the replayed exposure target"
        # Replay reads the immutable histogram/channel payloads. Legacy metadata
        # aliases of exposure/coverage may be writable; visible acceptance is
        # captured separately below rather than sharing those aliases.
        arrays = [data.signal, data.errors, data.mask, data.num_events,
                  *(axis.values for axis in data.axes)]
        for channel in data.auxiliary_channels.values():
            arrays.append(channel.values)
            if channel.errors is not None:
                arrays.append(channel.errors)
        if any(array.flags.writeable for array in arrays):
            return None, "Mutable histogram contents cannot be replayed asynchronously"
        if extents is None or viewer._current_slice is None:
            return None, ""
        if viewer._waterfall_mode_active() or viewer._tiled_mode_active() or viewer._is_effective_1d():
            return None, "Source covariance replay is available for 2D box profiles"
        from .background_profile_queries import background_profile_slice_selection

        view = viewer._current_slice
        selected = np.isfinite(viewer.model._display_values(view))
        selected &= np.isfinite(np.asarray(view["errors"]))
        selected &= np.asarray(view["coverage_fraction"]) >= viewer.coverage_threshold
        selected = np.array(selected, dtype=bool, copy=True)
        selected.setflags(write=False)
        selections = dict(viewer.model._normalized_selections())
        descriptor = background_profile_slice_selection(data,
            displayed_dimensions=(viewer.model.x_dim, viewer.model.y_dim), selections=selections)
        sources = []
        for term in data.metadata[CACHED_BACKGROUND_REPLAY].get("terms", []):
            for source in term.get("sources", []):
                path = source.get("source_file", "")
                try:
                    stat = Path(path).stat()
                    identity = (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
                except OSError as error:
                    identity = ("unavailable", error.errno)
                sources.append((path, identity))
        key = (mdhisto_source_identity(data), id(data.metadata.get(CACHED_BACKGROUND_REPLAY)), tuple(sources),
               viewer.model.x_dim, viewer.model.y_dim, tuple(sorted(selections.items())),
               tuple(extents), float(viewer._roi_angle), bool(viewer.model.masked),
               viewer.model.channel, float(viewer.coverage_threshold),
               float(viewer.smoothing_x), float(viewer.smoothing_y),
               bool(viewer.smoothing_fill_nans), tuple(sorted(viewer.display_step_factors.items())),
               sha256(selected.tobytes()).digest())
        kwargs = dict(x_dim=viewer.model.x_dim, y_dim=viewer.model.y_dim,
                      selection=descriptor, extents=tuple(extents), angle=float(viewer._roi_angle),
                      display_selected=selected, coverage_threshold=float(viewer.coverage_threshold))
        return _Request(key, data, kwargs), ""

    def update(self, extents):
        """Retain a labelled preview and debounce its exact replacement."""
        if self._closed:
            return
        try:
            request, reason = self._capture(extents)
        except (ValueError, RuntimeError) as error:
            request, reason = None, str(error)
        relevant = _background_relevant(self.viewer.model.data)
        if request is None:
            self._key = None
            self._pending = None
            self._timer.stop()
            if self._cancel is not None:
                self._cancel.set()
            self.status = "approximate" if relevant else "inactive"
            self.reason = reason
            self._label()
            return
        self._key = request.key
        if self._ready is not None and self._ready[0] == request.key:
            self.status, self.reason = "exact", "Source replay includes shared background event covariance"
            self._render(self._ready[1])
        elif request.key == self._failed_key:
            self.status = "approximate"
            self.reason = self._failure_reason
            self._label()
        else:
            self.status, self.reason = "pending", "Diagonal preview; background covariance replay is pending"
            if (self._active is None or self._active.key != request.key
                    or self._cancel is not None and self._cancel.is_set()):
                self._pending = request
                if self._cancel is not None:
                    self._cancel.set()
                self._timer.start()
            self._label()

    def _label(self):
        viewer = self.viewer
        suffix = ("Background covariance replay pending; diagonal preview" if self.status == "pending"
                  else f"Diagonal uncertainty: {self.reason}" if self.status == "approximate" else "")
        for axis in (viewer.ax_xcut, viewer.ax_ycut, viewer.ax_fit_cut):
            if axis is not None:
                axis.set_title(suffix, fontsize="small")
        if self.status in {"approximate", "pending"} and viewer._current_box_profiles is not None:
            updates = {}
            for name in ("x_measurement", "y_measurement"):
                measurement = getattr(viewer._current_box_profiles, name)
                if measurement is not None:
                    metadata = dict(measurement.data.metadata)
                    metadata["background_profile_uncertainty"] = "diagonal_approximation"
                    metadata["background_replay_unavailable_reason"] = self.reason
                    updates[name] = replace(measurement, data=measurement.data.with_updates(metadata=metadata))
            viewer._current_box_profiles = replace(viewer._current_box_profiles, **updates)
            viewer._sync_cut_viewers()
        viewer._sync_export_controls()

    @Slot()
    def _launch(self):
        if self._closed or self._active is not None or self._pending is None:
            return
        request, self._pending = self._pending, None
        self._active = request
        cancel = self._cancel = Event()
        replay = self._replay
        if replay is None:
            from .background_profile_queries import replay_cached_background_box_profiles
            replay = replay_cached_background_box_profiles
        def progress(*args, **kwargs):
            if cancel.is_set():
                raise _Cancelled()
        def work():
            result, error = None, None
            try:
                progress()
                result = replay(request.data, **request.kwargs, progress_callback=progress)
                progress()
            except Exception as caught:
                error = caught
            # Closing never waits for this thread; QObject may already be gone.
            try:
                self.completed.emit(request, result, error)
            except RuntimeError:
                pass
        Thread(target=work, name="nfit-background-profile", daemon=True).start()

    @Slot(object, object, object)
    def _finished(self, request, result, error):
        self._active = None
        self._cancel = None
        if self._closed:
            return
        try:
            current, _reason = self._capture(self.viewer._roi_extents)
        except (ValueError, RuntimeError):
            current = None
        if current is None or current.key != self._key:
            self.update(self.viewer._roi_extents)
        if request.key == self._key:
            if error is None:
                self._ready = (request.key, result)
                self.status, self.reason = "exact", "Source replay includes shared background event covariance"
                self._render(result)
            elif not isinstance(error, _Cancelled):
                self._failed_key = request.key
                self._failure_reason = str(error)
                self.status, self.reason = "approximate", str(error)
                self._label()
                self.viewer.canvas.draw_idle()
        if self._pending is not None and not self._timer.isActive():
            self._launch()

    def _render(self, profiles):
        viewer = self.viewer
        image_limits = None if viewer.ax_image is None else (
            viewer.ax_image.get_xlim(), viewer.ax_image.get_ylim())
        viewer._current_box_profiles = profiles
        viewer._current_x_cut = profiles.x if len(profiles.x[0]) else None
        viewer._current_y_cut = profiles.y if len(profiles.y[0]) else None
        rotated = not np.isclose(viewer._roi_angle % 360, 0., atol=1e-10)
        for dimension, axis, cut in (("x", viewer.ax_xcut, profiles.x),
                                     ("y", viewer.ax_ycut, profiles.y)):
            if axis is None:
                continue
            axis.clear()
            label = ("Box " + dimension + " · " if rotated else "") + viewer.model._axis_label(
                viewer.model.x_dim if dimension == "x" else viewer.model.y_dim)
            value_label = profiles.value_label + " (background covariance replay)"
            if dimension == "x":
                axis.errorbar(cut[0], cut[1], yerr=cut[2], fmt="-", lw=1.2)
                axis.set(xlabel=label, ylabel=value_label)
            else:
                axis.errorbar(cut[1], cut[0], xerr=cut[2], fmt="-", lw=1.2)
                axis.set(xlabel=value_label, ylabel=label)
        # Cut axes share display coordinates with the image. Clearing/replacing
        # their artists must not change the user's map limits or clamp its ROI.
        if image_limits is not None:
            viewer.ax_image.set_xlim(image_limits[0])
            viewer.ax_image.set_ylim(image_limits[1])
        viewer._apply_figure_font_size()
        viewer._apply_axis_linewidth()
        self._label()
        viewer._sync_cut_viewers()
        viewer.canvas.draw_idle()

    def export_allowed(self):
        if self.status == "pending":
            return False
        if self.status == "exact":
            try:
                current, _reason = self._capture(self.viewer._roi_extents)
            except (ValueError, RuntimeError):
                return False
            return current is not None and current.key == self._key
        return True

    def export_tooltip(self):
        if self.status == "pending":
            return "Save becomes available when source covariance replay finishes. " + self.reason
        if self.status == "approximate":
            return "Save an approximate profile with its diagonal uncertainty declaration. " + self.reason
        return "Save the current profile and its uncertainty declaration. " + self.reason
