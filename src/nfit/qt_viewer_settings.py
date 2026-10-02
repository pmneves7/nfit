"""Qt clipboard controls for portable data-viewer presentation settings."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtWidgets

from .viewer_settings import compatible_viewer_settings, copy_viewer_settings

_MIME = "application/x-nfit-viewer-settings+json"


_POINT_MODEL_FIELDS = (
    "color_scale", "color_alpha", "auto_limits", "power_gamma", "sigma_n",
    "iqr_n", "percentile_n", "autoscale", "symmetric_about_zero",
    "manual_vmin", "manual_vmax", "channel", "coverage_threshold",
)


def _axis_names(viewer):
    if getattr(viewer.model, "is_point_list", False):
        return [*viewer.model.point_coordinates, "__signal__"]
    return [axis.name for axis in viewer.data.axes]


def _capture_settings(viewer):
    if not getattr(viewer.model, "is_point_list", False):
        return viewer.current_plot_settings()
    values = vars(viewer._capture_dataset_state()).copy()
    for key in ("model", "display_step_factors", "tile_dim", "tile_range", "tile_step"):
        values.pop(key, None)
    values["show_histogram_axes"] = values.pop("histogram_axes")
    values.update({key: getattr(viewer.model, key) for key in _POINT_MODEL_FIELDS})
    values.update(x_dim=viewer.model.x_key, y_dim="__signal__",
                  cmap=viewer.model._effective_cmap(), view_mode="slice")
    return values


def _apply_settings(viewer, settings):
    if not getattr(viewer.model, "is_point_list", False):
        viewer.apply_plot_settings(settings)
        return
    # Use the same state restoration as Hold view settings for point-list plots.
    state = viewer._capture_dataset_state()
    for key in vars(state):
        if key not in {"model", "tile_dim", "display_step_factors"} and key in settings:
            setattr(state, key, settings[key])
    if "show_histogram_axes" in settings:
        state.histogram_axes = bool(settings["show_histogram_axes"])
    for key in _POINT_MODEL_FIELDS:
        if key in settings and (key != "channel" or settings[key] in state.model.point_channels):
            setattr(state.model, key, settings[key])
    cmap = settings.get("cmap")
    if cmap is not None:
        state.model.cmap_reversed = str(cmap).endswith("_r")
        state.model.cmap = str(cmap).removesuffix("_r")
    state.model.masked = bool(settings.get("apply_masks", state.model.masked))
    if settings.get("x_dim") in state.model.point_coordinates:
        state.model.x_key = settings["x_dim"]
    viewer._restore_dataset_state(state)


class _ClipboardAvailability(QtCore.QObject):
    def __init__(self, button):
        super().__init__(button)
        self.button = button
        QtWidgets.QApplication.clipboard().dataChanged.connect(self.update)
        self.update()

    @QtCore.Slot()
    def update(self):
        mime = QtWidgets.QApplication.clipboard().mimeData()
        self.button.setEnabled(mime is not None and mime.hasFormat(_MIME))


def add_settings_buttons(viewer, layout) -> None:
    """Install clipboard actions at the right of the viewer's top bar."""
    clipboard = QtWidgets.QApplication.clipboard()
    copy_button = QtWidgets.QPushButton("Copy settings")
    paste_button = QtWidgets.QPushButton("Paste settings")
    viewer.copy_settings_button = copy_button
    viewer.paste_settings_button = paste_button
    copy_button.setObjectName("data_viewer_copy_settings")
    paste_button.setObjectName("data_viewer_paste_settings")
    copy_button.setToolTip(
        "Copy axes, limits, hidden-axis ranges, colors, smoothing, box cuts, "
        "and figure styling for another data viewer."
    )
    paste_button.setToolTip(
        "Apply copied settings supported by this viewer, matching axes by name. "
        "Keep this viewer's dataset, binning, and lattice metadata."
    )

    def copy_settings() -> None:
        payload = copy_viewer_settings(
            _capture_settings(viewer), _axis_names(viewer)
        )
        payload["hold_view_settings"] = viewer.hold_view_settings
        mime = QtCore.QMimeData()
        encoded = json.dumps(payload).encode("utf-8")
        mime.setData(_MIME, encoded)
        mime.setText(encoded.decode("utf-8"))
        clipboard.setMimeData(mime)

    def paste_settings() -> None:
        try:
            payload = json.loads(bytes(clipboard.mimeData().data(_MIME)))
            settings = compatible_viewer_settings(
                payload, _axis_names(viewer)
            )
            # Preserve the current mode if the source mode cannot be transferred.
            if "view_mode" not in settings:
                settings["view_mode"] = _capture_settings(viewer)["view_mode"]
            _apply_settings(viewer, settings)
            viewer.hold_view_settings_check.setChecked(
                bool(payload.get("hold_view_settings", viewer.hold_view_settings))
            )
        except (ValueError, TypeError, KeyError, IndexError) as error:
            QtWidgets.QMessageBox.warning(viewer.window, "Paste settings", str(error))

    copy_button.clicked.connect(copy_settings)
    paste_button.clicked.connect(paste_settings)
    paste_button._nfit_clipboard_availability = _ClipboardAvailability(paste_button)
    layout.addWidget(copy_button)
    layout.addWidget(paste_button)
