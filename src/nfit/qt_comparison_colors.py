"""Independent color controls for the background-comparison difference panel."""

from __future__ import annotations

from typing import Any

from .slice_comparison import apply_difference_colors, default_difference_colors


class DiffColorController:
    """Own difference colors without changing the signal viewer's color model."""

    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer
        self.settings = default_difference_colors()
        self.group = None
        self._controls: dict[str, Any] = {}

    def build(self, controls_layout: Any) -> None:
        """Insert a complete color group and publish conventional widget aliases."""
        from PySide6 import QtWidgets

        from .colormaps import IMAGE_COLORMAP_GROUPS, populate_qt_colormap_combo
        from .qt_slice_controls import _compact_combobox, _make_float_spinbox

        self.group = QtWidgets.QGroupBox("Diff color")
        self.group.setObjectName("viewer_diff_color_group")
        self.viewer.diff_color_group = self.group
        layout = QtWidgets.QGridLayout(self.group)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)

        def publish(name: str, widget: Any, tooltip: str) -> Any:
            widget.setObjectName(f"diff_{name}")
            widget.setToolTip(tooltip)
            self._controls[name] = widget
            setattr(self.viewer, f"diff_{name}", widget)
            return widget

        cmap = publish("cmap_combo", QtWidgets.QComboBox(),
                       "Choose the colormap used for the background difference.")
        populate_qt_colormap_combo(cmap, IMAGE_COLORMAP_GROUPS)
        reverse = publish("cmap_reverse_button", QtWidgets.QPushButton("Reverse"),
                          "Reverse the difference colormap.")
        scale = publish("scale_combo", QtWidgets.QComboBox(),
                        "Choose linear, logarithmic, symmetric logarithmic, asinh, or power difference colors.")
        scale.addItems(self.viewer.model.COLOR_SCALES)
        limits = publish("limits_combo", QtWidgets.QComboBox(),
                         "Choose how automatic difference color limits are estimated.")
        limits.addItems(self.viewer.model.AUTO_LIMITS)
        for combo in (cmap, scale, limits):
            _compact_combobox(combo)
        autoscale = publish("autoscale_check", QtWidgets.QCheckBox("Autoscale"),
                            "Recompute difference color limits when the displayed data or selection changes.")
        symmetric = publish("symmetric_about_zero_check", QtWidgets.QCheckBox("Symmetric about 0"),
                            "Use equal-magnitude negative and positive difference limits; editing either manual limit updates both.")
        vmin = publish("vmin_spin", _make_float_spinbox(),
                       "Manual lower difference color limit; editing turns autoscale off.")
        vmax = publish("vmax_spin", _make_float_spinbox(),
                       "Manual upper difference color limit; editing turns autoscale off.")
        gamma = publish("gamma_spin", _make_float_spinbox(1.0e-6, 20.0),
                        "Exponent used by power difference color scaling.")
        alpha = publish("alpha_spin", _make_float_spinbox(-20.0, 20.0),
                        "Exponentially shift difference colors within the fixed range without changing limits or ticks.")
        alpha.setSingleStep(0.1)
        limit_n = publish("limit_n_spin", _make_float_spinbox(0.0, 50.0),
                          "Width parameter for the selected automatic difference color-limit rule.")
        gamma_label = QtWidgets.QLabel("gamma")
        limit_n_label = QtWidgets.QLabel("N")
        self._controls.update(gamma_label=gamma_label, limit_n_label=limit_n_label)
        self.viewer.diff_gamma_label = gamma_label
        self.viewer.diff_limit_n_label = limit_n_label
        layout.addWidget(QtWidgets.QLabel("Colormap"), 0, 0)
        layout.addWidget(cmap, 0, 1)
        layout.addWidget(reverse, 0, 2, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Scale"), 1, 0)
        layout.addWidget(scale, 1, 1)
        layout.addWidget(gamma_label, 1, 2)
        layout.addWidget(gamma, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Alpha"), 2, 0)
        layout.addWidget(alpha, 2, 1, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Auto limits"), 3, 0)
        layout.addWidget(limits, 3, 1)
        layout.addWidget(limit_n_label, 3, 2)
        layout.addWidget(limit_n, 3, 3)
        layout.addWidget(autoscale, 4, 1)
        layout.addWidget(symmetric, 4, 2, 1, 2)
        layout.addWidget(QtWidgets.QLabel("vmin"), 5, 0)
        layout.addWidget(vmin, 5, 1)
        layout.addWidget(QtWidgets.QLabel("vmax"), 5, 2)
        layout.addWidget(vmax, 5, 3)
        controls_layout.addWidget(self.group)
        self._sync_controls()
        cmap.currentTextChanged.connect(lambda value: self._set("cmap", value))
        reverse.clicked.connect(lambda: self._set("cmap_reversed", not self.settings["cmap_reversed"]))
        scale.currentTextChanged.connect(lambda value: self._set("color_scale", value))
        limits.currentTextChanged.connect(self._set_auto_limits)
        autoscale.toggled.connect(self._set_autoscale)
        symmetric.toggled.connect(self._set_symmetric)
        vmin.valueChanged.connect(lambda value: self._set_manual_limit("manual_vmin", value))
        vmax.valueChanged.connect(lambda value: self._set_manual_limit("manual_vmax", value))
        gamma.valueChanged.connect(lambda value: self._set("power_gamma", float(value)))
        alpha.valueChanged.connect(lambda value: self._set("color_alpha", float(value)))
        limit_n.valueChanged.connect(self._set_limit_n)
        self.sync_visibility()

    def snapshot(self) -> dict[str, Any]:
        """Return an independent serializable color-settings dictionary."""
        return dict(self.settings)

    def restore(self, settings: dict[str, Any]) -> None:
        """Restore saved values, supplying defaults for omitted settings."""
        from types import SimpleNamespace

        model = SimpleNamespace()
        apply_difference_colors(model, settings)
        self.settings = {key: getattr(model, key) for key in default_difference_colors()}
        self._sync_controls()

    def apply_to_model(self, model: Any) -> None:
        """Apply the current settings to a temporary comparison plotting model."""
        apply_difference_colors(model, self.settings)

    def sync_visibility(self) -> None:
        """Show the group only while background-comparison panels are active."""
        if self.group is not None:
            self.group.setVisible(bool(self.viewer._background_panels_active()))

    def sync_limits(self, vmin: float, vmax: float) -> None:
        """Refresh displayed limits without triggering edits or plot updates."""
        for name, value in (("vmin_spin", vmin), ("vmax_spin", vmax)):
            widget = self._controls.get(name)
            if widget is not None:
                was_blocked = widget.blockSignals(True)
                try:
                    widget.setValue(float(value))
                finally:
                    widget.blockSignals(was_blocked)

    def _sync_controls(self) -> None:
        if self.group is None:
            return
        values = {
            "cmap_combo": self.settings["cmap"],
            "scale_combo": self.settings["color_scale"],
            "limits_combo": self.settings["auto_limits"],
            "autoscale_check": self.settings["autoscale"],
            "symmetric_about_zero_check": self.settings["symmetric_about_zero"],
            "gamma_spin": self.settings["power_gamma"],
            "alpha_spin": self.settings["color_alpha"],
            "limit_n_spin": self.settings[self._limit_key()],
        }
        for name, value in values.items():
            widget = self._controls[name]
            was_blocked = widget.blockSignals(True)
            try:
                if name.endswith("combo"):
                    widget.setCurrentText(str(value))
                elif name.endswith("check"):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(float(value))
            finally:
                widget.blockSignals(was_blocked)
        if self.settings["manual_vmin"] is not None and self.settings["manual_vmax"] is not None:
            self.sync_limits(self.settings["manual_vmin"], self.settings["manual_vmax"])
        for name in ("gamma_label", "gamma_spin"):
            self._controls[name].setVisible(self.settings["color_scale"] == "power")
        for name in ("limit_n_label", "limit_n_spin"):
            self._controls[name].setVisible(self.settings["auto_limits"] != "min/max")

    def _changed(self) -> None:
        self._sync_controls()
        self.viewer.update_plot(reuse_slice=True)

    def _set(self, name: str, value: Any) -> None:
        self.settings[name] = value
        self._changed()

    def _limit_key(self) -> str:
        return {"N IQR": "iqr_n", "Nth percentile": "percentile_n"}.get(
            self.settings["auto_limits"], "sigma_n"
        )

    def _set_auto_limits(self, value: str) -> None:
        self.settings.update(auto_limits=value, autoscale=True)
        self._changed()

    def _set_autoscale(self, enabled: bool) -> None:
        if not enabled:
            self._capture_limits()
        self._set("autoscale", bool(enabled))

    def _capture_limits(self) -> None:
        for key, name in (("manual_vmin", "vmin_spin"), ("manual_vmax", "vmax_spin")):
            self.settings[key] = float(self._controls[name].value())

    def _set_symmetric(self, enabled: bool) -> None:
        self.settings["symmetric_about_zero"] = bool(enabled)
        if enabled:
            magnitude = max(abs(float(self._controls[name].value()))
                            for name in ("vmin_spin", "vmax_spin")) or 1.0
            self.sync_limits(-magnitude, magnitude)
            if not self.settings["autoscale"]:
                self._capture_limits()
        self._changed()

    def _set_manual_limit(self, name: str, value: float) -> None:
        self._capture_limits()
        if self.settings["symmetric_about_zero"]:
            magnitude = abs(float(value)) or 1.0
            self.settings.update(manual_vmin=-magnitude, manual_vmax=magnitude)
        else:
            self.settings[name] = float(value)
        self.settings["autoscale"] = False
        self._changed()

    def _set_limit_n(self, value: float) -> None:
        self.settings[self._limit_key()] = float(value)
        self.settings["autoscale"] = True
        self._changed()
