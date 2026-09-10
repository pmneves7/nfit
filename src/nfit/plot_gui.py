"""Minimal presentation window for a persisted :class:`PlotEntry`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .file_dialogs import get_save_file_name
from .plot_recipes import plot_script, render_plot


class PlotWindow:
    """Render one saved plot with controls hidden until explicitly requested."""

    def __init__(self, entry, data, *, fit_entry=None, project_path=None, on_update: Callable | None = None):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from PySide6 import QtCore, QtGui, QtWidgets

        self.entry = entry
        self.data = data
        self.project_path = project_path
        self.on_update = on_update
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(f"nfit Plot - {entry.name}")
        self.window.resize(980, 720)
        self.fit_entry = fit_entry
        self.figure = render_plot(entry, data, fit_entry=fit_entry)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setToolTip("Saved plot preview. Use the Plot menu to copy, save, edit, or export its script.")
        self.window.setCentralWidget(self.canvas)

        toolbar = self.window.addToolBar("Plot")
        toolbar.setMovable(False)
        menu_button = QtWidgets.QToolButton()
        menu_button.setText("Plot")
        menu_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_button.setToolTip("Plot output and editing actions.")
        menu = QtWidgets.QMenu(menu_button)
        actions = [
            ("Copy figure", "Copy the rendered figure image to the clipboard.", self.copy_figure),
            ("Save figure", "Save the rendered figure as PNG, PDF, or SVG.", self.save_figure),
            ("Open plot controls", "Show the hidden dock containing editable plot settings.", self.open_controls),
            ("Copy generating script", "Copy a GUI-free Python script that recreates this plot.", self.copy_script),
            ("Save generating script", "Save a GUI-free Python script that recreates this plot.", self.save_script),
        ]
        for label, tip, callback in actions:
            action = menu.addAction(label)
            action.setToolTip(tip)
            action.setStatusTip(tip)
            action.triggered.connect(callback)
        menu_button.setMenu(menu)
        toolbar.addWidget(menu_button)

        self.dock = QtWidgets.QDockWidget("Plot controls", self.window)
        self.dock.setObjectName("plot_controls_dock")
        self.dock.setToolTip("Edit saved settings and redraw the plot.")
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        self.title = QtWidgets.QLineEdit(str(entry.settings.get("title", "")))
        self.title.setToolTip("Optional title displayed above the primary axes.")
        self.xlabel = QtWidgets.QLineEdit(str(entry.settings.get("xlabel", "")))
        self.xlabel.setToolTip("Optional replacement for the horizontal axis label.")
        self.ylabel = QtWidgets.QLineEdit(str(entry.settings.get("ylabel", "")))
        self.ylabel.setToolTip("Optional replacement for the vertical axis label.")
        self.width = QtWidgets.QDoubleSpinBox()
        self.width.setRange(1.0, 30.0)
        self.width.setValue(float(entry.settings.get("figsize", (8.0, 6.5))[0]))
        self.width.setToolTip("Saved figure width in inches.")
        self.height = QtWidgets.QDoubleSpinBox()
        self.height.setRange(1.0, 30.0)
        self.height.setValue(float(entry.settings.get("figsize", (8.0, 6.5))[1]))
        self.height.setToolTip("Saved figure height in inches.")
        redraw = QtWidgets.QPushButton("Apply")
        redraw.setToolTip("Apply these settings to the saved plot and redraw it.")
        redraw.clicked.connect(self.apply_controls)
        form.addRow("Title", self.title)
        form.addRow("X label", self.xlabel)
        form.addRow("Y label", self.ylabel)
        form.addRow("Width (in)", self.width)
        form.addRow("Height (in)", self.height)
        form.addRow(redraw)
        self.dock.setWidget(panel)
        self.window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, self.window)
        copy_shortcut.activated.connect(self.copy_figure)

    def show(self):
        self.window.show()
        return self

    def open_controls(self):
        self.dock.show()
        self.dock.raise_()

    def apply_controls(self):
        self.entry.settings.update(
            {
                "title": self.title.text(),
                "xlabel": self.xlabel.text(),
                "ylabel": self.ylabel.text(),
                "figsize": (self.width.value(), self.height.value()),
            }
        )
        self.figure.clear()
        old_canvas = self.canvas
        self.figure = render_plot(self.entry, self.data, fit_entry=self.fit_entry)
        old_canvas.figure = None
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.window.setCentralWidget(self.canvas)
        if self.on_update is not None:
            self.on_update(self.entry)

    def copy_figure(self):
        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())

    def save_figure(self):

        path, _filter = get_save_file_name(
            self.window, "Save figure", f"{self.entry.name}.png", "Images (*.png *.pdf *.svg)"
        )
        if path:
            self.figure.savefig(path, dpi=300)

    def script(self) -> str:
        return plot_script(self.entry, project_path=self.project_path or Path("project.nfit"))

    def copy_script(self):
        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setText(self.script())

    def save_script(self):

        path, _filter = get_save_file_name(
            self.window, "Save generating script", f"{self.entry.name}.py", "Python scripts (*.py)"
        )
        if path:
            Path(path).write_text(self.script(), encoding="utf-8")
