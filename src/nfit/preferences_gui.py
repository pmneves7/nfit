"""Application preferences, organized into pages independent of project state."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from .colormaps import USER_COLORMAPS, open_user_colormap_folder, user_colormap_directory


class PreferencesDialog(QtWidgets.QDialog):
    """Application-wide preferences; the initial page manages custom palettes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preferences_dialog")
        self.setWindowTitle("nfit Preferences")
        self.resize(620, 400)
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setToolTip("Application preferences shared by all projects.")
        layout.addWidget(self.tabs)

        page = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(page)
        description = QtWidgets.QLabel(
            "Add your own color tables for slice, tiled-slice, waterfall, and volume plots."
        )
        description.setWordWrap(True)
        body.addWidget(description)
        body.addWidget(QtWidgets.QLabel("Custom colormap folder"))
        row = QtWidgets.QHBoxLayout()
        self.folder_path = QtWidgets.QLineEdit(str(user_colormap_directory()))
        self.folder_path.setReadOnly(True)
        self.folder_path.setObjectName("preferences_colormap_path")
        self.folder_path.setToolTip(
            "Shared palette folder. Select and copy this path; NFIT_COLORMAP_DIR can override the default."
        )
        row.addWidget(self.folder_path, 1)
        self.open_folder_button = QtWidgets.QPushButton("Open folder…")
        self.open_folder_button.setObjectName("preferences_open_colormap_folder")
        self.open_folder_button.setToolTip("Create the folder if needed and open it in your file manager.")
        self.open_folder_button.clicked.connect(lambda: open_user_colormap_folder(self))
        row.addWidget(self.open_folder_button)
        body.addLayout(row)
        if os.environ.get("NFIT_COLORMAP_DIR"):
            body.addWidget(QtWidgets.QLabel("Folder set by NFIT_COLORMAP_DIR."))

        help_text = QtWidgets.QLabel(
            "Use .csv, .txt, or .rgb files with one RGB triplet per row, ordered from low "
            "to high. Separate values with commas or spaces. Use decimals in 0–1 or "
            "integers in 0–255, with at least two rows. Lines beginning with # are comments.\n\n"
            "For example, parula.csv appears as user_parula. To export it from MATLAB:\n"
            "writematrix(parula(256), 'parula.csv')\n\n"
            "Restart nfit after adding or editing files. Custom RGB files must accompany "
            "saved plots when sharing them with another computer."
        )
        help_text.setWordWrap(True)
        help_text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(help_text)
        loaded = QtWidgets.QLabel(
            "Loaded this session: " + (", ".join(USER_COLORMAPS) if USER_COLORMAPS else "No custom colormaps")
        )
        loaded.setWordWrap(True)
        body.addWidget(loaded)
        body.addStretch()
        self.tabs.addTab(page, "Colormaps")
        from .performance_gui import PerformancePage

        self.tabs.addTab(PerformancePage(self), "Performance")
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setToolTip("Close application preferences.")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
