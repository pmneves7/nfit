"""Application preferences, organized into pages independent of project state."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from .app_distribution import application_version
from .application_preferences import (
    application_settings,
    default_continuous_colormap,
    default_waterfall_colormap,
    set_default_continuous_colormap,
    set_default_waterfall_colormap,
)
from .colormaps import (
    IMAGE_COLORMAP_GROUPS,
    USER_COLORMAPS,
    WATERFALL_COLORMAP_GROUPS,
    open_user_colormap_folder,
    populate_qt_colormap_combo,
    user_colormap_directory,
)


class PreferencesDialog(QtWidgets.QDialog):
    """Application-wide preferences; the initial page manages custom palettes."""

    def __init__(self, parent=None, *, settings: QtCore.QSettings | None = None):
        super().__init__(parent)
        self.settings = application_settings() if settings is None else settings
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

        defaults = QtWidgets.QFormLayout()
        self.continuous_colormap = QtWidgets.QComboBox()
        self.continuous_colormap.setObjectName("preferences_continuous_colormap")
        self.continuous_colormap.setToolTip(
            "Default colormap for newly opened image, slice, tiled-slice, and volume plots."
        )
        populate_qt_colormap_combo(self.continuous_colormap, IMAGE_COLORMAP_GROUPS)
        self.continuous_colormap.setCurrentText(
            default_continuous_colormap(self.settings)
        )
        defaults.addRow("Continuous plots", self.continuous_colormap)

        self.waterfall_colormap = QtWidgets.QComboBox()
        self.waterfall_colormap.setObjectName("preferences_waterfall_colormap")
        self.waterfall_colormap.setToolTip(
            "Default colormap for trace sequences in newly opened waterfall plots."
        )
        populate_qt_colormap_combo(self.waterfall_colormap, WATERFALL_COLORMAP_GROUPS)
        self.waterfall_colormap.setCurrentText(default_waterfall_colormap(self.settings))
        defaults.addRow("Waterfall plots", self.waterfall_colormap)
        body.addLayout(defaults)
        local_note = QtWidgets.QLabel(
            "These defaults are stored for this local nfit installation and are not saved in projects."
        )
        local_note.setWordWrap(True)
        body.addWidget(local_note)
        self.continuous_colormap.currentTextChanged.connect(
            lambda name: set_default_continuous_colormap(name, self.settings)
        )
        self.waterfall_colormap.currentTextChanged.connect(
            lambda name: set_default_waterfall_colormap(name, self.settings)
        )

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
        self.version_label = QtWidgets.QLabel(f"nfit version {application_version()}")
        self.version_label.setObjectName("preferences_version")
        self.version_label.setToolTip("Version of the nfit build currently running.")
        layout.addWidget(self.version_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setToolTip("Close application preferences.")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
