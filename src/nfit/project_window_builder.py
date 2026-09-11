"""Construction helpers for :class:`nfit.project_gui.NfitProjectExplorer`.

The explorer retains ownership of widget state and callbacks. This module only
assembles that object graph in the original construction order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_project_window(
    explorer: Any,
    *,
    project_window_class_factory: Callable[[], type],
    project_tree_class_factory: Callable[[], type],
    refreshing_combo_class_factory: Callable[[], type],
) -> None:
    """Construct and connect the project explorer main window."""

    from PySide6 import QtCore, QtGui, QtWidgets

    self = explorer
    toolbar, file_button, menu, splitter = _build_window_shell(
        self,
        QtCore,
        QtGui,
        QtWidgets,
        project_window_class_factory,
    )
    _build_navigation_panel(
        self,
        QtCore,
        QtGui,
        QtWidgets,
        toolbar,
        file_button,
        menu,
        splitter,
        project_tree_class_factory,
    )
    right_panel, right_layout, title_row, actions_row = _build_work_area(
        self,
        QtCore,
        QtGui,
        QtWidgets,
        refreshing_combo_class_factory,
    )
    _build_fit_controls(self, QtWidgets)
    _build_fit_settings_panel(self, QtWidgets)
    _assemble_project_window(
        self,
        QtWidgets,
        splitter,
        right_panel,
        right_layout,
        title_row,
        actions_row,
    )
    from .app_updates_gui import add_update_actions

    add_update_actions(explorer)


def _build_window_shell(
    self: Any,
    QtCore: Any,
    QtGui: Any,
    QtWidgets: Any,
    project_window_class_factory: Callable[[], type],
) -> tuple[Any, Any, Any, Any]:
    """Build the main window, project toolbar, and central splitter."""

    window_class = project_window_class_factory()
    self.window = window_class(self)
    self.window.setWindowTitle("nfit Project Explorer")
    self._external_change_timer = QtCore.QTimer(self.window)
    self._external_change_timer.setInterval(1500)
    self._external_change_timer.timeout.connect(
        self.check_for_external_project_change
    )

    toolbar = QtWidgets.QToolBar("Project")
    toolbar.setMovable(False)
    self.window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
    file_button = QtWidgets.QToolButton()
    file_button.setObjectName("file_menu_button")
    file_button.setText("File")
    file_button.setToolTip("Open project file operations and application Preferences.")
    file_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QtWidgets.QMenu(file_button)
    menu.setToolTipsVisible(True)
    self.file_menu = menu
    new_action = menu.addAction("New", self.new_project)
    new_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
    new_action.setToolTip("Start a new empty nfit project.")
    new_action.setStatusTip("Start a new empty nfit project.")
    open_action = menu.addAction("Open", self.open_project)
    open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
    open_action.setToolTip("Open a saved nfit project file.")
    open_action.setStatusTip("Open a saved nfit project file.")
    self.recent_projects_menu = menu.addMenu("Recent projects")
    self.recent_projects_menu.setToolTipsVisible(True)
    self.recent_projects_menu.setToolTip("Open one of the most recently used nfit project files.")
    self.recent_projects_menu.aboutToShow.connect(self._refresh_recent_projects_menu)
    self.reload_project_action = menu.addAction(
        "Reload from Disk", self.reload_project_from_disk
    )
    self.reload_project_action.setShortcut(
        QtGui.QKeySequence.StandardKey.Refresh
    )
    self.reload_project_action.setToolTip(
        "Reload the current project file, with a confirmation before discarding unsaved changes."
    )
    self.reload_project_action.setStatusTip(
        "Reload the current project from disk."
    )
    menu.addSeparator()
    self.cache_binnings_action = menu.addAction("Cache binnings")
    self.cache_binnings_action.setObjectName("cache_binnings_action")
    self.cache_binnings_action.setCheckable(True)
    self.cache_binnings_action.setChecked(
        bool(self.project.settings.get("cache_binnings", False))
    )
    self.cache_binnings_action.setToolTip(
        "Update stale binnings and embed rebinned datasets and composites in this project when saving."
    )
    self.cache_binnings_action.setStatusTip(
        self.cache_binnings_action.toolTip()
    )
    self.cache_binnings_action.triggered.connect(
        lambda: self._set_cache_binnings_enabled(
            self.cache_binnings_action.isChecked()
        )
    )
    save_action = menu.addAction("Save", self.save)
    save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
    save_action.setToolTip("Save the current project to its existing project file.")
    save_action.setStatusTip("Save the current project to its existing project file.")
    save_as_action = menu.addAction("Save As", self.save_as)
    save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
    save_as_action.setToolTip("Choose a new file path and save the current project there.")
    save_as_action.setStatusTip("Choose a new file path and save the current project there.")
    menu.addSeparator()
    self.preferences_action = menu.addAction("Preferences…", self.show_preferences)
    self.preferences_action.setObjectName("preferences_action")
    self.preferences_action.setToolTip("Open application preferences, including the custom colormap folder.")
    self.preferences_action.setStatusTip(self.preferences_action.toolTip())
    menu.addSeparator()
    close_action = menu.addAction("Close", self.close_project)
    close_action.setShortcut(QtGui.QKeySequence.StandardKey.Close)
    close_action.setToolTip("Close the current project after prompting to save unsaved changes.")
    close_action.setStatusTip("Close the current project after prompting to save unsaved changes.")
    quit_action = menu.addAction("Quit", self.quit_application)
    quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
    quit_action.setToolTip("Quit nfit after prompting to save unsaved project changes.")
    quit_action.setStatusTip("Quit nfit after prompting to save unsaved project changes.")
    file_button.setMenu(menu)
    toolbar.addWidget(file_button)

    help_button = QtWidgets.QToolButton()
    help_button.setObjectName("help_button")
    help_button.setText("Help")
    help_button.setToolTip("Open the local nfit documentation home page in your default web browser.")
    help_button.clicked.connect(self.show_help)
    toolbar.addWidget(help_button)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    self.window.setCentralWidget(splitter)
    return toolbar, file_button, menu, splitter


def _build_navigation_panel(
    self: Any,
    QtCore: Any,
    QtGui: Any,
    QtWidgets: Any,
    toolbar: Any,
    file_button: Any,
    menu: Any,
    splitter: Any,
    project_tree_class_factory: Callable[[], type],
) -> None:
    """Build the project tree and its navigation actions."""

    left_panel = QtWidgets.QWidget()
    left_layout = QtWidgets.QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(6)

    project_tree_class = project_tree_class_factory()
    self.tree = project_tree_class(self)
    self.tree.setToolTip(
        "Project explorer tree. Select items to edit them, expand folders to navigate, "
        "drag supported items to reorder or move them, drop data files to import them or an nfit project "
        "file to open it, and right-click for actions."
    )
    self.tree.setHeaderHidden(True)
    self.tree.setIndentation(18)
    self.tree.setDragEnabled(True)
    self.tree.setAcceptDrops(True)
    # DragDrop (not DropOnly) lets the view initiate drags via our custom
    # startDrag; DropOnly disables drag initiation entirely.
    self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
    self.tree.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
    self.tree.setDropIndicatorShown(True)
    # Allow shift/ctrl(cmd) range and multi selection for dragging several
    # datasets into a group at once.
    self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    self.tree.setEditTriggers(
        QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
    )
    self.tree.currentItemChanged.connect(lambda _current, _previous: self._sync_details())
    self.tree.itemSelectionChanged.connect(self._prune_tree_selection)
    self.tree.itemChanged.connect(self._tree_item_changed)
    self.tree.itemExpanded.connect(self._dataset_page_expanded)
    self.tree.itemCollapsed.connect(self._dataset_page_collapsed)
    toolbar_font = QtGui.QFont(self.tree.font())
    if toolbar_font.pointSize() > 0:
        toolbar_font.setPointSize(toolbar_font.pointSize() + 1)
    else:
        toolbar_font.setPointSizeF(toolbar_font.pointSizeF() + 1.0)
    toolbar.setFont(toolbar_font)
    file_button.setFont(toolbar_font)
    menu.setFont(toolbar_font)
    left_layout.addWidget(self.tree, 1)

    tree_expand_row = QtWidgets.QHBoxLayout()
    tree_expand_row.setContentsMargins(8, 0, 8, 0)
    self.expand_all_button = QtWidgets.QPushButton("Expand groups")
    self.collapse_all_button = QtWidgets.QPushButton("Collapse all")
    self.expand_all_button.setToolTip(
        "Expand the project structure while leaving lazy run pages collapsed."
    )
    self.collapse_all_button.setToolTip("Collapse the tree to the top-level workspaces.")
    self.expand_all_button.clicked.connect(self.expand_all)
    self.collapse_all_button.clicked.connect(self.collapse_all)
    tree_expand_row.addWidget(self.expand_all_button)
    tree_expand_row.addWidget(self.collapse_all_button)
    left_layout.addLayout(tree_expand_row)

    tree_button_row = QtWidgets.QHBoxLayout()
    self.tree_action_layout = tree_button_row
    tree_button_row.setContentsMargins(8, 0, 8, 8)
    self.create_group_button = QtWidgets.QPushButton("Create workspace")
    self.open_analysis_button = QtWidgets.QPushButton("Open Analysis Window")
    self.open_analysis_button.setObjectName("tree_open_analysis_button")
    self.delete_button = QtWidgets.QPushButton("Delete")
    self.create_group_button.setToolTip("Create a new top-level workspace and immediately rename it.")
    self.open_analysis_button.setToolTip(
        "Open the Analysis Window for this workspace and resume the selected analysis recipe."
    )
    self.delete_button.setToolTip(
        "Delete the selected workspace, dataset, mask, model, fit, analysis, or plot item(s) when allowed. "
        "Shift- or Ctrl/Command-click to select and delete several items of the same kind at once."
    )
    self.create_group_button.clicked.connect(self.create_data_group)
    self.open_analysis_button.clicked.connect(self.open_data_playground_for_selection)
    self.delete_button.clicked.connect(self.delete_selected)
    tree_button_row.addWidget(self.create_group_button)
    tree_button_row.addWidget(self.open_analysis_button)
    tree_button_row.addWidget(self.delete_button)
    left_layout.addLayout(tree_button_row)
    splitter.addWidget(left_panel)


def _build_work_area(
    self: Any,
    QtCore: Any,
    QtGui: Any,
    QtWidgets: Any,
    refreshing_combo_class_factory: Callable[[], type],
) -> tuple[Any, Any, Any, Any]:
    """Build selection details, dataset actions, and model selectors."""

    right_panel = QtWidgets.QWidget()
    right_layout = QtWidgets.QVBoxLayout(right_panel)
    right_layout.setContentsMargins(14, 14, 14, 14)
    right_layout.setSpacing(10)

    self.title_label = QtWidgets.QLabel("Project")
    title_font = QtGui.QFont()
    title_font.setPointSize(18)
    title_font.setBold(True)
    self.title_label.setFont(title_font)
    title_row = QtWidgets.QHBoxLayout()
    title_row.addWidget(self.title_label, 1)
    self.enabled_check = QtWidgets.QCheckBox("Enabled")
    self.enabled_check.setToolTip(
        "Include or exclude the selected dataset, dataset group, mask, or model. "
        "On a Masks or Backgrounds folder, apply the state to every contained item. "
        "Disabling a dataset group omits all descendants from fitting without "
        "changing their individual enabled states."
    )
    self.enabled_check.checkStateChanged.connect(
        lambda state: self._set_selected_enabled(
            state == QtCore.Qt.CheckState.Checked
        )
    )
    title_row.addWidget(self.enabled_check)
    self.fit_weight_widget = QtWidgets.QWidget()
    fit_weight_layout = QtWidgets.QHBoxLayout(self.fit_weight_widget)
    fit_weight_layout.setContentsMargins(0, 0, 0, 0)
    fit_weight_layout.setSpacing(6)
    fit_weight_layout.addWidget(QtWidgets.QLabel("Fit weight"))
    self.fit_weight_spin = QtWidgets.QDoubleSpinBox()
    self.fit_weight_spin.setToolTip(
        "Relative fitting weight for the selected dataset. Larger values make this dataset "
        "count more in the fit. Set zero to omit it from optimization and evaluate its model "
        "once afterward for visualization."
    )
    self.fit_weight_spin.setRange(0.0, 1.0e12)
    self.fit_weight_spin.setDecimals(6)
    self.fit_weight_spin.setSingleStep(0.1)
    self.fit_weight_spin.setValue(1.0)
    self.fit_weight_spin.valueChanged.connect(self._set_selected_dataset_fit_weight)
    fit_weight_layout.addWidget(self.fit_weight_spin)
    fit_weight_layout.addWidget(QtWidgets.QLabel("Scale"))
    self.scale_factor_spin = QtWidgets.QDoubleSpinBox()
    self.scale_factor_spin.setObjectName("dataset_scale_factor")
    self.scale_factor_spin.setToolTip(
        "Dataset calibration scale. Use 1 for normalized data. When Fit scale is "
        "unchecked it is applied before viewing and fitting; when checked it is "
        "the initial guess for a fitted dataset scale."
    )
    self.scale_factor_spin.setRange(1.0e-12, 1.0e12)
    self.scale_factor_spin.setDecimals(6)
    self.scale_factor_spin.setSingleStep(0.1)
    self.scale_factor_spin.setValue(1.0)
    self.scale_factor_spin.valueChanged.connect(self._set_selected_dataset_scale_factor)
    fit_weight_layout.addWidget(self.scale_factor_spin)
    self.scale_factor_fit_check = QtWidgets.QCheckBox("Fit scale")
    self.scale_factor_fit_check.setObjectName("dataset_scale_factor_vary")
    self.scale_factor_fit_check.setToolTip(
        "Treat the selected dataset's scale as a fitted parameter. The Scale value is used as the initial guess, "
        "then the fitted value is written back to the dataset. The fitted scale multiplies the signal and its "
        "uncertainty; avoid an initial value of zero. Changing this option controls the next fit and "
        "does not alter the current plot. Toggling an individual dataset removes it from any shared scale."
    )
    self.scale_factor_fit_check.toggled.connect(self._set_selected_dataset_scale_factor_vary)
    fit_weight_layout.addWidget(self.scale_factor_fit_check)
    title_row.addWidget(self.fit_weight_widget)
    # Temperature and applied field live in the Conditions panel
    # (built below), between the Dataset and Axes detail panels, to keep
    # this top row uncluttered.
    self._build_sample_environment_panel()

    # Bulk editors for nested dataset groups: blank when descendants differ,
    # editing overwrites the fit weight / scale of every descendant dataset.
    self.group_bulk_widget = QtWidgets.QWidget()
    group_bulk_layout = QtWidgets.QHBoxLayout(self.group_bulk_widget)
    group_bulk_layout.setContentsMargins(0, 0, 0, 0)
    group_bulk_layout.setSpacing(6)
    group_bulk_layout.addWidget(QtWidgets.QLabel("Fit weight"))
    self.group_fit_weight_edit = QtWidgets.QLineEdit()
    self.group_fit_weight_edit.setObjectName("group_fit_weight_edit")
    self.group_fit_weight_edit.setToolTip(
        "Bulk edit the fit weight for every dataset in this dataset group. Blank means descendant values differ."
    )
    self.group_fit_weight_edit.setPlaceholderText("(mixed)")
    self.group_fit_weight_edit.setMaximumWidth(90)
    self.group_fit_weight_edit.editingFinished.connect(
        lambda: self._set_group_bulk_value("fit_weight", self.group_fit_weight_edit.text())
    )
    group_bulk_layout.addWidget(self.group_fit_weight_edit)
    group_bulk_layout.addWidget(QtWidgets.QLabel("Scale"))
    self.group_scale_edit = QtWidgets.QLineEdit()
    self.group_scale_edit.setObjectName("group_scale_edit")
    self.group_scale_edit.setToolTip(
        "Bulk edit the scale factor for every dataset in this dataset group. Blank means descendant values differ."
    )
    self.group_scale_edit.setPlaceholderText("(mixed)")
    self.group_scale_edit.setMaximumWidth(90)
    self.group_scale_edit.editingFinished.connect(
        lambda: self._set_group_bulk_value("scale_factor", self.group_scale_edit.text())
    )
    group_bulk_layout.addWidget(self.group_scale_edit)
    self.group_scale_fit_check = QtWidgets.QCheckBox("Share fitted scale")
    self.group_scale_fit_check.setObjectName("group_scale_factor_vary")
    self.group_scale_fit_check.setToolTip(
        "Fit one calibration scale shared by every descendant dataset. "
        "Unchecking keeps their scales fitted independently."
    )
    self.group_scale_fit_check.toggled.connect(self._set_group_shared_scale)
    group_bulk_layout.addWidget(self.group_scale_fit_check)
    title_row.addWidget(self.group_bulk_widget)
    self.background_bulk_widget = QtWidgets.QWidget()
    background_bulk_layout = QtWidgets.QHBoxLayout(self.background_bulk_widget)
    background_bulk_layout.setContentsMargins(0, 0, 0, 0)
    background_bulk_layout.setSpacing(6)
    background_bulk_layout.addWidget(QtWidgets.QLabel("Scale"))
    self.background_bulk_scale_edit = QtWidgets.QLineEdit()
    self.background_bulk_scale_edit.setObjectName("background_bulk_scale_edit")
    self.background_bulk_scale_edit.setPlaceholderText("(mixed)")
    self.background_bulk_scale_edit.setMaximumWidth(90)
    self.background_bulk_scale_edit.setToolTip(
        "Bulk edit the multiplier for every background in this folder. "
        "Blank means the background scales differ."
    )
    self.background_bulk_scale_edit.editingFinished.connect(
        lambda: self._set_background_bulk_scale(
            self.background_bulk_scale_edit.text()
        )
    )
    background_bulk_layout.addWidget(self.background_bulk_scale_edit)
    title_row.addWidget(self.background_bulk_widget)
    self.details_label = QtWidgets.QLabel()
    self.details_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    self.details_label.setWordWrap(True)
    self.details_label.setAlignment(
        QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
    )
    self.details_scroll = QtWidgets.QScrollArea()
    self.details_scroll.setWidgetResizable(True)
    self.details_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    self.details_widget = QtWidgets.QWidget()
    self.details_layout = QtWidgets.QVBoxLayout(self.details_widget)
    self.details_layout.setContentsMargins(0, 0, 0, 0)
    self.details_layout.setSpacing(8)
    self.details_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    self.details_scroll.setWidget(self.details_widget)

    actions_row = QtWidgets.QHBoxLayout()
    self.import_dataset_button = QtWidgets.QPushButton("Import dataset")
    self.add_model_button = QtWidgets.QPushButton("Add model")
    self.new_analysis_button = QtWidgets.QPushButton("New analysis")
    self.view_slice_button = QtWidgets.QPushButton("View in data viewer")
    self.load_dataset_button = QtWidgets.QPushButton("Load now")
    self.reload_data_button = QtWidgets.QPushButton("Reload data")
    self.reload_data_button.setObjectName("reload_data_button")
    self.add_mask_button = QtWidgets.QPushButton("Add mask")
    self.add_background_button = QtWidgets.QPushButton("Add background")
    self.add_dataset_group_button = QtWidgets.QPushButton("New dataset group")
    self.save_dataset_button = QtWidgets.QPushButton("Save dataset")
    self.import_dataset_button.setToolTip("Import one or more data files into the selected workspace or dataset group.")
    self.add_model_button.setToolTip("Add a new model component to the selected workspace.")
    self.new_analysis_button.setToolTip(
        "Open the Analysis Window for this workspace with a fresh analysis recipe."
    )
    self.view_slice_button.setToolTip(
        "Open a new, independent data viewer for the selected workspace or dataset."
    )
    self.load_dataset_button.setToolTip("Load this dataset from disk now so its axes, data, and metadata are available.")
    self.reload_data_button.setToolTip(
        "Reread the selected dataset, or every dataset in the selected collection, "
        "from its configured source and refresh dependent views."
    )
    self.add_mask_button.setToolTip("Create a new mask under the selected dataset or shared mask folder.")
    self.add_background_button.setToolTip(
        "Attach a powder |Q|-energy dataset as a scaled background for the "
        "selected dataset or composite dataset group."
    )
    self.add_dataset_group_button.setToolTip("Create a nested dataset group for organizing related datasets and shared masks.")
    self.save_dataset_button.setToolTip("Export the selected dataset, including current nfit processing, to a data file.")
    self.import_dataset_button.clicked.connect(self.import_dataset_dialog)
    self.add_model_button.clicked.connect(self.add_model_to_selection)
    self.new_analysis_button.clicked.connect(self.new_analysis_for_selection)
    self.view_slice_button.clicked.connect(self.open_slice_viewer_for_selection)
    self.load_dataset_button.clicked.connect(self.load_dataset_for_selection)
    self.reload_data_button.clicked.connect(self.reload_data_for_selection)
    self.add_mask_button.clicked.connect(self.add_mask_to_selection)
    self.add_background_button.clicked.connect(self.add_background_to_selection)
    self.add_dataset_group_button.clicked.connect(self.add_dataset_group_to_selection)
    self.save_dataset_button.clicked.connect(self.save_dataset_for_selection)
    actions_row.addWidget(self.import_dataset_button)
    actions_row.addWidget(self.add_model_button)
    actions_row.addWidget(self.new_analysis_button)
    actions_row.addWidget(self.view_slice_button)
    actions_row.addWidget(self.load_dataset_button)
    actions_row.addWidget(self.reload_data_button)
    actions_row.addWidget(self.add_mask_button)
    actions_row.addWidget(self.add_background_button)
    actions_row.addWidget(self.add_dataset_group_button)
    actions_row.addWidget(self.save_dataset_button)
    actions_row.addStretch(1)

    mask_combo_class = refreshing_combo_class_factory()
    self.mask_type_combo = mask_combo_class(self._refresh_mask_type_combo)
    self.mask_type_combo.setObjectName("mask_type_combo")
    self.mask_type_combo.setToolTip(
        "Choose the exclusion-region shape for the selected mask. Parameters describe the area to mask out; "
        "new masks default to excluding nothing."
    )
    self.mask_type_combo.currentTextChanged.connect(self._set_selected_mask_type)
    self.mask_parameter_widget = QtWidgets.QWidget()
    self.mask_parameter_layout = QtWidgets.QGridLayout(self.mask_parameter_widget)
    self.mask_parameter_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    self.mask_parameter_layout.setColumnStretch(1, 1)
    model_combo_class = refreshing_combo_class_factory()
    self.model_selector_widget = QtWidgets.QWidget()
    self.model_selector_widget.setObjectName("model_selector_widget")
    model_selector_layout = QtWidgets.QHBoxLayout(
        self.model_selector_widget
    )
    model_selector_layout.setContentsMargins(0, 0, 0, 0)
    model_selector_layout.setSpacing(8)
    self.model_category_combo = QtWidgets.QComboBox()
    self.model_category_combo.setObjectName("model_category_combo")
    self.model_category_combo.setToolTip(
        "Choose a broad model family. The model list on the right is "
        "filtered to this category."
    )
    self.model_category_combo.currentIndexChanged.connect(
        self._set_selected_model_category
    )
    self.model_type_combo = model_combo_class(self._refresh_model_type_combo)
    self.model_type_combo.setObjectName("model_type_combo")
    self.model_type_combo.setToolTip(
        "Choose the model function within the selected category."
    )
    self.model_type_combo.currentTextChanged.connect(self._set_selected_model_type)
    model_selector_layout.addWidget(self.model_category_combo, 1)
    model_selector_layout.addWidget(self.model_type_combo, 2)
    self.model_parameter_scroll = QtWidgets.QScrollArea()
    self.model_parameter_scroll.setObjectName("model_parameter_scroll")
    self.model_parameter_scroll.setWidgetResizable(True)
    self.model_parameter_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    self.model_parameter_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.model_parameter_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.model_parameter_scroll.setMinimumHeight(240)
    self.model_parameter_widget = QtWidgets.QWidget()
    self.model_parameter_layout = QtWidgets.QGridLayout(self.model_parameter_widget)
    self.model_parameter_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    self.model_parameter_layout.setColumnStretch(1, 1)
    self.model_parameter_scroll.setWidget(self.model_parameter_widget)
    return right_panel, right_layout, title_row, actions_row


def _build_fit_controls(self: Any, QtWidgets: Any) -> None:
    """Build optimizer controls and saved fit or plot actions."""

    fit_editor = QtWidgets.QWidget()
    fit_editor_layout = QtWidgets.QHBoxLayout(fit_editor)
    fit_editor_layout.setContentsMargins(0, 0, 0, 0)
    self.fit_optimizer_combo = QtWidgets.QComboBox()
    self.fit_optimizer_combo.addItems(["least_squares"])
    self.fit_optimizer_combo.setToolTip("Choose the optimizer used when fitting from this fit state.")
    self.fit_optimizer_combo.currentTextChanged.connect(self._set_selected_fit_optimizer)
    self.fit_loss_combo = QtWidgets.QComboBox()
    self.fit_loss_combo.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
    self.fit_loss_combo.setToolTip(
        "Least-squares loss. Use linear for ordinary chi-squared; robust losses reduce the influence of outliers."
    )
    self.fit_loss_combo.currentTextChanged.connect(self._set_selected_fit_controls_config)
    self.fit_covariance_mode_combo = QtWidgets.QComboBox()
    self.fit_covariance_mode_combo.addItem(
        "Use absolute data uncertainties",
        "absolute",
    )
    self.fit_covariance_mode_combo.addItem(
        "Estimate scale from residuals",
        "residual",
    )
    self.fit_covariance_mode_combo.setToolTip(
        "Controls covariance-derived parameter uncertainties. The default trusts the supplied one-sigma data "
        "uncertainties. Residual scaling multiplies the covariance by reduced chi-squared and cannot distinguish "
        "underestimated statistical errors, systematics, correlations, outliers, or model inadequacy."
    )
    self.fit_covariance_mode_combo.currentIndexChanged.connect(
        self._set_selected_fit_controls_config
    )
    self.fit_f_scale_spin = QtWidgets.QDoubleSpinBox()
    self.fit_f_scale_spin.setRange(1.0e-9, 1.0e9)
    self.fit_f_scale_spin.setDecimals(6)
    self.fit_f_scale_spin.setValue(1.0)
    self.fit_f_scale_spin.setToolTip(
        "Residual scale where robust losses begin down-weighting points. With normalized residuals, 1.0 means about one sigma."
    )
    self.fit_f_scale_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_finite_difference_workers_spin = QtWidgets.QSpinBox()
    self.fit_finite_difference_workers_spin.setObjectName(
        "fit_finite_difference_workers_spin"
    )
    self.fit_finite_difference_workers_spin.setRange(-1, 256)
    self.fit_finite_difference_workers_spin.setValue(-1)
    self.fit_finite_difference_workers_spin.setToolTip(
        "Parallel residual evaluations used for numerical parameter "
        "derivatives. Use -1 for a conservative automatic CPU choice or "
        "1 for serial evaluation. Analytic Jacobians ignore this setting."
    )
    self.fit_finite_difference_workers_spin.valueChanged.connect(
        self._set_selected_fit_controls_config
    )
    self.fit_de_check = QtWidgets.QCheckBox("Differential evolution initialization")
    self.fit_de_check.setToolTip(
        "Search the bounded parameter space before least squares. Requires finite bounds on every fitted parameter."
    )
    self.fit_de_check.stateChanged.connect(self._set_selected_fit_controls_config)
    self.fit_de_maxiter_spin = QtWidgets.QSpinBox()
    self.fit_de_maxiter_spin.setRange(1, 10000)
    self.fit_de_maxiter_spin.setValue(60)
    self.fit_de_maxiter_spin.setToolTip("Maximum differential-evolution generations before least-squares polishing.")
    self.fit_de_maxiter_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_de_popsize_spin = QtWidgets.QSpinBox()
    self.fit_de_popsize_spin.setRange(2, 200)
    self.fit_de_popsize_spin.setValue(10)
    self.fit_de_popsize_spin.setToolTip("Population multiplier for differential evolution. Larger values explore more but run longer.")
    self.fit_de_popsize_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_de_workers_spin = QtWidgets.QSpinBox()
    self.fit_de_workers_spin.setRange(-1, 256)
    self.fit_de_workers_spin.setValue(1)
    self.fit_de_workers_spin.setToolTip(
        "Parallel worker threads for differential-evolution objective evaluations. Use 1 for serial execution or -1 for an automatic CPU-based choice."
    )
    self.fit_de_workers_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_check = QtWidgets.QCheckBox("Sample posterior with emcee")
    self.fit_emcee_check.setToolTip(
        "After least squares, run emcee walkers near the best fit to estimate posterior intervals and correlations."
    )
    self.fit_emcee_check.stateChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_walkers_spin = QtWidgets.QSpinBox()
    self.fit_emcee_walkers_spin.setObjectName("fit_posterior_walkers_spin")
    self.fit_emcee_walkers_spin.setRange(0, 10000)
    self.fit_emcee_walkers_spin.setValue(0)
    self.fit_emcee_walkers_spin.setToolTip("Number of emcee walkers. Use 0 to choose an automatic value from the parameter count.")
    self.fit_emcee_walkers_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_steps_spin = QtWidgets.QSpinBox()
    self.fit_emcee_steps_spin.setObjectName("fit_posterior_steps_spin")
    self.fit_emcee_steps_spin.setRange(1, 1000000)
    self.fit_emcee_steps_spin.setValue(1000)
    self.fit_emcee_steps_spin.setToolTip("Number of emcee steps per walker.")
    self.fit_emcee_steps_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_burn_spin = QtWidgets.QSpinBox()
    self.fit_emcee_burn_spin.setObjectName("fit_posterior_burn_spin")
    self.fit_emcee_burn_spin.setRange(0, 1000000)
    self.fit_emcee_burn_spin.setValue(200)
    self.fit_emcee_burn_spin.setToolTip("Initial emcee steps to discard before summarizing posterior samples.")
    self.fit_emcee_burn_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_thin_spin = QtWidgets.QSpinBox()
    self.fit_emcee_thin_spin.setObjectName("fit_posterior_thin_spin")
    self.fit_emcee_thin_spin.setRange(1, 10000)
    self.fit_emcee_thin_spin.setValue(1)
    self.fit_emcee_thin_spin.setToolTip("Keep every Nth emcee sample after burn-in.")
    self.fit_emcee_thin_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_seed_spin = QtWidgets.QSpinBox()
    self.fit_emcee_seed_spin.setObjectName("fit_posterior_seed_spin")
    self.fit_emcee_seed_spin.setRange(-1, 2147483647)
    self.fit_emcee_seed_spin.setValue(-1)
    self.fit_emcee_seed_spin.setToolTip(
        "Random seed for emcee. Use -1 for a fresh random initialization."
    )
    self.fit_emcee_seed_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_emcee_workers_spin = QtWidgets.QSpinBox()
    self.fit_emcee_workers_spin.setObjectName("fit_posterior_workers_spin")
    self.fit_emcee_workers_spin.setRange(-1, 256)
    self.fit_emcee_workers_spin.setValue(-1)
    self.fit_emcee_workers_spin.setToolTip(
        "Parallel worker threads for emcee log-probability evaluations. The default, -1, selects an automatic CPU-based worker count; use 1 for serial execution."
    )
    self.fit_emcee_workers_spin.valueChanged.connect(self._set_selected_fit_controls_config)
    self.fit_optimizer_config_editor = QtWidgets.QLineEdit("{}")
    self.fit_optimizer_config_editor.setToolTip(
        "Advanced JSON optimizer configuration for the selected fit state. GUI controls update this; edit directly for extra SciPy/emcee options."
    )
    self.fit_optimizer_config_editor.editingFinished.connect(self._set_selected_fit_optimizer_config)
    self.fit_branch_check = QtWidgets.QCheckBox("Branch timeline")
    self.fit_now_button = QtWidgets.QPushButton("Fit now")
    self.fit_corner_button = QtWidgets.QPushButton("Fit diagnostics")
    self.show_data_fit_button = QtWidgets.QPushButton("Show data and model")
    self.fit_export_report_button = QtWidgets.QPushButton("Export report...")
    self.fit_export_report_button.setObjectName("fit_export_report_button")
    self.fit_branch_check.setToolTip("Start a new nested fit timeline instead of appending to the current timeline.")
    self.fit_now_button.setToolTip("Run the optimizer from the selected fit state and store the result in the fit history.")
    self.fit_corner_button.setToolTip(
        "Open covariance/correlation heatmaps and posterior trace or corner-style plots when diagnostics are stored."
    )
    self.show_data_fit_button.setToolTip(
        "Open the data viewer with the model overlay enabled, using current model parameters or stored fit channels."
    )
    self.fit_export_report_button.setToolTip(
        "Export a publication-grade LaTeX or PDF report of this fit result: "
        "per-dataset statistics, the model Hamiltonian term by term, fitted "
        "parameters, and physics diagnostics."
    )
    self.fit_now_button.clicked.connect(self.start_fit_for_selection)
    self.fit_corner_button.clicked.connect(self.open_fit_diagnostics_plots_for_selection)
    self.show_data_fit_button.clicked.connect(self.show_data_and_fit_for_selection)
    self.fit_export_report_button.clicked.connect(self.export_fit_report_for_selection)
    self.fit_copy_script_button = QtWidgets.QPushButton("Copy fit script")
    self.fit_copy_script_button.setObjectName("fit_copy_script_button")
    self.fit_copy_script_button.setToolTip(
        "Copy a GUI-free Python script that restores this saved fit state. "
        "Set RUN_FIT=True in the script to rerun the optimizer."
    )
    self.fit_copy_script_button.clicked.connect(self.copy_fit_script_for_selection)
    self.fit_save_script_button = QtWidgets.QPushButton("Save fit script...")
    self.fit_save_script_button.setObjectName("fit_save_script_button")
    self.fit_save_script_button.setToolTip(
        "Save a GUI-free Python script that restores this saved fit state. "
        "The project must be saved so the script has a portable project path."
    )
    self.fit_save_script_button.clicked.connect(self.save_fit_script_for_selection)
    self.plot_open_button = QtWidgets.QPushButton("Open plot")
    self.plot_open_button.setObjectName("plot_open_button")
    self.plot_open_button.setToolTip(
        "Open this saved plot in a clean presentation window without viewer controls."
    )
    self.plot_open_button.clicked.connect(self.open_saved_plot_for_selection)
    self.plot_edit_button = QtWidgets.QPushButton("Open in data viewer")
    self.plot_edit_button.setObjectName("plot_edit_button")
    self.plot_edit_button.setToolTip(
        "Reopen this saved plot in the data viewer and restore its editable controls."
    )
    self.plot_edit_button.clicked.connect(self.edit_saved_plot_in_viewer)
    self.plot_copy_script_button = QtWidgets.QPushButton("Copy script")
    self.plot_copy_script_button.setObjectName("plot_copy_script_button")
    self.plot_copy_script_button.setToolTip(
        "Copy an editable, GUI-free Python script that recreates this saved plot."
    )
    self.plot_copy_script_button.clicked.connect(self.copy_plot_script_for_selection)
    self.plot_save_script_button = QtWidgets.QPushButton("Save script...")
    self.plot_save_script_button.setObjectName("plot_save_script_button")
    self.plot_save_script_button.setToolTip(
        "Save an editable, GUI-free Python script that recreates this saved plot. "
        "The project must be saved first."
    )
    self.plot_save_script_button.clicked.connect(self.save_plot_script_for_selection)
    fit_editor_layout.addWidget(self.fit_branch_check)
    fit_editor_layout.addStretch(1)
    self.fit_editor_widget = fit_editor


def _build_fit_settings_panel(self: Any, QtWidgets: Any) -> None:
    """Arrange optimizer, differential-evolution, and posterior settings."""

    self.fit_settings_panel = QtWidgets.QWidget()
    self.fit_settings_panel.setObjectName("fit_settings_panel")
    fit_settings_layout = QtWidgets.QVBoxLayout(self.fit_settings_panel)
    fit_settings_layout.setContentsMargins(0, 0, 0, 0)
    fit_settings_layout.setSpacing(8)

    optimizer_group = QtWidgets.QGroupBox("Optimizer")
    optimizer_group.setObjectName("fit_optimizer_settings_group")
    optimizer_layout = QtWidgets.QGridLayout(optimizer_group)
    optimizer_layout.setColumnStretch(1, 1)
    optimizer_layout.addWidget(QtWidgets.QLabel("Optimizer"), 0, 0)
    optimizer_layout.addWidget(self.fit_optimizer_combo, 0, 1)
    optimizer_layout.addWidget(QtWidgets.QLabel("Loss"), 1, 0)
    optimizer_layout.addWidget(self.fit_loss_combo, 1, 1)
    optimizer_layout.addWidget(QtWidgets.QLabel("Parameter uncertainty"), 2, 0)
    optimizer_layout.addWidget(self.fit_covariance_mode_combo, 2, 1)
    optimizer_layout.addWidget(QtWidgets.QLabel("Loss scale"), 3, 0)
    optimizer_layout.addWidget(self.fit_f_scale_spin, 3, 1)
    optimizer_layout.addWidget(
        QtWidgets.QLabel("Derivative workers"),
        4,
        0,
    )
    optimizer_layout.addWidget(
        self.fit_finite_difference_workers_spin,
        4,
        1,
    )
    optimizer_layout.addWidget(QtWidgets.QLabel("Advanced config"), 5, 0)
    optimizer_layout.addWidget(self.fit_optimizer_config_editor, 5, 1)
    fit_settings_layout.addWidget(optimizer_group)

    de_group = QtWidgets.QGroupBox("Differential Evolution")
    de_group.setObjectName("fit_de_settings_group")
    de_layout = QtWidgets.QGridLayout(de_group)
    de_layout.setColumnStretch(1, 1)
    de_layout.addWidget(self.fit_de_check, 0, 1)
    de_layout.addWidget(QtWidgets.QLabel("DE generations"), 1, 0)
    de_layout.addWidget(self.fit_de_maxiter_spin, 1, 1)
    de_layout.addWidget(QtWidgets.QLabel("DE population"), 2, 0)
    de_layout.addWidget(self.fit_de_popsize_spin, 2, 1)
    de_layout.addWidget(QtWidgets.QLabel("DE workers"), 3, 0)
    de_layout.addWidget(self.fit_de_workers_spin, 3, 1)
    fit_settings_layout.addWidget(de_group)

    posterior_group = QtWidgets.QGroupBox("Posterior")
    posterior_group.setObjectName("fit_posterior_settings_group")
    posterior_layout = QtWidgets.QGridLayout(posterior_group)
    self.fit_posterior_layout = posterior_layout
    posterior_layout.setColumnStretch(1, 1)
    posterior_layout.addWidget(self.fit_emcee_check, 0, 1)
    posterior_layout.addWidget(QtWidgets.QLabel("Walkers"), 1, 0)
    posterior_layout.addWidget(self.fit_emcee_walkers_spin, 1, 1)
    posterior_layout.addWidget(QtWidgets.QLabel("Steps"), 2, 0)
    posterior_layout.addWidget(self.fit_emcee_steps_spin, 2, 1)
    posterior_layout.addWidget(QtWidgets.QLabel("Burn-in"), 3, 0)
    posterior_layout.addWidget(self.fit_emcee_burn_spin, 3, 1)
    posterior_layout.addWidget(QtWidgets.QLabel("Thin"), 4, 0)
    posterior_layout.addWidget(self.fit_emcee_thin_spin, 4, 1)
    posterior_layout.addWidget(QtWidgets.QLabel("Seed"), 5, 0)
    posterior_layout.addWidget(self.fit_emcee_seed_spin, 5, 1)
    posterior_layout.addWidget(QtWidgets.QLabel("emcee workers"), 6, 0)
    posterior_layout.addWidget(self.fit_emcee_workers_spin, 6, 1)
    fit_settings_layout.addWidget(posterior_group)


def _assemble_project_window(
    self: Any,
    QtWidgets: Any,
    splitter: Any,
    right_panel: Any,
    right_layout: Any,
    title_row: Any,
    actions_row: Any,
) -> None:
    """Assemble the completed work area without changing widget order."""

    right_layout.addLayout(title_row)
    right_layout.addWidget(self.mask_type_combo)
    right_layout.addWidget(self.mask_parameter_widget)
    right_layout.addWidget(self.model_selector_widget)
    right_layout.addWidget(self.model_parameter_scroll, 5)
    right_layout.addWidget(self.fit_editor_widget)
    right_layout.addWidget(self.details_scroll, 1)
    right_layout.addLayout(actions_row)
    fit_actions_grid = QtWidgets.QGridLayout()
    fit_actions_grid.addWidget(self.fit_now_button, 0, 0)
    fit_actions_grid.addWidget(self.fit_corner_button, 0, 1)
    fit_actions_grid.addWidget(self.show_data_fit_button, 1, 0)
    fit_actions_grid.addWidget(self.fit_export_report_button, 1, 1)
    fit_actions_grid.addWidget(self.fit_copy_script_button, 2, 0)
    fit_actions_grid.addWidget(self.fit_save_script_button, 2, 1)
    right_layout.addLayout(fit_actions_grid)
    plot_actions_grid = QtWidgets.QGridLayout()
    plot_actions_grid.addWidget(self.plot_open_button, 0, 0)
    plot_actions_grid.addWidget(self.plot_edit_button, 0, 1)
    plot_actions_grid.addWidget(self.plot_copy_script_button, 1, 0)
    plot_actions_grid.addWidget(self.plot_save_script_button, 1, 1)
    right_layout.addLayout(plot_actions_grid)

    splitter.addWidget(right_panel)
    splitter.setSizes([360, 760])
    self._sync_window_title()
