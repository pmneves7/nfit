from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_data_playground_window(
    owner: Any,
    explorer: Any,
    *,
    available_analysis_types: Callable[[], tuple[str, ...] | list[str]],
    analysis_definition: Callable[[str], Any],
    diagnostics_factory: Callable[[], Any],
) -> None:
    """Build and connect the widgets owned by a data-playground controller."""
    from PySide6 import QtCore, QtWidgets

    owner.window = QtWidgets.QMainWindow(explorer.window)
    owner.window.setWindowTitle("nfit Analysis Window")
    owner.window.resize(900, 650)
    central = QtWidgets.QWidget()
    owner.window.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)

    analysis_row = QtWidgets.QHBoxLayout()
    owner.analysis_combo = QtWidgets.QComboBox()
    owner.analysis_combo.setToolTip("Select a saved analysis recipe or create a new one.")
    owner.new_button = QtWidgets.QPushButton("New")
    owner.new_button.setToolTip("Start a new analysis recipe.")
    owner.duplicate_button = QtWidgets.QPushButton("Duplicate")
    owner.duplicate_button.setToolTip(
        "Duplicate the selected recipe without its previous result."
    )
    owner.delete_button = QtWidgets.QPushButton("Delete")
    owner.delete_button.setToolTip(
        "Delete the selected recipe and its linked derived datasets."
    )
    analysis_row.addWidget(owner.analysis_combo, 1)
    analysis_row.addWidget(owner.new_button)
    analysis_row.addWidget(owner.duplicate_button)
    analysis_row.addWidget(owner.delete_button)
    layout.addLayout(analysis_row)

    selectors = QtWidgets.QHBoxLayout()
    owner.dataset_combo = QtWidgets.QComboBox()
    owner.dataset_combo.setToolTip("Dataset used as the primary analysis input.")
    owner.secondary_dataset_combo = QtWidgets.QComboBox()
    owner.secondary_dataset_combo.setToolTip(
        "Optional secondary input, such as an H, K, L peak table."
    )
    owner.additional_inputs_button = QtWidgets.QPushButton("Select runs...")
    owner.additional_inputs_button.setToolTip(
        "Choose all MDEvent rotation-angle datasets used by the angle-energy "
        "background estimator."
    )
    owner.additional_inputs_button.setVisible(False)
    owner.operation_combo = QtWidgets.QComboBox()
    for key in available_analysis_types():
        owner.operation_combo.addItem(analysis_definition(key).label, key)
    owner.operation_combo.setToolTip("Choose the dataset operation to configure.")
    owner.name_edit = QtWidgets.QLineEdit("Analysis")
    owner.name_edit.setToolTip("Editable name stored with this analysis recipe.")
    selectors.addWidget(owner.dataset_combo, 2)
    selectors.addWidget(owner.secondary_dataset_combo, 2)
    selectors.addWidget(owner.additional_inputs_button, 2)
    selectors.addWidget(owner.operation_combo, 2)
    selectors.addWidget(owner.name_edit, 2)
    layout.addLayout(selectors)

    owner.parameter_scroll = QtWidgets.QScrollArea()
    owner.parameter_scroll.setWidgetResizable(True)
    owner.parameter_panel = QtWidgets.QWidget()
    owner.parameter_form = QtWidgets.QFormLayout(owner.parameter_panel)
    owner.parameter_scroll.setWidget(owner.parameter_panel)
    owner.parameter_scroll.setMinimumHeight(240)

    owner.result_tabs = QtWidgets.QTabWidget()
    owner.result_tabs.setToolTip("Inspect analysis outputs, diagnostics, and provenance.")
    owner.results = QtWidgets.QWidget()
    owner.results.setToolTip("Latest analysis summary, output table, and artifact actions.")
    results_layout = QtWidgets.QVBoxLayout(owner.results)
    results_layout.setContentsMargins(4, 4, 4, 4)
    owner.result_summary = QtWidgets.QLabel(
        "Run an analysis to inspect its outputs."
    )
    owner.result_summary.setWordWrap(True)
    owner.result_summary.setTextInteractionFlags(
        owner.result_summary.textInteractionFlags()
        | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    owner.result_table = QtWidgets.QTableWidget()
    owner.result_table.setObjectName("analysis_result_table")
    owner.result_table.setToolTip(
        "Sortable analysis output. Select a Bragg reflection to update its "
        "diagnostic plots."
    )
    owner.result_table.setSortingEnabled(True)
    owner.result_table.horizontalHeader().setSortIndicatorShown(True)
    owner.result_table.setSelectionBehavior(
        QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    )
    owner.result_table.setSelectionMode(
        QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    )

    result_actions = QtWidgets.QHBoxLayout()
    owner.add_output_button = QtWidgets.QPushButton("Add to datasets")
    owner.add_output_button.setToolTip(
        "Create a disabled dataset from this analysis artifact. Analysis tables "
        "are not added automatically."
    )
    owner.add_output_button.setEnabled(False)
    owner.add_output_button.clicked.connect(owner.add_current_output_to_datasets)
    owner.view_peak_overlay_button = QtWidgets.QPushButton("View input with peaks")
    owner.view_peak_overlay_button.setToolTip(
        "Open the input dataset in the data viewer with accepted and rejected "
        "Bragg reflections marked."
    )
    owner.view_peak_overlay_button.setEnabled(False)
    owner.view_peak_overlay_button.clicked.connect(owner.view_input_with_peaks)
    owner.export_bragg_button = QtWidgets.QPushButton("Export .int")
    owner.export_bragg_button.setToolTip(
        "Export accepted Bragg reflections as a headerless, space-delimited .int "
        "file containing H, K, L, I, and dI."
    )
    owner.export_bragg_button.setEnabled(False)
    owner.export_bragg_button.clicked.connect(owner.export_current_bragg_int)
    result_actions.addWidget(owner.add_output_button)
    result_actions.addWidget(owner.view_peak_overlay_button)
    result_actions.addWidget(owner.export_bragg_button)
    result_actions.addStretch(1)
    results_layout.addWidget(owner.result_summary)
    results_layout.addWidget(owner.result_table, 1)
    results_layout.addLayout(result_actions)

    owner.diagnostics = QtWidgets.QWidget()
    owner.diagnostics.setToolTip(
        "Numerical quality summary and per-reflection integration diagnostics."
    )
    diagnostics_layout = QtWidgets.QVBoxLayout(owner.diagnostics)
    diagnostics_layout.setContentsMargins(4, 4, 4, 4)
    owner.diagnostic_summary = QtWidgets.QTextEdit()
    owner.diagnostic_summary.setReadOnly(True)
    owner.diagnostic_summary.setMaximumHeight(110)
    owner.diagnostic_summary.setToolTip(
        "Generated, accepted, rejected, and quality-threshold summary."
    )
    owner.bragg_diagnostics = diagnostics_factory()
    diagnostics_layout.addWidget(owner.diagnostic_summary)
    diagnostics_layout.addWidget(owner.bragg_diagnostics.widget, 1)

    owner.provenance = QtWidgets.QTextEdit()
    owner.provenance.setReadOnly(True)
    owner.provenance.setToolTip("Recipe hash and input fingerprints for this result.")
    owner.result_tabs.addTab(owner.results, "Results")
    owner.result_tabs.addTab(owner.diagnostics, "Diagnostics")
    owner.result_tabs.addTab(owner.provenance, "Provenance")
    owner.result_tabs.setMinimumHeight(180)

    owner.config_output_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
    owner.config_output_splitter.setObjectName("analysis_config_output_splitter")
    owner.config_output_splitter.setToolTip(
        "Drag this divider to allocate space between the analysis configuration "
        "and its outputs."
    )
    owner.config_output_splitter.setChildrenCollapsible(False)
    owner.config_output_splitter.addWidget(owner.parameter_scroll)
    owner.config_output_splitter.addWidget(owner.result_tabs)
    owner.config_output_splitter.setStretchFactor(0, 3)
    owner.config_output_splitter.setStretchFactor(1, 2)
    owner.config_output_splitter.setSizes([390, 250])
    layout.addWidget(owner.config_output_splitter, 1)

    commands = QtWidgets.QHBoxLayout()
    commands.addStretch(1)
    owner.run_button = QtWidgets.QPushButton("Run")
    owner.run_button.setToolTip(
        "Run this recipe without modifying the input dataset."
    )
    commands.addWidget(owner.run_button)
    layout.addLayout(commands)

    owner.operation_combo.currentIndexChanged.connect(owner._operation_changed)
    owner.analysis_combo.currentIndexChanged.connect(owner._select_analysis)
    owner.new_button.clicked.connect(owner.new_analysis)
    owner.duplicate_button.clicked.connect(owner.duplicate_analysis)
    owner.delete_button.clicked.connect(owner.delete_analysis)
    owner.run_button.clicked.connect(owner.run)
    owner.result_table.itemSelectionChanged.connect(owner._result_row_selected)
    owner.additional_inputs_button.clicked.connect(owner._choose_additional_inputs)
