"""Focused presentation of the four stages of a dataset collection workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def collection_workflow_tabs(
    sections: Mapping[str, Sequence[Any]], *, selected_tab: str = "Sources", parent=None
):
    """Arrange existing controls without adding scientific state or project containers."""
    from PySide6 import QtWidgets

    tabs = QtWidgets.QTabWidget(parent)
    tabs.setObjectName("collection_workflow_tabs")
    tabs.setDocumentMode(True)
    tabs.setMinimumHeight(480)
    tips = {
        "Sources": "Select source files or numbered runs and inspect the existing dataset group membership.",
        "Reduction": "Edit acquisition calibration, reconstruction, and per-run overrides independently of the output grid.",
        "Binning and combination": "Choose the output grid, measurement combination, symmetry and its uncertainty convention.",
        "Plots and cuts": "View the prepared output and create saved plots or cuts without changing source reduction settings.",
    }
    for title, widgets in sections.items():
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("collection_workflow_" + title.lower().replace(" ", "_"))
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(8)
        for widget in widgets:
            if widget is not None:
                layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(content)
        index = tabs.addTab(scroll, title)
        tabs.setTabToolTip(index, tips[title])
        if title == selected_tab:
            tabs.setCurrentIndex(index)
    for button in tabs.tabBar().findChildren(QtWidgets.QToolButton):
        button.setToolTip("Scroll workflow sections when the panel is narrow.")
    return tabs


def collection_plots_panel(
    plots, *, on_view: Callable, on_plot: Callable, on_copy_workflow=None, parent=None
):
    """Link the collection workflow to the existing viewer and saved plot actions."""
    from PySide6 import QtWidgets

    box = QtWidgets.QGroupBox("Plots and cuts", parent)
    box.setObjectName("collection_plots_panel")
    box.setToolTip("Plots and cuts use prepared data; display settings do not alter acquisition recipes.")
    layout = QtWidgets.QVBoxLayout(box)
    explanation = QtWidgets.QLabel(
        "Open the data viewer to select channels, integrate axes, make box cuts, and store plots. "
        "Saved plots are owned by the workspace and remain available in the project tree."
    )
    explanation.setWordWrap(True)
    explanation.setToolTip("Viewer plot and cut exports use the existing public plotting and measurement APIs.")
    layout.addWidget(explanation)
    view = QtWidgets.QPushButton("Open data viewer")
    view.setObjectName("collection_open_viewer")
    view.setToolTip("Open the selected collection using its controlling binning and current reduced-event cache.")
    view.clicked.connect(lambda _checked=False: on_view())
    layout.addWidget(view)
    if on_copy_workflow is not None:
        copy_workflow = QtWidgets.QPushButton("Copy complete workflow script")
        copy_workflow.setObjectName("collection_copy_workflow")
        copy_workflow.setToolTip(
            "Copy an editable public-API workflow including original sources, reduction settings, "
            "nested groups, binning and background dependencies. In-memory replacements and "
            "unsupported callable sources report an explicit export error."
        )
        copy_workflow.clicked.connect(lambda _checked=False: on_copy_workflow())
        layout.addWidget(copy_workflow)
    selector = QtWidgets.QComboBox()
    selector.setObjectName("collection_saved_plot_selector")
    selector.setToolTip("Choose a saved workspace plot; these are the same plots shown in the project tree.")
    for plot in plots:
        selector.addItem(plot.name, plot)
    layout.addWidget(selector)
    open_plot = QtWidgets.QPushButton("Open saved plot")
    open_plot.setObjectName("collection_open_saved_plot")
    open_plot.setToolTip("Select and render the saved plot using its persisted source, channel and slice settings.")
    open_plot.setEnabled(bool(plots))
    open_plot.clicked.connect(lambda _checked=False: on_plot(selector.currentData()))
    layout.addWidget(open_plot)
    return box


def build_collection_workflow(
    root, node, *, source_panel, membership_panel, reduction_panel, coordinate_panel,
    binning_panel, on_view, on_plot, on_copy_workflow=None, selected_tab="Sources", parent=None
):
    """Compose existing scientific controls through narrow presentation callbacks."""
    from PySide6 import QtWidgets

    from .reduction_recipes import reduction_family, reduction_settings_schema

    reduction = []
    binning = [binning_panel()]
    if reduction_family(node) is not None:
        reduction.append(reduction_panel(scopes={"reduction", "normalization", "coordinates", "storage"}))
        if any(field.scope == "histogram" for field in reduction_settings_schema(node)):
            binning.append(reduction_panel(scopes={"histogram"}, export=False))
    else:
        message = QtWidgets.QLabel(
            "No native acquisition reduction recipe is attached to this collection. "
            "Select a dataset to inspect its importer settings."
        )
        message.setWordWrap(True)
        message.setToolTip("Already reduced measurements retain their importer metadata and individual dataset processing controls.")
        reduction.append(message)
    if any(dataset.data_type.startswith("single_crystal") for dataset in node.iter_datasets()):
        reduction.append(coordinate_panel())
    return collection_workflow_tabs({
        "Sources": [source_panel(), membership_panel()],
        "Reduction": reduction,
        "Binning and combination": binning,
        "Plots and cuts": [collection_plots_panel(root.plots, on_view=on_view, on_plot=on_plot, on_copy_workflow=on_copy_workflow, parent=parent)],
    }, selected_tab=selected_tab, parent=parent)
