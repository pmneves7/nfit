"""Numbered-source selection and bounded asynchronous preview controls."""

from __future__ import annotations

import json

PREVIEW_ROWS = 1000


def source_selection_from_config(config):
    """Read saved selector fields through the same public source-selection type."""
    from .source_selection import SourceSelection

    return SourceSelection(
        directory=str(config.get("path", "")), prefix=str(config.get("prefix", "")),
        suffix=str(config.get("suffix", "")), expression=str(config.get("numors", "")),
        padding=int(config.get("padding", 0)),
    )


def build_source_selection_panel(
    config, *, on_setting_changed, on_enabled, on_import, on_files, on_clear, parent=None,
):
    """Build saved source fields and a metadata-only preview using public services."""
    from PySide6 import QtCore, QtWidgets
    from shiboken6 import isValid

    from .source_selection import (
        MAX_SOURCE_NUMBER_PADDING,
        RUN_EXPRESSION_HELP,
        resolve_source_selection,
    )

    box = QtWidgets.QGroupBox("Dataset importing", parent)
    box.setToolTip(
        "Select numbered runs by directory, naming pattern, and run expression, "
        "or choose explicit files. Preview retains grouped runs and repeated source identities."
    )
    outer = QtWidgets.QVBoxLayout(box)
    enabled = QtWidgets.QCheckBox("Import datasets from files")
    enabled.setObjectName("dataset_importing_enabled")
    enabled.setToolTip("Enable source selection for this workspace; the selection is saved in the project.")
    enabled.setChecked(bool(config.get("enabled", False)))
    outer.addWidget(enabled)
    controls = QtWidgets.QWidget()
    controls.setObjectName("dataset_importing_controls")
    controls.setEnabled(bool(config.get("enabled", False)))
    enabled.toggled.connect(controls.setEnabled)
    enabled.toggled.connect(on_enabled)
    layout = QtWidgets.QVBoxLayout(controls)
    layout.setContentsMargins(0, 0, 0, 0)
    form = QtWidgets.QFormLayout()
    fields = (
        ("Directory", "path", "Directory containing numbered files. Source data remain in their original location."),
        ("File prefix", "prefix", "Text before each run number, for example SEQ_."),
        ("File suffix", "suffix", "Text after each run number, for example .nxs.h5 or .001."),
        ("Run expression", "numors", "Run numbers, inclusive ranges, strides, sums, and repeated or blocked groups. Preview shows the resulting expression groups before import."),
    )
    edits = {}
    for caption, key, tooltip in fields:
        label = QtWidgets.QLabel(caption)
        label.setToolTip(tooltip)
        edit = QtWidgets.QLineEdit(str(config.get(key, "")))
        edit.setObjectName(f"dataset_importing_{key}")
        edit.setToolTip(tooltip)
        edit.editingFinished.connect(lambda key=key, edit=edit: on_setting_changed(key, edit.text()))
        form.addRow(label, edit)
        edits[key] = edit
    padding = QtWidgets.QSpinBox()
    padding.setObjectName("dataset_importing_padding")
    padding.setRange(0, MAX_SOURCE_NUMBER_PADDING)
    padding.setValue(int(config.get("padding", 0)))
    padding.setToolTip("Minimum run-number width, padded with leading zeros. Zero uses the ordinary run number.")
    padding.valueChanged.connect(lambda value: on_setting_changed("padding", str(value)))
    form.addRow("Number padding", padding)
    layout.addLayout(form)
    syntax = QtWidgets.QLabel(
        "Examples: 392985:393469,393500:393631; 10:2:20; 10>20; 10+12. "
        "Repeated and blocked GRASP expressions are also accepted."
    )
    syntax.setWordWrap(True)
    syntax.setToolTip("The saved expression is expanded by the same parser used by the public scripting API.")
    layout.addWidget(syntax)
    preserve_groups = QtWidgets.QCheckBox("Preserve expression groups as subfolders")
    preserve_groups.setObjectName("dataset_importing_preserve_groups")
    preserve_groups.setToolTip(
        "By default, all selected files join one dataset group. Enable to keep the "
        "expression groups in ordinary dataset subfolders."
    )
    preserve_groups.setChecked(bool(config.get("preserve_groups", False)))
    preserve_groups.toggled.connect(lambda value: on_setting_changed("preserve_groups", bool(value)))
    layout.addWidget(preserve_groups)
    actions = QtWidgets.QHBoxLayout()
    preview = QtWidgets.QPushButton("Preview runs")
    preview.setObjectName("dataset_importing_preview")
    preview.setToolTip("Expand the expression and check file availability and repeated source identities without importing or reducing data.")
    inspect = QtWidgets.QCheckBox("Inspect acquisition metadata")
    inspect.setObjectName("dataset_importing_inspect_metadata")
    inspect.setToolTip("Also read available run, instrument, energy, and acquisition metadata. Event arrays are not loaded.")
    actions.addWidget(preview)
    actions.addWidget(inspect)
    help_button = QtWidgets.QPushButton("Expression syntax")
    help_button.setObjectName("dataset_importing_expression_help")
    help_button.setToolTip("Show all supported run expressions and their acquisition grouping semantics.")

    def show_expression_help():
        dialog = QtWidgets.QDialog(box)
        dialog.setWindowTitle("Run expression syntax")
        dialog.resize(720, 540)
        help_layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit(RUN_EXPRESSION_HELP)
        text.setReadOnly(True)
        help_layout.addWidget(text)
        close = QtWidgets.QPushButton("Close")
        close.setToolTip("Close the syntax reference.")
        close.clicked.connect(dialog.accept)
        help_layout.addWidget(close)
        dialog.exec()

    help_button.clicked.connect(show_expression_help)
    actions.addWidget(help_button)
    actions.addStretch(1)
    layout.addLayout(actions)
    summary = QtWidgets.QLabel("Preview the requested runs before import.")
    summary.setObjectName("dataset_importing_preview_summary")
    summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    summary.setWordWrap(True)
    summary.setToolTip("Preview status, missing files, duplicate identities, and bounded table display.")
    layout.addWidget(summary)
    table = QtWidgets.QTableWidget(0, 6)
    table.setObjectName("dataset_importing_preview_table")
    table.setHorizontalHeaderLabels(["Expression group", "Run", "File", "Present", "Repeated", "Metadata"])
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setToolTip("Each source appearance retains its group and identity. An empty group represents missing data rather than a zero measurement.")
    table.setMinimumHeight(170)
    table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
    layout.addWidget(table)
    buttons = QtWidgets.QHBoxLayout()
    add_files = QtWidgets.QPushButton("Add files")
    add_files.setObjectName("dataset_importing_add_files")
    add_files.setToolTip("Choose explicit files for irregular names, containers, or data without run numbers.")
    add_files.clicked.connect(on_files)
    buttons.addWidget(add_files)
    import_button = QtWidgets.QPushButton("Import datasets")
    import_button.setObjectName("dataset_importing_import_range")
    import_button.setToolTip("Import the previewed selection using its saved grouping and reduction settings.")
    import_button.setEnabled(False)
    buttons.addWidget(import_button)
    export = QtWidgets.QPushButton("Copy source import script")
    export.setObjectName("dataset_importing_copy_source_script")
    export.setToolTip("Copy editable Python reproducing the source naming, expression, and optional subfolders through nfit's public import API.")
    export.setEnabled(False)

    def copy_source_script():
        from .source_selection_imports import source_selection_script

        QtWidgets.QApplication.clipboard().setText(source_selection_script(
            state["selection"], preserve_groups=preserve_groups.isChecked()
        ))

    export.clicked.connect(copy_source_script)
    buttons.addWidget(export)
    clear_button = QtWidgets.QPushButton("Clear datasets")
    clear_button.setObjectName("dataset_importing_clear")
    clear_button.setToolTip("Remove direct and nested datasets after confirmation; keep models, masks, and fit history.")
    clear_button.clicked.connect(on_clear)
    buttons.addWidget(clear_button)
    layout.addLayout(buttons)
    outer.addWidget(controls)
    state = {"token": 0, "selection": None}

    def current_selection():
        values = {key: edit.text() for key, edit in edits.items()}
        values["padding"] = padding.value()
        return source_selection_from_config(values)

    def changed(*_args):
        state["token"] += 1
        state["selection"] = None
        import_button.setEnabled(False)
        export.setEnabled(False)
        table.setRowCount(0)
        summary.setText("Settings changed; preview the requested runs again.")

    for edit in edits.values():
        edit.textChanged.connect(changed)
    padding.valueChanged.connect(changed)
    inspect.toggled.connect(changed)

    class PreviewSignals(QtCore.QObject):
        completed = QtCore.Signal(object)
        failed = QtCore.Signal(str)

    class PreviewJob(QtCore.QRunnable):
        def __init__(self, selection, metadata, jobs):
            super().__init__()
            self.selection = selection
            self.metadata = metadata
            self.jobs = jobs
            self.signals = PreviewSignals()

        def run(self):
            try:
                plan = resolve_source_selection(self.selection, inspect_metadata=self.metadata)
            except Exception as error:
                self.signals.failed.emit(str(error))
            else:
                self.signals.completed.emit(plan)
            finally:
                self.jobs.discard(self)

    class PreviewReceiver(QtCore.QObject):
        def __init__(self, token, selection):
            super().__init__(box)
            self.token = token
            self.selection = selection

        @QtCore.Slot(str)
        def failed(self, message):
            if not isValid(box):
                return
            preview.setEnabled(True)
            if self.token == state["token"]:
                summary.setText(message)
                import_button.setEnabled(False)
                export.setEnabled(False)
            self.deleteLater()

        @QtCore.Slot(object)
        def completed(self, plan):
            if not isValid(box):
                return
            preview.setEnabled(True)
            if self.token == state["token"]:
                _present_preview(table, summary, plan)
                state["selection"] = self.selection
                ready = bool(plan.slots) and not bool(plan.missing)
                import_button.setEnabled(ready)
                export.setEnabled(ready)
            self.deleteLater()

    def start_preview():
        try:
            selection = current_selection()
        except (TypeError, ValueError) as error:
            summary.setText(str(error))
            return
        state["token"] += 1
        token = state["token"]
        summary.setText("Resolving source groups and file availability…")
        import_button.setEnabled(False)
        export.setEnabled(False)
        application = QtWidgets.QApplication.instance()
        if not hasattr(application, "_nfit_source_preview_jobs"):
            application._nfit_source_preview_jobs = set()
        jobs = application._nfit_source_preview_jobs
        job = PreviewJob(selection, inspect.isChecked(), jobs)
        jobs.add(job)
        receiver = PreviewReceiver(token, selection)
        job.signals.failed.connect(receiver.failed)
        job.signals.completed.connect(receiver.completed)
        preview.setEnabled(False)
        QtCore.QThreadPool.globalInstance().start(job)

    preview.clicked.connect(start_preview)
    import_button.clicked.connect(lambda _checked=False: on_import(
        state["selection"], preserve_groups=preserve_groups.isChecked()
    ))
    return box


def _present_preview(table, summary, plan):
    """Render a bounded view of a fully resolved public source plan."""
    from PySide6 import QtWidgets

    rows = plan.appearances
    missing = set(plan.missing)
    duplicates = set(plan.duplicates)
    table.setRowCount(min(len(rows), PREVIEW_ROWS))
    for row_index, appearance in enumerate(rows[:PREVIEW_ROWS]):
        empty = appearance.run_number is None
        values = (
            f"{appearance.depth} / {appearance.stack}",
            "empty" if empty else appearance.run_number,
            appearance.path or "",
            "empty" if empty else "missing" if appearance.path in missing else "yes",
            "shared source" if appearance.source_identity in duplicates else "",
            json.dumps(plan.metadata.get(appearance.path, {}), default=str),
        )
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            table.setItem(row_index, column, item)
    summary.setText(
        f"{len(plan.slots):,} expression groups; {len(rows):,} appearances; "
        f"{len(plan.resolved_files):,} unique files; {len(plan.missing):,} missing files; "
        f"{len(plan.duplicates):,} repeated identities. "
        + (f"Showing the first {PREVIEW_ROWS:,} appearances." if len(rows) > PREVIEW_ROWS else "")
    )


def build_saved_source_selection_summary(metadata, *, parent=None):
    """Show an imported group's saved expression without expanding or loading files."""
    from PySide6 import QtCore, QtWidgets

    selection = metadata.get("source_selection")
    if not isinstance(selection, dict):
        recipe = metadata.get("reduction_recipe", {})
        selection = recipe.get("source_selection") if isinstance(recipe, dict) else None
    if not isinstance(selection, dict) or selection.get("mode") != "expression":
        return None
    box = QtWidgets.QGroupBox("Sources", parent)
    box.setObjectName("saved_source_selection_summary")
    box.setToolTip("Original source expression and resolved membership retained for reproducible imports.")
    layout = QtWidgets.QVBoxLayout(box)
    expression = str(selection.get("expression", selection.get("numors", "")))
    display = expression[:2000] + ("… (copy the expression to view it completely)" if len(expression) > 2000 else "")
    lines = [
        f"Run expression: {display}",
        f"Directory: {selection.get('directory', '')}",
        f"Naming: {selection.get('prefix', '')}[run]{selection.get('suffix', '')}; padding {selection.get('padding', 0)}",
        f"Resolved files: {len(selection.get('resolved_files', ())):,}",
        "Expression groups preserved as subfolders" if selection.get("preserve_groups", False) else "Selected runs imported into one dataset group",
    ]
    label = QtWidgets.QLabel("\n".join(lines))
    label.setObjectName("saved_source_selection_text")
    label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setToolTip("Saved selector information; viewing it does not reopen source files or reduction caches.")
    layout.addWidget(label)
    copy_expression = QtWidgets.QPushButton("Copy expression")
    copy_expression.setObjectName("saved_source_selection_copy_expression")
    copy_expression.setToolTip("Copy the complete stored run expression for reuse in a source selector.")
    copy_expression.clicked.connect(lambda _checked=False: QtWidgets.QApplication.clipboard().setText(expression))
    layout.addWidget(copy_expression)
    return box


def build_saved_source_selection_editor(group, *, on_apply, parent=None):
    """Edit an existing ordinary dataset group's numbered source membership.

    Preview is asynchronous and metadata-only. The callback must delegate edits
    to the public ``update_source_selection`` service; widgets do not import or
    reduce measurements themselves.
    """
    from PySide6 import QtWidgets

    selection = group.metadata.get("source_selection", {})
    if not selection:
        selection = group.metadata.get("reduction_recipe", {}).get("source_selection", {})
    paths = [entry.metadata.get("source_file") for entry in group.iter_datasets()]
    paths = [path for path in paths if path]
    from pathlib import Path

    config = {
        "enabled": True, "path": selection.get("directory", str(Path(paths[0]).parent) if paths else ""),
        "prefix": selection.get("prefix", ""), "suffix": selection.get("suffix", ""),
        "numors": selection.get("expression", ""), "padding": selection.get("padding", 0),
        "preserve_groups": selection.get("preserve_groups", bool(group.subgroups)),
    }
    if "source_selection_group" in group.metadata:
        # A subfolder edits only its own members, rather than replaying the
        # parent expression's complete selection.
        numbers = group.metadata["source_selection_group"].get("run_numbers", ())
        config["numors"] = "+".join(str(number) for number in numbers if number is not None) or "0"
        config["preserve_groups"] = False
    panel = build_source_selection_panel(
        config, on_setting_changed=lambda key, value: config.update({key: value}),
        on_enabled=lambda _value: None, on_import=on_apply,
        on_files=lambda: None, on_clear=lambda: None, parent=parent,
    )
    panel.setTitle("Sources")
    panel.setObjectName("saved_source_selection_editor")
    panel.setToolTip("Edit this group's selected runs. Retained runs keep their reduction caches and per-run settings; only added sources are inspected.")
    panel.findChild(QtWidgets.QCheckBox, "dataset_importing_enabled").hide()
    panel.findChild(QtWidgets.QPushButton, "dataset_importing_add_files").hide()
    panel.findChild(QtWidgets.QPushButton, "dataset_importing_clear").hide()
    apply = panel.findChild(QtWidgets.QPushButton, "dataset_importing_import_range")
    apply.setText("Apply sources")
    apply.setToolTip("Replace this group's source membership after preview. Retained sources keep their caches; dependent binnings become stale when membership changes.")
    return panel
