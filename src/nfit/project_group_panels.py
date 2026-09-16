"""Qt builders for dataset-group and composite detail panels."""

from __future__ import annotations

from typing import Any

# The compatibility dispatcher supplies these established project_gui globals
# at invocation time. Keeping the contract explicit avoids a reverse import.
DETAIL_DATASET_PAGE_SIZE: Any = None
DataGroup: Any = None
_CompositeScope: Any = None
_composite_rebin_status_text: Any = None
_composite_root: Any = None
_composite_scope: Any = None
_dataset_data_point_count: Any = None
_format_number: Any = None
_momentum_rebin_matrix: Any = None
_momentum_rebin_vector_text: Any = None
_parameter_to_text: Any = None
_peek_cached_composite_dataset_data: Any = None
_rebin_axis_bound_is_auto: Any = None
_rebin_axis_mode: Any = None
_rebin_axis_fractional: Any = None
_rebin_max_batch_mb: Any = None
_rebin_mean_weighting: Any = None
_rebin_minimum_coverage: Any = None
_rebin_minimum_samples: Any = None
_sanitize_rebin_axis_config: Any = None
_tooltip_table_corner_buttons: Any = None
data_group_composite_config: Any = None
data_group_composite_binnings: Any = None
data_group_composite_enabled: Any = None
data_group_composite_status: Any = None
data_type_label: Any = None
math: Any = None
metadata_rebin_rows: Any = None
symmetry_spec_from_config: Any = None


def _group_dataset_weights_group_box(self, group: DataGroup | _CompositeScope) -> Any:
    from PySide6 import QtCore, QtWidgets

    node = group.node if isinstance(group, _CompositeScope) else group
    datasets = list(node.datasets)
    child_groups = list(node.subgroups)
    showing_children = not datasets and bool(child_groups)
    box = QtWidgets.QGroupBox(
        "Child collections" if showing_children else "Datasets"
    )
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(10, 8, 10, 8)
    table = QtWidgets.QTableWidget()
    table.setObjectName(
        "group_child_collections_table"
        if showing_children
        else "group_datasets_table"
    )
    table.setToolTip(
        "A bounded view of direct datasets, or a hierarchy-preserving view of immediate child collections."
    )
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.setMinimumHeight(120)
    table.setMaximumHeight(320)
    table.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )

    if showing_children:
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Name", "Role", "Direct datasets", "Total datasets", "Enabled"]
        )
        table.setRowCount(len(child_groups))
        for row, child in enumerate(child_groups):
            scope = _composite_scope(_composite_root(group), child)
            values = (
                child.name,
                "Live composite" if data_group_composite_enabled(scope) else "Collection",
                str(len(child.datasets)),
                str(sum(1 for _dataset in child.iter_datasets())),
                "Yes" if child.enabled else "No",
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        note = QtWidgets.QLabel(
            "Immediate child collections are shown instead of flattening all descendant runs."
        )
        note.setWordWrap(True)
        note.setToolTip(
            "Select a child collection in the project tree to inspect its runs or composite settings."
        )
        layout.addWidget(note)
    else:
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Name", "Type", "Points", "Fit weight", "Scale"]
        )
        controls = QtWidgets.QHBoxLayout()
        page_label = QtWidgets.QLabel()
        page_label.setObjectName("group_datasets_page_label")
        page_combo = QtWidgets.QComboBox()
        page_combo.setObjectName("group_datasets_page")
        page_combo.setToolTip(
            "Choose which small page of direct datasets is shown. The full collection remains active."
        )
        page_count = max(math.ceil(len(datasets) / DETAIL_DATASET_PAGE_SIZE), 1)
        for page in range(page_count):
            start = page * DETAIL_DATASET_PAGE_SIZE
            end = min(start + DETAIL_DATASET_PAGE_SIZE, len(datasets))
            page_combo.addItem(f"{start + 1}-{end}", page)

        def populate(page: int) -> None:
            start = int(page) * DETAIL_DATASET_PAGE_SIZE
            visible = datasets[start : start + DETAIL_DATASET_PAGE_SIZE]
            table.setRowCount(len(visible))
            for row, dataset in enumerate(visible):
                values = (
                    dataset.name,
                    data_type_label(dataset.data_type),
                    _format_number(_dataset_data_point_count(dataset)),
                    _format_number(dataset.fit_weight),
                    _format_number(dataset.scale_factor),
                )
                for column, value in enumerate(values):
                    table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            end = start + len(visible)
            page_label.setText(
                f"Showing {start + 1}-{end} of {len(datasets)} direct datasets"
                if datasets
                else "No direct datasets"
            )

        page_combo.currentIndexChanged.connect(populate)
        populate(0)
        controls.addWidget(page_label)
        controls.addStretch(1)
        if page_count > 1:
            controls.addWidget(QtWidgets.QLabel("Page"))
            controls.addWidget(page_combo)
        layout.addLayout(controls)

    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setDefaultAlignment(
        QtCore.Qt.AlignmentFlag.AlignLeft
    )
    table.resizeColumnsToContents()
    _tooltip_table_corner_buttons(table, "Select all visible rows on this page.")
    layout.addWidget(table)
    box.setToolTip(
        "Direct datasets or immediate child collections. Run-heavy collections are paginated; "
        "all datasets still participate in the configured composite."
    )
    return box


def _group_composite_group_box(self, group: DataGroup | _CompositeScope) -> Any:
    from PySide6 import QtWidgets

    from .metadata_dimensions_gui import metadata_dimensions_panel, metadata_rebin_rows
    from .project_rebin_panels import add_rebin_assignment_items, add_rebin_mode_items

    box = QtWidgets.QGroupBox("Composite dataset")
    box.setToolTip(
        "Combine compatible enabled datasets in this collection into one rebinned effective dataset. "
        "Enable the checkbox to show the composite rebin controls. For MDEvent composites, detector-covered "
        "zero-count bins stay at signal zero and use a finite conservative Poisson uncertainty; only bins with "
        "no detector coverage are masked."
    )
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(10, 8, 10, 8)
    binnings = data_group_composite_binnings(group)
    selected_binning = self._selected_composite_binning(group)
    config = selected_binning["config"]
    binning_row = QtWidgets.QHBoxLayout()
    binning_row.addWidget(QtWidgets.QLabel("Binning"))
    binning_combo = QtWidgets.QComboBox()
    binning_combo.setObjectName("group_composite_binning")
    for item in binnings:
        binning_combo.addItem(
            f"{item['name']}{' (fit)' if item['fit'] else ''}", item["id"]
        )
    binning_combo.setCurrentIndex(max(binning_combo.findData(selected_binning["id"]), 0))
    binning_combo.setToolTip("Choose the named composite binning edited below.")
    binning_combo.currentIndexChanged.connect(
        lambda _index, combo=binning_combo: self._select_composite_binning(
            group, str(combo.currentData())
        )
    )
    add_binning = QtWidgets.QPushButton("Add…")
    add_binning.setObjectName("group_composite_add_binning")
    add_binning.setToolTip("Create a visualization binning from the fit binning.")
    add_binning.clicked.connect(lambda: self._add_composite_binning(group, duplicate=False))
    duplicate_binning = QtWidgets.QPushButton("Duplicate")
    duplicate_binning.setObjectName("group_composite_duplicate_binning")
    duplicate_binning.setToolTip("Duplicate the selected composite binning.")
    duplicate_binning.clicked.connect(lambda: self._add_composite_binning(group, duplicate=True))
    rename_binning = QtWidgets.QPushButton("Rename…")
    rename_binning.setObjectName("group_composite_rename_binning")
    rename_binning.setToolTip("Rename the selected composite binning.")
    rename_binning.clicked.connect(lambda: self._rename_composite_binning(group))
    remove_binning = QtWidgets.QPushButton("Remove")
    remove_binning.setObjectName("group_composite_remove_binning")
    remove_binning.setEnabled(not selected_binning["fit"])
    remove_binning.setToolTip("Remove the selected visualization binning and its cache.")
    remove_binning.clicked.connect(lambda: self._remove_composite_binning(group))
    fit_binning = QtWidgets.QCheckBox("Use for fitting")
    fit_binning.setObjectName("group_composite_fit_binning")
    fit_binning.setChecked(bool(selected_binning["fit"]))
    fit_binning.setEnabled(not selected_binning["fit"])
    fit_binning.setToolTip("Make this the sole composite binning used during fit iterations.")
    fit_binning.toggled.connect(
        lambda checked: checked and self._make_composite_fit_binning(group)
    )
    binning_row.addWidget(binning_combo, 1)
    binning_row.addWidget(add_binning)
    binning_row.addWidget(duplicate_binning)
    binning_row.addWidget(rename_binning)
    binning_row.addWidget(remove_binning)
    binning_row.addWidget(fit_binning)
    layout.addLayout(binning_row)
    can_combine, message = data_group_composite_status(group)
    enable_check = QtWidgets.QCheckBox("Combine enabled datasets into one effective dataset")
    enable_check.setObjectName("group_composite_enabled")
    enable_check.setChecked(bool(config.get("enabled", False)))
    enable_check.setEnabled(can_combine)
    enable_check.setToolTip(
        (
            "When checked, this collection plots and fits as one rebinned composite. "
            "The fitter receives it instead of the constituent datasets. "
            if selected_binning["fit"]
            else "Make this visualization-only composite rebin available in the data viewer. "
        )
        + "All enabled datasets must have the same data kind. "
        "Negative dataset scale factors subtract data."
    )
    enable_check.toggled.connect(lambda checked: self._set_group_composite_enabled(group, checked))
    settings_row = QtWidgets.QHBoxLayout()
    settings_row.addWidget(enable_check)
    settings_row.addStretch(1)
    copy_settings_button = QtWidgets.QPushButton("Copy settings")
    copy_settings_button.setObjectName("group_composite_copy_settings")
    copy_settings_button.setToolTip(
        "Copy every setting in this composite rebin panel to the system clipboard. "
        "The copied JSON can be pasted into a compatible dataset or dataset-group rebin panel."
    )
    copy_settings_button.clicked.connect(
        lambda _checked=False: self._copy_rebin_settings({
            **config, "metadata_dimensions": group.metadata.get("metadata_dimensions", [])
        })
    )
    paste_settings_button = QtWidgets.QPushButton("Paste settings")
    paste_settings_button.setObjectName("group_composite_paste_settings")
    paste_settings_button.setToolTip(
        "Replace every setting in this composite rebin panel with compatible nfit rebin settings "
        "from the system clipboard."
    )
    paste_settings_button.clicked.connect(
        lambda _checked=False: self._paste_group_composite_settings(group)
    )
    settings_row.addWidget(copy_settings_button)
    settings_row.addWidget(paste_settings_button)
    layout.addLayout(settings_row)
    message_label = QtWidgets.QLabel(message)
    message_label.setWordWrap(True)
    message_label.setToolTip("Composite status. All enabled datasets must have the same data kind before they can be combined.")
    layout.addWidget(message_label)

    controls = QtWidgets.QTabWidget()
    controls.setObjectName("group_composite_tabs")
    controls.setToolTip(
        "Switch between rebin settings, metadata dimensions, and output-bin information."
    )
    controls.setEnabled(can_combine)
    controls.setVisible(bool(config.get("enabled", False)))
    settings_tab = QtWidgets.QWidget()
    settings_tab.setObjectName("group_composite_settings_tab")
    controls_layout = QtWidgets.QGridLayout(settings_tab)
    controls_layout.setContentsMargins(0, 0, 0, 0)
    if isinstance(group.metadata.get("mdevent"), dict) or isinstance(
        group.metadata.get("raw_dgs"), dict
    ):
        coordinate_row = QtWidgets.QHBoxLayout()
        coordinate_label = QtWidgets.QLabel("Output coordinates")
        coordinate_combo = QtWidgets.QComboBox()
        coordinate_combo.setObjectName("group_composite_coordinate_mode")
        coordinate_combo.addItem("Single crystal (HKLE)", "hkle")
        coordinate_combo.addItem("Powder (|Q|, DeltaE)", "powder")
        coordinate_combo.setCurrentIndex(
            max(coordinate_combo.findData(config.get("coordinate_mode", "hkle")), 0)
        )
        coordinate_tooltip = (
            "Choose a four-dimensional single-crystal HKLE reduction or a direct "
            "two-dimensional powder |Q|, energy reduction. Both use proton-charge "
            "and detector-trajectory normalization."
        )
        coordinate_label.setToolTip(coordinate_tooltip)
        coordinate_combo.setToolTip(coordinate_tooltip)
        coordinate_combo.currentIndexChanged.connect(
            lambda _index, combo=coordinate_combo: self._set_group_composite_coordinate_mode(
                group, str(combo.currentData() or "hkle")
            )
        )
        coordinate_row.addWidget(coordinate_label)
        coordinate_row.addWidget(coordinate_combo)
        coordinate_row.addStretch(1)
        layout.addLayout(coordinate_row)
    axes = config.get("axes", [])
    raw_dgs = group.metadata.get("raw_dgs")
    is_corelli = (
        isinstance(raw_dgs, dict)
        and raw_dgs.get("format") == "corelli-correlation-nexus"
    )
    show_momentum_matrix = (
        config.get("coordinate_mode") != "powder"
        and len(axes) == 4
        and can_combine
    )
    header_row = 0
    if show_momentum_matrix:
        matrix_label = QtWidgets.QLabel("Momentum coordinates")
        matrix_edit = QtWidgets.QLineEdit(
            _parameter_to_text(_momentum_rebin_matrix(axes))
        )
        matrix_edit.setObjectName("group_composite_momentum_matrix")
        matrix_tooltip = (
            "Complete 3x3 output momentum-coordinate matrix. Each row contains [H, K, L] "
            "coefficients, and all three rows must be linearly independent. DeltaE remains a "
            "separate fixed coordinate and cannot be mixed with momentum."
        )
        matrix_label.setToolTip(matrix_tooltip)
        matrix_edit.setToolTip(matrix_tooltip)
        matrix_edit.editingFinished.connect(
            lambda editor=matrix_edit: self._set_group_composite_momentum_matrix(
                group, editor.text()
            )
        )
        controls_layout.addWidget(matrix_label, 0, 0)
        controls_layout.addWidget(matrix_edit, 0, 1, 1, 6)
        header_row = 1
    show_vectors = False
    headers = ["Axis"]
    if show_vectors:
        headers.append("Momentum row [H,K,L]")
    headers.extend(
        ["Min center", "Max center", "Value", "Edges", "Grid", "Mode"]
    )
    for column, label in enumerate(headers):
        controls_layout.addWidget(QtWidgets.QLabel(label), header_row, column)
    for row, axis_config in enumerate(axes, start=header_row + 1):
        axis_index = row - header_row - 1
        axis = _sanitize_rebin_axis_config(axis_config)
        axis_mode = _rebin_axis_mode(config, axis)
        resolution_key = (
            "num_bins"
            if axis_mode == "bins"
            else "tolerance"
            if axis_mode == "tolerance"
            else "step_size"
        )
        axis_label = QtWidgets.QLabel(
            str(axis.get("name", f"Axis {axis_index + 1}"))
        )
        axis_label.setObjectName(f"group_composite_axis_label_{axis_index}")
        controls_layout.addWidget(axis_label, row, 0)
        column_offset = 1
        if show_vectors:
            variable = str(
                axis.get("variable", ("H", "K", "L", "E")[axis_index])
            ).upper()
            if variable == "E":
                energy_label = QtWidgets.QLabel("fixed E")
                energy_label.setToolTip(
                    "Energy remains a separate DeltaE coordinate and cannot be mixed with H, K, or L."
                )
                controls_layout.addWidget(energy_label, row, column_offset)
            else:
                vector_edit = QtWidgets.QLineEdit(
                    _momentum_rebin_vector_text(axis, axis_index)
                )
                vector_edit.setObjectName(f"group_composite_axis_vector_{axis_index}")
                vector_edit.setMinimumWidth(110)
                vector_edit.setToolTip(
                    "One row of the 3x3 output momentum-coordinate matrix, expressed as [H, K, L] "
                    "coefficients. Its three rows must be linearly independent. Energy is kept as a "
                    "separate fixed coordinate. Examples include [1, 1, 0], [0, 0, 1], and [1, -1, 0]."
                )
                vector_edit.editingFinished.connect(
                    lambda index=axis_index, editor=vector_edit: self._set_group_composite_axis_vector(group, index, editor.text())
                )
                controls_layout.addWidget(vector_edit, row, column_offset)
            column_offset += 1
        for column, key in enumerate(("lower", "upper", resolution_key), start=column_offset):
            display_value = (
                ""
                if key == "step_size" and axis_mode in {"discrete", "edges"}
                else
                ""
                if key in {"lower", "upper"}
                and _rebin_axis_bound_is_auto(axis, key)
                else _parameter_to_text(axis.get(key))
            )
            edit = QtWidgets.QLineEdit(display_value)
            if key in {"lower", "upper"}:
                edit.setObjectName(f"group_composite_axis_{key}_{axis_index}")
                edit.setPlaceholderText("auto")
                edit.setEnabled(axis_mode in {"step", "bins"})
            if key == resolution_key:
                edit.setObjectName(f"group_composite_resolution_value_{axis_index}")
                edit.setEnabled(axis_mode not in {"discrete", "edges"})
            edit.setMinimumWidth(72)
            if key in {"lower", "upper"}:
                edit.setToolTip(
                    f"Optional composite rebin {key} bin center (interval edge for one-bin integration). Leave blank to cover the projected data and align uniform bins so zero is a bin center."
                )
            else:
                edit.setToolTip(
                    f"Composite rebin {('bin count' if key == 'num_bins' else 'clustering tolerance' if key == 'tolerance' else 'step size')} for this axis. "
                    "These bounds and bins are applied after all enabled datasets are scaled, weighted, and collected."
                )
            edit.editingFinished.connect(
                lambda index=axis_index, key=key, editor=edit: self._set_group_composite_axis_value(group, index, key, editor.text())
            )
            controls_layout.addWidget(edit, row, column)
        edges_edit = QtWidgets.QLineEdit(
            _parameter_to_text(axis.get("bin_edges", ""))
        )
        edges_edit.setObjectName(f"group_composite_axis_edges_{axis_index}")
        edges_edit.setMinimumWidth(150)
        edges_edit.setPlaceholderText("uniform")
        edges_edit.setEnabled(axis_mode == "edges")
        edges_edit.setToolTip(
            "Optional strictly increasing edge list for only this axis, for example "
            "[-2, -1, 0, 0.5, 2]. Leave blank to use the uniform Resolution setting."
        )
        edges_edit.editingFinished.connect(
            lambda index=axis_index, editor=edges_edit: self._set_group_composite_axis_edges(
                group, index, editor.text()
            )
        )
        controls_layout.addWidget(edges_edit, row, column_offset + 3)
        mode_combo = QtWidgets.QComboBox()
        mode_combo.setObjectName(f"group_composite_axis_mode_{axis_index}")
        add_rebin_mode_items(mode_combo)
        mode_combo.setCurrentIndex(max(mode_combo.findData(axis_mode), 0))
        mode_combo.setToolTip(
            "Choose how this axis's bin centers or edges are constructed. Point assignment is "
            "controlled separately; Discrete and Tolerance grids require discrete assignment."
        )
        mode_combo.currentIndexChanged.connect(
            lambda _index, index=axis_index, combo=mode_combo: self._set_group_composite_axis_mode(
                group, index, str(combo.currentData() or "step")
            )
        )
        controls_layout.addWidget(mode_combo, row, column_offset + 4)
        assignment_combo = QtWidgets.QComboBox()
        assignment_combo.setObjectName(f"group_composite_axis_assignment_{axis_index}")
        add_rebin_assignment_items(assignment_combo)
        assignment_combo.setCurrentIndex(
            max(assignment_combo.findData(_rebin_axis_fractional(config, axis)), 0)
        )
        corelli_energy = is_corelli and axis_index == len(axes) - 1
        assignment_combo.setEnabled(
            axis_mode not in {"discrete", "tolerance"} and not corelli_energy
        )
        assignment_combo.setToolTip(
            "CORELLI reconstructs each requested DeltaE bin centre as a separate, "
            "correlated energy channel, so energy assignment remains discrete."
            if corelli_energy
            else "Fractional distributes a point between neighboring bins on this axis. "
            "Discrete assigns it wholly to one bin. Tolerance always uses discrete "
            "assignment."
        )
        assignment_combo.currentIndexChanged.connect(
            lambda _index, index=axis_index, combo=assignment_combo: self._set_group_composite_axis_fractional(
                group, index, bool(combo.currentData())
            )
        )
        controls_layout.addWidget(assignment_combo, row, column_offset + 5)
    metadata_axis_count = metadata_rebin_rows(
        self,
        group,
        controls_layout,
        header_row + len(axes) + 1,
    )
    option_row = QtWidgets.QHBoxLayout()
    auto_check = QtWidgets.QCheckBox("Automatic rebinning")
    auto_check.setObjectName("group_composite_auto")
    auto_check.setChecked(bool(config.get("auto_rebin", True)))
    auto_check.setToolTip(
        "Automatically recompute the composite when rebin settings change. It turns off when an edit exceeds "
        "5,000,000 estimated point contributions or 2,000,000 output bins, and can then be manually re-enabled. "
        "With automatic rebinning off, edits are marked pending until Rebin now is pressed or an operation such as fitting or opening the data viewer "
        "requires an up-to-date composite."
    )
    auto_check.toggled.connect(lambda checked: self._set_group_composite_auto(group, checked))
    mean_label = QtWidgets.QLabel("Mean")
    mean_combo = QtWidgets.QComboBox()
    mean_combo.setObjectName("group_composite_mean_weighting")
    mean_combo.setToolTip(
        "Choose the composite averaging mode. A physical normalization denominator, when present, always contributes to the data weight. Inverse variance additionally uses 1/sigma^2; uniform does not. Dataset scale and fit-weight factors are also applied."
    )
    mean_combo.addItem("Inverse variance", "inverse_variance")
    mean_combo.addItem("Uniform", "uniform")
    mean_combo.setCurrentIndex(max(mean_combo.findData(_rebin_mean_weighting(config)), 0))
    mean_combo.currentIndexChanged.connect(
        lambda _index, combo=mean_combo: self._set_group_composite_mean_weighting(group, str(combo.currentData() or "uniform"))
    )
    batch_label = QtWidgets.QLabel("Batch target")
    batch_spin = QtWidgets.QSpinBox()
    batch_spin.setObjectName("group_composite_max_batch_mb")
    batch_spin.setRange(1, 1_048_576)
    batch_spin.setSuffix(" MiB")
    batch_spin.setValue(_rebin_max_batch_mb(config))
    batch_tooltip = (
        "Approximate per-batch working-memory target in MiB. Smaller batches usually use less temporary memory "
        "but require more computational time. This is not a cap on total rebinner memory use. The optimum depends "
        "on dataset size, output grid size, dimensionality, and available memory."
    )
    batch_label.setToolTip(batch_tooltip)
    batch_spin.setToolTip(batch_tooltip)
    batch_spin.valueChanged.connect(lambda value: self._set_group_composite_max_batch_mb(group, int(value)))
    symmetry = symmetry_spec_from_config(config.get("symmetry"))
    symmetry_check = QtWidgets.QCheckBox("Apply symmetry")
    symmetry_check.setObjectName("group_composite_symmetry_enabled")
    symmetry_check.setChecked(symmetry.enabled)
    symmetry_check.setToolTip("Apply reciprocal-HKL point-group operations before composite binning; energy is unchanged.")
    symmetry_check.toggled.connect(lambda checked: self._set_group_composite_symmetry_enabled(group, checked))
    symmetry_mode = QtWidgets.QComboBox()
    symmetry_mode.setObjectName("group_composite_symmetry_mode")
    for label, value in (("Space group", "space_group"), ("Point group", "point_group"), ("Operations", "operations"), ("Generators", "generators")):
        symmetry_mode.addItem(label, value)
    displayed_symmetry_mode = (
        symmetry.mode
        if symmetry.mode != "none"
        else str(config.get("symmetry", {}).get("last_mode", "space_group"))
    )
    symmetry_mode.setCurrentIndex(
        max(symmetry_mode.findData(displayed_symmetry_mode), 0)
    )
    symmetry_mode.setToolTip("Select the notation used by the symmetry expression.")
    symmetry_mode.currentIndexChanged.connect(lambda _index, combo=symmetry_mode: self._set_group_composite_symmetry_mode(group, str(combo.currentData())))
    symmetry_expression = QtWidgets.QLineEdit(symmetry.expression)
    symmetry_expression.setObjectName("group_composite_symmetry_expression")
    symmetry_expression.setPlaceholderText("P -1")
    symmetry_expression.setToolTip("Examples: P -1; -1; x,y,z;-x,-y,-z; rotate(order=3, axis=[1,1,1]).")
    symmetry_expression.editingFinished.connect(lambda editor=symmetry_expression: self._set_group_composite_symmetry_expression(group, editor.text()))
    coverage_label = QtWidgets.QLabel("Minimum coverage")
    coverage_edit = QtWidgets.QLineEdit(_format_number(_rebin_minimum_coverage(config)))
    coverage_edit.setObjectName("group_composite_minimum_coverage")
    coverage_edit.setMaximumWidth(70)
    coverage_tooltip = (
        "Mask composite output bins whose measured geometric support is below this fraction "
        "of the requested bin volume. Enter a value from 0 to 1."
    )
    coverage_label.setToolTip(coverage_tooltip)
    coverage_edit.setToolTip(coverage_tooltip)
    coverage_edit.editingFinished.connect(
        lambda editor=coverage_edit: self._set_group_composite_minimum_coverage(
            group, editor
        )
    )
    samples_label = QtWidgets.QLabel("Minimum samples")
    samples_edit = QtWidgets.QLineEdit(
        _format_number(_rebin_minimum_samples(config))
    )
    samples_edit.setObjectName("group_composite_minimum_samples")
    samples_edit.setMaximumWidth(70)
    samples_tooltip = (
        "Mask bins receiving less than this effective number of source samples. "
        "Fractional binning sums fractional sample contributions."
    )
    samples_label.setToolTip(samples_tooltip)
    samples_edit.setToolTip(samples_tooltip)
    samples_edit.editingFinished.connect(
        lambda editor=samples_edit: self._set_group_composite_minimum_samples(
            group, editor
        )
    )
    option_row.addWidget(auto_check)
    option_row.addWidget(mean_label)
    option_row.addWidget(mean_combo)
    option_row.addStretch(1)
    footer_row = header_row + len(axes) + metadata_axis_count + 1
    controls_layout.addLayout(option_row, footer_row, 0, 1, len(headers))
    quality_row = QtWidgets.QHBoxLayout()
    quality_row.addWidget(coverage_label)
    quality_row.addWidget(coverage_edit)
    quality_row.addSpacing(12)
    quality_row.addWidget(samples_label)
    quality_row.addWidget(samples_edit)
    quality_row.addSpacing(12)
    quality_row.addWidget(batch_label)
    quality_row.addWidget(batch_spin)
    self._add_rebin_performance_controls(quality_row, group=group, composite=True)
    quality_row.addStretch(1)
    controls_layout.addLayout(quality_row, footer_row + 1, 0, 1, len(headers))
    symmetry_row = QtWidgets.QHBoxLayout()
    symmetry_row.addWidget(symmetry_check)
    symmetry_row.addWidget(symmetry_mode)
    symmetry_row.addWidget(symmetry_expression, 1)
    controls_layout.addLayout(symmetry_row, footer_row + 2, 0, 1, len(headers))
    from .metadata_dimensions import MetadataDimension, metadata_rebin_axis_config
    from .project_rebin_panels import rebin_memory_estimate_label

    cached_rebin_data = _peek_cached_composite_dataset_data(
        group,
        config_override=None if selected_binning["fit"] else config,
        binning_id=None if selected_binning["fit"] else selected_binning["id"],
    )
    memory_config = {
        **config,
        "axes": [
            *config.get("axes", []),
            *(
                metadata_rebin_axis_config(MetadataDimension(**recipe))
                for recipe in group.metadata.get("metadata_dimensions", [])
            ),
        ],
    }
    memory_label = rebin_memory_estimate_label(
        memory_config, data=cached_rebin_data, object_prefix="group_composite"
    )
    controls_layout.addWidget(memory_label, footer_row + 3, 0, 1, len(headers))
    status_label = QtWidgets.QLabel(_composite_rebin_status_text(group, config))
    status_label.setObjectName("group_composite_status")
    status_label.setWordWrap(True)
    status_label.setToolTip(
        "Shows whether the cached composite rebin is current. Pending manual rebinning will be forced automatically for fit and viewer operations."
    )
    controls_layout.addWidget(status_label, footer_row + 4, 0, 1, len(headers))
    action_row = QtWidgets.QHBoxLayout()
    rebin_now_button = QtWidgets.QPushButton("Rebin now")
    rebin_now_button.setObjectName("group_composite_rebin_now")
    rebin_now_button.setEnabled(can_combine)
    rebin_now_button.setToolTip(
        "Compute the current composite rebin immediately and update the cached viewer/fit data. Use this when automatic rebinning is off."
    )
    rebin_now_button.clicked.connect(lambda: self.rebin_composite_now(group))
    action_row.addWidget(rebin_now_button)
    rebin_all_button = QtWidgets.QPushButton("Rebin all now")
    rebin_all_button.setObjectName("group_composite_rebin_all_now")
    rebin_all_button.setEnabled(can_combine)
    rebin_all_button.setVisible(len(binnings) > 1)
    rebin_all_button.setToolTip(
        "Compute every enabled named composite binning for this collection and update open data viewers."
    )
    rebin_all_button.clicked.connect(
        lambda: self.rebin_all_composite_binnings_now(group)
    )
    action_row.addWidget(rebin_all_button)
    materialize_button = QtWidgets.QPushButton("Create dataset from composite")
    materialize_button.setObjectName("group_composite_create")
    materialize_button.setEnabled(can_combine)
    materialize_button.setToolTip(
        "Compute this composite and store it inside the nfit project as an independent dataset. "
        "The materialized result can be used as a background or as an input to a later composite."
    )
    materialize_button.clicked.connect(
        lambda: self.materialize_composite_for_group(group)
    )
    action_row.addStretch(1)
    action_row.addWidget(materialize_button)
    controls_layout.addLayout(action_row, footer_row + 5, 0, 1, len(headers))
    controls.addTab(settings_tab, "Rebin settings")

    metadata_tab = QtWidgets.QWidget()
    metadata_tab.setObjectName("group_composite_metadata_tab")
    metadata_layout = QtWidgets.QVBoxLayout(metadata_tab)
    metadata_layout.addWidget(metadata_dimensions_panel(self, group, embedded=True))
    metadata_layout.addStretch(1)
    controls.addTab(metadata_tab, "Metadata dimensions")

    from .project_composite_physics import composite_physics_panel

    physics_panel = composite_physics_panel(self, group, config)
    if physics_panel is not None:
        controls.addTab(physics_panel, "Physics")

    from .project_rebin_panels import rebin_bin_information_widget

    controls.addTab(
        rebin_bin_information_widget(
            memory_config,
            data=cached_rebin_data,
            object_prefix="group_composite",
        ),
        "Bin information",
    )
    for button in controls.findChildren(QtWidgets.QToolButton):
        button.setToolTip("Scroll composite tabs when the tab bar is too narrow.")
    layout.addWidget(controls)
    return box
