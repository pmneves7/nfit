import numpy as np
import pytest

import nfit.mdevent as mdevent
from nfit import (
    DataGroup,
    NfitProject,
    append_mdevent_file,
    bin_mdevent_group,
    bin_mdevent_powder_group,
    estimate_mdevent_peak_memory,
    inspect_mdevent_workspace,
    load_detector_normalization,
    load_mdevent_run_points,
    load_project,
    mdevent_dataset_group,
    save_project,
)
from nfit.plotting import _mdhisto_channel_array
from nfit.project_gui import (
    NfitProjectExplorer,
    _composite_scope,
    _point_data_from_mdhisto_view,
    data_group_composite_config,
    import_dataset_paths,
    slice_viewer_datasets,
)


def _write_mdevent(path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        workspace = handle.create_group("MDEventWorkspace")
        workspace.create_dataset("dimensions", data=[4])
        for index, (name, units, frame, low, high) in enumerate(
            (
                ("Q_sample_x", "Angstrom^-1", "QSample", -2, 2),
                ("Q_sample_y", "Angstrom^-1", "QSample", -2, 2),
                ("Q_sample_z", "Angstrom^-1", "QSample", -2, 2),
                ("DeltaE", "DeltaE", "DeltaE", -1, 1),
            )
        ):
            workspace.attrs[f"dimension{index}"] = (
                f'<Dimension ID="Q{index}"><Name>{name}</Name><Units>{units}</Units>'
                f"<Frame>{frame}</Frame><UpperBounds>{high}</UpperBounds>"
                f"<LowerBounds>{low}</LowerBounds><NumberOfBins>5</NumberOfBins></Dimension>"
            )
        event_group = workspace.create_group("event_data")
        event_group.create_dataset(
            "event_data",
            data=np.asarray(
                [
                    [1, 1, 0, 0, 10, 0, 0, 0, 0],
                    [1, 1, 1, 0, 10, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
        )
        for index, charge in enumerate((1.0, 2.0)):
            experiment = workspace.create_group(f"experiment{index}")
            logs = experiment.create_group("logs")
            _log(logs, "run_number", np.asarray([100 + index], dtype="S8"))
            _log(logs, "Ei", [10.0])
            _log(logs, "gd_prtn_chrg", [charge])
            _log(logs, "omega", [0.0])
            _log(logs, "phi", [0.0])
            _log(logs, "chi", [0.0])
            _log(logs, "duration", [1.0])
            _log(logs, "CalculatedT0", [2.5])
            _log(logs, "processed_histogram_bins", [-1.0, 1.0])
            _log(logs, "RUBW_MATRIX", np.eye(3).reshape(-1))
            goniometer = logs.create_group("goniometer")
            goniometer.create_dataset("rotation_matrix", data=np.eye(3).reshape(-1))
            instrument = experiment.create_group("instrument")
            detectors = instrument.create_group("physical_detectors")
            detectors.create_dataset("detector_number", data=[10])
            detectors.create_dataset("polar_angle", data=[0.0])
            detectors.create_dataset("azimuthal_angle", data=[0.0])
            sample = experiment.create_group("sample")
            lattice = sample.create_group("oriented_lattice")
            lattice.create_dataset("orientation_matrix", data=np.eye(3) / (2.0 * np.pi))
            for name, value in zip(
                ("a", "b", "c", "alpha", "beta", "gamma"),
                (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
                strict=True,
            ):
                lattice.create_dataset(f"unit_cell_{name}", data=[value])


def _log(logs, name, values):
    group = logs.create_group(name)
    group.create_dataset("value", data=values)


def _write_normalization(path, value):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("mantid_workspace_1")
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector")
        detector.create_dataset("detector_list", data=[10])
        workspace = entry.create_group("workspace")
        workspace.create_dataset("values", data=[[value]])
        workspace.create_dataset("errors", data=[[0.1 if value else 0.0]])


def test_mdevent_metadata_grouping_and_hkl_run_loading(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)

    info = inspect_mdevent_workspace(source)
    group = mdevent_dataset_group(source)
    progress = []
    points = load_mdevent_run_points(
        group.datasets[0],
        batch_size=1,
        progress_callback=progress.append,
    )

    assert info.event_count == 2
    assert [run.run_number for run in info.runs] == ["100", "101"]
    assert len(group.datasets) == 2
    assert "ub_matrix" not in group.datasets[0].metadata
    np.testing.assert_allclose(points.H, [0.0])
    np.testing.assert_allclose(points.K, [0.0])
    np.testing.assert_allclose(points.L, [0.0])
    np.testing.assert_allclose(points.E, [0.0])
    assert [event["iteration"] for event in progress] == [0, 1, 2]
    assert all(event["stage"] == "mdevent_scan" for event in progress)
    assert progress[-1]["total"] == 2


def test_mdevent_reconstructs_missing_goniometer_matrix_from_axes(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    with h5py.File(source, "r+") as handle:
        workspace = handle["MDEventWorkspace"]
        for index in range(2):
            goniometer = workspace[f"experiment{index}/logs/goniometer"]
            del goniometer["rotation_matrix"]
            axis = goniometer.create_group("axis0")
            angle = axis.create_dataset("angle", data=[90.0])
            angle.attrs["unit"] = "deg"
            angle.attrs["sense"] = "CCW"
            axis.create_dataset("rotationaxis", data=[0.0, 1.0, 0.0])

    info = inspect_mdevent_workspace(source)

    expected = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(info.runs[0].goniometer, expected, atol=1.0e-15)


def test_detector_normalization_zero_values_define_mask(tmp_path):
    path = tmp_path / "van.nxs"
    _write_normalization(path, 0.0)

    normalization = load_detector_normalization(path)

    np.testing.assert_array_equal(normalization.masked_detector_ids, [10])


def test_native_mdevent_binning_uses_proton_charge_and_vanadium_coverage(tmp_path):
    source = tmp_path / "events.nxs"
    vanadium = tmp_path / "van.nxs"
    _write_mdevent(source)
    _write_normalization(vanadium, 2.0)
    group = mdevent_dataset_group(
        source,
        normalization_path=vanadium,
        mask_path=vanadium,
    )

    result = bin_mdevent_group(
        group,
        lower=[-1, -1, -1, -1],
        upper=[1, 1, 1, 1],
        num_bins=[1, 1, 1, 1],
    )

    # Two unit events over 2 * (charge 1 + charge 2) * 2 meV coverage.
    np.testing.assert_allclose(result.signal, [[[[1.0 / 6.0]]]])
    np.testing.assert_allclose(result.errors, [[[[np.sqrt(2.0) / 12.0]]]])
    np.testing.assert_allclose(result.num_events, [[[[2.0]]]])
    assert not result.mask.item()
    assert result.metadata["signal_semantics"] == "density"
    assert result.metadata["signal_semantics_source"] == "nfit_mdevent_reduction"


def test_native_mdevent_supports_one_nonuniform_axis_and_minimum_samples(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)

    result = bin_mdevent_group(
        group,
        lower=[-1, -1, -1, -1],
        upper=[1, 1, 1, 1],
        num_bins=[1, 1, 1, 1],
        bin_edges=[None, None, None, [-1.0, 0.0, 0.25, 1.0]],
        minimum_samples=3,
    )

    assert result.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(result.axes[3].values, [-1.0, 0.0, 0.25, 1.0])
    assert np.all(result.mask)
    assert result.metadata["rebin"]["minimum_samples"] == 3.0


def test_native_mdevent_powder_binning_uses_radial_trajectory_normalization(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)

    result = bin_mdevent_powder_group(
        group,
        lower=[0.0, -1.0],
        upper=[1.0, 1.0],
        num_bins=[1, 1],
    )

    assert result.shape == (1, 1)
    assert result.axes[0].name == "|Q|"
    np.testing.assert_allclose(result.signal, [[1.0 / 3.0]])
    np.testing.assert_allclose(result.errors, [[np.sqrt(2.0) / 6.0]])
    np.testing.assert_allclose(result.num_events, [[2.0]])
    assert not result.mask.item()
    assert result.metadata["signal_semantics_source"] == (
        "nfit_mdevent_powder_reduction"
    )


def test_native_mdevent_covered_zero_bins_are_finite_measured_zeros(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)

    result = bin_mdevent_group(
        group,
        lower=[-1, -1, -1, -1], upper=[1, 1, 1, 1],
        num_bins=[1, 1, 1, 2],
    )

    covered = result.metadata["normalization_denominator"] > 0.0
    covered_zero = covered & (result.num_events == 0.0)
    assert np.count_nonzero(covered_zero) == 1
    assert np.all(result.signal[covered_zero] == 0.0)
    assert np.all(np.isfinite(result.errors[covered_zero]))
    assert np.all(result.errors[covered_zero] > 0.0)
    np.testing.assert_allclose(
        result.errors[covered_zero],
        1.29 * result.metadata["event_weight_rms"] / result.metadata["normalization_denominator"][covered_zero],
    )
    assert result.metadata["zero_count_error_model"] == (
        "feldman_cousins_68_percent_upper_limit_scaled_by_rms_event_weight"
    )
    assert not np.any(result.mask[covered_zero])
    assert np.all(_mdhisto_channel_array(result, "signal")[covered_zero] == 0.0)
    fit_points = _point_data_from_mdhisto_view(result)
    assert np.count_nonzero(fit_points.valid_mask()) == 2


def test_native_mdevent_mask_removes_events_and_coverage(tmp_path):
    source = tmp_path / "events.nxs"
    mask = tmp_path / "mask.nxs"
    _write_mdevent(source)
    _write_normalization(mask, 0.0)
    group = mdevent_dataset_group(source, mask_path=mask)

    result = bin_mdevent_group(
        group,
        lower=[-1, -1, -1, -1],
        upper=[1, 1, 1, 1],
        num_bins=[1, 1, 1, 1],
    )

    assert result.num_events.item() == 0.0
    assert result.mask.item()


def test_native_mdevent_accepts_custom_hkl_basis_without_changing_totals(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    progress = []

    result = bin_mdevent_group(
        group,
        lower=[-2, -2, -2, -1], upper=[2, 2, 2, 1], num_bins=[1, 1, 1, 1],
        vectors=[[1, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, 1]],
        axis_names=["[H,H,0]", "L", "[K,-K,0]", "DeltaE"],
        progress_callback=progress.append,
    )

    assert result.num_events.item() == 2.0
    assert [axis.name for axis in result.axes] == ["[H,H,0]", "L", "[K,-K,0]", "DeltaE"]
    assert progress[0]["stage"] == "mdevent_events"
    assert "mdevent_normalization_setup" in {event["stage"] for event in progress}
    assert "mdevent_normalization" in {event["stage"] for event in progress}
    assert progress[-1]["stage"] == "mdevent_finalize"
    assert progress[-1]["iteration"] == progress[-1]["total"]


def test_symmetry_trajectory_numba_matches_python_fallback(monkeypatch, tmp_path):
    if mdevent._MDEVENT_NUMBA is None:
        pytest.skip("Numba MDEvent normalization is unavailable")
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        # Exercise geometry grouping as well as multiple symmetry operations.
        handle[
            "MDEventWorkspace/experiment1/instrument/physical_detectors/polar_angle"
        ][...] = [35.0]
    group = mdevent_dataset_group(source)
    operations = [np.eye(3), np.diag([1.0, 1.0, -1.0])]
    progress = []

    monkeypatch.setattr(mdevent._parallel, "num_threads", lambda: 1)
    compiled_single_worker = bin_mdevent_group(
        group,
        lower=[-2, -2, -2, -1],
        upper=[2, 2, 2, 1],
        num_bins=[2, 2, 2, 2],
        symmetry_operations=operations,
        progress_callback=progress.append,
    )
    monkeypatch.setattr(mdevent._parallel, "num_threads", lambda: 2)
    compiled = bin_mdevent_group(
        group,
        lower=[-2, -2, -2, -1],
        upper=[2, 2, 2, 1],
        num_bins=[2, 2, 2, 2],
        symmetry_operations=operations,
    )
    monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    fallback = bin_mdevent_group(
        group,
        lower=[-2, -2, -2, -1],
        upper=[2, 2, 2, 1],
        num_bins=[2, 2, 2, 2],
        symmetry_operations=operations,
    )

    np.testing.assert_allclose(compiled.signal, fallback.signal, equal_nan=True)
    np.testing.assert_allclose(compiled.errors, fallback.errors, equal_nan=True)
    np.testing.assert_allclose(
        compiled.metadata["normalization_denominator"],
        fallback.metadata["normalization_denominator"],
    )
    np.testing.assert_array_equal(compiled.mask, fallback.mask)
    np.testing.assert_allclose(
        compiled.metadata["normalization_denominator"],
        compiled_single_worker.metadata["normalization_denominator"],
    )
    event_updates = [
        event for event in progress if event["stage"] == "mdevent_events"
    ]
    assert event_updates[0]["total"] == 4
    assert event_updates[-1]["iteration"] == 4


def test_trajectory_workers_honor_cpu_ceiling_and_available_memory(monkeypatch):
    monkeypatch.setattr(mdevent._parallel, "num_threads", lambda: 16)
    monkeypatch.setattr(
        mdevent,
        "_available_memory_bytes",
        lambda: 2 * 1024**3,
    )

    assert mdevent._trajectory_worker_count(1024) == 16
    assert mdevent._trajectory_worker_count(16_000_000) == 4

    monkeypatch.setattr(
        mdevent,
        "_available_memory_bytes",
        lambda: 512 * 1024**3,
    )
    assert mdevent._trajectory_worker_count(1_470_183_435) == 11


def test_mdevent_memory_estimate_scales_with_output_grid_and_preflight_blocks(monkeypatch, tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    assert estimate_mdevent_peak_memory([20, 20, 20, 50]) < estimate_mdevent_peak_memory([200, 200, 200, 50])
    monkeypatch.setattr("nfit.mdevent._available_memory_bytes", lambda: 2 * 1024**3)

    with pytest.raises(MemoryError, match="estimated to peak"):
        bin_mdevent_group(
            group, lower=[-1] * 4, upper=[1] * 4,
            num_bins=[200, 200, 200, 50],
        )


def test_project_import_creates_nested_group_and_serializes_shared_setup_once(tmp_path):
    source = tmp_path / "events.nxs"
    vanadium = tmp_path / "van_test"
    project_path = tmp_path / "project.nfit"
    _write_mdevent(source)
    _write_normalization(vanadium, 1.0)
    root = DataGroup("Data")

    entries = import_dataset_paths(root, [source], data_type="single_crystal_inelastic")

    assert len(entries) == 2
    assert len(root.subgroups) == 1
    subgroup = root.subgroups[0]
    assert subgroup.metadata["mdevent"]["normalization_file"] == str(vanadium)
    assert subgroup.metadata["mdevent"]["mask_file"] == str(vanadium)
    assert "ub_matrix" not in entries[0].metadata
    save_project(NfitProject([root]), project_path)
    loaded = load_project(project_path)
    loaded_subgroup = loaded.data_groups[0].subgroups[0]
    assert len(loaded_subgroup.datasets) == 2
    assert loaded_subgroup.metadata["mdevent"]["ub_matrix"] == subgroup.metadata["mdevent"]["ub_matrix"]
    assert "ub_matrix" not in loaded_subgroup.datasets[0].metadata


def test_mdevent_import_reports_each_experiment_and_raw_runs_are_not_viewed(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    root = DataGroup("Data")
    progress = []

    import_dataset_paths(
        root, [source], data_type="single_crystal_inelastic",
        progress_callback=progress.append,
    )

    assert [(event["iteration"], event["total"]) for event in progress] == [(1, 2), (2, 2)]
    assert slice_viewer_datasets(root, use_composite=False) == ([], [])


def test_append_individual_mdevent_file_reuses_shared_orientation(tmp_path):
    first = tmp_path / "first.nxs"
    second = tmp_path / "second.nxs"
    _write_mdevent(first)
    _write_mdevent(second)
    group = mdevent_dataset_group(first)

    added = append_mdevent_file(group, second)

    assert len(added) == 2
    assert len(group.datasets) == 4
    assert group.metadata["mdevent"]["source_files"] == [str(first), str(second)]
    assert added[0].name == "run 100 (2)"


def test_mdevent_group_gui_exposes_shared_setup_and_defaults_manual(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    source = tmp_path / "events.nxs"
    vanadium = tmp_path / "van_test"
    _write_mdevent(source)
    _write_normalization(vanadium, 1.0)
    root = DataGroup("Data")
    import_dataset_paths(root, [source], data_type="single_crystal_inelastic")
    subgroup = root.subgroups[0]
    explorer = NfitProjectExplorer(NfitProject([root]))
    explorer._refresh_tree(select_dataset_group=subgroup)

    norm = explorer.details_widget.findChild(QtWidgets.QLineEdit, "mdevent_normalization_file")
    mask = explorer.details_widget.findChild(QtWidgets.QLineEdit, "mdevent_mask_file")
    ei = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "mdevent_incident_energy_override")
    t0 = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "mdevent_t0_override")
    ub = explorer.details_widget.findChild(QtWidgets.QLineEdit, "mdevent_ub_matrix")
    assert all(widget is not None and widget.toolTip() for widget in (norm, mask, ei, t0, ub))
    config = data_group_composite_config(_composite_scope(root, subgroup))
    assert config["enabled"] is True
    assert config["auto_rebin"] is False
    assert len(config["axes"]) == 4
    coordinate_mode = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_coordinate_mode"
    )
    assert coordinate_mode is not None and coordinate_mode.toolTip()
    assert coordinate_mode.currentData() == "hkle"
    matrix = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "group_composite_momentum_matrix"
    )
    assert matrix is not None and matrix.toolTip()
    matrix.setText("[[1, 1, 0], [0, 0, 1], [1, -1, 0]]")
    matrix.editingFinished.emit()
    assert [axis["vector"][:3] for axis in config["axes"][:3]] == [
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ]
    assert [axis["name"] for axis in config["axes"][:3]] == [
        "[H,H,0]",
        "[0,0,L]",
        "[K,-K,0]",
    ]

    boxes = explorer.details_widget.findChildren(QtWidgets.QGroupBox)
    titles = [box.title() for box in boxes]
    assert titles.index("Composite dataset") < titles.index("Datasets")

    ei.setValue(12.5)
    assert subgroup.metadata["mdevent"]["incident_energy_override"] == 12.5
    assert config["stale"] is True
    explorer._set_group_composite_coordinate_mode(
        _composite_scope(root, subgroup), "powder"
    )
    assert config["coordinate_mode"] == "powder"
    assert [axis["name"] for axis in config["axes"]] == ["|Q|", "DeltaE"]


def test_mdevent_rebin_warns_before_estimated_ram_overcommit(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    root = DataGroup("Data")
    import_dataset_paths(root, [source], data_type="single_crystal_inelastic")
    subgroup = root.subgroups[0]
    scope = _composite_scope(root, subgroup)
    data_group_composite_config(scope)["enabled"] = True
    explorer = NfitProjectExplorer(NfitProject([root]))
    warnings = []
    monkeypatch.setattr("nfit.project_gui.assess_mdevent_memory", lambda *args, **kwargs: (100 * 1024**3, 10 * 1024**3, True))
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]) or QtWidgets.QMessageBox.StandardButton.No,
    )

    assert explorer.rebin_composite_now(scope) is False
    assert "available RAM" in warnings[0]
    assert "_allow_memory_overcommit_once" not in data_group_composite_config(scope)
