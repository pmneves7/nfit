from pathlib import Path

import h5py
import numpy as np
import pytest

from nfit import project_gui
from nfit.dataset import PointData4D
from nfit.importers import IMPORTERS, import_with, probe_importers
from nfit.macs import import_macs_nexus, is_macs_nexus_file
from nfit.mdhisto import MDHistoData
from nfit.pipeline import DataGroup
from nfit.plotting import default_plot_grid_fill_neighbors
from nfit.project_gui import (
    GROUP_COMPOSITE_KEY,
    NfitProjectExplorer,
    composite_dataset_data,
    data_group_composite_config,
    import_dataset_paths,
    load_project,
    save_project,
)
from nfit.project_io import NfitProject
from nfit.rebin import rebin_nd


def _write_macs_nexus(
    path: Path, *, points: int = 24, energy_transfer: float = 1.2
) -> Path:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("facility", data=np.bytes_("NCNR"))
        das = entry.create_group("DAS_logs")
        experiment = das.create_group("experiment")
        experiment.create_dataset("instrument", data=np.bytes_("MACS"))

        scan = np.arange(points, dtype=float)
        spec = np.full((points, 20), 9.0)
        spec[:, 18] = 0.0
        spec[0, 18] = 1.0
        diff = np.full((points, 20), 100.0)
        for name, counts in (("specDetector", spec), ("diffDetector", diff)):
            detector = das.create_group(name)
            detector.create_dataset("counts", data=counts)
            detector.create_dataset("detectorEfficiency", data=np.ones(20))
            detector.create_dataset("roiMask", data=np.ones((points, 20), dtype=np.uint8))

        counter = das.create_group("counter")
        counter.create_dataset("liveMonitor", data=np.full(points, 2.0e5))
        ei = das.create_group("ei")
        ei.create_dataset("energy", data=np.full(points, 3.7 + energy_transfer))
        ef = das.create_group("ef")
        ef.create_dataset("energy", data=np.full(points, 3.7))
        sample_theta = das.create_group("sampleTheta")
        sample_theta.create_dataset("primaryNode", data=-30.0 + scan)
        kidney = das.create_group("kidneyMotor")
        kidney.create_dataset("softPosition", data=-20.0 + 0.1 * scan)

        state = das.create_group("sampleState")
        for name, value in (
            ("A", 8.24),
            ("B", 8.24),
            ("C", 8.24),
            ("Alpha", 90.0),
            ("Beta", 90.0),
            ("Gamma", 90.0),
            ("a3Zero", 0.0),
        ):
            state.create_dataset(name, data=np.full(points, value))
        state.create_dataset("refPlane1", data=np.tile([1.0, 1.0, 0.0], (points, 1)))
        state.create_dataset("refPlane2", data=np.tile([0.0, 0.0, 1.0], (points, 1)))

        common = das.create_group("anaTwoTheta")
        common.create_dataset("primaryNode", data=np.full(points, 89.0))
        for channel in range(1, 21):
            analyzer = das.create_group(f"anaTwoTheta{channel:02d}")
            angle = 91.5 if channel == 20 else 89.0
            analyzer.create_dataset("softPosition", data=np.full(points, angle))

        temp = das.create_group("temp")
        temp.create_dataset("primarySensor", data=np.asarray([np.bytes_("A")]))
        sensor = temp.create_group("sensor_A")
        sensor.create_dataset("average_value", data=np.full(points, 1.5))
    return path


def test_macs_probe_uses_internal_instrument_metadata(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.unusual")

    assert is_macs_nexus_file(source)
    matches = probe_importers(source, "single_crystal_inelastic")
    assert matches[0].importer_name == "macs_nexus"
    assert matches[0].confidence == 1.0
    assert IMPORTERS["macs_nexus"].can_read(source)

    not_macs = tmp_path / "named_like_macs.nxs.ng0"
    with h5py.File(not_macs, "w") as handle:
        handle.create_group("entry").attrs["NX_class"] = "NXentry"
    assert not is_macs_nexus_file(not_macs)


def test_macs_spec_and_diff_are_distinct_point_streams(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.nxs.ng0")
    spec = import_with(
        "macs_nexus",
        source,
        {"stream": "spec", "a3_offset_deg": 66.5},
    )
    diff = import_macs_nexus(source, {"stream": "diff", "a3_offset_deg": 66.5})

    assert isinstance(spec, PointData4D)
    assert spec.size == 24 * 20
    np.testing.assert_allclose(spec.E, 1.2)
    np.testing.assert_allclose(diff.E, 0.0)
    assert spec.metadata["detector_stream"] == "SPEC"
    assert diff.metadata["detector_stream"] == "DIFF"
    assert "elastic projection" in diff.metadata["coordinate_approximation"]
    assert spec.metadata["monitor_target"] == 1.0e6
    np.testing.assert_allclose(spec.intensity[:18], 45.0)
    np.testing.assert_allclose(spec.sigma[:18], 15.0)
    assert spec.temperature == pytest.approx(np.full(spec.size, 1.5))

    # Channel 19 is unresponsive and channel 20 is misset; neither condition is
    # inherited by the unanalysed DIFF detector stream.
    assert spec.metadata["masked_analyzer_channels"] == [19, 20]
    assert not np.any(spec.mask.reshape(24, 20)[:, 18:])
    assert np.all(diff.mask)
    assert np.all(diff.intensity == 500.0)


def test_macs_spec_recovers_final_energy_from_aligned_analyzers(tmp_path):
    source = _write_macs_nexus(tmp_path / "stale_ef.nxs.ng0")
    with h5py.File(source, "r+") as handle:
        logs = handle["entry/DAS_logs"]
        logs["ef/energy"][...] = 3.5
        logs["ef"].create_dataset("dSpacing", data=np.full(24, 3.35416))
        logs["anaTwoTheta/primaryNode"][...] = np.nan

    data = import_macs_nexus(source, {"stream": "spec"})

    theta = np.deg2rad(89.0 / 2.0)
    expected_ef = 81.8042 / (2.0 * 3.35416 * np.sin(theta)) ** 2
    np.testing.assert_allclose(data.E, 4.9 - expected_ef)
    assert data.metadata["fixed_final_energy_meV"] == pytest.approx(expected_ef)
    assert data.metadata["recorded_fixed_final_energy_meV"] == pytest.approx(3.5)
    assert "DAVE convention" in data.metadata["fixed_final_energy_source"]
    assert data.metadata["dataset_parameters"]["spectral_channels"][
        "final_energy_meV"
    ] == pytest.approx(expected_ef)
    assert data.metadata["masked_analyzer_channels"] == [19, 20]

    unmasked = import_macs_nexus(
        source,
        {"stream": "spec", "mask_misaligned_analyzers": False},
    )
    np.testing.assert_allclose(unmasked.E, data.E)
    assert np.all(unmasked.mask.reshape(24, 20)[:, 19])


def test_macs_batch_import_expands_streams_and_prepares_composites(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    first = _write_macs_nexus(tmp_path / "first.nxs.ng0")
    second = _write_macs_nexus(
        tmp_path / "second.nxs.ng0", energy_transfer=1.6
    )
    group = DataGroup("MACS")

    entries = import_dataset_paths(
        group,
        [first, second],
        data_type="single_crystal_inelastic",
        importer_options={
            str(first): {"a3_offset_deg": 66.5},
            str(second): {"a3_offset_deg": 66.5},
        },
    )

    assert len(entries) == 4
    assert [node.name for node in group.subgroups] == ["MACS SPEC", "MACS DIFF"]
    assert [len(node.datasets) for node in group.subgroups] == [2, 2]
    assert all(node.metadata[GROUP_COMPOSITE_KEY]["enabled"] for node in group.subgroups)
    assert {entry.data_type for entry in group.subgroups[0].datasets} == {
        "single_crystal_inelastic"
    }
    assert {entry.data_type for entry in group.subgroups[1].datasets} == {
        "single_crystal_energy_integrated"
    }
    assert group.lattice_parameters["a"] == pytest.approx(8.24)
    assert all(entry.metadata["import_options"]["stream"] for entry in entries)
    for node in group.subgroups:
        np.testing.assert_allclose(
            node.metadata["ub_matrix"], node.datasets[0].data.metadata["ub_matrix"]
        )
        scope = project_gui._composite_scope(group, node)
        config = data_group_composite_config(scope)
        assert np.prod([axis["num_bins"] for axis in config["axes"]]) <= 32**4
        composite = composite_dataset_data(scope)
        assert isinstance(composite, MDHistoData)
        assert composite.signal.size <= 2_000_000
        assert np.count_nonzero(~composite.mask) > 0
        assert default_plot_grid_fill_neighbors(composite) == (
            2 if node.name.startswith("MACS SPEC") else 0
        )
        np.testing.assert_allclose(
            composite.metadata["ub_matrix"], node.metadata["ub_matrix"]
        )

    project_path = tmp_path / "macs.nfit"
    save_project(NfitProject([group]), project_path)
    reloaded = load_project(project_path)
    lazy_root = reloaded.data_groups[0]
    assert all(dataset.data is None for dataset in lazy_root.iter_datasets())
    assert project_gui._dataset_collection_point_count(
        lazy_root.subgroups[0]
    ) == 2 * 24 * 20
    assert project_gui._has_slice_viewer_candidates(lazy_root)

    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    explorer = NfitProjectExplorer(reloaded)
    crystal_box = explorer._ub_setup_group_box(
        lazy_root, lazy_root.subgroups[0]
    )
    ub_display = crystal_box.findChild(QtWidgets.QLabel, "ub_matrix_display")
    assert ub_display is not None
    assert "UB matrix" in ub_display.text()
    assert "not set" not in ub_display.text()
    assert ub_display.toolTip().strip()
    explorer.window.close()


def test_separate_macs_imports_reuse_existing_stream_groups(tmp_path):
    first = _write_macs_nexus(tmp_path / "first.nxs.ng0")
    second = _write_macs_nexus(tmp_path / "second.nxs.ng0")
    group = DataGroup("MACS")

    import_dataset_paths(group, [first], data_type="single_crystal_inelastic")
    for node in group.subgroups:
        node.metadata.pop("importer")
        node.metadata.pop("source_stream")
    import_dataset_paths(group, [second], data_type="single_crystal_inelastic")

    assert [node.name for node in group.subgroups] == ["MACS SPEC", "MACS DIFF"]
    assert [len(node.datasets) for node in group.subgroups] == [2, 2]
    assert all(node.metadata["importer"] == "macs_nexus" for node in group.subgroups)


def test_separate_macs_import_can_create_new_stream_groups(tmp_path):
    first = _write_macs_nexus(tmp_path / "first.nxs.ng0")
    second = _write_macs_nexus(tmp_path / "second.nxs.ng0")
    group = DataGroup("MACS")

    import_dataset_paths(group, [first], data_type="single_crystal_inelastic")
    import_dataset_paths(
        group,
        [second],
        data_type="single_crystal_inelastic",
        stream_group_mode="new",
    )

    assert [node.name for node in group.subgroups] == [
        "MACS SPEC",
        "MACS DIFF",
        "MACS SPEC1",
        "MACS DIFF1",
    ]
    assert [len(node.datasets) for node in group.subgroups] == [1, 1, 1, 1]


def test_legacy_lazy_macs_entry_reports_points_without_loading(tmp_path):
    source = _write_macs_nexus(tmp_path / "scan.nxs.ng0", points=17)
    group = DataGroup("MACS")
    entries = import_dataset_paths(
        group, [source], data_type="single_crystal_inelastic"
    )
    spec = entries[0]
    spec.metadata.pop("source_point_count", None)
    spec.unload_data()

    assert project_gui._dataset_data_point_count(spec) == 17 * 20
    assert spec.data is None
    assert spec.metadata["source_point_count"] == 17 * 20


def test_macs_import_dialog_scientific_controls_have_tooltips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    explorer = NfitProjectExplorer(NfitProject([DataGroup("MACS")]))

    def reject_after_inspection(dialog):
        for name in (
            "macs_a3_offset",
            "macs_stream_group_mode",
            "macs_monitor_target",
            "macs_apply_efficiency",
            "macs_mask_alignment",
            "macs_alignment_tolerance",
            "macs_detect_dead",
            "macs_manual_channels",
        ):
            control = dialog.findChild(QtWidgets.QWidget, name)
            assert control is not None
            assert control.toolTip().strip()
        offset = dialog.findChild(QtWidgets.QLineEdit, "macs_a3_offset")
        assert offset is not None
        assert not offset.text()
        assert "a3Zero" in offset.placeholderText()
        group_mode = dialog.findChild(QtWidgets.QComboBox, "macs_stream_group_mode")
        assert group_mode is not None
        assert group_mode.currentData() == "reuse"
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QDialog, "exec", reject_after_inspection)
    assert explorer._prompt_macs_nexus_options([tmp_path / "scan.nxs.ng0"]) is False
    explorer.window.close()


def test_macs_import_dialog_returns_new_group_choice(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    explorer = NfitProjectExplorer(NfitProject([DataGroup("MACS")]))

    def choose_new_groups(dialog):
        group_mode = dialog.findChild(QtWidgets.QComboBox, "macs_stream_group_mode")
        group_mode.setCurrentIndex(group_mode.findData("new"))
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", choose_new_groups)
    result = explorer._prompt_macs_nexus_options([tmp_path / "scan.nxs.ng0"])

    assert isinstance(result, tuple)
    options, mode = result
    assert mode == "new"
    assert str(tmp_path / "scan.nxs.ng0") in options
    explorer.window.close()


REAL_MACS_FILE = Path(
    "/Users/pmneves/Library/CloudStorage/OneDrive-JohnsHopkins/_research/"
    "LiV2O4/2026_08_MACS/data/Ef3p7_et_1.2_244.nxs.ng0"
)
REAL_MACS_2MEV_FILE = REAL_MACS_FILE.with_name("Ef5p0_et_2_311.nxs.ng0")


@pytest.mark.skipif(not REAL_MACS_FILE.exists(), reason="local MACS validation file is absent")
def test_local_macs_file_masks_bad_spec_channel_and_matches_recorded_ptai_hkl():
    data = import_macs_nexus(REAL_MACS_FILE, {"stream": "spec"})

    assert data.metadata["masked_analyzer_channels"] == [19]
    assert not np.any(data.mask.reshape(-1, 20)[:, 18])
    with h5py.File(REAL_MACS_FILE, "r") as handle:
        sample = handle["entry/sample"]
        expected = np.column_stack([sample[name][()] for name in ("qh", "qk", "ql")])
    # The NeXus qh/qk/ql values refer to PTAI channel 2. DAVE and nfit use the
    # same 2.0721246 meV Å² conversion; remaining differences are float storage.
    actual = np.column_stack(
        (
            data.H.reshape(-1, 20)[:, 1],
            data.K.reshape(-1, 20)[:, 1],
            data.L.reshape(-1, 20)[:, 1],
        )
    )
    np.testing.assert_allclose(actual, expected, atol=1.1e-4, rtol=0.0)


@pytest.mark.skipif(
    not REAL_MACS_2MEV_FILE.exists(), reason="local MACS 2 meV validation file is absent"
)
def test_local_macs_2mev_file_ignores_stale_common_final_energy_log():
    data = import_macs_nexus(REAL_MACS_2MEV_FILE, {"stream": "spec"})

    assert np.nanmedian(data.E) == pytest.approx(2.001, abs=5.0e-4)
    assert data.metadata["fixed_final_energy_meV"] == pytest.approx(4.9991, abs=5.0e-4)
    assert data.metadata["recorded_fixed_final_energy_meV"] == pytest.approx(
        4.93386, abs=5.0e-5
    )


def _dave_arithmetic_stepgrid(
    x: np.ndarray,
    y: np.ndarray,
    signal: np.ndarray,
    error: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent expression of DAVE ``dm_stepgrid_bin`` arithmetic mode."""

    lower = np.asarray([np.min(x) - step / 2.0, np.min(y) - step / 2.0])
    upper = np.asarray([np.max(x) + step / 2.0, np.max(y) + step / 2.0])
    shape = np.ceil((upper - lower) / step).astype(int)
    indices = np.floor((np.column_stack((x, y)) - lower) / step).astype(int)
    sample_count = np.zeros(tuple(shape), dtype=int)
    signal_sum = np.zeros(tuple(shape), dtype=float)
    variance_sum = np.zeros(tuple(shape), dtype=float)
    np.add.at(sample_count, (indices[:, 0], indices[:, 1]), 1)
    np.add.at(signal_sum, (indices[:, 0], indices[:, 1]), signal)
    np.add.at(variance_sum, (indices[:, 0], indices[:, 1]), np.square(error))
    occupied = sample_count > 0
    binned_signal = np.full(tuple(shape), np.nan)
    binned_error = np.full(tuple(shape), np.nan)
    binned_signal[occupied] = signal_sum[occupied] / sample_count[occupied]
    binned_error[occupied] = (
        np.sqrt(variance_sum[occupied]) / sample_count[occupied]
    )
    return occupied, binned_signal, binned_error


@pytest.mark.skipif(not REAL_MACS_FILE.exists(), reason="local MACS validation file is absent")
def test_local_macs_hh0_l_slice_matches_dave_hard_grid_arithmetic_binning():
    data = import_macs_nexus(
        REAL_MACS_FILE,
        {"stream": "spec", "a3_offset_deg": 66.5},
    ).valid(require_positive_sigma=False)
    step = 0.05
    x = 0.5 * (data.H + data.K)
    y = data.L
    occupied, expected_signal, expected_error = _dave_arithmetic_stepgrid(
        x,
        y,
        data.intensity,
        data.sigma,
        step,
    )
    lower = [float(np.min(x)), float(np.min(y))]
    # Include DAVE's final (possibly beyond-data) center on the regular grid.
    upper = [lo + (size - 1) * step for lo, size in zip(lower, occupied.shape, strict=True)]
    actual = rebin_nd(
        data.intensity,
        np.column_stack((x, y)),
        data_errs=data.sigma,
        lower=lower,
        upper=upper,
        step_size=[step, step],
        fractional=False,
        mean_weighting="uniform",
    )

    np.testing.assert_array_equal(actual.n_samples > 0, occupied)
    np.testing.assert_allclose(actual.binned_data[occupied], expected_signal[occupied])
    np.testing.assert_allclose(
        actual.binned_data_errs[occupied], expected_error[occupied]
    )
