import numpy as np
import pytest

import nfit.mdevent as mdevent
from nfit import (
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
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
    project_powder_background_mdevent,
    save_project,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import MaskSpec
from nfit.plotting import _mdhisto_channel_array
from nfit.project_gui import (
    NfitProjectExplorer,
    _apply_composite_backgrounds,
    _composite_scope,
    _point_data_from_mdhisto_view,
    data_group_composite_config,
    import_dataset_paths,
    slice_viewer_datasets,
)


def test_mdevent_preflight_warns_above_half_available_ram(monkeypatch):
    bins = (10, 10, 10, 10)
    estimate = estimate_mdevent_peak_memory(bins)
    monkeypatch.setattr(mdevent, "_available_memory_bytes", lambda: estimate * 2)
    assert not mdevent.assess_mdevent_memory(bins)[2]
    monkeypatch.setattr(mdevent, "_available_memory_bytes", lambda: int(estimate * 1.8))
    assert mdevent.assess_mdevent_memory(bins)[2]


def test_full_compressed_cache_dialog_explains_save_and_discard(monkeypatch):
    from PySide6 import QtWidgets

    from nfit.project_cache_gui import CompressedCachePrompt
    from nfit.rebin_cache import CompressedBinning
    from tests.project_gui_test_support import _tiny_mdhisto_data

    explorer = NfitProjectExplorer(NfitProject([DataGroup("cache")]))
    prompt = CompressedCachePrompt(explorer.window)
    artifact = CompressedBinning.from_data(_tiny_mdhisto_data(1.0), max_bytes=10_000)
    seen = []

    def choose_discard(message):
        buttons = message.buttons()
        assert len(buttons) == 3
        assert all(button.toolTip() for button in buttons)
        seen.append(message.text())
        next(button for button in buttons if "Discard" in button.text()).click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_discard)
    prompt.request("named oldest bin", artifact)
    prompt.request("another old bin", artifact)
    assert "named oldest bin" in seen[0]
    assert "compressed NPZ" in seen[0]
    assert "single combined allowance" not in seen[0]
    assert len(seen) == 1


def test_compressed_cache_dialog_reuses_chosen_session_folder(
    monkeypatch, tmp_path
):
    from PySide6 import QtWidgets

    from nfit import project_cache_gui
    from nfit.project_cache_gui import CompressedCachePrompt
    from nfit.rebin_cache import CompressedBinning
    from tests.project_gui_test_support import _tiny_mdhisto_data

    explorer = NfitProjectExplorer(NfitProject([DataGroup("cache")]))
    prompt = CompressedCachePrompt(explorer.window)
    artifact = CompressedBinning.from_data(_tiny_mdhisto_data(1.0), max_bytes=10_000)
    prompts = []

    def choose_cache(message):
        prompts.append(message.text())
        next(
            button for button in message.buttons() if "disk cache" in button.text()
        ).click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_cache)
    monkeypatch.setattr(
        project_cache_gui,
        "get_existing_directory",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    first = prompt.request("first bin", artifact)
    second = prompt.request("second bin", artifact)

    assert first is not None and first.exists()
    assert second is not None and second.exists()
    assert first.parent == second.parent
    assert len(prompts) == 1
    first.unlink()
    second.unlink()
    prompt.cleanup()


def test_batch_cache_dialog_cancels_before_rebinning_by_default(monkeypatch):
    from PySide6 import QtWidgets

    from nfit.project_cache_gui import confirm_rebin_cache_preflight

    explorer = NfitProjectExplorer(NfitProject([DataGroup("cache")]))
    seen = []

    def choose_cancel(message):
        assert message.defaultButton().text() == "Cancel"
        assert all(button.toolTip() for button in message.buttons())
        seen.append(message.text())
        next(button for button in message.buttons() if button.text() == "Cancel").click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_cancel)
    assert not confirm_rebin_cache_preflight(
        explorer.window,
        operation="Opening the data viewer",
        result_count=3,
        added_bytes=700 * 1024**2,
        current_bytes=200 * 1024**2,
        projected_bytes=900 * 1024**2,
        limit_bytes=1024**3,
    )
    assert "download more RAM" in seen[0]
    assert "Opening the data viewer" in seen[0]


def test_batch_cache_dialog_can_choose_disk_before_start(monkeypatch):
    from PySide6 import QtWidgets

    from nfit.project_cache_gui import confirm_rebin_cache_preflight

    explorer = NfitProjectExplorer(NfitProject([DataGroup("cache")]))
    choices = []

    def choose_cache(message):
        next(
            button
            for button in message.buttons()
            if button.text().startswith("Choose disk cache")
        ).click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_cache)
    assert confirm_rebin_cache_preflight(
        explorer.window,
        operation="Running the fit",
        result_count=2,
        added_bytes=800 * 1024**2,
        current_bytes=200 * 1024**2,
        projected_bytes=1000 * 1024**2,
        limit_bytes=1024**3,
        choose_disk_cache=lambda: choices.append(True) or True,
    )
    assert choices == [True]


def test_full_compressed_cache_dialog_uses_active_modal_parent(monkeypatch):
    from PySide6 import QtCore, QtWidgets

    from nfit.project_cache_gui import CompressedCachePrompt
    from nfit.rebin_cache import CompressedBinning
    from tests.project_gui_test_support import _tiny_mdhisto_data

    explorer = NfitProjectExplorer(NfitProject([DataGroup("cache")]))
    progress = QtWidgets.QDialog(explorer.window)
    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress.show()
    QtWidgets.QApplication.processEvents()
    prompt = CompressedCachePrompt(explorer.window)
    artifact = CompressedBinning.from_data(_tiny_mdhisto_data(1.0), max_bytes=10_000)

    def choose_discard(message):
        assert message.parentWidget() is progress
        next(
            button for button in message.buttons() if "Discard" in button.text()
        ).click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_discard)
    prompt.request("old bin", artifact)
    progress.close()


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


@pytest.mark.parametrize("powder", [False, True])
def test_native_mdevent_applies_run_calibration_scales_and_fit_weights(tmp_path, powder):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    group.datasets[0].scale_factor = 2.0
    group.datasets[0].fit_weight = 2.0
    reducer = bin_mdevent_powder_group if powder else bin_mdevent_group
    bounds = (
        dict(lower=[0.0, -1.0], upper=[1.0, 1.0], num_bins=[1, 1])
        if powder
        else dict(lower=[-1.0] * 4, upper=[1.0] * 4, num_bins=[1] * 4)
    )

    result = reducer(group, **bounds)

    denominator = 8.0
    np.testing.assert_allclose(result.signal, 5.0 / denominator)
    np.testing.assert_allclose(result.errors, np.sqrt(17.0) / denominator)


def test_mdevent_coordinate_bounds_use_physical_q_for_powder_and_ub_for_hkl(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        events = handle["MDEventWorkspace/event_data/event_data"]
        values = np.asarray(events)
        values[:, 5] = [2.0, 4.0]
        events[...] = values
    group = mdevent_dataset_group(source)

    powder = mdevent.mdevent_coordinate_bounds(group, powder=True)
    np.testing.assert_allclose(powder, [(2.0, 4.0), (0.0, 0.0)])
    hkl = mdevent.mdevent_coordinate_bounds(group)
    np.testing.assert_allclose(hkl[0], (2.0, 4.0))

    # Changing UB must invalidate the cached HKL result, while powder remains
    # in physical inverse-angstrom coordinates.
    group.metadata["mdevent"]["ub_matrix"] = (
        2.0 * np.asarray(group.metadata["mdevent"]["ub_matrix"])
    ).tolist()
    changed = mdevent.mdevent_coordinate_bounds(group)
    np.testing.assert_allclose(changed[0], (1.0, 2.0))
    np.testing.assert_allclose(
        mdevent.mdevent_coordinate_bounds(group, powder=True), powder
    )


def test_lab_frame_mdevents_reject_hkle_paths_but_allow_powder(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    for dimension in group.metadata["mdevent"]["dimensions"][:3]:
        dimension["frame"] = "QLab"
        dimension["name"] = dimension["name"].replace("sample", "lab")

    message = "requires QSample momentum dimensions"
    with pytest.raises(ValueError, match=message):
        bin_mdevent_group(
            group, lower=[-1.0] * 4, upper=[1.0] * 4, num_bins=[1] * 4
        )
    with pytest.raises(ValueError, match=message):
        mdevent.mdevent_coordinate_bounds(group)
    assert mdevent.mdevent_coordinate_bounds(group, powder=True) == [
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        workspace = handle["MDEventWorkspace"]
        for index in range(3):
            workspace.attrs[f"dimension{index}"] = str(
                workspace.attrs[f"dimension{index}"]
            ).replace("QSample", "QLab").replace("Q_sample", "Q_lab")
    with pytest.raises(ValueError, match=message):
        load_mdevent_run_points(group.datasets[0])


def test_mdevent_coordinate_bounds_honor_detector_mask_and_report_empty_batches(tmp_path):
    source = tmp_path / "events.nxs"
    mask_path = tmp_path / "mask.nxs"
    _write_mdevent(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "r+") as handle:
        events = handle["MDEventWorkspace/event_data/event_data"]
        values = np.asarray(events)
        values[:, 4] = [10, 11]
        values[:, 5] = [1.0, 5.0]
        events[...] = values
    with h5py.File(mask_path, "w") as handle:
        entry = handle.create_group("mantid_workspace_1")
        detector = entry.create_group("instrument/detector")
        detector.create_dataset("detector_list", data=[10, 11])
        workspace = entry.create_group("workspace")
        workspace.create_dataset("values", data=[[1.0], [0.0]])
        workspace.create_dataset("errors", data=[[0.0], [0.0]])
    group = mdevent_dataset_group(source, mask_path=mask_path)
    bounds = mdevent.mdevent_coordinate_bounds(group, powder=True)
    np.testing.assert_allclose(bounds[0], (1.0, 1.0))

    with h5py.File(source, "r+") as handle:
        events = handle["MDEventWorkspace/event_data/event_data"]
        values = np.asarray(events)
        values[:, 2] = 99
        events[...] = values
    progress = []
    with pytest.raises(ValueError, match="no finite coordinates"):
        mdevent.mdevent_coordinate_bounds(
            group, max_batch_bytes=216, progress_callback=progress.append
        )
    assert [item["iteration"] for item in progress] == [1, 2]
    assert all(item["total"] == 2 for item in progress)


def test_mdevent_composite_resolves_auto_bounds_but_skips_scan_for_explicit_limits(
    monkeypatch, tmp_path
):
    import nfit.project_composites as composites

    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    subgroup = mdevent_dataset_group(source)
    root = DataGroup("Workspace", subgroups=[subgroup])
    config = data_group_composite_config(_composite_scope(root, subgroup))
    config.update(enabled=True, minimum_coverage=0)
    names = ("H", "K", "L", "DeltaE")
    config["axes"] = [
        {
            "name": name,
            "vector": np.eye(4)[index].tolist(),
            "mode": "step",
            "lower": 0.0,
            "upper": 0.0,
            "step_size": 0.5,
            "num_bins": 2,
            "auto_lower": True,
            "auto_upper": True,
            "auto_lower_value": 0.0,
            "auto_upper_value": 0.0,
        }
        for index, name in enumerate(names)
    ]
    calls = []
    monkeypatch.setattr(
        composites,
        "mdevent_coordinate_bounds",
        lambda *_args, **_kwargs: calls.append("scan") or [(0.2, 0.7)] * 4,
    )
    reduced = []

    def fake_reduce(_group, *, lower, upper, num_bins, **_kwargs):
        reduced.append((tuple(lower), tuple(upper)))
        shape = tuple(num_bins)
        axes = tuple(
            MDHistoAxis(
                name,
                np.linspace(lo, hi, count + 1),
                "meV" if name == "DeltaE" else "r.l.u.",
                "energy" if name == "DeltaE" else "momentum",
            )
            for name, lo, hi, count in zip(
                names, lower, upper, num_bins, strict=True
            )
        )
        return MDHistoData(
            axes, np.ones(shape), np.ones(shape), np.zeros(shape, dtype=bool),
            np.ones(shape), metadata={"signal_semantics": "density"},
        )

    monkeypatch.setattr(composites, "bin_mdevent_group", fake_reduce)
    composites.composite_dataset_data(root, node=subgroup, apply_spectral_channels=False)
    assert calls == ["scan"]
    np.testing.assert_allclose(reduced[-1][0], [0.0] * 4)
    np.testing.assert_allclose(reduced[-1][1], [0.5] * 4)

    for axis in config["axes"]:
        axis.update(
            lower=-2.0, upper=2.0, auto_lower=False, auto_upper=False
        )
    calls.clear()
    composites.composite_dataset_data(root, node=subgroup, apply_spectral_channels=False)
    assert calls == []
    np.testing.assert_allclose(reduced[-1][0], [-2.0] * 4)
    np.testing.assert_allclose(reduced[-1][1], [2.0] * 4)


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


@pytest.mark.parametrize("powder", [True, False])
def test_mdevent_masks_exclude_run_normalization_and_honor_additive_order(tmp_path, powder):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    # Both runs contribute one event, with exposures 2 and 4 respectively.
    # Removing the second must leave I=1/2, not 1/6 or 1/3.
    group.datasets[1].masks.append(MaskSpec("exclude", "energy_q_range", {"energy": [-1, 1]}))
    reducer = bin_mdevent_powder_group if powder else bin_mdevent_group
    bounds = dict(lower=[0, -1], upper=[1, 1], num_bins=[1, 1]) if powder else dict(
        lower=[-1] * 4, upper=[1] * 4, num_bins=[1] * 4
    )
    data = reducer(group, **bounds)
    np.testing.assert_allclose(data.signal, 0.5)
    np.testing.assert_allclose(data.errors, 0.5)
    np.testing.assert_allclose(data.metadata["normalization_denominator"], 2)
    np.testing.assert_allclose(data.num_events, 1)
    assert not data.mask.item()
    np.testing.assert_allclose(reducer(group, include_source_masks=False, **bounds).signal, 1 / 3)

    group.datasets[1].masks[0].parameters = {"energy": [8, 9]}
    combined = reducer(group, minimum_samples=2, **bounds)
    assert not combined.mask.item()
    np.testing.assert_allclose(combined.signal, 1 / 3)
    assert combined.metadata["rebin"]["minimum_samples"] == 2
    group.datasets[1].masks[0].parameters = {"energy": [-1, 1]}

    group.masks.append(MaskSpec("all", "energy_q_range", {"energy": [-1, 1]}))
    assert reducer(group, **bounds).mask.item()
    group.datasets[0].masks.append(
        MaskSpec("restore first", "energy_q_range", {"energy": [-1, 1]}, additive=True)
    )
    np.testing.assert_allclose(reducer(group, **bounds).signal, 0.5)


@pytest.mark.parametrize("owner", ["ancestor", "group", "dataset"])
def test_live_mdevent_background_applies_masks_and_keeps_target_valid(tmp_path, owner):
    from nfit.project_composites import composite_dataset_data

    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    background = mdevent_dataset_group(source)
    root = DataGroup("Workspace", subgroups=[background])
    config = data_group_composite_config(_composite_scope(root, background))
    config.update(enabled=True, coordinate_mode="powder", minimum_coverage=0)
    config["axes"] = [
        dict(name="|Q|", lower=0, upper=1, num_bins=2, bin_edges=[0, 0.5, 1], mode="edges"),
        dict(name="DeltaE", lower=-1, upper=1, num_bins=2, bin_edges=[-1, 0, 1], mode="edges"),
    ]
    mask = MaskSpec("exclude all", "energy_q_range", {"energy": [-2, 2]})
    owners = {"ancestor": [root], "group": [background], "dataset": background.datasets}[owner]
    for item in owners:
        item.masks.append(mask)
    masked = composite_dataset_data(root, node=background, apply_spectral_channels=False)
    assert np.all(masked.metadata["nfit_mask"])
    assert np.all(masked.mask)
    target = masked.with_updates(
        signal=np.full(masked.shape, 7.0), errors=np.ones(masked.shape),
        mask=np.zeros(masked.shape, dtype=bool), num_events=np.ones(masked.shape), metadata={},
    )
    # The actual live-source resolver must preserve the mask's provenance.
    sample = mdevent_dataset_group(source, name="sample")
    sample.backgrounds.append(BackgroundSpec("window", source_group=background, source_group_id=background.id))
    root.subgroups.append(sample)
    result = _apply_composite_backgrounds(_composite_scope(root, sample), target)
    np.testing.assert_allclose(result.signal, target.signal)
    np.testing.assert_allclose(result.errors, target.errors)
    assert not np.any(result.mask)


def test_mdevent_powder_background_projects_through_sample_trajectories(
    monkeypatch, tmp_path
):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    target = bin_mdevent_group(
        group,
        lower=[-1.0, -1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0, 1.0],
        num_bins=[1, 1, 1, 1],
    )
    background = MDHistoData(
        axes=(
            MDHistoAxis("|Q|", np.array([-0.5, 0.5, 1.5]), "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", np.array([-2.0, 0.0, 2.0]), "meV", "energy"),
        ),
        signal=np.array([[0.0, 0.0], [1.0, 1.0]]),
        errors=np.full((2, 2), 0.2),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        metadata={"signal_semantics": "density"},
    )

    progress = []
    projected = project_powder_background_mdevent(
        group,
        background,
        target,
        progress_callback=progress.append,
    )

    # The target voxel is centered at Q=0, where center interpolation would
    # return zero. Trajectory projection instead averages the finite |Q| path.
    assert 0.0 < projected.signal.item() < 1.0
    assert np.sqrt(0.5) * 0.2 <= projected.errors.item() <= 0.2
    assert not projected.mask.item()
    assert projected.metadata["background_projection"]["mode"] == "sample_trajectories"
    projection_progress = [
        event
        for event in progress
        if event["stage"] == "mdevent_background_projection"
    ]
    assert projection_progress[0]["iteration"] == 0
    assert projection_progress[-1]["iteration"] == projection_progress[-1]["total"]
    np.testing.assert_allclose(
        projected.auxiliary_channels["background_projection_coverage"].values,
        1.0,
    )

    monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", None)
    fallback = project_powder_background_mdevent(group, background, target)
    np.testing.assert_allclose(fallback.signal, projected.signal)
    np.testing.assert_allclose(fallback.errors, projected.errors)
    np.testing.assert_allclose(
        fallback.metadata["normalization_denominator"],
        projected.metadata["normalization_denominator"],
    )

    excluded = background.with_updates(
        mask=np.ones(background.shape, dtype=bool),
        metadata={**background.metadata, "nfit_mask": np.ones(background.shape, dtype=bool)},
    )
    zero = project_powder_background_mdevent(group, excluded, target)
    np.testing.assert_allclose(zero.signal, 0)
    np.testing.assert_allclose(zero.errors, 0)
    assert not zero.mask.item()
    missing = excluded.with_updates(metadata=background.metadata)
    assert project_powder_background_mdevent(group, missing, target).mask.item()

    background_entry = DatasetEntry(
        "Powder background",
        background,
        kind="mdhisto",
        data_type="powder_inelastic",
    )
    group.backgrounds.append(
        BackgroundSpec(
            "Projected powder",
            background_entry.id,
            projection="sample_trajectories",
            source_entry=background_entry,
        )
    )
    root = DataGroup("Workspace", datasets=[background_entry], subgroups=[group])
    corrected = _apply_composite_backgrounds(_composite_scope(root, group), target)
    np.testing.assert_allclose(corrected.signal, target.signal - projected.signal)
    assert corrected.metadata["background_subtractions"][0]["projection"]["mode"] == (
        "sample_trajectories"
    )


def test_native_mdevent_powder_binning_reports_trajectory_progress(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    progress = []

    bin_mdevent_powder_group(
        group,
        lower=[0.0, -1.0],
        upper=[1.0, 1.0],
        num_bins=[1, 1],
        progress_callback=progress.append,
    )

    trajectory_updates = [
        event
        for event in progress
        if "powder detector trajectories" in event.get("message", "")
    ]
    assert trajectory_updates
    assert trajectory_updates[-1]["iteration"] == trajectory_updates[-1]["total"]


def test_native_mdevent_powder_binning_stops_at_trajectory_checkpoint(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)

    def cancel_at_trajectory_start(event):
        if "powder detector trajectories" in event.get("message", ""):
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        bin_mdevent_powder_group(
            group,
            lower=[0.0, -1.0],
            upper=[1.0, 1.0],
            num_bins=[1, 1],
            progress_callback=cancel_at_trajectory_start,
        )


def test_native_mdevent_covered_zero_bins_are_finite_measured_zeros(
    monkeypatch, tmp_path
):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    group = mdevent_dataset_group(source)
    container_inputs = {}

    def capture_container_inputs(**kwargs):
        container_inputs.update(kwargs)
        return MDHistoData(**kwargs)

    monkeypatch.setattr(mdevent, "MDHistoData", capture_container_inputs)

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
    denominator = result.metadata["normalization_denominator"]
    assert denominator is result.auxiliary_channels["normalization_denominator"].values
    assert not denominator.flags.writeable
    assert result.signal is container_inputs["signal"]
    assert result.errors is container_inputs["errors"]
    assert result.mask is container_inputs["mask"]
    assert result.num_events is container_inputs["num_events"]
    fit_points = _point_data_from_mdhisto_view(result)
    assert np.count_nonzero(fit_points.valid_mask()) == 2


def test_native_mdevent_powder_shares_immutable_normalization_storage(tmp_path):
    source = tmp_path / "events.nxs"
    _write_mdevent(source)
    result = bin_mdevent_powder_group(
        mdevent_dataset_group(source),
        lower=[0.0, -1.0],
        upper=[1.0, 1.0],
        num_bins=[1, 2],
    )

    denominator = result.metadata["normalization_denominator"]
    assert denominator is result.auxiliary_channels["normalization_denominator"].values
    assert not denominator.flags.writeable


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


def test_persistent_trajectory_accumulator_matches_per_batch_wrapper():
    kernels = mdevent._MDEVENT_NUMBA
    if kernels is None:
        pytest.skip("Numba MDEvent normalization is unavailable")
    edges = (
        np.linspace(-3.0, 3.0, 5),
        np.linspace(-3.0, 3.0, 4),
        np.linspace(-3.0, 3.0, 4),
        np.linspace(-2.0, 8.0, 6),
    )
    shape = np.asarray([edge.size - 1 for edge in edges], dtype=np.int64)

    def batch(theta, phi, charge, inverse):
        return (
            np.asarray(theta),
            np.asarray(phi),
            np.ones(len(theta)),
            np.asarray([inverse]),
            np.asarray([12.0]),
            np.asarray([[-2.0, 8.0]]),
            np.asarray([charge]),
            *edges,
            shape,
        )

    batches = (
        batch([0.3, 0.8], [0.1, 1.2], 1.0, np.eye(3)),
        batch([0.4, 1.0, 1.4], [-0.2, 0.6, 2.0], 2.0, np.diag([1.0, -1.0, 1.0])),
    )
    expected = sum(
        (kernels.run_trajectory_normalization(*args, workers=2) for args in batches),
        start=np.zeros(int(np.prod(shape))),
    )
    accumulator = kernels.trajectory_normalization_accumulator(
        *edges, shape, workers=2
    )
    for args in batches:
        accumulator.accumulate(*args)
    np.testing.assert_allclose(accumulator.result(), expected)
    assert not accumulator.partial.flags.writeable
    with pytest.raises(RuntimeError, match="finalized"):
        accumulator.accumulate(*batches[0])

    single = kernels.trajectory_normalization_accumulator(*edges, shape, workers=1)
    single.accumulate(*batches[0])
    single_result = single.result()
    assert np.shares_memory(single_result, single.partial)
    assert not single_result.flags.writeable
    assert not single_result.base.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        single_result[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        single_result.base[0, 0] = 0.0
    with pytest.raises(RuntimeError, match="finalized"):
        single.accumulate(*batches[0])

    eager = kernels.trajectory_normalization_accumulator(
        *edges, shape, workers=2, eager=True
    )
    for args in batches:
        eager.accumulate(*args)
    np.testing.assert_array_equal(eager.result(), accumulator.result())
    assert not eager.partial.flags.writeable
    with pytest.raises(RuntimeError, match="finalized"):
        eager.accumulate(*batches[0])

    eager_single = kernels.trajectory_normalization_accumulator(
        *edges, shape, workers=1, eager=True
    )
    eager_single.accumulate(*batches[0])
    eager_single_result = eager_single.result()
    np.testing.assert_array_equal(eager_single_result, single_result)
    assert np.shares_memory(eager_single_result, eager_single.partial)
    assert not eager_single_result.flags.writeable


def test_eager_trajectory_allocation_restores_numba_threads_on_failure(monkeypatch):
    kernels = mdevent._MDEVENT_NUMBA
    if kernels is None:
        pytest.skip("Numba MDEvent normalization is unavailable")
    active = [7]

    monkeypatch.setattr(kernels, "get_num_threads", lambda: active[0])
    monkeypatch.setattr(kernels, "set_num_threads", lambda value: active.__setitem__(0, value))

    def fail_allocation(*args):
        assert active[0] == 3
        raise MemoryError("allocation failed")

    monkeypatch.setattr(kernels, "_eager_trajectory_partial", fail_allocation)
    edges = tuple(np.linspace(-1.0, 1.0, 3) for _ in range(4))
    with pytest.raises(MemoryError, match="allocation failed"):
        kernels.trajectory_normalization_accumulator(
            *edges, np.asarray([2, 2, 2, 2]), workers=3, eager=True
        )
    assert active[0] == 7


def test_eager_trajectory_allocation_obeys_workspace_boundary(monkeypatch):
    monkeypatch.setattr(
        "nfit.performance.scientific_memory_limit_bytes", lambda: 16 * 1024**3
    )
    limit = 2 * 1024**3
    assert mdevent._trajectory_eager_partial(limit // (4 * 8), 4)
    assert not mdevent._trajectory_eager_partial(limit // (4 * 8) + 1, 4)

    monkeypatch.setattr(
        "nfit.performance.scientific_memory_limit_bytes", lambda: 128 * 1024**3
    )
    limit = mdevent.MDEVENT_EAGER_PARTIAL_MAX_BYTES
    assert mdevent._trajectory_eager_partial(limit // (8 * 8), 8)
    assert not mdevent._trajectory_eager_partial(limit // (8 * 8) + 1, 8)


def test_persistent_trajectory_accumulator_stops_at_batch_checkpoint(monkeypatch):
    calls = []
    allocation_modes = []

    class Accumulator:
        def accumulate(self, *args):
            calls.append(args)

        def result(self):
            raise AssertionError("cancelled accumulation must not be reduced")

    class Kernels:
        @staticmethod
        def trajectory_normalization_accumulator(*args, workers, eager):
            allocation_modes.append(eager)
            return Accumulator()

    detector_payloads = [
        (np.asarray([1]), np.asarray([0.4]), np.asarray([0.2]), np.asarray([1.0])),
        (np.asarray([2]), np.asarray([0.7]), np.asarray([0.5]), np.asarray([1.0])),
    ]
    run_payloads = [
        (np.eye(3), 12.0, np.asarray([-2.0, 8.0]), 1.0, 0),
        (np.eye(3), 12.0, np.asarray([-2.0, 8.0]), 1.0, 0),
        (np.eye(3), 12.0, np.asarray([-2.0, 8.0]), 1.0, 0),
    ]
    monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", Kernels)
    monkeypatch.setattr(
        mdevent, "_trajectory_payloads", lambda *args, **kwargs: (detector_payloads, run_payloads)
    )
    monkeypatch.setattr(
        mdevent, "_trajectory_worker_count", lambda output_size, **kwargs: 1
    )
    monkeypatch.setattr(mdevent, "_trajectory_eager_partial", lambda *args: True)
    completed = []

    def cancel_after_second_batch(event):
        if event["stage"] == "mdevent_normalization" and event["iteration"]:
            completed.append(event["iteration"])
            if len(completed) == 2:
                raise RuntimeError("cancelled")

    edges = tuple(np.linspace(-2.0, 2.0, 3) for _ in range(4))
    with pytest.raises(RuntimeError, match="cancelled"):
        mdevent._trajectory_normalization_from_payloads(
            detector_payloads,
            run_payloads,
            edges,
            (2, 2, 2, 2),
            progress_callback=cancel_after_second_batch,
            max_batch_tasks=1,
        )
    assert completed == [1, 2]
    assert len(calls) == 2
    assert allocation_modes == [True]


def test_trajectory_workers_honor_cpu_ceiling_and_available_memory(monkeypatch):
    monkeypatch.setattr(mdevent._parallel, "num_threads", lambda: 16)
    monkeypatch.setattr(
        "nfit.performance.transient_rebin_memory_limit_bytes",
        lambda available: available // 4,
    )
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


def test_mdevent_available_memory_uses_shared_platform_probe(monkeypatch):
    monkeypatch.setattr(
        "nfit.performance.available_memory_bytes",
        lambda: 64 * 1024**3,
    )

    assert mdevent._available_memory_bytes() == 64 * 1024**3


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


def test_trajectory_workers_reserve_live_arrays(monkeypatch):
    monkeypatch.setattr(mdevent._parallel, "num_threads", lambda: 64)
    monkeypatch.setattr(mdevent, "_available_memory_bytes", lambda: 512 * 1024**3)
    monkeypatch.setattr(
        "nfit.performance.transient_rebin_memory_limit_bytes", lambda available: 128000 * 1024**2
    )
    bins = 464_011_821
    assert mdevent._trajectory_worker_count(bins, reserved_bytes=4 * bins * 8) == 32
    assert mdevent._trajectory_worker_count(bins, reserved_bytes=128000 * 1024**2) == 1


def test_mdevent_peak_estimate_includes_private_normalization_grids(monkeypatch):
    monkeypatch.setattr(mdevent, "_MDEVENT_NUMBA", object())
    monkeypatch.setattr(mdevent, "_trajectory_worker_count", lambda *args, **kwargs: 32)
    bins = 464_011_821
    estimate = estimate_mdevent_peak_memory([bins], max_batch_bytes=1)
    assert estimate == mdevent.MDEVENT_FIXED_MEMORY_BYTES + 36 * bins * 8 + 1
