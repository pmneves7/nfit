import copy

import h5py
import numpy as np
import pytest

import nfit
from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from nfit.metadata_dimensions import (
    metadata_dimension_coordinates,
    metadata_dimension_indices,
)
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
from nfit.plotting import MDHistoSliceViewer
from nfit.project_gui import (
    PLOT_SOURCE_COMPOSITE_KEY,
    _apply_spectral_channel_view,
    _composite_cache_signature,
    _point_data_from_mdhisto_view,
    _saved_plot_source_views,
    data_group_composite_config,
)


def points(temperature, signal, *, sigma=None, name="scan"):
    signal = np.asarray(signal, dtype=float)
    data = nfit.PointData4D(
        H=np.zeros(signal.size),
        K=np.zeros(signal.size),
        L=np.zeros(signal.size),
        E=np.ones(signal.size),
        intensity=signal,
        sigma=np.ones(signal.size) if sigma is None else sigma,
        temperature=temperature,
    )
    return DatasetEntry(name, data, parameters={"temperature": float(np.mean(temperature))})


def group(entries):
    result = DataGroup("series", datasets=entries)
    config = data_group_composite_config(result)
    config.update(enabled=True, fractional=True, minimum_coverage=0, mean_weighting="uniform")
    for i, axis in enumerate(config["axes"]):
        axis.update(
            bin_edges=[-0.5, 0.5] if i < 3 else [0.5, 1.5],
            mode="edges",
            auto_lower=False,
            auto_upper=False,
            auto_step_size=False,
        )
    return result


def temperature(**kwargs):
    return nfit.MetadataDimension("Temperature", "parameters/temperature", "K", **kwargs)


@pytest.mark.parametrize("metadata_axes", [False, True])
@pytest.mark.parametrize("source_rebinned", [False, True])
def test_raw_group_background_automatically_follows_sample_grid_and_preserves_source(
    metadata_axes, source_rebinned
):
    from nfit.pipeline import BackgroundSpec, MaskSpec

    reference = points(50, [2, 100], sigma=[3, 1], name="reference")
    reference.replace_data(reference.data.with_updates(mask=[True, False]))
    g = group([points(5, [10], name="cold"), points(10, [20], name="warm"), reference])
    if source_rebinned:
        source_config = nfit.dataset_rebin_config(reference)
        source_config["enabled"] = True
        source_edges = ([-1, 0, 1], [-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5])
        for axis, bin_edges in zip(source_config["axes"], source_edges, strict=True):
            axis.update(
                mode="edges",
                bin_edges=bin_edges,
                auto_lower=False,
                auto_upper=False,
                auto_step_size=False,
            )
    else:
        reference.parameters["rebin"] = {"enabled": False}
    original = reference.data
    recipe = copy.deepcopy(reference.parameters["rebin"])
    g.backgrounds.append(BackgroundSpec("50 K", reference.id, scale=0.5, source_entry=reference))
    if metadata_axes:
        nfit.set_metadata_dimensions(g, [temperature()])
    result = nfit.composite_dataset_data(g)
    np.testing.assert_allclose(result.signal.ravel(), [9, 19, 1] if metadata_axes else [14])
    np.testing.assert_allclose(result.errors.ravel()**2, [3.25, 3.25, 11.25] if metadata_axes else [2.75])
    assert reference.data is original
    assert reference.parameters["rebin"] == recipe
    if metadata_axes:
        np.testing.assert_array_equal(result.axes[-1].centers, [5, 10, 50])
    # Overrides used by saved plots and scripts must also control the background.
    config = copy.deepcopy(data_group_composite_config(g))
    config["axes"][0].update(
        name="H new", bin_edges=[-1, 0.25, 1], mode="edges"
    )
    changed = nfit.composite_dataset_data(g, config_override=config)
    assert changed.shape[0] == 2
    np.testing.assert_allclose(changed.signal[0].ravel(), result.signal.ravel())
    np.testing.assert_allclose(changed.signal[1].ravel(), result.signal.ravel())
    signature = _composite_cache_signature(g)
    reference.masks.append(MaskSpec("cut", type="box", parameters={"H": [0, 1]}))
    assert _composite_cache_signature(g) != signature


def test_raw_background_composite_script_round_trip(tmp_path):
    from nfit.pipeline import BackgroundSpec
    from tests.test_macs import _write_macs_nexus

    entries = []
    for i in range(2):
        filename = _write_macs_nexus(tmp_path / f"source{i}.nxs", energy_transfer=1.0)
        entries.append(nfit.dataset_entry_from_path(filename))
        entries[i].parameters["temperature"] = [5, 50][i]
        entries[i].parameters["rebin"] = {"enabled": False}
    g = group(entries)
    for axis in data_group_composite_config(g)["axes"][:3]:
        axis["bin_edges"] = [-10, 10]
    before = nfit.composite_dataset_data(g)
    g.backgrounds.append(BackgroundSpec("50 K", entries[1].id, source_entry=entries[1]))
    nfit.set_metadata_dimensions(g, [temperature()])
    path = tmp_path / "background.nfit"
    nfit.save_project(nfit.NfitProject([g]), path)
    restored = nfit.load_project(path)
    script = nfit.composite_workflow_script(restored, g.name)
    namespace = {"__name__": "test_workflow"}
    exec(compile(script, "background.py", "exec"), namespace)
    output = namespace["run"]()
    np.testing.assert_allclose(output.signal.ravel(), [0, 0], atol=1e-10)
    # Before combines two identical independent runs; subtraction adds their variances.
    np.testing.assert_allclose(output.errors[..., 0], 2 * before.errors)
    np.testing.assert_allclose(output.errors[..., 1], 0)


def test_six_temperatures_are_independent_even_with_fractional_binning(tmp_path):
    targets = [5, 10, 20, 30, 40, 50]
    entries = [points(t + 0.1, [i + 1], name=str(t)) for i, t in enumerate(targets)]
    # A repeated run shares its temperature only, with statistical uncertainties.
    entries.append(points(10.2, [6], sigma=[3]))
    entries[0].scale_factor = 2
    g = group(entries)
    before = [entry.data for entry in entries]
    nfit.set_metadata_dimensions(g, [temperature(centers=targets)])
    data = nfit.composite_dataset_data(g)
    assert data.shape == (1, 1, 1, 1, 6)
    np.testing.assert_array_equal(data.axes[-1].centers, targets)
    np.testing.assert_allclose(data.signal.ravel(), [2, 4, 3, 4, 5, 6])
    np.testing.assert_allclose(data.errors.ravel(), [2, np.sqrt(10) / 2, 1, 1, 1, 1])
    assert not data.mask.any()
    assert all(entry.data is original for entry, original in zip(entries, before, strict=True))
    assert not data.signal.flags.writeable
    view = MDHistoSliceViewer(data, x_dim=4, y_dim=3).slice_arrays()
    np.testing.assert_array_equal(view["x_centers"], targets)
    np.testing.assert_allclose(view["signal"].ravel(), data.signal.ravel())
    fit_points = _point_data_from_mdhisto_view(data)
    np.testing.assert_array_equal(fit_points.temperature, targets)
    np.testing.assert_array_equal(fit_points.E, np.ones(6))
    path = tmp_path / "data.npz"
    write_dataset_artifact(data, path)
    restored = read_dataset_artifact(path)
    np.testing.assert_array_equal(restored.axes[-1].centers, targets)
    np.testing.assert_array_equal(restored.signal, data.signal)


def test_pointwise_coordinates_masks_weights_and_empty_centers():
    entry = points([5, 10, 5, 20], [1, 9, 1000, 25])
    entry.replace_data(entry.data.with_updates(mask=[True, True, False, True]))
    g = group([entry])
    nfit.set_metadata_dimensions(
        g, [nfit.MetadataDimension("T", "temperature", "K", "per_point", [5, 10, 20, 30])]
    )
    data = nfit.composite_dataset_data(g)
    np.testing.assert_allclose(data.signal.ravel()[:3], [1, 9, 25])
    assert data.mask.ravel()[-1]
    assert data.num_events.ravel()[-1] == 0
    assert np.isnan(data.signal.ravel()[-1])


def test_temperature_dependent_spectral_conversion_uses_each_slice():
    from nfit.spectral_channels import default_spectral_channel_config

    g = group([points(5, [1]), points(50, [1])])
    nfit.set_metadata_dimensions(g, [temperature()])
    data = nfit.composite_dataset_data(g)
    config = default_spectral_channel_config()
    config.update(source_unit="mbarn/sr/meV/f.u.", kf_ki_state="removed")
    entry = DatasetEntry(
        "composite",
        data,
        data_type="single_crystal_inelastic",
        parameters={"spectral_channels": config},
    )
    converted = _apply_spectral_channel_view(entry, data)
    for i, t in enumerate([5, 50]):
        single = nfit.composite_dataset_data(group([points(t, [1])]))
        reference = _apply_spectral_channel_view(
            DatasetEntry(
                "single",
                single,
                data_type="single_crystal_inelastic",
                parameters={"temperature": t, "spectral_channels": config},
            ),
            single,
        )
        np.testing.assert_allclose(
            converted.auxiliary_channels["dynamic_susceptibility"].values[..., i],
            reference.auxiliary_channels["dynamic_susceptibility"].values,
        )


def test_saved_plot_metadata_axes_are_independent_of_live_edits():
    from nfit.pipeline import PlotEntry

    g = group([points(5, [1]), points(10, [3])])
    nfit.set_metadata_dimensions(g, [temperature()])
    recipe = {
        "config": copy.deepcopy(data_group_composite_config(g)),
        "metadata_dimensions": copy.deepcopy(g.metadata["metadata_dimensions"]),
    }
    plot = PlotEntry("saved", "slice_2d", settings={PLOT_SOURCE_COMPOSITE_KEY: recipe})
    nfit.set_metadata_dimensions(g, [])
    views, _ = _saved_plot_source_views(g, plot)
    np.testing.assert_array_equal(views[0].axes[-1].centers, [5, 10])
    assert nfit.composite_dataset_data(g).signal.ndim == 4


def test_two_dimensions_leave_missing_combinations_masked():
    a, b = points(5, [1]), points(10, [3])
    a.metadata["angle"], b.metadata["angle"] = 0, 90
    g = group([a, b])
    nfit.set_metadata_dimensions(
        g, [temperature(), nfit.MetadataDimension("Angle", "metadata/angle", "degree")]
    )
    data = nfit.composite_dataset_data(g)
    np.testing.assert_allclose(
        data.signal.reshape(2, 2), [[1, np.nan], [np.nan, 3]], equal_nan=True
    )
    np.testing.assert_array_equal(data.mask.reshape(2, 2), [[False, True], [True, False]])


@pytest.mark.parametrize("values,match", [([7.5], "equidistant"), ([12], "outside")])
def test_assignment_rejects_ambiguous_or_outlying_values(values, match):
    with pytest.raises(ValueError, match=match):
        metadata_dimension_indices(values, [5, 10], 0.5)


@pytest.mark.parametrize("centers", [[5, 5], [10, 5], [np.nan], []])
def test_coordinate_validation(centers):
    with pytest.raises(ValueError, match="strictly increasing"):
        temperature(centers=centers)


def test_hdf_channel_selection_aliases_summary_and_scan_alignment(tmp_path):
    path = tmp_path / "scan.nxs"
    with h5py.File(path, "w") as file:
        logs = file.create_group("entry/DAS_logs/temp")
        values = logs.create_dataset("value", data=[5.1, 10.1])
        values.attrs["units"] = "K"
        logs.create_dataset("average_value", data=[7.6]).attrs["units"] = "K"
        file.create_group("entry/data")["temp"] = logs
        file["entry/data"].create_dataset("scan_temperature", data=[5.1, 10.1]).attrs["units"] = "K"
    entry = points(7.6, np.arange(40.0))
    entry.metadata["source_file"] = str(path)
    entry.replace_data(
        entry.data.with_updates(metadata={"importer": "macs_nexus", "scan_point_shape": [2, 20]})
    )
    assert nfit.metadata_channels(entry)["entry/data/temp/value"] == "K"
    spec = nfit.MetadataDimension("Temperature", "entry>data>temp>average_value", "K")
    np.testing.assert_allclose(metadata_dimension_coordinates(entry, spec), [7.6])
    spec = nfit.MetadataDimension("Temperature", "entry/data/scan_temperature", "K", "per_point")
    np.testing.assert_allclose(
        metadata_dimension_coordinates(entry, spec), np.repeat([5.1, 10.1], 20)
    )
    spec = nfit.MetadataDimension("Temperature", "entry/DAS_logs/temp/value", "K", "per_point")
    with pytest.raises(ValueError, match="alignment adapter"):
        metadata_dimension_coordinates(entry, spec)


def test_bad_metadata_is_not_silently_skipped():
    entry = points(5, [1, 2])
    entry.metadata["log"] = [5, 10, 20]
    with pytest.raises(ValueError, match="not aligned"):
        metadata_dimension_coordinates(
            entry, nfit.MetadataDimension("T", "metadata/log", "K", "per_point")
        )
    with pytest.raises(KeyError):
        metadata_dimension_coordinates(entry, nfit.MetadataDimension("T", "metadata/missing"))
    with pytest.raises(ValueError, match="units"):
        metadata_dimension_coordinates(entry, nfit.MetadataDimension("T", "temperature", "degree"))
    entry.metadata["log"] = [np.nan]
    with pytest.raises(ValueError, match="finite"):
        metadata_dimension_coordinates(entry, nfit.MetadataDimension("T", "metadata/log"))


def test_cache_tracks_metadata_edits_and_pointwise_interior_values():
    entry = points([5, 10, 20], [1, 2, 3])
    entry.metadata["log"] = [5, 10, 20]
    g = group([entry])
    nfit.set_metadata_dimensions(g, [nfit.MetadataDimension("T", "metadata/log", "K", "per_point")])
    before = _composite_cache_signature(g)
    entry.metadata["log"][1] = 15
    assert _composite_cache_signature(g) != before


def test_histogram_sources_keep_a_common_grid_and_missing_regions():
    axes = (nfit.MDHistoAxis("E", np.array([0, 1, 2]), "meV", "energy"),)

    def entry(t, values):
        data = nfit.MDHistoData(axes, np.array(values), np.ones(2), np.zeros(2, bool), np.ones(2))
        return DatasetEntry(str(t), data, parameters={"temperature": t})

    g = DataGroup("hist", datasets=[entry(5, [1, 2]), entry(10, [3, 4])])
    config = data_group_composite_config(g)
    config.update(fractional=False, minimum_coverage=0)
    config["axes"][0].update(bin_edges=[0, 1, 2], auto_lower=False, auto_upper=False)
    nfit.set_metadata_dimensions(g, [temperature()])
    data = nfit.composite_dataset_data(g)
    np.testing.assert_allclose(data.signal, [[1, 3], [2, 4]])


@pytest.mark.parametrize("binning", [None, {"bin_edges": [0, 15, 25]}])
def test_project_and_editable_script_round_trip(tmp_path, binning):
    g = group([points(5, [1]), points(10, [4])])
    for i, entry in enumerate(g.datasets):
        source = tmp_path / f"source_{i}.npz"
        histogram = nfit.composite_dataset_data(group([entry]))
        nfit.save_dataset_file(entry.copy(data=histogram), source, use_view=False)
        g.datasets[i] = nfit.dataset_entry_from_path(source)
    config = copy.deepcopy(data_group_composite_config(g))
    node = DatasetGroup("temperatures", datasets=g.datasets, metadata={"composite": config})
    g.datasets = []
    g.subgroups.append(node)
    nfit.set_metadata_dimensions(node, [temperature(centers=[5, 10, 20], binning=binning)])
    project = nfit.NfitProject([g])
    path = tmp_path / "test.nfit"
    nfit.save_project(project, path)
    restored = nfit.load_project(path)
    script = nfit.composite_workflow_script(restored, g.name, node_id=node.id)
    assert "METADATA_DIMENSIONS" in script and "REBIN_CONFIG" in script
    namespace = {"__name__": "test_workflow"}
    exec(compile(script, "composite.py", "exec"), namespace)
    data = namespace["run"]()
    expected = [1, 4, np.nan] if binning is None else [2.5, np.nan]
    np.testing.assert_allclose(data.signal.ravel(), expected, equal_nan=True)
    np.testing.assert_array_equal(data.axes[-1].centers, [5, 10, 20] if binning is None else [7.5, 20])


def test_metadata_panel_and_dialog_controls_have_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.metadata_dimensions_gui import metadata_dimensions_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    g = group([points(5, [1])])

    class Explorer:
        def _after_group_composite_changed(self, group):
            pass

    panel = metadata_dimensions_panel(Explorer(), g)
    seen = []

    def inspect(dialog):
        for widget in dialog.findChildren(QtWidgets.QWidget):
            if widget.objectName().startswith("metadata_dimension_"):
                assert widget.toolTip(), widget.objectName()
                seen.append(widget.objectName())
        dialog.findChild(QtWidgets.QPushButton, "metadata_dimension_check").click()
        assert (
            "5"
            in dialog.findChild(
                QtWidgets.QPlainTextEdit, "metadata_dimension_preview"
            ).toPlainText()
        )
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        return 1

    monkeypatch.setattr(QtWidgets.QDialog, "exec", inspect)
    for widget in panel.findChildren(QtWidgets.QWidget):
        if widget.objectName().startswith("metadata_dimensions_"):
            assert widget.toolTip()
    panel.findChild(QtWidgets.QPushButton, "metadata_dimensions_add").click()
    assert "metadata_dimension_source" in seen
    assert g.metadata["metadata_dimensions"][0]["name"] == "Temperature"
    panel.close()
    app.processEvents()


@pytest.mark.parametrize(
    "binning",
    [
        {"lower": 5, "upper": 25, "step": 10},
        {"lower": 5, "upper": 25, "num_bins": 3},
        {"bin_edges": [0, 10, 20, 30]},
    ],
)
def test_metadata_rebin_preserves_source_weights_and_whole_bin_assignment(binning):
    g = group(
        [
            points(5, [1, 1, 1]),
            points(8, [9]),
            points(10, [20]),
            points(30, [30]),
            points(40, [1000]),
        ]
    )
    nfit.set_metadata_dimensions(g, [temperature(binning=binning)])
    data = nfit.composite_dataset_data(g)
    np.testing.assert_array_equal(data.axes[-1].values, [0, 10, 20, 30])
    np.testing.assert_array_equal(data.axes[-1].centers, [5, 15, 25])
    # Combine the original samples, not the two run means. Edge 10 goes right;
    # final edge 30 is included and 40 is discarded, despite fractional HKLE.
    np.testing.assert_allclose(data.signal.ravel(), [3, 20, 30])
    np.testing.assert_allclose(data.errors.ravel(), [0.5, 1, 1])
    np.testing.assert_array_equal(_point_data_from_mdhisto_view(data).temperature, [5, 15, 25])


def test_metadata_tolerance_mode_clusters_nearby_values_without_interpolation():
    g = group(
        [
            points(-0.398, [1]),
            points(0.003, [2]),
            points(0.398, [3]),
            points(0.403, [5]),
            points(1.199, [4]),
            points(2.001, [6]),
            points(2.799, [7]),
            points(3.602, [8]),
        ]
    )
    nfit.set_metadata_dimensions(g, [temperature(binning={"tolerance": 0.1})])

    data = nfit.composite_dataset_data(g)

    np.testing.assert_allclose(
        data.axes[-1].centers,
        [-0.398, 0.003, 0.4005, 1.199, 2.001, 2.799, 3.602],
    )
    np.testing.assert_allclose(data.signal.ravel(), [1, 2, 4, 4, 6, 7, 8])


def test_metadata_fractional_mode_uses_the_central_nd_rebinner(monkeypatch):
    import nfit.project_rebinning as project_rebinning

    g = group([points([0.0, 0.5, 1.0], [0.0, 10.0, 20.0])])
    nfit.set_metadata_dimensions(
        g,
        [
            nfit.MetadataDimension(
                "Temperature",
                "temperature",
                "K",
                "per_point",
                binning={
                    "lower": 0.0,
                    "upper": 1.0,
                    "step": 1.0,
                    "fractional": True,
                },
            )
        ],
    )
    calls = []
    original = project_rebinning.rebin_nd

    def recording_rebin(*args, **kwargs):
        calls.append(np.asarray(args[1]).shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(project_rebinning, "rebin_nd", recording_rebin)
    data = nfit.composite_dataset_data(g)

    assert calls == [5]
    np.testing.assert_allclose(data.signal.ravel(), [10 / 3, 50 / 3])
    assert data.metadata["rebin"]["fractional_axes"] == [True] * 5


def test_pointwise_rebin_uses_nominal_centers_before_grouping_and_preserves_empty_bins():
    g = group([points([4.9, 10.1, 20.1], [2, 4, 8])])
    spec = nfit.MetadataDimension(
        "T", "temperature", "K", "per_point", [5, 10, 20], binning={"bin_edges": [5, 15, 25, 35]}
    )
    nfit.set_metadata_dimensions(g, [spec])
    data = nfit.composite_dataset_data(g)
    np.testing.assert_allclose(data.signal.ravel(), [3, 8, np.nan], equal_nan=True)
    assert data.mask.ravel()[-1]
    nfit.set_metadata_dimensions(g, [temperature(binning={"lower": 0, "upper": 30, "num_bins": 1})])
    assert nfit.composite_dataset_data(g).signal.item() == pytest.approx(14 / 3)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"step": 0, "lower": 0, "upper": 10},
        {"num_bins": 1.5, "lower": 0, "upper": 10},
        {"step": 1, "num_bins": 2, "lower": 0, "upper": 10},
        {"step": 1, "lower": 10, "upper": 0},
        {"bin_edges": [0, 0]},
        {"bin_edges": [0, np.nan]},
        {"tolerance": 0},
    ],
)
def test_metadata_binning_rejects_invalid_grids(config):
    with pytest.raises(ValueError, match="metadata"):
        temperature(binning=config)


def temperature_volume():
    from nfit.metadata_dimensions import discrete_metadata_axis

    temps = [1.54, 5.24, 10.34, 20.12, 30.06, 39.28]
    axes = (
        nfit.MDHistoAxis("H", [-0.5, 0.5, 1.5], "rlu", "h"),
        nfit.MDHistoAxis("K", [-0.5, 0.5, 1.5], "rlu", "k"),
        nfit.MDHistoAxis("L", [-0.5, 0.5], "rlu", "l"),
        nfit.MDHistoAxis("DeltaE", [0.5, 1.5], "meV", "energy_transfer"),
        discrete_metadata_axis(temperature(), temps),
    )
    values = np.broadcast_to(np.arange(1.0, 7.0), (2, 2, 1, 1, 6)).copy()
    return nfit.MDHistoData(
        axes, values, np.ones_like(values), np.zeros_like(values, bool), np.ones_like(values)
    )


@pytest.mark.parametrize("restricted", [False, True])
def test_refresh_expands_full_metadata_tile_range_and_keeps_zero_panel(monkeypatch, restricted):
    from dataclasses import replace

    from nfit.plotting import prepare_mdhisto_tiled_slices

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    data = temperature_volume()
    signal = data.signal.copy()
    signal[..., -1] = 0
    data = data.with_updates(signal=signal)
    old_axis = replace(data.axes[-1], values=data.axes[-1].values[:-1], metadata={
        **data.axes[-1].metadata, "discrete_centers": data.axes[-1].centers[:-1].tolist(),
    })
    old = data.with_updates(axes=(*data.axes[:-1], old_axis), signal=data.signal[..., :-1],
        errors=data.errors[..., :-1], mask=data.mask[..., :-1], num_events=data.num_events[..., :-1])
    viewer = nfit.QtMDHistoSliceViewer(old, x_dim=0, y_dim=1)
    viewer.tile_dim = 4
    viewer.tile_range = (5.24, 20.12) if restricted else (1.54, 30.06)
    viewer.replace_datasets(data)
    assert viewer.tile_range == ((5.24, 20.12) if restricted else (1.54, 39.28))
    panels = prepare_mdhisto_tiled_slices(data, x_dim=0, y_dim=1, tile_dim=4)
    assert len(panels) == 6
    np.testing.assert_allclose(panels[-1].values, 0)
    viewer.window.close()


def test_same_named_composites_are_distinguished_by_collection_path():
    from nfit.project_gui import _composite_dataset_name, _composite_scope

    a, b = DatasetGroup("MACS SPEC"), DatasetGroup("MACS SPEC")
    root = DataGroup("workspace", subgroups=[DatasetGroup("series1", subgroups=[a]), DatasetGroup("series2", subgroups=[b])])
    assert _composite_dataset_name(_composite_scope(root, a)) == "series1/MACS SPEC Composite"
    assert _composite_dataset_name(_composite_scope(root, b)) == "series2/MACS SPEC Composite"


def test_reference_slice_is_visible_zero_and_background_toggle_forces_manual_viewer_refresh(monkeypatch):
    from PySide6 import QtWidgets

    from nfit.pipeline import BackgroundSpec
    from nfit.project_gui import _cached_composite_dataset_data

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    def fail_dialog(*args):
        pytest.fail(str(args))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", fail_dialog)
    reference = points(50, [2], name="50 K")
    g = group([points(5, [10]), reference])
    nfit.set_metadata_dimensions(g, [temperature()])
    background = BackgroundSpec("reference", reference.id, source_entry=reference)
    g.backgrounds.append(background)
    data_group_composite_config(g)["auto_rebin"] = False
    data = _cached_composite_dataset_data(g, force_rebin=True)
    np.testing.assert_allclose(data.signal.ravel(), [8, 0])
    np.testing.assert_allclose(data.errors.ravel(), [np.sqrt(2), 0])
    assert not data.mask.any()
    explorer = nfit.NfitProjectExplorer(nfit.NfitProject([g]))
    monkeypatch.setattr(explorer, "_confirm_save_before_closing_project", lambda: True)
    viewer = explorer.open_slice_viewer(g)
    explorer._update_background(g, g, background, enabled=False)
    np.testing.assert_allclose(viewer.data.signal.ravel(), [10, 2])
    explorer._update_background(g, g, background, enabled=True)
    np.testing.assert_allclose(viewer.data.signal.ravel(), [8, 0])
    assert not viewer.data.mask.any()
    viewer.window.close()
    explorer.window.close()


@pytest.mark.parametrize("typed,expected", [("50", 49.8), ("-5", 1.5), ("20", 20)])
@pytest.mark.parametrize("commit", ["enter", "focus"])
def test_viewer_range_input_accepts_then_clamps_on_commit(monkeypatch, typed, expected, commit):
    from PySide6 import QtCore, QtTest, QtWidgets

    from nfit.qt_slice_controls import _make_float_spinbox

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)
    spin = _make_float_spinbox(1.5, 49.8)
    spin.setValue(10)
    other = QtWidgets.QLineEdit()
    layout.addWidget(spin)
    layout.addWidget(other)
    window.show()
    spin.setFocus()
    app.processEvents()
    spin.selectAll()
    QtTest.QTest.keyClicks(spin, typed)
    assert spin.lineEdit().text() == typed
    assert spin.value() == 10
    if commit == "enter":
        QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Return)
    else:
        other.setFocus()
    app.processEvents()
    assert spin.value() == pytest.approx(expected)
    assert float(spin.cleanText()) == pytest.approx(expected)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_irregular_metadata_tiles_and_saved_plot_use_each_exact_coordinate():
    import matplotlib.pyplot as plt

    from nfit.pipeline import PlotEntry
    from nfit.plot_recipes import render_plot
    from nfit.plotting import prepare_mdhisto_tiled_slices

    data = temperature_volume()
    panels = prepare_mdhisto_tiled_slices(data, x_dim=0, y_dim=1, tile_dim=4)
    np.testing.assert_array_equal([panel.coordinate for panel in panels], data.axes[4].centers)
    for i, panel in enumerate(panels):
        np.testing.assert_allclose(panel.values, i + 1)
    subset = prepare_mdhisto_tiled_slices(data, x_dim=0, y_dim=1, tile_dim=4, tile_range=(6, 31))
    np.testing.assert_allclose([panel.coordinate for panel in subset], [10.34, 20.12, 30.06])
    plot = PlotEntry(
        "temperatures",
        "mdhisto_tiled_slices",
        settings={
            "x_dim": 0,
            "y_dim": 1,
            "tile_dim": "Temperature",
            "tile_step_auto": True,
            "tile_step": 4,
        },
    )
    figure = render_plot(plot, data)
    labels = [text.get_text() for ax in figure.axes for text in ax.texts]
    assert all(any(f"{value:.1f}" in label for label in labels) for value in data.axes[4].centers)
    plt.close(figure)


@pytest.mark.parametrize("composite", [False, True, "parent"])
def test_viewer_selects_clicked_collection_and_metadata_axis_supports_slice_and_integration(
    monkeypatch, composite
):
    from PySide6 import QtWidgets

    import nfit.project_gui as gui

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    first = DatasetGroup("other scans", datasets=[DatasetEntry("other", temperature_volume())])
    target = DatasetGroup(
        "MACS SPEC", datasets=[DatasetEntry("temperature series", temperature_volume())]
    )
    root = DataGroup(
        "workspace", subgroups=[first, DatasetGroup("temp_dependence", subgroups=[target])]
    )
    if composite:
        owner = root.subgroups[1] if composite == "parent" else target
        owner.metadata["composite"] = {"enabled": True}
    target_name = f"{owner.name} {gui.GROUP_COMPOSITE_NAME}" if composite else "temperature series"
    explorer = nfit.NfitProjectExplorer(nfit.NfitProject([root]))
    iterator = QtWidgets.QTreeWidgetItemIterator(explorer.tree)
    while iterator.value():
        item = iterator.value()
        if (
            explorer._dataset_group_for_item(item) is target
            and explorer._objects_for_item(item)[-1] == "dataset_group"
        ):
            explorer.tree.setCurrentItem(item)
            break
        iterator += 1
    # Supply prepared views to focus this test on explorer/viewer selection;
    # scientific composite binning is exercised above.
    monkeypatch.setattr(
        gui,
        "slice_viewer_datasets",
        lambda *a, **kw: ([temperature_volume(), temperature_volume()], ["other", target_name]),
    )
    viewer = explorer.open_slice_viewer_for_selection()
    assert viewer.dataset_combo.currentText() == target_name
    viewer.x_combo.setCurrentIndex(viewer.x_combo.findText("Temperature"))
    assert viewer.model.x_dim == 4
    np.testing.assert_array_equal(
        viewer.model.slice_arrays()["x_centers"], viewer.data.axes[4].centers
    )
    viewer.x_combo.setCurrentIndex(viewer.x_combo.findText("H"))
    viewer.y_combo.setCurrentIndex(viewer.y_combo.findText("K"))
    controls = viewer.hidden_controls[4]
    controls.integrate.setChecked(False)
    controls.value.setValue(10.34)
    np.testing.assert_allclose(viewer.model.slice_arrays()["signal"], 3)
    controls.integrate.setChecked(True)
    controls.low.setValue(5.24)
    controls.high.setValue(20.12)
    np.testing.assert_allclose(viewer.model.slice_arrays()["signal"], 2 + 3 + 4)
    viewer.tile_dim = 4
    viewer.tile_step_auto = True
    assert "tile_step=None" in viewer._tiled_figure_script()
    assert viewer._effective_tile_step() is None
    viewer.window.close()
    explorer.window.close()


def test_metadata_rebin_gui_rows_follow_energy_and_edit_the_public_recipe(monkeypatch):
    from PySide6 import QtWidgets

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    g = group([points(5, [1]), points(10, [3])])
    nfit.set_metadata_dimensions(g, [temperature()])
    explorer = nfit.NfitProjectExplorer(nfit.NfitProject([g]))
    monkeypatch.setattr(explorer, "_after_group_composite_changed", lambda group: None)
    panel = explorer._group_composite_group_box(g)
    label = panel.findChild(QtWidgets.QLabel, "metadata_rebin_label_0")
    energy = panel.findChild(QtWidgets.QLabel, "group_composite_axis_label_3")
    grid = label.parentWidget().layout()
    assert (
        grid.getItemPosition(grid.indexOf(label))[0]
        == grid.getItemPosition(grid.indexOf(energy))[0] + 1
    )
    tabs = panel.findChild(QtWidgets.QTabWidget, "group_composite_tabs")
    assert tabs.indexOf(label.parentWidget()) == 0
    for widget in panel.findChildren(QtWidgets.QWidget):
        if widget.objectName().startswith("metadata_rebin_"):
            assert widget.toolTip(), widget.objectName()
    mode = panel.findChild(QtWidgets.QComboBox, "metadata_rebin_mode_0")
    mode.setCurrentIndex(mode.findData("step"))
    assert g.metadata["metadata_dimensions"][0]["binning"]["step"] == 5
    assignment = panel.findChild(
        QtWidgets.QComboBox, "metadata_rebin_assignment_0"
    )
    assert assignment.isEnabled()
    assignment.setCurrentIndex(assignment.findData(True))
    assert g.metadata["metadata_dimensions"][0]["binning"]["fractional"] is True
    panel.findChild(QtWidgets.QPushButton, "group_composite_copy_settings").click()
    destination = group([points(5, [7])])
    assert explorer._paste_group_composite_settings(destination)
    assert destination.metadata["metadata_dimensions"] == g.metadata["metadata_dimensions"]
    mode.setCurrentIndex(mode.findData("discrete"))
    assert g.metadata["metadata_dimensions"][0]["binning"] is None
    mode.setCurrentIndex(mode.findData("tolerance"))
    assert g.metadata["metadata_dimensions"][0]["binning"]["tolerance"] > 0
    panel.close()
    explorer.window.close()


def test_temperature_dependent_model_fits_all_metadata_slices_and_emits_5d_overlay():
    from nfit.cross_section import intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.pipeline import ModelComponentSpec
    from nfit.spin_fluctuations import local_relaxational_chipp

    targets = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    expected = intensity_from_chipp(
        local_relaxational_chipp(1.0, chi_loc=2.4, gamma=3.1),
        1.0,
        targets,
        polarization=ISOTROPIC_POLARIZATION,
    )
    g = group(
        [points(t, [value], sigma=[0.01]) for t, value in zip(targets, expected, strict=True)]
    )
    nfit.set_metadata_dimensions(g, [temperature()])
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"scale": 1.0, "chi_loc": 1.0, "gamma": 3.1},
        fit_parameters={"chi_loc": True},
    )
    g.models[component.name] = component
    inputs, _ = nfit.fit_dataset_inputs(g, purpose="fit")
    np.testing.assert_array_equal(inputs[0].data.temperature, targets)
    result = nfit.perform_group_fit(g)
    assert result["goodness"]["status"] == "converged"
    assert component.parameters["chi_loc"] == pytest.approx(2.4, rel=1e-6)
    channels = next(iter(result["channels"].values()))
    assert np.asarray(channels["fit"]).shape == (1, 1, 1, 1, 6)
