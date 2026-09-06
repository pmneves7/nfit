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
            auto_lower=False,
            auto_upper=False,
            auto_step_size=False,
        )
    return result


def temperature(**kwargs):
    return nfit.MetadataDimension("Temperature", "parameters/temperature", "K", **kwargs)


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


def test_project_and_editable_script_round_trip(tmp_path):
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
    nfit.set_metadata_dimensions(node, [temperature(centers=[5, 10, 20])])
    project = nfit.NfitProject([g])
    path = tmp_path / "test.nfit"
    nfit.save_project(project, path)
    restored = nfit.load_project(path)
    script = nfit.composite_workflow_script(restored, g.name, node_id=node.id)
    assert "METADATA_DIMENSIONS" in script and "REBIN_CONFIG" in script
    namespace = {"__name__": "test_workflow"}
    exec(compile(script, "composite.py", "exec"), namespace)
    data = namespace["run"]()
    np.testing.assert_allclose(data.signal.ravel(), [1, 4, np.nan], equal_nan=True)
    np.testing.assert_array_equal(data.axes[-1].centers, [5, 10, 20])


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
