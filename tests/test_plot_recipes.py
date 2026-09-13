from __future__ import annotations

import numpy as np

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DataGroup, DatasetEntry
from nfit.plot_recipes import new_plot_entry, plot_script, render_plot
from nfit.project_gui import NfitProject, _project_from_dict, _project_to_dict


def _data() -> MDHistoData:
    axes = (
        MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
        MDHistoAxis("K", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
    )
    shape = (2, 2)
    return MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape))


def test_plot_recipe_renders_and_exports_backend_script(tmp_path):
    entry = new_plot_entry("Map", "dataset-id", {"x_dim": "H", "y_dim": "K", "title": "Map"}, plot_type="mdhisto_slice")
    figure = render_plot(entry, _data())
    assert figure.axes[0].get_title() == "Map"
    script = plot_script(entry, project_path=tmp_path / "sample.nfit")
    assert "PySide" not in script
    compile(script, "generated_plot.py", "exec")


def test_plot_recipe_restores_major_gridlines_and_shared_style():
    entry = new_plot_entry(
        "Map",
        "dataset-id",
        {
            "x_dim": "H",
            "y_dim": "K",
            "show_major_gridlines": True,
            "brillouin_zone_color": "#123456",
            "brillouin_zone_linewidth": 2.25,
            "brillouin_zone_alpha": 0.6,
        },
        plot_type="mdhisto_slice",
    )

    axis = render_plot(entry, _data()).axes[0]
    gridlines = [line for line in axis.get_xgridlines() if line.get_visible()]

    assert gridlines
    assert all(line.get_color() == "#123456" for line in gridlines)
    assert all(line.get_linewidth() == 2.25 for line in gridlines)
    assert all(line.get_alpha() == 0.6 for line in gridlines)


def test_slice_recipe_preserves_empty_bins_from_legacy_fill_recipe():
    axes = (
        MDHistoAxis("H", np.arange(4.0), "rlu", "momentum"),
        MDHistoAxis("K", np.arange(4.0), "rlu", "momentum"),
    )
    signal = np.array(
        [[np.nan, 2.0, np.nan], [4.0, np.nan, np.nan], [np.nan, np.nan, 8.0]]
    )
    data = MDHistoData(
        axes,
        signal,
        np.ones_like(signal),
        np.zeros(signal.shape, bool),
        np.ones_like(signal),
    )
    entry = new_plot_entry(
        "Map",
        "dataset-id",
        {"x_dim": "K", "y_dim": "H", "empty_bin_fill_neighbors": 0},
        plot_type="mdhisto_slice",
    )

    entry.settings["empty_bin_fill_neighbors"] = 2
    rendered = render_plot(entry, data).axes[0].collections[0].get_array()

    assert np.ma.getmaskarray(rendered)[1, 1]


def test_plot_entries_persist_with_the_project_schema():
    dataset = DatasetEntry("scan", None, id="dataset-id", metadata={"source_file": "scan.nxs"})
    plot = new_plot_entry("Map", dataset.id, {"x_dim": "H", "y_dim": "K"}, plot_type="mdhisto_slice")
    project = NfitProject([DataGroup("workspace", datasets=[dataset], plots=[plot])])
    payload = _project_to_dict(project)
    assert payload["version"] == 4
    restored = _project_from_dict(payload).data_groups[0].plots[0]
    assert restored.id == plot.id
    assert restored.sources[0].dataset_id == dataset.id


def test_waterfall_recipe_renders_and_persists_multiple_dataset_sources():
    source_first = _data()
    axes = (
        MDHistoAxis("fixed", np.array([-0.5, 0.5]), "", "unknown"),
        MDHistoAxis("Q", np.arange(5.0), "1/angstrom", "momentum"),
    )
    first = source_first.with_updates(
        signal=source_first.signal.reshape(1, 4),
        errors=source_first.errors.reshape(1, 4),
        mask=source_first.mask.reshape(1, 4),
        num_events=source_first.num_events.reshape(1, 4),
        axes=axes,
    )
    source_second = _data()
    second = source_second.with_updates(
        signal=source_second.signal.reshape(1, 4),
        errors=source_second.errors.reshape(1, 4),
        mask=source_second.mask.reshape(1, 4),
        num_events=source_second.num_events.reshape(1, 4),
        axes=axes,
    )
    entry = new_plot_entry(
        "Cuts",
        "first",
        {
            "view_mode": "waterfall",
            "x_dim": "Q",
            "y_dim": "fixed",
            "waterfall_dataset_names": ["first", "second"],
            "waterfall_offset": 0.5,
            "waterfall_trace_label_suffix": " at 6 K",
            "waterfall_trace_label_font_size": 13.0,
            "waterfall_trace_label_color": "#000000",
        },
        plot_type="mdhisto_waterfall",
        dataset_ids=["first", "second"],
    )

    figure = render_plot(entry, [first, second])

    assert len(entry.sources) == 2
    assert [source.dataset_id for source in entry.sources] == ["first", "second"]
    assert len(figure.axes[0]._nfit_waterfall_traces) == 2
    assert {text.get_text() for text in figure.axes[0].texts} == {
        "first at 6 K",
        "second at 6 K",
    }
    assert all(
        text.get_fontsize() == 13.0
        for text in figure.axes[0].texts
    )
    assert all(text.get_color() == "#000000" for text in figure.axes[0].texts)


def test_tiled_slice_recipe_restores_labels_and_local_colorbars():
    axes = (
        *_data().axes,
        MDHistoAxis("Temperature", np.arange(5.0), "K", "temperature"),
    )
    signal = np.arange(16.0).reshape(2, 2, 4)
    data = MDHistoData(
        axes,
        signal,
        np.ones_like(signal),
        np.zeros_like(signal, dtype=bool),
        np.ones_like(signal),
    )
    entry = new_plot_entry(
        "Temperature maps",
        "dataset-id",
        {
            "view_mode": "tiled_slices",
            "x_dim": "H",
            "y_dim": "K",
            "tile_dim": "Temperature",
            "tile_range": (0.5, 3.5),
            "tile_step": 1.0,
            "show_tile_labels": False,
            "tile_local_color_scales": True,
        },
        plot_type="mdhisto_tiled_slices",
    )

    figure = render_plot(entry, data)

    assert len(figure._nfit_tiled_axes) == 4
    assert len(figure._nfit_tiled_colorbars) == 4
    assert len(figure.axes) == 8
    assert all(not axis.texts for axis in figure._nfit_tiled_axes)
    assert [
        colorbar.ax.yaxis.label.get_text()
        for colorbar in figure._nfit_tiled_colorbars
    ] == ["", r"$I(\mathbf{Q},E)$ (a.u.)", "", r"$I(\mathbf{Q},E)$ (a.u.)"]

    entry.settings.update(
        show_tile_labels=True, tile_label_decimals=0,
        tile_label_prefix="T = ", tile_label_unit="K", tile_label_si_prefix="m",
    )
    labeled = render_plot(entry, data)
    assert labeled._nfit_tiled_axes[0].texts[0].get_text() == "T = 500 mK"


def test_fit_comparison_recipe_can_render_model_through_data_masks():
    data = _data().mutable_copy()
    data.mask[0, 0] = True
    data.metadata["fit"] = np.full(data.shape, 2.0)
    data.metadata["residual"] = np.full(data.shape, -1.0)
    entry = new_plot_entry(
        "Fit",
        "dataset-id",
        {"x_dim": "K", "y_dim": "H", "unmask_model": True},
        plot_type="fit_comparison",
    )

    figure = render_plot(entry, data)
    fit_axis = next(axis for axis in figure.axes if axis.get_title() == "Fit")
    residual_axis = next(axis for axis in figure.axes if axis.get_title() == "Residual")
    assert np.all(np.isfinite(fit_axis.collections[0].get_array()))
    assert np.all(np.isfinite(residual_axis.collections[0].get_array()))
