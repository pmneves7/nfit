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


def test_plot_entries_persist_with_the_project_schema():
    dataset = DatasetEntry("scan", None, id="dataset-id", metadata={"source_file": "scan.nxs"})
    plot = new_plot_entry("Map", dataset.id, {"x_dim": "H", "y_dim": "K"}, plot_type="mdhisto_slice")
    project = NfitProject([DataGroup("workspace", datasets=[dataset], plots=[plot])])
    payload = _project_to_dict(project)
    assert payload["version"] == 3
    restored = _project_from_dict(payload).data_groups[0].plots[0]
    assert restored.id == plot.id
    assert restored.sources[0].dataset_id == dataset.id


def test_fit_comparison_recipe_can_render_model_through_data_masks():
    data = _data()
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
