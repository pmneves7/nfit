from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from nfit.kpath import KPathNode, prepare_mdhisto_kpath
from nfit.mapped_archive import array_storage_nbytes
from nfit.viewer_data import DeferredViewerDatasets, ViewerDatasetDescriptor
from tests.plotting_test_data import tiny_mdhisto_data


def test_retired_kpath_releases_prepared_source_metadata(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = tiny_mdhisto_data()
    data = data.with_updates(metadata={"exposure": np.ones(data.shape)})
    source_exposure = weakref.ref(data.metadata["exposure"])
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    prepared = prepare_mdhisto_kpath(
        data, (KPathNode("A", (0, 0, 0)), KPathNode("B", (1, 1, 0))),
        lattice_parameters={"a": 4, "b": 4, "c": 4, "alpha": 90, "beta": 90, "gamma": 90},
    )
    child = SimpleNamespace(window=QtWidgets.QMainWindow(viewer.window),
        data=data, _prepared=prepared, figure=Figure())
    child.figure._nfit_kpath_data = prepared
    viewer._child_viewers.append(child)

    items = viewer.loaded_resource_items()
    assert any(payload is prepared for _, _, payload in items)
    assert prepared.metadata["exposure"] is data.metadata["exposure"]
    del items, prepared, data
    viewer.window.close()
    gc.collect()

    assert child.data is child._prepared is None
    assert not hasattr(child.figure, "_nfit_kpath_data")
    assert source_exposure() is None


def test_volume_inventory_is_stable_and_shutdown_releases_buffers(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    calls = []
    catalog = DeferredViewerDatasets(
        (ViewerDatasetDescriptor("loaded"), ViewerDatasetDescriptor("cold")),
        lambda index: calls.append(index) or tiny_mdhisto_data(),
    )
    viewer = QtMDHistoSliceViewer(catalog, x_dim=3, y_dim=2)

    class Grid:
        def __init__(self):
            self.point_data = {"rendered": np.arange(16.)}
            self.cell_data = {"cells": np.ones(8)}

        @property
        def points(self):
            pytest.fail("inventory constructed a full coordinate array")

    class Panel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.datasets = [viewer.data]
            self.data = viewer.data
            self.current_grid = Grid()
            self.current_surface = self.current_render_grid = None
            self.hidden_controls = {}

        def shutdown(self):
            # The renderer closes before its source references are released.
            assert self.data is not None

    panel = Panel()
    rendered = weakref.ref(panel.current_grid.point_data["rendered"])
    viewer.volume_panel = panel
    viewer.content_stack.addWidget(panel)
    first = viewer.loaded_resource_items()[-1][2]
    second = viewer.loaded_resource_items()[-1][2]
    assert first is second
    assert array_storage_nbytes(first).total == array_storage_nbytes(viewer.data).total + 24 * 8
    assert calls == [0]
    panel.current_grid.point_data["rendered"] = np.arange(32.)
    assert viewer.loaded_resource_items()[-1][2] is first
    gc.collect()
    assert rendered() is None
    rendered = weakref.ref(panel.current_grid.point_data["rendered"])
    assert array_storage_nbytes(first).total == array_storage_nbytes(viewer.data).total + 40 * 8
    del first, second
    viewer.window.close()
    gc.collect()

    assert panel.datasets == []
    assert panel.data is panel.current_grid is panel.current_surface is panel.current_render_grid is None
    assert not hasattr(panel, "_nfit_resource_payload")
    assert viewer.volume_panel is None
    assert rendered() is None
    assert catalog.cached_items() == ()
