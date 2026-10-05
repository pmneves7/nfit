"""Metadata-only viewer ownership and explicit release of closed-window arrays."""

from __future__ import annotations

from typing import Any

from .dataset import PointListData
from .mdhisto import MDHistoData
from .qt_operation_guard import close_operation_window
from .viewer_data import DeferredViewerDatasets, ViewerDatasetDescriptor


def _volume_resource_payload(panel):
    """Expose existing grid buffers without constructing coordinates or slices."""

    grids = tuple(getattr(panel, name, None) for name in (
        "current_grid", "current_surface", "current_render_grid"
    ))
    datasets = getattr(panel, "datasets", ())
    data = getattr(panel, "data", None)
    values = [datasets, data]
    for grid in grids:
        if grid is not None:
            for name in ("point_data", "cell_data"):
                arrays = getattr(grid, name, None)
                if arrays is not None:
                    values.extend(arrays.values())
    # Keep a stable object identity between catalog scans, refreshing buffer
    # references even when the renderer replaces arrays on an existing grid.
    # Never access grid.points: a rectilinear grid computes that full array.
    payload = getattr(panel, "_nfit_resource_payload", None)
    if payload is None:
        payload = panel._nfit_resource_payload = []
    payload[:] = values
    return payload


def release_volume_resources(panel) -> None:
    """Release the retired panel's buffers before deferred Qt deletion."""

    panel.datasets = []
    panel.data = None
    panel.current_grid = panel.current_surface = panel.current_render_grid = None
    panel.hidden_controls = {}
    panel.__dict__.pop("_nfit_resource_payload", None)


def loaded_viewer_resources(viewer) -> tuple[tuple[int, ViewerDatasetDescriptor, Any], ...]:
    """List held dataset payloads without decoding an unloaded catalog entry.

    Retained presentation states can still reference the previous payload
    after a catalog refresh. Include those references so the resource
    manager accounts for all of the arrays that this viewer keeps alive.
    """

    if getattr(viewer, "_data_released", False):
        return ()
    loaded = (
        list(viewer.datasets.cached_items())
        if isinstance(viewer.datasets, DeferredViewerDatasets)
        else list(enumerate(viewer.datasets))
    )
    loaded.extend(
        (index, state.model.data)
        for index, state in enumerate(viewer._dataset_states)
        if state is not None
    )
    unique = {}
    for index, payload in loaded:
        if payload is None:
            continue
        descriptor = (
            viewer.datasets.descriptors[index]
            if isinstance(viewer.datasets, DeferredViewerDatasets)
            else ViewerDatasetDescriptor(
                viewer.dataset_names[index],
                source_dataset_name=viewer.source_dataset_names[index],
                binning_name=viewer.binning_names[index],
                binning_id=str(payload.metadata.get("binning_id", "fit")),
                group_key=viewer.dataset_group_keys[index],
                crystal_context=viewer.crystal_contexts[index],
            )
        )
        unique[(index, id(payload))] = (index, descriptor, payload)
    for child in viewer._child_viewers:
        if hasattr(child, "loaded_resource_items"):
            for index, descriptor, payload in child.loaded_resource_items():
                unique[(index, id(payload))] = (index, descriptor, payload)
        else:
            payloads = (getattr(child, "data", None), getattr(child, "_prepared", None))
            for payload in payloads:
                if not isinstance(payload, (MDHistoData, PointListData)):
                    continue
                descriptor = ViewerDatasetDescriptor(
                    f"{child.window.windowTitle()} · {payload.metadata.get('source_dataset_name', viewer.dataset_names[viewer.dataset_index])}",
                    source_dataset_name=payload.metadata.get(
                        "source_dataset_name", viewer.source_dataset_names[viewer.dataset_index]
                    ),
                    binning_name=payload.metadata.get("binning_name", viewer.binning_names[viewer.dataset_index]),
                    binning_id=str(payload.metadata.get("binning_id", "fit")),
                )
                unique[(viewer.dataset_index, id(payload))] = (viewer.dataset_index, descriptor, payload)
    if viewer.volume_panel is not None:
        payload = _volume_resource_payload(viewer.volume_panel)
        descriptor = ViewerDatasetDescriptor(
            f"Volume view · {viewer.dataset_names[viewer.dataset_index]}",
            source_dataset_name=viewer.source_dataset_names[viewer.dataset_index],
            binning_name=viewer.binning_names[viewer.dataset_index],
        )
        unique[(viewer.dataset_index, id(payload))] = (viewer.dataset_index, descriptor, payload)
    return tuple(unique.values())


def release_viewer_resources(viewer) -> None:
    """Release arrays held by a closed viewer, retaining its last plot recipe.

    Project cache references are managed separately. Call after closing the
    window so controls cannot read an unloaded model while it is visible.
    """

    if getattr(viewer, "_data_released", False):
        return
    viewer._released_plot_settings = viewer.current_plot_settings()
    viewer._close_volume_panel()
    viewer._close_cut_viewers()
    for child in tuple(viewer._child_viewers):
        close_operation_window(child.window)
        if hasattr(child, "release_loaded_data"):
            child.release_loaded_data()
        else:
            # K-path viewers owned by this window retain the source cube.
            child.data = None
            child._prepared = None
            child.figure.__dict__.pop("_nfit_kpath_data", None)
            child.figure.clear()
    viewer._child_viewers.clear()
    if viewer._background_profiles is not None:
        viewer._background_profiles.close()
        viewer._background_profiles = None
    if isinstance(viewer.datasets, DeferredViewerDatasets):
        viewer.datasets.release()
    else:
        viewer.datasets = []
    viewer._dataset_states.clear()
    viewer._current_slice = None
    viewer._current_slice_source_key = None
    viewer._current_tiled_slices.clear()
    viewer._current_waterfall_traces.clear()
    viewer._current_box_profiles = None
    viewer._current_x_cut = viewer._current_y_cut = None
    viewer.data = None
    viewer.model = None
    viewer._data_released = True
    # Matplotlib artists may hold slices which are views of entire cubes.
    viewer.figure.clear()
    viewer.image = viewer.colorbar = None
    viewer._compare_axes.clear()
    viewer._compare_colorbars.clear()
    viewer._tile_axes.clear()
    viewer._tile_colorbar_axes.clear()
    viewer._tile_colorbars.clear()
