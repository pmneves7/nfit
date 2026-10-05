"""Portable display settings shared by viewer clipboard and Python callers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy


def copy_viewer_settings(settings: Mapping[str, object], axis_names: Sequence[str]) -> dict:
    """Capture presentation settings with named axes, independent of data identity."""
    values = deepcopy(dict(settings))
    for key in (
        "dataset_name", "binning_name", "waterfall_dataset_names", "figsize",
        "brillouin_zone_spacegroup", "brillouin_zone_lattice_parameters",
    ):
        values.pop(key, None)
    return {"format": "nfit-viewer-settings", "version": 1,
            "axis_names": list(axis_names), "settings": values}


def compatible_viewer_settings(payload: Mapping, axis_names: Sequence[str]) -> dict:
    """Remap hidden dimensions by name and omit settings for unavailable axes.

    Pass the result to a viewer's public ``apply_plot_settings`` method. Dataset,
    binning, model identity and lattice metadata remain owned by the destination.
    """
    if payload.get("format") != "nfit-viewer-settings" or payload.get("version") != 1:
        raise ValueError("The clipboard does not contain supported nfit viewer settings.")
    source_names = payload["axis_names"]
    if not isinstance(source_names, list) or not all(isinstance(n, str) for n in source_names):
        raise ValueError("Invalid source axis names.")
    values = copy_viewer_settings(payload["settings"], source_names)["settings"]
    names = list(axis_names)
    for key in ("selections", "integrate_checks"):
        remapped = {}
        for index, value in dict(values.get(key, {})).items():
            index = int(index)
            if 0 <= index < len(source_names) and source_names[index] in names:
                remapped[names.index(source_names[index])] = value
        values[key] = remapped
    if values.get("y_dim") not in names:
        values.pop("waterfall_center_bounds", None)
    compatible_xy = (
        values.get("x_dim") in names
        and values.get("y_dim") in names
        and (values.get("x_dim") != values.get("y_dim") or len(names) == 1)
    )
    if not compatible_xy:
        for key in ("x_dim", "y_dim", "xlim", "ylim", "x_step", "y_step",
                    "roi_extents", "roi_angle", "roi_enabled", "show_box_tool"):
            values.pop(key, None)
    if values.get("tile_dim") not in names:
        for key in ("tile_dim", "tile_range", "tile_step"):
            values.pop(key, None)
        if values.get("view_mode") == "tiled_slices":
            values.pop("view_mode", None)
    # Saved plot recipes currently describe the 2D controls, not volume state.
    if values.get("view_mode") == "volumetric":
        values.pop("view_mode", None)
    return values
