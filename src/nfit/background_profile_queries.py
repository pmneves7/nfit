"""Explicit original-grid profile selection for cached background covariance.

Selection descriptors contain only grid identity and dimension/index choices.
They read no events and retain no histogram arrays. Full-grid memberships and
source replay are created only by the explicit profile functions below.
"""
from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass, replace

import numpy as np

from .box_cuts import BoxProfiles, box_profile_selection
from .cached_background_replay import replay_cached_background_profile
from .measurement_dependencies import SourceReplayRequired


@dataclass(frozen=True)
class CachedBackgroundSliceSelection:
    """Bounded runtime selection tied to one immutable original histogram.

    Recreate this descriptor after replacing, scaling or transforming the source
    histogram. Hidden ranges use inclusive original-grid indices. This runtime
    descriptor is not a substitute for the persisted scientific replay recipe.
    """

    source_identity: int
    grid_signature: str
    displayed_dimensions: tuple[int, ...]
    selections: tuple[tuple[int, int, int], ...]


def _grid_signature(data):
    digest = hashlib.sha256()
    digest.update(str(data.shape).encode())
    for axis in data.axes:
        digest.update(np.ascontiguousarray(axis.values).tobytes())
        digest.update(np.ascontiguousarray(axis.centers).tobytes())
        digest.update(repr((axis.name, axis.units, axis.kind, axis.frame)).encode())
    return digest.hexdigest()


def background_profile_slice_selection(data, *, displayed_dimensions, selections=None):
    """Capture a lazy selection in original histogram indices without event I/O.

    ``displayed_dimensions`` is an ordered one- or two-axis tuple. ``selections``
    maps every other dimension to an index or an inclusive ``(start, stop)``
    pair, as returned by the slice viewer's normalized selection. Missing hidden
    selections choose the middle bin. Public scripts may specify these indices
    directly; no viewer or Qt object is required.
    """
    dimensions = tuple(operator.index(dim) for dim in displayed_dimensions)
    if len(dimensions) not in (1, 2) or len(set(dimensions)) != len(dimensions):
        raise ValueError("Select one or two distinct displayed dimensions")
    if any(dim < 0 or dim >= len(data.shape) for dim in dimensions):
        raise ValueError("Displayed dimension lies outside the cached grid")
    selections = {} if selections is None else dict(selections)
    if any(dim in dimensions or dim < 0 or dim >= len(data.shape) for dim in selections):
        raise ValueError("Selections must refer only to hidden original-grid dimensions")
    hidden = []
    for dim, size in enumerate(data.shape):
        if dim in dimensions:
            continue
        raw = selections.get(dim, size // 2)
        if isinstance(raw, (tuple, list)):
            if len(raw) != 2:
                raise ValueError("Hidden index ranges must contain two inclusive endpoints")
            start, stop = map(operator.index, raw)
        else:
            start = stop = operator.index(raw)
        if start < 0 or stop < start or stop >= size:
            raise ValueError("Hidden index selection lies outside the cached grid")
        hidden.append((dim, start, stop))
    return CachedBackgroundSliceSelection(id(data), _grid_signature(data), dimensions, tuple(hidden))


def _resolve_selection(data, dimensions, selections, selection):
    if selection is None:
        return background_profile_slice_selection(data, displayed_dimensions=dimensions, selections=selections)
    if selections is not None:
        raise ValueError("Supply selections or a captured selection descriptor, not both")
    if not isinstance(selection, CachedBackgroundSliceSelection):
        raise ValueError("Expected a cached-background slice selection descriptor")
    if (selection.source_identity != id(data) or selection.grid_signature != _grid_signature(data)
            or selection.displayed_dimensions != tuple(dimensions)):
        raise SourceReplayRequired("Cached background selection changed; capture it from the original histogram again")
    expected = background_profile_slice_selection(data, displayed_dimensions=dimensions,
        selections={dim: (start, stop) for dim, start, stop in selection.selections})
    if expected != selection:
        raise SourceReplayRequired("Cached background selection is inconsistent with its original grid")
    return selection


def _hidden_membership(data, selection):
    included = np.ones(data.shape, bool)
    weights = np.ones([1] * len(data.shape))
    for dim, start, stop in selection.selections:
        shape = [1] * len(data.shape)
        shape[dim] = data.shape[dim]
        positions = np.arange(data.shape[dim])
        included &= ((positions >= start) & (positions <= stop)).reshape(shape)
        # Only integrated axes contribute bin width to geometric coverage.
        if stop > start:
            widths = np.diff(data.axes[dim].values)
            weights = weights * widths.reshape(shape)
    return included, weights


def _lift_plane(values, data, x_dim, y_dim):
    values = np.asarray(values)
    shape = [1] * len(data.shape)
    shape[x_dim], shape[y_dim] = data.shape[x_dim], data.shape[y_dim]
    ordered = values if y_dim < x_dim else values.T
    return np.broadcast_to(ordered.reshape(shape), data.shape)


def _coordinate_profile(profile, data, dimension, selection, *, extents=None, angle=0., side=None):
    source_axis = data.axes[dimension]
    axis = replace(profile.data.axes[0], name=source_axis.name, units=source_axis.units,
                   kind=source_axis.kind, frame=source_axis.frame)
    query = {"selection": [list(item) for item in selection.selections],
             "displayed_dimensions": list(selection.displayed_dimensions),
             "membership": "original_cached_cell_centers",
             "aggregation_target": "original_cached_cell_exposure_weighted_field_mean"}
    if extents is not None:
        query.update(extents=list(map(float, extents)), angle_degrees=float(angle), side=side)
        if not np.isclose(float(angle) % 360, 0., atol=1e-10):
            x_axis, y_axis = [data.axes[dim] for dim in selection.displayed_dimensions]
            axis = replace(axis, name=f"Box {side} · {source_axis.name}", frame=None,
                           units=x_axis.units if x_axis.units == y_axis.units else "",
                           kind=x_axis.kind if x_axis.kind == y_axis.kind else "unknown")
            query["coordinate_convention"] = "rotation_in_displayed_axis_coordinates"
    output = profile.data.with_updates(axes=(axis,), metadata={**profile.data.metadata, "background_profile_query": query})
    return replace(profile, data=output)


def replay_cached_background_box_profiles(
    data, *, x_dim, y_dim, extents, angle=0., selections=None, selection=None,
    display_selected=None, coverage_threshold=0., progress_callback=None,
    max_batch_bytes=64 * 1024**2,
):
    """Replay exact background marginal errors for regular/rotated box cuts.

    Membership is evaluated on original cached-grid centers. Each hidden axis
    uses the supplied inclusive index selection; every selected original cell
    contributes to one x and one y profile bin. ``display_selected`` optionally
    excludes visible 2D pixels (for example a slice coverage threshold) before
    mapping back to original cells. Original masks always apply. This explicit
    operation is for unsmoothed primary signal on the original displayed grid;
    it does not support unmasking, model channels or coarsened preview geometry.

    The retained recipe chooses background exposure for a background field and
    sample exposure for a subtracted field. Sample variance retains its recorded
    treatment; exact background marginals do not certify cross-bin independence.
    No event reads occur until this function is invoked.
    """
    selection = _resolve_selection(data, (x_dim, y_dim), selections, selection)
    view = {"x_edges": data.axes[x_dim].values, "x_centers": data.axes[x_dim].centers,
            "y_edges": data.axes[y_dim].values, "y_centers": data.axes[y_dim].centers}
    plane = box_profile_selection(view, extents, angle)
    selected = plane.selected
    if display_selected is not None:
        visible = np.asarray(display_selected)
        if visible.dtype.kind != "b" or visible.shape != selected.shape:
            raise ValueError("Visible box selection must be boolean and match the original displayed grid")
        selected = selected & visible
    if not len(plane.x_edges):
        empty = (np.array([]), np.array([]), np.array([]))
        return BoxProfiles(empty, empty, selected)
    hidden, widths = _hidden_membership(data, selection)
    included = hidden & _lift_plane(selected, data, x_dim, y_dim)
    profiles = []
    for side, dim, indices, edges, centers, coverage in (
        ("x", x_dim, plane.x_indices, plane.x_edges, plane.x_centers, plane.x_coverage_weights),
        ("y", y_dim, plane.y_indices, plane.y_edges, plane.y_centers, plane.y_coverage_weights),
    ):
        geometric = widths
        if coverage is not None:
            geometric = widths * _lift_plane(np.broadcast_to(coverage, selected.shape), data, x_dim, y_dim)
        profile = replay_cached_background_profile(data, selected=included,
            indices=_lift_plane(indices, data, x_dim, y_dim), edges=edges, centers=centers,
            coverage_weights=geometric, coverage_threshold=coverage_threshold,
            progress_callback=progress_callback, max_batch_bytes=max_batch_bytes)
        profiles.append(_coordinate_profile(profile, data, dim, selection,
            extents=extents, angle=angle, side=side))
    return BoxProfiles(profiles[0].arrays, profiles[1].arrays, selected, *profiles)


def replay_cached_background_slice_profile(
    data, *, axis, selections=None, selection=None, limits=None,
    coverage_threshold=0., progress_callback=None, max_batch_bytes=64 * 1024**2,
):
    """Replay a regular 1D cut while retaining original coordinate edges/centers.

    ``limits`` selects inclusive coordinate-center bounds on the output axis.
    Other axes use the same hidden index/range rules as box profile queries.
    """
    axis = operator.index(axis)
    selection = _resolve_selection(data, (axis,), selections, selection)
    centers = data.axes[axis].centers
    positions = np.arange(len(centers))
    if limits is not None:
        if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[1] < limits[0]:
            raise ValueError("Slice limits must contain finite increasing center bounds")
        positions = positions[(centers >= limits[0]) & (centers <= limits[1])]
    if not len(positions):
        raise ValueError("Slice limits contain no original histogram centers")
    start, stop = positions[0], positions[-1]
    shape = [1] * len(data.shape)
    shape[axis] = len(centers)
    all_positions = np.arange(len(centers)).reshape(shape)
    hidden, widths = _hidden_membership(data, selection)
    selected = hidden & (all_positions >= start) & (all_positions <= stop)
    indices = np.broadcast_to(np.clip(all_positions-start, 0, stop-start), data.shape)
    profile = replay_cached_background_profile(data, selected=selected, indices=indices,
        edges=data.axes[axis].values[start:stop+2], centers=centers[start:stop+1],
        coverage_weights=widths, coverage_threshold=coverage_threshold,
        progress_callback=progress_callback, max_batch_bytes=max_batch_bytes)
    return _coordinate_profile(profile, data, axis, selection)
