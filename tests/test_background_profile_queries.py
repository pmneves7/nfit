"""Original-grid membership, coordinate output and explicit covariance replay."""
import ast
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from nfit import (
    background_profile_slice_selection,
    replay_cached_background_box_profiles,
    replay_cached_background_slice_profile,
)
from nfit.backgrounds import subtract_aligned_background
from nfit.box_cuts import box_profile_selection
from nfit.cached_background_replay import clear_cached_background_profile_queries
from nfit.mdevent_background import project_measured_background_mdevent
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.measurement_profiles import MeasurementProfile
from tests.test_mdevent_background_statistics import _directional_fixture, _target


@pytest.fixture
def cached(tmp_path):
    clear_cached_background_profile_queries()
    sample, source = _directional_fixture(tmp_path)
    target = _target(sample, 2)
    return target, project_measured_background_mdevent(sample, source, target)


def test_regular_box_replays_shared_event_before_squaring_and_preserves_axes(cached):
    _, background = cached
    result = replay_cached_background_box_profiles(background, x_dim=0, y_dim=3,
        selections={1: 0, 2: 0}, extents=(-2., .5, -.5, .5))
    np.testing.assert_allclose(result.x[1], [4., 4.])
    np.testing.assert_allclose(result.x[2]**2, [2., 2.])
    assert result.y[1].item() == pytest.approx(4.)
    assert result.y[2].item()**2 == pytest.approx(2.)
    assert result.y_measurement.data.metadata["background_profile_uncertainty"] == "source_covariance"
    assert result.x_measurement.data.axes[0].name == background.axes[0].name
    assert result.y_measurement.data.axes[0].units == background.axes[3].units
    np.testing.assert_array_equal(result.x[0], background.axes[0].centers)
    np.testing.assert_array_equal(result.x_measurement.data.axes[0].values, background.axes[0].values)


def test_subtracted_box_uses_sample_exposure_and_recorded_sample_variance(cached):
    sample, background = cached
    result = replay_cached_background_box_profiles(subtract_aligned_background(sample, background),
        x_dim=0, y_dim=3, extents=(-2., .5, -.5, .5))
    assert result.y[1].item() == pytest.approx(-2.)
    assert result.y[2].item()**2 == pytest.approx(2.5)
    assert result.y_measurement.data.metadata["measurement_target"] == "sample_exposure_weighted_subtracted_field_mean"


def test_rotated_box_uses_same_shared_event_covariance_and_marks_mixed_coordinate_units(cached):
    _, background = cached
    result = replay_cached_background_box_profiles(background, x_dim=0, y_dim=3,
        extents=(-2., .5, -.5, .5), angle=10.)
    assert result.selected.all()
    np.testing.assert_allclose(result.x_measurement.data.errors[~result.x_measurement.data.mask]**2, 2.)
    assert result.y[1].item() == pytest.approx(4.)
    assert result.y[2].item()**2 == pytest.approx(2.)
    assert result.x_measurement.data.axes[0].name.startswith("Box x")
    assert result.x_measurement.data.axes[0].units == ""
    assert result.x_measurement.data.metadata["background_profile_query"]["coordinate_convention"] == "rotation_in_displayed_axis_coordinates"


def test_one_dimensional_slice_keeps_discrete_coordinate_and_selected_limits(cached):
    _, background = cached
    result = replay_cached_background_slice_profile(background, axis=0,
        selections={1: 0, 2: 0, 3: 0}, limits=(-2., -.7))
    assert result.data.signal.item() == pytest.approx(4.)
    assert result.data.errors.item()**2 == pytest.approx(2.)
    np.testing.assert_array_equal(result.arrays[0], background.axes[0].centers[:1])
    np.testing.assert_array_equal(result.data.axes[0].values, background.axes[0].values[:2])


def test_visible_selection_excludes_pixels_and_original_mask_still_applies(cached):
    _, background = cached
    result = replay_cached_background_box_profiles(background, x_dim=0, y_dim=3,
        extents=(-2., .5, -.5, .5), display_selected=np.array([[True, False]]))
    assert result.y[1].item() == pytest.approx(4.)
    assert result.y[2].item()**2 == pytest.approx(2.)
    assert result.x_measurement.data.mask[1]
    masked = background.with_updates(mask=np.array([False, True]).reshape(background.shape))
    result = replay_cached_background_box_profiles(masked, x_dim=0, y_dim=3,
        extents=(-2., .5, -.5, .5), coverage_threshold=.6)
    assert result.y_measurement.data.auxiliary_channels["coverage_fraction"].values.item() == .5
    assert result.y_measurement.data.mask.item()


def _grid():
    edges = ([0., 1., 2., 3.], [0., 1., 4.], [-2., 0., 2., 4.], [0., 2., 5.])
    axes = tuple(MDHistoAxis(f"axis {index}", np.array(values), "", "unknown")
                 for index, values in enumerate(edges))
    shape = tuple(len(values)-1 for values in edges)
    return MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape))


@pytest.mark.parametrize("dims", [(0, 2), (2, 0), (1, 3), (3, 1)])
@pytest.mark.parametrize("angle", [0., 27., 90.])
def test_box_membership_lifts_original_indices_in_either_axis_order_with_hidden_widths(monkeypatch, dims, angle):
    import nfit.background_profile_queries as service
    data = _grid()
    x_dim, y_dim = dims
    hidden = {dim: (0, data.shape[dim]-1) for dim in range(4) if dim not in dims}
    extents = (data.axes[x_dim].values[0], data.axes[x_dim].values[-1],
               data.axes[y_dim].values[0], data.axes[y_dim].values[-1])
    requests = []
    def capture(_data, **kwargs):
        requests.append(kwargs)
        bins = len(kwargs["edges"])-1
        axis = MDHistoAxis("Profile", kwargs["edges"], "", "unknown",
            metadata={} if kwargs["centers"] is None else {"discrete_centers": kwargs["centers"].tolist()})
        return MeasurementProfile(MDHistoData((axis,), np.zeros(bins), np.zeros(bins),
            np.zeros(bins, bool), np.zeros(bins)), MeasurementContract(
                kind="continuous", estimator="uniform_mean", quantity="test field", value_units="U"))
    monkeypatch.setattr(service, "replay_cached_background_profile", capture)
    result = replay_cached_background_box_profiles(data, x_dim=x_dim, y_dim=y_dim,
        selections=hidden, extents=extents, angle=angle)
    plane = box_profile_selection({"x_centers": data.axes[x_dim].centers,
        "y_centers": data.axes[y_dim].centers, "x_edges": data.axes[x_dim].values,
        "y_edges": data.axes[y_dim].values}, extents, angle)
    for coordinate in np.ndindex(data.shape):
        pixel = (coordinate[y_dim], coordinate[x_dim])
        width = np.prod([np.diff(data.axes[dim].values)[coordinate[dim]] for dim in hidden])
        for request, assignments, coverage in (
            (requests[0], plane.x_indices, plane.x_coverage_weights),
            (requests[1], plane.y_indices, plane.y_coverage_weights),
        ):
            assert request["selected"][coordinate] == plane.selected[pixel]
            assert request["indices"][coordinate] == assignments[pixel]
            if coverage is not None:
                extra = np.broadcast_to(coverage, plane.selected.shape)[pixel]
            else:
                extra = 1.
            assert np.broadcast_to(request["coverage_weights"], data.shape)[coordinate] == width*extra
    np.testing.assert_array_equal(result.selected, plane.selected)


def test_selection_capture_is_lazy_bounded_and_invalidated_by_copy_or_transform(cached, monkeypatch):
    import h5py
    _, background = cached
    def no_read(*args, **kwargs):
        raise AssertionError("Selection descriptor read original events")
    monkeypatch.setattr(h5py, "File", no_read)
    selection = background_profile_slice_selection(background, displayed_dimensions=(0, 3), selections={1: 0, 2: 0})
    assert not any(isinstance(getattr(selection, field.name), np.ndarray) for field in fields(selection))
    for changed in (background.with_updates(), background.with_updates(signal=2*background.signal)):
        with pytest.raises(SourceReplayRequired, match="selection changed"):
            replay_cached_background_box_profiles(changed, x_dim=0, y_dim=3,
                selection=selection, extents=(-2., .5, -.5, .5))
    with pytest.raises(SourceReplayRequired, match="selection changed"):
        replay_cached_background_box_profiles(background, x_dim=0, y_dim=3,
            selection=replace(selection, grid_signature="other"), extents=(-2., .5, -.5, .5))


@pytest.mark.parametrize("kwargs", [{"displayed_dimensions": (0, 0)},
    {"displayed_dimensions": (0, 5)}, {"displayed_dimensions": (0, 1), "selections": {0: 0}},
    {"displayed_dimensions": (0, 1), "selections": {2: (-1, 0)}},
    {"displayed_dimensions": (0, 1), "selections": {2: (1, 0)}}])
def test_invalid_original_index_selection_fails(kwargs):
    with pytest.raises(ValueError):
        background_profile_slice_selection(_grid(), **kwargs)


def test_query_services_are_public_and_gui_independent():
    import nfit
    import nfit.background_profile_queries as service
    assert nfit.replay_cached_background_box_profiles is service.replay_cached_background_box_profiles
    tree = ast.parse(Path(service.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("Qt" in name or "gui" in name or name in {"plotting_core", "project_data", "mdevent"}
                   for name in imports)


def test_regular_zero_width_center_selection_preserves_existing_preview_rule():
    view = {"x_centers": np.array([.5, 1.5]), "y_centers": np.array([.5]),
            "x_edges": np.array([0., 1., 2.]), "y_edges": np.array([0., 1.])}
    selected = box_profile_selection(view, (.5, .5, .5, .5))
    np.testing.assert_array_equal(selected.selected, [[True, False]])
    np.testing.assert_array_equal(selected.x_edges, [0., 1.])
