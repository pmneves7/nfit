"""Headless figure/CSV consumers preserve replayed targets and approximation labels."""

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nfit import (
    plot_mdhisto_slice,
    project_measured_background_mdevent,
    replay_cached_background_profile,
    save_measurement_profile_csv,
)
from nfit.backgrounds import subtract_aligned_background
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.measurement_profiles import MeasurementProfile
from tests.test_mdevent_background_statistics import _directional_fixture, _target


def _subtracted(tmp_path):
    sample, source = _directional_fixture(tmp_path)
    target = _target(sample, 2)
    return subtract_aligned_background(target,
        project_measured_background_mdevent(sample, source, target))


def test_static_figure_replays_original_grid_and_labels_separate_diagonal_sum(tmp_path):
    data = _subtracted(tmp_path)
    figure = plot_mdhisto_slice(data, x_dim=0, y_dim=3,
        show_histogram_axes=True, roi_extents=(-2, .5, -.5, .5),
        background_uncertainty='replay')
    image, ycut, _, xcut = figure.axes
    # The vertical profile pools both spatial cells; one background observation
    # stays one primitive, giving Var=.5 sample +2 background rather than 1.25.
    expected = replay_cached_background_profile(data,
        selected=np.ones(data.shape, bool), indices=np.zeros(data.shape, int), edges=[-.5, .5])
    np.testing.assert_allclose(ycut.lines[0].get_xdata(), expected.data.signal)
    bar = ycut.collections[0].get_segments()[0]
    assert abs(bar[1, 0]-bar[0, 0])/2 == pytest.approx(np.sqrt(2.5))
    assert 'diagonal uncertainty' in image.get_title(loc='left')
    assert 'diagonal' not in xcut.get_ylabel()
    plt.close(figure)


@pytest.mark.parametrize('kwargs', [dict(channel='errors'), dict(smoothing_sigma_x=1.),
    dict(smoothing_sigma_y=1.), dict(background_uncertainty='unknown')])
def test_static_replay_rejects_unsupported_treatments_before_drawing(tmp_path, kwargs):
    data = _subtracted(tmp_path)
    options = dict(background_uncertainty='replay')
    options.update(kwargs)
    with pytest.raises(ValueError, match='Background'):
        plot_mdhisto_slice(data, **options)


def test_static_replay_rejects_coarsened_grid(tmp_path):
    data = _subtracted(tmp_path)
    with pytest.raises(ValueError, match='original displayed grid'):
        plot_mdhisto_slice(data, x_dim=0, y_dim=3, x_step=5.,
            show_histogram_axes=True, roi_extents=(-2, .5, -.5, .5),
            background_uncertainty='replay')
    plt.close('all')


def test_profile_csv_exact_guard_runs_before_creating_any_output(tmp_path):
    data = _subtracted(tmp_path)
    profile = replay_cached_background_profile(data,
        selected=np.ones(data.shape, bool), indices=np.zeros(data.shape, int), edges=[0, 1])
    preview = MeasurementProfile(profile.data.with_updates(metadata={**profile.data.metadata,
        'background_profile_uncertainty':'diagonal_approximation_preview'}), profile.contract)
    path = tmp_path/'profile.csv'
    with pytest.raises(SourceReplayRequired, match='approximate'):
        save_measurement_profile_csv(path, preview, coordinate_name='E',
            require_exact_background_uncertainty=True)
    assert not path.exists()
    assert not path.with_suffix('.csv.json').exists()
    save_measurement_profile_csv(path, profile, coordinate_name='E',
        require_exact_background_uncertainty=True)
    metadata = json.loads(path.with_suffix('.csv.json').read_text())['metadata']
    assert metadata['background_profile_uncertainty'] == 'source_covariance'
    assert metadata['cross_profile_bin_covariance'] == 'not_retained_requires_source_replay'


def test_replayed_marginal_display_does_not_allow_later_cross_bin_aggregation(tmp_path):
    from nfit import replay_cached_background_box_profiles
    from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
    from nfit.plotting_core import MDHistoSliceViewer, coarsen_mdhisto_view
    from nfit.qt_box_cut_viewers import CutViewerContext, _profile_dataset

    data = _subtracted(tmp_path)
    profiles = replay_cached_background_box_profiles(data, x_dim=0, y_dim=3,
        extents=(-2, .5, -.5, .5))
    context = CutViewerContext(data, 0, 3, lambda dim: data.axes[dim].name, 'sample')
    marginal = _profile_dataset(context, 'x', profiles.x, profiles.x_measurement)
    view = MDHistoSliceViewer(marginal, x_dim=1, y_dim=0).slice_arrays()
    np.testing.assert_allclose(view['errors'].ravel(), profiles.x[2])
    with pytest.raises(ValueError, match='payload or source replay'):
        coarsen_mdhisto_view(view, x_step=5.)

    # Even an explicitly source-covariance-marked histogram cannot pool two
    # marginal cells lacking the joint source dependencies.
    repeated = MDHistoData(
        (MDHistoAxis('hidden', np.arange(3.), '', 'unknown'), *marginal.axes),
        np.repeat(marginal.signal[None], 2, axis=0),
        np.repeat(marginal.errors[None], 2, axis=0),
        np.repeat(marginal.mask[None], 2, axis=0),
        np.repeat(marginal.num_events[None], 2, axis=0),
        metadata=marginal.metadata,
        auxiliary_channels={name: MDHistoChannel(np.repeat(channel.values[None], 2, axis=0))
                            for name, channel in marginal.auxiliary_channels.items()},
    )
    model = MDHistoSliceViewer(repeated, x_dim=2, y_dim=1)
    model.selections[0] = (.5, 1.5)
    model.integrate_checks[0] = True
    with pytest.raises(ValueError, match='payload or source replay'):
        model.slice_arrays()
