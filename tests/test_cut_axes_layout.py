"""Box profile locators must respect the space reserved for decorations."""

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nfit.plotting_core import align_mdhisto_cut_axes


@pytest.mark.parametrize("font_size", [12, 18, 24])
def test_cut_axes_retain_label_margins_and_stable_spacing(font_size):
    figure = Figure(figsize=(11, 8.5), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 3, width_ratios=[1, .19, .045],
                              height_ratios=[1, .235])
    image = figure.add_subplot(grid[0, 0])
    ycut = figure.add_subplot(grid[0, 1], sharey=image)
    figure.add_subplot(grid[0, 2])
    xcut = figure.add_subplot(grid[1, 0], sharex=image)
    image.set(xlim=(-2, 2), ylim=(0, 50), xlabel="[K,-K,0] (r.l.u.)", ylabel="ΔE (meV)")
    xcut.set(xlabel="[K,-K,0] (r.l.u.)", ylabel="Weighted mean")
    ycut.set(xlabel="Weighted mean", ylabel="ΔE (meV)")
    for axis in figure.axes:
        axis.tick_params(labelsize=font_size)
        axis.xaxis.label.set_size(font_size)
        axis.yaxis.label.set_size(font_size)
    align_mdhisto_cut_axes(image, xcut, ycut)

    for size, ratio in (((11, 8.5), .235), ((9, 7), .35), ((11, 8.5), .235)):
        figure.set_size_inches(*size)
        grid.set_height_ratios([1, ratio])
        figure.canvas.draw()
        expected = np.array([axis.get_position().bounds for axis in (image, xcut, ycut)])
        for _ in range(4):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            # The lower profile must leave the main plot's x label unobscured.
            assert image.xaxis.label.get_window_extent(renderer).y0 > xcut.get_window_extent(renderer).y1
            np.testing.assert_allclose(xcut.get_position().bounds[::2],
                                       image.get_position().bounds[::2], atol=1.e-8)
            np.testing.assert_allclose(ycut.get_position().bounds[1::2],
                                       image.get_position().bounds[1::2], atol=1.e-8)
            np.testing.assert_allclose(xcut.get_position().bounds[1::2],
                                       xcut.get_position(original=True).bounds[1::2], atol=1.e-8)
            np.testing.assert_allclose(ycut.get_position().bounds[::2],
                                       ycut.get_position(original=True).bounds[::2], atol=1.e-8)
            np.testing.assert_allclose(
                [axis.get_position().bounds for axis in (image, xcut, ycut)],
                expected, atol=1.e-6,
            )
