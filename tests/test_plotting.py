import matplotlib

matplotlib.use("Agg")

import numpy as np

from metallix import PointData4D
from metallix.plotting import plot_2d_map, plot_energy_cut, plot_q_cut


def test_plotting_helpers_return_axes():
    data = PointData4D(
        H=[0.0, 0.1, 0.2, 0.3],
        K=[0.0, 0.0, 0.1, 0.1],
        L=[0.0, 0.0, 0.0, 0.0],
        E=[1.0, 2.0, 3.0, 4.0],
        intensity=[1.0, 2.0, 1.5, 2.5],
        sigma=[0.1, 0.1, 0.1, 0.1],
    )
    assert plot_energy_cut(data) is not None
    assert plot_q_cut(data, "H") is not None
    assert plot_2d_map(data.H, data.K, np.asarray(data.intensity)) is not None

