import numpy as np
import pytest

from nfit.analysis.coordinates import (
    bin_edges,
    physical_axis_vectors,
    physical_coordinate_arrays,
    q_bin_volume,
    q_modulus_for_spectral,
    signal_semantics,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData


def _data():
    axes = tuple(
        MDHistoAxis(name, np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
        for name in ("[H,H,0]", "[H,-H,0]", "L")
    )
    shape = (2, 2, 2)
    return MDHistoData(
        axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape),
        metadata={
            "lattice_parameters": {"a": 2 * np.pi, "b": 2 * np.pi, "c": 2 * np.pi},
            "signal_semantics": "density",
        },
    )


def test_projected_coordinates_and_q_volume():
    data = _data()
    np.testing.assert_allclose(
        physical_axis_vectors(data)[:, :3],
        [[1, 1, 0], [1, -1, 0], [0, 0, 1]],
    )
    coords = physical_coordinate_arrays(data)
    assert coords["H"][0, 0, 0] == pytest.approx(1.0)
    assert coords["K"][0, 0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(q_bin_volume(data), 2.0)
    assert signal_semantics(data) == "density"


def test_center_axis_edges_are_reconstructed():
    axis = MDHistoAxis("H", np.array([1.0, 2.0, 4.0]), "rlu", "momentum")
    np.testing.assert_allclose(bin_edges(axis, 3), [0.5, 1.5, 3.0, 5.0])


def test_q_modulus_spectral_grid_keeps_energy_dimension_singleton():
    axes = (MDHistoAxis("|Q|", np.array([0.0, 1.0, 2.0]), "1/angstrom", "momentum"), MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0, 3.0]), "meV", "energy"))
    shape = (2, 3)
    data = MDHistoData(axes, np.ones(shape), np.ones(shape), np.zeros(shape, bool), np.ones(shape))
    q = q_modulus_for_spectral(data)
    assert q.shape == (2, 1)
    np.testing.assert_allclose(q[:, 0], [0.5, 1.5])
