import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from nfit import (
    KPathNode,
    plot_mdhisto_kpath,
    prepare_mdhisto_kpath,
    sample_kpath,
    standard_kpath,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData


def _hkl_energy_data() -> MDHistoData:
    signal = np.arange(3 * 1 * 1 * 2, dtype=float).reshape(3, 1, 1, 2) + 1.0
    return MDHistoData(
        axes=(
            MDHistoAxis("[H,0,0]", np.asarray([-0.5, 0.5, 1.5, 2.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[0,K,0]", np.asarray([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[0,0,L]", np.asarray([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", np.asarray([0.0, 1.0, 2.0]), "meV", "energy"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={"rlu_to_inv_angstrom_matrix": np.eye(3).tolist()},
    )


def test_sample_and_extract_absolute_higher_zone_path_with_propagated_errors():
    data = _hkl_energy_data()
    nodes = (
        KPathNode("X1", (1.0, 0.0, 0.0)),
        KPathNode("X2", (2.0, 0.0, 0.0)),
    )

    sampling = sample_kpath(nodes, np.eye(3), step_inv_angstrom=1.0)
    result = prepare_mdhisto_kpath(
        data,
        nodes,
        step_inv_angstrom=1.0,
        transverse_width_inv_angstrom=0.1,
    )

    np.testing.assert_allclose(sampling.points_hkl, [[1, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(result.signal, data.signal[1:, 0, 0, :])
    np.testing.assert_allclose(result.errors, 1.0)
    assert result.metadata["kpath"]["nodes"][0]["hkl"] == [1.0, 0.0, 0.0]


def test_kpath_tube_average_propagates_independent_voxel_errors():
    data = _hkl_energy_data()
    result = prepare_mdhisto_kpath(
        data,
        [KPathNode("start", (0.0, 0.0, 0.0)), KPathNode("stop", (2.0, 0.0, 0.0))],
        step_inv_angstrom=2.0,
        transverse_width_inv_angstrom=0.1,
    )

    np.testing.assert_allclose(
        result.signal[0],
        np.mean(data.signal[:2, 0, 0, :], axis=0),
    )
    np.testing.assert_allclose(result.errors[0], 1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(result.num_events[0], 2.0)


def test_kpath_plot_switches_between_names_coordinates_and_guides():
    result = prepare_mdhisto_kpath(
        _hkl_energy_data(),
        [
            {"label": "X", "hkl": [0, 0, 0]},
            {"label": "R", "hkl": [2, 0, 0]},
        ],
        step_inv_angstrom=1.0,
        transverse_width_inv_angstrom=0.1,
    )

    figure = plot_mdhisto_kpath(result, label_mode="both", show_guides=True)

    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
        "X\n(0 0 0)",
        "R\n(2 0 0)",
    ]
    assert len(figure.axes[0].lines) == 2


def test_standard_kpath_uses_conventional_hkl_for_centered_lattice():
    lattice = {
        "a": 4.0, "b": 4.0, "c": 4.0,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
    }

    path = standard_kpath(lattice, "F m -3 m")

    assert len(path) >= 2
    assert any(node.label == "Γ" for node in path)
    assert all(len(node.hkl) == 3 for node in path)


def test_standard_kpath_transforms_back_to_skewed_conventional_cell():
    path = standard_kpath(
        {
            "a": 4.0, "b": 5.0, "c": 6.0,
            "alpha": 70.0, "beta": 80.0, "gamma": 75.0,
        },
        "P 1",
    )

    # The triclinic special points are corners of the reciprocal primitive
    # parallelepiped. Their coordinates must remain half-integers in the
    # original skewed cell, rather than coordinates in ASE's rotated standard
    # cell.
    assert all(
        np.allclose(np.asarray(node.hkl) * 2.0, np.round(np.asarray(node.hkl) * 2.0))
        for node in path
    )


def test_qt_kpath_viewer_exports_custom_higher_zone_path():
    pytest.importorskip("PySide6")
    from nfit.qt_kpath_viewer import QtKPathViewer

    viewer = QtKPathViewer(
        _hkl_energy_data(),
        lattice_parameters={
            "a": 4.0, "b": 4.0, "c": 4.0,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        },
        spacegroup="P 1",
    )
    viewer.set_nodes(
        [KPathNode("A", (1.0, 0.0, 0.0)), KPathNode("B", (2.0, 0.0, 0.0))]
    )

    assert viewer.redraw()
    assert "'hkl': [2.0, 0.0, 0.0]" in viewer.script()
    assert "plot_mdhisto_kpath" in viewer.script()


def test_slice_viewer_offers_kpath_viewer_only_for_four_dimensional_hkl_data():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_hkl_energy_data())

    assert not viewer.open_kpath_viewer_button.isHidden()
    assert viewer.open_kpath_viewer_button.toolTip()
