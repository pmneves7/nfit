import h5py
import numpy as np

from metallix import load_mantid_mdhisto_nxs, point_data_from_hyspec_hhl


def test_load_mantid_mdhisto_nxs_reads_axes_and_arrays(tmp_path):
    path = tmp_path / "tiny_mdhisto.nxs"
    signal = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)

    with h5py.File(path, "w") as handle:
        workspace = handle.create_group("MDHistoWorkspace")
        workspace.attrs["NX_class"] = "NXentry"
        workspace.create_dataset("coordinate_system", data=np.array([2], dtype=np.uint32))
        workspace.create_dataset("visual_normalization", data=np.array([1], dtype=np.uint32))
        data = workspace.create_group("data")
        data.attrs["NX_class"] = "NXdata"

        for name, values, long_name, units, frame in (
            ("D3", np.linspace(-1.0, 1.0, 3), "DeltaE", "meV", "General Frame"),
            ("D2", np.linspace(-0.2, 0.2, 4), "[H,-H,0]", "r.l.u.", "HKL"),
            ("D1", np.linspace(-2.0, 2.0, 5), "[0,0,L]", "r.l.u.", "HKL"),
            ("D0", np.linspace(0.0, 1.0, 6), "[H,H,0]", "r.l.u.", "HKL"),
        ):
            axis = data.create_dataset(name, data=values)
            axis.attrs["long_name"] = long_name
            axis.attrs["units"] = units
            axis.attrs["frame"] = frame

        signal_dataset = data.create_dataset("signal", data=signal)
        signal_dataset.attrs["axes"] = "D3:D2:D1:D0"
        signal_dataset.attrs["signal"] = 1
        data.create_dataset("errors_squared", data=np.full(signal.shape, 4.0))
        data.create_dataset("mask", data=np.zeros(signal.shape, dtype=np.int8))
        data.create_dataset("num_events", data=np.full(signal.shape, 7.0))
        oriented_lattice = workspace.create_group("experiment99/sample/oriented_lattice")
        oriented_lattice.create_dataset("orientation_matrix", data=np.eye(3))

    imported = load_mantid_mdhisto_nxs(path)

    assert imported.shape == signal.shape
    np.testing.assert_allclose(imported.signal, signal)
    np.testing.assert_allclose(imported.errors, 2.0)
    np.testing.assert_array_equal(imported.mask, False)
    np.testing.assert_allclose(imported.num_events, 7.0)
    assert imported.coordinate_system == 2
    assert imported.visual_normalization == 1

    assert [axis.name for axis in imported.axes] == [
        "DeltaE",
        "[H,-H,0]",
        "[0,0,L]",
        "[H,H,0]",
    ]
    assert [axis.kind for axis in imported.axes] == [
        "energy",
        "momentum",
        "momentum",
        "momentum",
    ]
    np.testing.assert_allclose(imported.axes[0].centers, [-0.5, 0.5])
    assert imported.metadata["signal_axes"] == ("D3", "D2", "D1", "D0")
    assert "data/signal" in imported.metadata["nexus"]
    np.testing.assert_allclose(imported.metadata["oriented_lattice"]["orientation_matrix"], np.eye(3))
    assert imported.metadata["oriented_lattice"]["orientation_matrix_path"].endswith(
        "/experiment99/sample/oriented_lattice/orientation_matrix"
    )

    points, summary = point_data_from_hyspec_hhl(imported, temperature=1.8)

    assert points.size == signal.size
    assert summary["total_q_trajectories_with_data"] == 3 * 4 * 5
    assert summary["used_q_trajectories"] == 3 * 4 * 5
    assert summary["initial_valid_points"] == signal.size
    np.testing.assert_allclose(points.E[:2], [-0.5, 0.5])
    hh0 = imported.axes[3].centers[0]
    hmh0 = imported.axes[1].centers[0]
    l0 = imported.axes[2].centers[0]
    np.testing.assert_allclose(points.H[0], hh0 + hmh0)
    np.testing.assert_allclose(points.K[0], hh0 - hmh0)
    np.testing.assert_allclose(points.L[0], l0)
    np.testing.assert_allclose(points.intensity[:2], [0.0, 60.0])
    assert points.temperature == 1.8
