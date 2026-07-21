import h5py
import numpy as np

from nfit import (
    attach_fit_comparisons,
    hyspec_hhl_fit_comparison_from_points,
    hyspec_hhl_point_indices,
    load_mantid_mdhisto_nxs,
    point_data_from_hyspec_hhl,
)


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

    imported = load_mantid_mdhisto_nxs(path, copy_metadata=True)

    assert imported.shape == signal.shape
    np.testing.assert_allclose(imported.signal, signal)
    np.testing.assert_allclose(imported.errors, 2.0)
    np.testing.assert_array_equal(imported.mask, False)
    np.testing.assert_allclose(imported.num_events, 7.0)
    assert imported.coordinate_system == 2
    assert imported.visual_normalization == 1
    assert imported.metadata["signal_semantics"] == "density"
    assert imported.metadata["signal_semantics_source"] == "mantid_mdhisto_workspace"

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
    assert [axis.role for axis in imported.axes] == [
        "energy_transfer",
        "momentum_projection",
        "momentum_projection",
        "momentum_projection",
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

    indices = hyspec_hhl_point_indices(imported, points)
    np.testing.assert_allclose(imported.signal[indices], points.intensity)

    fit_values = np.full(points.intensity.shape, 5.0)
    comparison = hyspec_hhl_fit_comparison_from_points(
        imported,
        points,
        fit_values,
        model_name="constant",
        result_name="fit 0",
    )
    attach_fit_comparisons(imported, [comparison])
    result_view = comparison.results[0]

    assert imported.metadata["fit_comparisons"] == [comparison]
    np.testing.assert_allclose(result_view.data.signal[indices], points.intensity)
    assert np.count_nonzero(~result_view.data.mask) == points.size
    assert np.count_nonzero(~result_view.fit.mask) == points.size
    np.testing.assert_allclose(result_view.fit.signal[indices], 5.0)
    np.testing.assert_allclose(
        result_view.residual.signal[indices],
        (points.intensity - 5.0) / points.sigma,
    )


def test_load_mantid_mdhisto_nxs_does_not_copy_large_metadata_by_default(tmp_path):
    path = tmp_path / "tiny_mdhisto.nxs"
    signal = np.ones((1, 1), dtype=float)

    with h5py.File(path, "w") as handle:
        workspace = handle.create_group("MDHistoWorkspace")
        data = workspace.create_group("data")
        for name, values in (("D1", np.array([0.0, 1.0])), ("D0", np.array([0.0, 1.0]))):
            axis = data.create_dataset(name, data=values)
            axis.attrs["long_name"] = name
            axis.attrs["units"] = "r.l.u."
        signal_dataset = data.create_dataset("signal", data=signal)
        signal_dataset.attrs["axes"] = "D1:D0"
        data.create_dataset("errors_squared", data=np.ones_like(signal))
        data.create_dataset("mask", data=np.zeros_like(signal, dtype=np.int8))
        data.create_dataset("num_events", data=np.ones_like(signal))
        experiment = workspace.create_group("experiment0")
        experiment.create_dataset("large_raw_payload", data=np.arange(32))

    imported = load_mantid_mdhisto_nxs(path)

    assert "nexus" not in imported.metadata
    assert imported.metadata["source_file"] == str(path)
