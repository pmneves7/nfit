import copy

import h5py
import numpy as np

from nfit import load_mantid_mdhisto_nxs
from nfit.mdhisto import MDHistoAxis, MDHistoData


def test_energy_axis_migrates_legacy_deltae_unit_to_mev():
    axis = MDHistoAxis(
        "DeltaE",
        np.asarray([-1.0, 0.0, 1.0]),
        "DeltaE",
        "energy",
    )

    assert axis.name == "DeltaE"
    assert axis.units == "meV"


def test_mdhisto_numeric_payload_requires_explicit_mutable_copy():
    axis = MDHistoAxis("DeltaE", np.array([0.0, 1.0]), "meV", "energy")
    data = MDHistoData(
        axes=(axis,),
        signal=np.array([1.0]),
        errors=np.array([0.1]),
        mask=np.array([False]),
        num_events=np.array([2.0]),
    )

    assert not data.signal.flags.writeable
    assert not data.axes[0].values.flags.writeable
    editable = data.mutable_copy()
    editable.signal[0] = 4.0
    replacement = editable.immutable_copy()

    assert data.signal[0] == 1.0
    assert replacement.signal[0] == 4.0
    assert not replacement.signal.flags.writeable

    restored = copy.deepcopy(data).immutable_copy()
    assert not restored.signal.flags.writeable
    assert not restored.axes[0].values.flags.writeable


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


def test_load_corelli_correlation_chopper_output_masks_invalid_bins_and_reads_lattice(
    tmp_path,
):
    directory = tmp_path / "CORELLI" / "normalization"
    directory.mkdir(parents=True)
    path = directory / "sample_cc_sub_bkg.nxs"
    signal = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, 6.0]])
    variance = np.array([[1.0, 4.0, np.nan], [9.0, -1.0, 16.0]])
    events = np.ones_like(signal)

    with h5py.File(path, "w") as handle:
        workspace = handle.create_group("MDHistoWorkspace")
        workspace.attrs["QConvention"] = "Crystallography"
        data = workspace.create_group("data")
        for name, long_name, size in (
            ("D1", "[0,K,0]", 3),
            ("D0", "[H,0,0]", 4),
        ):
            axis = data.create_dataset(name, data=np.arange(size, dtype=float))
            axis.attrs["long_name"] = long_name
            axis.attrs["units"] = "r.l.u."
            axis.attrs["frame"] = "HKL"
        signal_dataset = data.create_dataset("signal", data=signal)
        signal_dataset.attrs["axes"] = "D1:D0"
        data.create_dataset("errors_squared", data=variance)
        data.create_dataset("mask", data=np.zeros_like(signal, dtype=np.int8))
        data.create_dataset("num_events", data=events)
        instrument = workspace.create_group("experiment0/instrument")
        instrument.create_dataset("name", data=np.asarray([b" "]))
        lattice = workspace.create_group("experiment0/sample/oriented_lattice")
        lattice.create_dataset("orientation_matrix", data=np.eye(3) * 0.125)
        for key, value in {
            "a": 4.3,
            "b": 4.3,
            "c": 11.1,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }.items():
            lattice.create_dataset(f"unit_cell_{key}", data=np.asarray([value]))

    imported = load_mantid_mdhisto_nxs(path)

    np.testing.assert_array_equal(
        imported.mask,
        [[False, True, True], [False, True, False]],
    )
    assert np.isnan(imported.errors[0, 2])
    assert np.isnan(imported.errors[1, 1])
    assert imported.metadata["invalid_bin_count"] == 3
    assert imported.metadata["instrument_name"] == "CORELLI"
    assert imported.metadata["instrument_name_source"] == "source_path"
    assert imported.metadata["suggested_data_type"] == "single_crystal_elastic"
    assert imported.metadata["q_convention"] == "Crystallography"
    assert imported.metadata["reduction_provenance"] == {
        "format": "Mantid MDHistoWorkspace",
        "scattering_mode": "elastic",
        "elastic_discrimination": "correlation_chopper",
        "correlation_chopper": True,
        "correlation_chopper_source": "source_filename",
    }
    assert imported.metadata["lattice_parameters"] == {
        "a": 4.3,
        "b": 4.3,
        "c": 11.1,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    np.testing.assert_allclose(imported.metadata["ub_matrix"], np.eye(3) * 0.125)


def test_load_wand2_output_reads_instrument_and_wavelength(tmp_path):
    path = tmp_path / "wand_output.nxs"
    signal = np.ones((2, 2), dtype=float)

    with h5py.File(path, "w") as handle:
        workspace = handle.create_group("MDHistoWorkspace")
        workspace.attrs["QConvention"] = "Inelastic"
        data = workspace.create_group("data")
        for name, long_name, units in (
            ("D1", "[0,K,0]", "in 0.827 A^-1"),
            ("D0", "[H,0,0]", "in 0.476 A^-1"),
        ):
            axis = data.create_dataset(name, data=np.arange(3, dtype=float))
            axis.attrs["long_name"] = long_name
            axis.attrs["units"] = units
            axis.attrs["frame"] = "HKL"
        signal_dataset = data.create_dataset("signal", data=signal)
        signal_dataset.attrs["axes"] = "D1:D0"
        data.create_dataset("errors_squared", data=np.ones_like(signal))
        data.create_dataset("mask", data=np.zeros_like(signal, dtype=np.int8))
        data.create_dataset("num_events", data=np.ones_like(signal))
        instrument = workspace.create_group("experiment0/instrument")
        instrument.create_dataset("name", data=np.asarray([b"WAND"]))
        logs = workspace.create_group("experiment0/logs/wavelength")
        logs.create_dataset("value", data=np.asarray([1.486]))

    imported = load_mantid_mdhisto_nxs(path)

    assert imported.metadata["instrument_name"] == "WAND²"
    assert imported.metadata["instrument_name_source"] == "nexus"
    assert imported.metadata["incident_wavelength"] == 1.486
    assert imported.metadata["incident_wavelength_unit"] == "angstrom"
    assert imported.metadata["suggested_data_type"] == "single_crystal_elastic"
    assert imported.metadata["reduction_provenance"] == {
        "format": "Mantid MDHistoWorkspace",
        "scattering_mode": "monochromatic_elastic",
    }
