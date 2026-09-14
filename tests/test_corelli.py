import numpy as np
import pytest

from nfit import (
    bin_corelli_group,
    bin_raw_dgs_group,
    corelli_dataset_group,
    inspect_corelli_run,
    is_corelli_raw_nexus_file,
    raw_dgs_dataset_group,
)
from nfit.corelli import (
    CORELLI_TOF_US_PER_M_SQRT_MEV,
    _ChopperTiming,
    _corelli_bin_contributions,
    _corelli_fractional_axes,
    _correlation_weights,
    _geometry_from_solid_angle,
    _load_corelli_detector_mask,
    _load_corelli_flux,
    _solve_incident_energy,
)
from nfit.raw_dgs import _detector_geometry


def _write_corelli(path, *, incident_energy=50.0, delta_e=0.0):
    h5py = pytest.importorskip("h5py")
    l1 = 20.0
    l2 = np.sqrt(5.0)
    tof = float(
        CORELLI_TOF_US_PER_M_SQRT_MEV
        * (l1 / np.sqrt(incident_energy) + l2 / np.sqrt(incident_energy - delta_e))
    )
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0" name="CORELLI">
      <component type="moderator"><location z="-20"/></component>
      <component type="correlation-chopper"><location z="-2"/></component>
      <component type="panel" idlist="panel"><location/></component>
      <component-link name="correlation-chopper"><parameter name="sequence" type="string"><value val="180 180"/></parameter></component-link>
      <type name="moderator" is="Source"/><type name="correlation-chopper" is="chopper"/>
      <type name="pixel" is="detector"/>
      <type name="panel"><component type="pixel"><location x="1" y="0" z="2"/></component></type>
      <idlist idname="panel"><id start="42" end="42"/></idlist>
    </instrument>"""
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("run_number", data=[b"7"])
        instrument = entry.create_group("instrument")
        instrument.create_group("instrument_xml").create_dataset(
            "data", data=np.frombuffer(xml.encode(), dtype="u1")
        )
        bank = entry.create_group("bank1_events")
        bank.create_dataset("event_id", data=[42])
        bank.create_dataset("event_time_offset", data=[tof])
        bank.create_dataset("event_index", data=[0])
        bank.create_dataset("event_time_zero", data=[0.0])
        logs = entry.create_group("DASlogs")
        tdc = logs.create_group("chopper4_TDC")
        tdc.create_dataset("time", data=np.arange(0.0, 0.021, 0.001))
        tdc.create_dataset("value", data=np.zeros(21, dtype=np.uint32))
        speed = logs.create_group("BL9:Chop:Skf4:MotorSpeed")
        speed.create_dataset("value", data=[1000.0])
        charge = logs.create_group("proton_charge")
        charge.create_dataset("value", data=[3.6e9])
        for index, value in enumerate((10.0, 20.0, 30.0), start=1):
            axis = logs.create_group(f"BL9:Mot:Sample:Axis{index}")
            axis.create_dataset("value", data=[value])


def test_corelli_metadata_and_finite_energy_solver(tmp_path):
    source = tmp_path / "CORELLI_7.nxs.h5"
    _write_corelli(source, incident_energy=50.0, delta_e=4.0)

    assert is_corelli_raw_nexus_file(source)
    info = inspect_corelli_run(source)
    assert info.run_number == "7"
    assert info.event_count == 1
    assert info.source_to_chopper == pytest.approx(18.0)
    assert info.goniometer_angles == pytest.approx((10.0, 20.0, 30.0))
    l2 = np.array([np.sqrt(5.0)])
    tof = np.array(
        [
            CORELLI_TOF_US_PER_M_SQRT_MEV
            * (20.0 / np.sqrt(50.0) + l2[0] / np.sqrt(46.0))
        ]
    )
    solved = _solve_incident_energy(tof, l2, 4.0, 20.0, (10.0, 100.0))
    assert solved[0] == pytest.approx(50.0, abs=1e-6)


def test_corelli_fractional_assignment_splits_momentum_but_not_energy():
    edges = (
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([-0.5, 0.5]),
    )
    assignments = _corelli_fractional_axes(None, dimensions=4)
    flat, points, weights = _corelli_bin_contributions(
        np.array([[1.0, 1.0, 1.0, 0.0]]),
        edges,
        (2, 2, 2, 1),
        assignments,
    )

    np.testing.assert_array_equal(assignments, [True, True, True, False])
    np.testing.assert_array_equal(np.sort(flat), np.arange(8))
    np.testing.assert_array_equal(points, np.zeros(8, dtype=int))
    np.testing.assert_allclose(weights, np.full(8, 0.125))
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="energy assignment must be discrete"):
        _corelli_fractional_axes([True, True, True, True], dimensions=4)


def test_corelli_compiled_fractional_deposition_matches_numpy():
    from nfit import corelli as corelli_module

    if corelli_module._CORELLI_NUMBA is None:
        pytest.skip("Numba is unavailable")
    edges = np.array([0.0, 1.0, 2.0])
    shape = np.array([2, 2, 2, 1], dtype=np.int64)
    assignments = np.array([True, True, True, False])
    values = np.zeros(8)
    variances = np.zeros(8)
    counts = np.zeros(8)

    corelli_module._CORELLI_NUMBA._deposit_hkle(
        values,
        variances,
        counts,
        1.0,
        1.0,
        1.0,
        0,
        edges,
        edges,
        edges,
        shape,
        assignments,
        2.0,
    )

    np.testing.assert_allclose(values, np.full(8, 0.25))
    np.testing.assert_allclose(variances, np.full(8, 0.0625))
    np.testing.assert_allclose(counts, np.full(8, 0.125))


def test_corelli_weights_match_mantid_open_closed_convention():
    timing = _ChopperTiming(
        tdc_ns=np.array([0.0, 1_000_000.0, 2_000_000.0]),
        period_ns=1_000_000.0,
        sequence_edges_deg=np.array([180.0, 360.0]),
        duty_cycle=0.5,
    )
    weights, valid = _correlation_weights(
        np.array([1_250_000.0, 1_750_000.0]), timing, 0.0
    )
    np.testing.assert_array_equal(valid, [True, True])
    # Mantid labels even sequence intervals absorbing and odd intervals open.
    np.testing.assert_allclose(weights, [-1.0, 1.0])


def test_corelli_group_reconstructs_requested_energy_channel(tmp_path):
    source = tmp_path / "CORELLI_7.nxs.h5"
    _write_corelli(source)
    group = corelli_dataset_group([source])
    group.metadata["raw_dgs"]["timing_offset_ns"] = 0
    group.metadata["raw_dgs"]["bad_pulse_threshold"] = 0

    result = bin_corelli_group(
        group,
        lower=[-100.0, -100.0, -100.0, -1.0],
        upper=[100.0, 100.0, 100.0, 1.0],
        num_bins=[1, 1, 1, 1],
        max_batch_bytes=128,
    )

    assert result.num_events.item() == 1.0
    assert np.isfinite(result.signal.item())
    assert result.errors.item() > 0.0
    assert not result.mask.item()
    reconstruction = result.metadata["corelli_reconstruction"]
    assert reconstruction["energy_sampling"] == "requested_DeltaE_bin_centres"
    assert reconstruction["fractional_axes"] == [True, True, True, False]
    assert result.metadata["rebin"]["fractional_axes"] == [True, True, True, False]
    assert "correlated" in reconstruction["channel_covariance"]


def test_generic_raw_import_dispatches_corelli_batches(tmp_path, monkeypatch):
    from nfit import corelli as corelli_module

    sources = []
    for run in range(3):
        source = tmp_path / f"CORELLI_{run}.nxs.h5"
        _write_corelli(source)
        sources.append(source)
    progress = []

    group = raw_dgs_dataset_group(sources, progress_callback=progress.append)
    group.metadata["raw_dgs"]["timing_offset_ns"] = 0
    group.metadata["raw_dgs"]["bad_pulse_threshold"] = 0
    geometry_calls = []
    detector_geometry = corelli_module._detector_geometry

    def tracked_geometry(path):
        geometry_calls.append(path)
        return detector_geometry(path)

    monkeypatch.setattr(corelli_module, "_detector_geometry", tracked_geometry)
    result = bin_raw_dgs_group(
        group,
        lower=[-100.0, -100.0, -100.0, -1.0],
        upper=[100.0, 100.0, 100.0, 1.0],
        num_bins=[1, 1, 1, 1],
        max_batch_bytes=128,
    )

    assert group.metadata["raw_dgs"]["format"] == "corelli-correlation-nexus"
    assert len(group.datasets) == 3
    assert result.num_events.item() == 3.0
    assert geometry_calls == [sources[0]]
    assert [event["iteration"] for event in progress] == [1, 2, 3]
    assert all(event["stage"] == "raw_dgs_import" for event in progress)


def test_corelli_compiled_paths_match_numpy(tmp_path, monkeypatch):
    from nfit import corelli as corelli_module

    if corelli_module._CORELLI_NUMBA is None:
        pytest.skip("Numba is unavailable")
    source = tmp_path / "CORELLI_7.nxs.h5"
    _write_corelli(source)
    group = corelli_dataset_group([source])
    group.metadata["raw_dgs"]["timing_offset_ns"] = 0
    group.metadata["raw_dgs"]["bad_pulse_threshold"] = 0
    settings = {
        "lower": [0.0, -1.0],
        "upper": [100.0, 1.0],
        "num_bins": [1, 3],
        "max_batch_bytes": 128,
    }
    monkeypatch.setattr(corelli_module, "CORELLI_NUMBA_MIN_HYPOTHESES", 0)
    compiled = corelli_module.bin_corelli_powder_group(group, **settings)
    hkle_settings = {
        "lower": [-100.0, -100.0, -100.0, -1.0],
        "upper": [100.0, 100.0, 100.0, 1.0],
        "num_bins": [1, 1, 1, 3],
        "max_batch_bytes": 128,
    }
    compiled_hkle = corelli_module.bin_corelli_group(group, **hkle_settings)
    monkeypatch.setattr(corelli_module, "_CORELLI_NUMBA", None)
    numpy_result = corelli_module.bin_corelli_powder_group(group, **settings)
    numpy_hkle = corelli_module.bin_corelli_group(group, **hkle_settings)

    np.testing.assert_allclose(compiled.signal, numpy_result.signal)
    np.testing.assert_allclose(compiled.errors, numpy_result.errors)
    np.testing.assert_allclose(compiled.num_events, numpy_result.num_events)
    np.testing.assert_allclose(compiled_hkle.signal, numpy_hkle.signal)
    np.testing.assert_allclose(compiled_hkle.errors, numpy_hkle.errors)
    np.testing.assert_allclose(compiled_hkle.num_events, numpy_hkle.num_events)


def test_corelli_single_energy_uses_vectorized_path(tmp_path, monkeypatch):
    from nfit import corelli as corelli_module

    source = tmp_path / "CORELLI_7.nxs.h5"
    _write_corelli(source)
    group = corelli_dataset_group([source])
    group.metadata["raw_dgs"]["timing_offset_ns"] = 0
    group.metadata["raw_dgs"]["bad_pulse_threshold"] = 0

    class UnexpectedCompiledBackend:
        def run_hkle(self, *args, **kwargs):
            raise AssertionError("single-energy reconstruction should stay vectorized")

    monkeypatch.setattr(corelli_module, "_CORELLI_NUMBA", UnexpectedCompiledBackend())
    monkeypatch.setattr(corelli_module, "CORELLI_NUMBA_MIN_HYPOTHESES", 0)
    result = bin_corelli_group(
        group,
        lower=[-100.0, -100.0, -100.0, -1.0],
        upper=[100.0, 100.0, 100.0, 1.0],
        num_bins=[1, 1, 1, 1],
        max_batch_bytes=128,
    )

    assert result.num_events.item() == 1.0


def test_corelli_flux_maps_grouped_bank_spectra_to_detector_ids(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "flux.nxs"
    with h5py.File(source, "w") as handle:
        workspace = handle.create_group("mantid_workspace_1")
        data = workspace.create_group("workspace")
        data.create_dataset("axis1", data=[1.0, 2.0, 3.0])
        data.create_dataset(
            "values", data=[[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]
        )
        detector = workspace.create_group("instrument").create_group("detector")
        detector.create_dataset("detector_list", data=[10, 11, 20])
        detector.create_dataset("detector_count", data=[2, 1])
        detector.create_dataset("detector_index", data=[0, 2])

    flux = _load_corelli_flux(source)
    density = flux.density_for_ids(
        np.array([11, 20, 99]), np.array([1.5, 2.5, 2.0])
    )
    np.testing.assert_allclose(density, [1.0, 2.0, 0.0])


def test_corelli_reads_mantid_xml_detector_masks(tmp_path):
    source = tmp_path / "mask.xml"
    source.write_text(
        "<detector-masking><group><detids>10-12, 20</detids></group></detector-masking>"
    )
    mask = _load_corelli_detector_mask(source)
    np.testing.assert_allclose(mask.value_for_ids(np.array([9, 10, 12, 13, 20])), [1, 0, 0, 1, 0])


def test_corelli_reads_mantid_spherical_calibrated_detector_positions(tmp_path):
    h5py = pytest.importorskip("h5py")
    raw = tmp_path / "CORELLI_7.nxs.h5"
    solid = tmp_path / "solid_angle.nxs"
    _write_corelli(raw)
    with h5py.File(solid, "w") as handle:
        workspace = handle.create_group("mantid_workspace_1")
        detector = workspace.create_group("instrument").create_group("detector")
        detector.create_dataset("detector_list", data=[42])
        detector.create_dataset("detector_positions", data=[[2.0, 90.0, 0.0]])

    geometry = _geometry_from_solid_angle(solid, _detector_geometry(raw))
    positions, _he3, valid = geometry.event_geometry_for_ids(np.array([42]))

    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_allclose(positions, [[2.0, 0.0, 0.0]], atol=1e-12)


def test_corelli_group_panel_exposes_timing_wavelength_and_correction_files(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.pipeline import DataGroup
    from nfit.project_gui import NfitProjectExplorer
    from nfit.project_io import NfitProject

    source = tmp_path / "CORELLI_7.nxs.h5"
    _write_corelli(source)
    subgroup = corelli_dataset_group([source])
    root = DataGroup("sample", subgroups=[subgroup])
    explorer = NfitProjectExplorer(NfitProject([root]))
    explorer._refresh_tree(select_dataset_group=subgroup)

    names = (
        "raw_dgs_normalization_file",
        "raw_dgs_flux_file",
        "raw_dgs_mask_file",
        "raw_dgs_timing_offset_ns",
        "raw_dgs_wavelength_min_angstrom",
        "raw_dgs_wavelength_max_angstrom",
        "raw_dgs_ki_kf_normalization",
        "raw_dgs_he3_detector_efficiency_correction",
        "raw_dgs_ub_matrix",
    )
    for name in names:
        widget = explorer.details_widget.findChild(QtWidgets.QWidget, name)
        assert widget is not None and widget.toolTip()
